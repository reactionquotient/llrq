#!/usr/bin/env python3
"""
Multi-mode LLRQ fitting for CM (Competitive-Mixed) rate law dynamics.

This module integrates advanced LLRQ fitting methods with CM dynamics simulation,
providing multi-exponential, piecewise constant K(t), and offset-drive fitting
capabilities.

Key features:
- Multi-mode exponential fitting with shared equilibrium constants
- Piecewise constant K(t) fitting via segmentation
- Single-mode fitting with constant external drive
- Comparison with standard single-mode LLRQ fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import least_squares

# Import CM dynamics functions
from cm_rate_law_integrated import (
    CMParams,
    simulate,
    sample_params,
    compute_reaction_quotient,
    fit_llrq_parameter_sweep,
    OUT_DIR,
)

# =============================================================================
# Multi-mode LLRQ fitting functions (cleaned from special_fitting.py)
# =============================================================================


def _compute_lnQ(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute ln(Q) from concentration arrays."""
    Q = compute_reaction_quotient(A, B, C)
    return np.log(Q)


def fit_llrq_multi_exp(
    ts_list: List[np.ndarray], A_list: List[np.ndarray], B_list: List[np.ndarray], C_list: List[np.ndarray], M: int = 2
) -> Dict[str, Any]:
    """
    Fit ln Q(t) with a mixture of M exponentials that share a common Keq across runs.

    Model: y_r(t) = ln Q_r(t) - ln Keq = y0_r * sum_m w_m * exp(-k_m t)

    Parameters:
    -----------
    ts_list : List[np.ndarray]
        Time arrays for each experimental run
    A_list, B_list, C_list : List[np.ndarray]
        Concentration arrays for each run
    M : int
        Number of exponential modes (default: 2)

    Returns:
    --------
    Dict containing fitted parameters (k, w, Keq, y0) and predictions
    """
    R = len(ts_list)
    assert R == len(A_list) == len(B_list) == len(C_list), "All lists must have same length"

    lnQ_list = [_compute_lnQ(A_list[r], B_list[r], C_list[r]) for r in range(R)]

    # Parameterization helpers for numerical stability
    def softplus(x):
        # More numerically stable softplus
        x = np.clip(x, -500, 500)  # Prevent overflow
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def softmax(b):
        b = np.clip(b, -500, 500)  # Prevent overflow
        e = np.exp(b - np.max(b))
        return e / (e.sum() + 1e-12)  # Prevent division by zero

    def unpack_params(theta):
        """Unpack parameter vector into physical parameters."""
        off = 0
        th_k = theta[off : off + M]
        off += M  # Rate parameters (transformed)
        b_w = theta[off : off + M]
        off += M  # Weight parameters (logit)
        lnKeq = theta[off]
        off += 1  # Log equilibrium constant
        y0 = theta[off : off + R]
        off += R  # Initial deviations for each run

        # Ensure rates are ordered: k[0] < k[1] < ... < k[M-1]
        k = np.empty(M)
        k[0] = softplus(th_k[0])
        for m in range(1, M):
            k[m] = k[m - 1] + softplus(th_k[m])

        w = softmax(b_w)  # Weights sum to 1
        Keq = np.exp(lnKeq)
        return k, w, Keq, y0

    def y_model(t, k, w, y0):
        """Multi-exponential model: y(t) = y0 * sum_m w_m exp(-k_m t)"""
        # Clip k*t to prevent overflow
        kt_outer = np.outer(t, k)
        kt_outer = np.clip(kt_outer, 0, 500)  # Prevent overflow in exp
        E = np.exp(-kt_outer)  # shape (T, M)
        return y0 * (E @ w)

    def residuals(theta):
        """Residual function for least squares fitting."""
        k, w, Keq, y0s = unpack_params(theta)
        res = []
        for r in range(R):
            t = ts_list[r]
            lnQ = lnQ_list[r]
            y_obs = lnQ - np.log(Keq)
            y_fit = y_model(t, k, w, y0s[r])
            res.append(y_obs - y_fit)
        return np.concatenate(res)

    # Initialize parameters more conservatively
    t_all = np.concatenate(ts_list)
    tmin, tmax = float(t_all.min()), float(t_all.max())

    # Choose rate guesses spanning the time window more conservatively
    if tmax > 1e-6:
        k_typical = 1.0 / tmax  # Typical relaxation rate
        ks0 = np.array([k_typical * 0.5, k_typical * 2.0]) if M == 2 else np.linspace(k_typical * 0.1, k_typical * 5.0, M)
    else:
        ks0 = np.logspace(-1, 1, M)  # Default range

    # Use more conservative initialization for softplus inverse
    th_k0 = np.log(np.maximum(ks0, 1e-6))  # Avoid log(0)
    b_w0 = np.zeros(M)  # Equal weights initially

    # Estimate initial Keq from tail values
    Keq0 = np.median([np.median(np.exp(lnQ[-max(5, len(lnQ) // 5) :])) for lnQ in lnQ_list])
    Keq0 = max(Keq0, 1e-12)
    lnKeq0 = np.log(Keq0)

    # Initial deviations
    y00 = np.array([lnQ_list[r][0] - lnKeq0 for r in range(R)])

    theta0 = np.concatenate([th_k0, b_w0, [lnKeq0], y00])

    # Check initial residuals to catch obvious problems
    try:
        res0 = residuals(theta0)
        if not np.all(np.isfinite(res0)):
            print("Initial residuals not finite, using fallback initialization")
            # Fallback initialization
            th_k0 = np.array([-1.0, 0.0]) if M == 2 else np.linspace(-2, 0, M)
            theta0 = np.concatenate([th_k0, b_w0, [lnKeq0], y00])
            res0 = residuals(theta0)
            if not np.all(np.isfinite(res0)):
                raise ValueError("Cannot find finite initial point")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return {"success": False, "message": f"Initialization failed: {e}"}

    # Fit parameters
    try:
        out = least_squares(residuals, theta0, max_nfev=20000, ftol=1e-12, xtol=1e-12)
        if not out.success:
            print(f"Optimization did not converge: {out.message}")
        k, w, Keq, y0s = unpack_params(out.x)

        # Generate predictions
        preds = []
        for r in range(R):
            t = ts_list[r]
            lnQ_pred = np.log(Keq) + y_model(t, k, w, y0s[r])
            preds.append(lnQ_pred)

        return {
            "success": out.success,
            "cost": float(out.cost),
            "k": k,
            "w": w,
            "Keq": Keq,
            "y0": y0s,
            "lnQ_pred_list": preds,
            "lnQ_obs_list": lnQ_list,
            "times": ts_list,
            "message": out.message,
        }

    except Exception as e:
        print(f"Multi-exponential fitting failed: {e}")
        return {"success": False, "message": str(e)}


def fit_llrq_with_offset(
    ts_list: List[np.ndarray], A_list: List[np.ndarray], B_list: List[np.ndarray], C_list: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Fit affine LLRQ model: d/dt ln Q = -k (ln Q - ln Keq) + u

    Solution: y(t) = (y0 - u/k) e^{-k t} + u/k
    where y(t) = ln Q(t) - ln Keq

    Shared (k, u, Keq) across runs, run-specific y0.

    Parameters:
    -----------
    ts_list, A_list, B_list, C_list : Lists of arrays for each experimental run

    Returns:
    --------
    Dict containing fitted parameters and predictions
    """
    R = len(ts_list)
    lnQ_list = [_compute_lnQ(A_list[r], B_list[r], C_list[r]) for r in range(R)]

    def unpack(theta):
        k = np.exp(theta[0])  # >0
        u = theta[1]  # can be any real
        lnKeq = theta[2]
        y0 = theta[3 : 3 + R]
        return k, u, lnKeq, y0

    def model_y(t, k, u, y0):
        """Affine LLRQ solution."""
        if k < 1e-12:  # avoid division by zero
            return y0 + u * t
        return (y0 - u / k) * np.exp(-k * t) + u / k

    def residuals(theta):
        k, u, lnKeq, y0s = unpack(theta)
        res = []
        for r in range(R):
            t = ts_list[r]
            y_obs = lnQ_list[r] - lnKeq
            y_fit = model_y(t, k, u, y0s[r])
            res.append(y_obs - y_fit)
        return np.concatenate(res)

    # Initialize parameters
    Keq0 = np.median([np.median(np.exp(lnQ[-max(5, len(lnQ) // 5) :])) for lnQ in lnQ_list])
    lnKeq0 = float(np.log(max(Keq0, 1e-12)))
    k0 = 1.0 / max(1e-6 + max([ts.max() for ts in ts_list]), 1e-3)
    theta0 = np.concatenate([[np.log(k0)], [0.0], [lnKeq0], [lnQ[0] - lnKeq0 for lnQ in lnQ_list]])

    try:
        out = least_squares(residuals, theta0, max_nfev=20000)
        k, u, lnKeq, y0s = unpack(out.x)
        Keq = float(np.exp(lnKeq))

        # Generate predictions
        lnQ_pred_list = []
        for r in range(R):
            y_fit = model_y(ts_list[r], k, u, y0s[r])
            lnQ_pred_list.append(lnKeq + y_fit)

        return {
            "success": out.success,
            "message": out.message,
            "cost": float(out.cost),
            "k": float(k),
            "u": float(u),
            "Keq": Keq,
            "y0": y0s,
            "lnQ_pred_list": lnQ_pred_list,
            "lnQ_obs_list": lnQ_list,
            "times": ts_list,
        }

    except Exception as e:
        print(f"Offset fitting failed: {e}")
        return {"success": False, "message": str(e)}


# =============================================================================
# Integration functions
# =============================================================================


def run_cm_multi_mode_comparison(
    params: Optional[CMParams] = None,
    seed: int = 20250912,
    t_span: Tuple[float, float] = (0.0, 3.0),
    n_points: int = 200,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run CM simulations and compare single-mode vs multi-mode LLRQ fitting.

    Parameters:
    -----------
    params : CMParams, optional
        CM parameters. If None, will sample random parameters
    seed : int
        Random seed for parameter generation
    t_span : tuple
        Time span for simulation
    n_points : int
        Number of time points
    save_plots : bool
        Whether to save comparison plots

    Returns:
    --------
    Dict containing all results and fitted models
    """
    if params is None:
        params = sample_params(seed=seed)

    print("CM Parameters:")
    print(f"  u: {params.u:.4f}")
    print(f"  kM_A: {params.kM_A:.4f}, kM_B: {params.kM_B:.4f}, kM_C: {params.kM_C:.4f}")
    print(f"  k_plus: {params.k_plus:.4f}, k_minus: {params.k_minus:.4f}")

    # Compute true equilibrium constant
    true_Keq = (params.k_plus / params.k_minus) * (params.kM_C**2) / (params.kM_A * params.kM_B)
    print(f"  True Keq: {true_Keq:.4e}")

    # Define time evaluation points
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # Set up forward and reverse experiments
    experiments = {
        "forward": [0.20 * params.kM_A, 1.25 * params.kM_B, 2e-4 * params.kM_C],
        "reverse": [0.05 * params.kM_A, 0.32 * params.kM_B, 0.50 * params.kM_C],
    }

    # Simulate CM dynamics for both experiments
    results = {}
    ts_list, A_list, B_list, C_list = [], [], [], []

    print("\nRunning CM simulations...")
    for name, y0 in experiments.items():
        print(f"  {name} experiment: y0 = {y0}")
        t, A, B, C, v = simulate(t_span, y0, params, t_eval=t_eval)
        results[name] = {"t": t, "A": A, "B": B, "C": C, "v": v}
        ts_list.append(t)
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)

    # Fit single-mode LLRQ to forward experiment (for comparison)
    print("\nFitting single-mode LLRQ to forward experiment...")
    single_mode_result = fit_llrq_parameter_sweep(
        results["forward"]["t"],
        results["forward"]["A"],
        results["forward"]["B"],
        results["forward"]["C"],
        true_Keq,
        fit_on="lnQ",
    )

    # Fit multi-mode LLRQ across both experiments
    print("\nFitting 2-mode LLRQ across both experiments...")
    multi_mode_result = fit_llrq_multi_exp(ts_list, A_list, B_list, C_list, M=2)

    # Fit with offset across both experiments
    print("\nFitting LLRQ with offset across both experiments...")
    offset_result = fit_llrq_with_offset(ts_list, A_list, B_list, C_list)

    # Create comparison plots
    if save_plots:
        create_comparison_plots(results, single_mode_result, multi_mode_result, offset_result, true_Keq)

    return {
        "cm_results": results,
        "single_mode": single_mode_result,
        "multi_mode": multi_mode_result,
        "offset": offset_result,
        "true_Keq": true_Keq,
        "params": params,
    }


def create_comparison_plots(cm_results: Dict, single_mode: Dict, multi_mode: Dict, offset: Dict, true_Keq: float):
    """Create comprehensive comparison plots."""

    fig = plt.figure(figsize=(20, 12))

    # Extract data
    t_fwd = cm_results["forward"]["t"]
    t_rev = cm_results["reverse"]["t"]

    # Compute ln(Q) for true CM data
    lnQ_fwd_true = _compute_lnQ(cm_results["forward"]["A"], cm_results["forward"]["B"], cm_results["forward"]["C"])
    lnQ_rev_true = _compute_lnQ(cm_results["reverse"]["A"], cm_results["reverse"]["B"], cm_results["reverse"]["C"])

    # Forward experiment ln(Q) comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t_fwd, lnQ_fwd_true, "k-", linewidth=2, label="CM (true)")
    if single_mode.get("ln_Q_fit") is not None:
        ax1.plot(t_fwd, single_mode["ln_Q_fit"], "b--", linewidth=2, label=f'Single-mode (K={single_mode["K_fit"]:.2f})')
    if multi_mode.get("success"):
        ax1.plot(
            t_fwd,
            multi_mode["lnQ_pred_list"][0],
            "r:",
            linewidth=2,
            label=f'2-mode (K={multi_mode["k"][0]:.2f}, {multi_mode["k"][1]:.2f})',
        )
    ax1.axhline(np.log(true_Keq), color="g", linestyle=":", alpha=0.5, label="ln(Keq)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("ln(Q)")
    ax1.set_title("Forward Experiment: ln(Q)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reverse experiment ln(Q) comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t_rev, lnQ_rev_true, "k-", linewidth=2, label="CM (true)")
    if multi_mode.get("success"):
        ax2.plot(t_rev, multi_mode["lnQ_pred_list"][1], "r:", linewidth=2, label=f"2-mode (shared Keq)")
    if offset.get("success"):
        ax2.plot(t_rev, offset["lnQ_pred_list"][1], "m--", linewidth=2, label=f'Offset (u={offset["u"]:.3f})')
    ax2.axhline(np.log(true_Keq), color="g", linestyle=":", alpha=0.5, label="ln(Keq)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("ln(Q)")
    ax2.set_title("Reverse Experiment: ln(Q)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Residuals comparison for forward
    ax3 = plt.subplot(2, 3, 3)
    if single_mode.get("ln_Q_fit") is not None:
        res_single = lnQ_fwd_true - single_mode["ln_Q_fit"]
        ax3.plot(t_fwd, res_single, "b-", linewidth=1, label="Single-mode residuals")
    if multi_mode.get("success"):
        res_multi = lnQ_fwd_true - multi_mode["lnQ_pred_list"][0]
        ax3.plot(t_fwd, res_multi, "r-", linewidth=1, label="2-mode residuals")
    ax3.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Residuals")
    ax3.set_title("Forward Residuals")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Model comparison summary
    ax4 = plt.subplot(2, 3, 4)
    models = []
    costs = []
    colors = []

    if single_mode.get("r_squared_lnQ") is not None:
        models.append(f'Single\n(R²={single_mode["r_squared_lnQ"]:.3f})')
        costs.append(single_mode.get("cost", 0))
        colors.append("blue")

    if multi_mode.get("success"):
        models.append(f'2-mode\n(cost={multi_mode["cost"]:.3e})')
        costs.append(multi_mode["cost"])
        colors.append("red")

    if offset.get("success"):
        models.append(f'Offset\n(cost={offset["cost"]:.3e})')
        costs.append(offset["cost"])
        colors.append("magenta")

    if costs:
        ax4.bar(models, costs, color=colors, alpha=0.7)
        ax4.set_ylabel("Cost Function")
        ax4.set_title("Model Comparison")
        ax4.set_yscale("log")

    # Concentration comparison for forward (multi-mode)
    if multi_mode.get("success"):
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(t_fwd, cm_results["forward"]["A"], "b-", linewidth=2, label="A (CM)")
        ax5.plot(t_fwd, cm_results["forward"]["B"], "g-", linewidth=2, label="B (CM)")
        ax5.plot(t_fwd, cm_results["forward"]["C"], "r-", linewidth=2, label="C (CM)")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Concentration")
        ax5.set_title("Forward Concentrations")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Effective rate analysis for multi-mode
    if multi_mode.get("success"):
        ax6 = plt.subplot(2, 3, 6)
        k = multi_mode["k"]
        w = multi_mode["w"]

        # Compute effective rate: K_eff(t) = sum(k_i * w_i * exp(-k_i*t)) / sum(w_i * exp(-k_i*t))
        def K_eff_curve(t, k, w):
            E = np.exp(-np.outer(t, k))  # (T, M)
            num = (k * E).dot(w)
            den = E.dot(w)
            return num / den

        Keff = K_eff_curve(t_fwd, k, w)
        ax6.plot(t_fwd, Keff, "r-", linewidth=2, label="K_eff(t) from 2-mode")
        if single_mode.get("K_fit") is not None:
            ax6.axhline(single_mode["K_fit"], color="b", linestyle="--", label=f'Single K = {single_mode["K_fit"]:.2f}')
        ax6.set_xlabel("Time")
        ax6.set_ylabel("Effective Rate")
        ax6.set_title("Time-varying Effective Rate")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.suptitle("CM vs Multi-mode LLRQ Fitting Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = OUT_DIR / f"cm_multimode_comparison_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {fig_path}")
    plt.show()


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CM Multi-Mode LLRQ Fitting Comparison")
    print("=" * 80)

    # Run the complete comparison
    comparison_results = run_cm_multi_mode_comparison(seed=20250912, t_span=(0.0, 3.0), n_points=220, save_plots=True)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    single = comparison_results["single_mode"]
    multi = comparison_results["multi_mode"]
    offset = comparison_results["offset"]

    if single.get("r_squared_lnQ") is not None:
        print(f"Single-mode LLRQ:")
        print(f"  K = {single['K_fit']:.4f}")
        print(f"  R² (ln Q) = {single['r_squared_lnQ']:.4f}")

    if multi.get("success"):
        print(f"\n2-mode LLRQ:")
        print(f"  K = {multi['k']}")
        print(f"  w = {multi['w']}")
        print(f"  Keq = {multi['Keq']:.4e}")
        print(f"  Cost = {multi['cost']:.6e}")

    if offset.get("success"):
        print(f"\nOffset LLRQ:")
        print(f"  k = {offset['k']:.4f}")
        print(f"  u = {offset['u']:.4f}")
        print(f"  Keq = {offset['Keq']:.4e}")
        print(f"  Cost = {offset['cost']:.6e}")

    print(f"\nTrue Keq = {comparison_results['true_Keq']:.4e}")
