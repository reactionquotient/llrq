#!/usr/bin/env python3
"""
Test generalization of sparse log-linear fits across different initial conditions.

This script:
1. Fits a sparse 2-mode model to one CM trajectory
2. Generates multiple test trajectories with different initial conditions
3. Tests how well the trained model generalizes to new trajectories
4. Analyzes generalization performance across different scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import sys
import os
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# Import the log-linear fitting function and CM utilities
sys.path.append(os.path.dirname(__file__))
from lnQ_loglinear_fit_cvxpy import fit_lnQ_loglinear_cvx
from K_matrix_from_modes import build_symmetric_K_and_x0, x1_of_t

# Output directory
OUT_DIR = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CMParams:
    u: float
    kM_A: float
    kM_B: float
    kM_C: float
    k_plus: float
    k_minus: float


def rate_cm(a, b, c, p: CMParams):
    """CM rate law"""
    a_p = a / p.kM_A
    b_p = b / p.kM_B
    c_p = c / p.kM_C
    num = p.k_plus * a_p * b_p - p.k_minus * (c_p**2)
    den = (1.0 + a_p) * (1.0 + b_p) + (1.0 + c_p) ** 2 - 1.0
    return p.u * num / den


def rhs(t, y, p: CMParams):
    """ODE right-hand side for CM kinetics"""
    A, B, C = y
    v = rate_cm(A, B, C, p)
    return [-v, -v, 2.0 * v]


def simulate_cm_trajectory(t_span, y0, params: CMParams, t_eval=None, rtol=1e-8, atol=1e-10):
    """Simulate CM trajectory"""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(lambda t, y: rhs(t, y, params), t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def compute_reaction_quotient(A, B, C):
    """Compute Q = C^2 / (A * B) for reaction A + B <-> 2C"""
    eps = 1e-12
    return (C + eps) ** 2 / ((A + eps) * (B + eps))


def compute_true_keq(params):
    """Compute true equilibrium constant from CM parameters."""
    return (params.k_plus / params.k_minus) * (params.kM_C**2) / (params.kM_A * params.kM_B)


def load_training_data():
    """Load the latest CM trajectory data for training."""
    # Find the most recent trajectory files
    csv_files = list(OUT_DIR.glob("cm_timeseries_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CM trajectory files found. Run cm_rate_law_integrated.py first.")

    # Get forward trajectory for training
    forward_files = [f for f in csv_files if "forward" in f.name]
    if not forward_files:
        raise FileNotFoundError("No forward trajectory file found for training.")

    forward_file = max(forward_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading training data from: {forward_file.name}")

    forward_df = pd.read_csv(forward_file)

    # Load parameters
    param_files = list(OUT_DIR.glob("cm_params_*.json"))
    if param_files:
        param_file = max(param_files, key=lambda x: x.stat().st_mtime)
        with open(param_file) as f:
            params_dict = json.load(f)
        params = CMParams(**params_dict)
    else:
        raise FileNotFoundError("No parameter file found. Run cm_rate_law_integrated.py first.")

    return forward_df, params


def fit_sparse_model(t, ln_Q, true_keq=None):
    """Fit sparse model with cardinality constraint."""
    print("Fitting sparse model with cardinality=2...")

    fit_params = {
        "m": 60,
        "alpha": 1e-2,
        "beta": 1e-5,
        "gamma_l1": 1e-4,
        "cardinality": 2,
        "solver": "MOSEK",
    }

    fit = fit_lnQ_loglinear_cvx(
        t,
        ln_Q,
        m=fit_params["m"],
        alpha=fit_params["alpha"],
        beta=fit_params["beta"],
        gamma_l1=fit_params["gamma_l1"],
        cardinality=fit_params["cardinality"],
        solver=fit_params["solver"],
    )

    if fit is None:
        raise RuntimeError("Sparse model fitting failed")

    print(f"Training fit: R² = {fit.r2:.4f}, K_eq = {fit.K_eq():.4e}")
    print(f"Active modes: {fit.active_idx.size}/{fit.lambdas.size}")

    if true_keq:
        keq_error = abs(fit.K_eq() - true_keq) / true_keq * 100
        print(f"True K_eq = {true_keq:.4e}, Error = {keq_error:.1f}%")

    return fit


def generate_test_trajectories(params, n_tests=12):
    """Generate test trajectories with different initial conditions."""
    print(f"\nGenerating {n_tests} test trajectories...")

    test_cases = []
    t_eval = np.linspace(0.0, 2.5, 200)

    # Test case scenarios
    scenarios = [
        # Various forward reactions
        {"name": "forward_balanced", "y0": [1.0 * params.kM_A, 1.0 * params.kM_B, 0.1 * params.kM_C]},
        {"name": "forward_A_excess", "y0": [2.0 * params.kM_A, 0.5 * params.kM_B, 0.05 * params.kM_C]},
        {"name": "forward_B_excess", "y0": [0.5 * params.kM_A, 2.0 * params.kM_B, 0.05 * params.kM_C]},
        {"name": "forward_high_start", "y0": [3.0 * params.kM_A, 3.0 * params.kM_B, 0.02 * params.kM_C]},
        # Various reverse reactions
        {"name": "reverse_balanced", "y0": [0.05 * params.kM_A, 0.05 * params.kM_B, 1.0 * params.kM_C]},
        {"name": "reverse_high_C", "y0": [0.02 * params.kM_A, 0.02 * params.kM_B, 2.0 * params.kM_C]},
        {"name": "reverse_some_AB", "y0": [0.2 * params.kM_A, 0.2 * params.kM_B, 1.5 * params.kM_C]},
        # Near equilibrium starts
        {"name": "near_eq_1", "y0": [0.3 * params.kM_A, 0.3 * params.kM_B, 0.8 * params.kM_C]},
        {"name": "near_eq_2", "y0": [0.6 * params.kM_A, 0.6 * params.kM_B, 0.5 * params.kM_C]},
        # Extreme cases
        {"name": "very_low_start", "y0": [0.1 * params.kM_A, 0.1 * params.kM_B, 0.01 * params.kM_C]},
        {"name": "mixed_extreme", "y0": [0.01 * params.kM_A, 3.0 * params.kM_B, 0.5 * params.kM_C]},
        {"name": "all_moderate", "y0": [0.8 * params.kM_A, 0.8 * params.kM_B, 0.3 * params.kM_C]},
    ]

    for i, scenario in enumerate(scenarios[:n_tests]):
        print(f"  Generating {scenario['name']}...")
        t, A, B, C = simulate_cm_trajectory((0.0, 2.5), scenario["y0"], params, t_eval=t_eval)
        Q = compute_reaction_quotient(A, B, C)
        ln_Q = np.log(Q)

        test_cases.append(
            {"name": scenario["name"], "t": t, "A": A, "B": B, "C": C, "Q": Q, "ln_Q": ln_Q, "y0": scenario["y0"]}
        )

    return test_cases


def evaluate_model_on_trajectory(trained_fit, t, ln_Q_true):
    """Evaluate trained model on a new trajectory."""
    # Use the trained model parameters to predict ln(Q)
    # Get active modes
    active_lambdas = trained_fit.K.diagonal()
    active_weights = trained_fit.z0 / trained_fit.s

    # Get equilibrium value and initial value
    ln_Keq = trained_fit.b  # Equilibrium value from the fit
    ln_Q_initial = ln_Q_true[0]  # Initial value of this trajectory

    # Calculate initial displacement from equilibrium
    displacement = ln_Q_initial - ln_Keq

    # Compute the exponential decay terms that go from 1 to 0
    # Normalize the weights so they sum to 1 at t=0
    weight_sum = np.sum(active_weights)
    if abs(weight_sum) < 1e-12:
        # Fallback if weights are degenerate
        normalized_weights = np.ones_like(active_weights) / len(active_weights)
    else:
        normalized_weights = active_weights / weight_sum

    # Compute exponential terms that decay from 1 to 0
    exp_decay = (normalized_weights[:, np.newaxis] * np.exp(-active_lambdas[:, np.newaxis] * t)).sum(axis=0)

    # The correct exponential relaxation model:
    # ln_Q(t) = ln_Keq + displacement * exp_decay
    # This naturally converges to ln_Keq as t -> infinity
    ln_Q_pred = ln_Keq + displacement * exp_decay

    # Calculate metrics
    r2 = 1 - np.sum((ln_Q_true - ln_Q_pred) ** 2) / np.sum((ln_Q_true - np.mean(ln_Q_true)) ** 2)
    rmse = np.sqrt(np.mean((ln_Q_true - ln_Q_pred) ** 2))

    return ln_Q_pred, r2, rmse


def reconstruct_concentrations_from_lnQ(t, ln_Q_fit, A_orig, B_orig, C_orig):
    """Reconstruct concentrations A(t), B(t), C(t) from fitted ln(Q) using conservation laws."""
    # Initial concentrations
    A0, B0, C0 = A_orig[0], B_orig[0], C_orig[0]

    # Conservation constants for A + B <-> 2C
    alpha = A0 + C0 / 2
    beta = B0 + C0 / 2

    Q_fit = np.exp(ln_Q_fit)

    # For each time point, solve Q = C^2/((alpha - C/2)(beta - C/2)) for C
    C_fit = np.zeros_like(t)
    A_fit = np.zeros_like(t)
    B_fit = np.zeros_like(t)

    for i, Q_val in enumerate(Q_fit):
        # Quadratic equation coefficients
        a = Q_val / 4 - 1
        b = -Q_val * (alpha + beta) / 2
        c = Q_val * alpha * beta

        # Solve quadratic equation
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            C_fit[i] = C_orig[i]  # Fallback
        else:
            sqrt_disc = np.sqrt(discriminant)
            C1 = (-b + sqrt_disc) / (2 * a) if abs(a) > 1e-12 else -c / b
            C2 = (-b - sqrt_disc) / (2 * a) if abs(a) > 1e-12 else -c / b

            # Choose physically meaningful solution
            A1 = alpha - C1 / 2
            B1 = beta - C1 / 2
            A2 = alpha - C2 / 2
            B2 = beta - C2 / 2

            if C1 >= 0 and A1 >= 0 and B1 >= 0:
                if C2 >= 0 and A2 >= 0 and B2 >= 0:
                    # Both valid, pick closer to original
                    err1 = abs(C1 - C_orig[i])
                    err2 = abs(C2 - C_orig[i])
                    C_fit[i] = C1 if err1 <= err2 else C2
                else:
                    C_fit[i] = C1
            elif C2 >= 0 and A2 >= 0 and B2 >= 0:
                C_fit[i] = C2
            else:
                C_fit[i] = C_orig[i]  # Fallback

        # Compute A and B from conservation laws
        A_fit[i] = alpha - C_fit[i] / 2
        B_fit[i] = beta - C_fit[i] / 2

    return A_fit, B_fit, C_fit


def test_generalization(trained_fit, test_cases, params):
    """Test generalization of trained model on test cases."""
    print("\nTesting generalization across different initial conditions...")

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"  Testing {test_case['name']}...")

        # Apply trained model (no refitting!)
        ln_Q_pred, r2, rmse_lnQ = evaluate_model_on_trajectory(trained_fit, test_case["t"], test_case["ln_Q"])
        # ln_Q_pred, r2, rmse_lnQ = evaluate_model_on_trajectory_with_refit(
        #     trained_fit, test_case["t"], test_case["ln_Q"], A=test_case["A"], B=test_case["B"], C=test_case["C"], params=params
        # )

        # Reconstruct concentrations
        A_fit, B_fit, C_fit = reconstruct_concentrations_from_lnQ(
            test_case["t"], ln_Q_pred, test_case["A"], test_case["B"], test_case["C"]
        )

        # Calculate concentration RMSEs
        rmse_A = np.sqrt(np.mean((test_case["A"] - A_fit) ** 2))
        rmse_B = np.sqrt(np.mean((test_case["B"] - B_fit) ** 2))
        rmse_C = np.sqrt(np.mean((test_case["C"] - C_fit) ** 2))

        # Calculate R² in concentration space (overall for A, B, C combined)
        # Stack all true and predicted concentrations
        conc_true = np.hstack([test_case["A"], test_case["B"], test_case["C"]])
        conc_pred = np.hstack([A_fit, B_fit, C_fit])
        r2_conc = 1 - np.sum((conc_true - conc_pred) ** 2) / np.sum((conc_true - np.mean(conc_true)) ** 2)

        result = {
            "name": test_case["name"],
            "r2": r2,
            "r2_conc": r2_conc,
            "rmse_lnQ": rmse_lnQ,
            "rmse_A": rmse_A,
            "rmse_B": rmse_B,
            "rmse_C": rmse_C,
            "ln_Q_true": test_case["ln_Q"],
            "ln_Q_pred": ln_Q_pred,
            "A_true": test_case["A"],
            "B_true": test_case["B"],
            "C_true": test_case["C"],
            "A_fit": A_fit,
            "B_fit": B_fit,
            "C_fit": C_fit,
            "t": test_case["t"],
            "y0": test_case["y0"],
        }
        results.append(result)

        print(f"    R²(lnQ) = {r2:.4f}, R²(conc) = {r2_conc:.4f}, RMSE(ln Q) = {rmse_lnQ:.4f}")
        print(f"    Concentration RMSE: A={rmse_A:.4f}, B={rmse_B:.4f}, C={rmse_C:.4f}")

    return results


def plot_generalization_results(results):
    """Create comprehensive visualization of generalization results."""
    n_cases = len(results)

    # Main grid plot: ln(Q) predictions vs actual
    fig1 = plt.figure(figsize=(16, 12))

    for i, result in enumerate(results):
        ax = plt.subplot(3, 4, i + 1)
        ax.plot(result["t"], result["ln_Q_true"], "k-", linewidth=2, label="True", alpha=0.8)
        ax.plot(result["t"], result["ln_Q_pred"], "r--", linewidth=2, label="Predicted", alpha=0.7)
        ax.set_title(f"{result['name']}\nR²={result['r2']:.3f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("ln(Q)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Generalization Test: ln(Q) Predictions vs True Values", fontsize=16)
    plt.tight_layout()

    # Save plot
    filepath1 = OUT_DIR / "generalization_lnQ_grid.png"
    plt.savefig(filepath1, dpi=150, bbox_inches="tight")
    print(f"Saved ln(Q) grid plot to: {filepath1}")

    # Concentration trajectories grid plot
    fig3 = plt.figure(figsize=(16, 12))

    for i, result in enumerate(results):
        ax = plt.subplot(3, 4, i + 1)

        # Plot true concentrations (solid lines)
        ax.plot(result["t"], result["A_true"], "b-", linewidth=2, label="A (true)", alpha=0.8)
        ax.plot(result["t"], result["B_true"], "r-", linewidth=2, label="B (true)", alpha=0.8)
        ax.plot(result["t"], result["C_true"], "g-", linewidth=2, label="C (true)", alpha=0.8)

        # Plot predicted concentrations (dashed lines)
        ax.plot(result["t"], result["A_fit"], "b--", linewidth=2, label="A (pred)", alpha=0.7)
        ax.plot(result["t"], result["B_fit"], "r--", linewidth=2, label="B (pred)", alpha=0.7)
        ax.plot(result["t"], result["C_fit"], "g--", linewidth=2, label="C (pred)", alpha=0.7)

        ax.set_title(f"{result['name']}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        if i == 0:  # Only show legend on first subplot to avoid clutter
            ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Generalization Test: Concentration Predictions vs True Values", fontsize=16)
    plt.tight_layout()

    # Save concentration plot
    filepath3 = OUT_DIR / "generalization_concentration_grid.png"
    plt.savefig(filepath3, dpi=150, bbox_inches="tight")
    print(f"Saved concentration grid plot to: {filepath3}")

    # Performance summary plot
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    names = [r["name"] for r in results]
    r2_values = [r["r2"] for r in results]
    rmse_lnQ_values = [r["rmse_lnQ"] for r in results]

    # R² bar plot
    bars1 = ax1.bar(range(len(names)), r2_values, color="steelblue", alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("R²")
    ax1.set_title("R² Score Across Test Cases")
    ax1.grid(True, alpha=0.3)

    # Add R² values on bars
    for bar, r2 in zip(bars1, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.005, f"{r2:.3f}", ha="center", va="bottom", fontsize=8)

    # RMSE ln(Q) plot
    bars2 = ax2.bar(range(len(names)), rmse_lnQ_values, color="darkorange", alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("RMSE ln(Q)")
    ax2.set_title("ln(Q) RMSE Across Test Cases")
    ax2.grid(True, alpha=0.3)

    # Concentration RMSE heatmap
    conc_rmse_data = np.array([[r["rmse_A"], r["rmse_B"], r["rmse_C"]] for r in results])
    im = ax3.imshow(conc_rmse_data.T, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["A", "B", "C"])
    ax3.set_title("Concentration RMSE Heatmap")
    plt.colorbar(im, ax=ax3, shrink=0.8)

    # Overall performance scatter
    total_conc_rmse = [np.sqrt((r["rmse_A"] ** 2 + r["rmse_B"] ** 2 + r["rmse_C"] ** 2) / 3) for r in results]
    ax4.scatter(r2_values, total_conc_rmse, c=rmse_lnQ_values, s=60, alpha=0.7, cmap="viridis")
    ax4.set_xlabel("R² (ln Q)")
    ax4.set_ylabel("Total Concentration RMSE")
    ax4.set_title("Performance Overview")
    ax4.grid(True, alpha=0.3)

    # Add case names as annotations
    for i, name in enumerate(names):
        ax4.annotate(
            name.replace("_", " "),
            (r2_values[i], total_conc_rmse[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    plt.colorbar(ax4.collections[0], ax=ax4, label="RMSE ln(Q)", shrink=0.8)

    plt.suptitle("Generalization Performance Summary", fontsize=16)
    plt.tight_layout()

    # Save plot
    filepath2 = OUT_DIR / "generalization_summary.png"
    plt.savefig(filepath2, dpi=150, bbox_inches="tight")
    print(f"Saved performance summary to: {filepath2}")

    # Show plots
    try:
        plt.show()
    except Exception as e:
        print(f"Display warning: {e}")


def generate_report(trained_fit, results, true_keq):
    """Generate comprehensive generalization report."""
    print("\n" + "=" * 80)
    print("GENERALIZATION TEST REPORT")
    print("=" * 80)

    print(f"\nTraining Model Summary:")
    print(f"  Sparse model: {trained_fit.active_idx.size} active modes out of {trained_fit.lambdas.size}")
    print(f"  Training R²: {trained_fit.r2:.4f}")
    print(f"  Fitted K_eq: {trained_fit.K_eq():.4e}")
    if true_keq:
        print(f"  True K_eq: {true_keq:.4e}")
        print(f"  K_eq error: {abs(trained_fit.K_eq() - true_keq)/true_keq*100:.1f}%")

    print(f"\nGeneralization Results Across {len(results)} Test Cases:")
    print("-" * 80)
    print(f"{'Case Name':<20} {'R²':<8} {'RMSE(lnQ)':<12} {'RMSE(A)':<10} {'RMSE(B)':<10} {'RMSE(C)':<10}")
    print("-" * 80)

    r2_values = []
    rmse_lnQ_values = []

    for result in results:
        r2_values.append(result["r2"])
        rmse_lnQ_values.append(result["rmse_lnQ"])
        print(
            f"{result['name']:<20} {result['r2']:<8.4f} {result['rmse_lnQ']:<12.4f} "
            f"{result['rmse_A']:<10.4f} {result['rmse_B']:<10.4f} {result['rmse_C']:<10.4f}"
        )

    print("-" * 80)

    # Summary statistics
    avg_r2 = np.mean(r2_values)
    min_r2 = np.min(r2_values)
    max_r2 = np.max(r2_values)
    std_r2 = np.std(r2_values)

    avg_rmse_lnQ = np.mean(rmse_lnQ_values)

    print(f"\nSummary Statistics:")
    print(f"  Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"  R² range: [{min_r2:.4f}, {max_r2:.4f}]")
    print(f"  Average RMSE(ln Q): {avg_rmse_lnQ:.4f}")

    # Performance categories
    excellent = sum(1 for r2 in r2_values if r2 > 0.95)
    good = sum(1 for r2 in r2_values if 0.9 <= r2 <= 0.95)
    fair = sum(1 for r2 in r2_values if 0.8 <= r2 < 0.9)
    poor = sum(1 for r2 in r2_values if r2 < 0.8)

    print(f"\nPerformance Distribution:")
    print(f"  Excellent (R² > 0.95): {excellent}/{len(results)} cases ({excellent/len(results)*100:.1f}%)")
    print(f"  Good (0.9 ≤ R² ≤ 0.95): {good}/{len(results)} cases ({good/len(results)*100:.1f}%)")
    print(f"  Fair (0.8 ≤ R² < 0.9): {fair}/{len(results)} cases ({fair/len(results)*100:.1f}%)")
    print(f"  Poor (R² < 0.8): {poor}/{len(results)} cases ({poor/len(results)*100:.1f}%)")

    print(f"\nConclusion:")
    if avg_r2 > 0.9:
        print(f"The sparse 2-mode model shows EXCELLENT generalization across diverse initial conditions.")
    elif avg_r2 > 0.8:
        print(f"The sparse 2-mode model shows GOOD generalization across diverse initial conditions.")
    elif avg_r2 > 0.7:
        print(f"The sparse 2-mode model shows FAIR generalization across diverse initial conditions.")
    else:
        print(f"The sparse 2-mode model shows POOR generalization across diverse initial conditions.")

    print(f"This suggests the sparse exponential mixture approximation captures ")
    print(f"{'universal' if avg_r2 > 0.9 else 'somewhat universal' if avg_r2 > 0.8 else 'limited'} ")
    print(f"dynamics of the CM system beyond the specific training trajectory.")


def main():
    """Main generalization test function."""
    print("=" * 60)
    print("SPARSE MODEL GENERALIZATION TEST")
    print("=" * 60)

    # 1. Load training data and fit sparse model
    print("\n1. Loading training data and fitting sparse model...")
    train_df, params = load_training_data()

    t = train_df["t"].values
    A = train_df["A"].values
    B = train_df["B"].values
    C = train_df["C"].values
    Q = compute_reaction_quotient(A, B, C)
    ln_Q = np.log(Q)

    true_keq = compute_true_keq(params)
    print(f"True equilibrium constant: {true_keq:.4e}")

    # Fit sparse model
    trained_fit = fit_sparse_model(t, ln_Q, true_keq)

    # 2. Generate test trajectories
    print(f"\n2. Generating test trajectories with different initial conditions...")
    test_cases = generate_test_trajectories(params, n_tests=12)

    # 3. Test generalization
    print(f"\n3. Testing generalization...")
    results = test_generalization(trained_fit, test_cases, params)

    # 4. Visualization
    print(f"\n4. Creating visualization...")
    plot_generalization_results(results)

    # 5. Generate report
    print(f"\n5. Generating report...")
    generate_report(trained_fit, results, true_keq)

    print(f"\nGeneralization test complete!")


if __name__ == "__main__":
    main()
