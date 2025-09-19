#!/usr/bin/env python3
"""
LLRQ fit for ATP/ADP/AMP glucose-pulse data (v2).

Key changes vs v1:
- Separate ridge penalties for K and u (alpha_K, alpha_u) so the fit doesn't dump all dynamics into u.
- Optional constraints: --no_u (forces u=0), --u_late_zero (forces u_late=0).
- CV selects alpha_K while holding alpha_u fixed (defaults to 10*alpha_K) to prefer small u.
- Clearer printing with more precision.
- Adds an option to smooth x(t) with Savitzky-Golay before differencing (helps noisy Δx).

Data file required: nucleotides_timeseries.csv
"""

import argparse, math, sys, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter

    HAVE_SG = True
except Exception:
    HAVE_SG = False


def finite_mask(a):
    return np.isfinite(a).all(axis=-1) if a.ndim > 1 else np.isfinite(a)


def moving_average(x, w=3):
    if w <= 1:
        return x.copy()
    k = int(w)
    pad = (k - 1) // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    c = np.convolve(xpad, np.ones(k) / k, mode="valid")
    return c


def nearest_spd(A, eps=1e-9):
    """Project a symmetric matrix A to the nearest SPD (clip eigenvalues)."""
    B = (A + A.T) / 2.0
    w, V = np.linalg.eigh(B)
    w_clipped = np.maximum(w, eps)
    return (V * w_clipped) @ V.T


def pretty_mat(M, prec=6):
    return "\n".join(["[" + ", ".join([f"{v:+.{prec}f}" for v in row]) + "]" for row in M])


@dataclass
class FitResult:
    K: np.ndarray
    u_early: np.ndarray
    u_late: np.ndarray
    alpha_K: float
    alpha_u: float
    Keq: np.ndarray
    cv_scores: List[Tuple[float, float]]
    resid_rms: float
    ev: np.ndarray
    evecs: np.ndarray


def build_x_times(
    df_one: pd.DataFrame, Keq: np.ndarray, sg_window: int = 0, sg_order: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    one = df_one.sort_values("time_s").copy()
    t = one["time_s"].to_numpy()
    ATP = one["ATP_mean"].to_numpy()
    ADP = one["ADP_mean"].to_numpy()
    AMP = one["AMP_mean"].to_numpy()
    Q1 = ADP / ATP
    Q2 = AMP / ADP
    x = np.c_[np.log(Q1 / Keq[0]), np.log(Q2 / Keq[1])]
    m = finite_mask(x) & np.isfinite(t)
    t, x = t[m], x[m]
    if sg_window and HAVE_SG and len(t) >= sg_window:
        # smooth x lightly to stabilize Δx, preserving endpoints
        for j in range(2):
            x[:, j] = savgol_filter(x[:, j], sg_window, sg_order, mode="interp")
    return t, x


def assemble_discrete_rows(groups, t_break: float):
    X_rows, Y_rows, T_rows = [], [], []
    for name, (t, x) in groups.items():
        if len(t) < 3:
            continue
        dt = np.diff(t)
        dt = np.maximum(dt, 1e-9)
        early = (t[:-1] <= t_break).astype(float)[:, None]
        late = 1.0 - early
        Xk = -(dt[:, None] * x[:-1])  # n x 2
        n = len(dt)
        # We'll build per-dimension designs when solving; here keep raw blocks handy
        # Store columns: [Xk_col0, Xk_col1, uE1, uL1, uE2, uL2] (we'll slice for each dim)
        X = np.zeros((n, 6))
        X[:, 0:2] = Xk
        X[:, 2:4] = dt[:, None] * np.c_[early, late]  # for dim 1
        X[:, 4:6] = dt[:, None] * np.c_[early, late]  # for dim 2 (same indicators)
        Y = x[1:] - x[:-1]  # n x 2
        X_rows.append(X)
        Y_rows.append(Y)
        T_rows.append(t[:-1])
    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    T = np.concatenate(T_rows)
    return X, Y, T


def solve_weighted_ridge(X, y, alpha_K, alpha_u, dim):
    """
    dim=0 or 1. Design columns per dim: [Xk_col0, Xk_col1, uE_dim, uL_dim].
    Apply different ridge penalties to K (first 2 cols) and u (last 2 cols).
    """
    Xd = np.c_[X[:, 0:2], X[:, 2 + 2 * dim : 4 + 2 * dim]]
    d = Xd.shape[1]
    XtX = Xd.T @ Xd
    # build diagonal ridge with separate weights
    R = np.diag([alpha_K, alpha_K, alpha_u, alpha_u])
    A = XtX + R
    b = Xd.T @ y
    theta = np.linalg.solve(A, b)
    # Map to components
    Krow = -theta[:2]
    uE = theta[2]
    uL = theta[3]
    return Krow, uE, uL


def fit_once(groups, Keq, alpha_K=1e-3, alpha_u=None, t_break=60.0, enforce_spd=True, no_u=False, u_late_zero=False):
    X, Y, _ = assemble_discrete_rows(groups, t_break)
    if alpha_u is None:
        alpha_u = 10.0 * alpha_K
    K = np.zeros((2, 2))
    uE = np.zeros(2)
    uL = np.zeros(2)
    for dim in (0, 1):
        y = Y[:, dim]
        if no_u:
            # No u: only K columns
            Xd = X[:, 0:2]
            XtX = Xd.T @ Xd
            A = XtX + np.eye(2) * alpha_K
            b = Xd.T @ y
            theta = np.linalg.solve(A, b)
            K[dim, :] = -theta[:2]
            uE[dim] = 0.0
            uL[dim] = 0.0
        else:
            Krow, uE_dim, uL_dim = solve_weighted_ridge(X, y, alpha_K, alpha_u, dim)
            K[dim, :] = Krow
            uE[dim] = uE_dim
            uL[dim] = 0.0 if u_late_zero else uL_dim

    # Symmetrize and enforce SPD
    K = (K + K.T) / 2.0
    if enforce_spd:
        K = nearest_spd(K, eps=1e-9)

    # Residuals
    # Build predictions of Δx
    X1 = np.c_[-X[:, 0:2], X[:, 2:4]]  # dim 1
    X2 = np.c_[-X[:, 0:2], X[:, 4:6]]  # dim 2
    th1 = np.r_[K[0, :], uE[0], uL[0]]
    th2 = np.r_[K[1, :], uE[1], uL[1]]
    Yhat = np.c_[X1 @ th1, X2 @ th2]
    resid = Y - Yhat
    resid_rms = float(np.sqrt((resid**2).mean()))
    w, V = np.linalg.eigh(K)
    return K, uE, uL, resid_rms, w, V


def simulate_forward(t, x0, K, u_early, u_late, t_break):
    x = np.zeros((len(t), 2))
    x[0] = x0
    for k in range(len(t) - 1):
        dt = max(t[k + 1] - t[k], 1e-9)
        u = u_early if t[k] <= t_break else u_late
        dx = -K @ x[k] + u
        x[k + 1] = x[k] + dt * dx
    return x


def loorep_cv(groups, Keq, alphaKs, t_break, alpha_u=None, no_u=False, u_late_zero=False):
    scores = []
    keys = list(groups.keys())
    for aK in alphaKs:
        ll = []
        for hold in keys:
            train = {k: v for k, v in groups.items() if k != hold}
            K, uE, uL, _, _, _ = fit_once(
                train, Keq, alpha_K=aK, alpha_u=alpha_u, t_break=t_break, enforce_spd=True, no_u=no_u, u_late_zero=u_late_zero
            )
            t, x = groups[hold]
            if len(t) < 3:
                continue
            xhat = simulate_forward(t, x[0], K, uE, uL, t_break)
            mse = float(((x - xhat) ** 2).mean())
            ll.append(-mse)
        scores.append((aK, float(np.mean(ll)) if ll else np.nan))
    return scores


def main():
    ap = argparse.ArgumentParser(description="LLRQ fit on nucleotide data (v2)")
    ap.add_argument("--csv", default="nucleotides_timeseries.csv")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--keq_from", type=float, default=300.0)
    ap.add_argument("--t_break", type=float, default=60.0)
    ap.add_argument("--alphaK_grid", type=str, default="1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--alphaK", type=float, default=None, help="Override alpha_K (skip CV)")
    ap.add_argument("--alphaU", type=float, default=None, help="Override alpha_u (default 10*alpha_K)")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--savefig", default="nucleotide_k_matrix_fit_v2.png")
    ap.add_argument("--no_u", action="store_true", help="Force u=0 for all times")
    ap.add_argument("--u_late_zero", action="store_true", help="Force u_late=0 (late drive off)")
    ap.add_argument("--sg_window", type=int, default=0, help="Savitzky-Golay window (odd, >=3). 0 disables.")
    ap.add_argument("--sg_order", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["time_s"] >= args.tmin].copy()

    tail = df[df["time_s"] >= args.keq_from]
    Keq1 = tail["ADP_mean"].median() / tail["ATP_mean"].median()
    Keq2 = tail["AMP_mean"].median() / tail["ADP_mean"].median()
    Keq = np.array([Keq1, Keq2], float)

    groups = {}
    for name, g in df.groupby("dataset"):
        t, x = build_x_times(g, Keq, sg_window=args.sg_window, sg_order=args.sg_order)
        if len(t) >= 3:
            groups[name] = (t, x)

    if len(groups) == 0:
        raise RuntimeError("No usable replicate had >=3 time points after filtering.")

    if args.alphaK is None:
        alphaKs = [float(s) for s in args.alphaK_grid.split(",")]
        cv_scores = loorep_cv(
            groups, Keq, alphaKs, args.t_break, alpha_u=args.alphaU, no_u=args.no_u, u_late_zero=args.u_late_zero
        )
        best_alphaK = sorted([s for s in cv_scores if np.isfinite(s[1])], key=lambda z: z[1], reverse=True)[0][0]
    else:
        cv_scores = []
        best_alphaK = float(args.alphaK)

    K, uE, uL, resid_rms, w, V = fit_once(
        groups,
        Keq,
        alpha_K=best_alphaK,
        alpha_u=args.alphaU,
        t_break=args.t_break,
        enforce_spd=True,
        no_u=args.no_u,
        u_late_zero=args.u_late_zero,
    )

    # Report
    print("\n=== LLRQ Fit Summary (v2) ===")
    print(f"Keq (ADP/ATP, AMP/ADP) = {Keq}")
    print(f"alpha_K (ridge on K): {best_alphaK:.2e}")
    print(f"alpha_u (ridge on u): {(args.alphaU if args.alphaU is not None else 10.0*best_alphaK):.2e}")
    print("\nK (SPD, s^-1):")
    print(pretty_mat(K, prec=6))
    print("\nEigenvalues (s^-1):", w, "  → time constants τ (s):", 1.0 / np.maximum(w, 1e-12))
    print("\nu_early:", uE, "   u_late:", uL)
    print(f"Residual RMS in Δx: {resid_rms:.4e}")

    # Plots
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3)

    axA = fig.add_subplot(gs[0, 0])
    colors = {"ATP_mean": "tab:blue", "ADP_mean": "tab:orange", "AMP_mean": "tab:green"}
    for name, g in df.groupby("dataset"):
        t = g["time_s"].to_numpy()
        for col in ["ATP_mean", "ADP_mean", "AMP_mean"]:
            axA.plot(
                t,
                g[col],
                "o-",
                ms=3,
                alpha=0.8,
                color=colors[col],
                label=col if name == list(df["dataset"].unique())[0] else None,
            )
    axA.axvline(0, color="red", ls="--", lw=1)
    axA.set_title("Nucleotide Concentrations")
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("μmol/gDW")
    axA.legend(loc="best", fontsize=8)

    axB = fig.add_subplot(gs[0, 1])
    for name, (t, x) in groups.items():
        axB.plot(t, x[:, 0], "o-", ms=3, alpha=0.85, label=f"{name} Q1 ADP/ATP")
        axB.plot(t, x[:, 1], "o-", ms=3, alpha=0.85, label=f"{name} Q2 AMP/ADP")
    axB.axhline(0, color="k", lw=1)
    axB.set_title("Log-Deviations from Equilibrium")
    axB.set_xlabel("Time (s)")
    axB.set_ylabel("ln(Q/Keq)")
    axB.legend(fontsize=7)

    # Discrete differences vs fit
    axC = fig.add_subplot(gs[0, 2])
    X, Y, tstep = assemble_discrete_rows(groups, args.t_break)
    X1 = np.c_[-X[:, 0:2], X[:, 2:4]]
    X2 = np.c_[-X[:, 0:2], X[:, 4:6]]
    Yhat = np.c_[X1 @ np.r_[K[0, :], uE[0], uL[0]], X2 @ np.r_[K[1, :], uE[1], uL[1]]]
    axC.plot(tstep, Y[:, 0], "o", ms=3, alpha=0.5, label="Δx1 data")
    axC.plot(tstep, Yhat[:, 0], "-", lw=1.5, label="Δx1 fit")
    axC.plot(tstep, Y[:, 1], "o", ms=3, alpha=0.5, label="Δx2 data")
    axC.plot(tstep, Yhat[:, 1], "-", lw=1.5, label="Δx2 fit")
    axC.axvline(args.t_break, color="gray", ls="--")
    axC.set_title("Discrete Differences vs Fit")
    axC.set_xlabel("Time (s)")
    axC.set_ylabel("Δx per step")
    axC.legend(fontsize=7)

    # CV curve (if available)
    axD = fig.add_subplot(gs[1, 0])
    if cv_scores:
        a_vals = [a for a, _ in cv_scores]
        sc = [s for _, s in cv_scores]
        axD.semilogx(a_vals, sc, "o-")
        axD.axvline(best_alphaK, color="r", ls="--")
        axD.set_title("LORO CV (−MSE)")
        axD.set_xlabel("alpha_K")
        axD.set_ylabel("Score")

    # Heatmap
    axE = fig.add_subplot(gs[1, 1])
    im = axE.imshow(K, cmap="RdBu_r")
    axE.set_xticks([0, 1])
    axE.set_xticklabels(["ATP→ADP", "ADP→AMP"], rotation=30)
    axE.set_yticks([0, 1])
    axE.set_yticklabels(["ATP→ADP", "ADP→AMP"])
    for i in range(2):
        for j in range(2):
            axE.text(j, i, f"{K[i,j]:+.4f}", ha="center", va="center", fontsize=9, color="k")
    axE.set_title("K Matrix (s⁻¹)")
    fig.colorbar(im, ax=axE, fraction=0.046, pad=0.04)

    # Forward sim on densest replicate
    axF = fig.add_subplot(gs[1, 2])
    key = max(groups.keys(), key=lambda k: len(groups[k][0]))
    t, x = groups[key]
    xhat = simulate_forward(t, x[0], K, uE, uL, args.t_break)
    axF.plot(t, x[:, 0], "o", ms=3, label="x1 data")
    axF.plot(t, xhat[:, 0], "-", lw=1.5, label="x1 sim")
    axF.plot(t, x[:, 1], "o", ms=3, label="x2 data")
    axF.plot(t, xhat[:, 1], "-", lw=1.5, label="x2 sim")
    axF.axhline(0, color="k", lw=1)
    axF.set_title(f"Forward Simulation ({key})")
    axF.set_xlabel("Time (s)")
    axF.set_ylabel("ln(Q/Keq)")
    axF.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(args.savefig, dpi=180)
    print(f"\nSaved figure → {args.savefig}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sys.exit(main())
