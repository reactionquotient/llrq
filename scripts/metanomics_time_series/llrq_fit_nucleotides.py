#!/usr/bin/env python3
"""
LLRQ fit for ATP/ADP/AMP glucose-pulse data.

Implements discrete-time log-linear reaction quotient dynamics:
  x_{k+1} - x_k ≈ -Δt K x_k + Δt u(t_k)
with x = ln(Q/Keq), Q1 = ADP/ATP, Q2 = AMP/ADP.

- Estimates Keq from late-time data
- Fits symmetric positive-definite K with ridge (α)
- Fits piecewise-constant drive u in two segments (early/late)
- Leave-one-replicate-out CV over α grid
- Bootstrap CIs for K entries and eigenvalues
- Plots and saves figure 'nucleotide_k_matrix_fit.png'

Data file required: nucleotides_timeseries.csv
"""

import argparse, math, sys, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- Utilities ---------------------------------


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
    """Project a symmetric matrix A to the nearest SPD (Higham 2002, simplified)."""
    # symmetrize
    B = (A + A.T) / 2.0
    # eigen clip
    w, V = np.linalg.eigh(B)
    w_clipped = np.maximum(w, eps)
    return (V * w_clipped) @ V.T


def pretty_mat(M, prec=4):
    return "\n".join(["[" + ", ".join([f"{v:+.{prec}f}" for v in row]) + "]" for row in M])


# -------------------------- Core LLRQ pieces ---------------------------


@dataclass
class FitResult:
    K: np.ndarray  # 2x2 SPD matrix
    u_early: np.ndarray  # length-2
    u_late: np.ndarray  # length-2
    alpha: float
    Keq: np.ndarray  # length-2, for (Q1,Q2)
    cv_scores: List[Tuple[float, float]]  # (alpha, mean_LL) pairs
    resid_rms: float
    ev: np.ndarray  # eigenvalues of K
    evecs: np.ndarray  # eigenvectors of K
    boot_samples: Dict[str, np.ndarray]  # bootstrap arrays


def build_x_times(df_one: pd.DataFrame, Keq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return times t and x_k = ln(Q/Keq) for one replicate."""
    one = df_one.sort_values("time_s").copy()
    t = one["time_s"].to_numpy()
    ATP = one["ATP_mean"].to_numpy()
    ADP = one["ADP_mean"].to_numpy()
    AMP = one["AMP_mean"].to_numpy()
    Q1 = ADP / ATP
    Q2 = AMP / ADP
    x = np.c_[np.log(Q1 / Keq[0]), np.log(Q2 / Keq[1])]
    m = finite_mask(x) & np.isfinite(t)
    return t[m], x[m]


def assemble_discrete_rows(
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]], t_break: float, smooth_dt: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, Y, meta_t) rows over all replicates for regression."""
    X_rows, Y_rows, T_rows = [], [], []
    for name, (t, x) in groups.items():
        if len(t) < 3:
            continue
        dt = np.diff(t)
        # guard small/zero dt
        dt = np.maximum(dt, 1e-9)
        if smooth_dt:
            dt = moving_average(dt, w=3)
        early = (t[:-1] <= t_break).astype(float)[:, None]
        late = 1.0 - early
        # features: [-Δt * x_k] (2x2), + Δt*u_early (2), + Δt*u_late (2)
        n = len(dt)
        X = np.zeros((n, 2 * 2 + 2 + 2))
        # K block: Y ≈ (-Δt)*K*x_k + Δt*u_seg  → flatten by columns
        # We'll solve row-wise later; here we prepare shared features.
        # For convenience, we'll fit each reaction dimension separately.
        Xk = -(dt[:, None] * x[:-1])  # n x 2
        # Place columns for K acting on x: we'll duplicate this when solving per-dimension
        # We just store Xk and append segment indicators for u
        X[:, :2] = Xk
        X[:, 2:4] = Xk  # placeholder; we'll slice properly per-dim
        # u features
        X[:, 4:6] = dt[:, None] * early
        X[:, 6:8] = dt[:, None] * late
        Y = x[1:] - x[:-1]  # n x 2
        X_rows.append(X)
        Y_rows.append(Y)
        T_rows.append(t[:-1])
    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    meta_t = np.concatenate(T_rows)
    return X, Y, meta_t


def solve_ridge_for_dim(X, y, alpha):
    """Closed-form ridge: θ = (X^T X + αI)^(-1) X^T y."""
    d = X.shape[1]
    XtX = X.T @ X
    A = XtX + alpha * np.eye(d)
    b = X.T @ y
    theta = np.linalg.solve(A, b)
    return theta


def pack_params(theta_dim1, theta_dim2):
    # Each theta has length 8 with structure:
    # columns: [K* x columns for dim?, u_early(2), u_late(2)]
    # We'll reconstruct K (2x2): rows are reaction dims
    # For our X construction: first 2 columns belong to dim-1 K row, next 2 to dim-2 K row.
    K = np.zeros((2, 2))
    K[0, :] = -np.array(theta_dim1[:2])  # minus sign because X used -Δt x_k
    K[1, :] = -np.array(theta_dim2[2:4])  # note: from second theta, columns 2:4
    u_early = np.array([theta_dim1[4], theta_dim2[4]])
    u_late = np.array([theta_dim1[6], theta_dim2[6]])
    return K, u_early, u_late


def fit_once(groups, Keq, alpha=1e-2, t_break=60.0, enforce_spd=True):
    X, Y, _ = assemble_discrete_rows(groups, t_break)
    # Build per-dimension design by selecting appropriate columns
    # dim1 uses [0:2, 4,6]; dim2 uses [2:4, 4,6]
    X1 = X[:, [0, 1, 4, 6]]  # K row 1 cols + u_early1 + u_late1
    X2 = X[:, [2, 3, 4, 6]]  # K row 2 cols + u_early2 + u_late2
    theta1 = solve_ridge_for_dim(X1, Y[:, 0], alpha)
    theta2 = solve_ridge_for_dim(X2, Y[:, 1], alpha)
    # Map back
    # Expand to common 8-len layout to reuse packer
    theta1_full = np.zeros(8)
    theta2_full = np.zeros(8)
    theta1_full[:2] = theta1[:2]
    theta1_full[4] = theta1[2]
    theta1_full[6] = theta1[3]
    theta2_full[2:4] = theta2[:2]
    theta2_full[4] = theta2[2]
    theta2_full[6] = theta2[3]
    K, u_early, u_late = pack_params(theta1_full, theta2_full)
    # Enforce symmetry → SPD
    K = (K + K.T) / 2.0
    if enforce_spd:
        K = nearest_spd(K, eps=1e-9)
    # Residual RMS (for reporting)
    # Predict Δx:
    Xpred1 = np.c_[-X[:, 0:2], X[:, 4], X[:, 6]] @ np.r_[K[0, :], u_early[0], u_late[0]]
    Xpred2 = np.c_[-X[:, 2:4], X[:, 4], X[:, 6]] @ np.r_[K[1, :], u_early[1], u_late[1]]
    Yhat = np.c_[Xpred1, Xpred2]
    resid = Y - Yhat
    resid_rms = float(np.sqrt((resid**2).mean()))
    w, V = np.linalg.eigh(K)
    return K, u_early, u_late, resid_rms, w, V


def simulate_forward(t, x0, K, u_early, u_late, t_break):
    """Euler simulate x over times t using fitted K and piecewise u."""
    x = np.zeros((len(t), 2))
    x[0] = x0
    for k in range(len(t) - 1):
        dt = max(t[k + 1] - t[k], 1e-9)
        u = u_early if t[k] <= t_break else u_late
        dx = -K @ x[k] + u
        x[k + 1] = x[k] + dt * dx
    return x


def loorep_cv(groups_all, Keq, alphas, t_break):
    scores = []
    keys = list(groups_all.keys())
    for a in alphas:
        ll = []
        for hold in keys:
            train = {k: v for k, v in groups_all.items() if k != hold}
            K, uE, uL, _, _, _ = fit_once(train, Keq, alpha=a, t_break=t_break)
            # score on held-out by simulating forward from its first point
            t, x = groups_all[hold]
            if len(t) < 3:
                continue
            xhat = simulate_forward(t, x[0], K, uE, uL, t_break)
            # negative MSE as "log-likelihood-like"
            mse = float(((x - xhat) ** 2).mean())
            ll.append(-mse)
        scores.append((a, float(np.mean(ll)) if ll else np.nan))
    return scores


def bootstrap_params(groups, Keq, alpha, t_break, B=500, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    K_list, ev_list = [], []
    for b in range(B):
        # sample time rows with replacement within each replicate (pairs of consecutive points)
        boot_groups = {}
        for k in keys:
            t, x = groups[k]
            if len(t) < 3:
                boot_groups[k] = (t, x)
                continue
            idx = np.arange(len(t))
            # resample contiguous steps by index permutation with replacement
            # We'll just keep original order to maintain monotonic t, but subsample steps.
            step_idx = rng.choice(np.arange(len(t) - 1), size=len(t) - 1, replace=True)
            # rebuild time and x with chosen steps
            t_new = [t[0]]
            x_new = [x[0]]
            for s in step_idx:
                t_new.append(t[s + 1])
                x_new.append(x[s + 1])
            boot_groups[k] = (np.array(t_new), np.vstack(x_new))
        K, uE, uL, _, w, _ = fit_once(boot_groups, Keq, alpha=alpha, t_break=t_break)
        K_list.append(K)
        ev_list.append(w)
    Ks = np.stack(K_list, axis=0)
    evs = np.stack(ev_list, axis=0)
    return {"K": Ks, "eigs": evs}


# -------------------------- Pipeline -----------------------------------


def main():
    ap = argparse.ArgumentParser(description="LLRQ fit on nucleotide data")
    ap.add_argument("--csv", default="nucleotides_timeseries.csv")
    ap.add_argument("--tmin", type=float, default=0.0, help="Use data with time >= tmin")
    ap.add_argument("--keq_from", type=float, default=300.0, help="Estimate Keq using time >= this")
    ap.add_argument("--t_break", type=float, default=60.0, help="Break point between early/late u (seconds)")
    ap.add_argument(
        "--alpha_grid",
        type=str,
        default="1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2",
        help="Comma-separated ridge values",
    )
    ap.add_argument("--alpha", type=float, default=None, help="Override α (skip CV)")
    ap.add_argument("--bootstrap", type=int, default=400, help="Bootstrap samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--savefig", default="nucleotide_k_matrix_fit.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["time_s"] >= args.tmin].copy()
    # Estimate Keq pooled from late times
    tail = df[df["time_s"] >= args.keq_from]
    Keq1 = tail["ADP_mean"].median() / tail["ATP_mean"].median()
    Keq2 = tail["AMP_mean"].median() / tail["ADP_mean"].median()
    Keq = np.array([Keq1, Keq2], float)

    # Build per-replicate x(t)
    groups = {}
    for name, g in df.groupby("dataset"):
        t, x = build_x_times(g, Keq)
        if len(t) >= 3:
            groups[name] = (t, x)

    if len(groups) == 0:
        raise RuntimeError("No usable replicate had >=3 time points after filtering.")

    # CV over alpha, unless alpha provided
    if args.alpha is None:
        alphas = [float(s) for s in args.alpha_grid.split(",")]
        cv_scores = loorep_cv(groups, Keq, alphas, args.t_break)
        # pick best α (max mean score)
        best_alpha = sorted([s for s in cv_scores if np.isfinite(s[1])], key=lambda z: z[1], reverse=True)[0][0]
    else:
        cv_scores = []
        best_alpha = float(args.alpha)

    # Final fit
    K, uE, uL, resid_rms, w, V = fit_once(groups, Keq, alpha=best_alpha, t_break=args.t_break)

    # Bootstrap CIs
    boots = bootstrap_params(groups, Keq, alpha=best_alpha, t_break=args.t_break, B=args.bootstrap, seed=args.seed)
    K_b = boots["K"]
    ev_b = boots["eigs"]
    K_ci = np.quantile(K_b, [0.16, 0.84], axis=0)  # ~68% CI
    ev_ci = np.quantile(ev_b, [0.16, 0.84], axis=0)

    # ----------- Reporting -----------
    print("\n=== LLRQ Fit Summary ===")
    print(f"Keq (ADP/ATP, AMP/ADP) = {Keq}")
    print(f"α (ridge): {best_alpha:.2e}")
    print("\nK (SPD, s^-1):")
    print(pretty_mat(K))
    print("\n68% CI for K entries (lower, upper):")
    for i in range(2):
        for j in range(2):
            lo, hi = K_ci[0, i, j], K_ci[1, i, j]
            print(f"K[{i},{j}] ∈ [{lo:+.4f}, {hi:+.4f}]")
    print("\nEigenvalues (s^-1) and ~68% CI:")
    for i, lam in enumerate(w):
        print(f"λ{i+1} = {lam:.5f}  CI≈ [{ev_ci[0,i]:.5f}, {ev_ci[1,i]:.5f}]  → τ≈ {1.0/lam:.1f} s")
    print("\nu_early:", uE, "   u_late:", uL)
    print(f"Residual RMS in Δx: {resid_rms:.4e}")

    # ----------- Plots -----------
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3)

    # Panel A: concentrations
    axA = fig.add_subplot(gs[0, 0])
    colors = {"ATP_mean": "tab:blue", "ADP_mean": "tab:orange", "AMP_mean": "tab:green"}
    for name, g in df.groupby("dataset"):
        t = g["time_s"].to_numpy()
        for col in ["ATP_mean", "ADP_mean", "AMP_mean"]:
            axA.plot(t, g[col], "o-", ms=3, alpha=0.8, color=colors[col], label=col if name == "pulse_1" else None)
    axA.axvline(0, color="red", ls="--", lw=1)
    axA.set_title("Nucleotide Concentrations")
    axA.set_xlabel("Time (s)")
    axA.set_ylabel("μmol/gDW")
    axA.legend(loc="best", fontsize=8)

    # Panel B: x=ln(Q/Keq)
    axB = fig.add_subplot(gs[0, 1])
    for name, (t, x) in groups.items():
        axB.plot(t, x[:, 0], "o-", ms=3, alpha=0.85, label=f"{name} Q1 ADP/ATP")
        axB.plot(t, x[:, 1], "o-", ms=3, alpha=0.85, label=f"{name} Q2 AMP/ADP")
    axB.axhline(0, color="k", lw=1)
    axB.set_title("Log-Deviations from Equilibrium")
    axB.set_xlabel("Time (s)")
    axB.set_ylabel("ln(Q/Keq)")
    axB.legend(fontsize=7)

    # Panel C: discrete differences vs fit (derivative-like)
    axC = fig.add_subplot(gs[0, 2])
    X, Y, tstep = assemble_discrete_rows(groups, args.t_break)
    X1 = X[:, [0, 1, 4, 6]]
    X2 = X[:, [2, 3, 4, 6]]
    Yhat1 = X1 @ np.r_[-K[0, :], uE[0], uL[0]]
    Yhat2 = X2 @ np.r_[-K[1, :], uE[1], uL[1]]
    axC.plot(tstep, Y[:, 0], "o", ms=3, alpha=0.5, label="Δx1 data")
    axC.plot(tstep, Yhat1, "-", lw=1.5, label="Δx1 fit")
    axC.plot(tstep, Y[:, 1], "o", ms=3, alpha=0.5, label="Δx2 data")
    axC.plot(tstep, Yhat2, "-", lw=1.5, label="Δx2 fit")
    axC.axvline(args.t_break, color="gray", ls="--")
    axC.set_title("Discrete Differences vs Fit")
    axC.set_xlabel("Time (s)")
    axC.set_ylabel("Δx per step")
    axC.legend(fontsize=7)

    # Panel D: CV curve
    axD = fig.add_subplot(gs[1, 0])
    if len(
        cv_scores := loorep_cv(
            groups,
            Keq,
            [best_alpha] if not args.alpha is None else [float(s) for s in args.alpha_grid.split(",")],
            args.t_break,
        )
    ):
        a_vals = [a for a, _ in cv_scores]
        sc = [s for _, s in cv_scores]
        axD.semilogx(a_vals, sc, "o-")
        axD.axvline(best_alpha, color="r", ls="--")
        axD.set_title("Leave-one-replicate-out CV (−MSE)")
        axD.set_xlabel("α")
        axD.set_ylabel("Score")

    # Panel E: heatmap of K with entries + CIs
    axE = fig.add_subplot(gs[1, 1])
    im = axE.imshow(K, cmap="RdBu_r")
    axE.set_xticks([0, 1])
    axE.set_xticklabels(["ATP→ADP", "ADP→AMP"], rotation=30)
    axE.set_yticks([0, 1])
    axE.set_yticklabels(["ATP→ADP", "ADP→AMP"])
    for i in range(2):
        for j in range(2):
            axE.text(
                j,
                i,
                f"{K[i,j]:+.3f}\n[{K_ci[0,i,j]:+.3f},{K_ci[1,i,j]:+.3f}]",
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
    axE.set_title("K Matrix (s⁻¹) with ~68% CIs")
    fig.colorbar(im, ax=axE, fraction=0.046, pad=0.04)

    # Panel F: simulate one replicate vs data
    axF = fig.add_subplot(gs[1, 2])
    # choose the replicate with densest times
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
