"""Fit an ICNN vector field to mass-action ln(Q/Keq) dynamics (3 reactions).

Overview
--------
This example demonstrates learning a thermodynamically-safe vector field
in reaction-quotient log space x = ln(Q/Keq) by training an ICNN-based
model on samples (x, xdot) generated from mass action kinetics.

Network: A ⇌ B ⇌ C ⇌ D (three reversible reactions)

What it does:
- Simulates true mass action for multiple initial conditions (pure NumPy RK4)
- Computes x(t) = ln(Q/Keq) and exact xdot(t) = S^T (c_dot / c)
- Trains the ICNN vector field f(x) = -M(x) ∇Φ(x) on (x, xdot)
- Rolls out the learned dynamics from a held-out initial x0 and compares
  ln(Q/Keq) trajectories to mass action

Requirements: numpy, matplotlib, torch

Run:
    python examples/icnn_fit_vs_mass_action_three_rxn.py

Notes:
- This example uses a small, self-contained mass-action integrator (RK4), so
  it does not require tellurium.
- To speed up or stabilize training, you can reduce epochs or widths below.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Reaction network (A ⇌ B ⇌ C ⇌ D)
# -------------------------------
def build_chain3_network():
    species_ids = ["A", "B", "C", "D"]
    reaction_ids = ["R1", "R2", "R3"]  # A⇌B, B⇌C, C⇌D
    # Stoichiometric matrix S (species x reactions)
    S = np.array(
        [
            [-1.0, 0.0, 0.0],  # A
            [1.0, -1.0, 0.0],  # B
            [0.0, 1.0, -1.0],  # C
            [0.0, 0.0, 1.0],  # D
        ]
    )
    return species_ids, reaction_ids, S


def reactant_product_matrices(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return A (reactant stoich) and B (product stoich), both >= 0."""
    A = np.maximum(-S, 0.0)
    B = np.maximum(S, 0.0)
    return A, B


# -------------------------------
# Mass action ODE and integrator
# -------------------------------
def mass_action_rhs(
    c: np.ndarray, S: np.ndarray, A: np.ndarray, B: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray
) -> np.ndarray:
    """Compute c_dot = S v(c) with mass action v_j = k+ prod(c_i^A_ij) - k- prod(c_i^B_ij)."""
    # Numerically stable via logs
    log_c = np.log(np.clip(c, 1e-12, None))  # (n_species,)
    r_fwd = np.exp(A.T @ log_c)  # (n_reactions,)
    r_rev = np.exp(B.T @ log_c)  # (n_reactions,)
    v = k_plus * r_fwd - k_minus * r_rev
    return S @ v


def rk4_simulate(
    c0: np.ndarray, t: np.ndarray, S: np.ndarray, A: np.ndarray, B: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray
) -> np.ndarray:
    """Fixed-step RK4 for mass action; returns concentrations over time (T, n_species)."""
    c = np.array(c0, dtype=float)
    T = len(t)
    out = np.zeros((T, len(c0)))
    out[0] = c
    for i in range(1, T):
        dt = float(t[i] - t[i - 1])
        k1 = mass_action_rhs(c, S, A, B, k_plus, k_minus)
        k2 = mass_action_rhs(c + 0.5 * dt * k1, S, A, B, k_plus, k_minus)
        k3 = mass_action_rhs(c + 0.5 * dt * k2, S, A, B, k_plus, k_minus)
        k4 = mass_action_rhs(c + dt * k3, S, A, B, k_plus, k_minus)
        c = c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        c = np.maximum(c, 1e-12)  # positivity safeguard
        out[i] = c
    return out


# -------------------------------
# Dataset: x = ln(Q/Keq), xdot = S^T (c_dot / c)
# -------------------------------
def compute_lnQ_and_xdot(
    traj_c: np.ndarray, t: np.ndarray, S: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute x(t)=ln(Q/Keq) and xdot(t) exactly from c, cdot.

    x = S^T ln c - ln(Keq), with Keq = k+ / k-
    xdot = S^T (c_dot / c)
    """
    A = np.maximum(-S, 0.0)
    B = np.maximum(S, 0.0)
    keq = k_plus / k_minus
    lnK = np.log(keq)
    T, n_species = traj_c.shape
    n_reactions = S.shape[1]

    X = np.zeros((T, n_reactions))
    V = np.zeros((T, n_reactions))

    for i in range(T):
        c = traj_c[i]
        log_c = np.log(np.clip(c, 1e-12, None))
        Q = np.exp(S.T @ log_c)
        x = np.log(np.clip(Q, 1e-300, None)) - lnK
        X[i] = x

        # exact c_dot at this c
        c_dot = mass_action_rhs(c, S, A, B, k_plus, k_minus)
        V[i] = S.T @ (c_dot / np.clip(c, 1e-12, None))

    return X, V


def collect_dataset(
    S: np.ndarray,
    k_plus: np.ndarray,
    k_minus: np.ndarray,
    c0_list: List[np.ndarray],
    t_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate multiple mass-action trajectories and stack (x, xdot)."""
    A, B = reactant_product_matrices(S)
    X_all = []
    V_all = []
    T_all = []
    for c0 in c0_list:
        traj_c = rk4_simulate(c0, t_eval, S, A, B, k_plus, k_minus)
        X, V = compute_lnQ_and_xdot(traj_c, t_eval, S, k_plus, k_minus)
        X_all.append(X)
        V_all.append(V)
        T_all.append(t_eval.copy())
    X_all = np.concatenate(X_all, axis=0)
    V_all = np.concatenate(V_all, axis=0)
    return X_all, V_all, t_eval


# -------------------------------
# Import ICNN implementation from scripts/icnn.py
# -------------------------------
def import_icnn_module():
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    icnn_path = root / "scripts" / "icnn.py"
    spec = importlib.util.spec_from_file_location("llrq_icnn_module", icnn_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# -------------------------------
# Training and evaluation
# -------------------------------
@dataclass
class TrainCfg:
    epochs: int = 6000
    batch: int = 1024
    lr: float = 1e-3
    wd: float = 1e-6
    lambda_pass: float = 0.05
    eval_every: int = 200
    widths: Tuple[int, int] = (64, 64)
    learn_mobility: bool = False
    mob_rank: int = 0


def time_block_split(
    X: np.ndarray, V: np.ndarray, val_frac: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple random split; for true time-blocked, adapt per-trajectory indices."""
    N = X.shape[0]
    idx = np.random.permutation(N)
    n_val = int(math.ceil(val_frac * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], V[tr_idx], X[val_idx], V[val_idx]


def main():
    # Network and rates
    species_ids, reaction_ids, S = build_chain3_network()
    k_plus = np.array([2.0, 1.0, 1.5], dtype=float)
    k_minus = np.array([1.0, 2.0, 1.0], dtype=float)
    keq = k_plus / k_minus

    # Multiple far-from-eq initial conditions with conserved total ~3.5
    c0_list = [
        np.array([3.2, 0.22, 0.06, 0.02], dtype=float),
        np.array([0.02, 0.06, 0.22, 3.2], dtype=float),
        np.array([2.0, 0.9, 0.4, 0.2], dtype=float),
    ]

    t_eval = np.linspace(0.0, 10.0, 500)

    # Build dataset
    X, V, _ = collect_dataset(S, k_plus, k_minus, c0_list, t_eval)

    # Import ICNN module
    icnn = import_icnn_module()

    # Train/val split and standardize
    Xtr, Vtr, Xval, Vval = time_block_split(X, V, val_frac=0.2)
    Xtr_t = icnn.torch.tensor(Xtr, device=icnn.DEVICE).float()
    Vtr_t = icnn.torch.tensor(Vtr, device=icnn.DEVICE).float()
    Xval_t = icnn.torch.tensor(Xval, device=icnn.DEVICE).float()
    Vval_t = icnn.torch.tensor(Vval, device=icnn.DEVICE).float()

    Xtr_s, Xval_s, (mu_x, sd_x) = icnn.standardize(Xtr_t, Xval_t)
    Vtr_s, Vval_s, _ = icnn.standardize(Vtr_t, Vval_t)

    # Model
    dim = X.shape[1]
    cfg = TrainCfg()
    phi = icnn.ICNN(dim=dim, widths=cfg.widths).to(icnn.DEVICE)
    M = icnn.Mobility(dim=dim, rank=(cfg.mob_rank if cfg.mob_rank > 0 else dim), learn_mobility=cfg.learn_mobility).to(
        icnn.DEVICE
    )
    model = icnn.Model(phi=phi, M=M)

    # Pack config matching icnn.train expectations
    class C:
        pass

    C.epochs = cfg.epochs
    C.batch = cfg.batch
    C.lr = cfg.lr
    C.wd = cfg.wd
    C.lambda_pass = cfg.lambda_pass
    C.eval_every = cfg.eval_every

    print("Training ICNN on (x, xdot) from mass action...")
    model = icnn.train(model, Xtr_s, Vtr_s, Xval_s, Vval_s, C)

    # Choose a held-out initial condition and compare rollouts
    c0_test = np.array([3.2, 0.22, 0.06, 0.02], dtype=float)
    A, B = reactant_product_matrices(S)
    traj_c = rk4_simulate(c0_test, t_eval, S, A, B, k_plus, k_minus)
    X_true, _ = compute_lnQ_and_xdot(traj_c, t_eval, S, k_plus, k_minus)

    # Rollout the learned field in standardized x-space
    dt = float(t_eval[1] - t_eval[0])
    x0 = X_true[0:1]
    x0_s = (icnn.torch.tensor(x0, device=icnn.DEVICE).float() - mu_x) / icnn.torch.clamp_min(sd_x, 1e-6)
    traj_s = icnn.rollout(model, x0_s, dt=dt, steps=len(t_eval) - 1).cpu().numpy()
    traj_pred = traj_s * sd_x.cpu().numpy() + mu_x.cpu().numpy()  # unstandardize

    # Metrics
    rmse = float(np.sqrt(np.mean((traj_pred - X_true) ** 2)))
    print(f"\nTrajectory RMSE in x=ln(Q/Keq) against mass action: {rmse:.4f}")

    # Plot ln(Q/Keq) for each reaction
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t_eval, X_true[:, j], label="Mass Action (true)", linestyle=":")
        ax.plot(t_eval, traj_pred[:, j], label="ICNN learned", linestyle="-")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(f"ln(Q{j+1}/Keq{j+1})")
        if j == 0:
            ax.set_title("ICNN fit to mass action: ln(Q/Keq) trajectories (3 rxns)")
    axes[-1].set_xlabel("Time")
    axes[0].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
