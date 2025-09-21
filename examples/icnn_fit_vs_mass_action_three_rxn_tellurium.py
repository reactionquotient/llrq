"""ICNN fit to mass action ln(Q/Keq) using Tellurium backend (3 reactions).

This example uses tellurium/roadrunner directly (no llrq import) to generate
(x, xdot) training data in ln(Q/Keq) space, trains the ICNN vector field from
scripts/icnn.py in-process, and compares rollouts to mass action.

Why no llrq import? Importing llrq pulls SciPy via solver, which can clash
with some tellurium envs. This script composes the Antimony model directly
to avoid that dependency for data generation.

Run inside the Tellurium conda environment:
    conda run -n tellurium python examples/icnn_fit_vs_mass_action_three_rxn_tellurium.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Prefer antimony + roadrunner to avoid heavy SciPy imports through tellurium
import antimony
import roadrunner


# -------------------------------
# Helpers: network and mass-action ops
# -------------------------------
def chain3_stoichiometry() -> Tuple[List[str], List[str], np.ndarray]:
    species_ids = ["A", "B", "C", "D"]
    reaction_ids = ["R1", "R2", "R3"]
    S = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return species_ids, reaction_ids, S


def reactant_product_matrices(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.maximum(-S, 0.0)
    B = np.maximum(S, 0.0)
    return A, B


def mass_action_rhs(
    c: np.ndarray, S: np.ndarray, A: np.ndarray, B: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray
) -> np.ndarray:
    log_c = np.log(np.clip(c, 1e-12, None))
    r_fwd = np.exp(A.T @ log_c)
    r_rev = np.exp(B.T @ log_c)
    v = k_plus * r_fwd - k_minus * r_rev
    return S @ v


# -------------------------------
# Dataset building from Tellurium traces
# -------------------------------
def build_antimony_model(k_plus: np.ndarray, k_minus: np.ndarray, c_init: np.ndarray) -> str:
    # Species and parameters
    lines = ["model chain3"]
    lines.append("")
    lines.append("// Species with initial concentrations")
    species = ["A", "B", "C", "D"]
    for sid, val in zip(species, c_init):
        lines.append(f"  {sid} = {float(val)};")
    lines.append("")

    lines.append("// Rate constants")
    for i in range(3):
        lines.append(f"  kf{i+1} = {float(k_plus[i])};")
        lines.append(f"  kr{i+1} = {float(k_minus[i])};")
    lines.append("")

    lines.append("// Reversible reactions as pairs of irreversible mass-action steps")
    lines.append("  R1f: A -> B; kf1*A;")
    lines.append("  R1r: B -> A; kr1*B;")
    lines.append("  R2f: B -> C; kf2*B;")
    lines.append("  R2r: C -> B; kr2*C;")
    lines.append("  R3f: C -> D; kf3*C;")
    lines.append("  R3r: D -> C; kr3*D;")
    lines.append("")
    lines.append("end")
    return "\n".join(lines)


def simulate_mass_action_rr(
    k_plus: np.ndarray, k_minus: np.ndarray, c0: np.ndarray, t_eval: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ant = build_antimony_model(k_plus, k_minus, c0)
    # Convert Antimony to SBML via antimony module, then load into RoadRunner
    antimony.clearPreviousLoads()
    rc = antimony.loadAntimonyString(ant)
    if rc < 0:
        raise RuntimeError("Failed to load Antimony model: " + antimony.getLastError())
    sbml = antimony.getSBMLString(None)
    if not sbml:
        raise RuntimeError("Failed to get SBML string from Antimony")

    rr = roadrunner.RoadRunner(sbml)
    # RoadRunner simulate: start, end, points -> uniform grid
    T = len(t_eval)
    result = rr.simulate(float(t_eval[0]), float(t_eval[-1]), T)
    # Columns: [time, A, B, C, D] (default floating species)
    conc = np.array(result)[:, 1:5]
    return conc, t_eval


def build_dataset_from_mass_action(
    k_plus: np.ndarray, k_minus: np.ndarray, c0_list: List[np.ndarray], t_eval: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    species_ids, reaction_ids, S = chain3_stoichiometry()
    A, B = reactant_product_matrices(S)
    lnK = np.log(k_plus / k_minus)

    X_list, V_list = [], []
    for c0 in c0_list:
        conc, _ = simulate_mass_action_rr(k_plus, k_minus, c0, t_eval)
        # x = S^T ln c - lnK
        log_c = np.log(np.clip(conc, 1e-12, None))
        X = (S.T @ log_c.T).T - lnK[None, :]

        # exact xdot = S^T (c_dot / c)
        V = np.zeros_like(X)
        for i in range(len(t_eval)):
            cdot = mass_action_rhs(conc[i], S, A, B, k_plus, k_minus)
            V[i] = S.T @ (cdot / np.clip(conc[i], 1e-12, None))

        X_list.append(X)
        V_list.append(V)

    X = np.concatenate(X_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    return X, V, S


# -------------------------------
# Import ICNN code from scripts/icnn.py
# -------------------------------
def import_icnn_module():
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    icnn_path = root / "scripts" / "icnn.py"
    spec = importlib.util.spec_from_file_location("llrq_icnn_module", icnn_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        name = getattr(e, "name", "")
        if name == "torch":
            raise RuntimeError(
                "PyTorch is required for ICNN training. Install in the tellurium env, e.g.,\n"
                "  conda install -c pytorch pytorch cpuonly\n"
                "or\n"
                "  pip install torch\n"
            )
        raise
    return mod


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


def main():
    # Stoichiometry and rates
    species_ids, reaction_ids, S = chain3_stoichiometry()
    k_plus = np.array([2.0, 1.0, 1.5], dtype=float)
    k_minus = np.array([1.0, 2.0, 1.0], dtype=float)

    # Initial conditions (diverse, conserved total around 3.5)
    c0_list = [
        np.array([3.2, 0.22, 0.06, 0.02], dtype=float),
        np.array([0.02, 0.06, 0.22, 3.2], dtype=float),
        np.array([2.0, 0.9, 0.4, 0.2], dtype=float),
    ]
    t_eval = np.linspace(0.0, 10.0, 500)

    # Dataset from tellurium-based mass action
    X, V, S = build_dataset_from_mass_action(k_plus, k_minus, c0_list, t_eval)

    # Import ICNN module and prepare tensors
    icnn = import_icnn_module()
    X_t = icnn.torch.tensor(X, device=icnn.DEVICE).float()
    V_t = icnn.torch.tensor(V, device=icnn.DEVICE).float()

    # Random split (for proper time-blocking, split per-trajectory indices)
    N = X_t.shape[0]
    idx = icnn.torch.randperm(N, device=icnn.DEVICE)
    n_val = int(math.ceil(0.2 * N))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Xval = X_t[tr_idx], X_t[val_idx]
    Vtr, Vval = V_t[tr_idx], V_t[val_idx]

    # Standardize
    Xtr_s, Xval_s, (mu_x, sd_x) = icnn.standardize(Xtr, Xval)
    Vtr_s, Vval_s, _ = icnn.standardize(Vtr, Vval)

    # Model
    dim = X.shape[1]
    cfg = TrainCfg()
    phi = icnn.ICNN(dim=dim, widths=cfg.widths).to(icnn.DEVICE)
    M = icnn.Mobility(dim=dim, rank=(cfg.mob_rank if cfg.mob_rank > 0 else dim), learn_mobility=cfg.learn_mobility).to(
        icnn.DEVICE
    )
    model = icnn.Model(phi=phi, M=M)

    class C:
        pass

    C.epochs = cfg.epochs
    C.batch = cfg.batch
    C.lr = cfg.lr
    C.wd = cfg.wd
    C.lambda_pass = cfg.lambda_pass
    C.eval_every = cfg.eval_every

    print("Training ICNN on tellurium mass-action (x, xdot) samples...")
    model = icnn.train(model, Xtr_s, Vtr_s, Xval_s, Vval_s, C)

    # Hold-out rollout test on one IC (mass action reference via Tellurium)
    c0_test = np.array([3.2, 0.22, 0.06, 0.02], dtype=float)
    conc_ref, _ = simulate_mass_action_rr(k_plus, k_minus, c0_test, t_eval)
    X_true = (S.T @ np.log(np.clip(conc_ref, 1e-12, None)).T).T - np.log(k_plus / k_minus)[None, :]

    # ICNN rollout in standardized coordinates
    dt = float(t_eval[1] - t_eval[0])
    x0 = X_true[0:1]
    x0_s = (icnn.torch.tensor(x0, device=icnn.DEVICE).float() - mu_x) / icnn.torch.clamp_min(sd_x, 1e-6)
    traj_s = icnn.rollout(model, x0_s, dt=dt, steps=len(t_eval) - 1).cpu().numpy()
    traj_pred = traj_s * sd_x.cpu().numpy() + mu_x.cpu().numpy()

    rmse = float(np.sqrt(np.mean((traj_pred - X_true) ** 2)))
    print(f"\nTrajectory RMSE in x=ln(Q/Keq) (ICNN vs mass action): {rmse:.4f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t_eval, X_true[:, j], label="Mass Action (Tellurium)", linestyle=":")
        ax.plot(t_eval, traj_pred[:, j], label="ICNN learned", linestyle="-")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(f"ln(Q{j+1}/Keq{j+1})")
        if j == 0:
            ax.set_title("ICNN fit vs Mass Action (Tellurium): ln(Q/Keq), 3 rxns")
    axes[-1].set_xlabel("Time")
    axes[0].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
