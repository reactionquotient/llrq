"""Twelve-reaction network: compare linear and accelerated LLRQ vs mass action.

Network has heterogeneous stoichiometries (bimolecular, trimolecular, branching).

We simulate:
- True mass action (Tellurium) if available
- Linear LLRQ (baseline)
- Accelerated LLRQ (exp/sinh grid + scalar gain fitted from mass action)

We print SSE on ln(Q/Keq) across all 12 reactions and make concise plots:
1) ||ln(Q/Keq)||_2 vs time (three curves)
2) Concentrations of four species (A,B,C,D) vs time (mass action dotted; linear dashed; accelerated solid)

Run (Tellurium env):
    conda run -n tellurium python examples/accelerated_vs_mass_action_twelve_rxn.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from llrq import LLRQDynamics, LLRQSolver, ReactionNetwork, fit_gain_from_mass_action

try:
    from llrq.mass_action_simulator import MassActionSimulator

    HAS_TELLURIUM = True
except ImportError:
    HAS_TELLURIUM = False


def build_network() -> ReactionNetwork:
    # Species: A,B,C,D,E,F,G,H,I,J,K,L (we'll use A..H for stoichiometry below to keep it tractable)
    species_ids = ["A", "B", "C", "D", "E", "F", "G", "H"]
    reaction_ids = [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
        "R10",
        "R11",
        "R12",
    ]

    # Stoichiometry (species x reactions): positive = products, negative = reactants
    # R1: 2A + B -> C
    # R2: C -> A + D
    # R3: B + D -> E
    # R4: 2E -> F
    # R5: C + E -> G
    # R6: G -> A + F
    # R7: D + F -> H
    # R8: H -> E + G
    # R9: 2B -> A
    # R10: C + H -> 2D
    # R11: E + F -> A + G
    # R12: 3A -> B + C
    S = np.array(
        [
            # A  B  C  D  E  F  G  H
            [-2, +1, 0, 0, 0, +1, 0, 0, +1, 0, +1, -3],  # A
            [-1, 0, -1, 0, 0, 0, 0, 0, -2, 0, 0, +1],  # B
            [+1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, +1],  # C
            [0, +1, -1, 0, 0, 0, 0, 0, 0, +2, 0, 0],  # D
            [0, 0, +1, -2, -1, 0, 0, +1, 0, 0, -1, 0],  # E
            [0, 0, 0, +1, 0, +1, -1, 0, 0, 0, -1, 0],  # F
            [0, 0, 0, 0, +1, -1, +1, +1, 0, 0, +1, 0],  # G
            [0, 0, 0, 0, 0, 0, +1, -1, 0, -1, 0, 0],  # H
        ],
        dtype=float,
    )

    species_info = {
        "A": {"initial_concentration": 2.5},
        "B": {"initial_concentration": 1.2},
        "C": {"initial_concentration": 0.8},
        "D": {"initial_concentration": 0.6},
        "E": {"initial_concentration": 0.4},
        "F": {"initial_concentration": 0.3},
        "G": {"initial_concentration": 0.2},
        "H": {"initial_concentration": 0.1},
    }

    return ReactionNetwork(species_ids, [f"R{i+1}" for i in range(12)], S, species_info=species_info)


def simulate_linear(dynamics: LLRQDynamics, c0: np.ndarray, t_eval: np.ndarray) -> dict:
    solver = LLRQSolver(dynamics)
    return solver.solve(c0, t_eval, method="numerical", enforce_conservation=True)


def simulate_accelerated(dynamics: LLRQDynamics, c0: np.ndarray, t_eval: np.ndarray) -> dict:
    solver = LLRQSolver(dynamics)
    return solver.solve(c0, t_eval, method="numerical", enforce_conservation=True)


def simulate_mass_action(
    network: ReactionNetwork, k_plus: np.ndarray, k_minus: np.ndarray, c0: np.ndarray, t_eval: np.ndarray
) -> dict | None:
    if not HAS_TELLURIUM:
        print("Tellurium not available; skipping true mass action simulation.")
        return None
    # Build rate constants mapping directly (no need for LLRQ metadata)
    rate_constants = {rid: (float(k_plus[i]), float(k_minus[i])) for i, rid in enumerate(network.reaction_ids)}
    try:
        sim = MassActionSimulator(network, rate_constants)
        sim.set_concentrations(c0)
        return sim.simulate(t_eval)
    except Exception as e:
        print(f"Mass action simulation unavailable ({e}); continuing without it.")
        return None


def lnQ_dev(result: dict, keq: np.ndarray) -> np.ndarray:
    Q = result["reaction_quotients"]
    return np.log(np.maximum(Q, 1e-300) / keq[None, :])


def sse_lnQ(res: dict, ref: dict, keq: np.ndarray) -> float:
    a = lnQ_dev(res, keq)
    b = lnQ_dev(ref, keq)
    n = min(a.shape[0], b.shape[0])
    return float(np.mean((a[:n] - b[:n]) ** 2))


def build_llrq_linear(network: ReactionNetwork, c_star: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray) -> LLRQDynamics:
    # Build linear relaxation without solving for equilibrium globally
    lr = network.compute_linear_relaxation_matrix(c_star, k_plus, k_minus)
    K_lin = lr["K"]
    Keq = np.maximum(k_plus / k_minus, 1e-12)
    dyn = LLRQDynamics(network, equilibrium_constants=Keq, relaxation_matrix=K_lin)
    # Provide minimal mass-action metadata to enable accelerated fitting
    try:
        K_red, B = network._reduce_to_image_space(K_lin)
    except Exception:
        # Fallback: identity basis
        B = np.eye(len(network.reaction_ids))
        K_red = B.T @ K_lin @ B
    dyn._mass_action_data = {"K": K_lin, "K_reduced": K_red, "basis": B}
    dyn._mass_action_mode = "local"
    dyn._forward_rates = np.array(k_plus, dtype=float)
    dyn._backward_rates = np.array(k_minus, dtype=float)
    dyn._equilibrium_point = np.array(c_star, dtype=float)
    return dyn


def _reconstruct_concentrations_if_needed(result: dict, network: ReactionNetwork) -> np.ndarray:
    C = result.get("concentrations", None)
    if C is not None:
        return C
    Q = result["reaction_quotients"]
    # Solve S^T u = ln Q for u = ln c (min-norm)
    S = network.S
    S_T = S.T.toarray() if hasattr(S, "toarray") else np.array(S.T, dtype=float)
    lnQ = np.log(np.maximum(Q, 1e-300))
    C_rec = np.zeros((Q.shape[0], S_T.shape[1]))
    for i in range(Q.shape[0]):
        u, *_ = np.linalg.lstsq(S_T, lnQ[i], rcond=None)
        C_rec[i] = np.exp(np.clip(u, -50, 50))
    return C_rec


def concise_plots(t: np.ndarray, keq: np.ndarray, lin: dict, acc: dict, ma: dict | None, network: ReactionNetwork) -> None:
    # Panel 1: norm of ln(Q/Keq)
    ln_lin = lnQ_dev(lin, keq)
    ln_acc = lnQ_dev(acc, keq)
    ln_ma = lnQ_dev(ma, keq) if ma is not None else None

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, np.linalg.norm(ln_lin, axis=1), label="Linear", linestyle="--")
    axes[0].plot(t, np.linalg.norm(ln_acc, axis=1), label="Accelerated", linestyle="-")
    if ln_ma is not None:
        axes[0].plot(t, np.linalg.norm(ln_ma, axis=1), label="Mass Action", linestyle=":")
    axes[0].set_ylabel("||ln(Q/Keq)||₂")
    axes[0].legend()
    axes[0].set_title("12-Reaction Network: lnQ norm and select concentrations")

    # Panel 2: concentrations of selected species (A,B,C,D)
    linC = _reconstruct_concentrations_if_needed(lin, network)
    accC = _reconstruct_concentrations_if_needed(acc, network)
    maC = ma.get("concentrations") if ma is not None else None
    sel = [0, 1, 2, 3]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels = ["[A]", "[B]", "[C]", "[D]"]
    for idx, col, lab in zip(sel, colors, labels):
        axes[1].plot(t, linC[:, idx], linestyle="--", color=col, label=f"{lab} Linear")
        axes[1].plot(t, accC[:, idx], linestyle="-", color=col, label=f"{lab} Accelerated")
        if maC is not None:
            axes[1].plot(t, maC[:, idx], linestyle=":", color=col, label=f"{lab} Mass")
    axes[1].set_ylabel("Concentration")
    axes[1].set_xlabel("Time")
    axes[1].legend(ncol=3, fontsize=8)

    fig.tight_layout()
    plt.show()


def main() -> None:
    network = build_network()

    # Equilibrium point and mass action parameters
    c_star = np.array([2.5, 1.2, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])
    # Forward rates (choose positive values)
    k_plus = np.array(
        [
            1.5,  # R1
            0.9,  # R2
            1.2,  # R3
            0.7,  # R4
            1.1,  # R5
            0.8,  # R6
            1.0,  # R7
            0.6,  # R8
            1.3,  # R9
            0.9,  # R10
            1.0,  # R11
            0.5,  # R12
        ]
    )
    # Choose k_minus to be Wegscheider-consistent with c_star: Keq_j = Q_j(c*)
    Q_star = network.compute_reaction_quotients(c_star)
    keq = np.maximum(Q_star, 1e-12)
    k_minus = k_plus / keq

    # Far-from-equilibrium initial condition (respect positivity)
    c0 = np.array([5.0, 0.25, 0.1, 0.05, 0.03, 0.02, 0.02, 0.01])

    t_eval = np.linspace(0.0, 12.0, 600)

    # Linear dynamics (build directly to avoid global equilibrium checks)
    lin_dyn = build_llrq_linear(network, c_star, k_plus, k_minus)

    # Mass action trace (if available)
    res_ma = simulate_mass_action(network, k_plus, k_minus, c0, t_eval)

    # Grid search for exp/sinh
    variants = ["sinh", "exp"]
    xi_grid = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Seed at t=0 based on |ln(Q/Keq)| mean
    Q0 = network.compute_reaction_quotients(c0)
    xi_grid.append(float(np.clip(np.mean(np.abs(np.log(np.maximum(Q0, 1e-300) / keq))), 0.5, 6.0)))
    xi_grid = sorted(set([float(v) for v in xi_grid]))

    best = None
    for var in variants:
        for xi in xi_grid:
            dyn = build_llrq_linear(network, c_star, k_plus, k_minus)
            dyn.enable_accelerated_relaxation(variant=var, xi_star=xi, delta=1e-3)
            res = simulate_accelerated(dyn, c0, t_eval)
            if res_ma is not None:
                sse = sse_lnQ(res, res_ma, keq)
            else:
                res_lin = simulate_linear(lin_dyn, c0, t_eval)
                sse = float(np.mean((lnQ_dev(res, keq) - lnQ_dev(res_lin, keq)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, var, xi, res, dyn)

    # Fit scalar gain from mass action early segment
    if HAS_TELLURIUM and res_ma is not None:
        beta = fit_gain_from_mass_action(lin_dyn, res_ma, t_eval, early_fraction=0.25, gamma=1.0)
        dyn_gain = build_llrq_linear(network, c_star, k_plus, k_minus)
        dyn_gain.enable_accelerated_relaxation(variant="gain", gain_beta=beta, gain_gamma=1.0)
        res_gain = simulate_accelerated(dyn_gain, c0, t_eval)
        sse_gain = sse_lnQ(res_gain, res_ma, keq)
        if best is None or sse_gain < best[0]:
            best = (sse_gain, "gain", 0.0, res_gain, dyn_gain)

    sse_best, var_best, xi_best, res_acc, dyn_acc = best
    res_lin = simulate_linear(lin_dyn, c0, t_eval)

    if res_ma is not None:
        sse_lin = sse_lnQ(res_lin, res_ma, keq)
        print(
            f"Chosen variant={var_best}, xi_star={xi_best:.3f} | "
            f"SSE lnQ (sum over 12 rxns): linear={sse_lin:.4g}, accelerated={sse_best:.4g}"
        )

    concise_plots(t_eval, keq, res_lin, res_acc, res_ma, network)


if __name__ == "__main__":
    main()
