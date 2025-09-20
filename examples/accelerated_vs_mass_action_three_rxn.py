"""Compare linear and accelerated LLRQ vs true mass action for 3 reactions.

Network: A ⇌ B ⇌ C ⇌ D (three reversible reactions).

The script:
- Builds the chain network and assigns mass-action rates
- Simulates true mass action with Tellurium (if available)
- Simulates linear LLRQ and accelerated LLRQ
- Auto-tunes accelerated model: tries exp/sinh with a small ξ* grid, and
  fits the scalar-gain 'gain' variant from the mass-action trace, then picks
  the best performer by SSE in ln(Q/Keq) across all three reactions.
- Plots ln(Q/Keq) for each reaction and concentrations [A],[B],[C],[D].

Run (with Tellurium):
    conda run -n tellurium python examples/accelerated_vs_mass_action_three_rxn.py
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


def build_chain3_network() -> ReactionNetwork:
    species_ids = ["A", "B", "C", "D"]
    reaction_ids = ["R1", "R2", "R3"]  # A⇌B, B⇌C, C⇌D
    S = np.array(
        [
            [-1, 0, 0],  # A
            [1, -1, 0],  # B
            [0, 1, -1],  # C
            [0, 0, 1],  # D
        ]
    )

    species_info = {
        "A": {"initial_concentration": 1.5},
        "B": {"initial_concentration": 1.0},
        "C": {"initial_concentration": 0.6},
        "D": {"initial_concentration": 0.4},
    }

    return ReactionNetwork(species_ids, reaction_ids, S, species_info=species_info)


def simulate_linear(dynamics: LLRQDynamics, c0: np.ndarray, t_eval: np.ndarray) -> dict:
    solver = LLRQSolver(dynamics)
    return solver.solve(c0, t_eval, method="numerical", enforce_conservation=True)


def simulate_accelerated(dynamics: LLRQDynamics, c0: np.ndarray, t_eval: np.ndarray) -> dict:
    solver = LLRQSolver(dynamics)
    return solver.solve(c0, t_eval, method="numerical", enforce_conservation=True)


def simulate_mass_action(dynamics: LLRQDynamics, network: ReactionNetwork, c0: np.ndarray, t_eval: np.ndarray) -> dict | None:
    if not HAS_TELLURIUM:
        print("Tellurium not available; skipping true mass action simulation.")
        return None
    sim = MassActionSimulator.from_llrq_dynamics(dynamics, network)
    sim.set_concentrations(c0)
    return sim.simulate(t_eval)


def lnQ_dev(result: dict, keq: np.ndarray) -> np.ndarray:
    Q = result["reaction_quotients"]
    return np.log(Q / keq[None, :])


def sse_lnQ(res: dict, ref: dict, keq: np.ndarray) -> float:
    a = lnQ_dev(res, keq)
    b = lnQ_dev(ref, keq)
    return float(np.mean((a - b) ** 2))


def plot_results(t: np.ndarray, keq: np.ndarray, linear: dict, accelerated: dict, mass_action: dict | None) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ln_lin = lnQ_dev(linear, keq)
    ln_acc = lnQ_dev(accelerated, keq)
    ln_ma = lnQ_dev(mass_action, keq) if mass_action is not None else None

    for j, ax in enumerate(axes[:3]):
        ax.plot(t, ln_lin[:, j], label="Linear LLRQ", linestyle="--")
        ax.plot(t, ln_acc[:, j], label="Accelerated LLRQ", linestyle="-")
        if ln_ma is not None:
            ax.plot(t, ln_ma[:, j], label="Mass Action", linestyle=":")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(f"ln(Q{j+1}/Keq{j+1})")
        ax.legend()
        if j == 0:
            ax.set_title("Three-Reaction Chain: ln(Q/Keq) and Concentrations")

    # Concentrations panel
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    labels_lin = ["[A] Linear", "[B] Linear", "[C] Linear", "[D] Linear"]
    labels_acc = ["[A] Accelerated", "[B] Accelerated", "[C] Accelerated", "[D] Accelerated"]
    labels_ma = ["[A] Mass Action", "[B] Mass Action", "[C] Mass Action", "[D] Mass Action"]

    for i in range(4):
        axes[3].plot(t, linear["concentrations"][:, i], linestyle="--", color=colors[i], label=labels_lin[i])
        axes[3].plot(t, accelerated["concentrations"][:, i], linestyle="-", color=colors[i], label=labels_acc[i])
        if mass_action is not None:
            axes[3].plot(t, mass_action["concentrations"][:, i], linestyle=":", color=colors[i], label=labels_ma[i])
    axes[3].set_ylabel("Concentration")
    axes[3].set_xlabel("Time")
    axes[3].legend(ncol=4, fontsize=9)

    fig.tight_layout()
    plt.show()


def main() -> None:
    network = build_chain3_network()

    # Mass action parameters and equilibrium point
    c_star = np.array([1.5, 1.0, 0.6, 0.4])
    k_plus = np.array([2.0, 1.0, 1.5])
    k_minus = np.array([1.0, 2.0, 1.0])
    keq = k_plus / k_minus

    # Far-from-eq initial condition (conserved total ~3.5)
    c0 = np.array([3.2, 0.22, 0.06, 0.02])

    t_eval = np.linspace(0.0, 10.0, 500)

    # Linear dynamics
    lin_dyn = LLRQDynamics.from_mass_action(
        network=network,
        forward_rates=k_plus,
        backward_rates=k_minus,
        initial_concentrations=c_star,
        mode="equilibrium",
        relaxation_mode="linear",
    )

    # Mass action trace
    res_ma = simulate_mass_action(lin_dyn, network, c0, t_eval)

    # Grid over exp/sinh; combine with fitted gain variant
    variants = ["sinh", "exp"]
    xi_grid = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Seed with mean |ln(Q/Keq)| at t=0
    Q0 = network.compute_reaction_quotients(c0)
    x0 = np.abs(np.log(Q0 / keq))
    xi_grid.append(float(np.clip(np.mean(x0), 0.5, 6.0)))
    xi_grid = sorted(set([float(v) for v in xi_grid]))

    best = None
    for var in variants:
        for xi in xi_grid:
            dyn = LLRQDynamics.from_mass_action(
                network=network,
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="equilibrium",
                relaxation_mode="accelerated",
                relaxation_kwargs={"variant": var, "xi_star": xi, "delta": 1e-3},
            )
            res = simulate_accelerated(dyn, c0, t_eval)
            if res_ma is not None:
                sse = sse_lnQ(res, res_ma, keq)
            else:
                res_lin = simulate_linear(lin_dyn, c0, t_eval)
                sse = float(np.mean((lnQ_dev(res, keq) - lnQ_dev(res_lin, keq)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, var, xi, res, dyn)

    # Fit gain from mass-action early segment and test it
    if HAS_TELLURIUM and res_ma is not None:
        beta = fit_gain_from_mass_action(lin_dyn, res_ma, t_eval, early_fraction=0.25, gamma=1.0)
        dyn_gain = LLRQDynamics.from_mass_action(
            network=network,
            forward_rates=k_plus,
            backward_rates=k_minus,
            initial_concentrations=c_star,
            mode="equilibrium",
            relaxation_mode="accelerated",
            relaxation_kwargs={"variant": "gain", "gain_beta": beta, "gain_gamma": 1.0},
        )
        res_gain = simulate_accelerated(dyn_gain, c0, t_eval)
        sse_gain = sse_lnQ(res_gain, res_ma, keq)
        if best is None or sse_gain < best[0]:
            best = (sse_gain, "gain", 0.0, res_gain, dyn_gain)

    # Final pick
    sse_best, var_best, xi_best, res_acc, dyn_acc = best

    # Linear result (single run)
    res_lin = simulate_linear(lin_dyn, c0, t_eval)

    if res_ma is not None:
        sse_lin = sse_lnQ(res_lin, res_ma, keq)
        print(
            f"Chosen variant={var_best}, xi_star={xi_best:.3f} | "
            f"SSE lnQ (sum over 3 rxns): linear={sse_lin:.4g}, accelerated={sse_best:.4g}"
        )

    plot_results(t_eval, keq, res_lin, res_acc, res_ma)


if __name__ == "__main__":
    main()
