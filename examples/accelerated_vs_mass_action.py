"""Compare linear and accelerated LLRQ dynamics against true mass action kinetics.

This example simulates a simple reversible reaction A ⇌ B starting far from
equilibrium. It generates three trajectories:

1. True mass action kinetics simulated with Tellurium (if available)
2. Linear LLRQ approximation (default relaxation law)
3. Accelerated LLRQ relaxation (modal exponential-speedup law)

Run inside the Tellurium conda environment to enable the mass action trace:

    conda run -n tellurium python examples/accelerated_vs_mass_action.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from llrq import LLRQDynamics, LLRQSolver, ReactionNetwork

try:
    from llrq.mass_action_simulator import MassActionSimulator

    HAS_TELLURIUM = True
except ImportError:  # Tellurium / roadrunner not installed
    HAS_TELLURIUM = False


def build_reversible_network() -> ReactionNetwork:
    """Create an A ⇌ B reaction network with metadata."""

    species_ids = ["A", "B"]
    reaction_ids = ["R1"]
    stoich = np.array([[-1], [1]])

    # Provide baseline concentrations (equilibrium point) for reference
    species_info = {
        "A": {"initial_concentration": 1.0},
        "B": {"initial_concentration": 2.0},
    }

    return ReactionNetwork(species_ids, reaction_ids, stoich, species_info=species_info)


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

    simulator = MassActionSimulator.from_llrq_dynamics(dynamics, network)
    simulator.set_concentrations(c0)
    return simulator.simulate(t_eval)


def plot_results(t: np.ndarray, keq: float, linear: dict, accelerated: dict, mass_action: dict | None) -> None:
    lnQ_linear = np.log(linear["reaction_quotients"][:, 0] / keq)
    lnQ_accel = np.log(accelerated["reaction_quotients"][:, 0] / keq)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(t, lnQ_linear, label="Linear LLRQ", linestyle="--")
    axes[0].plot(t, lnQ_accel, label="Accelerated LLRQ", linestyle="-")

    if mass_action is not None:
        lnQ_mass = np.log(mass_action["reaction_quotients"][:, 0] / keq)
        axes[0].plot(t, lnQ_mass, label="Mass Action", linestyle=":")

    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("ln(Q/Keq)")
    axes[0].legend()
    axes[0].set_title("Reaction force relaxation (log quotient deviation)")

    axes[1].plot(t, linear["reaction_quotients"][:, 0], label="Linear LLRQ", linestyle="--")
    axes[1].plot(t, accelerated["reaction_quotients"][:, 0], label="Accelerated LLRQ", linestyle="-")

    if mass_action is not None:
        axes[1].plot(t, mass_action["reaction_quotients"][:, 0], label="Mass Action", linestyle=":")

    axes[1].set_ylabel("Reaction Quotient Q")
    axes[1].set_xlabel("")
    axes[1].legend()

    # Concentrations (A, B)
    axes[2].plot(t, linear["concentrations"][:, 0], label="[A] Linear", linestyle="--", color="#1f77b4")
    axes[2].plot(t, linear["concentrations"][:, 1], label="[B] Linear", linestyle="--", color="#ff7f0e")
    axes[2].plot(t, accelerated["concentrations"][:, 0], label="[A] Accelerated", linestyle="-", color="#1f77b4")
    axes[2].plot(t, accelerated["concentrations"][:, 1], label="[B] Accelerated", linestyle="-", color="#ff7f0e")

    if mass_action is not None:
        axes[2].plot(t, mass_action["concentrations"][:, 0], label="[A] Mass Action", linestyle=":", color="#1f77b4")
        axes[2].plot(t, mass_action["concentrations"][:, 1], label="[B] Mass Action", linestyle=":", color="#ff7f0e")

    axes[2].set_ylabel("Concentration")
    axes[2].set_xlabel("Time")
    axes[2].legend(ncol=3, fontsize=9)

    fig.tight_layout()
    plt.show()


def main() -> None:
    network = build_reversible_network()

    # Mass action parameters and equilibrium point
    c_star = np.array([1.0, 2.0])
    k_plus = np.array([2.0])
    k_minus = np.array([1.0])
    keq = float(k_plus[0] / k_minus[0])

    # Initial condition far from equilibrium but respecting conservation (A + B = 3)
    c0 = np.array([2.9, 0.1])

    t_eval = np.linspace(0.0, 6.0, 300)

    linear_dyn = LLRQDynamics.from_mass_action(
        network=network,
        forward_rates=k_plus,
        backward_rates=k_minus,
        initial_concentrations=c_star,
        mode="equilibrium",
        relaxation_mode="linear",
    )

    # Choose xi_star based on initial deviation magnitude to fit tails where trajectory starts.
    # We'll also try a small grid and pick the best against mass-action if available.
    Q0 = network.compute_reaction_quotients(c0)[0]
    x0 = float(np.log(Q0 / keq))
    xi_star_guess = float(np.clip(abs(x0), 0.5, 4.0))

    # Compute mass-action trace first (for metric)
    mass_action_result = simulate_mass_action(linear_dyn, network, c0, t_eval)

    # Sweep xi_star over a small grid and choose best fit against mass action (if available)
    xi_grid = sorted(set([xi_star_guess, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]))
    variants = ["sinh", "exp"]
    best = None
    for var in variants:
        for xi in xi_grid:
            dyn_acc = LLRQDynamics.from_mass_action(
                network=network,
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="equilibrium",
                relaxation_mode="accelerated",
                relaxation_kwargs={"xi_star": xi, "delta": 1e-3, "variant": var},
            )
            res_acc = simulate_accelerated(dyn_acc, c0, t_eval)
            if mass_action_result is None:
                # Fall back: compare to linear as a proxy if mass-action missing
                res_lin = simulate_linear(linear_dyn, c0, t_eval)
                lnQ_lin = np.log(res_lin["reaction_quotients"][:, 0] / keq)
                lnQ_acc = np.log(res_acc["reaction_quotients"][:, 0] / keq)
                sse = float(np.mean((lnQ_acc - lnQ_lin) ** 2))
            else:
                lnQ_ma = np.log(mass_action_result["reaction_quotients"][:, 0] / keq)
                lnQ_acc = np.log(res_acc["reaction_quotients"][:, 0] / keq)
                sse = float(np.mean((lnQ_acc - lnQ_ma) ** 2))
            if best is None or sse < best[0]:
                best = (sse, xi, var, res_acc, dyn_acc)

    # Final accelerated pick
    _, xi_star, variant_chosen, accelerated_result, accelerated_dyn = best

    # Get linear result (single run)
    linear_result = simulate_linear(linear_dyn, c0, t_eval)

    if mass_action_result is not None:
        lnQ_ma = np.log(mass_action_result["reaction_quotients"][:, 0] / keq)
        lnQ_lin = np.log(linear_result["reaction_quotients"][:, 0] / keq)
        lnQ_acc = np.log(accelerated_result["reaction_quotients"][:, 0] / keq)
        sse_lin = float(np.mean((lnQ_lin - lnQ_ma) ** 2))
        sse_acc = float(np.mean((lnQ_acc - lnQ_ma) ** 2))
        print(
            f"Chosen variant={variant_chosen}, xi_star={xi_star:.3f} | "
            f"SSE lnQ: linear={sse_lin:.4g}, accelerated={sse_acc:.4g}"
        )

    plot_results(t_eval, keq, linear_result, accelerated_result, mass_action_result)


if __name__ == "__main__":
    main()
