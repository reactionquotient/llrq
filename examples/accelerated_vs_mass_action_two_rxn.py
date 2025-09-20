"""Compare linear and accelerated (sinh) LLRQ dynamics vs true mass action for two reactions.

Network: A ⇌ B ⇌ C (two reversible reactions)

This script:
- Builds the chain network and mass-action parameters
- Simulates true mass action with Tellurium (if available)
- Simulates linear LLRQ and accelerated LLRQ (sinh/exp variants)
- Auto-tunes variant and xi_star against mass-action ln(Q/Keq) to minimize SSE
- Plots ln(Q/Keq) for both reactions and raw concentrations [A],[B],[C]

Run inside the Tellurium conda environment for mass-action traces:

    conda run -n tellurium python examples/accelerated_vs_mass_action_two_rxn.py
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


def build_chain_network() -> ReactionNetwork:
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2"]  # A ⇌ B ⇌ C
    # Stoichiometric matrix (species x reactions)
    S = np.array(
        [
            [-1, 0],  # A consumed in R1
            [1, -1],  # B produced in R1, consumed in R2
            [0, 1],  # C produced in R2
        ]
    )

    species_info = {
        "A": {"initial_concentration": 1.0},
        "B": {"initial_concentration": 1.5},
        "C": {"initial_concentration": 0.5},
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

    simulator = MassActionSimulator.from_llrq_dynamics(dynamics, network)
    simulator.set_concentrations(c0)
    return simulator.simulate(t_eval)


def lnQ_dev(result: dict, keq: np.ndarray) -> np.ndarray:
    Q = result["reaction_quotients"]  # shape (T, r)
    return np.log(Q / keq[None, :])


def plot_results(t: np.ndarray, keq: np.ndarray, linear: dict, accelerated: dict, mass_action: dict | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # ln(Q/Keq) for R1 and R2
    lnQ_lin = lnQ_dev(linear, keq)
    lnQ_acc = lnQ_dev(accelerated, keq)

    axes[0].plot(t, lnQ_lin[:, 0], label="Linear LLRQ", linestyle="--")
    axes[0].plot(t, lnQ_acc[:, 0], label="Accelerated LLRQ", linestyle="-")
    if mass_action is not None:
        lnQ_ma = lnQ_dev(mass_action, keq)
        axes[0].plot(t, lnQ_ma[:, 0], label="Mass Action", linestyle=":")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("ln(Q1/Keq1)")
    axes[0].legend()
    axes[0].set_title("Two-Reaction Chain: ln(Q/Keq) and Concentrations")

    axes[1].plot(t, lnQ_lin[:, 1], label="Linear LLRQ", linestyle="--")
    axes[1].plot(t, lnQ_acc[:, 1], label="Accelerated LLRQ", linestyle="-")
    if mass_action is not None:
        lnQ_ma = lnQ_dev(mass_action, keq)
        axes[1].plot(t, lnQ_ma[:, 1], label="Mass Action", linestyle=":")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("ln(Q2/Keq2)")
    axes[1].legend()

    # Concentrations: color by species, linestyle by model
    species_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    labels_lin = ["[A] Linear", "[B] Linear", "[C] Linear"]
    labels_acc = ["[A] Accelerated", "[B] Accelerated", "[C] Accelerated"]
    labels_ma = ["[A] Mass Action", "[B] Mass Action", "[C] Mass Action"]

    for i in range(3):
        axes[2].plot(t, linear["concentrations"][:, i], linestyle="--", color=species_colors[i], label=labels_lin[i])
        axes[2].plot(t, accelerated["concentrations"][:, i], linestyle="-", color=species_colors[i], label=labels_acc[i])
        if mass_action is not None:
            axes[2].plot(
                t,
                mass_action["concentrations"][:, i],
                linestyle=":",
                color=species_colors[i],
                label=labels_ma[i],
            )
    axes[2].set_ylabel("Concentration")
    axes[2].set_xlabel("Time")
    axes[2].legend(ncol=3, fontsize=9)

    fig.tight_layout()
    plt.show()


def sse_lnQ(acc_result: dict, mass_action_result: dict, keq: np.ndarray) -> float:
    lnQ_acc = lnQ_dev(acc_result, keq)
    lnQ_ma = lnQ_dev(mass_action_result, keq)
    return float(np.mean((lnQ_acc - lnQ_ma) ** 2))


def main() -> None:
    network = build_chain_network()

    # Mass action parameters and equilibrium point
    c_star = np.array([1.0, 1.5, 0.5])
    k_plus = np.array([2.0, 1.0])
    k_minus = np.array([1.0, 2.0])
    keq = k_plus / k_minus

    # Initial far-from-equilibrium condition with conserved total (sum ~3.0)
    c0 = np.array([2.85, 0.12, 0.03])

    t_eval = np.linspace(0.0, 8.0, 400)

    # Build linear dynamics
    linear_dyn = LLRQDynamics.from_mass_action(
        network=network,
        forward_rates=k_plus,
        backward_rates=k_minus,
        initial_concentrations=c_star,
        mode="equilibrium",
        relaxation_mode="linear",
    )

    # Mass action trace (if available)
    mass_action_result = simulate_mass_action(linear_dyn, network, c0, t_eval)

    # Auto-tune accelerated variant and xi_star
    xi_candidates = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Seed by average initial |ln(Q/Keq)| across both reactions
    Q0 = network.compute_reaction_quotients(c0)
    x0 = np.abs(np.log(Q0 / keq))
    xi_candidates.append(float(np.clip(np.mean(x0), 0.5, 6.0)))
    xi_candidates = sorted(set([float(x) for x in xi_candidates]))

    variants = ["sinh", "exp", "gain"]
    best = None
    for var in variants:
        for xi in xi_candidates:
            kwargs = {"xi_star": xi, "delta": 1e-3, "variant": var}
            dyn_acc = LLRQDynamics.from_mass_action(
                network=network,
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="equilibrium",
                relaxation_mode="accelerated",
                relaxation_kwargs=kwargs,
            )
            res_acc = simulate_accelerated(dyn_acc, c0, t_eval)
            if mass_action_result is None:
                # Compare against linear as a fallback proxy in absence of mass action
                res_lin = simulate_linear(linear_dyn, c0, t_eval)
                lnQ_lin = lnQ_dev(res_lin, keq)
                lnQ_acc = lnQ_dev(res_acc, keq)
                sse = float(np.mean((lnQ_acc - lnQ_lin) ** 2))
            else:
                sse = sse_lnQ(res_acc, mass_action_result, keq)
            if best is None or sse < best[0]:
                best = (sse, var, xi, res_acc, dyn_acc)

    # If mass action available, try fitting 'gain' beta directly via least squares over early window
    if HAS_TELLURIUM and mass_action_result is not None:
        # Build linear solver to fetch reduced basis and K_red
        lin_solver = LLRQSolver(linear_dyn)
        B = lin_solver._B
        K_red = B.T @ linear_dyn.K @ B
        lnKeq_consistent = lin_solver._lnKeq_consistent

        # Compute reduced y(t) from mass action result
        lnQ_ma = np.log(np.maximum(mass_action_result["reaction_quotients"], 1e-300))
        y_ma = (B.T @ (lnQ_ma.T - lnKeq_consistent[:, None])).T  # shape (T, rankS)

        # Numerical dy/dt
        dt = float(t_eval[1] - t_eval[0])
        dydt = np.gradient(y_ma, dt, axis=0)

        # Directional slopes ratio r(t) = s*_dir / s_lin_dir over a window
        T = len(t_eval)
        idx_end = int(0.25 * T)  # early segment where far-from-eq dominates
        ratios = []
        radii = []
        for i in range(1, idx_end):
            y = y_ma[i]
            ynorm2 = float(np.dot(y, y))
            if ynorm2 < 1e-14:
                continue
            s_lin_dir = float(y @ (K_red @ y) / ynorm2)
            s_star_dir = float(-np.dot(dydt[i], y) / ynorm2)
            if s_lin_dir > 1e-12 and np.isfinite(s_star_dir):
                ratios.append(s_star_dir / s_lin_dir)
                radii.append(float(np.linalg.norm(y)))

        if ratios:
            ratios = np.array(ratios)
            radii = np.array(radii)
            # Fit alpha(r) = 1 + beta * (cosh(r) - 1) via least squares
            phi = np.cosh(radii) - 1.0
            beta_fit = float(np.maximum(0.0, np.dot(phi, (ratios - 1.0)) / (np.dot(phi, phi) + 1e-12)))

            # Build gain-variant accelerated dynamics using fitted beta
            dyn_gain = LLRQDynamics.from_mass_action(
                network=network,
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="equilibrium",
                relaxation_mode="accelerated",
                relaxation_kwargs={"variant": "gain", "gain_beta": beta_fit, "gain_gamma": 1.0},
            )
            res_gain = simulate_accelerated(dyn_gain, c0, t_eval)
            sse_gain = sse_lnQ(res_gain, mass_action_result, keq)
            if best is None or sse_gain < best[0]:
                best = (sse_gain, "gain", 0.0, res_gain, dyn_gain)

    # Final accelerated pick
    _, var_choice, xi_star, accelerated_result, accelerated_dyn = best

    # Linear result
    linear_result = simulate_linear(linear_dyn, c0, t_eval)

    if mass_action_result is not None:
        sse_lin = sse_lnQ(linear_result, mass_action_result, keq)
        sse_acc = sse_lnQ(accelerated_result, mass_action_result, keq)
        print(
            f"Chosen variant={var_choice}, xi_star={xi_star:.3f} | "
            f"SSE lnQ (sum over both rxns): linear={sse_lin:.4g}, accelerated={sse_acc:.4g}"
        )

    plot_results(t_eval, keq, linear_result, accelerated_result, mass_action_result)


if __name__ == "__main__":
    main()
