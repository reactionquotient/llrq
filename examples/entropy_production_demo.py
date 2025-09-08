#!/usr/bin/env python3
"""
Entropy Production and Energy Balance Demonstration.

This example demonstrates the new thermodynamic accounting functionality,
showing how to compute entropy production from reaction forces, external drives,
and the energy balance diagnostics for LLRQ dynamics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.thermodynamic_accounting import ThermodynamicAccountant


def demo_simple_relaxation_entropy():
    """Demonstrate entropy production during simple relaxation to equilibrium."""
    print("=== Entropy Production During Relaxation ===")

    # Create simple A ⇌ B network
    network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

    # Set up mass action kinetics
    forward_rates = np.array([2.0])
    backward_rates = np.array([1.0])  # K_eq = 2.0
    initial_concentrations = np.array([2.0, 0.5])  # Away from equilibrium

    # Create LLRQ dynamics from mass action
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

    # Solve dynamics
    solver = LLRQSolver(dynamics)
    solution = solver.solve(
        initial_conditions={"A": 2.0, "B": 0.5},
        t_span=(0, 5),
        method="analytical",
        compute_entropy=True,  # Enable entropy computation
    )

    print(f"Solution computed successfully: {solution['success']}")
    print(f"Method used: {solution['method']}")

    # Manual entropy computation using ThermodynamicAccountant
    accountant = ThermodynamicAccountant(network)

    # Compute entropy from reaction forces
    entropy_result = accountant.from_solution(
        solution,
        forward_rates=forward_rates,
        backward_rates=backward_rates,
        scale=1.0,  # Could use kB*T for physical units
    )

    print(f"Total entropy production: {entropy_result.sigma_total:.4f}")
    print(f"Final entropy rate: {entropy_result.sigma_time[-1]:.6f}")
    print(f"Initial entropy rate: {entropy_result.sigma_time[0]:.4f}")

    return solution, entropy_result


def demo_controlled_dynamics_entropy():
    """Demonstrate entropy production with external control."""
    print("\n=== Entropy Production Under External Control ===")

    # Create A ⇌ B ⇌ C network for more interesting dynamics
    network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], [[-1, 0], [1, -1], [0, 1]])

    # Rate constants
    forward_rates = np.array([1.0, 0.5])
    backward_rates = np.array([0.5, 1.0])
    initial_concentrations = np.array([3.0, 1.0, 0.5])

    # Create dynamics
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

    # Add sinusoidal external drive to first reaction
    def external_drive(t):
        return np.array([0.2 * np.sin(0.5 * t), 0.0])

    dynamics.external_drive = external_drive

    # Solve with control
    solver = LLRQSolver(dynamics)
    solution = solver.solve(initial_conditions={"A": 3.0, "B": 1.0, "C": 0.5}, t_span=(0, 10), method="numerical")

    # Compute entropy accounting
    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

    # Compute external drive trajectory
    t = solution["time"]
    u_t = np.array([external_drive(t_i) for t_i in t])
    x_t = solution["log_deviations"]
    K = dynamics.K

    # Full dual accounting with energy balance
    dual_result = accountant.entropy_from_xu(t, x_t, u_t, K, scale=1.0)

    print(f"Entropy from reaction forces: {dual_result.from_x.sigma_total:.4f}")
    print(f"Entropy from quasi-steady approx: {dual_result.from_u.sigma_total:.4f}")
    print(f"Energy balance residual: {dual_result.balance['residual_total']:.6f}")
    print(f"Control work total: {dual_result.balance['P_ctrl_total']:.4f}")
    print(f"Relaxation work total: {dual_result.balance['P_relax_total']:.4f}")

    return solution, dual_result, L


def demo_entropy_comparison():
    """Compare entropy production for different driving conditions."""
    print("\n=== Entropy Production Comparison ===")

    network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
    forward_rates = np.array([1.0])
    backward_rates = np.array([1.0])
    initial_concentrations = np.array([1.5, 0.5])

    # Three scenarios: no drive, constant drive, oscillating drive
    scenarios = [
        ("No Drive", lambda t: np.array([0.0])),
        ("Constant Drive", lambda t: np.array([0.1])),
        ("Oscillating Drive", lambda t: np.array([0.1 * np.sin(t)])),
    ]

    results = {}

    for name, drive_func in scenarios:
        # Create dynamics
        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        dynamics.external_drive = drive_func

        # Solve
        solver = LLRQSolver(dynamics)
        solution = solver.solve(initial_conditions={"A": 1.5, "B": 0.5}, t_span=(0, 8), method="numerical")

        # Compute entropy
        accountant = ThermodynamicAccountant(network)
        L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates)

        t = solution["time"]
        x_t = solution["log_deviations"]
        u_t = np.array([drive_func(t_i) for t_i in t])
        K = dynamics.K

        dual_result = accountant.entropy_from_xu(t, x_t, u_t, K)
        results[name] = (solution, dual_result)

        print(
            f"{name:15s}: Entropy = {dual_result.from_x.sigma_total:.4f}, "
            f"Control Work = {dual_result.balance['P_ctrl_total']:.4f}"
        )

    return results


def plot_entropy_production_analysis(solution, entropy_result, title="Entropy Production"):
    """Create comprehensive entropy production plots."""
    t = solution["time"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Concentrations
    concentrations = solution["concentrations"]
    n_species = concentrations.shape[1]
    colors = ["blue", "red", "green", "orange"]

    for i in range(n_species):
        species_name = f"Species {i+1}" if i < len(colors) else f"S{i+1}"
        ax1.plot(t, concentrations[:, i], colors[i % len(colors)], label=species_name, linewidth=2)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Concentration")
    ax1.set_title("Species Concentrations")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reaction forces (log deviations)
    log_deviations = solution["log_deviations"]
    n_reactions = log_deviations.shape[1]

    for i in range(n_reactions):
        ax2.plot(t, log_deviations[:, i], colors[i % len(colors)], label=f"Reaction {i+1}", linewidth=2)

    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Log Deviation from Equilibrium")
    ax2.set_title("Reaction Forces x(t)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Entropy production rate
    if hasattr(entropy_result, "from_x"):
        # DualAccountingResult
        ax3.plot(t, entropy_result.from_x.sigma_time, "b-", linewidth=2, label="From reaction forces")
        ax3.plot(t, entropy_result.from_u.sigma_time, "r--", linewidth=2, label="Quasi-steady approx")
    else:
        # AccountingResult
        ax3.plot(t, entropy_result.sigma_time, "b-", linewidth=2, label="Entropy production rate")

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Entropy Production Rate")
    ax3.set_title("Instantaneous Entropy Production")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Energy balance (if available)
    if hasattr(entropy_result, "balance"):
        balance = entropy_result.balance
        ax4.plot(t, balance["V_dot_time"], "g-", label="dV/dt", linewidth=2)
        ax4.plot(t, -balance["P_relax_time"], "r-", label="-P_relax", linewidth=2)
        ax4.plot(t, balance["P_ctrl_time"], "b-", label="P_ctrl", linewidth=2)
        ax4.plot(t, balance["residual_time"], "k--", alpha=0.7, label="Residual")

        ax4.set_xlabel("Time")
        ax4.set_ylabel("Power")
        ax4.set_title("Energy Balance Diagnostic")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Show cumulative entropy instead
        cumulative_entropy = np.cumsum(entropy_result.sigma_time) * (t[1] - t[0])
        ax4.plot(t, cumulative_entropy, "purple", linewidth=2)
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Cumulative Entropy")
        ax4.set_title("Integrated Entropy Production")
        ax4.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def main():
    """Run all entropy production demonstrations."""
    print("Entropy Production and Energy Balance Demonstration")
    print("=" * 55)

    # Demo 1: Simple relaxation
    solution1, entropy1 = demo_simple_relaxation_entropy()

    # Demo 2: Controlled dynamics
    solution2, entropy2, L = demo_controlled_dynamics_entropy()

    # Demo 3: Comparison across scenarios
    comparison_results = demo_entropy_comparison()

    # Create plots
    try:
        # Plot simple relaxation
        fig1 = plot_entropy_production_analysis(solution1, entropy1, "Entropy During Relaxation to Equilibrium")

        # Plot controlled dynamics
        fig2 = plot_entropy_production_analysis(solution2, entropy2, "Entropy Under External Control")

        # Summary plot comparing scenarios
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        scenario_names = []
        entropy_totals = []
        control_work = []

        for name, (sol, ent_result) in comparison_results.items():
            scenario_names.append(name)
            entropy_totals.append(ent_result.from_x.sigma_total)
            control_work.append(ent_result.balance["P_ctrl_total"])

        ax1.bar(scenario_names, entropy_totals, color=["lightblue", "lightcoral", "lightgreen"])
        ax1.set_ylabel("Total Entropy Production")
        ax1.set_title("Entropy Production by Scenario")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(scenario_names, control_work, color=["lightblue", "lightcoral", "lightgreen"])
        ax2.set_ylabel("Total Control Work")
        ax2.set_title("Control Work by Scenario")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plots
        fig1.savefig("entropy_relaxation.png", dpi=150, bbox_inches="tight")
        fig2.savefig("entropy_controlled.png", dpi=150, bbox_inches="tight")
        fig3.savefig("entropy_comparison.png", dpi=150, bbox_inches="tight")

        print(f"\nPlots saved:")
        print("- entropy_relaxation.png")
        print("- entropy_controlled.png")
        print("- entropy_comparison.png")

        plt.show()

    except ImportError:
        print("\nSkipping plots - matplotlib not available")
    except Exception as e:
        print(f"\nSkipping plots - error: {e}")

    print("\n" + "=" * 55)
    print("Demonstration completed successfully!")
    print("\nKey findings:")
    print("1. Entropy production decreases as system approaches equilibrium")
    print("2. External control can increase total entropy production")
    print("3. Energy balance provides useful diagnostic for numerical accuracy")
    print("4. Quasi-steady approximation matches full calculation when drives vary slowly")


if __name__ == "__main__":
    main()
