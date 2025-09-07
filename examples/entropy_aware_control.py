#!/usr/bin/env python3
"""
Entropy-Aware Steady-State Control Demonstration.

This example demonstrates how to design constant control inputs that trade off
between reaching a target steady state and minimizing entropy production cost.

The key equation is the control-metric quadratic:
σ_u(t) = u(t)^T M u(t), where M := K^{-T} L K^{-1}

This is a textbook pullback of the metric L by the map u ↦ x = K^{-1} u:
||u||_M^2 = ||K^{-1} u||_L^2
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.control import LLRQController
from llrq.thermodynamic_accounting import ThermodynamicAccountant


def demo_entropy_control_tradeoff():
    """Demonstrate entropy-control tradeoff for different entropy weights."""
    print("=== Entropy-Aware Steady-State Control Demonstration ===\n")

    # Create A ⇌ B ⇌ C reaction network
    network = ReactionNetwork(
        species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
    )

    # Set up mass action kinetics
    forward_rates = np.array([2.0, 1.5])
    backward_rates = np.array([1.0, 0.8])
    initial_concentrations = np.array([2.0, 1.0, 0.5])

    print(f"Network: A ⇌ B ⇌ C")
    print(f"Forward rates: {forward_rates}")
    print(f"Backward rates: {backward_rates}")
    print(f"Initial concentrations: {initial_concentrations}")

    # Create LLRQ dynamics
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

    # Create solver and controller
    solver = LLRQSolver(dynamics)
    controller = LLRQController(solver, controlled_reactions=[0, 1])  # Control both reactions

    print(f"Relaxation matrix K:\n{dynamics.K}")
    print(f"Equilibrium constants: {np.exp(solver._lnKeq_consistent)}")

    # Compute Onsager conductance
    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")
    print(f"Onsager conductance L:\n{L}")

    # Define target state (away from equilibrium)
    x_target = np.array([0.5, -0.3])  # Target reaction forces
    print(f"Target reaction forces: {x_target}")

    # Compute control entropy metric
    M = controller.compute_control_entropy_metric(L)
    print(f"Control entropy metric M = K^{{-T}} L K^{{-1}}:\n{M}")

    # Explore tradeoff curve by varying entropy weight
    entropy_weights = np.logspace(-3, 2, 20)  # λ from 0.001 to 100
    results = []

    print(f"\n{'λ':>8} {'Track Err':>10} {'Entropy':>10} {'Total Cost':>12} {'||u||':>8}")
    print("-" * 60)

    for lam in entropy_weights:
        result = controller.compute_entropy_aware_steady_state_control(
            x_target=x_target, L=L, entropy_weight=lam, controlled_reactions_only=True
        )

        control_norm = np.linalg.norm(result["u_optimal"])
        results.append(result)

        print(
            f"{lam:8.3f} {result['tracking_error']:10.4f} {result['entropy_rate']:10.4f} "
            f"{result['total_cost']:12.4f} {control_norm:8.4f}"
        )

    # Special cases: exact tracking (λ=0) and minimal entropy (large λ)
    print(f"\n=== Special Cases ===")

    # Exact tracking (no entropy penalty)
    exact_result = controller.compute_entropy_aware_steady_state_control(
        x_target=x_target, L=L, entropy_weight=0.0, controlled_reactions_only=True
    )
    print(f"Exact tracking (λ=0):")
    print(f"  Control input: {exact_result['u_optimal']}")
    print(f"  Achieved state: {exact_result['x_achieved']}")
    print(f"  Tracking error: {exact_result['tracking_error']:.6f}")
    print(f"  Entropy rate: {exact_result['entropy_rate']:.4f}")

    # High entropy penalty
    minimal_entropy_result = controller.compute_entropy_aware_steady_state_control(
        x_target=x_target, L=L, entropy_weight=1000.0, controlled_reactions_only=True
    )
    print(f"\nHigh entropy penalty (λ=1000):")
    print(f"  Control input: {minimal_entropy_result['u_optimal']}")
    print(f"  Achieved state: {minimal_entropy_result['x_achieved']}")
    print(f"  Tracking error: {minimal_entropy_result['tracking_error']:.4f}")
    print(f"  Entropy rate: {minimal_entropy_result['entropy_rate']:.6f}")

    return results, exact_result, minimal_entropy_result


def plot_tradeoff_analysis(results):
    """Plot the Pareto frontier of tracking error vs entropy production."""

    # Extract data for plotting
    entropy_weights = [r["entropy_weight"] for r in results]
    tracking_errors = [r["tracking_error"] for r in results]
    entropy_rates = [r["entropy_rate"] for r in results]
    total_costs = [r["total_cost"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Entropy-Aware Control Tradeoff Analysis", fontsize=14)

    # Plot 1: Pareto frontier (tracking error vs entropy)
    ax = axes[0, 0]
    ax.loglog(tracking_errors, entropy_rates, "bo-", markersize=4)
    ax.set_xlabel("Tracking Error ||x - x_target||²")
    ax.set_ylabel("Entropy Production Rate σ_u")
    ax.set_title("Pareto Frontier")
    ax.grid(True, alpha=0.3)

    # Annotate extreme points
    ax.annotate(
        "λ=0\n(exact tracking)",
        xy=(tracking_errors[0], entropy_rates[0]),
        xytext=(tracking_errors[0] * 10, entropy_rates[0]),
        arrowprops=dict(arrowstyle="->", alpha=0.7),
        fontsize=8,
    )
    ax.annotate(
        "λ→∞\n(minimal entropy)",
        xy=(tracking_errors[-1], entropy_rates[-1]),
        xytext=(tracking_errors[-1] * 0.1, entropy_rates[-1] * 10),
        arrowprops=dict(arrowstyle="->", alpha=0.7),
        fontsize=8,
    )

    # Plot 2: Cost components vs entropy weight
    ax = axes[0, 1]
    ax.loglog(entropy_weights, tracking_errors, "r-", label="Tracking Error")
    ax.loglog(entropy_weights, entropy_rates, "b-", label="Entropy Rate")
    ax.loglog(entropy_weights, total_costs, "g--", label="Total Cost")
    ax.set_xlabel("Entropy Weight λ")
    ax.set_ylabel("Cost Components")
    ax.set_title("Cost vs Entropy Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Control effort
    control_norms = [np.linalg.norm(r["u_optimal"]) for r in results]
    ax = axes[1, 0]
    ax.loglog(entropy_weights, control_norms, "mo-", markersize=4)
    ax.set_xlabel("Entropy Weight λ")
    ax.set_ylabel("Control Effort ||u||")
    ax.set_title("Control Effort vs Entropy Weight")
    ax.grid(True, alpha=0.3)

    # Plot 4: Efficiency ratio (tracking/entropy)
    efficiency = np.array(tracking_errors) / np.array(entropy_rates)
    ax = axes[1, 1]
    ax.loglog(entropy_weights, efficiency, "co-", markersize=4)
    ax.set_xlabel("Entropy Weight λ")
    ax.set_ylabel("Efficiency (Tracking/Entropy)")
    ax.set_title("Control Efficiency")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_simulation_validation():
    """Validate the steady-state predictions by running actual simulations."""
    print(f"\n=== Simulation Validation ===")

    # Same setup as before
    network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

    forward_rates = np.array([2.0])
    backward_rates = np.array([1.0])
    initial_concentrations = np.array([1.0, 1.0])

    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

    solver = LLRQSolver(dynamics)
    controller = LLRQController(solver, controlled_reactions=[0])

    # Compute Onsager conductance
    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

    # Target state
    x_target = np.array([0.5])

    # Compute optimal control for moderate entropy weight
    result = controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=1.0)

    u_optimal = result["u_optimal"]
    x_predicted = result["x_achieved"]

    print(f"Optimal control: {u_optimal[0]:.4f}")
    print(f"Predicted steady state: {x_predicted[0]:.4f}")
    print(f"Target state: {x_target[0]:.4f}")

    # Simulate with constant control to verify
    dynamics.external_drive = lambda t: u_optimal

    solution = solver.solve(initial_conditions={"A": 1.0, "B": 1.0}, t_span=(0, 10), method="numerical")

    # Check final state
    x_final_sim = solution["log_deviations"][-1]

    print(f"Simulated final state: {x_final_sim[0]:.4f}")
    print(f"Prediction error: {abs(x_final_sim[0] - x_predicted[0]):.6f}")

    # Verify entropy calculation
    actual_entropy = accountant.entropy_from_xu(
        solution["time"], solution["log_deviations"], np.array([u_optimal for _ in solution["time"]]), dynamics.K, L
    )

    predicted_entropy = result["entropy_rate"]
    print(f"Predicted entropy rate: {predicted_entropy:.4f}")
    print(f"Simulated entropy rate (final): {actual_entropy.from_u.sigma_time[-1]:.4f}")


if __name__ == "__main__":
    # Run demonstrations
    results, exact_result, minimal_entropy_result = demo_entropy_control_tradeoff()

    # Create plots
    plot_tradeoff_analysis(results)

    # Validate with simulation
    demo_simulation_validation()

    print(f"\nDemonstration completed successfully!")
