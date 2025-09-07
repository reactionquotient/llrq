#!/usr/bin/env python3
"""
Simple example demonstrating the LLRQ package.

This example shows how to:
1. Create a simple A ⇌ B reaction system
2. Solve the log-linear dynamics
3. Visualize the results
"""

import matplotlib.pyplot as plt
import numpy as np

import llrq


def main():
    print("LLRQ Package Example: Simple A ⇌ B Reaction")
    print("=" * 50)

    # Create a simple reaction system
    network, dynamics, solver, visualizer = llrq.simple_reaction(
        reactant_species="A",
        product_species="B",
        equilibrium_constant=2.0,
        relaxation_rate=1.0,
        initial_concentrations={"A": 1.0, "B": 0.1},
    )

    # Print network summary
    print("\nReaction Network:")
    print(network.summary())

    # Solve the dynamics
    solution = solver.solve(initial_conditions={"A": 1.0, "B": 0.1}, t_span=(0, 10), method="analytical")

    print(f"\nSolution successful: {solution['success']}")
    print(f"Method used: {solution['method']}")

    # Plot results
    fig = visualizer.plot_dynamics(solution)
    plt.show()

    # Demonstrate single reaction analysis
    print("\nSingle reaction analysis:")
    single_solution = solver.solve_single_reaction(
        reaction_id="R1", initial_concentrations={"A": 1.0, "B": 0.1}, t_span=(0, 10)
    )

    fig2 = visualizer.plot_single_reaction(reaction_id="R1", initial_concentrations={"A": 1.0, "B": 0.1}, t_span=(0, 10))
    plt.show()

    # Demonstrate external drive
    print("\nExternal drive example:")

    def step_drive(t):
        return np.array([0.5 if t > 5 else 0.0])

    def oscillating_drive(t):
        return np.array([0.3 * np.sin(2 * np.pi * t)])

    fig3 = visualizer.plot_external_drive_response(
        initial_conditions={"A": 1.0, "B": 0.1},
        drive_functions=[step_drive, oscillating_drive],
        drive_labels=["Step drive", "Oscillating drive"],
        t_span=(0, 10),
    )
    plt.show()

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
