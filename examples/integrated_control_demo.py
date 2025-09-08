#!/usr/bin/env python3
"""
Demonstration of the integrated mass action control workflow.

This example shows how the core ideas from linear_vs_mass_action.py have been
integrated into the main LLRQ codebase for easy use.

Workflow demonstrated:
1. Setup reaction with initial concentration
2. Choose target point
3. Figure out static control input to get to target point
4. Simulate controlled dynamics with mass action
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import llrq


def main():
    """Demonstrate the integrated controlled simulation workflow."""
    print("LLRQ Integrated Control Workflow Demonstration")
    print("=" * 50)

    # Step 1: Setup reaction network with initial concentrations
    print("\n1. Setting up 3-cycle reaction network...")

    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2", "R3"]

    # Stoichiometric matrix (3 species x 3 reactions)
    S = np.array(
        [[-1, 0, 1], [1, -1, 0], [0, 1, -1]]  # A: -1 in R1, 0 in R2, +1 in R3  # B: +1 in R1, -1 in R2, 0 in R3
    )  # C: 0 in R1, +1 in R2, -1 in R3

    # Species info with initial concentrations
    species_info = {
        "A": {"name": "A", "initial_concentration": 2.0, "compartment": "cell", "boundary_condition": False},
        "B": {"name": "B", "initial_concentration": 0.2, "compartment": "cell", "boundary_condition": False},
        "C": {"name": "C", "initial_concentration": 0.1, "compartment": "cell", "boundary_condition": False},
    }

    # Reaction info
    reaction_info = [
        {"id": "R1", "name": "A ⇌ B", "reactants": [("A", 1.0)], "products": [("B", 1.0)], "reversible": True},
        {"id": "R2", "name": "B ⇌ C", "reactants": [("B", 1.0)], "products": [("C", 1.0)], "reversible": True},
        {"id": "R3", "name": "C ⇌ A", "reactants": [("C", 1.0)], "products": [("A", 1.0)], "reversible": True},
    ]

    # Create network
    network = llrq.ReactionNetwork(species_ids, reaction_ids, S, species_info, reaction_info)

    print(f"✓ Network created with {len(species_ids)} species and {len(reaction_ids)} reactions")

    # Step 2: Choose target point
    print("\\n2. Choosing target concentrations...")

    initial_concentrations = {"A": 2.0, "B": 0.2, "C": 0.1}
    target_concentrations = {"A": 0.8, "B": 1.2, "C": 0.3}  # Total mass = 2.3 (same as initial)

    print(f"Initial: {initial_concentrations}")
    print(f"Target:  {target_concentrations}")
    print(f"Total mass change: {sum(initial_concentrations.values()):.1f} → {sum(target_concentrations.values()):.1f}")

    # Method 1: One-line simulation using high-level API
    print("\\n3. Method 1: One-line controlled simulation")
    print("-" * 40)

    result = llrq.simulate_to_target(
        network,
        initial_concentrations=initial_concentrations,
        target_concentrations=target_concentrations,
        controlled_reactions=["R1", "R3"],  # Control reactions 1 and 3
        t_span=(0, 100),
        method="linear",
        forward_rates=[3.0, 1.0, 3.0],  # Thermodynamically consistent
        backward_rates=[1.5, 2.0, 3.0],  # Keq = [2.0, 0.5, 1.0] → product = 1.0 ✓
        feedback_gain=2.0,
    )

    print(f"✓ Simulation completed in {len(result['time'])} time steps")
    print(f"Final concentrations: {dict(zip(species_ids, result['concentrations'][-1]))}")
    print(f"Final total mass: {np.sum(result['concentrations'][-1]):.3f}")

    # Method 2: Advanced workflow using ControlledSimulation class
    print("\\n4. Method 2: Advanced controlled simulation workflow")
    print("-" * 50)

    # Create ControlledSimulation object
    controlled_sim = llrq.ControlledSimulation.from_mass_action(
        network=network,
        forward_rates=[3.0, 1.0, 3.0],
        backward_rates=[1.5, 2.0, 3.0],
        initial_concentrations=[2.0, 0.2, 0.1],  # For equilibrium computation
        controlled_reactions=["R1", "R3"],
    )

    print("✓ ControlledSimulation created from mass action parameters")

    # Simulate to target
    result2 = controlled_sim.simulate_to_target(
        initial_concentrations=initial_concentrations,
        target_state=target_concentrations,
        t_span=(0, 100),
        method="linear",
        feedback_gain=2.0,
    )

    print(f"✓ Target simulation completed")

    # Analyze performance
    metrics = controlled_sim.analyze_performance(result2, target_concentrations)

    print("\\nPerformance Analysis:")
    print(f"  Final tracking error: {metrics['final_error']:.6f}")
    print(f"  RMS tracking error: {metrics['rms_error']:.6f}")
    print(f"  Maximum tracking error: {metrics['max_error']:.6f}")
    print(f"  Settling time: {metrics['settling_time']:.1f}s" if metrics["settling_time"] else "  Settling time: Not achieved")
    print(f"  Steady state achieved: {metrics['steady_state_achieved']}")

    # Method 3: Compare linear vs mass action (if tellurium is available)
    print("\\n5. Method 3: Compare linear LLRQ vs mass action")
    print("-" * 50)

    try:
        comparison = llrq.compare_control_methods(
            network,
            initial_concentrations=initial_concentrations,
            target_concentrations=target_concentrations,
            controlled_reactions=["R1", "R3"],
            t_span=(0, 100),
            forward_rates=[3.0, 1.0, 3.0],
            backward_rates=[1.5, 2.0, 3.0],
            feedback_gain=2.0,
        )

        print("✓ Comparison completed")

        # Analyze comparison
        linear_final = comparison["linear_result"]["concentrations"][-1]
        mass_action_final = comparison["mass_action_result"]["concentrations"][-1]

        print(f"Linear LLRQ final:   {dict(zip(species_ids, linear_final))}")
        print(f"Mass action final:   {dict(zip(species_ids, mass_action_final))}")
        print(f"Method difference:   {np.linalg.norm(linear_final - mass_action_final):.6f}")

        # Performance comparison
        linear_metrics = controlled_sim.analyze_performance(comparison["linear_result"], target_concentrations)
        mass_action_metrics = controlled_sim.analyze_performance(comparison["mass_action_result"], target_concentrations)

        print("\\nPerformance Comparison:")
        print(f"  Linear LLRQ RMS error:    {linear_metrics['rms_error']:.6f}")
        print(f"  Mass action RMS error:    {mass_action_metrics['rms_error']:.6f}")
        print(f"  Difference:               {abs(linear_metrics['rms_error'] - mass_action_metrics['rms_error']):.6f}")

    except Exception as e:
        print(f"Mass action comparison not available: {e}")
        print("(Install tellurium for mass action simulation)")

    # Summary
    print("\\n" + "=" * 50)
    print("🎉 Integration Demo Complete!")
    print("\\nKey Benefits:")
    print("• One-line controlled simulation: llrq.simulate_to_target()")
    print("• Advanced workflows with ControlledSimulation class")
    print("• Automatic equilibrium computation from rate constants")
    print("• Built-in performance analysis")
    print("• Seamless linear vs mass action comparisons")
    print("• All core workflow steps from linear_vs_mass_action.py now integrated")


if __name__ == "__main__":
    main()
