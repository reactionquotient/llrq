#!/usr/bin/env python3
"""
Selective Removal Example with Moiety Dynamics

Demonstrates moiety dynamics for a system with selective species removal,
such as membrane separation or precipitation. Uses a simple A + B ⇌ C
reaction network where species can be selectively removed.

Key concepts:
- Moiety-respecting removal: L @ R = A_y @ L for some A_y
- Non-moiety-respecting removal: creates coupling between x and y blocks
- Design considerations for control in the presence of selectivity

System: A + B ⇌ C with selective removal of C (product separation)
- Conservation: [A] + [C] = const, [B] + [C] = const  (2 moieties)
- Selective removal preferentially removes C via membrane/precipitation
- Demonstrates both moiety-respecting and non-respecting cases
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.open_system_network import OpenSystemNetwork
from llrq.moiety_dynamics import MoietyDynamics
from llrq.moiety_controller import MoietyController
from llrq.open_system_solver import OpenSystemSolver


def create_selective_removal_example():
    """Create A + B ⇌ C system with selective product removal."""

    # System setup: A + B ⇌ C
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1"]
    S = np.array(
        [
            [-1.0],  # A coefficient in R1 (A + B → C)
            [-1.0],  # B coefficient in R1
            [1.0],
        ]
    )  # C coefficient in R1

    print("=== Selective Removal Example: A + B ⇌ C ===\n")
    print("Stoichiometric matrix S:")
    print(S)

    # Create base network to check conservation laws
    from llrq.reaction_network import ReactionNetwork

    base_network = ReactionNetwork(species_ids, reaction_ids, S)
    conservation_laws = base_network.find_conservation_laws()
    print(f"\nConservation laws (L matrix):")
    print(conservation_laws)
    print("Conservation law interpretations:")
    for i, law in enumerate(conservation_laws):
        species_terms = [f"{law[j]:.1f}*[{species_ids[j]}]" for j in range(len(species_ids)) if abs(law[j]) > 1e-10]
        print(f"  Law {i+1}: {' + '.join(species_terms)} = constant")

    return species_ids, reaction_ids, S, conservation_laws


def demonstrate_moiety_respecting_removal():
    """Demonstrate moiety-respecting removal (uniform removal rates)."""

    print("\n" + "=" * 50)
    print("CASE 1: MOIETY-RESPECTING REMOVAL")
    print("=" * 50)

    species_ids, reaction_ids, S, L = create_selective_removal_example()

    # Moiety-respecting removal: uniform removal rate for all species
    # R = diag([r, r, r]) with r = removal rate
    # This satisfies L @ R = r * L = A_y @ L with A_y = r * I

    removal_rate = 0.1
    removal_matrix = removal_rate * np.eye(3)  # Uniform removal

    print(f"Removal matrix R (uniform removal, rate = {removal_rate}):")
    print(removal_matrix)

    # Check if moiety-respecting
    LR = L @ removal_matrix
    L_pinv = np.linalg.pinv(L)
    A_y = LR @ L_pinv
    reconstructed = A_y @ L

    print(f"\nL @ R =")
    print(LR)
    print(f"A_y @ L =")
    print(reconstructed)
    print(f"Is moiety-respecting: {np.allclose(LR, reconstructed, atol=1e-10)}")
    print(f"A_y matrix:")
    print(A_y)

    # Create open system network
    network = OpenSystemNetwork(
        species_ids=species_ids,
        reaction_ids=reaction_ids,
        stoichiometric_matrix=S,
        flow_config={
            "type": "batch_with_removal",
            "removal_matrix": removal_matrix,
            "inlet_composition": [0.0, 0.0, 0.0],  # No inlet
        },
    )

    # Create moiety dynamics
    K = 1.0  # Reaction rate
    Keq = np.array([2.0])  # Equilibrium constant favoring products

    moiety_dynamics = MoietyDynamics(network=network, K=K, equilibrium_constants=Keq)

    # Configure removal dynamics
    moiety_dynamics.configure_moiety_respecting_removal(
        removal_matrix=removal_matrix,
        inlet_totals=np.array([0.0, 0.0]),  # No inlet
    )

    return network, moiety_dynamics


def demonstrate_non_moiety_respecting_removal():
    """Demonstrate non-moiety-respecting removal (selective removal)."""

    print("\n" + "=" * 50)
    print("CASE 2: NON-MOIETY-RESPECTING REMOVAL")
    print("=" * 50)

    species_ids, reaction_ids, S, L = create_selective_removal_example()

    # Non-moiety-respecting removal: selective removal of C only
    # This breaks the moiety conservation structure
    removal_matrix = np.array(
        [
            [0.0, 0.0, 0.0],  # A not removed
            [0.0, 0.0, 0.0],  # B not removed
            [0.0, 0.0, 0.5],  # C removed at rate 0.5
        ]
    )

    print(f"Removal matrix R (selective C removal):")
    print(removal_matrix)

    # Check if moiety-respecting
    LR = L @ removal_matrix
    L_pinv = np.linalg.pinv(L)
    A_y = LR @ L_pinv
    reconstructed = A_y @ L

    print(f"\nL @ R =")
    print(LR)
    print(f"A_y @ L =")
    print(reconstructed)
    print(f"Is moiety-respecting: {np.allclose(LR, reconstructed, atol=1e-10)}")
    print(f"Error in moiety-respecting assumption:")
    print(LR - reconstructed)

    # Create open system network
    try:
        network = OpenSystemNetwork(
            species_ids=species_ids,
            reaction_ids=reaction_ids,
            stoichiometric_matrix=S,
            flow_config={"type": "selective_removal", "removal_matrix": removal_matrix, "inlet_composition": [0.0, 0.0, 0.0]},
        )

        print(f"Network created successfully")
        print(f"Is moiety-respecting: {network.is_moiety_respecting_flow()}")

    except ValueError as e:
        print(f"Warning: {e}")
        # Create anyway for demonstration
        network = OpenSystemNetwork(species_ids=species_ids, reaction_ids=reaction_ids, stoichiometric_matrix=S)
        network.flow_config = {"type": "selective_removal", "removal_matrix": removal_matrix}
        network.removal_matrix = removal_matrix

    # Create moiety dynamics (may have coupling)
    K = 1.0
    Keq = np.array([2.0])

    moiety_dynamics = MoietyDynamics(network=network, K=K, equilibrium_constants=Keq)

    # Try to configure (will show warnings about non-respecting nature)
    try:
        moiety_dynamics.configure_moiety_respecting_removal(removal_matrix=removal_matrix, inlet_totals=np.array([0.0, 0.0]))
    except ValueError as e:
        print(f"Configuration error: {e}")
        # Manual configuration for demonstration
        moiety_dynamics.A_y = -A_y  # Use approximate A_y
        moiety_dynamics.g_y = np.zeros(2)

    return network, moiety_dynamics


def simulate_and_compare():
    """Simulate both cases and compare results."""

    print("\n" + "=" * 50)
    print("SIMULATION COMPARISON")
    print("=" * 50)

    # Get both systems
    network1, dynamics1 = demonstrate_moiety_respecting_removal()
    network2, dynamics2 = demonstrate_non_moiety_respecting_removal()

    # Initial conditions
    c0 = np.array([1.5, 1.0, 0.2])  # Some A, B, and small amount of C

    print(f"\nInitial concentrations: [A] = {c0[0]}, [B] = {c0[1]}, [C] = {c0[2]}")

    # Check initial conservation laws
    L = network1.find_conservation_laws()
    conserved_quantities = L @ c0
    print(f"Initial conserved quantities: {conserved_quantities}")

    # Simulation parameters
    t_final = 10.0
    dt = 0.05
    t = np.arange(0, t_final + dt, dt)

    # Simulate both systems
    solver1 = OpenSystemSolver(dynamics1, network1)
    solver2 = OpenSystemSolver(dynamics2, network2)

    try:
        result1 = solver1.solve_analytical(c0, t)
        print("Case 1 simulation successful")
    except Exception as e:
        print(f"Case 1 simulation error: {e}")
        result1 = None

    try:
        result2 = solver2.solve_analytical(c0, t)
        print("Case 2 simulation successful")
    except Exception as e:
        print(f"Case 2 simulation error: {e}")
        result2 = None

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Selective Removal: Moiety-Respecting vs Non-Respecting", fontsize=14)

    colors = ["blue", "red", "green"]
    species_labels = ["[A]", "[B]", "[C]"]

    # Plot concentrations for both cases
    for case_idx, (result, case_name) in enumerate([(result1, "Moiety-Respecting"), (result2, "Non-Respecting")]):
        if result is None:
            continue

        # Concentrations
        ax = axes[case_idx, 0]
        for i, (color, label) in enumerate(zip(colors, species_labels)):
            ax.plot(t, result["concentrations"][:, i], color=color, label=label, linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration (M)")
        ax.set_title(f"{case_name}: Concentrations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Moiety totals
        ax = axes[case_idx, 1]
        if "y" in result and result["y"].shape[1] > 0:
            for i in range(result["y"].shape[1]):
                ax.plot(t, result["y"][:, i], label=f"Moiety {i+1}", linewidth=2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Moiety Total")
        ax.set_title(f"{case_name}: Moiety Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reaction quotients
        ax = axes[case_idx, 2]
        if "Q" in result:
            ax.semilogy(t, result["Q"][:, 0], "purple", label="Q = [C]/([A][B])", linewidth=2)
            ax.axhline(dynamics1.Keq[0], color="k", linestyle="--", label=f"Keq = {dynamics1.Keq[0]}", alpha=0.7)
        ax.set_xlabel("Time")
        ax.set_ylabel("Reaction Quotient")
        ax.set_title(f"{case_name}: Quotient Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Analysis
    if result1 is not None and result2 is not None:
        print(f"\n=== Analysis ===")

        # Final concentrations
        c1_final = result1["concentrations"][-1]
        c2_final = result2["concentrations"][-1]

        print(f"Final concentrations (moiety-respecting): {c1_final}")
        print(f"Final concentrations (selective): {c2_final}")

        # Conservation law violations
        conserved1_final = L @ c1_final
        conserved2_final = L @ c2_final

        print(f"Final conserved quantities (moiety-respecting): {conserved1_final}")
        print(f"Initial conserved quantities: {conserved_quantities}")
        print(f"Conservation error (moiety-respecting): {np.linalg.norm(conserved1_final - conserved_quantities):.6f}")

        print(f"Final conserved quantities (selective): {conserved2_final}")
        print(f"Conservation error (selective): {np.linalg.norm(conserved2_final - conserved_quantities):.6f}")

        # Product removal efficiency
        c_removed_1 = c0[2] - c1_final[2]
        c_removed_2 = c0[2] - c2_final[2]

        print(f"Product C removed (moiety-respecting): {c_removed_1:.3f} ({100*c_removed_1/c0[2]:.1f}%)")
        print(f"Product C removed (selective): {c_removed_2:.3f} ({100*c_removed_2/c0[2]:.1f}%)")


def main():
    """Run the selective removal example."""

    try:
        simulate_and_compare()

        print(f"\n=== Key Insights ===")
        print(f"1. Moiety-respecting removal maintains block-triangular structure")
        print(f"2. Selective removal creates coupling between quotient and total dynamics")
        print(f"3. Both approaches can be effective for product separation")
        print(f"4. Control design must account for moiety structure")
        print(f"5. Selective removal may offer better separation efficiency")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory with llrq package available")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
