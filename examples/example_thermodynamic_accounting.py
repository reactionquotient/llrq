#!/usr/bin/env python3
"""
Example demonstrating thermodynamic accounting with Onsager conductance.

This example shows how to use the new thermodynamic accounting features
added to ReactionNetwork, including Onsager conductance, reaction forces,
and detailed balance checking.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.reaction_network import ReactionNetwork


def example_simple_reaction():
    """Demonstrate thermodynamic accounting for simple A ⇌ B reaction."""
    print("=== Simple A ⇌ B Reaction ===")

    # Create network: A ⇌ B
    network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

    # Rate constants
    k_plus = np.array([2.0])  # Forward: A → B
    k_minus = np.array([1.0])  # Reverse: B → A
    # Equilibrium constant K_eq = k+/k- = 2.0

    print(f"Rate constants: k+ = {k_plus[0]}, k- = {k_minus[0]}")
    print(f"Equilibrium constant K_eq = {k_plus[0]/k_minus[0]}")

    # Test at equilibrium: [B]/[A] = K_eq = 2.0
    c_eq = np.array([1.0, 2.0])
    print(f"\nAt equilibrium concentrations [A]={c_eq[0]}, [B]={c_eq[1]}:")

    # Compute Onsager conductance
    onsager_eq = network.compute_onsager_conductance(c_eq, k_plus, k_minus)
    print(f"  Near equilibrium: {onsager_eq['near_equilibrium']}")
    print(f"  Mode used: {onsager_eq['mode_used']}")
    print(f"  Onsager conductance L: {onsager_eq['L'][0,0]:.4f}")

    # Check detailed balance
    balance_eq = network.check_detailed_balance(c_eq, k_plus, k_minus)
    print(f"  Detailed balance satisfied: {balance_eq['detailed_balance']}")
    print(f"  Forward flux: {balance_eq['forward_flux'][0]:.4f}")
    print(f"  Reverse flux: {balance_eq['reverse_flux'][0]:.4f}")

    # Test away from equilibrium: [B]/[A] = 0.5 ≠ K_eq
    c_neq = np.array([4.0, 2.0])
    print(f"\nAway from equilibrium [A]={c_neq[0]}, [B]={c_neq[1]}:")

    onsager_neq = network.compute_onsager_conductance(c_neq, k_plus, k_minus)
    print(f"  Near equilibrium: {onsager_neq['near_equilibrium']}")
    print(f"  Mode used: {onsager_neq['mode_used']}")
    print(f"  Onsager conductance L: {onsager_neq['L'][0,0]:.4f}")

    # Reaction forces
    forces = network.compute_reaction_forces(c_neq, k_plus, k_minus)
    print(f"  Reaction force x: {forces[0]:.4f}")
    print(f"  (Negative force drives reaction forward)")

    balance_neq = network.check_detailed_balance(c_neq, k_plus, k_minus)
    print(f"  Detailed balance satisfied: {balance_neq['detailed_balance']}")
    print(f"  Max imbalance: {balance_neq['max_imbalance']:.4f}")


def example_bimolecular_reaction():
    """Demonstrate thermodynamic accounting for A + B ⇌ C reaction."""
    print("\n=== Bimolecular A + B ⇌ C Reaction ===")

    # Create network: A + B ⇌ C
    network = ReactionNetwork(["A", "B", "C"], ["R1"], [[-1], [-1], [1]])

    k_plus = np.array([1.0])  # Forward: A + B → C
    k_minus = np.array([0.5])  # Reverse: C → A + B
    # K_eq = 2.0

    print(f"Rate constants: k+ = {k_plus[0]}, k- = {k_minus[0]}")
    print(f"Equilibrium constant K_eq = {k_plus[0]/k_minus[0]}")

    # At equilibrium: [C]/([A]*[B]) = K_eq = 2.0
    c_eq = np.array([1.0, 1.0, 2.0])  # Q = 2.0/(1.0*1.0) = 2.0 = K_eq
    print(f"\nAt equilibrium [A]={c_eq[0]}, [B]={c_eq[1]}, [C]={c_eq[2]}:")

    # Thermodynamic analysis
    onsager = network.compute_onsager_conductance(c_eq, k_plus, k_minus)
    forces = network.compute_reaction_forces(c_eq, k_plus, k_minus)
    B_matrix = network.compute_flux_response_matrix(c_eq)
    linear_K = network.compute_linear_relaxation_matrix(c_eq, k_plus, k_minus)

    print(f"  Reaction quotient Q: {onsager['reaction_forces'][0] + np.log(k_plus[0]/k_minus[0]):.4f}")
    print(f"  Equilibrium constant K_eq: {k_plus[0]/k_minus[0]:.4f}")
    print(f"  Reaction force: {forces[0]:.6f} (should be ~0 at equilibrium)")
    print(f"  Onsager conductance L: {onsager['L'][0,0]:.4f}")
    print(f"  Flux response matrix B: {B_matrix[0,0]:.4f}")
    print(f"  Linear relaxation K = BL: {linear_K['K'][0,0]:.4f}")


def example_auto_mode_selection():
    """Demonstrate automatic mode selection for Onsager conductance."""
    print("\n=== Auto Mode Selection Example ===")

    network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
    k_plus = np.array([1.0])
    k_minus = np.array([1.0])  # K_eq = 1.0

    # Test different concentration ratios
    test_cases = [
        ("Near equilibrium", [1.0, 1.0]),  # [B]/[A] = 1.0 ≈ K_eq = 1.0
        ("Slightly off", [1.0, 0.8]),  # [B]/[A] = 0.8
        ("Moderately off", [1.0, 0.5]),  # [B]/[A] = 0.5
        ("Far from equilibrium", [1.0, 0.1]),  # [B]/[A] = 0.1
    ]

    for description, concentrations in test_cases:
        c = np.array(concentrations)
        result = network.compute_onsager_conductance(c, k_plus, k_minus, mode="auto")

        print(f"{description:20s}: [A]={c[0]:.1f}, [B]={c[1]:.1f} → {result['mode_used']:11s} mode")


def plot_conductance_vs_concentration():
    """Plot how Onsager conductance varies with concentration."""
    print("\n=== Plotting Conductance vs Concentration ===")

    network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
    k_plus = np.array([2.0])
    k_minus = np.array([1.0])  # K_eq = 2.0

    # Vary [B] while keeping [A] = 1.0
    B_concentrations = np.logspace(-1, 1, 50)  # 0.1 to 10
    A_concentration = 1.0

    L_values = []
    modes = []
    forces = []

    for B_conc in B_concentrations:
        c = np.array([A_concentration, B_conc])
        result = network.compute_onsager_conductance(c, k_plus, k_minus, mode="auto")

        L_values.append(result["L"][0, 0])
        modes.append(result["mode_used"])
        forces.append(result["reaction_forces"][0])

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Onsager conductance
    ax1.loglog(B_concentrations, L_values, "b-", linewidth=2)
    ax1.axvline(2.0, color="red", linestyle="--", alpha=0.7, label="Equilibrium [B] = 2.0")
    ax1.set_xlabel("[B] concentration")
    ax1.set_ylabel("Onsager conductance L")
    ax1.set_title("Onsager Conductance vs Concentration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reaction forces
    ax2.semilogx(B_concentrations, forces, "g-", linewidth=2)
    ax2.axhline(0, color="red", linestyle="--", alpha=0.7, label="Equilibrium (force = 0)")
    ax2.axvline(2.0, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("[B] concentration")
    ax2.set_ylabel("Reaction force x")
    ax2.set_title("Reaction Forces vs Concentration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("thermodynamic_accounting_example.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'thermodynamic_accounting_example.png'")


if __name__ == "__main__":
    print("Thermodynamic Accounting Example")
    print("================================")

    # Run examples
    example_simple_reaction()
    example_bimolecular_reaction()
    example_auto_mode_selection()

    # Create plot (optional - requires matplotlib)
    try:
        plot_conductance_vs_concentration()
    except ImportError:
        print("\nSkipping plot - matplotlib not available")
    except Exception as e:
        print(f"\nSkipping plot - error: {e}")

    print("\nExample completed successfully!")
