#!/usr/bin/env python3
"""
Test script for mass action dynamics matrix computation.

This script validates the implementation of the dynamics matrix K
computation from mass action networks.
"""

import os
import sys

import numpy as np

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llrq import LLRQDynamics, ReactionNetwork


def test_simple_reversible_reaction():
    """Test A ⇌ B reaction."""
    print("Testing simple A ⇌ B reaction...")

    # Network: A ⇌ B
    species_ids = ["A", "B"]
    reaction_ids = ["R1"]
    S = np.array([[-1], [1]])  # A -> B

    network = ReactionNetwork(species_ids, reaction_ids, S)

    # Parameters
    c_star = np.array([1.0, 2.0])  # Equilibrium concentrations
    k_plus = np.array([2.0])  # Forward rate
    k_minus = np.array([1.0])  # Backward rate

    # Test equilibrium mode
    result = network.compute_dynamics_matrix(
        forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
    )

    K = result["K"]
    print(f"Dynamics matrix K:\n{K}")
    print(f"Eigenvalues: {result['eigenanalysis']['eigenvalues']}")
    print(f"Is stable: {result['eigenanalysis']['is_stable']}")

    # Test factory method
    dynamics = LLRQDynamics.from_mass_action(
        network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
    )

    print(f"Equilibrium constants: {dynamics.Keq}")
    print(f"Relaxation matrix shape: {dynamics.K.shape}")

    # Basic validation
    assert K.shape == (1, 1), f"Expected K shape (1,1), got {K.shape}"
    assert K[0, 0] > 0, f"Expected positive diagonal element, got {K[0, 0]}"

    print("✓ Simple reaction test passed!\n")


def test_two_reaction_network():
    """Test A ⇌ B ⇌ C network."""
    print("Testing A ⇌ B ⇌ C network...")

    # Network: A ⇌ B ⇌ C
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2"]
    S = np.array([[-1, 0], [1, -1], [0, 1]])  # A  # B  # C

    network = ReactionNetwork(species_ids, reaction_ids, S)

    # Parameters
    c_star = np.array([1.0, 1.5, 0.5])  # Equilibrium concentrations
    k_plus = np.array([2.0, 1.0])  # Forward rates
    k_minus = np.array([1.0, 2.0])  # Backward rates

    # Test equilibrium mode
    result = network.compute_dynamics_matrix(
        forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium", reduce_to_image=True
    )

    K = result["K"]
    K_reduced = result.get("K_reduced")

    print(f"Full dynamics matrix K:\n{K}")
    if K_reduced is not None:
        print(f"Reduced dynamics matrix K_reduced:\n{K_reduced}")
        print(f"Basis matrix shape: {result['basis'].shape}")

    print(f"Eigenvalues: {result['eigenanalysis']['eigenvalues']}")
    print(f"Is stable: {result['eigenanalysis']['is_stable']}")

    # Test nonequilibrium mode
    result_neq = network.compute_dynamics_matrix(
        forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="nonequilibrium"
    )

    print(f"Nonequilibrium K:\n{result_neq['K']}")

    # Basic validation
    assert K.shape == (2, 2), f"Expected K shape (2,2), got {K.shape}"
    assert result["eigenanalysis"]["is_stable"], "System should be stable"

    print("✓ Two reaction network test passed!\n")


def test_symmetry_enforcement():
    """Test symmetry enforcement."""
    print("Testing symmetry enforcement...")

    # Simple A ⇌ B reaction
    species_ids = ["A", "B"]
    reaction_ids = ["R1"]
    S = np.array([[-1], [1]])

    network = ReactionNetwork(species_ids, reaction_ids, S)

    c_star = np.array([1.0, 1.0])
    k_plus = np.array([1.0])
    k_minus = np.array([1.0])

    # Without symmetry enforcement
    result1 = network.compute_dynamics_matrix(
        forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, enforce_symmetry=False
    )

    # With symmetry enforcement
    result2 = network.compute_dynamics_matrix(
        forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, enforce_symmetry=True
    )

    K1 = result1["K"]
    K2 = result2["K"]

    print(f"Original K:\n{K1}")
    print(f"Symmetrized K:\n{K2}")

    # Check symmetry
    is_symmetric = np.allclose(K2, K2.T)
    print(f"Is symmetrized K symmetric: {is_symmetric}")

    assert is_symmetric, "Enforced symmetry should produce symmetric matrix"

    print("✓ Symmetry enforcement test passed!\n")


def test_conservation_laws():
    """Test with conservation laws."""
    print("Testing conservation laws...")

    # Closed system A + B ⇌ C + D
    species_ids = ["A", "B", "C", "D"]
    reaction_ids = ["R1"]
    S = np.array([[-1], [-1], [1], [1]])  # A  # B  # C  # D

    network = ReactionNetwork(species_ids, reaction_ids, S)

    # Check conservation laws
    conservation = network.find_conservation_laws()
    print(f"Conservation matrix shape: {conservation.shape}")
    print(f"Conservation laws:\n{conservation}")

    c_star = np.array([0.5, 0.5, 1.0, 1.0])
    k_plus = np.array([1.0])
    k_minus = np.array([0.5])

    result = network.compute_dynamics_matrix(
        forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium", reduce_to_image=True
    )

    print(f"Dynamics matrix shape: {result['K'].shape}")
    if "K_reduced" in result:
        print(f"Reduced matrix shape: {result['K_reduced'].shape}")

    print("✓ Conservation laws test passed!\n")


def run_all_tests():
    """Run all test cases."""
    print("Running mass action dynamics matrix tests...")
    print("=" * 50)

    try:
        test_simple_reversible_reaction()
        test_two_reaction_network()
        test_symmetry_enforcement()
        test_conservation_laws()

        print("=" * 50)
        print("All tests passed successfully! ✓")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
