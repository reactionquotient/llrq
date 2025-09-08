#!/usr/bin/env python3
"""
Direct test of dynamics computation without package imports.
"""

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import pinv


# Direct copy of the essential parts to test the algorithm
class TestReactionNetwork:
    """Minimal version for testing dynamics matrix computation."""

    def __init__(self, species_ids, reaction_ids, stoichiometric_matrix):
        self.species_ids = species_ids
        self.reaction_ids = reaction_ids
        self.S = np.array(stoichiometric_matrix, dtype=float)
        self.n_species = len(species_ids)
        self.n_reactions = len(reaction_ids)

    def compute_dynamics_matrix(self, equilibrium_point, forward_rates, backward_rates, mode="equilibrium"):
        """Test version of dynamics matrix computation."""
        c_star = np.array(equilibrium_point)
        k_plus = np.array(forward_rates)
        k_minus = np.array(backward_rates)

        D_star_inv = np.diag(1.0 / np.maximum(c_star, 1e-12))

        if mode == "equilibrium":
            # Compute φ_j = k_j^+ (c*)^(ν_j^reac) = k_j^- (c*)^(ν_j^prod)
            phi = np.zeros(self.n_reactions)
            for j in range(self.n_reactions):
                nu_reac = np.maximum(-self.S[:, j], 0)
                phi_forward = k_plus[j] * np.prod(c_star**nu_reac)
                phi[j] = phi_forward

            Phi = np.diag(phi)
            K = self.S.T @ D_star_inv @ self.S @ Phi
        else:
            raise NotImplementedError("Only equilibrium mode implemented in test")

        eigenvals, eigenvecs = np.linalg.eig(K)

        return {
            "K": K,
            "phi": phi,
            "eigenanalysis": {
                "eigenvalues": eigenvals,
                "eigenvectors": eigenvecs,
                "is_stable": np.all(eigenvals.real >= -1e-12),
            },
        }


def test_simple_reaction():
    """Test A ⇌ B reaction."""
    print("Testing A ⇌ B reaction...")

    network = TestReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

    result = network.compute_dynamics_matrix(
        equilibrium_point=[1.0, 2.0], forward_rates=[2.0], backward_rates=[1.0], mode="equilibrium"
    )

    K = result["K"]
    print(f"K matrix: {K}")
    print(f"φ coefficients: {result['phi']}")
    print(f"Eigenvalues: {result['eigenanalysis']['eigenvalues']}")
    print(f"Is stable: {result['eigenanalysis']['is_stable']}")

    # Validation
    assert K.shape == (1, 1)
    assert K[0, 0] > 0
    assert result["eigenanalysis"]["is_stable"]

    print("✓ Simple reaction test passed!")


def test_two_reactions():
    """Test A ⇌ B ⇌ C network."""
    print("\nTesting A ⇌ B ⇌ C network...")

    network = TestReactionNetwork(["A", "B", "C"], ["R1", "R2"], [[-1, 0], [1, -1], [0, 1]])

    result = network.compute_dynamics_matrix(
        equilibrium_point=[1.0, 1.5, 0.5], forward_rates=[2.0, 1.0], backward_rates=[1.0, 2.0], mode="equilibrium"
    )

    K = result["K"]
    print(f"K matrix:\n{K}")
    print(f"φ coefficients: {result['phi']}")
    print(f"Eigenvalues: {result['eigenanalysis']['eigenvalues']}")
    print(f"Is stable: {result['eigenanalysis']['is_stable']}")

    # Validation
    assert K.shape == (2, 2)
    assert result["eigenanalysis"]["is_stable"]

    print("✓ Two reaction network test passed!")


def test_algorithm_components():
    """Test individual components of the algorithm."""
    print("\nTesting algorithm components...")

    # Simple case: A ⇌ B with known solution
    # At equilibrium: k_plus * [A] = k_minus * [B]
    # So if [A] = 1, [B] = 2, then k_plus/k_minus = 2
    # φ = k_plus * [A] = k_minus * [B] should be equal

    c_star = np.array([1.0, 2.0])
    k_plus = 2.0
    k_minus = 1.0

    # Check flux consistency
    phi_forward = k_plus * c_star[0] ** 1  # A^1
    phi_backward = k_minus * c_star[1] ** 1  # B^1

    print(f"Forward flux: {phi_forward}")
    print(f"Backward flux: {phi_backward}")
    print(f"Ratio: {phi_forward/phi_backward}")

    # For equilibrium, these should be equal (Keq = k_plus/k_minus = [B]/[A])
    expected_ratio = (k_plus / k_minus) / (c_star[1] / c_star[0])
    print(f"Expected ratio: {expected_ratio}")

    assert np.isclose(
        phi_forward, phi_backward, rtol=0.1
    ), "Forward and backward fluxes should be approximately equal at equilibrium"

    print("✓ Algorithm components test passed!")


def main():
    """Run all tests."""
    print("Testing mass action dynamics matrix computation")
    print("=" * 50)

    try:
        test_simple_reaction()
        test_two_reactions()
        test_algorithm_components()

        print("\n" + "=" * 50)
        print("All tests passed successfully! ✓")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
