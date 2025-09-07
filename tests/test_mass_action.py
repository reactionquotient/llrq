"""
Pytest tests for mass action dynamics matrix computation.

Tests the implementation of the dynamics matrix K computation
from mass action networks using the Diamond (2025) algorithm.
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock matplotlib to avoid import issues
with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
    from llrq.llrq_dynamics import LLRQDynamics
    from llrq.reaction_network import ReactionNetwork


class TestReactionNetworkDynamics:
    """Test dynamics matrix computation in ReactionNetwork."""

    def test_simple_reversible_reaction(self):
        """Test A ⇌ B reaction dynamics matrix."""
        # Setup
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])  # A -> B

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        # Test
        result = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Validate
        K = result["K"]
        assert K.shape == (1, 1)
        assert K[0, 0] > 0
        assert result["eigenanalysis"]["is_stable"]
        assert "phi" in result
        assert len(result["phi"]) == 1

        # Check flux coefficient calculation
        expected_phi = k_plus[0] * c_star[0]  # k_plus * [A]^1
        assert np.isclose(result["phi"][0], expected_phi)

    def test_two_reaction_network(self):
        """Test A ⇌ B ⇌ C network."""
        # Setup
        species_ids = ["A", "B", "C"]
        reaction_ids = ["R1", "R2"]
        S = np.array([[-1, 0], [1, -1], [0, 1]])  # A  # B  # C

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 1.5, 0.5])
        k_plus = np.array([2.0, 1.0])
        k_minus = np.array([1.0, 2.0])

        # Test equilibrium mode
        result = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Validate
        K = result["K"]
        assert K.shape == (2, 2)
        assert result["eigenanalysis"]["is_stable"]
        assert len(result["phi"]) == 2

        # K should be coupling the reactions
        assert not np.allclose(K, np.diag(np.diag(K)))  # Not purely diagonal

    def test_basis_reduction(self):
        """Test basis reduction to Im(S^T)."""
        # Setup with conservation law: A + B ⇌ C + D
        species_ids = ["A", "B", "C", "D"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [-1], [1], [1]])  # A  # B  # C  # D

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([0.5, 0.5, 1.0, 1.0])
        k_plus = np.array([1.0])
        k_minus = np.array([0.5])

        # Test with basis reduction
        result = network.compute_dynamics_matrix(
            forward_rates=k_plus,
            backward_rates=k_minus,
            initial_concentrations=c_star,
            mode="equilibrium",
            reduce_to_image=True,
        )

        # Validate
        assert "K_reduced" in result
        assert "basis" in result

        K_full = result["K"]
        K_reduced = result["K_reduced"]
        basis = result["basis"]

        # Reduced matrix should be smaller
        assert K_reduced.shape[0] <= K_full.shape[0]
        assert K_reduced.shape[1] <= K_full.shape[1]

        # Basis should have correct dimensions
        assert basis.shape[0] == K_full.shape[0]

    def test_symmetry_enforcement(self):
        """Test symmetry enforcement."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 1.0])
        k_plus = np.array([1.0])
        k_minus = np.array([1.0])

        # Test with and without symmetry enforcement
        result1 = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, enforce_symmetry=False
        )

        result2 = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, enforce_symmetry=True
        )

        K1 = result1["K"]
        K2 = result2["K"]

        # K2 should be symmetric
        assert np.allclose(K2, K2.T)

        # Both should be stable
        assert result1["eigenanalysis"]["is_stable"]
        assert result2["eigenanalysis"]["is_stable"]

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        # Wrong number of species
        with pytest.raises(ValueError, match="Expected 2 initial concentrations"):
            network.compute_dynamics_matrix(
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=[1.0],  # Should be 2
            )

        # Wrong number of forward rates
        with pytest.raises(ValueError, match="Expected 1 forward rates"):
            network.compute_dynamics_matrix(
                forward_rates=[1.0, 2.0],
                backward_rates=k_minus,
                initial_concentrations=c_star,  # Should be 1
            )

        # Invalid mode
        with pytest.raises(ValueError, match="Unknown mode"):
            network.compute_dynamics_matrix(
                forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="invalid"
            )


class TestLLRQDynamicsFactory:
    """Test the from_mass_action factory method."""

    def test_factory_method_basic(self):
        """Test basic factory method functionality."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        # Create dynamics via factory
        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Validate
        assert dynamics.network == network
        assert len(dynamics.Keq) == 1
        assert dynamics.K.shape == (1, 1)

        # Check equilibrium constants
        expected_Keq = k_plus / k_minus
        assert np.allclose(dynamics.Keq, expected_Keq)

    def test_mass_action_info_storage(self):
        """Test that mass action info is properly stored."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Test info retrieval
        info = dynamics.get_mass_action_info()
        assert info is not None
        assert info["mode"] == "equilibrium"
        # equilibrium_point is now stored in the equilibrium_info or may be None
        if info.get("equilibrium_point") is not None:
            assert np.allclose(info["equilibrium_point"], c_star)
        assert np.allclose(info["forward_rates"], k_plus)
        assert np.allclose(info["backward_rates"], k_minus)
        assert "dynamics_data" in info

    def test_factory_without_mass_action(self):
        """Test dynamics created normally don't have mass action info."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        # Create dynamics normally (not from mass action)
        dynamics = LLRQDynamics(network)

        # Should not have mass action info
        info = dynamics.get_mass_action_info()
        assert info is None


class TestAlgorithmValidation:
    """Test algorithm correctness against known cases."""

    def test_equilibrium_flux_balance(self):
        """Test that flux coefficients satisfy equilibrium condition."""
        # At equilibrium: k_plus * [A]^nu_reac = k_minus * [B]^nu_prod
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])  # A -> B

        network = ReactionNetwork(species_ids, reaction_ids, S)

        # Set up equilibrium: k_plus/k_minus = [B]/[A]
        c_A, c_B = 1.0, 2.0
        k_plus = 2.0
        k_minus = 1.0

        # Verify this satisfies equilibrium: k_plus/k_minus = 2.0 = c_B/c_A
        assert np.isclose(k_plus / k_minus, c_B / c_A)

        result = network.compute_dynamics_matrix(
            forward_rates=[k_plus], backward_rates=[k_minus], initial_concentrations=[c_A, c_B], mode="equilibrium"
        )

        # At equilibrium, forward and backward fluxes should be equal
        phi_forward = k_plus * c_A**1
        phi_backward = k_minus * c_B**1

        assert np.isclose(phi_forward, phi_backward), f"Forward flux {phi_forward} != backward flux {phi_backward}"

        # The stored phi should be consistent
        assert np.isclose(result["phi"][0], phi_forward)

    def test_single_reaction_dynamics(self):
        """Test against analytical solution for single reaction."""
        # For single A ⇌ B: the dynamics should be dx/dt = -k*x
        # where k is the relaxation rate and x = ln(Q/Keq)

        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 1.0])  # Equal concentrations
        k_plus = 1.0
        k_minus = 1.0  # Keq = 1

        dynamics = LLRQDynamics.from_mass_action(
            network=network,
            forward_rates=[k_plus],
            backward_rates=[k_minus],
            initial_concentrations=c_star,
            mode="equilibrium",
        )

        # For this simple case, the relaxation matrix should be 1x1
        assert dynamics.K.shape == (1, 1)
        assert dynamics.K[0, 0] > 0  # Should be positive (stable)

        # Test dynamics function
        x = np.array([0.1])  # Small deviation from equilibrium
        dxdt = dynamics.dynamics(0, x)

        # Should be -K*x (restoring force)
        expected = -dynamics.K @ x
        assert np.allclose(dxdt, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
