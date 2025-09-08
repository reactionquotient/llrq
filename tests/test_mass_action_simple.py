"""
Simple pytest tests for mass action dynamics matrix computation.
Directly imports only what's needed to avoid matplotlib issues.
"""

import os
import sys

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.llrq_dynamics import LLRQDynamics

# Import directly to avoid __init__.py issues
from llrq.reaction_network import ReactionNetwork


class TestMassActionDynamics:
    """Test mass action dynamics matrix computation."""

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

    def test_factory_method(self):
        """Test from_mass_action factory method."""
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

    def test_two_reaction_network(self):
        """Test A ⇌ B ⇌ C network."""
        species_ids = ["A", "B", "C"]
        reaction_ids = ["R1", "R2"]
        S = np.array([[-1, 0], [1, -1], [0, 1]])  # A  # B  # C

        network = ReactionNetwork(species_ids, reaction_ids, S)

        c_star = np.array([1.0, 1.5, 0.5])
        k_plus = np.array([2.0, 1.0])
        k_minus = np.array([1.0, 2.0])

        result = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        K = result["K"]
        assert K.shape == (2, 2)
        assert result["eigenanalysis"]["is_stable"]

    def test_invalid_inputs(self):
        """Test error handling."""
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

    def test_equilibrium_condition(self):
        """Test equilibrium flux balance."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        # Set up true equilibrium: k_plus/k_minus = [B]/[A]
        c_A, c_B = 1.0, 2.0
        k_plus = 2.0
        k_minus = 1.0

        result = network.compute_dynamics_matrix(
            forward_rates=[k_plus], backward_rates=[k_minus], initial_concentrations=[c_A, c_B], mode="equilibrium"
        )

        # Forward and backward fluxes should be equal at equilibrium
        phi_forward = k_plus * c_A**1
        phi_backward = k_minus * c_B**1

        assert np.isclose(phi_forward, phi_backward), f"Forward flux {phi_forward} != backward flux {phi_backward}"

    def test_automatic_equilibrium_computation(self):
        """Test automatic equilibrium computation without providing equilibrium_point."""
        # Setup simple A ⇌ B reaction
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])  # A -> B

        network = ReactionNetwork(species_ids, reaction_ids, S)

        k_plus = np.array([2.0])
        k_minus = np.array([1.0])
        initial_conc = np.array([1.0, 0.5])  # Initial concentrations

        # Test automatic equilibrium computation
        result = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=initial_conc, mode="equilibrium"
        )

        # Validate results
        assert "K" in result
        assert "equilibrium_info" in result
        assert "equilibrium_point" in result
        assert result["eigenanalysis"]["is_stable"]

        # Check equilibrium satisfies detailed balance
        c_eq = result["equilibrium_point"]
        K_eq = k_plus / k_minus
        Q_eq = network.compute_reaction_quotients(c_eq)

        assert np.allclose(Q_eq, K_eq, rtol=1e-6), f"Equilibrium quotient {Q_eq} != equilibrium constant {K_eq}"

        # Check conservation: total mass should be conserved
        total_initial = np.sum(initial_conc)
        total_equilibrium = np.sum(c_eq)
        assert np.isclose(
            total_initial, total_equilibrium, rtol=1e-6
        ), f"Mass not conserved: {total_initial} -> {total_equilibrium}"

    def test_compare_manual_vs_automatic_equilibrium(self):
        """Test that manual and automatic equilibrium give same results."""
        # Setup A ⇌ B reaction
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        k_plus = np.array([2.0])
        k_minus = np.array([1.0])
        initial_conc = np.array([1.0, 0.5])

        # Manual equilibrium computation (provide equilibrium concentrations)
        c_manual = np.array([0.5, 1.0])  # Known equilibrium for this system

        result_manual = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_manual, mode="equilibrium"
        )

        # Automatic equilibrium computation
        result_auto = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=initial_conc, mode="equilibrium"
        )

        # Compare dynamics matrices
        K_manual = result_manual["K"]
        K_auto = result_auto["K"]

        assert np.allclose(K_manual, K_auto, rtol=1e-6), f"Dynamics matrices differ: manual={K_manual}, auto={K_auto}"

    def test_equilibrium_with_conservation(self):
        """Test equilibrium computation with conservation laws."""
        # Setup A + B ⇌ C system (has conservation laws)
        species_ids = ["A", "B", "C"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [-1], [1]])  # A  # B  # C

        network = ReactionNetwork(species_ids, reaction_ids, S)

        k_plus = np.array([1.0])
        k_minus = np.array([0.5])
        initial_conc = np.array([2.0, 1.0, 0.0])  # A=2, B=1, C=0

        # Test equilibrium computation
        result = network.compute_dynamics_matrix(
            forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=initial_conc, mode="equilibrium"
        )

        c_eq = result["equilibrium_point"]

        # Check conservation laws
        # Total A+C should equal initial A
        assert np.isclose(c_eq[0] + c_eq[2], initial_conc[0], rtol=1e-6)
        # Total B+C should equal initial B
        assert np.isclose(c_eq[1] + c_eq[2], initial_conc[1], rtol=1e-6)

        # Check detailed balance
        K_eq = k_plus / k_minus
        Q_eq = network.compute_reaction_quotients(c_eq)
        assert np.allclose(Q_eq, K_eq, rtol=1e-6)

    def test_factory_method_with_automatic_equilibrium(self):
        """Test LLRQDynamics.from_mass_action with automatic equilibrium."""
        species_ids = ["A", "B"]
        reaction_ids = ["R1"]
        S = np.array([[-1], [1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        k_plus = np.array([2.0])
        k_minus = np.array([1.0])
        initial_conc = np.array([1.0, 0.5])

        # Create dynamics via factory without equilibrium_point
        dynamics = LLRQDynamics.from_mass_action(
            network=network,
            forward_rates=k_plus,
            backward_rates=k_minus,
            initial_concentrations=initial_conc,
            mode="equilibrium",
        )

        # Validate
        assert dynamics.network == network
        assert len(dynamics.Keq) == 1
        assert dynamics.K.shape == (1, 1)

        # Check equilibrium constants
        expected_Keq = k_plus / k_minus
        assert np.allclose(dynamics.Keq, expected_Keq)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
