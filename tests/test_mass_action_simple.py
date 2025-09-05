"""
Simple pytest tests for mass action dynamics matrix computation.
Directly imports only what's needed to avoid matplotlib issues.
"""

import numpy as np
import pytest
import sys
import os

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly to avoid __init__.py issues
from llrq.reaction_network import ReactionNetwork
from llrq.llrq_dynamics import LLRQDynamics


class TestMassActionDynamics:
    """Test mass action dynamics matrix computation."""

    def test_simple_reversible_reaction(self):
        """Test A ⇌ B reaction dynamics matrix."""
        # Setup
        species_ids = ['A', 'B']
        reaction_ids = ['R1']
        S = np.array([[-1], [1]])  # A -> B
        
        network = ReactionNetwork(species_ids, reaction_ids, S)
        
        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])
        
        # Test
        result = network.compute_dynamics_matrix(
            equilibrium_point=c_star,
            forward_rates=k_plus,
            backward_rates=k_minus,
            mode='equilibrium'
        )
        
        # Validate
        K = result['K']
        assert K.shape == (1, 1)
        assert K[0, 0] > 0
        assert result['eigenanalysis']['is_stable']
        assert 'phi' in result
        assert len(result['phi']) == 1

    def test_factory_method(self):
        """Test from_mass_action factory method."""
        species_ids = ['A', 'B']
        reaction_ids = ['R1']
        S = np.array([[-1], [1]])
        
        network = ReactionNetwork(species_ids, reaction_ids, S)
        
        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])
        
        # Create dynamics via factory
        dynamics = LLRQDynamics.from_mass_action(
            network=network,
            equilibrium_point=c_star,
            forward_rates=k_plus,
            backward_rates=k_minus,
            mode='equilibrium'
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
        species_ids = ['A', 'B', 'C']
        reaction_ids = ['R1', 'R2']
        S = np.array([
            [-1, 0],   # A
            [1, -1],   # B
            [0, 1]     # C
        ])
        
        network = ReactionNetwork(species_ids, reaction_ids, S)
        
        c_star = np.array([1.0, 1.5, 0.5])
        k_plus = np.array([2.0, 1.0])
        k_minus = np.array([1.0, 2.0])
        
        result = network.compute_dynamics_matrix(
            equilibrium_point=c_star,
            forward_rates=k_plus,
            backward_rates=k_minus,
            mode='equilibrium'
        )
        
        K = result['K']
        assert K.shape == (2, 2)
        assert result['eigenanalysis']['is_stable']

    def test_invalid_inputs(self):
        """Test error handling."""
        species_ids = ['A', 'B']
        reaction_ids = ['R1']
        S = np.array([[-1], [1]])
        
        network = ReactionNetwork(species_ids, reaction_ids, S)
        
        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])
        
        # Wrong number of species
        with pytest.raises(ValueError, match="Expected 2 equilibrium concentrations"):
            network.compute_dynamics_matrix(
                equilibrium_point=[1.0],  # Should be 2
                forward_rates=k_plus,
                backward_rates=k_minus
            )

    def test_equilibrium_condition(self):
        """Test equilibrium flux balance."""
        species_ids = ['A', 'B']
        reaction_ids = ['R1']
        S = np.array([[-1], [1]])
        
        network = ReactionNetwork(species_ids, reaction_ids, S)
        
        # Set up true equilibrium: k_plus/k_minus = [B]/[A]
        c_A, c_B = 1.0, 2.0
        k_plus = 2.0
        k_minus = 1.0
        
        result = network.compute_dynamics_matrix(
            equilibrium_point=[c_A, c_B],
            forward_rates=[k_plus],
            backward_rates=[k_minus],
            mode='equilibrium'
        )
        
        # Forward and backward fluxes should be equal at equilibrium
        phi_forward = k_plus * c_A**1
        phi_backward = k_minus * c_B**1
        
        assert np.isclose(phi_forward, phi_backward), \
            f"Forward flux {phi_forward} != backward flux {phi_backward}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])