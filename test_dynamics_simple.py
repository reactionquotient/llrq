#!/usr/bin/env python3
"""
Simple test script for mass action dynamics matrix computation.
Tests the core functionality without full package import.
"""

import numpy as np
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llrq.reaction_network import ReactionNetwork
from llrq.llrq_dynamics import LLRQDynamics


def test_simple_reversible():
    """Test A ⇌ B reaction."""
    print("Testing simple A ⇌ B reaction...")
    
    # Network: A ⇌ B
    species_ids = ['A', 'B']
    reaction_ids = ['R1']
    S = np.array([[-1], [1]])  # A -> B
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    
    # Parameters
    c_star = np.array([1.0, 2.0])  # Equilibrium concentrations
    k_plus = np.array([2.0])       # Forward rate
    k_minus = np.array([1.0])      # Backward rate
    
    # Test equilibrium mode
    result = network.compute_dynamics_matrix(
        equilibrium_point=c_star,
        forward_rates=k_plus,
        backward_rates=k_minus,
        mode='equilibrium'
    )
    
    K = result['K']
    print(f"Dynamics matrix K:\n{K}")
    print(f"Eigenvalues: {result['eigenanalysis']['eigenvalues']}")
    print(f"Is stable: {result['eigenanalysis']['is_stable']}")
    
    # Basic validation
    assert K.shape == (1, 1), f"Expected K shape (1,1), got {K.shape}"
    assert K[0, 0] > 0, f"Expected positive diagonal element, got {K[0, 0]}"
    
    print("✓ Simple reaction test passed!\n")


def test_factory_method():
    """Test the from_mass_action factory method."""
    print("Testing from_mass_action factory method...")
    
    species_ids = ['A', 'B']
    reaction_ids = ['R1']
    S = np.array([[-1], [1]])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    
    c_star = np.array([1.0, 2.0])
    k_plus = np.array([2.0])
    k_minus = np.array([1.0])
    
    dynamics = LLRQDynamics.from_mass_action(
        network=network,
        equilibrium_point=c_star,
        forward_rates=k_plus,
        backward_rates=k_minus,
        mode='equilibrium'
    )
    
    print(f"Equilibrium constants: {dynamics.Keq}")
    print(f"Relaxation matrix shape: {dynamics.K.shape}")
    
    # Test mass action info
    info = dynamics.get_mass_action_info()
    assert info is not None, "Should have mass action info"
    assert info['mode'] == 'equilibrium', "Should store mode correctly"
    
    print("✓ Factory method test passed!\n")


def run_tests():
    """Run all test cases."""
    print("Running mass action dynamics tests...")
    print("=" * 40)
    
    try:
        test_simple_reversible()
        test_factory_method()
        
        print("All tests passed successfully! ✓")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)