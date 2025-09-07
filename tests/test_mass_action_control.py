"""
Comprehensive test suite for mass action control implementation.

This test suite validates the mathematical correctness of LLRQ control
applied to mass action kinetics via asymmetric rate modifications.
"""

import os
import sys

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.integrations.mass_action_drive import (
    apply_llrq_drive_to_rates,
    compute_equilibrium_shift,
    validate_llrq_control_mapping,
)
from llrq.llrq_dynamics import LLRQDynamics
from llrq.mass_action_simulator import MassActionSimulator
from llrq.reaction_network import ReactionNetwork
from llrq.solver import LLRQSolver


class TestSingleReaction:
    """Test LLRQ control on simple A ⇌ B reaction (analytically solvable)."""

    def setup_method(self):
        """Set up single reaction A ⇌ B."""
        self.species_ids = ["A", "B"]
        self.reaction_ids = ["R1"]
        self.S = np.array([[-1], [1]])  # A -> B

        self.species_info = {"A": {"initial_concentration": 2.0}, "B": {"initial_concentration": 0.5}}

        self.reaction_info = [{"id": "R1"}]

        self.network = ReactionNetwork(self.species_ids, self.reaction_ids, self.S, self.species_info, self.reaction_info)

        # Use realistic rate constants
        self.kf, self.kr = 2.0, 1.0  # Keq = 2.0
        self.rate_constants = {"R1": (self.kf, self.kr)}

        # Set up LLRQ with proper relaxation matrix
        Keq = np.array([self.kf / self.kr])
        K = np.array([[self.kf + self.kr]])  # Proper relaxation rate

        self.dynamics = LLRQDynamics(self.network, Keq, K)
        self.solver = LLRQSolver(self.dynamics)

    def test_equilibrium_without_control(self):
        """Test that uncontrolled system reaches correct equilibrium."""
        sim = MassActionSimulator(
            self.network,
            self.rate_constants,
            B=self.solver._B,
            K_red=self.solver._B.T @ self.solver.dynamics.K @ self.solver._B,
        )

        # Simulate to equilibrium
        t_eval = np.linspace(0, 10, 1000)  # Long enough to reach equilibrium
        result = sim.simulate(t_eval)

        # Check final equilibrium
        c_final = result["concentrations"][-1]
        Q_final = result["reaction_quotients"][-1]

        # Should match Keq = 2.0
        expected_Keq = self.kf / self.kr
        np.testing.assert_allclose(Q_final[0], expected_Keq, rtol=1e-2, err_msg="System should reach correct equilibrium")

        print(f"Final Q: {Q_final[0]}, Expected Keq: {expected_Keq}")

    def test_control_mapping_consistency(self):
        """Test that control mapping math is consistent."""
        sim = MassActionSimulator(
            self.network,
            self.rate_constants,
            B=self.solver._B,
            K_red=self.solver._B.T @ self.solver.dynamics.K @ self.solver._B,
        )

        # Test control input
        u_red = np.array([0.5])  # Should shift ln(Keq) by ~0.5

        # Apply control
        sim.apply_llrq_control(u_red)

        # Check rate modifications
        new_rates = sim.get_current_rates()
        kf_new, kr_new = new_rates["R1"]

        # Compute equilibrium shift
        Keq_old = self.kf / self.kr
        Keq_new = kf_new / kr_new
        delta_computed = np.log(Keq_new / Keq_old)

        # Expected shift from LLRQ theory
        K_red = sim.K_red
        Delta = np.linalg.lstsq(K_red, u_red, rcond=None)[0]
        delta_expected = sim.B @ Delta

        print(f"Control u_red: {u_red}")
        print(f"Delta: {Delta}")
        print(f"delta_expected: {delta_expected}")
        print(f"delta_computed: {delta_computed}")
        print(f"Keq_old: {Keq_old}, Keq_new: {Keq_new}")

        np.testing.assert_allclose(
            delta_computed, delta_expected[0], rtol=1e-10, err_msg="Control mapping should be mathematically consistent"
        )

    def test_steady_state_control(self):
        """Test that constant control drives to predicted equilibrium."""
        sim = MassActionSimulator(
            self.network,
            self.rate_constants,
            B=self.solver._B,
            K_red=self.solver._B.T @ self.solver.dynamics.K @ self.solver._B,
        )

        # Target: shift Keq by factor of 2
        target_ln_shift = np.log(2.0)

        # Compute required control
        K_red = sim.K_red
        Delta = target_ln_shift  # For single reaction, delta = Delta
        u_red_required = K_red @ np.array([Delta])

        print(f"Target ln shift: {target_ln_shift}")
        print(f"Required u_red: {u_red_required}")
        print(f"K_red: {K_red}")

        # Control function (constant control)
        def control_func(t, Q):
            return u_red_required

        # Simulate
        t_eval = np.linspace(0, 10, 1000)
        result = sim.simulate(t_eval, control_func)

        # Check final equilibrium
        Q_final = result["reaction_quotients"][-1]
        expected_Q = (self.kf / self.kr) * np.exp(target_ln_shift)

        print(f"Final Q: {Q_final[0]}, Expected Q: {expected_Q}")

        np.testing.assert_allclose(
            Q_final[0], expected_Q, rtol=5e-2, err_msg="Constant control should drive to predicted equilibrium"
        )


class TestThreeCycleValidation:
    """Test validation functions on 3-cycle network."""

    def setup_method(self):
        """Set up 3-cycle network."""
        self.species_ids = ["A", "B", "C"]
        self.reaction_ids = ["R1", "R2", "R3"]
        self.S = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]])

        self.species_info = {
            "A": {"initial_concentration": 2.0},
            "B": {"initial_concentration": 0.2},
            "C": {"initial_concentration": 0.1},
        }

        self.reaction_info = [{"id": rid} for rid in self.reaction_ids]

        self.network = ReactionNetwork(self.species_ids, self.reaction_ids, self.S, self.species_info, self.reaction_info)

        # Thermodynamically consistent rate constants for cycle
        # Must satisfy: Keq1 × Keq2 × Keq3 = 1 for detailed balance
        # Choose: Keq1 = 2.0, Keq2 = 0.5, then Keq3 = 1.0 to satisfy constraint
        self.rate_constants = {
            "R1": (3.0, 1.5),  # Keq = 2.0
            "R2": (1.0, 2.0),  # Keq = 0.5
            "R3": (3.0, 3.0),  # Keq = 1.0 (2.0 × 0.5 × 1.0 = 1.0 ✓)
        }

        # Extract rate constants and compute proper K matrix
        kf = np.array([self.rate_constants[rid][0] for rid in self.reaction_ids])
        kr = np.array([self.rate_constants[rid][1] for rid in self.reaction_ids])
        Keq = kf / kr

        # Proper relaxation matrix (diagonal approximation)
        K = np.diag(kf + kr)

        self.dynamics = LLRQDynamics(self.network, Keq, K)
        self.solver = LLRQSolver(self.dynamics)

    def test_control_mapping_validation(self):
        """Test the control mapping validation function."""
        # Get LLRQ matrices
        B = self.solver._B
        K_red = B.T @ self.solver.dynamics.K @ B

        # Base rate constants
        kf_base = np.array([self.rate_constants[rid][0] for rid in self.reaction_ids])
        kr_base = np.array([self.rate_constants[rid][1] for rid in self.reaction_ids])

        # Test control
        u_red = np.array([0.2, -0.1])

        # Validate mapping
        validation = validate_llrq_control_mapping(B, K_red, kf_base, kr_base, u_red)

        print("Validation results:")
        for key, value in validation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        assert validation["consistent"], "Control mapping should be mathematically consistent"
        assert validation["max_error"] < 1e-10, f"Max error too large: {validation['max_error']}"

    def test_equilibrium_consistency(self):
        """Test that mass action equilibrium matches LLRQ Keq."""
        sim = MassActionSimulator(
            self.network,
            self.rate_constants,
            B=self.solver._B,
            K_red=self.solver._B.T @ self.solver.dynamics.K @ self.solver._B,
        )

        # Simulate to equilibrium
        t_eval = np.linspace(0, 20, 2000)
        result = sim.simulate(t_eval)

        # Final Q values
        Q_final = result["reaction_quotients"][-1]

        # Expected Keq from rate constants
        expected_Keq = np.array(
            [
                self.rate_constants["R1"][0] / self.rate_constants["R1"][1],
                self.rate_constants["R2"][0] / self.rate_constants["R2"][1],
                self.rate_constants["R3"][0] / self.rate_constants["R3"][1],
            ]
        )

        print(f"Final Q: {Q_final}")
        print(f"Expected Keq: {expected_Keq}")
        print(f"LLRQ consistent Keq: {self.solver._Keq_consistent}")

        # Check that they match within reasonable tolerance
        np.testing.assert_allclose(
            Q_final, expected_Keq, rtol=1e-2, err_msg="Mass action equilibrium should match rate constant ratios"
        )


def test_mass_action_drive_functions():
    """Test the mass action drive helper functions."""
    # Simple test case
    kf_base = np.array([2.0, 1.0, 1.5])
    kr_base = np.array([1.0, 2.0, 0.5])

    # Mock LLRQ matrices
    B = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])  # Simple example
    K_red = np.array([[3.0, 0.1], [0.1, 3.0]])

    u_red = np.array([0.3, -0.2])

    # Test control application
    kf_new, kr_new = apply_llrq_drive_to_rates(kf_base, kr_base, B, K_red, u_red)

    # Test equilibrium shift computation
    delta = compute_equilibrium_shift(kf_base, kr_base, kf_new, kr_new)

    print(f"Base rates: kf={kf_base}, kr={kr_base}")
    print(f"New rates: kf={kf_new}, kr={kr_new}")
    print(f"Equilibrium shift: {delta}")

    # Basic sanity checks
    assert np.all(kf_new > 0), "Forward rates should be positive"
    assert np.all(kr_new > 0), "Reverse rates should be positive"
    assert len(delta) == len(kf_base), "Delta should match number of reactions"


if __name__ == "__main__":
    # Run specific tests for debugging
    print("Testing single reaction control...")
    test = TestSingleReaction()
    test.setup_method()
    test.test_equilibrium_without_control()
    test.test_control_mapping_consistency()
    test.test_steady_state_control()

    print("\nTesting three-cycle validation...")
    test3 = TestThreeCycleValidation()
    test3.setup_method()
    test3.test_control_mapping_validation()
    test3.test_equilibrium_consistency()

    print("\nTesting mass action drive functions...")
    test_mass_action_drive_functions()

    print("\nAll tests completed!")
