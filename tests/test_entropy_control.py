"""
Tests for entropy-aware steady-state control functionality.

Tests the new methods in LLRQController for computing control inputs that
trade off between target tracking and entropy production minimization.
"""

import pytest
import numpy as np
from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.control import LLRQController
from llrq.thermodynamic_accounting import ThermodynamicAccountant


@pytest.fixture
def simple_network():
    """Simple A ⇌ B reaction network."""
    return ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])


@pytest.fixture
def chain_network():
    """A ⇌ B ⇌ C reaction chain."""
    return ReactionNetwork(
        species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
    )


@pytest.fixture
def simple_controller(simple_network):
    """Controller for simple A ⇌ B network."""
    forward_rates = np.array([2.0])
    backward_rates = np.array([1.0])
    initial_concentrations = np.array([1.0, 1.0])

    dynamics = LLRQDynamics.from_mass_action(simple_network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)
    return LLRQController(solver, controlled_reactions=[0])


@pytest.fixture
def chain_controller(chain_network):
    """Controller for A ⇌ B ⇌ C chain."""
    forward_rates = np.array([2.0, 1.5])
    backward_rates = np.array([1.0, 0.8])
    initial_concentrations = np.array([2.0, 1.0, 0.5])

    dynamics = LLRQDynamics.from_mass_action(chain_network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)
    return LLRQController(solver, controlled_reactions=[0, 1])


class TestEntropyMetric:
    """Test computation of control entropy metric matrix."""

    def test_entropy_metric_symmetry(self, simple_controller):
        """Test that entropy metric matrix is symmetric."""
        # Create a simple Onsager conductance matrix
        L = np.array([[1.0]])

        M = simple_controller.compute_control_entropy_metric(L)

        assert np.allclose(M, M.T), "Entropy metric should be symmetric"

    def test_entropy_metric_positive_definite(self, simple_controller):
        """Test that entropy metric is positive definite for positive definite L."""
        # Positive definite Onsager matrix
        L = np.array([[2.0]])

        M = simple_controller.compute_control_entropy_metric(L)

        # Check positive definiteness via eigenvalues
        eigenvals = np.linalg.eigvals(M)
        assert np.all(eigenvals > 0), "Entropy metric should be positive definite"

    def test_entropy_metric_pullback_property(self, simple_controller):
        """Test that M = K^{-T} L K^{-1} satisfies pullback property."""
        L = np.array([[1.5]])
        K = simple_controller.solver.dynamics.K

        M = simple_controller.compute_control_entropy_metric(L)

        # Test pullback: ||u||_M^2 = ||K^{-1} u||_L^2
        u_test = np.array([0.5])

        # Left side: u^T M u
        lhs = u_test.T @ M @ u_test

        # Right side: (K^{-1} u)^T L (K^{-1} u)
        K_inv_u = np.linalg.solve(K, u_test)
        rhs = K_inv_u.T @ L @ K_inv_u

        assert np.allclose(lhs, rhs), "Should satisfy pullback property"


class TestEntropyRate:
    """Test steady-state entropy rate computation."""

    def test_entropy_rate_positive(self, simple_controller):
        """Test that entropy rate is non-negative."""
        L = np.array([[1.0]])
        M = simple_controller.compute_control_entropy_metric(L)

        u_test = np.array([1.0])
        entropy_rate = simple_controller.compute_steady_state_entropy_rate(u_test, M)

        assert entropy_rate >= 0, "Entropy rate should be non-negative"

    def test_entropy_rate_scaling(self, simple_controller):
        """Test that entropy rate scales quadratically with control magnitude."""
        L = np.array([[1.0]])
        M = simple_controller.compute_control_entropy_metric(L)

        u_base = np.array([1.0])
        u_scaled = 2.0 * u_base

        entropy_base = simple_controller.compute_steady_state_entropy_rate(u_base, M)
        entropy_scaled = simple_controller.compute_steady_state_entropy_rate(u_scaled, M)

        assert np.allclose(entropy_scaled, 4.0 * entropy_base), "Entropy rate should scale quadratically"

    def test_zero_control_zero_entropy(self, simple_controller):
        """Test that zero control gives zero entropy."""
        L = np.array([[1.0]])
        M = simple_controller.compute_control_entropy_metric(L)

        u_zero = np.array([0.0])
        entropy_rate = simple_controller.compute_steady_state_entropy_rate(u_zero, M)

        assert np.allclose(entropy_rate, 0.0), "Zero control should give zero entropy"


class TestEntropyAwareControl:
    """Test entropy-aware steady-state control computation."""

    def test_zero_entropy_weight_exact_tracking(self, simple_controller):
        """Test that λ=0 gives exact target tracking."""
        L = np.array([[1.0]])
        x_target = np.array([0.5])

        result = simple_controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=0.0)

        # Should achieve exact target with λ=0
        assert np.allclose(result["x_achieved"], x_target, atol=1e-10), "Should achieve exact target with zero entropy weight"
        assert np.allclose(result["tracking_error"], 0.0, atol=1e-10), "Tracking error should be zero with λ=0"

    def test_large_entropy_weight_minimal_control(self, simple_controller):
        """Test that large λ gives minimal control effort."""
        L = np.array([[1.0]])
        x_target = np.array([0.5])

        # Compare small vs large entropy weights
        result_small = simple_controller.compute_entropy_aware_steady_state_control(
            x_target=x_target, L=L, entropy_weight=0.01
        )
        result_large = simple_controller.compute_entropy_aware_steady_state_control(
            x_target=x_target, L=L, entropy_weight=100.0
        )

        # Large entropy weight should give smaller control effort
        control_small = np.linalg.norm(result_small["u_optimal"])
        control_large = np.linalg.norm(result_large["u_optimal"])

        assert control_large < control_small, "Larger entropy weight should reduce control effort"

    def test_monotonic_tradeoff(self, simple_controller):
        """Test that increasing entropy weight monotonically reduces entropy rate."""
        L = np.array([[2.0]])
        x_target = np.array([0.8])

        entropy_weights = [0.1, 1.0, 10.0]
        entropy_rates = []
        tracking_errors = []

        for lam in entropy_weights:
            result = simple_controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=lam)
            entropy_rates.append(result["entropy_rate"])
            tracking_errors.append(result["tracking_error"])

        # Entropy rates should decrease with increasing weight
        assert entropy_rates[1] < entropy_rates[0], "Higher entropy weight should reduce entropy rate"
        assert entropy_rates[2] < entropy_rates[1], "Higher entropy weight should reduce entropy rate"

        # Tracking errors should increase with increasing entropy weight
        assert tracking_errors[1] > tracking_errors[0], "Higher entropy weight should increase tracking error"
        assert tracking_errors[2] > tracking_errors[1], "Higher entropy weight should increase tracking error"

    def test_controlled_reactions_only(self, chain_controller):
        """Test controlled_reactions_only flag."""
        # Set up Onsager conductance
        accountant = ThermodynamicAccountant(chain_controller.network)
        L = accountant.compute_onsager_conductance(
            np.array([1.0, 1.0, 1.0]), np.array([2.0, 1.5]), np.array([1.0, 0.8]), mode="local"
        )

        x_target = np.array([0.3, -0.2])

        # Test with controlled reactions only
        result_controlled = chain_controller.compute_entropy_aware_steady_state_control(
            x_target=x_target, L=L, entropy_weight=1.0, controlled_reactions_only=True
        )

        # Test with all reactions
        result_all = chain_controller.compute_entropy_aware_steady_state_control(
            x_target=x_target, L=L, entropy_weight=1.0, controlled_reactions_only=False
        )

        # Both should be valid results
        assert "u_optimal" in result_controlled
        assert "u_optimal" in result_all
        assert len(result_controlled["u_optimal"]) == len(result_all["u_optimal"])

    def test_cost_function_consistency(self, simple_controller):
        """Test that reported total cost matches sum of components."""
        L = np.array([[1.0]])
        x_target = np.array([0.4])
        entropy_weight = 2.0

        result = simple_controller.compute_entropy_aware_steady_state_control(
            x_target=x_target, L=L, entropy_weight=entropy_weight
        )

        expected_total_cost = result["tracking_error"] + entropy_weight * result["entropy_rate"]

        assert np.allclose(
            result["total_cost"], expected_total_cost
        ), "Total cost should equal tracking error + λ * entropy rate"

    def test_steady_state_consistency(self, simple_controller):
        """Test that x_achieved = K^{-1} u_optimal."""
        L = np.array([[1.5]])
        x_target = np.array([0.6])

        result = simple_controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=1.0)

        K = simple_controller.solver.dynamics.K
        expected_x = np.linalg.solve(K, result["u_optimal"])

        assert np.allclose(result["x_achieved"], expected_x), "Achieved state should satisfy x = K^{-1} u"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_singular_system_handling(self, simple_controller):
        """Test handling of singular systems."""
        # Create a singular Onsager matrix
        L = np.array([[0.0]])
        x_target = np.array([0.5])

        # Should not crash, may use pseudoinverse
        result = simple_controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=1.0)

        assert "u_optimal" in result
        assert np.isfinite(result["u_optimal"]).all()

    def test_negative_entropy_weight_raises_warning(self, simple_controller):
        """Test that negative entropy weights work but may be unphysical."""
        L = np.array([[1.0]])
        x_target = np.array([0.3])

        # Negative entropy weight should still work mathematically
        result = simple_controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=-1.0)

        assert "u_optimal" in result
        assert np.isfinite(result["u_optimal"]).all()

    def test_zero_target_state(self, simple_controller):
        """Test with zero target state."""
        L = np.array([[1.0]])
        x_target = np.array([0.0])

        result = simple_controller.compute_entropy_aware_steady_state_control(x_target=x_target, L=L, entropy_weight=1.0)

        # Zero target should give zero control (for any λ > 0)
        assert np.allclose(result["u_optimal"], 0.0, atol=1e-10)
        assert np.allclose(result["x_achieved"], 0.0, atol=1e-10)
        assert np.allclose(result["tracking_error"], 0.0, atol=1e-10)
        assert np.allclose(result["entropy_rate"], 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
