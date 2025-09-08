"""
Tests for entropy production and thermodynamic accounting functionality.

Tests the new ThermodynamicAccountant class and entropy production calculations
from both reaction forces (x) and external drives (u).
"""

import os
import sys

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.thermodynamic_accounting import ThermodynamicAccountant, AccountingResult, DualAccountingResult
from llrq.reaction_network import ReactionNetwork
from llrq.llrq_dynamics import LLRQDynamics
from llrq.solver import LLRQSolver


class TestAccountingDataClasses:
    """Test the accounting result dataclasses."""

    def test_accounting_result_creation(self):
        """Test AccountingResult creation and access."""
        sigma_time = np.array([1.0, 2.0, 3.0])
        sigma_total = 6.0
        result = AccountingResult(sigma_time, sigma_total, "test note")

        np.testing.assert_array_equal(result.sigma_time, sigma_time)
        assert result.sigma_total == sigma_total
        assert result.notes == "test note"

    def test_dual_accounting_result_creation(self):
        """Test DualAccountingResult creation and access."""
        result_x = AccountingResult(np.array([1.0, 2.0]), 3.0)
        result_u = AccountingResult(np.array([0.5, 1.0]), 1.5, "quasi-steady")
        balance = {"residual_total": 0.1}

        dual_result = DualAccountingResult(result_x, result_u, balance)

        assert dual_result.from_x == result_x
        assert dual_result.from_u == result_u
        assert dual_result.balance == balance


class TestThermodynamicAccountant:
    """Test the ThermodynamicAccountant class."""

    def test_init_with_network(self):
        """Test initialization with just network."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        assert accountant.network == network
        assert accountant.L is None
        assert accountant.n_reactions == 1

    def test_init_with_onsager_conductance(self):
        """Test initialization with pre-computed Onsager conductance."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[2.0]])
        accountant = ThermodynamicAccountant(network, L)

        assert accountant.network == network
        np.testing.assert_array_equal(accountant.L, L)

    def test_quad_time_series_helper(self):
        """Test the _quad_time_series helper function."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        # Simple test case
        t = np.array([0.0, 1.0, 2.0])
        Y = np.array([[1.0], [2.0], [1.0]])  # (3, 1)
        M = np.array([[1.0]])  # (1, 1)

        q_t, integral = accountant._quad_time_series(t, Y, M)

        # q(t) = Y @ M @ Y^T = [1, 4, 1] for this case
        expected_q = np.array([1.0, 4.0, 1.0])
        np.testing.assert_array_equal(q_t, expected_q)

        # Integral should be positive
        assert integral > 0

    def test_sym_psd_helper(self):
        """Test the _sym_psd symmetrization and PSD clipping."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        # Test symmetric matrix (no change)
        M_sym = np.array([[2.0, 1.0], [1.0, 3.0]])
        result = accountant._sym_psd(M_sym)
        np.testing.assert_allclose(result, M_sym)

        # Test asymmetric matrix (gets symmetrized)
        M_asym = np.array([[2.0, 0.5], [1.5, 3.0]])
        result = accountant._sym_psd(M_asym)
        expected = 0.5 * (M_asym + M_asym.T)
        np.testing.assert_allclose(result, expected)

        # Test PSD clipping
        M_neg = np.array([[1.0, 2.0], [2.0, 1.0]])  # Has negative eigenvalue
        result = accountant._sym_psd(M_neg, eps=0.1)
        eigenvals = np.linalg.eigvals(result)
        assert np.all(eigenvals >= 0.1 - 1e-10)


class TestEntropyFromX:
    """Test entropy production from reaction forces x(t)."""

    def test_simple_entropy_from_x(self):
        """Test entropy computation from reaction forces."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[1.0]])
        accountant = ThermodynamicAccountant(network, L)

        # Simple test trajectory
        t = np.linspace(0, 2, 100)
        x = np.sin(t).reshape(-1, 1)  # (100, 1)

        result = accountant.entropy_from_x(t, x, scale=2.0)

        assert isinstance(result, AccountingResult)
        assert len(result.sigma_time) == len(t)
        assert result.sigma_total > 0  # Should be positive for this case
        assert result.notes == ""

        # Check scaling
        result_scaled = accountant.entropy_from_x(t, x, scale=4.0)
        np.testing.assert_allclose(result_scaled.sigma_time, 2 * result.sigma_time)
        assert abs(result_scaled.sigma_total - 2 * result.sigma_total) < 1e-10

    def test_entropy_from_x_no_conductance(self):
        """Test that entropy_from_x fails gracefully without L matrix."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)  # No L matrix

        t = np.array([0.0, 1.0])
        x = np.array([[0.1], [0.2]])

        with pytest.raises(ValueError, match="No Onsager conductance matrix"):
            accountant.entropy_from_x(t, x)

    def test_entropy_from_x_with_psd_clipping(self):
        """Test entropy computation with PSD clipping."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Conductance matrix with small negative eigenvalue
        L = np.array([[1.0]])  # This is fine, but test the clipping logic
        accountant = ThermodynamicAccountant(network, L)

        t = np.array([0.0, 1.0, 2.0])
        x = np.array([[0.1], [0.2], [0.15]])

        result = accountant.entropy_from_x(t, x, psd_clip=0.01)

        assert isinstance(result, AccountingResult)
        assert len(result.sigma_time) == 3
        assert np.all(np.isfinite(result.sigma_time))


class TestEntropyFromU:
    """Test quasi-steady entropy production from external drives u(t)."""

    def test_simple_entropy_from_u(self):
        """Test entropy computation from external drives."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[2.0]])
        K = np.array([[1.0]])
        accountant = ThermodynamicAccountant(network, L)

        # Simple drive trajectory
        t = np.linspace(0, 2, 50)
        u = 0.5 * np.ones((50, 1))  # Constant drive

        result = accountant.entropy_from_u(t, u, K)

        assert isinstance(result, AccountingResult)
        assert len(result.sigma_time) == len(t)
        assert result.sigma_total > 0
        assert "quasi-steady" in result.notes

    def test_entropy_from_u_matrix_inversion(self):
        """Test that K matrix inversion is handled properly."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], [[-1, 0], [1, -1], [0, 1]])

        L = np.eye(2) * 2.0
        K = np.array([[2.0, 0.5], [0.5, 1.0]])  # Well-conditioned
        accountant = ThermodynamicAccountant(network, L)

        t = np.linspace(0, 1, 20)
        u = np.random.rand(20, 2) * 0.1  # Small random drives

        result = accountant.entropy_from_u(t, u, K)

        assert isinstance(result, AccountingResult)
        assert np.all(np.isfinite(result.sigma_time))
        assert np.isfinite(result.sigma_total)

    def test_entropy_from_u_no_conductance(self):
        """Test that entropy_from_u fails without L matrix."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        t = np.array([0.0, 1.0])
        u = np.array([[0.1], [0.2]])
        K = np.array([[1.0]])

        with pytest.raises(ValueError, match="No Onsager conductance matrix"):
            accountant.entropy_from_u(t, u, K)


class TestDualAccounting:
    """Test dual entropy accounting (from x and u) with energy balance."""

    def test_entropy_from_xu_basic(self):
        """Test dual entropy accounting."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[1.5]])
        K = np.array([[2.0]])
        accountant = ThermodynamicAccountant(network, L)

        # Consistent trajectories
        t = np.linspace(0, 2, 50)
        x = np.exp(-2 * t).reshape(-1, 1) * 0.1  # Decaying reaction force
        u = np.zeros((50, 1))  # No external drive

        result = accountant.entropy_from_xu(t, x, u, K)

        assert isinstance(result, DualAccountingResult)
        assert isinstance(result.from_x, AccountingResult)
        assert isinstance(result.from_u, AccountingResult)
        assert isinstance(result.balance, dict)

        # Check balance keys
        expected_keys = [
            "V_dot_time",
            "V_dot_total",
            "P_relax_time",
            "P_relax_total",
            "P_ctrl_time",
            "P_ctrl_total",
            "residual_time",
            "residual_total",
            "comment",
        ]
        for key in expected_keys:
            assert key in result.balance

    def test_entropy_from_xu_energy_balance(self):
        """Test energy balance diagnostic."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[1.0]])
        K = np.array([[1.0]])
        accountant = ThermodynamicAccountant(network, L)

        # Well-behaved trajectories that should satisfy the balance
        t = np.linspace(0, 3, 100)
        x = np.exp(-t).reshape(-1, 1) * 0.2  # Exponential decay
        u = np.zeros((100, 1))  # No control

        result = accountant.entropy_from_xu(t, x, u, K)

        # Energy balance should be reasonably satisfied
        # dV/dt + P_relax - P_ctrl ≈ 0
        residual = result.balance["residual_total"]
        V_change = result.balance["V_dot_total"]

        # For this simple case with u=0, residual should be small
        assert abs(residual) < 0.1  # Allow some numerical error

    def test_entropy_from_xu_with_control(self):
        """Test dual accounting with non-zero control inputs."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[2.0]])
        K = np.array([[1.0]])
        accountant = ThermodynamicAccountant(network, L)

        t = np.linspace(0, 2, 60)
        x = 0.1 * np.sin(t).reshape(-1, 1)
        u = 0.05 * np.cos(t).reshape(-1, 1)  # Oscillating control

        result = accountant.entropy_from_xu(t, x, u, K)

        # Both entropy estimates should be finite
        assert np.all(np.isfinite(result.from_x.sigma_time))
        assert np.all(np.isfinite(result.from_u.sigma_time))
        assert np.isfinite(result.from_x.sigma_total)
        assert np.isfinite(result.from_u.sigma_total)

        # Control power should be non-zero
        assert abs(result.balance["P_ctrl_total"]) > 1e-6


class TestFromSolutionIntegration:
    """Test integration with LLRQSolver results."""

    def test_from_solution_basic(self):
        """Test processing solver results for entropy."""
        # Create simple A <-> B system
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Create dynamics from mass action
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])
        initial_concentrations = np.array([1.0, 0.5])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

        solver = LLRQSolver(dynamics)

        # Solve dynamics
        solution = solver.solve(initial_conditions={"A": 1.0, "B": 0.5}, t_span=(0, 2), method="analytical")

        # Create accountant and process solution
        accountant = ThermodynamicAccountant(network)

        result = accountant.from_solution(solution, forward_rates=forward_rates, backward_rates=backward_rates, scale=1.0)

        assert isinstance(result, AccountingResult)
        assert len(result.sigma_time) == len(solution["time"])
        assert np.all(np.isfinite(result.sigma_time))
        assert np.isfinite(result.sigma_total)

    def test_from_solution_missing_data(self):
        """Test error handling for incomplete solution data."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        # Missing required fields
        incomplete_solution = {"time": np.array([0, 1, 2])}

        with pytest.raises(ValueError, match="log_deviations"):
            accountant.from_solution(incomplete_solution)

    def test_from_solution_no_rates(self):
        """Test error handling when rates not provided."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        solution = {
            "time": np.array([0, 1, 2]),
            "log_deviations": np.array([[0.1], [0.0], [-0.1]]),
            "initial_concentrations": np.array([1.0, 1.0]),
        }

        with pytest.raises(ValueError, match="forward_rates"):
            accountant.from_solution(solution)


class TestOnsagerIntegration:
    """Test integration with existing Onsager conductance methods."""

    def test_compute_and_cache_onsager(self):
        """Test computing and caching Onsager conductance."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        accountant = ThermodynamicAccountant(network)

        concentrations = np.array([1.0, 2.0])
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])

        # Should compute and cache L
        L = accountant.compute_onsager_conductance(concentrations, forward_rates, backward_rates)

        assert accountant.L is not None
        np.testing.assert_array_equal(accountant.L, L)
        assert L.shape == (1, 1)
        assert L[0, 0] > 0

    def test_entropy_with_precomputed_onsager(self):
        """Test entropy computation with pre-computed Onsager conductance."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Pre-compute Onsager conductance
        concentrations = np.array([1.0, 1.0])
        forward_rates = np.array([1.0])
        backward_rates = np.array([1.0])

        onsager_result = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates)
        L = onsager_result["L"]

        # Create accountant with pre-computed L
        accountant = ThermodynamicAccountant(network, L)

        # Test entropy computation
        t = np.array([0, 0.5, 1.0])
        x = np.array([[0.1], [0.0], [-0.1]])

        result = accountant.entropy_from_x(t, x)

        assert isinstance(result, AccountingResult)
        assert len(result.sigma_time) == 3
        assert np.all(np.isfinite(result.sigma_time))


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_small_values_stability(self):
        """Test behavior with very small values."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[1e-3]])  # Small conductance
        accountant = ThermodynamicAccountant(network, L)

        t = np.array([0, 1, 2])
        x = np.array([[1e-6], [1e-7], [1e-8]])  # Very small reaction forces

        result = accountant.entropy_from_x(t, x)

        assert isinstance(result, AccountingResult)
        assert np.all(np.isfinite(result.sigma_time))
        assert np.isfinite(result.sigma_total)

    def test_large_values_stability(self):
        """Test behavior with large values."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[1e3]])  # Large conductance
        accountant = ThermodynamicAccountant(network, L)

        t = np.linspace(0, 1, 10)
        x = np.ones((10, 1)) * 10  # Large reaction forces

        result = accountant.entropy_from_x(t, x)

        assert isinstance(result, AccountingResult)
        assert np.all(np.isfinite(result.sigma_time))
        assert np.isfinite(result.sigma_total)
        assert result.sigma_total > 0

    def test_zero_trajectories(self):
        """Test behavior with zero trajectories."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        L = np.array([[1.0]])
        K = np.array([[1.0]])
        accountant = ThermodynamicAccountant(network, L)

        t = np.array([0, 0.5, 1.0])
        x = np.zeros((3, 1))
        u = np.zeros((3, 1))

        result = accountant.entropy_from_xu(t, x, u, K)

        assert isinstance(result, DualAccountingResult)
        assert np.allclose(result.from_x.sigma_time, 0)
        assert np.allclose(result.from_u.sigma_time, 0)
        assert abs(result.from_x.sigma_total) < 1e-12
        assert abs(result.from_u.sigma_total) < 1e-12
