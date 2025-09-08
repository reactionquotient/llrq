"""
Comprehensive tests for LLRQDynamics class.

Tests log-linear reaction quotient dynamics including log deviations,
dynamics function, analytical solutions, and external drives.
"""

import os
import sys
import warnings

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.llrq_dynamics import LLRQDynamics
from llrq.reaction_network import ReactionNetwork


class TestLLRQDynamicsInitialization:
    """Test LLRQDynamics initialization and setup."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        dynamics = LLRQDynamics(network)

        assert dynamics.network == network
        assert dynamics.n_reactions == 1
        assert np.array_equal(dynamics.Keq, np.ones(1))  # Default Keq = 1
        assert np.array_equal(dynamics.K, np.eye(1))  # Default K = I

    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([2.0])
        K = np.array([[1.5]])

        dynamics = LLRQDynamics(network, Keq, K)

        assert np.array_equal(dynamics.Keq, Keq)
        assert np.array_equal(dynamics.K, K)

    def test_initialization_with_external_drive(self):
        """Test initialization with external drive function."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        def drive_func(t):
            return np.array([np.sin(t)])

        dynamics = LLRQDynamics(network, external_drive=drive_func)

        assert dynamics.external_drive == drive_func
        test_val = dynamics.external_drive(0.0)
        assert np.isclose(test_val[0], 0.0)

    def test_invalid_keq_length(self):
        """Test error for wrong number of equilibrium constants."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([1.0, 2.0])  # Wrong length!

        with pytest.raises(ValueError, match="Expected 1 equilibrium constants"):
            LLRQDynamics(network, Keq)

    def test_invalid_k_shape(self):
        """Test error for wrong relaxation matrix shape."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0, 2.0]])  # Wrong shape!

        with pytest.raises(ValueError, match="Expected relaxation matrix shape"):
            LLRQDynamics(network, relaxation_matrix=K)

    def test_warnings_for_defaults(self):
        """Test that warnings are issued for default values."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLRQDynamics(network)

            # Should have warnings about defaults
            assert len(w) >= 2
            assert any("equilibrium constants" in str(warning.message) for warning in w)
            assert any("relaxation matrix" in str(warning.message) for warning in w)


class TestLogDeviations:
    """Test log deviation computations."""

    def test_log_deviation_basic(self):
        """Test basic log deviation calculation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([2.0])

        dynamics = LLRQDynamics(network, Keq)

        Q = np.array([4.0])
        x = dynamics.compute_log_deviation(Q)

        # x = ln(Q/Keq) = ln(4.0/2.0) = ln(2)
        expected = np.array([np.log(2.0)])
        assert np.allclose(x, expected)

    def test_log_deviation_equilibrium(self):
        """Test log deviation at equilibrium."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([2.0])

        dynamics = LLRQDynamics(network, Keq)

        Q = Keq.copy()  # At equilibrium
        x = dynamics.compute_log_deviation(Q)

        # Should be zero at equilibrium
        assert np.allclose(x, np.zeros(1))

    def test_log_deviation_multiple_reactions(self):
        """Test log deviations for multiple reactions."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        Keq = np.array([2.0, 0.5])

        dynamics = LLRQDynamics(network, Keq)

        Q = np.array([4.0, 1.0])
        x = dynamics.compute_log_deviation(Q)

        expected = np.array([np.log(4.0 / 2.0), np.log(1.0 / 0.5)])
        assert np.allclose(x, expected)

    def test_log_deviation_zero_handling(self):
        """Test handling of zero reaction quotients."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([2.0])

        dynamics = LLRQDynamics(network, Keq)

        Q = np.array([0.0])  # Zero quotient
        x = dynamics.compute_log_deviation(Q)

        # Should not be NaN or -Inf
        assert np.isfinite(x).all()

    def test_compute_reaction_quotients_from_deviations(self):
        """Test converting log deviations back to reaction quotients."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([2.0])

        dynamics = LLRQDynamics(network, Keq)

        x = np.array([np.log(3.0)])  # ln(3)
        Q = dynamics.compute_reaction_quotients(x)

        # Q = Keq * exp(x) = 2.0 * exp(ln(3)) = 2.0 * 3 = 6.0
        expected = np.array([6.0])
        assert np.allclose(Q, expected)

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion Q -> x -> Q."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        Keq = np.array([2.0])

        dynamics = LLRQDynamics(network, Keq)

        Q_orig = np.array([4.0])
        x = dynamics.compute_log_deviation(Q_orig)
        Q_recovered = dynamics.compute_reaction_quotients(x)

        assert np.allclose(Q_orig, Q_recovered)

    def test_invalid_quotient_count(self):
        """Test error for wrong number of reaction quotients."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)

        with pytest.raises(ValueError, match="Expected 1 reaction quotients"):
            dynamics.compute_log_deviation(np.array([1.0, 2.0]))


class TestDynamicsFunction:
    """Test the dynamics function dx/dt = -K*x + u(t)."""

    def test_dynamics_zero_external_drive(self):
        """Test dynamics with zero external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.5]])

        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x = np.array([2.0])
        dxdt = dynamics.dynamics(0.0, x)

        # dx/dt = -K*x = -1.5 * 2.0 = -3.0
        expected = np.array([-3.0])
        assert np.allclose(dxdt, expected)

    def test_dynamics_with_external_drive(self):
        """Test dynamics with non-zero external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])

        def drive_func(t):
            return np.array([0.5])  # Constant drive

        dynamics = LLRQDynamics(network, relaxation_matrix=K, external_drive=drive_func)

        x = np.array([1.0])
        dxdt = dynamics.dynamics(0.0, x)

        # dx/dt = -K*x + u = -1.0*1.0 + 0.5 = -0.5
        expected = np.array([-0.5])
        assert np.allclose(dxdt, expected)

    def test_dynamics_multiple_reactions(self):
        """Test dynamics for multiple reactions."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[2.0, -0.5], [-0.5, 1.0]])

        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x = np.array([1.0, -0.5])
        dxdt = dynamics.dynamics(0.0, x)

        # dx/dt = -K*x
        expected = -K @ x
        assert np.allclose(dxdt, expected)

    def test_dynamics_time_varying_drive(self):
        """Test dynamics with time-varying external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])

        def drive_func(t):
            return np.array([np.sin(t)])

        dynamics = LLRQDynamics(network, relaxation_matrix=K, external_drive=drive_func)

        x = np.array([0.0])

        # At t=0: u(0) = sin(0) = 0
        dxdt_0 = dynamics.dynamics(0.0, x)
        assert np.allclose(dxdt_0, np.array([0.0]))

        # At t=π/2: u(π/2) = sin(π/2) = 1
        dxdt_pi2 = dynamics.dynamics(np.pi / 2, x)
        assert np.allclose(dxdt_pi2, np.array([1.0]))

    def test_dynamics_equilibrium_point(self):
        """Test dynamics at equilibrium point."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])

        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x_eq = np.array([0.0])  # Equilibrium: ln(Q/Keq) = 0
        dxdt = dynamics.dynamics(0.0, x_eq)

        # Should be zero at equilibrium
        assert np.allclose(dxdt, np.array([0.0]))

    def test_invalid_state_dimension(self):
        """Test error for wrong state vector dimension."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)

        with pytest.raises(ValueError, match="Expected 1 state variables"):
            dynamics.dynamics(0.0, np.array([1.0, 2.0]))


class TestAnalyticalSolution:
    """Test analytical solution methods."""

    def test_analytical_solution_constant_drive(self):
        """Test analytical solution with constant external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[2.0]])
        u0 = 0.5

        def drive_func(t):
            return np.array([u0])

        dynamics = LLRQDynamics(network, relaxation_matrix=K, external_drive=drive_func)

        x0 = np.array([1.0])
        t = np.array([0.0, 1.0, 2.0])

        x_t = dynamics.analytical_solution(x0, t)

        assert x_t.shape == (3, 1)
        assert np.allclose(x_t[0], x0)  # Initial condition

    def test_analytical_solution_no_drive(self):
        """Test analytical solution with no external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])

        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x0 = np.array([2.0])
        t = np.array([0.0, 1.0, 2.0])

        x_t = dynamics.analytical_solution(x0, t)

        # Should decay exponentially: x(t) = x0 * exp(-K*t)
        expected = x0 * np.exp(-K[0, 0] * t).reshape(-1, 1)
        assert np.allclose(x_t, expected)

    def test_analytical_solution_multiple_reactions(self):
        """Test analytical solution for multiple reactions."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[2.0, 0.0], [0.0, 1.0]])  # Diagonal for simplicity

        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x0 = np.array([1.0, -0.5])
        t = np.array([0.0, 1.0])

        x_t = dynamics.analytical_solution(x0, t)

        assert x_t.shape == (2, 2)
        assert np.allclose(x_t[0], x0)

    def test_analytical_solution_singular_matrix(self):
        """Test analytical solution with singular K matrix."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[0.0]])  # Singular!

        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x0 = np.array([1.0])
        t = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_t = dynamics.analytical_solution(x0, t)

            # Should warn about singular matrix
            assert any("singular" in str(warning.message) for warning in w)

    def test_analytical_solution_time_varying_warning(self):
        """Test warning for time-varying external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])

        def drive_func(t):
            return np.array([t])  # Time-varying!

        dynamics = LLRQDynamics(network, relaxation_matrix=K, external_drive=drive_func)

        x0 = np.array([1.0])
        t = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_t = dynamics.analytical_solution(x0, t)

            # Should warn about time-varying drive
            assert any("time-varying" in str(warning.message) for warning in w)

    def test_invalid_initial_condition_length(self):
        """Test error for wrong initial condition length."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)

        with pytest.raises(ValueError, match="Expected 1 initial conditions"):
            dynamics.analytical_solution(np.array([1.0, 2.0]), np.array([0.0, 1.0]))


class TestSingleReactionSolution:
    """Test single reaction analytical solution."""

    def test_single_reaction_no_drive(self):
        """Test single reaction solution without external drive."""
        dynamics = LLRQDynamics(ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]])))

        Q0 = 4.0
        Keq = 2.0
        k = 1.5
        t = np.array([0.0, 1.0, 2.0])

        t_out, Q_out = dynamics.single_reaction_solution(Q0, Keq, k, t=t)

        assert np.array_equal(t_out, t)
        assert len(Q_out) == len(t)
        assert np.isclose(Q_out[0], Q0)  # Initial condition

    def test_single_reaction_with_drive(self):
        """Test single reaction solution with external drive."""
        dynamics = LLRQDynamics(ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]])))

        Q0 = 1.0
        Keq = 1.0
        k = 1.0

        def u_func(t):
            return 0.5  # Constant drive

        t = np.array([0.0, 1.0])
        t_out, Q_out = dynamics.single_reaction_solution(Q0, Keq, k, u_func, t)

        assert len(Q_out) == len(t)
        assert np.isclose(Q_out[0], Q0)

    def test_single_reaction_default_time(self):
        """Test single reaction solution with default time points."""
        dynamics = LLRQDynamics(ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]])))

        Q0 = 2.0
        Keq = 1.0
        k = 1.0

        t_out, Q_out = dynamics.single_reaction_solution(Q0, Keq, k)

        assert len(t_out) > 0
        assert len(Q_out) == len(t_out)
        assert np.isclose(Q_out[0], Q0)


class TestFromMassAction:
    """Test factory method for creating dynamics from mass action."""

    def test_from_mass_action_basic(self):
        """Test basic from_mass_action factory method."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        assert isinstance(dynamics, LLRQDynamics)
        assert dynamics.network == network
        assert len(dynamics.Keq) == 1
        assert dynamics.K.shape == (1, 1)

    def test_from_mass_action_equilibrium_constants(self):
        """Test that equilibrium constants are computed correctly."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([3.0])
        k_minus = np.array([1.5])

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        expected_Keq = k_plus / k_minus
        assert np.allclose(dynamics.Keq, expected_Keq)

    def test_get_mass_action_info(self):
        """Test retrieval of mass action information."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        info = dynamics.get_mass_action_info()

        assert info is not None
        assert info["mode"] == "equilibrium"
        assert np.allclose(info["equilibrium_point"], c_star)
        assert np.allclose(info["forward_rates"], k_plus)
        assert np.allclose(info["backward_rates"], k_minus)
        assert "dynamics_data" in info

    def test_no_mass_action_info_for_normal_creation(self):
        """Test that normally created dynamics don't have mass action info."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        dynamics = LLRQDynamics(network)

        info = dynamics.get_mass_action_info()
        assert info is None

    def test_from_mass_action_multiple_reactions(self):
        """Test from_mass_action with multiple reactions."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        c_star = np.array([1.0, 1.5, 0.5])
        k_plus = np.array([2.0, 1.0])
        k_minus = np.array([1.0, 2.0])

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        assert len(dynamics.Keq) == 2
        assert dynamics.K.shape == (2, 2)

        expected_Keq = k_plus / k_minus
        assert np.allclose(dynamics.Keq, expected_Keq)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
