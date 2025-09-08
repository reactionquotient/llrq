"""
Comprehensive tests for LLRQSolver class.

Tests numerical and analytical solvers, conservation enforcement,
different initial condition formats, and error handling.
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
from llrq.solver import LLRQSolver


class TestLLRQSolverInitialization:
    """Test LLRQSolver initialization."""

    def test_initialization_basic(self):
        """Test basic solver initialization."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        assert solver.dynamics == dynamics
        assert solver.network == network

    def test_initialization_with_complex_system(self):
        """Test initialization with multi-reaction system."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[2.0, -0.5], [-0.5, 1.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        assert solver.dynamics == dynamics
        assert solver.network == network


class TestSolverBasic:
    """Test basic solver functionality."""

    def test_solve_with_concentration_initial_conditions(self):
        """Test solving with concentration initial conditions."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0])
        t_span = (0.0, 2.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert "time" in result
        assert "concentrations" in result
        assert "reaction_quotients" in result
        assert "log_deviations" in result
        assert result["success"]

        # Check shapes
        n_points = len(result["time"])
        assert result["concentrations"].shape == (n_points, 2)
        assert result["reaction_quotients"].shape == (n_points, 1)
        assert result["log_deviations"].shape == (n_points, 1)

    def test_solve_with_dict_initial_conditions(self):
        """Test solving with dictionary initial conditions."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_conditions = {"A": 3.0, "B": 0.5}
        t_span = (0.0, 1.0)

        result = solver.solve(initial_conditions, t_span, method="numerical")

        assert result["success"]
        # Check that initial conditions were parsed correctly
        expected_c0 = np.array([3.0, 0.5])
        assert np.allclose(result["initial_concentrations"], expected_c0)

    def test_solve_with_time_array(self):
        """Test solving with explicit time array."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_eval = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        result = solver.solve(initial_concentrations, t_eval, method="numerical")

        assert np.array_equal(result["time"], t_eval)
        assert result["success"]

    def test_solve_auto_method_selection(self):
        """Test automatic method selection."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, method="auto")

        assert result["success"]
        assert result["method"] in ["analytical", "numerical"]

    def test_solve_analytical_method(self):
        """Test analytical solution method."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 0.5])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, method="analytical")

        assert result["success"] or result["method"] == "numerical"  # May fallback
        assert len(result["time"]) > 0

    def test_solve_numerical_method(self):
        """Test numerical solution method."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]
        assert result["method"] == "numerical"


class TestSolverMultipleReactions:
    """Test solver with multiple reactions."""

    def test_solve_two_reaction_system(self):
        """Test A ⇌ B ⇌ C system."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[2.0, -0.1], [-0.1, 1.5]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([3.0, 1.0, 0.5])
        t_span = (0.0, 2.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]
        assert result["concentrations"].shape[1] == 3  # 3 species
        assert result["reaction_quotients"].shape[1] == 2  # 2 reactions

    def test_solve_complex_network(self):
        """Test more complex reaction network."""
        # A + B ⇌ C, C ⇌ D
        network = ReactionNetwork(["A", "B", "C", "D"], ["R1", "R2"], np.array([[-1, 0], [-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 2.0, 0.1, 0.1])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]
        assert result["concentrations"].shape[1] == 4
        assert result["reaction_quotients"].shape[1] == 2

    def test_solve_with_coupling_matrix(self):
        """Test with coupled reactions (off-diagonal K elements)."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        K = np.array([[1.0, 0.5], [0.2, 2.0]])  # Coupled reactions
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0, 1.0])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]


class TestConservationEnforcement:
    """Test conservation law enforcement."""

    def test_solve_with_conservation_enforcement(self):
        """Test solution with conservation law enforcement."""
        # A ⇌ B system has A + B = constant
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([3.0, 1.0])
        t_span = (0.0, 2.0)

        result = solver.solve(initial_concentrations, t_span, enforce_conservation=True, method="numerical")

        assert result["success"]

        # Check conservation: A + B should remain constant
        if result["concentrations"] is not None:
            total_mass = result["concentrations"].sum(axis=1)
            initial_total = initial_concentrations.sum()

            # Should be approximately conserved
            assert np.allclose(total_mass, initial_total, rtol=1e-3)

    def test_solve_without_conservation_enforcement(self):
        """Test solution without conservation enforcement."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, enforce_conservation=False, method="numerical")

        assert result["success"]

    def test_solve_multiple_conservation_laws(self):
        """Test system with multiple conservation laws."""
        # Two separate reactions: A ⇌ B, C ⇌ D
        network = ReactionNetwork(["A", "B", "C", "D"], ["R1", "R2"], np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0, 3.0, 0.5])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, enforce_conservation=True, method="numerical")

        assert result["success"]


class TestExternalDrives:
    """Test solver with external drives."""

    def test_solve_with_constant_drive(self):
        """Test solution with constant external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        def drive_func(t):
            return np.array([0.5])

        dynamics = LLRQDynamics(network, external_drive=drive_func)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 2.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]

    def test_solve_with_time_varying_drive(self):
        """Test solution with time-varying external drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        def drive_func(t):
            return np.array([np.sin(t)])

        dynamics = LLRQDynamics(network, external_drive=drive_func)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 0.5])
        t_span = (0.0, 2 * np.pi)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]

    def test_solve_analytical_with_drive_fallback(self):
        """Test that analytical method falls back with time-varying drive."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        def drive_func(t):
            return np.array([t])  # Time-varying

        dynamics = LLRQDynamics(network, external_drive=drive_func)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 1.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = solver.solve(initial_concentrations, t_span, method="analytical")

            # Should succeed but possibly with warnings
            assert result["success"]


class TestErrorHandling:
    """Test error handling in solver."""

    def test_invalid_initial_conditions_dict(self):
        """Test error for invalid species in initial conditions dict."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        # Include invalid species 'C'
        initial_conditions = {"A": 1.0, "B": 2.0, "C": 3.0}
        t_span = (0.0, 1.0)

        with pytest.raises((ValueError, KeyError)):
            solver.solve(initial_conditions, t_span)

    def test_missing_species_in_dict(self):
        """Test handling of missing species in initial conditions dict."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        # Missing species 'B'
        initial_conditions = {"A": 1.0}
        t_span = (0.0, 1.0)

        # Should handle gracefully (set missing to zero or raise error)
        try:
            result = solver.solve(initial_conditions, t_span)
            # If it succeeds, check that missing species was set to zero
            if result["success"]:
                assert result["initial_concentrations"][1] == 0.0  # B should be 0
        except (ValueError, KeyError):
            # It's also acceptable to raise an error
            pass

    def test_invalid_method(self):
        """Test error for invalid solution method."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 1.0)

        with pytest.raises(ValueError):
            solver.solve(initial_concentrations, t_span, method="invalid_method")

    def test_negative_concentrations(self):
        """Test handling of negative initial concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([-1.0, 2.0])  # Negative concentration
        t_span = (0.0, 1.0)

        # Should either handle gracefully or raise meaningful error
        try:
            result = solver.solve(initial_concentrations, t_span, method="numerical")
            # If it succeeds, check that solution is meaningful
            if result["success"]:
                assert np.isfinite(result["concentrations"]).all()
        except ValueError as e:
            # It's acceptable to raise an error for negative concentrations
            assert "negative" in str(e).lower() or "concentration" in str(e).lower()


class TestSolverOptions:
    """Test various solver options and parameters."""

    def test_solve_with_custom_n_points(self):
        """Test solving with custom number of time points."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 2.0)
        n_points = 50

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=n_points)

        assert result["success"]
        assert len(result["time"]) == n_points

    def test_solve_with_solver_options(self):
        """Test solving with additional solver options."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 0.5])
        t_span = (0.0, 1.0)

        # Pass additional options to numerical solver
        result = solver.solve(initial_concentrations, t_span, method="numerical", rtol=1e-8, atol=1e-10)

        assert result["success"]

    def test_solve_short_time_span(self):
        """Test solving over very short time span."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 1e-6)  # Very short time

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]

    def test_solve_long_time_span(self):
        """Test solving over long time span."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[0.1]])  # Slow relaxation
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0])
        t_span = (0.0, 100.0)  # Long time

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]


class TestSolverIntegration:
    """Test solver integration with other components."""

    def test_solve_from_mass_action_dynamics(self):
        """Test solving dynamics created from mass action."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        c_star = np.array([1.0, 2.0])
        k_plus = np.array([2.0])
        k_minus = np.array([1.0])

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([0.5, 1.0])
        t_span = (0.0, 3.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]

        # Should approach equilibrium
        if result["concentrations"] is not None:
            final_Q = result["reaction_quotients"][-1]
            expected_Keq = k_plus / k_minus
            # Should be close to equilibrium at the end
            assert np.allclose(final_Q, expected_Keq, rtol=0.1)

    def test_solve_with_sbml_derived_network(self):
        """Test solving with network derived from SBML-like data."""
        sbml_data = {
            "species_ids": ["A", "B", "C"],
            "reaction_ids": ["R1", "R2"],
            "stoichiometric_matrix": np.array([[-1, 0], [1, -1], [0, 1]]),
            "species": {
                "A": {"initial_concentration": 2.0},
                "B": {"initial_concentration": 1.0},
                "C": {"initial_concentration": 0.1},
            },
            "reactions": [],
            "parameters": {},
        }

        network = ReactionNetwork.from_sbml_data(sbml_data)
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        # Use initial concentrations from SBML data
        initial_concentrations = network.get_initial_concentrations()
        t_span = (0.0, 2.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        assert result["success"]
        assert np.allclose(result["initial_concentrations"], np.array([2.0, 1.0, 0.1]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
