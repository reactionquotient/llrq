"""
Comprehensive tests for edge cases and error conditions in LLRQ package.

Tests zero concentrations, singular matrices, invalid inputs, numerical stability,
and other boundary conditions that could cause failures.
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


class TestZeroAndNegativeConcentrations:
    """Test handling of zero and negative concentrations."""

    def test_zero_initial_concentrations(self):
        """Test with zero initial concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)

        # Zero concentrations should be handled gracefully
        zero_concentrations = np.array([0.0, 0.0])
        Q = network.compute_reaction_quotients(zero_concentrations)

        # Should not be NaN or Inf
        assert np.isfinite(Q).all()

    def test_single_zero_concentration(self):
        """Test with one species having zero concentration."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        concentrations = np.array([0.0, 1.0])
        Q = network.compute_reaction_quotients(concentrations)

        # Should handle gracefully
        assert np.isfinite(Q).all()

    def test_negative_concentrations_error(self):
        """Test error handling for negative concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        # Negative concentrations should be handled appropriately
        negative_concentrations = np.array([-1.0, 2.0])
        t_span = (0.0, 1.0)

        try:
            result = solver.solve(negative_concentrations, t_span, method="numerical")
            # If it succeeds, check that result is meaningful
            if result["success"]:
                assert np.isfinite(result["concentrations"]).all()
        except ValueError:
            # It's also acceptable to raise an error for negative concentrations
            pass

    def test_very_small_concentrations(self):
        """Test with very small but positive concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)

        tiny_concentrations = np.array([1e-12, 1e-15])
        Q = network.compute_reaction_quotients(tiny_concentrations)

        # Should not underflow to zero or become infinite
        assert np.isfinite(Q).all()
        assert not np.any(Q == 0)

    def test_concentration_underflow_protection(self):
        """Test protection against concentration underflow."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Test log deviation computation with very small quotients
        Keq = np.array([1.0])
        dynamics = LLRQDynamics(network, Keq)

        tiny_Q = np.array([1e-100])
        x = dynamics.compute_log_deviation(tiny_Q)

        # Should not be -Inf
        assert np.isfinite(x).all()

    def test_zero_equilibrium_constants(self):
        """Test handling of zero equilibrium constants."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Zero equilibrium constant is problematic but should be handled
        try:
            Keq = np.array([0.0])
            dynamics = LLRQDynamics(network, Keq)

            Q = np.array([1.0])
            x = dynamics.compute_log_deviation(Q)

            # Should not be NaN or Inf
            assert np.isfinite(x).all()

        except (ValueError, FloatingPointError):
            # It's acceptable to raise an error for zero Keq
            pass


class TestSingularMatrices:
    """Test handling of singular and ill-conditioned matrices."""

    def test_singular_stoichiometric_matrix(self):
        """Test with singular stoichiometric matrix."""
        # Create dependent reactions: R1 = A -> B, R2 = 2*(A -> B)
        network = ReactionNetwork(["A", "B"], ["R1", "R2"], np.array([[-1, -2], [1, 2]]))

        # Matrix operations should handle rank deficiency
        independent = network.get_independent_reactions()

        # Should identify only one independent reaction
        assert len(independent) <= 1

    def test_singular_dynamics_matrix(self):
        """Test with singular dynamics matrix K."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Singular K matrix
        K = np.array([[0.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        # Should handle gracefully in analytical solution
        x0 = np.array([1.0])
        t = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_t = dynamics.analytical_solution(x0, t)

            # Should warn about singularity
            assert any("singular" in str(warning.message).lower() for warning in w)

    def test_ill_conditioned_matrix(self):
        """Test with ill-conditioned matrix."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        # Create ill-conditioned K matrix
        K = np.array([[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]])  # Nearly singular
        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        # Should handle numerical issues gracefully
        x0 = np.array([1.0, -1.0])
        t = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_t = dynamics.analytical_solution(x0, t)

            # Might issue warnings about conditioning
            assert np.isfinite(x_t).all()

    def test_rank_deficient_conservation_matrix(self):
        """Test conservation law computation with rank deficient system."""
        # Create system where S has zero rows (species not involved)
        network = ReactionNetwork(["A", "B", "C"], ["R1"], np.array([[-1], [1], [0]]))  # C not involved

        C = network.find_conservation_laws()

        # Should find conservation laws including the trivial one for C
        assert C.shape[0] >= 1

    def test_pseudoinverse_fallback(self):
        """Test pseudoinverse fallback for singular matrices."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        c_star = np.array([1e-12, 1e-12])  # Nearly zero equilibrium
        k_plus = np.array([1.0])
        k_minus = np.array([1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = network.compute_dynamics_matrix(
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="nonequilibrium",  # More likely to hit singularities
            )

            # Should complete without crashing
            assert "K" in result


class TestInvalidInputs:
    """Test various invalid inputs and their error handling."""

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched array dimensions."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Wrong number of concentrations
        with pytest.raises(ValueError):
            network.compute_reaction_quotients([1.0])  # Should be 2

        # Wrong number of forward rates
        with pytest.raises(ValueError):
            network.compute_dynamics_matrix(
                forward_rates=[1.0, 2.0],
                backward_rates=[1.0],
                initial_concentrations=[1.0, 2.0],  # Should be 1
            )

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        # Empty species list
        with pytest.raises((ValueError, IndexError)):
            ReactionNetwork([], [], np.array([]).reshape(0, 0))

    def test_invalid_stoichiometric_matrix_shape(self):
        """Test invalid stoichiometric matrix shapes."""
        # Wrong shape matrix
        with pytest.raises(ValueError):
            ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1], [0]]))  # 3x1 for 2 species

    def test_invalid_string_inputs(self):
        """Test invalid string inputs."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Invalid reaction ID
        with pytest.raises(ValueError):
            network.compute_single_reaction_quotient("invalid_id", [1.0, 2.0])

    def test_non_numeric_inputs(self):
        """Test non-numeric inputs."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # String instead of numeric array
        with pytest.raises((TypeError, ValueError)):
            network.compute_reaction_quotients(["not", "numeric"])

    def test_infinite_inputs(self):
        """Test infinite input values."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Infinite concentrations
        infinite_concentrations = np.array([np.inf, 1.0])

        try:
            Q = network.compute_reaction_quotients(infinite_concentrations)
            # If it doesn't raise an error, should produce meaningful result
            assert not np.isnan(Q).any()
        except (ValueError, OverflowError):
            # It's acceptable to reject infinite inputs
            pass

    def test_nan_inputs(self):
        """Test NaN input values."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # NaN concentrations
        nan_concentrations = np.array([np.nan, 1.0])

        with pytest.raises((ValueError, FloatingPointError)):
            network.compute_reaction_quotients(nan_concentrations)


class TestNumericalStability:
    """Test numerical stability under various conditions."""

    def test_extreme_rate_ratios(self):
        """Test with extreme ratios of forward/backward rates."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        c_star = np.array([1.0, 1.0])
        k_plus = np.array([1e10])
        k_minus = np.array([1e-10])  # Extreme ratio

        try:
            dynamics = LLRQDynamics.from_mass_action(
                network=network,
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="equilibrium",
            )

            # Should produce finite K matrix
            assert np.isfinite(dynamics.K).all()

            # Equilibrium constants should be finite
            assert np.isfinite(dynamics.Keq).all()

        except (OverflowError, ValueError):
            # Acceptable to fail with extreme ratios
            pass

    def test_extreme_concentrations(self):
        """Test with extreme concentration values."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Very large and very small concentrations
        extreme_concentrations = np.array([1e100, 1e-100])

        try:
            Q = network.compute_reaction_quotients(extreme_concentrations)

            # Should not overflow or underflow inappropriately
            assert np.isfinite(Q).all()

        except (OverflowError, FloatingPointError):
            # May legitimately overflow with extreme values
            pass

    def test_stiff_dynamics(self):
        """Test with stiff dynamics (widely separated time scales)."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        # Create stiff system with fast and slow reactions
        K = np.array([[1e6, 0], [0, 1e-6]])  # Very different eigenvalues
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0, 1.0])
        t_span = (0.0, 1.0)

        try:
            # May need small time steps for stiff system
            result = solver.solve(initial_concentrations, t_span, method="numerical", rtol=1e-6)

            if result["success"]:
                # Should not have numerical artifacts
                assert np.isfinite(result["concentrations"]).all()

        except (ValueError, RuntimeError):
            # Stiff systems may legitimately fail without specialized solvers
            pass

    def test_numerical_derivatives_accuracy(self):
        """Test accuracy of numerical derivatives in matrix computation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Test with concentrations near machine precision limits
        c_star = np.array([1e-8, 1e8])
        k_plus = np.array([1.0])
        k_minus = np.array([1.0])

        try:
            result = network.compute_dynamics_matrix(
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="nonequilibrium",  # Uses numerical derivatives
            )

            # Should produce reasonable K matrix
            assert np.isfinite(result["K"]).all()

        except (ValueError, FloatingPointError):
            # May fail with extreme concentration ranges
            pass

    def test_matrix_exponential_stability(self):
        """Test stability of matrix exponential computation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Large eigenvalue that might cause exp() overflow
        K = np.array([[100.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x0 = np.array([10.0])  # Large initial deviation
        t = np.array([0.0, 0.1, 1.0])  # Include some reasonable times

        try:
            x_t = dynamics.analytical_solution(x0, t)

            # Should not overflow
            assert np.isfinite(x_t).all()

        except (OverflowError, FloatingPointError):
            # May legitimately overflow with large K and times
            pass


class TestMemoryAndPerformanceEdgeCases:
    """Test edge cases related to memory and performance."""

    def test_large_sparse_matrix(self):
        """Test with large sparse stoichiometric matrix."""
        # Create large system with mostly zeros
        n_species = 100
        n_reactions = 50

        species_ids = [f"S{i}" for i in range(n_species)]
        reaction_ids = [f"R{i}" for i in range(n_reactions)]

        # Create sparse matrix (most entries zero)
        S = np.zeros((n_species, n_reactions))
        for j in range(n_reactions):
            # Each reaction involves only 2-3 species
            involved_species = np.random.choice(n_species, size=2, replace=False)
            S[involved_species[0], j] = -1  # Reactant
            S[involved_species[1], j] = 1  # Product

        network = ReactionNetwork(species_ids, reaction_ids, S)

        # Basic operations should work
        assert network.n_species == n_species
        assert network.n_reactions == n_reactions

        # Concentration computation should handle large arrays
        large_concentrations = np.ones(n_species)
        Q = network.compute_reaction_quotients(large_concentrations)
        assert len(Q) == n_reactions

    def test_zero_time_span(self):
        """Test with zero time span."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (0.0, 0.0)  # Zero duration

        try:
            result = solver.solve(initial_concentrations, t_span, method="numerical")

            if result["success"]:
                # Should return initial conditions
                assert len(result["time"]) >= 1
                assert np.allclose(result["concentrations"][0], initial_concentrations)

        except ValueError:
            # May reject zero time span
            pass

    def test_single_time_point(self):
        """Test with single time point."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_eval = np.array([0.0])  # Single time point

        result = solver.solve(initial_concentrations, t_eval, method="numerical")

        if result["success"]:
            assert len(result["time"]) == 1
            assert np.allclose(result["concentrations"][0], initial_concentrations)

    def test_backwards_time_span(self):
        """Test with backwards time span."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0])
        t_span = (1.0, 0.0)  # Backwards time

        try:
            result = solver.solve(initial_concentrations, t_span, method="numerical")
            # Some solvers can handle backwards integration
            if result["success"]:
                assert len(result["time"]) > 0
        except ValueError:
            # It's acceptable to reject backwards time
            pass


class TestSpecialNetworkTopologies:
    """Test special network topologies that might cause issues."""

    def test_disconnected_network(self):
        """Test network with disconnected components."""
        # A ⇌ B and C ⇌ D (disconnected)
        network = ReactionNetwork(["A", "B", "C", "D"], ["R1", "R2"], np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))

        # Should find multiple conservation laws
        C = network.find_conservation_laws()
        assert C.shape[0] >= 2  # At least two conservation laws

        # Should work with dynamics
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([1.0, 1.0, 2.0, 1.0])
        t_span = (0.0, 1.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")
        assert result["success"]

    def test_self_loop_reaction(self):
        """Test reaction where species reacts with itself."""
        # 2A ⇌ B reaction
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-2], [1]]))

        concentrations = np.array([2.0, 1.0])
        Q = network.compute_reaction_quotients(concentrations)

        # Q = [B]/[A]^2 = 1.0/4.0 = 0.25
        assert np.isclose(Q[0], 0.25)

    def test_catalytic_reaction(self):
        """Test catalytic reaction where catalyst is not consumed."""
        # A + E ⇌ B + E (E is catalyst)
        network = ReactionNetwork(["A", "E", "B"], ["R1"], np.array([[-1], [0], [1]]))  # E not consumed

        concentrations = np.array([2.0, 1.0, 3.0])
        Q = network.compute_reaction_quotients(concentrations)

        # Q = [B]/[A] (E cancels out in quotient)
        assert np.isclose(Q[0], 3.0 / 2.0)

    def test_empty_reaction(self):
        """Test 'reaction' with no net change."""
        # Trivial reaction: 0 → 0
        network = ReactionNetwork(["A"], ["R1"], np.array([[0]]))

        concentrations = np.array([1.0])
        Q = network.compute_reaction_quotients(concentrations)

        # Q = 1 for trivial reaction
        assert np.isclose(Q[0], 1.0)


class TestFloatingPointEdgeCases:
    """Test floating point edge cases and precision issues."""

    def test_precision_loss_in_quotients(self):
        """Test precision loss in quotient calculations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Nearly equal concentrations
        concentrations = np.array([1.0, 1.0 + 1e-15])
        Q = network.compute_reaction_quotients(concentrations)

        # Should handle precision gracefully
        assert np.isfinite(Q).all()
        assert Q[0] > 1.0  # Should detect the small difference

    def test_cancellation_in_log_computation(self):
        """Test cancellation errors in log computations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        Keq = np.array([1.0 + 1e-15])  # Nearly 1
        dynamics = LLRQDynamics(network, Keq)

        Q = np.array([1.0 + 1e-15])  # Also nearly 1
        x = dynamics.compute_log_deviation(Q)

        # Should handle near-cancellation
        assert np.isfinite(x).all()

    def test_denormal_numbers(self):
        """Test handling of denormal (subnormal) numbers."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Denormal concentrations
        denormal_concentrations = np.array([1e-323, 1e-322])  # Near machine limit

        try:
            Q = network.compute_reaction_quotients(denormal_concentrations)
            assert np.isfinite(Q).all()
        except FloatingPointError:
            # May legitimately fail with denormal numbers
            pass

    def test_mixed_precision_inputs(self):
        """Test mixed precision inputs."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Mix float32 and float64
        concentrations = np.array([np.float32(1.0), np.float64(2.0)])
        Q = network.compute_reaction_quotients(concentrations)

        # Should handle mixed precision
        assert np.isfinite(Q).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
