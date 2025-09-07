"""
Unit tests for CVXpy-based control optimization.

These tests verify the correctness and robustness of the cvxpy integration,
including error handling when cvxpy is not available, comparison with analytical
solutions, and testing of pre-built templates and custom callbacks.
"""

import pytest
import numpy as np
import numpy.testing as npt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.control import LLRQController
from llrq.cvx_control import (
    CVXController,
    CVXObjectives,
    CVXConstraints,
)
import cvxpy as cp


class TestCVXController:
    """Test the main CVXController class functionality."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple A ⇌ B system for testing."""
        network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])
        initial_concentrations = np.array([1.5, 0.5])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return {
            "network": network,
            "dynamics": dynamics,
            "solver": solver,
            "forward_rates": forward_rates,
            "backward_rates": backward_rates,
            "initial_concentrations": initial_concentrations,
        }

    @pytest.fixture
    def chain_system(self):
        """Create A ⇌ B ⇌ C system for more complex testing."""
        network = ReactionNetwork(
            species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
        )

        forward_rates = np.array([2.0, 1.5])
        backward_rates = np.array([1.0, 0.8])
        initial_concentrations = np.array([2.0, 1.0, 0.5])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return {
            "network": network,
            "dynamics": dynamics,
            "solver": solver,
            "forward_rates": forward_rates,
            "backward_rates": backward_rates,
            "initial_concentrations": initial_concentrations,
        }

    def test_cvx_controller_initialization(self, simple_system):
        """Test CVXController initialization."""
        controller = CVXController(simple_system["solver"])

        assert controller.solver is simple_system["solver"]
        assert controller.network is simple_system["network"]
        assert hasattr(controller, "G")
        assert hasattr(controller, "B")

    def test_cvx_controller_with_controlled_reactions(self, chain_system):
        """Test CVXController with specific controlled reactions."""
        # Control only the first reaction
        controller = CVXController(chain_system["solver"], controlled_reactions=[0])

        assert len(controller.controlled_reactions) == 1
        assert controller.controlled_reactions[0] == 0
        assert controller.G.shape == (2, 1)  # 2 reactions, 1 controlled

    def test_default_objective_tracking(self, simple_system):
        """Test default objective with target tracking."""
        controller = CVXController(simple_system["solver"])

        x_target = np.array([0.5])
        result = controller.compute_cvx_control(x_target=x_target, constraints_fn=CVXConstraints.steady_state())

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        assert result["variables"]["x"] is not None
        assert len(result["variables"]["x"].value) == 1

    def test_default_objective_no_target(self, simple_system):
        """Test default objective without target (minimize control effort)."""
        controller = CVXController(simple_system["solver"])

        result = controller.compute_cvx_control()

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        assert result["u_optimal"] is not None
        # Without target, should minimize control effort (close to zero)
        npt.assert_allclose(result["u_optimal"], 0, atol=1e-6)


class TestCVXObjectives:
    """Test pre-built objective function templates."""

    @pytest.fixture
    def chain_system(self):
        """Create A ⇌ B ⇌ C system."""
        network = ReactionNetwork(
            species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
        )

        forward_rates = np.array([2.0, 1.5])
        backward_rates = np.array([1.0, 0.8])
        initial_concentrations = np.array([2.0, 1.0, 0.5])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return solver

    def test_sparse_control_objective(self, chain_system):
        """Test L1-regularized sparse control objective."""
        controller = CVXController(chain_system)
        x_target = np.array([0.5, -0.3])

        # Test sparse control with high sparsity weight
        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.sparse_control(sparsity_weight=10.0),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        u_sparse = result["u_optimal"]

        # Compare with low sparsity weight
        result_dense = controller.compute_cvx_control(
            objective_fn=CVXObjectives.sparse_control(sparsity_weight=0.01),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
        )

        u_dense = result_dense["u_optimal"]

        # Sparse solution should have smaller L1 norm
        assert np.linalg.norm(u_sparse, 1) <= np.linalg.norm(u_dense, 1) + 1e-6

    def test_multi_objective(self, chain_system):
        """Test multi-objective optimization."""
        controller = CVXController(chain_system)
        x_target = np.array([0.4, -0.2])

        weights = {"tracking": 1.0, "control": 0.5, "sparsity": 0.1}

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective(weights),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        assert result["u_optimal"] is not None

        # Verify achieved state is close to target
        x_achieved = result["variables"]["x"].value
        tracking_error = np.linalg.norm(x_achieved - x_target)
        assert tracking_error < 1.0  # Should be reasonable

    def test_robust_tracking_objective(self, chain_system):
        """Test robust tracking objective."""
        controller = CVXController(chain_system)
        x_target = np.array([0.3, -0.1])

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.robust_tracking(uncertainty_weight=0.2),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        assert result["u_optimal"] is not None


class TestCVXConstraints:
    """Test pre-built constraint function templates."""

    @pytest.fixture
    def chain_system(self):
        """Create A ⇌ B ⇌ C system."""
        network = ReactionNetwork(
            species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
        )

        forward_rates = np.array([2.0, 1.5])
        backward_rates = np.array([1.0, 0.8])
        initial_concentrations = np.array([2.0, 1.0, 0.5])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return solver

    def test_steady_state_constraint(self, chain_system):
        """Test steady-state constraint K*x = u."""
        controller = CVXController(chain_system)
        x_target = np.array([0.2, -0.1])

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]

        # Verify steady-state constraint is satisfied
        u = result["u_optimal"]
        x = result["variables"]["x"].value
        K = chain_system.dynamics.K

        npt.assert_allclose(K @ x, u, atol=1e-6)

    def test_box_bounds_constraint(self, chain_system):
        """Test box constraints on control inputs."""
        controller = CVXController(chain_system)

        u_min, u_max = -1.0, 2.0

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"control": 1.0}),
            constraints_fn=CVXConstraints.box_bounds(u_min=u_min, u_max=u_max),
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]

        # Verify bounds are satisfied
        u = result["u_optimal"]
        assert np.all(u >= u_min - 1e-6)
        assert np.all(u <= u_max + 1e-6)

    def test_control_budget_l1(self, chain_system):
        """Test L1 control budget constraint."""
        controller = CVXController(chain_system)

        budget = 1.5

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
            constraints_fn=CVXConstraints.combine(
                CVXConstraints.steady_state(), CVXConstraints.control_budget(budget, norm_type=1)
            ),
            x_target=np.array([0.1, -0.05]),
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]

        # Verify budget constraint
        u = result["u_optimal"]
        assert np.linalg.norm(u, 1) <= budget + 1e-6

    def test_control_budget_l2(self, chain_system):
        """Test L2 control budget constraint."""
        controller = CVXController(chain_system)

        budget = 1.0

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
            constraints_fn=CVXConstraints.combine(
                CVXConstraints.steady_state(), CVXConstraints.control_budget(budget, norm_type=2)
            ),
            x_target=np.array([0.08, -0.04]),
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]

        # Verify budget constraint
        u = result["u_optimal"]
        assert np.linalg.norm(u, 2) <= budget + 1e-6

    def test_state_bounds_constraint(self, chain_system):
        """Test state bounds constraint."""
        controller = CVXController(chain_system)

        x_min, x_max = -0.5, 0.5

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"control": 1.0}),
            constraints_fn=CVXConstraints.combine(
                CVXConstraints.steady_state(), CVXConstraints.state_bounds(x_min=x_min, x_max=x_max)
            ),
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]

        # Verify state bounds
        x = result["variables"]["x"].value
        assert np.all(x >= x_min - 1e-6)
        assert np.all(x <= x_max + 1e-6)

    def test_combine_constraints(self, chain_system):
        """Test combining multiple constraints."""
        controller = CVXController(chain_system)

        combined_constraints = CVXConstraints.combine(
            CVXConstraints.steady_state(),
            CVXConstraints.box_bounds(u_min=-1.0, u_max=1.0),
            CVXConstraints.control_budget(1.5, norm_type=1),
        )

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
            constraints_fn=combined_constraints,
            x_target=np.array([0.1, -0.05]),
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]

        # Verify all constraints
        u = result["u_optimal"]
        x = result["variables"]["x"].value
        K = chain_system.dynamics.K

        # Steady state
        npt.assert_allclose(K @ x, u, atol=1e-6)
        # Box bounds
        assert np.all(u >= -1.0 - 1e-6)
        assert np.all(u <= 1.0 + 1e-6)
        # L1 budget
        assert np.linalg.norm(u, 1) <= 1.5 + 1e-6


class TestCustomCallbacks:
    """Test custom objective and constraint callbacks."""

    @pytest.fixture
    def simple_system(self):
        """Create simple system for testing callbacks."""
        network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])
        initial_concentrations = np.array([1.0, 1.0])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return solver

    def test_custom_objective_callback(self, simple_system):
        """Test custom objective function callback."""
        controller = CVXController(simple_system)

        def custom_objective(variables, params):
            """Simple custom objective: minimize control squared."""
            u = variables["u"]
            return cp.sum_squares(u)

        result = controller.compute_cvx_control(objective_fn=custom_objective)

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        # Should minimize to near zero
        npt.assert_allclose(result["u_optimal"], 0, atol=1e-6)

    def test_custom_constraint_callback(self, simple_system):
        """Test custom constraint function callback."""
        controller = CVXController(simple_system)

        def custom_constraints(variables, params):
            """Custom constraint: u must equal 1.0."""
            u = variables["u"]
            return [u == 1.0]

        result = controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"control": 1.0}), constraints_fn=custom_constraints
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        npt.assert_allclose(result["u_optimal"], 1.0, atol=1e-6)

    def test_callback_parameter_passing(self, simple_system):
        """Test that parameters are correctly passed to callbacks."""
        controller = CVXController(simple_system)

        def parameter_aware_objective(variables, params):
            """Objective that uses custom parameters."""
            u = variables["u"]
            weight = params.get("custom_weight", 1.0)
            return weight * cp.sum_squares(u)

        # Test with different weights
        result1 = controller.compute_cvx_control(objective_fn=parameter_aware_objective, custom_weight=1.0)

        result2 = controller.compute_cvx_control(objective_fn=parameter_aware_objective, custom_weight=10.0)

        # Both should succeed
        assert result1["status"] in ["optimal", "optimal_inaccurate"]
        assert result2["status"] in ["optimal", "optimal_inaccurate"]


class TestComparisonWithAnalytical:
    """Test that cvxpy solutions match analytical solutions where applicable."""

    @pytest.fixture
    def simple_system(self):
        """Create simple A ⇌ B system."""
        network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

        forward_rates = np.array([3.0])
        backward_rates = np.array([1.5])
        initial_concentrations = np.array([1.5, 0.5])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return solver

    def test_unconstrained_steady_state_control(self, simple_system):
        """Test that unconstrained CVX matches analytical steady-state control."""
        analytical_controller = LLRQController(simple_system)
        cvx_controller = CVXController(simple_system)

        # Define target in reduced coordinates
        y_target = np.array([0.5])  # Target reduced state

        # Analytical solution
        u_analytical = analytical_controller.compute_steady_state_control(y_target)

        # CVX solution with steady-state constraint
        result = cvx_controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=simple_system._B @ y_target,  # Convert to reaction forces
        )

        assert result["status"] in ["optimal", "optimal_inaccurate"]
        u_cvx = result["u_optimal"]

        # Should match within numerical tolerance
        npt.assert_allclose(u_analytical, u_cvx, rtol=1e-4)

    def test_entropy_aware_control_comparison(self, simple_system):
        """Test CVX entropy-aware control against analytical version."""
        # This test would require setting up entropy metrics
        # For now, we'll test that both methods produce reasonable results

        analytical_controller = LLRQController(simple_system)
        cvx_controller = CVXController(simple_system)

        x_target = np.array([0.3])

        # Create Onsager conductance matrix for analytical controller
        L = np.array([[1.0]])

        # Compute entropy metric M for CVX controller
        M = analytical_controller.compute_control_entropy_metric(L)

        # Analytical entropy-aware control (uses L and computes M internally)
        result_analytical = analytical_controller.compute_entropy_aware_steady_state_control(x_target, L, entropy_weight=0.1)
        u_analytical = result_analytical["u_optimal"]

        # CVX entropy-aware control (uses pre-computed M)
        result_cvx = cvx_controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0, "entropy": 0.1}),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
            M=M,
        )

        assert result_cvx["status"] in ["optimal", "optimal_inaccurate"]

        # Compare solutions - should match now that both use the same entropy metric
        u_cvx = result_cvx["u_optimal"]
        npt.assert_allclose(u_analytical, u_cvx, rtol=1e-3)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def simple_system(self):
        """Create simple system."""
        network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])
        initial_concentrations = np.array([1.0, 1.0])

        dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
        solver = LLRQSolver(dynamics)

        return solver

    def test_infeasible_problem(self, simple_system):
        """Test handling of infeasible optimization problems."""
        controller = CVXController(simple_system)

        def infeasible_constraints(variables, params):
            """Create infeasible constraints."""
            u = variables["u"]
            return [
                u >= 10.0,
                u <= -10.0,  # Impossible!
            ]

        result = controller.compute_cvx_control(constraints_fn=infeasible_constraints)

        # Should detect infeasibility
        assert result["status"] == "infeasible"
        assert result["u_optimal"] is None

    def test_invalid_objective_callback(self, simple_system):
        """Test handling of invalid objective callbacks."""
        controller = CVXController(simple_system)

        def invalid_objective(variables, params):
            """Objective that will cause an error."""
            raise ValueError("This is a test error")

        with pytest.raises(ValueError, match="This is a test error"):
            controller.compute_cvx_control(objective_fn=invalid_objective)

    def test_missing_cvxpy_in_callback(self, simple_system):
        """Test callback that tries to use cvxpy functions without import."""
        controller = CVXController(simple_system)

        def bad_objective(variables, params):
            """Try to use cp without it being in scope."""
            u = variables["u"]
            return params["missing_cp"].sum_squares(u)

        with pytest.raises(KeyError):
            controller.compute_cvx_control(objective_fn=bad_objective)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__])
