"""CVXpy-based control optimization for LLRQ systems.

This module provides a flexible interface for solving custom control optimization
problems using cvxpy. It supports both pre-built templates for common problems
and fully custom objective functions and constraints.

The key design principle is to provide callback functions for objectives and
constraints that receive cvxpy variables and return cvxpy expressions/constraints.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import cvxpy as cp

from .control import LLRQController


class CVXController(LLRQController):
    """LLRQ controller with cvxpy-based optimization capabilities.

    Extends the analytical LLRQController with the ability to solve custom
    optimization problems for control input design using cvxpy.

    The key method is compute_cvx_control() which accepts callback functions
    for defining objectives and constraints in terms of cvxpy variables.
    """

    def __init__(self, solver, controlled_reactions: Optional[list] = None):
        """Initialize CVX controller.

        Args:
            solver: LLRQSolver with computed basis matrices
            controlled_reactions: List of reaction IDs or indices to control.
                                If None, controls all reactions.
        """
        super().__init__(solver, controlled_reactions)

    def compute_cvx_control(
        self,
        objective_fn: Optional[Callable] = None,
        constraints_fn: Optional[Callable] = None,
        x_target: Optional[np.ndarray] = None,
        y_target: Optional[np.ndarray] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        **problem_params,
    ) -> Dict[str, Any]:
        """Solve a custom cvxpy optimization problem for control input.

        Args:
            objective_fn: Function that takes (variables, params) and returns
                         a cvxpy expression to minimize. If None, uses default.
            constraints_fn: Function that takes (variables, params) and returns
                           a list of cvxpy constraints. If None, no constraints.
            x_target: Target reaction force state (r,)
            y_target: Target reduced state (rankS,)
            solver_options: Additional options to pass to cvxpy solver
            **problem_params: Additional parameters passed to objective/constraint functions

        Returns:
            Dictionary containing:
            - u_optimal: Optimal control input
            - problem: The cvxpy Problem object
            - variables: Dictionary of cvxpy variables used
            - objective_value: Optimal objective value
            - status: Solver status string
        """
        # Determine problem dimensions
        r = len(self.network.reaction_ids)
        m = len(self.controlled_reactions)

        # Create cvxpy variables based on problem structure
        variables = {}

        # Always create control variable
        variables["u_controlled"] = cp.Variable(m, name="u_controlled")

        # Create full control variable for convenience
        variables["u"] = self.G @ variables["u_controlled"]

        # Create state and target variables
        variables["x"] = cp.Variable(r, name="x")
        variables["y"] = cp.Variable(self.rankS, name="y")

        # Build parameter dictionary
        params = {
            "controller": self,
            "solver": self.solver,
            "x_target": x_target,
            "y_target": y_target,
            "G": self.G,
            "B": self.B,
            "A": self.A,
            "K": self.solver.dynamics.K,
            "rankS": self.rankS,
            **problem_params,
        }

        # Build objective
        if objective_fn is not None:
            objective = objective_fn(variables, params)
        else:
            objective = self._default_objective(variables, params)

        # Build constraints
        constraints = []
        if constraints_fn is not None:
            constraints.extend(constraints_fn(variables, params))

        # Create and solve problem
        problem = cp.Problem(cp.Minimize(objective), constraints)

        solver_opts = solver_options or {}
        problem.solve(**solver_opts)

        # Extract results
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            warnings.warn(f"CVX problem status: {problem.status}")

        # Get optimal control (map from controlled to full space)
        if variables["u_controlled"].value is not None:
            u_optimal = self.G @ variables["u_controlled"].value
        else:
            u_optimal = None

        return {
            "u_optimal": u_optimal,
            "u_controlled": variables["u_controlled"].value,
            "problem": problem,
            "variables": variables,
            "objective_value": problem.value,
            "status": problem.status,
            "params": params,
        }

    def _default_objective(self, variables, params):
        """Default objective: minimize control effort or track target."""
        u = variables["u"]

        if "x_target" in params and params["x_target"] is not None and "x" in variables:
            # Track target state
            return cp.sum_squares(variables["x"] - params["x_target"])
        elif "y_target" in params and params["y_target"] is not None and "y" in variables:
            # Track target reduced state
            return cp.sum_squares(variables["y"] - params["y_target"])
        else:
            # Just minimize control effort
            return cp.sum_squares(u)


class CVXObjectives:
    """Pre-built objective function templates for common control problems."""

    @staticmethod
    def sparse_control(sparsity_weight: float = 1.0, tracking_weight: float = 1.0):
        """L1-regularized objective for sparse control.

        Args:
            sparsity_weight: Weight on L1 penalty for control sparsity
            tracking_weight: Weight on tracking error (if target provided)

        Returns:
            Objective function compatible with compute_cvx_control
        """

        def objective(variables, params):
            u = variables["u"]
            obj = sparsity_weight * cp.norm(u, 1)

            # Add tracking term if target available
            if "x_target" in params and params["x_target"] is not None and "x" in variables:
                obj += tracking_weight * cp.sum_squares(variables["x"] - params["x_target"])
            elif "y_target" in params and params["y_target"] is not None and "y" in variables:
                obj += tracking_weight * cp.sum_squares(variables["y"] - params["y_target"])

            return obj

        return objective

    @staticmethod
    def multi_objective(weights: Dict[str, float]):
        """Multi-objective optimization with customizable weights.

        Args:
            weights: Dictionary with keys:
                - 'tracking': Weight on tracking error
                - 'control': Weight on control effort (L2)
                - 'entropy': Weight on entropy production (requires M matrix)
                - 'sparsity': Weight on control sparsity (L1)

        Returns:
            Objective function compatible with compute_cvx_control
        """

        def objective(variables, params):
            u = variables["u"]
            obj = 0

            # Tracking term
            if "tracking" in weights and weights["tracking"] > 0:
                if "x_target" in params and params["x_target"] is not None and "x" in variables:
                    obj += weights["tracking"] * cp.sum_squares(variables["x"] - params["x_target"])
                elif "y_target" in params and params["y_target"] is not None and "y" in variables:
                    obj += weights["tracking"] * cp.sum_squares(variables["y"] - params["y_target"])

            # Control effort term (L2)
            if "control" in weights and weights["control"] > 0:
                obj += weights["control"] * cp.sum_squares(u)

            # Control sparsity term (L1)
            if "sparsity" in weights and weights["sparsity"] > 0:
                obj += weights["sparsity"] * cp.norm(u, 1)

            # Entropy production term
            if "entropy" in weights and weights["entropy"] > 0 and "M" in params:
                M = params["M"]
                # Formulate entropy cost explicitly in terms of u = K x when x is present
                # This avoids any ambiguity and matches the analytical formulation.
                if "x" in variables and "K" in params:
                    u_for_entropy = params["K"] @ variables["x"]
                else:
                    u_for_entropy = u
                obj += weights["entropy"] * cp.quad_form(u_for_entropy, M)

            return obj

        return objective

    @staticmethod
    def robust_tracking(uncertainty_weight: float = 0.1):
        """Robust tracking objective that penalizes sensitivity to uncertainty.

        Args:
            uncertainty_weight: Weight on robustness term

        Returns:
            Objective function compatible with compute_cvx_control
        """

        def objective(variables, params):
            if "x_target" not in params or "x" not in variables:
                raise ValueError("Robust tracking requires x_target and x variables")

            x = variables["x"]
            x_target = params["x_target"]

            # Primary tracking objective
            tracking_error = cp.sum_squares(x - x_target)

            # Robustness term: penalize large state values that are sensitive to perturbations
            robustness_penalty = uncertainty_weight * cp.sum_squares(x)

            return tracking_error + robustness_penalty

        return objective


class CVXConstraints:
    """Pre-built constraint function templates for common control problems."""

    @staticmethod
    def steady_state():
        """Steady-state constraint: K*x = u (or equivalently x = K^{-1}*u).

        Returns:
            Constraints function compatible with compute_cvx_control
        """

        def constraints(variables, params):
            if "x" not in variables:
                return []

            u = variables["u"]
            x = variables["x"]
            K = params["K"]

            return [K @ x == u]

        return constraints

    @staticmethod
    def box_bounds(u_min: Optional[Union[float, np.ndarray]] = None, u_max: Optional[Union[float, np.ndarray]] = None):
        """Box constraints on control inputs.

        Args:
            u_min: Minimum control values (scalar or array)
            u_max: Maximum control values (scalar or array)

        Returns:
            Constraints function compatible with compute_cvx_control
        """

        def constraints(variables, params):
            u = variables["u"]
            cons = []

            if u_min is not None:
                cons.append(u >= u_min)
            if u_max is not None:
                cons.append(u <= u_max)

            return cons

        return constraints

    @staticmethod
    def control_budget(total_budget: float, norm_type: int = 1):
        """Total control budget constraint.

        Args:
            total_budget: Maximum total control effort
            norm_type: 1 for L1 norm, 2 for L2 norm

        Returns:
            Constraints function compatible with compute_cvx_control
        """

        def constraints(variables, params):
            u = variables["u"]

            if norm_type == 1:
                return [cp.norm(u, 1) <= total_budget]
            elif norm_type == 2:
                return [cp.norm(u, 2) <= total_budget]
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

        return constraints

    @staticmethod
    def state_bounds(x_min: Optional[Union[float, np.ndarray]] = None, x_max: Optional[Union[float, np.ndarray]] = None):
        """Bounds on state variables.

        Args:
            x_min: Minimum state values (scalar or array)
            x_max: Maximum state values (scalar or array)

        Returns:
            Constraints function compatible with compute_cvx_control
        """

        def constraints(variables, params):
            if "x" not in variables:
                return []

            x = variables["x"]
            cons = []

            if x_min is not None:
                cons.append(x >= x_min)
            if x_max is not None:
                cons.append(x <= x_max)

            return cons

        return constraints

    @staticmethod
    def combine(*constraint_fns):
        """Combine multiple constraint functions into one.

        Args:
            *constraint_fns: Variable number of constraint functions

        Returns:
            Combined constraints function
        """

        def constraints(variables, params):
            all_constraints = []
            for fn in constraint_fns:
                all_constraints.extend(fn(variables, params))
            return all_constraints

        return constraints


def create_entropy_aware_cvx_controller(solver, controlled_reactions, L):
    """Convenience function to create CVX controller with entropy metric pre-computed.

    Args:
        solver: LLRQSolver instance
        controlled_reactions: List of controlled reactions
        L: Onsager conductance matrix

    Returns:
        CVXController instance with entropy metric M in default parameters
    """
    controller = CVXController(solver, controlled_reactions)

    # Pre-compute entropy metric
    M = controller.compute_control_entropy_metric(L)

    # Create a wrapper that automatically includes M
    original_compute = controller.compute_cvx_control

    def compute_with_entropy(*args, **kwargs):
        if "M" not in kwargs:
            kwargs["M"] = M
        return original_compute(*args, **kwargs)

    controller.compute_cvx_control = compute_with_entropy
    return controller
