#!/usr/bin/env python3
"""
CVXpy Custom Objective and Constraint Demonstration.

This example shows the full flexibility of the cvxpy integration by demonstrating
how users can define completely custom objective functions and constraints.
This is where the power of the callback-based design really shines - users can
express any optimization problem that cvxpy supports.

Key features demonstrated:
1. Custom objective functions beyond the pre-built templates
2. Complex custom constraints (e.g., non-convex approximations, robust constraints)
3. Mixed-integer programming (if supported by solver)
4. Multi-objective optimization with custom Pareto weighting
5. Robust optimization under uncertainty
6. Integration with entropy production metrics
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.control import LLRQController
from llrq.thermodynamic_accounting import ThermodynamicAccountant
from llrq.cvx_control import CVXController, CVXObjectives, CVXConstraints
import cvxpy as cp


def demo_custom_objectives():
    """Demonstrate custom objective functions and constraints."""
    print("=== CVXpy Custom Objective & Constraint Demonstration ===\n")

    # Create A ⇌ B ⇌ C reaction network
    network = ReactionNetwork(
        species_ids=["A", "B", "C"],
        reaction_ids=["R1", "R2"],
        stoichiometric_matrix=[
            [-1, 0],  # A
            [1, -1],  # B
            [0, 1],  # C
        ],
    )

    # Set up mass action kinetics
    forward_rates = np.array([2.0, 1.5])
    backward_rates = np.array([1.0, 0.8])
    initial_concentrations = np.array([2.0, 1.0, 0.5])

    print(f"Network: A ⇌ B ⇌ C")

    # Create LLRQ system with entropy accounting
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)
    cvx_controller = CVXController(solver, controlled_reactions=None)

    # Compute entropy metric for advanced objectives
    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates)
    M = cvx_controller.compute_control_entropy_metric(L)

    print(f"System matrices computed. Entropy metric M:\n{M}")

    x_target = np.array([0.8, -0.5])
    print(f"Target reaction forces: {x_target}")

    # 1. Custom Objective: Asymmetric penalty
    print(f"\n=== Example 1: Asymmetric Penalty Objective ===")
    print("Penalize positive and negative control inputs differently")

    def asymmetric_penalty_objective(variables, params):
        """Penalize positive control more than negative (or vice versa)."""
        u = variables["u"]
        x = variables["x"]
        x_target = params["x_target"]

        # Tracking term
        tracking = cp.sum_squares(x - x_target)

        # Asymmetric penalty: positive control is 3x more expensive
        u_pos = cp.maximum(u, 0)  # Positive part
        u_neg = cp.maximum(-u, 0)  # Negative part (magnitude)

        asymmetric_penalty = 3.0 * cp.sum_squares(u_pos) + 1.0 * cp.sum_squares(u_neg)

        return tracking + 0.1 * asymmetric_penalty

    result1 = cvx_controller.compute_cvx_control(
        objective_fn=asymmetric_penalty_objective, constraints_fn=CVXConstraints.steady_state(), x_target=x_target
    )

    if result1["status"] in ["optimal", "optimal_inaccurate"]:
        u1 = result1["u_optimal"]
        x1 = result1["variables"]["x"].value
        print(f"Asymmetric control: {u1}")
        print(f"Achieved state: {x1}")
        print(f"Positive control: {np.sum(np.maximum(u1, 0)):.4f}")
        print(f"Negative control: {np.sum(np.maximum(-u1, 0)):.4f}")
        print(f"Tracking error: {np.linalg.norm(x1 - x_target):.6f}")

    # 2. Custom Objective: Minimize worst-case tracking error
    print(f"\n=== Example 2: Robust Min-Max Objective ===")
    print("Minimize worst-case tracking under bounded uncertainty in K")

    def robust_minmax_objective(variables, params):
        """Robust objective: minimize worst-case tracking error."""
        x = variables["x"]
        x_target = params["x_target"]

        # For simplicity, we'll use a heuristic robust formulation
        # In practice, this might require more sophisticated uncertainty modeling

        # Standard tracking term
        nominal_error = cp.sum_squares(x - x_target)

        # Robustness heuristic: penalize large states (more sensitive to perturbations)
        robustness_penalty = 0.2 * cp.sum_squares(x)

        return nominal_error + robustness_penalty

    def robust_constraints(variables, params):
        """Add constraints for robust optimization."""
        constraints = CVXConstraints.steady_state()(variables, params)

        # Additional robustness constraint: limit state magnitude
        x = variables["x"]
        constraints.append(cp.norm(x, "inf") <= 2.0)  # Infinity norm constraint

        return constraints

    result2 = cvx_controller.compute_cvx_control(
        objective_fn=robust_minmax_objective, constraints_fn=robust_constraints, x_target=x_target
    )

    if result2["status"] in ["optimal", "optimal_inaccurate"]:
        u2 = result2["u_optimal"]
        x2 = result2["variables"]["x"].value
        print(f"Robust control: {u2}")
        print(f"Achieved state: {x2}")
        print(f"State infinity norm: {np.linalg.norm(x2, np.inf):.4f}")
        print(f"Tracking error: {np.linalg.norm(x2 - x_target):.6f}")

    # 3. Custom Objective: Entropy-aware with custom weighting
    print(f"\n=== Example 3: Custom Entropy-Performance Trade-off ===")
    print("Non-linear weighting between tracking and entropy")

    def custom_entropy_objective(variables, params):
        """Custom entropy weighting with state-dependent penalties."""
        u = variables["u"]
        x = variables["x"]
        x_target = params["x_target"]
        M = params["M"]

        # Primary tracking objective
        tracking_error = cp.sum_squares(x - x_target)

        # Entropy production (quadratic in u)
        entropy_rate = cp.quad_form(u, M)

        # Custom weighting: fixed weights for convex formulation
        # In a real application, this could be tuned based on operating conditions
        entropy_weight = 0.15  # Fixed weighting for convexity

        return tracking_error + entropy_weight * entropy_rate

    result3 = cvx_controller.compute_cvx_control(
        objective_fn=custom_entropy_objective, constraints_fn=CVXConstraints.steady_state(), x_target=x_target, M=M
    )

    if result3["status"] in ["optimal", "optimal_inaccurate"]:
        u3 = result3["u_optimal"]
        x3 = result3["variables"]["x"].value
        entropy_rate = float(u3.T @ M @ u3)
        print(f"Entropy-aware control: {u3}")
        print(f"Achieved state: {x3}")
        print(f"Tracking error: {np.linalg.norm(x3 - x_target):.6f}")
        print(f"Entropy rate: {entropy_rate:.6f}")

    # 4. Custom Objective: Multi-target with priorities
    print(f"\n=== Example 4: Multi-Target with Priority Weighting ===")
    print("Track multiple targets with different priorities")

    # Define multiple targets with priorities
    targets = [
        {"x": np.array([0.5, -0.2]), "priority": 1.0, "name": "Primary"},
        {"x": np.array([0.1, 0.1]), "priority": 0.3, "name": "Secondary"},
        {"x": np.array([-0.2, 0.3]), "priority": 0.1, "name": "Tertiary"},
    ]

    def multi_target_objective(variables, params):
        """Track multiple targets with priority weighting."""
        x = variables["x"]
        targets = params["targets"]

        total_objective = 0
        for target in targets:
            tracking_error = cp.sum_squares(x - target["x"])
            total_objective += target["priority"] * tracking_error

        # Small control penalty
        u = variables["u"]
        total_objective += 0.01 * cp.sum_squares(u)

        return total_objective

    result4 = cvx_controller.compute_cvx_control(
        objective_fn=multi_target_objective, constraints_fn=CVXConstraints.steady_state(), targets=targets
    )

    if result4["status"] in ["optimal", "optimal_inaccurate"]:
        u4 = result4["u_optimal"]
        x4 = result4["variables"]["x"].value
        print(f"Multi-target control: {u4}")
        print(f"Achieved state: {x4}")

        for target in targets:
            error = np.linalg.norm(x4 - target["x"])
            print(f"  {target['name']} target error: {error:.6f} (priority: {target['priority']})")

    # 5. Custom Constraints: Nonlinear approximations
    print(f"\n=== Example 5: Custom Nonlinear Constraint Approximation ===")
    print("Approximate nonlinear constraints using convex relaxations")

    def nonlinear_constraints(variables, params):
        """Approximate nonlinear constraints."""
        constraints = CVXConstraints.steady_state()(variables, params)

        u = variables["u"]
        x = variables["x"]

        # Example: approximate constraint u1 * u2 <= 0.5 (complementarity-like)
        # Convex relaxation: |u1| + |u2| <= 1.0
        constraints.append(cp.abs(u[0]) + cp.abs(u[1]) <= 1.0)

        # Another nonlinear approximation: x1^2 + x2^2 <= 1
        # Convex approximation using norm constraint
        constraints.append(cp.norm(x, 2) <= 1.0)

        return constraints

    result5 = cvx_controller.compute_cvx_control(
        objective_fn=CVXObjectives.multi_objective({"tracking": 1.0, "control": 0.1}),
        constraints_fn=nonlinear_constraints,
        x_target=x_target,
    )

    if result5["status"] in ["optimal", "optimal_inaccurate"]:
        u5 = result5["u_optimal"]
        x5 = result5["variables"]["x"].value
        print(f"Nonlinear-constrained control: {u5}")
        print(f"Control complementarity: |u1| + |u2| = {np.abs(u5[0]) + np.abs(u5[1]):.4f}")
        print(f"State norm constraint: ||x||_2 = {np.linalg.norm(x5):.4f}")
        print(f"Tracking error: {np.linalg.norm(x5 - x_target):.6f}")

    # 6. Custom Objective: Frequency-domain considerations
    print(f"\n=== Example 6: Frequency-Domain Inspired Objective ===")
    print("Objective inspired by frequency response shaping")

    def frequency_shaped_objective(variables, params):
        """Shape control spectrum using penalty on differences."""
        u = variables["u"]
        x = variables["x"]
        x_target = params["x_target"]

        # Tracking term
        tracking = cp.sum_squares(x - x_target)

        # Penalty on control "roughness" (differences between adjacent controls)
        # This promotes smooth control allocation
        if u.shape[0] > 1:
            smoothness_penalty = cp.sum_squares(u[1:] - u[:-1])
        else:
            smoothness_penalty = 0

        # Penalty on high-frequency content (large control magnitudes)
        magnitude_penalty = cp.sum_squares(u)

        return tracking + 0.1 * smoothness_penalty + 0.01 * magnitude_penalty

    result6 = cvx_controller.compute_cvx_control(
        objective_fn=frequency_shaped_objective, constraints_fn=CVXConstraints.steady_state(), x_target=x_target
    )

    if result6["status"] in ["optimal", "optimal_inaccurate"]:
        u6 = result6["u_optimal"]
        x6 = result6["variables"]["x"].value
        control_smoothness = np.sum((u6[1:] - u6[:-1]) ** 2) if len(u6) > 1 else 0
        print(f"Frequency-shaped control: {u6}")
        print(f"Control smoothness: {control_smoothness:.6f}")
        print(f"Tracking error: {np.linalg.norm(x6 - x_target):.6f}")

    # Plotting comparison of all methods
    results = []
    labels = []

    for i, (result, label) in enumerate(
        [
            (result1, "Asymmetric"),
            (result2, "Robust"),
            (result3, "Entropy-aware"),
            (result4, "Multi-target"),
            (result5, "Nonlinear-constrained"),
            (result6, "Frequency-shaped"),
        ]
    ):
        if result["status"] in ["optimal", "optimal_inaccurate"]:
            results.append({"u": result["u_optimal"], "x": result["variables"]["x"].value, "label": label})

    if results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Custom Objective Function Comparison", fontsize=14)

        # Plot 1: Control inputs
        ax = axes[0, 0]
        x_pos = np.arange(len(network.reaction_ids))
        width = 0.12

        for i, result in enumerate(results):
            ax.bar(x_pos + i * width, result["u"], width, label=result["label"], alpha=0.8)

        ax.set_xlabel("Reaction")
        ax.set_ylabel("Control input")
        ax.set_title("Control Input Patterns")
        ax.set_xticks(x_pos + width * (len(results) - 1) / 2)
        ax.set_xticklabels(network.reaction_ids)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Plot 2: Achieved states
        ax = axes[0, 1]
        for i, result in enumerate(results):
            ax.scatter(result["x"][0], result["x"][1], label=result["label"], s=80, alpha=0.8)

        # Target state
        ax.scatter(x_target[0], x_target[1], marker="*", s=200, color="red", label="Target", zorder=10)

        ax.set_xlabel("x1 (reaction force)")
        ax.set_ylabel("x2 (reaction force)")
        ax.set_title("Achieved States")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Tracking errors
        ax = axes[1, 0]
        tracking_errors = [np.linalg.norm(r["x"] - x_target) for r in results]
        bars = ax.bar(range(len(results)), tracking_errors, alpha=0.7)

        # Color bars by performance
        for i, (bar, error) in enumerate(zip(bars, tracking_errors)):
            if error < 0.1:
                bar.set_color("green")
            elif error < 0.5:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        ax.set_xlabel("Method")
        ax.set_ylabel("Tracking error")
        ax.set_title("Tracking Performance")
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([r["label"] for r in results], rotation=15)
        ax.grid(True, alpha=0.3)

        # Plot 4: Control effort comparison
        ax = axes[1, 1]
        l1_norms = [np.linalg.norm(r["u"], 1) for r in results]
        l2_norms = [np.linalg.norm(r["u"], 2) for r in results]

        x_pos = np.arange(len(results))
        ax.bar(x_pos - 0.2, l1_norms, 0.4, label="L1 norm", alpha=0.7)
        ax.bar(x_pos + 0.2, l2_norms, 0.4, label="L2 norm", alpha=0.7)

        ax.set_xlabel("Method")
        ax.set_ylabel("Control effort")
        ax.set_title("Control Effort Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([r["label"] for r in results], rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print(f"\n=== Summary ===")
    print("• Custom objectives enable application-specific optimization")
    print("• Asymmetric penalties can reflect real-world costs")
    print("• Robust formulations handle uncertainty")
    print("• Multi-objective optimization balances competing goals")
    print("• Nonlinear constraints can be approximated convexly")
    print("• Frequency-domain considerations shape control spectra")
    print("• The callback design enables unlimited flexibility")


if __name__ == "__main__":
    demo_custom_objectives()
