#!/usr/bin/env python3
"""
CVXpy Constrained Control Demonstration.

This example demonstrates how to solve control problems with various constraints
using cvxpy optimization. Real control systems often have physical limits on
actuator outputs, safety constraints on states, or resource budgets.

Key features demonstrated:
1. Box constraints on control inputs (min/max bounds)
2. Control budget constraints (L1 and L2 norms)
3. State constraints (safety bounds)
4. Multi-objective optimization with constraints
5. Feasibility analysis and constraint violation handling
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.control import LLRQController
from llrq.cvx_control import CVXController, CVXObjectives, CVXConstraints
import cvxpy as cp


def demo_constrained_control():
    """Demonstrate various types of constrained control problems."""
    print("=== CVXpy Constrained Control Demonstration ===\n")

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
    forward_rates = np.array([2.5, 1.8])
    backward_rates = np.array([1.2, 0.9])
    initial_concentrations = np.array([2.0, 1.0, 0.5])

    print(f"Network: A ⇌ B ⇌ C")
    print(f"Forward rates: {forward_rates}")
    print(f"Backward rates: {backward_rates}")
    print(f"Initial concentrations: {initial_concentrations}")

    # Create LLRQ dynamics and controllers
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)

    analytical_controller = LLRQController(solver, controlled_reactions=None)
    cvx_controller = CVXController(solver, controlled_reactions=None)

    print(f"System dimensions: {solver._rankS} states, {len(network.reaction_ids)} reactions")

    # Define challenging target state
    x_target = np.array([1.2, -0.8])  # Aggressive target requiring large control
    print(f"Target reaction forces: {x_target}")

    # Unconstrained solution for comparison
    print(f"\n=== Unconstrained Control (Baseline) ===")
    u_unconstrained = analytical_controller.compute_steady_state_control(solver._B.T @ x_target)
    print(f"Unconstrained control: {u_unconstrained}")
    print(f"Control range: [{u_unconstrained.min():.3f}, {u_unconstrained.max():.3f}]")
    print(f"L1 norm: {np.linalg.norm(u_unconstrained, 1):.4f}")
    print(f"L2 norm: {np.linalg.norm(u_unconstrained):.4f}")

    # 1. Box constraints
    print(f"\n=== Box Constrained Control ===")
    u_bounds = [-2.0, 3.0]  # Asymmetric bounds to make it interesting
    print(f"Control bounds: [{u_bounds[0]}, {u_bounds[1]}]")

    result_box = cvx_controller.compute_cvx_control(
        objective_fn=CVXObjectives.multi_objective({"tracking": 1.0, "control": 0.01}),
        constraints_fn=CVXConstraints.combine(
            CVXConstraints.steady_state(), CVXConstraints.box_bounds(u_min=u_bounds[0], u_max=u_bounds[1])
        ),
        x_target=x_target,
    )

    if result_box["status"] in ["optimal", "optimal_inaccurate"]:
        u_box = result_box["u_optimal"]
        x_achieved_box = result_box["variables"]["x"].value

        print(f"Box-constrained control: {u_box}")
        print(f"Control range: [{u_box.min():.3f}, {u_box.max():.3f}]")
        print(f"Achieved state: {x_achieved_box}")
        print(f"Tracking error: {np.linalg.norm(x_achieved_box - x_target):.6f}")
        print(f"Bound violations: {np.sum((u_box < u_bounds[0]) | (u_box > u_bounds[1]))}")
    else:
        print(f"Box-constrained optimization failed: {result_box['status']}")

    # 2. L1 budget constraint
    print(f"\n=== L1 Budget Constrained Control ===")
    l1_budget = 3.0
    print(f"L1 budget: {l1_budget}")

    result_l1 = cvx_controller.compute_cvx_control(
        objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
        constraints_fn=CVXConstraints.combine(
            CVXConstraints.steady_state(), CVXConstraints.control_budget(l1_budget, norm_type=1)
        ),
        x_target=x_target,
    )

    if result_l1["status"] in ["optimal", "optimal_inaccurate"]:
        u_l1 = result_l1["u_optimal"]
        x_achieved_l1 = result_l1["variables"]["x"].value

        print(f"L1-budget control: {u_l1}")
        print(f"L1 norm: {np.linalg.norm(u_l1, 1):.4f} (budget: {l1_budget})")
        print(f"Achieved state: {x_achieved_l1}")
        print(f"Tracking error: {np.linalg.norm(x_achieved_l1 - x_target):.6f}")
    else:
        print(f"L1-budget optimization failed: {result_l1['status']}")

    # 3. L2 budget constraint
    print(f"\n=== L2 Budget Constrained Control ===")
    l2_budget = 2.0
    print(f"L2 budget: {l2_budget}")

    result_l2 = cvx_controller.compute_cvx_control(
        objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
        constraints_fn=CVXConstraints.combine(
            CVXConstraints.steady_state(), CVXConstraints.control_budget(l2_budget, norm_type=2)
        ),
        x_target=x_target,
    )

    if result_l2["status"] in ["optimal", "optimal_inaccurate"]:
        u_l2 = result_l2["u_optimal"]
        x_achieved_l2 = result_l2["variables"]["x"].value

        print(f"L2-budget control: {u_l2}")
        print(f"L2 norm: {np.linalg.norm(u_l2):.4f} (budget: {l2_budget})")
        print(f"Achieved state: {x_achieved_l2}")
        print(f"Tracking error: {np.linalg.norm(x_achieved_l2 - x_target):.6f}")
    else:
        print(f"L2-budget optimization failed: {result_l2['status']}")

    # 4. State constraints (safety bounds)
    print(f"\n=== State Constrained Control ===")
    x_bounds = [-1.5, 1.5]  # Symmetric state bounds
    print(f"State bounds: [{x_bounds[0]}, {x_bounds[1]}]")

    result_state = cvx_controller.compute_cvx_control(
        objective_fn=CVXObjectives.multi_objective({"tracking": 1.0, "control": 0.1}),
        constraints_fn=CVXConstraints.combine(
            CVXConstraints.steady_state(), CVXConstraints.state_bounds(x_min=x_bounds[0], x_max=x_bounds[1])
        ),
        x_target=x_target,
    )

    if result_state["status"] in ["optimal", "optimal_inaccurate"]:
        u_state = result_state["u_optimal"]
        x_achieved_state = result_state["variables"]["x"].value

        print(f"State-constrained control: {u_state}")
        print(f"Achieved state: {x_achieved_state}")
        print(f"State range: [{x_achieved_state.min():.3f}, {x_achieved_state.max():.3f}]")
        print(f"Tracking error: {np.linalg.norm(x_achieved_state - x_target):.6f}")
        print(f"State bound violations: {np.sum((x_achieved_state < x_bounds[0]) | (x_achieved_state > x_bounds[1]))}")
    else:
        print(f"State-constrained optimization failed: {result_state['status']}")

    # 5. Multi-constraint problem
    print(f"\n=== Multi-Constraint Problem ===")
    print("Combining box bounds + L1 budget + state bounds")

    def multi_constraints(variables, params):
        """Combine multiple constraint types."""
        constraints = []

        # Steady state
        constraints.extend(CVXConstraints.steady_state()(variables, params))

        # Relaxed box bounds
        constraints.extend(CVXConstraints.box_bounds(u_min=-1.5, u_max=2.5)(variables, params))

        # Relaxed L1 budget
        constraints.extend(CVXConstraints.control_budget(4.0, norm_type=1)(variables, params))

        # Relaxed state bounds
        constraints.extend(CVXConstraints.state_bounds(x_min=-1.0, x_max=1.0)(variables, params))

        return constraints

    result_multi = cvx_controller.compute_cvx_control(
        objective_fn=CVXObjectives.multi_objective({"tracking": 1.0, "control": 0.05}),
        constraints_fn=multi_constraints,
        x_target=x_target,
    )

    if result_multi["status"] in ["optimal", "optimal_inaccurate"]:
        u_multi = result_multi["u_optimal"]
        x_achieved_multi = result_multi["variables"]["x"].value

        print(f"Multi-constrained control: {u_multi}")
        print(f"Control range: [{u_multi.min():.3f}, {u_multi.max():.3f}]")
        print(f"L1 norm: {np.linalg.norm(u_multi, 1):.4f}")
        print(f"Achieved state: {x_achieved_multi}")
        print(f"State range: [{x_achieved_multi.min():.3f}, {x_achieved_multi.max():.3f}]")
        print(f"Tracking error: {np.linalg.norm(x_achieved_multi - x_target):.6f}")
    else:
        print(f"Multi-constrained optimization failed: {result_multi['status']}")

    # 6. Feasibility analysis
    print(f"\n=== Feasibility Analysis ===")
    print("Testing increasingly tight constraints to find feasibility limits")

    # Test different L2 budget levels
    l2_budgets = np.logspace(-1, 1, 10)  # From 0.1 to 10
    feasible_budgets = []
    tracking_errors = []

    for budget in l2_budgets:
        result = cvx_controller.compute_cvx_control(
            objective_fn=CVXObjectives.multi_objective({"tracking": 1.0}),
            constraints_fn=CVXConstraints.combine(
                CVXConstraints.steady_state(), CVXConstraints.control_budget(budget, norm_type=2)
            ),
            x_target=x_target,
            solver_options={"verbose": False},
        )

        if result["status"] in ["optimal", "optimal_inaccurate"]:
            feasible_budgets.append(budget)
            x_achieved = result["variables"]["x"].value
            tracking_errors.append(np.linalg.norm(x_achieved - x_target))

    print(f"Feasible L2 budgets: {len(feasible_budgets)}/10")
    if feasible_budgets:
        print(f"Minimum feasible budget: {min(feasible_budgets):.4f}")
        print(f"Best tracking error: {min(tracking_errors):.6f}")

    # Plotting results
    results_to_plot = []
    labels = []

    if "u_unconstrained" in locals():
        results_to_plot.append(u_unconstrained)
        labels.append("Unconstrained")

    if result_box["status"] in ["optimal", "optimal_inaccurate"]:
        results_to_plot.append(u_box)
        labels.append("Box constrained")

    if result_l1["status"] in ["optimal", "optimal_inaccurate"]:
        results_to_plot.append(u_l1)
        labels.append("L1 budget")

    if result_l2["status"] in ["optimal", "optimal_inaccurate"]:
        results_to_plot.append(u_l2)
        labels.append("L2 budget")

    if result_multi["status"] in ["optimal", "optimal_inaccurate"]:
        results_to_plot.append(u_multi)
        labels.append("Multi-constraint")

    if results_to_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Constrained Control Comparison", fontsize=14)

        # Plot 1: Control input patterns
        ax = axes[0, 0]
        x_pos = np.arange(len(network.reaction_ids))
        width = 0.15

        for i, (u, label) in enumerate(zip(results_to_plot, labels)):
            ax.bar(x_pos + i * width, u, width, label=label)

        ax.set_xlabel("Reaction")
        ax.set_ylabel("Control input")
        ax.set_title("Control Input Patterns")
        ax.set_xticks(x_pos + width * (len(results_to_plot) - 1) / 2)
        ax.set_xticklabels(network.reaction_ids)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Control norms
        ax = axes[0, 1]
        l1_norms = [np.linalg.norm(u, 1) for u in results_to_plot]
        l2_norms = [np.linalg.norm(u, 2) for u in results_to_plot]

        x_pos = np.arange(len(labels))
        ax.bar(x_pos - 0.2, l1_norms, 0.4, label="L1 norm", alpha=0.7)
        ax.bar(x_pos + 0.2, l2_norms, 0.4, label="L2 norm", alpha=0.7)

        ax.set_xlabel("Control method")
        ax.set_ylabel("Control norm")
        ax.set_title("Control Effort Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Feasibility analysis
        ax = axes[1, 0]
        if feasible_budgets and tracking_errors:
            ax.semilogx(feasible_budgets, tracking_errors, "bo-")
            ax.set_xlabel("L2 budget")
            ax.set_ylabel("Tracking error")
            ax.set_title("Tracking vs Budget Trade-off")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Feasibility analysis\nnot available", ha="center", va="center", transform=ax.transAxes)

        # Plot 4: Constraint satisfaction
        ax = axes[1, 1]
        constraint_types = ["Box", "L1 budget", "L2 budget", "State bounds"]
        satisfaction = []

        for i, result in enumerate([result_box, result_l1, result_l2, result_state]):
            if result["status"] in ["optimal", "optimal_inaccurate"]:
                satisfaction.append(1)  # Satisfied
            elif result["status"] == "infeasible":
                satisfaction.append(0)  # Infeasible
            else:
                satisfaction.append(0.5)  # Unknown/failed

        colors = ["green" if s == 1 else "red" if s == 0 else "orange" for s in satisfaction]
        ax.bar(constraint_types, satisfaction, color=colors, alpha=0.7)
        ax.set_ylabel("Constraint satisfaction")
        ax.set_title("Constraint Feasibility")
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plt.show()

    print(f"\n=== Summary ===")
    print("• Box constraints limit individual control inputs")
    print("• Budget constraints limit total control effort")
    print("• State constraints ensure safety bounds")
    print("• Multiple constraints can be combined")
    print("• Feasibility analysis helps identify constraint limits")
    print("• CVXpy provides flexible framework for custom constraints")


if __name__ == "__main__":
    demo_constrained_control()
