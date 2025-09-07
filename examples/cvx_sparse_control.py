#!/usr/bin/env python3
"""
CVXpy Sparse Control Demonstration.

This example shows how to use cvxpy-based optimization to design sparse control
inputs using L1 regularization. The L1 penalty promotes solutions that use
fewer control inputs, which is useful when actuating many reactions is expensive
or when we want to identify the minimal set of reactions needed for control.

Key features demonstrated:
1. L1-regularized sparse control design
2. Comparison with analytical dense control
3. Trade-off curves between sparsity and tracking performance
4. Custom objective functions with cvxpy
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.control import LLRQController

try:
    from llrq.cvx_control import CVXController, CVXObjectives, CVXConstraints

    CVXPY_AVAILABLE = True
except ImportError as e:
    CVXPY_AVAILABLE = False
    print(f"CVXpy not available: {e}")
    print("Install with: pip install 'llrq[cvx]' or pip install cvxpy>=1.7.2")


def demo_sparse_control():
    """Demonstrate sparse control design using L1 regularization."""
    if not CVXPY_AVAILABLE:
        print("This example requires cvxpy. Skipping demonstration.")
        return

    print("=== CVXpy Sparse Control Demonstration ===\n")

    # Create a larger reaction network: A ⇌ B ⇌ C ⇌ D
    network = ReactionNetwork(
        species_ids=["A", "B", "C", "D"],
        reaction_ids=["R1", "R2", "R3"],
        stoichiometric_matrix=[
            [-1, 0, 0],  # A
            [1, -1, 0],  # B
            [0, 1, -1],  # C
            [0, 0, 1],  # D
        ],
    )

    # Set up mass action kinetics with different rates
    forward_rates = np.array([3.0, 1.5, 2.0])
    backward_rates = np.array([1.0, 0.5, 1.5])
    initial_concentrations = np.array([2.0, 1.0, 0.5, 0.2])

    print(f"Network: A ⇌ B ⇌ C ⇌ D")
    print(f"Forward rates: {forward_rates}")
    print(f"Backward rates: {backward_rates}")
    print(f"Initial concentrations: {initial_concentrations}")

    # Create LLRQ dynamics and controllers
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)

    # Create both analytical and CVX controllers (control all reactions)
    analytical_controller = LLRQController(solver, controlled_reactions=None)
    cvx_controller = CVXController(solver, controlled_reactions=None)

    print(f"System dimensions: {solver._rankS} states, {len(network.reaction_ids)} reactions")
    print(f"Relaxation matrix K:\n{dynamics.K}")

    # Define target state in reduced coordinates (easier to work with)
    y_target = np.array([0.5, -0.3, 0.2])  # Target reduced state
    print(f"Target reduced state: {y_target}")

    # Compute corresponding target forces for CVX comparison
    x_target = solver._B @ y_target  # x = B @ y in reduced space
    print(f"Corresponding target forces: {x_target}")

    # 1. Analytical dense control (for comparison)
    print(f"\n=== Analytical Dense Control ===")
    u_analytical = analytical_controller.compute_steady_state_control(y_target)
    print(f"Analytical control: {u_analytical}")
    print(f"Control sparsity (L0): {np.sum(np.abs(u_analytical) > 1e-6)}/3")
    print(f"Control effort (L2): {np.linalg.norm(u_analytical):.4f}")
    print(f"Control effort (L1): {np.linalg.norm(u_analytical, 1):.4f}")

    # Verify achieved state
    x_achieved_analytical = np.linalg.solve(dynamics.K, u_analytical)
    tracking_error_analytical = np.linalg.norm(x_achieved_analytical - x_target) ** 2
    print(f"Achieved state: {x_achieved_analytical}")
    print(f"Tracking error: {tracking_error_analytical:.6f}")

    # 2. Sparse control with different sparsity weights
    print(f"\n=== Sparse Control with Different Weights ===")
    sparsity_weights = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    sparse_results = []

    print(f"{'Sparsity λ':>10} {'L0 Sparsity':>12} {'L1 Norm':>10} {'L2 Norm':>10} {'Track Error':>12}")
    print("-" * 66)

    for lam in sparsity_weights:
        # Use pre-built sparse objective
        result = cvx_controller.compute_cvx_control(
            objective_fn=CVXObjectives.sparse_control(sparsity_weight=lam, tracking_weight=1.0),
            constraints_fn=CVXConstraints.steady_state(),
            x_target=x_target,
        )

        if result["status"] in ["optimal", "optimal_inaccurate"]:
            u_sparse = result["u_optimal"]
            x_achieved = result["variables"]["x"].value

            # Compute metrics
            l0_sparsity = np.sum(np.abs(u_sparse) > 1e-4)
            l1_norm = np.linalg.norm(u_sparse, 1)
            l2_norm = np.linalg.norm(u_sparse)
            tracking_error = np.linalg.norm(x_achieved - x_target) ** 2

            sparse_results.append(
                {
                    "lambda": lam,
                    "u_optimal": u_sparse,
                    "x_achieved": x_achieved,
                    "l0_sparsity": l0_sparsity,
                    "l1_norm": l1_norm,
                    "l2_norm": l2_norm,
                    "tracking_error": tracking_error,
                    "objective_value": result["objective_value"],
                }
            )

            print(f"{lam:10.1f} {l0_sparsity:12.0f} {l1_norm:10.4f} {l2_norm:10.4f} {tracking_error:12.6f}")
        else:
            print(f"{lam:10.1f} {'FAILED':>12} {'':>10} {'':>10} {'':>12}")

    # 3. Custom objective: minimize L1 norm subject to tracking constraint
    print(f"\n=== Custom Objective: Exact Tracking with Minimal L1 Norm ===")

    def minimal_l1_objective(variables, params):
        """Custom objective: minimize L1 norm of control."""
        return params["cvxpy"].norm(variables["u"], 1)

    def exact_tracking_constraints(variables, params):
        """Custom constraints: exact tracking + steady state."""
        constraints = CVXConstraints.steady_state()(variables, params)

        # Add exact tracking constraint
        x = variables["x"]
        x_target = params["x_target"]
        tolerance = params.get("tracking_tolerance", 1e-6)
        constraints.append(params["cvxpy"].norm(x - x_target, 2) <= tolerance)

        return constraints

    # We need to pass cvxpy module to use in constraints
    import cvxpy as cp

    result = cvx_controller.compute_cvx_control(
        objective_fn=minimal_l1_objective,
        constraints_fn=exact_tracking_constraints,
        x_target=x_target,
        cvxpy=cp,
        tracking_tolerance=1e-8,
    )

    if result["status"] in ["optimal", "optimal_inaccurate"]:
        u_minimal = result["u_optimal"]
        x_achieved = result["variables"]["x"].value

        print(f"Minimal L1 control: {u_minimal}")
        print(f"L0 sparsity: {np.sum(np.abs(u_minimal) > 1e-4)}/3")
        print(f"L1 norm: {np.linalg.norm(u_minimal, 1):.4f}")
        print(f"Achieved state: {x_achieved}")
        print(f"Tracking error: {np.linalg.norm(x_achieved - x_target):.8f}")
    else:
        print(f"Optimization failed with status: {result['status']}")

    # 4. Plotting results
    if sparse_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Sparse Control Trade-offs", fontsize=14)

        lambdas = [r["lambda"] for r in sparse_results]
        l0_sparsities = [r["l0_sparsity"] for r in sparse_results]
        l1_norms = [r["l1_norm"] for r in sparse_results]
        tracking_errors = [r["tracking_error"] for r in sparse_results]

        # Plot 1: Sparsity vs regularization weight
        ax = axes[0, 0]
        ax.plot(lambdas, l0_sparsities, "bo-", label="L0 sparsity")
        ax.axhline(np.sum(np.abs(u_analytical) > 1e-6), color="r", linestyle="--", label="Analytical")
        ax.set_xlabel("Sparsity weight λ")
        ax.set_ylabel("Number of active controls")
        ax.set_title("Control Sparsity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: L1 norm vs regularization weight
        ax = axes[0, 1]
        ax.plot(lambdas, l1_norms, "go-", label="Sparse control")
        ax.axhline(np.linalg.norm(u_analytical, 1), color="r", linestyle="--", label="Analytical")
        ax.set_xlabel("Sparsity weight λ")
        ax.set_ylabel("L1 norm")
        ax.set_title("Control L1 Norm")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Tracking error vs regularization weight
        ax = axes[1, 0]
        ax.semilogy(lambdas, tracking_errors, "ro-", label="Sparse control")
        ax.axhline(tracking_error_analytical, color="r", linestyle="--", label="Analytical")
        ax.set_xlabel("Sparsity weight λ")
        ax.set_ylabel("Tracking error")
        ax.set_title("Tracking Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Control inputs for different sparsity levels
        ax = axes[1, 1]
        for i, result in enumerate([sparse_results[0], sparse_results[2], sparse_results[-1]]):
            lam = result["lambda"]
            u = result["u_optimal"]
            ax.bar(np.arange(3) + i * 0.25, u, 0.25, label=f"λ = {lam}")

        ax.bar(np.arange(3) + len([0, 2, -1]) * 0.25, u_analytical, 0.25, label="Analytical", alpha=0.7, color="red")
        ax.set_xlabel("Reaction index")
        ax.set_ylabel("Control input")
        ax.set_title("Control Input Patterns")
        ax.set_xticks(np.arange(3) + 0.375)
        ax.set_xticklabels(["R1", "R2", "R3"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print(f"\n=== Summary ===")
    print("• L1 regularization successfully promotes sparse control solutions")
    print("• Higher sparsity weights reduce the number of active control inputs")
    print("• Trade-off between sparsity and tracking accuracy is clearly visible")
    print("• Custom objectives and constraints provide maximum flexibility")


if __name__ == "__main__":
    demo_sparse_control()
