#!/usr/bin/env python3
"""Demo: K matrix constraint building with physical bounds.

This example demonstrates the core functionality of the simplified k_estimation module:
1. Creating KMatrixEstimator for constraint building
2. Building CVXPY constraints with physical bounds
3. Using constraints in optimization problems
4. Validating K matrices against physical bounds

This focuses only on the constraint-building functionality, not full estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import llrq


def create_example_networks():
    """Create example reaction networks for testing."""

    networks = []

    # 1. Simple reversible reaction: A ⇌ B
    species_1 = ["A", "B"]
    reactions_1 = ["forward", "backward"]
    S_1 = np.array([[-1, 1], [1, -1]])
    c_eq_1 = np.array([2.0, 1.0])
    networks.append(("A ⇌ B", llrq.ReactionNetwork(species_1, reactions_1, S_1), c_eq_1))

    # 2. Linear pathway: A → B → C
    species_2 = ["A", "B", "C"]
    reactions_2 = ["R1", "R2"]
    S_2 = np.array([[-1, 0], [1, -1], [0, 1]])
    c_eq_2 = np.array([3.0, 1.5, 0.5])
    networks.append(("A → B → C", llrq.ReactionNetwork(species_2, reactions_2, S_2), c_eq_2))

    return networks


def demonstrate_constraint_building():
    """Demonstrate building CVXPY constraints with physical bounds."""

    print("=== K Matrix Constraint Building Demo ===\n")

    # Create a simple network
    print("1. Setting up reaction network A ⇌ B...")
    species = ["A", "B"]
    reactions = ["forward", "backward"]
    S = np.array([[-1, 1], [1, -1]])  # A ⇌ B

    network = llrq.ReactionNetwork(species, reactions, S)
    c_eq = np.array([2.0, 1.0])  # Equilibrium concentrations

    print(f"   Species: {network.species_ids}")
    print(f"   Reactions: {network.reaction_ids}")
    print(f"   Stoichiometry matrix:\n{network.S}")

    # Create estimator
    print("\n2. Creating KMatrixEstimator...")
    estimator = llrq.KMatrixEstimator(network, c_eq)

    print(f"   A matrix (stoichiometric):\n{estimator.A}")
    print(f"   A eigenvalues: {np.linalg.eigvals(estimator.A)}")

    # Define physical bounds
    print("\n3. Setting up physical bounds...")

    # Example bounds from enzyme kinetics
    enzyme_concs = np.array([1e-6, 1e-6])  # 1 μM enzymes
    kcat_values = np.array([100.0, 80.0])  # s^-1
    km_values = np.array([10e-6, 15e-6])  # 10-15 μM

    bounds_data = llrq.enzyme_to_conductance(enzyme_concs, kcat_values, km_values, c_eq, network.S)

    g_diag_min = bounds_data["g_diag_min"]
    g_diag_max = bounds_data["g_diag_max"]

    spectral_bounds = llrq.compute_spectral_caps((g_diag_min, g_diag_max))
    gamma_min = spectral_bounds["gamma_min"]
    gamma_max = spectral_bounds["gamma_max"]

    print(f"   G diagonal bounds: [{g_diag_min}, {g_diag_max}]")
    print(f"   G spectral bounds: [{gamma_min:.3f}, {gamma_max:.3f}]")

    # Build constraints
    print("\n4. Building CVXPY constraints...")

    K = cp.Variable((2, 2))  # Decision variable

    constraints, W = estimator.build_k_bounds_constraints(
        K, g_diag_min=g_diag_min, g_diag_max=g_diag_max, gamma_min=gamma_min, gamma_max=gamma_max
    )

    print(f"   Built {len(constraints)} constraints")
    print(f"   W variable shape: {W.shape}, PSD: {W.is_psd()}")

    return estimator, constraints, W, K, (g_diag_min, g_diag_max, gamma_min, gamma_max)


def solve_constrained_optimization(estimator, constraints, W, K, bounds):
    """Solve optimization problems with physical bounds."""

    g_diag_min, g_diag_max, gamma_min, gamma_max = bounds

    print("\n5. Solving constrained optimization problems...")

    # Problem 1: Minimize Frobenius norm (find smallest feasible K)
    print("\n   Problem 1: Minimize ||K||_F subject to physical bounds")

    objective1 = cp.Minimize(cp.norm(K, "fro"))
    problem1 = cp.Problem(objective1, constraints)

    problem1.solve(verbose=False)
    print(f"   Status: {problem1.status}")
    print(f"   Optimal value: {problem1.value:.6f}")

    if problem1.status in ["optimal", "optimal_inaccurate"]:
        K_min = K.value
        print(f"   Minimal K matrix:\n{K_min}")

        # Validate solution
        G_min = estimator.A_invsqrt @ W.value @ estimator.A_invsqrt
        print(f"   Corresponding G diagonal: {np.diag(G_min)}")

    # Problem 2: Fit to a target matrix
    print("\n   Problem 2: Fit K to target matrix with physical bounds")

    # Target: diagonal matrix
    K_target = np.array([[3.0, 0.0], [0.0, 2.0]])

    objective2 = cp.Minimize(cp.norm(K - K_target, "fro"))
    problem2 = cp.Problem(objective2, constraints)

    problem2.solve(verbose=False)
    print(f"   Status: {problem2.status}")
    print(f"   Optimal value: {problem2.value:.6f}")

    if problem2.status in ["optimal", "optimal_inaccurate"]:
        K_fit = K.value
        print(f"   Fitted K matrix:\n{K_fit}")
        print(f"   Target K matrix:\n{K_target}")

        error = np.linalg.norm(K_fit - K_target, "fro") / np.linalg.norm(K_target, "fro")
        print(f"   Relative error: {error:.3f}")

    return {
        "minimal": K_min if problem1.status in ["optimal", "optimal_inaccurate"] else None,
        "fitted": K_fit if problem2.status in ["optimal", "optimal_inaccurate"] else None,
        "target": K_target,
    }


def validate_solutions(estimator, solutions, bounds):
    """Validate that solutions satisfy physical bounds."""

    g_diag_min, g_diag_max, gamma_min, gamma_max = bounds

    print("\n6. Validating solutions against physical bounds...")

    for name, K_matrix in solutions.items():
        if K_matrix is None:
            continue

        print(f"\n   Validating {name} K matrix:")

        # Use physical bounds validation
        validation = llrq.validate_physical_consistency(
            K_matrix, estimator.A, (g_diag_min, g_diag_max), {"gamma_min": gamma_min, "gamma_max": gamma_max}
        )

        print(f"     Physically consistent: {validation['is_physically_consistent']}")

        if not validation["is_physically_consistent"]:
            print("     Violations:")
            for violation_type, magnitude in validation["violations"].items():
                print(f"       {violation_type}: {magnitude:.6f}")

        # Show G matrix properties
        G = validation["G_matrix"]
        G_eigenvals = validation["G_eigenvalues"]

        print(f"     G diagonal: {np.diag(G)}")
        print(f"     G eigenvalues: {G_eigenvals.real}")

        # Compute timescales
        timescales = llrq.estimate_reaction_timescales(K_matrix)
        finite_timescales = timescales["timescales"][np.isfinite(timescales["timescales"])]

        if len(finite_timescales) > 0:
            print(f"     Relaxation timescales: {finite_timescales}")
            print(f"     Fastest/slowest: {np.min(finite_timescales):.3f} / {np.max(finite_timescales):.3f}")


def create_visualization(estimator, solutions):
    """Create visualization of results."""

    print("\n7. Creating visualization...")

    # Filter out None solutions
    valid_solutions = {name: K for name, K in solutions.items() if K is not None}

    if len(valid_solutions) < 2:
        print("   Not enough valid solutions for visualization")
        return

    fig, axes = plt.subplots(1, len(valid_solutions), figsize=(5 * len(valid_solutions), 4))

    if len(valid_solutions) == 1:
        axes = [axes]

    # Find global color scale
    all_values = np.concatenate([K.flatten() for K in valid_solutions.values()])
    vmin, vmax = np.min(all_values), np.max(all_values)

    for i, (name, K_matrix) in enumerate(valid_solutions.items()):
        ax = axes[i]

        im = ax.imshow(K_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name.title()} K Matrix")
        ax.set_xlabel("Reaction")
        ax.set_ylabel("Reaction")

        # Add text values
        for row in range(K_matrix.shape[0]):
            for col in range(K_matrix.shape[1]):
                ax.text(col, row, f"{K_matrix[row,col]:.2f}", ha="center", va="center", fontsize=12, weight="bold")

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("/tmp/k_constraints_demo.png", dpi=150, bbox_inches="tight")
    print("   Visualization saved to /tmp/k_constraints_demo.png")

    return fig


def demonstrate_multiple_networks():
    """Demonstrate constraint building on different network topologies."""

    print("\n8. Testing multiple network topologies...")

    networks = create_example_networks()

    for network_name, network, c_eq in networks:
        print(f"\n   Network: {network_name}")
        print(f"   Size: {network.n_species} species, {network.n_reactions} reactions")

        try:
            # Create estimator
            estimator = llrq.KMatrixEstimator(network, c_eq)

            # Simple bounds
            n_reactions = network.n_reactions
            g_diag_max = np.full(n_reactions, 5.0)  # Conservative bounds

            # Test constraint building
            K = cp.Variable((n_reactions, n_reactions))
            constraints, W = estimator.build_k_bounds_constraints(K, g_diag_max=g_diag_max)

            # Test feasibility
            objective = cp.Minimize(cp.norm(K, "fro"))
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)

            print(f"     Constraint building: ✓ ({len(constraints)} constraints)")
            print(f"     Optimization status: {problem.status}")

            if problem.status in ["optimal", "optimal_inaccurate"]:
                K_opt = K.value
                G_opt = estimator.A_invsqrt @ W.value @ estimator.A_invsqrt
                G_eigs = np.linalg.eigvals(G_opt).real
                print(f"     G eigenvalue range: [{np.min(G_eigs):.3f}, {np.max(G_eigs):.3f}]")

        except Exception as e:
            print(f"     Error: {e}")


def main():
    """Run complete demonstration of K matrix constraint functionality."""

    print("K Matrix Physical Bounds Constraints Demo")
    print("=========================================")
    print("This demo shows how to use KMatrixEstimator to build CVXPY constraints")
    print("for K matrices with physical bounds, and solve optimization problems.\n")

    # Core demonstration
    estimator, constraints, W, K, bounds = demonstrate_constraint_building()
    solutions = solve_constrained_optimization(estimator, constraints, W, K, bounds)
    validate_solutions(estimator, solutions, bounds)

    # Visualization
    fig = create_visualization(estimator, solutions)

    # Multiple networks
    demonstrate_multiple_networks()

    print("\n=== Demo Summary ===")
    print("This demo demonstrated:")
    print("• Creating KMatrixEstimator for constraint building")
    print("• Computing physical bounds from enzyme parameters")
    print("• Building CVXPY constraints with build_k_bounds_constraints()")
    print("• Solving constrained optimization problems")
    print("• Validating solutions against physical bounds")
    print("• Testing on multiple network topologies")
    print("\nKey capabilities:")
    print("• Ensures K = A*G with G ≥ 0 (thermodynamic consistency)")
    print("• Enforces diagonal bounds on G from enzyme kinetics")
    print("• Enforces spectral bounds on G eigenvalues")
    print("• Integrates with CVXPY for custom optimization problems")
    print("• Works with arbitrary reaction network topologies")

    return {"estimator": estimator, "solutions": solutions, "constraints": constraints}


if __name__ == "__main__":
    results = main()
    plt.show()
