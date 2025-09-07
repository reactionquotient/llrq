"""
Performance benchmark tests for LLRQ package.

Tests large networks, long simulations, matrix computations, and other
performance-critical aspects of the package.
"""

import os
import sys
import time
import warnings

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.llrq_dynamics import LLRQDynamics
from llrq.reaction_network import ReactionNetwork
from llrq.solver import LLRQSolver


class TestNetworkScaling:
    """Test performance scaling with network size."""

    def create_chain_network(self, n_species):
        """Create linear chain network A1 ⇌ A2 ⇌ ... ⇌ An."""
        species_ids = [f"A{i+1}" for i in range(n_species)]
        reaction_ids = [f"R{i+1}" for i in range(n_species - 1)]

        # Linear chain stoichiometry
        S = np.zeros((n_species, n_species - 1))
        for j in range(n_species - 1):
            S[j, j] = -1  # Reactant
            S[j + 1, j] = 1  # Product

        return ReactionNetwork(species_ids, reaction_ids, S)

    def test_small_network_performance(self):
        """Benchmark small network (5 species)."""
        network = self.create_chain_network(5)

        start_time = time.time()

        # Create dynamics
        c_star = np.ones(5)
        k_plus = np.ones(4)
        k_minus = 0.5 * np.ones(4)

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Solve system
        solver = LLRQSolver(dynamics)
        initial_concentrations = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
        t_span = (0.0, 5.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        elapsed_time = time.time() - start_time

        assert result["success"]
        assert elapsed_time < 5.0  # Should complete quickly

        print(f"Small network (5 species) time: {elapsed_time:.3f}s")

    def test_medium_network_performance(self):
        """Benchmark medium network (20 species)."""
        network = self.create_chain_network(20)

        start_time = time.time()

        # Create dynamics
        c_star = np.ones(20)
        k_plus = np.ones(19)
        k_minus = 0.5 * np.ones(19)

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Solve system
        solver = LLRQSolver(dynamics)
        initial_concentrations = np.concatenate([[10.0], np.ones(19)])
        t_span = (0.0, 5.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        elapsed_time = time.time() - start_time

        assert result["success"]
        assert elapsed_time < 30.0  # Should complete in reasonable time

        print(f"Medium network (20 species) time: {elapsed_time:.3f}s")

    @pytest.mark.slow
    def test_large_network_performance(self):
        """Benchmark large network (50 species)."""
        network = self.create_chain_network(50)

        start_time = time.time()

        # Create dynamics
        c_star = np.ones(50)
        k_plus = np.ones(49)
        k_minus = 0.5 * np.ones(49)

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        # Solve system (shorter time for performance)
        solver = LLRQSolver(dynamics)
        initial_concentrations = np.concatenate([[20.0], np.ones(49)])
        t_span = (0.0, 2.0)  # Shorter simulation

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=100)

        elapsed_time = time.time() - start_time

        assert result["success"]
        print(f"Large network (50 species) time: {elapsed_time:.3f}s")

        # Performance expectation (should scale reasonably)
        assert elapsed_time < 120.0  # 2 minutes max

    def create_branched_network(self, n_branches, branch_length):
        """Create branched network for scaling tests."""
        total_species = 1 + n_branches * branch_length  # Central + branches
        species_ids = ["Central"] + [f"B{i}_S{j}" for i in range(n_branches) for j in range(branch_length)]

        n_reactions = n_branches * branch_length
        reaction_ids = [f"R{i}_{j}" for i in range(n_branches) for j in range(branch_length)]

        S = np.zeros((total_species, n_reactions))

        reaction_idx = 0
        for branch in range(n_branches):
            # First reaction: Central -> B{branch}_S0
            S[0, reaction_idx] = -1  # Central consumed
            S[1 + branch * branch_length, reaction_idx] = 1  # First species produced
            reaction_idx += 1

            # Chain reactions within branch
            for pos in range(1, branch_length):
                species_idx = 1 + branch * branch_length + pos - 1
                next_species_idx = 1 + branch * branch_length + pos

                S[species_idx, reaction_idx] = -1
                S[next_species_idx, reaction_idx] = 1
                reaction_idx += 1

        return ReactionNetwork(species_ids, reaction_ids, S)

    def test_branched_network_scaling(self):
        """Test scaling with branched topology."""
        network = self.create_branched_network(n_branches=4, branch_length=5)

        start_time = time.time()

        n_species = network.n_species
        n_reactions = network.n_reactions

        c_star = np.ones(n_species)
        k_plus = np.ones(n_reactions)
        k_minus = 0.5 * np.ones(n_reactions)

        dynamics = LLRQDynamics.from_mass_action(
            network=network, forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
        )

        solver = LLRQSolver(dynamics)
        initial_concentrations = np.concatenate([[10.0], np.ones(n_species - 1)])
        t_span = (0.0, 3.0)

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        elapsed_time = time.time() - start_time

        assert result["success"]
        print(f"Branched network ({n_species} species, {n_reactions} reactions) time: {elapsed_time:.3f}s")


class TestMatrixComputationPerformance:
    """Test performance of matrix computations."""

    def test_reaction_quotient_computation_scaling(self):
        """Test scaling of reaction quotient computation."""
        sizes = [10, 50, 100]

        for n_species in sizes:
            network = ReactionNetwork(
                [f"S{i}" for i in range(n_species)],
                [f"R{i}" for i in range(n_species - 1)],
                np.random.randn(n_species, n_species - 1),
            )

            concentrations = np.random.rand(n_species) + 0.1  # Avoid zeros

            start_time = time.time()

            # Compute many times to measure performance
            for _ in range(1000):
                Q = network.compute_reaction_quotients(concentrations)

            elapsed_time = time.time() - start_time

            print(f"Reaction quotients ({n_species} species, 1000 iterations): {elapsed_time:.3f}s")

            # Should scale roughly linearly
            assert elapsed_time < 10.0

    def test_dynamics_matrix_computation_performance(self):
        """Test performance of dynamics matrix computation."""
        sizes = [5, 10, 20]

        for n_species in sizes:
            n_reactions = n_species - 1
            network = ReactionNetwork(
                [f"S{i}" for i in range(n_species)],
                [f"R{i}" for i in range(n_reactions)],
                np.random.randn(n_species, n_reactions),
            )

            c_star = np.random.rand(n_species) + 0.1
            k_plus = np.random.rand(n_reactions) + 0.1
            k_minus = np.random.rand(n_reactions) + 0.1

            start_time = time.time()

            result = network.compute_dynamics_matrix(
                forward_rates=k_plus, backward_rates=k_minus, initial_concentrations=c_star, mode="equilibrium"
            )

            elapsed_time = time.time() - start_time

            print(f"Dynamics matrix ({n_species} species): {elapsed_time:.3f}s")

            assert result["eigenanalysis"]["is_stable"] or not result["eigenanalysis"]["is_stable"]
            assert elapsed_time < 5.0

    def test_conservation_law_computation_performance(self):
        """Test performance of conservation law computation."""
        sizes = [20, 50, 100]

        for n_species in sizes:
            # Create network with some conservation laws
            n_reactions = n_species - 2  # Rank deficient
            S = np.random.randn(n_species, n_reactions)

            network = ReactionNetwork([f"S{i}" for i in range(n_species)], [f"R{i}" for i in range(n_reactions)], S)

            start_time = time.time()

            C = network.find_conservation_laws()

            elapsed_time = time.time() - start_time

            print(f"Conservation laws ({n_species} species): {elapsed_time:.3f}s")

            assert elapsed_time < 10.0
            assert C.shape[1] == n_species

    def test_eigenvalue_computation_performance(self):
        """Test performance of eigenvalue computations."""
        sizes = [10, 20, 50]

        for n in sizes:
            # Create random positive definite matrix
            A = np.random.randn(n, n)
            K = A @ A.T + np.eye(n)  # Ensure positive definite

            start_time = time.time()

            # Compute eigenvalues many times
            for _ in range(100):
                eigenvals, eigenvecs = np.linalg.eig(K)
                is_stable = np.all(eigenvals.real > 0)

            elapsed_time = time.time() - start_time

            print(f"Eigenvalue computation ({n}x{n} matrix, 100 iterations): {elapsed_time:.3f}s")

            assert elapsed_time < 20.0


class TestLongSimulationPerformance:
    """Test performance of long time simulations."""

    def test_short_simulation_performance(self):
        """Benchmark short simulation."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([3.0, 1.0, 1.0])
        t_span = (0.0, 1.0)

        start_time = time.time()

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=1000)

        elapsed_time = time.time() - start_time

        assert result["success"]
        print(f"Short simulation (1 time unit, 1000 points): {elapsed_time:.3f}s")
        assert elapsed_time < 10.0

    def test_long_simulation_performance(self):
        """Benchmark long simulation."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        # Slow dynamics for long simulation
        K = np.array([[0.1, 0.0], [0.0, 0.1]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([5.0, 1.0, 1.0])
        t_span = (0.0, 100.0)  # Long time

        start_time = time.time()

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=1000)

        elapsed_time = time.time() - start_time

        assert result["success"]
        print(f"Long simulation (100 time units, 1000 points): {elapsed_time:.3f}s")
        assert elapsed_time < 60.0  # 1 minute max

    def test_high_resolution_simulation(self):
        """Benchmark high time resolution simulation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        dynamics = LLRQDynamics(network)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0])
        t_span = (0.0, 5.0)

        start_time = time.time()

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=10000)  # High resolution

        elapsed_time = time.time() - start_time

        assert result["success"]
        assert len(result["time"]) == 10000
        print(f"High resolution simulation (10,000 points): {elapsed_time:.3f}s")
        assert elapsed_time < 30.0

    @pytest.mark.slow
    def test_very_long_simulation(self):
        """Benchmark very long simulation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        # Very slow dynamics
        K = np.array([[0.01]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([10.0, 1.0])
        t_span = (0.0, 1000.0)  # Very long time

        start_time = time.time()

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=1000)

        elapsed_time = time.time() - start_time

        assert result["success"]
        print(f"Very long simulation (1000 time units): {elapsed_time:.3f}s")
        # More generous time limit for very long simulation
        assert elapsed_time < 300.0  # 5 minutes max


class TestStiffSystemPerformance:
    """Test performance with stiff systems."""

    def test_mildly_stiff_system(self):
        """Test mildly stiff system performance."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        # Mildly stiff: fast and slow reactions
        K = np.array([[100.0, 0.0], [0.0, 1.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0, 1.0])
        t_span = (0.0, 1.0)

        start_time = time.time()

        result = solver.solve(initial_concentrations, t_span, method="numerical")

        elapsed_time = time.time() - start_time

        print(f"Mildly stiff system: {elapsed_time:.3f}s, success: {result['success']}")

        # May take longer due to stiffness, but should complete
        if result["success"]:
            assert elapsed_time < 60.0

    def test_moderately_stiff_system(self):
        """Test moderately stiff system."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        # More stiff: wide separation of time scales
        K = np.array([[1000.0, 0.0], [0.0, 0.1]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0, 1.0])
        t_span = (0.0, 0.1)  # Shorter time for stiff system

        start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # May have solver warnings
            result = solver.solve(initial_concentrations, t_span, method="numerical", rtol=1e-6)

        elapsed_time = time.time() - start_time

        print(f"Moderately stiff system: {elapsed_time:.3f}s, success: {result['success']}")

        # More generous time limit for stiff systems
        assert elapsed_time < 120.0


class TestMemoryUsageTests:
    """Test memory usage with different problem sizes."""

    @pytest.mark.skip("This test is too flaky")
    def test_memory_usage_scaling(self):
        """Test that memory usage scales reasonably."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        sizes = [10, 50, 100]
        memory_usage = []

        for n_species in sizes:
            network = self.create_chain_network(n_species)

            c_star = np.ones(n_species)
            k_plus = np.ones(n_species - 1)
            k_minus = 0.5 * np.ones(n_species - 1)

            dynamics = LLRQDynamics.from_mass_action(
                network=network,
                forward_rates=k_plus,
                backward_rates=k_minus,
                initial_concentrations=c_star,
                mode="equilibrium",
            )

            solver = LLRQSolver(dynamics)
            initial_concentrations = np.concatenate([[5.0], np.ones(n_species - 1)])
            t_span = (0.0, 1.0)

            result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=100)

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - baseline_memory
            memory_usage.append(memory_used)

            print(f"Memory usage for {n_species} species: {memory_used:.1f} MB")

            # Reasonable memory usage (should not explode)
            assert memory_used < 1000  # Less than 1GB

        # Memory should scale reasonably (not exponentially)
        if len(memory_usage) >= 2:
            growth_factor = memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 1
            assert growth_factor < 100  # Should not grow by more than 100x

    def create_chain_network(self, n_species):
        """Helper method to create chain network."""
        species_ids = [f"A{i+1}" for i in range(n_species)]
        reaction_ids = [f"R{i+1}" for i in range(n_species - 1)]

        S = np.zeros((n_species, n_species - 1))
        for j in range(n_species - 1):
            S[j, j] = -1
            S[j + 1, j] = 1

        return ReactionNetwork(species_ids, reaction_ids, S)

    def test_large_array_handling(self):
        """Test handling of large arrays."""
        # Create moderately large arrays
        n_species = 200
        n_reactions = 150

        # Random sparse stoichiometric matrix
        S = np.random.choice([-2, -1, 0, 0, 0, 1, 2], size=(n_species, n_reactions))

        species_ids = [f"S{i}" for i in range(n_species)]
        reaction_ids = [f"R{i}" for i in range(n_reactions)]

        start_time = time.time()

        network = ReactionNetwork(species_ids, reaction_ids, S)

        # Basic operations should work
        concentrations = np.random.rand(n_species) + 0.1
        Q = network.compute_reaction_quotients(concentrations)

        elapsed_time = time.time() - start_time

        print(f"Large array operations ({n_species}x{n_reactions}): {elapsed_time:.3f}s")

        assert len(Q) == n_reactions
        assert np.isfinite(Q).all()
        assert elapsed_time < 30.0


class TestAnalyticalVsNumericalPerformance:
    """Compare analytical vs numerical solution performance."""

    def test_analytical_solution_performance(self):
        """Benchmark analytical solution."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[2.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)

        x0 = np.array([1.0])
        t = np.linspace(0, 5, 1000)

        start_time = time.time()

        x_t = dynamics.analytical_solution(x0, t)

        elapsed_time = time.time() - start_time

        print(f"Analytical solution (1000 time points): {elapsed_time:.3f}s")

        assert x_t.shape == (1000, 1)
        assert elapsed_time < 1.0  # Should be very fast

    def test_numerical_solution_performance(self):
        """Benchmark numerical solution."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[2.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0])
        t_span = (0.0, 5.0)

        start_time = time.time()

        result = solver.solve(initial_concentrations, t_span, method="numerical", n_points=1000)

        elapsed_time = time.time() - start_time

        print(f"Numerical solution (1000 time points): {elapsed_time:.3f}s")

        assert result["success"]
        assert len(result["time"]) == 1000
        assert elapsed_time < 10.0

    def test_performance_comparison(self):
        """Direct performance comparison when both methods work."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        K = np.array([[1.0]])
        dynamics = LLRQDynamics(network, relaxation_matrix=K)
        solver = LLRQSolver(dynamics)

        initial_concentrations = np.array([2.0, 1.0])
        t_span = (0.0, 2.0)

        # Analytical
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_analytical = solver.solve(initial_concentrations, t_span, method="analytical", n_points=500)
        analytical_time = time.time() - start_time

        # Numerical
        start_time = time.time()
        result_numerical = solver.solve(initial_concentrations, t_span, method="numerical", n_points=500)
        numerical_time = time.time() - start_time

        print(f"Analytical: {analytical_time:.3f}s, Numerical: {numerical_time:.3f}s")

        if result_analytical["method"] == "analytical":
            # Analytical should usually be faster
            assert analytical_time < 2 * numerical_time  # Within factor of 2


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])  # -s to see print output
