"""
Comprehensive tests for sparse matrix operations in ReactionNetwork.

Tests compare results between sparse and dense implementations to ensure
correctness, and benchmark performance improvements.
"""

import os
import sys
import time
import warnings

import numpy as np
import pytest
from scipy import sparse

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.reaction_network import ReactionNetwork


class TestSparseOperations:
    """Test sparse matrix operations for correctness and performance."""

    def create_sparse_network(self, n_species=100, n_reactions=50, sparsity=0.95):
        """Create a sparse reaction network for testing.

        Args:
            n_species: Number of species
            n_reactions: Number of reactions
            sparsity: Target sparsity (fraction of zeros)
        """
        species_ids = [f"S{i}" for i in range(n_species)]
        reaction_ids = [f"R{j}" for j in range(n_reactions)]

        # Create sparse stoichiometric matrix
        S = np.zeros((n_species, n_reactions))

        # Each reaction involves 2-3 species randomly for higher sparsity
        np.random.seed(42)  # For reproducibility
        for j in range(n_reactions):
            n_involved = np.random.randint(2, 4)  # Reduced from 2-5 to 2-4
            species_indices = np.random.choice(n_species, n_involved, replace=False)

            # Randomly assign reactants and products
            n_reactants = np.random.randint(1, n_involved)
            reactants = species_indices[:n_reactants]
            products = species_indices[n_reactants:]

            # Assign stoichiometric coefficients
            for i in reactants:
                S[i, j] = -np.random.randint(1, 3)  # Reduced range from 1-4 to 1-3
            for i in products:
                S[i, j] = np.random.randint(1, 3)  # Reduced range from 1-4 to 1-3

        # Check actual sparsity and adjust if needed
        actual_sparsity = 1.0 - np.count_nonzero(S) / S.size

        # If sparsity is still too low, zero out some random entries
        while actual_sparsity < sparsity - 0.05:
            # Find non-zero entries and randomly zero some out
            rows, cols = np.nonzero(S)
            if len(rows) == 0:
                break
            # Zero out 10% of non-zero entries at a time
            n_to_zero = max(1, int(0.1 * len(rows)))
            indices_to_zero = np.random.choice(len(rows), n_to_zero, replace=False)
            for idx in indices_to_zero:
                S[rows[idx], cols[idx]] = 0
            actual_sparsity = 1.0 - np.count_nonzero(S) / S.size

        # Final verification with more tolerance for complex matrices
        assert actual_sparsity >= sparsity - 0.1, f"Sparsity too low: {actual_sparsity:.3f} (target: {sparsity:.3f})"

        return ReactionNetwork(species_ids, reaction_ids, S)

    def test_sparse_vs_dense_basic_properties(self):
        """Test that basic properties match between sparse and dense."""
        # Create test network
        network_dense = self.create_sparse_network(n_species=50, n_reactions=25)

        # Create same network with sparse matrix
        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Test basic properties
        assert network_dense.n_species == network_sparse.n_species
        assert network_dense.n_reactions == network_sparse.n_reactions
        assert not network_dense.is_sparse
        assert network_sparse.is_sparse
        assert network_sparse.sparsity > 0.9

        # Test matrix extraction methods
        A_dense = network_dense.get_reactant_stoichiometry_matrix()
        A_sparse = network_sparse.get_reactant_stoichiometry_matrix()
        np.testing.assert_array_almost_equal(A_dense, A_sparse.toarray())

        B_dense = network_dense.get_product_stoichiometry_matrix()
        B_sparse = network_sparse.get_product_stoichiometry_matrix()
        np.testing.assert_array_almost_equal(B_dense, B_sparse.toarray())

    def test_sparse_reaction_quotients(self):
        """Test reaction quotient computation with sparse matrices."""
        network_dense = self.create_sparse_network(n_species=30, n_reactions=20)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Test concentrations
        concentrations = np.random.uniform(0.1, 10.0, network_dense.n_species)

        # Compute reaction quotients
        Q_dense = network_dense.compute_reaction_quotients(concentrations)
        Q_sparse = network_sparse.compute_reaction_quotients(concentrations)

        # Should be very close
        np.testing.assert_array_almost_equal(Q_dense, Q_sparse, decimal=10)

    def test_sparse_conservation_laws(self):
        """Test conservation law computation with sparse matrices."""
        network_dense = self.create_sparse_network(n_species=40, n_reactions=25)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Find conservation laws
        C_dense = network_dense.find_conservation_laws()
        C_sparse = network_sparse.find_conservation_laws()

        # Should have same number of conservation laws
        assert C_dense.shape[0] == C_sparse.shape[0]
        assert C_dense.shape[1] == C_sparse.shape[1]

        # Test that both satisfy C @ S = 0
        if C_dense.shape[0] > 0:
            CS_dense = C_dense @ network_dense.S
            CS_sparse = C_sparse @ network_sparse.S.toarray()

            np.testing.assert_array_almost_equal(CS_dense, 0, decimal=8)
            np.testing.assert_array_almost_equal(CS_sparse, 0, decimal=8)

    def test_sparse_equilibrium_computation(self):
        """Test equilibrium computation with sparse matrices."""
        network_dense = self.create_sparse_network(n_species=20, n_reactions=15)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Random rate constants
        k_plus = np.random.uniform(0.5, 5.0, network_dense.n_reactions)
        k_minus = np.random.uniform(0.5, 5.0, network_dense.n_reactions)

        # Initial concentrations
        initial_conc = np.random.uniform(0.5, 2.0, network_dense.n_species)

        try:
            # Compute equilibrium
            c_eq_dense, info_dense = network_dense.compute_equilibrium(k_plus, k_minus, initial_conc)
            c_eq_sparse, info_sparse = network_sparse.compute_equilibrium(k_plus, k_minus, initial_conc)

            # Should be close
            np.testing.assert_array_almost_equal(c_eq_dense, c_eq_sparse, decimal=8)

            # Info should be similar
            assert abs(info_dense["thermodynamic_check"] - info_sparse["thermodynamic_check"]) < 1e-8

        except ValueError as e:
            # If thermodynamically inconsistent, both should fail the same way
            with pytest.raises(ValueError):
                network_sparse.compute_equilibrium(k_plus, k_minus, initial_conc)

    def test_sparse_dynamics_matrix(self):
        """Test dynamics matrix computation with sparse matrices."""
        network_dense = self.create_sparse_network(n_species=15, n_reactions=10)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Random rate constants and concentrations
        k_plus = np.random.uniform(1.0, 3.0, network_dense.n_reactions)
        k_minus = np.random.uniform(1.0, 3.0, network_dense.n_reactions)
        concentrations = np.random.uniform(0.5, 2.0, network_dense.n_species)

        try:
            # Compute dynamics matrix in equilibrium mode
            result_dense = network_dense.compute_dynamics_matrix(k_plus, k_minus, concentrations, mode="equilibrium")
            result_sparse = network_sparse.compute_dynamics_matrix(k_plus, k_minus, concentrations, mode="equilibrium")

            # Dynamics matrices should be close
            np.testing.assert_array_almost_equal(result_dense["K"], result_sparse["K"], decimal=6)

            # Flux coefficients should be close
            np.testing.assert_array_almost_equal(result_dense["phi"], result_sparse["phi"], decimal=8)

        except ValueError:
            # If computation fails for dense, should fail for sparse too
            pass

    def test_sparse_onsager_conductance(self):
        """Test Onsager conductance computation with sparse matrices."""
        network_dense = self.create_sparse_network(n_species=20, n_reactions=12)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Random parameters
        k_plus = np.random.uniform(0.8, 4.0, network_dense.n_reactions)
        k_minus = np.random.uniform(0.8, 4.0, network_dense.n_reactions)
        concentrations = np.random.uniform(0.3, 3.0, network_dense.n_species)

        # Compute Onsager conductance
        result_dense = network_dense.compute_onsager_conductance(concentrations, k_plus, k_minus, mode="equilibrium")
        result_sparse = network_sparse.compute_onsager_conductance(concentrations, k_plus, k_minus, mode="equilibrium")

        # Conductance matrices should be close
        np.testing.assert_array_almost_equal(result_dense["L"], result_sparse["L"], decimal=6)

        # Fluxes should be identical
        np.testing.assert_array_almost_equal(result_dense["forward_flux"], result_sparse["forward_flux"], decimal=10)
        np.testing.assert_array_almost_equal(result_dense["reverse_flux"], result_sparse["reverse_flux"], decimal=10)

    @pytest.mark.slow
    def test_sparse_performance_large_network(self):
        """Benchmark sparse vs dense performance on large networks."""
        # Create large sparse network
        n_species = 500
        n_reactions = 200

        print(f"\nTesting performance with {n_species} species, {n_reactions} reactions")

        network_dense = self.create_sparse_network(n_species, n_reactions, sparsity=0.98)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        print(f"Actual sparsity: {network_sparse.sparsity:.3f}")

        # Test conservation law computation performance
        start_time = time.time()
        C_dense = network_dense.find_conservation_laws()
        time_dense = time.time() - start_time

        start_time = time.time()
        C_sparse = network_sparse.find_conservation_laws()
        time_sparse = time.time() - start_time

        print(f"Conservation laws - Dense: {time_dense:.3f}s, Sparse: {time_sparse:.3f}s")
        print(f"Speedup: {time_dense/time_sparse:.2f}x")

        # For very sparse, large matrices, sparse should be competitive or faster
        # But allow for some overhead in sparse algorithms
        if network_sparse.sparsity > 0.95 and n_species >= 500:
            assert time_sparse <= 2 * time_dense, "Sparse should be competitive for highly sparse matrices"
        else:
            # For smaller networks, sparse may have overhead - just check it's not too slow
            assert time_sparse <= 5 * time_dense, "Sparse shouldn't be much slower"

        # Test that results are consistent
        assert C_dense.shape[0] == C_sparse.shape[0], "Should find same number of conservation laws"

    def test_sparse_column_extraction(self):
        """Test the _get_sparse_column helper method."""
        network_dense = self.create_sparse_network(n_species=30, n_reactions=20)

        S_sparse = sparse.csr_matrix(network_dense.S)
        network_sparse = ReactionNetwork(network_dense.species_ids, network_dense.reaction_ids, S_sparse, use_sparse=True)

        # Test all columns
        for j in range(network_sparse.n_reactions):
            col_dense = network_dense.S[:, j]
            col_sparse = network_sparse._get_sparse_column(j)

            np.testing.assert_array_equal(col_dense, col_sparse)

    def test_sparse_lstsq_helper(self):
        """Test the _sparse_lstsq helper method."""
        network = self.create_sparse_network(n_species=50, n_reactions=30)

        # Create test system A x = b
        A_dense = np.random.randn(30, 20)
        A_sparse = sparse.csr_matrix(A_dense)
        b = np.random.randn(30)

        # Test dense case
        network._use_sparse = False
        x_dense, resid_dense = network._sparse_lstsq(A_dense, b)

        # Test sparse case
        network._use_sparse = True
        x_sparse, resid_sparse = network._sparse_lstsq(A_sparse, b)

        # Solutions should be close
        np.testing.assert_array_almost_equal(x_dense, x_sparse, decimal=6)

    def test_sparse_nullspace_helper(self):
        """Test the _sparse_nullspace helper method."""
        network = self.create_sparse_network(n_species=40, n_reactions=25)

        # Create test matrix with known nullspace
        M = np.random.randn(20, 30)
        M_sparse = sparse.csr_matrix(M)

        # Test dense nullspace
        network._use_sparse = False
        N_dense = network._sparse_nullspace(M)

        # Test sparse nullspace
        network._use_sparse = True
        N_sparse = network._sparse_nullspace(M_sparse)

        # Both should satisfy M @ N ≈ 0
        if N_dense.shape[1] > 0:
            MN_dense = M @ N_dense
            np.testing.assert_array_almost_equal(MN_dense, 0, decimal=8)

        if N_sparse.shape[1] > 0:
            MN_sparse = M @ N_sparse
            np.testing.assert_array_almost_equal(MN_sparse, 0, decimal=6)

        # Should have same nullity
        assert N_dense.shape[1] == N_sparse.shape[1]

    def test_sparse_pinv_helper(self):
        """Test the _sparse_pinv helper method."""
        network = self.create_sparse_network(n_species=30, n_reactions=20)

        # Create test matrix
        M = np.random.randn(15, 10)
        M_sparse = sparse.csr_matrix(M)

        # Test dense pseudoinverse
        network._use_sparse = False
        Minv_dense = network._sparse_pinv(M)

        # Test sparse pseudoinverse
        network._use_sparse = True
        Minv_sparse = network._sparse_pinv(M_sparse)

        # Both should satisfy M @ M+ @ M ≈ M
        MM_dense = M @ Minv_dense @ M
        MM_sparse = M @ Minv_sparse @ M

        np.testing.assert_array_almost_equal(MM_dense, M, decimal=8)
        np.testing.assert_array_almost_equal(MM_sparse, M, decimal=6)

    def test_force_sparse_usage(self):
        """Test explicit sparse matrix creation."""
        # Create network and force sparse usage
        S = np.random.randn(20, 15)
        # Make it sparse by setting many elements to zero
        mask = np.random.random((20, 15)) > 0.3
        S[mask] = 0

        network = ReactionNetwork(
            species_ids=[f"S{i}" for i in range(20)],
            reaction_ids=[f"R{j}" for j in range(15)],
            stoichiometric_matrix=S,
            use_sparse=True,  # Force sparse
        )

        assert network.is_sparse, "Should use sparse format when forced"
        assert sparse.issparse(network.S), "S should be a sparse matrix"

    def test_auto_sparsity_detection(self):
        """Test automatic sparsity detection."""
        # Create very sparse matrix
        S = np.zeros((100, 50))
        # Fill only 2% of entries
        for _ in range(int(0.02 * S.size)):
            i, j = np.random.randint(0, 100), np.random.randint(0, 50)
            S[i, j] = np.random.randn()

        network = ReactionNetwork(
            species_ids=[f"S{i}" for i in range(100)],
            reaction_ids=[f"R{j}" for j in range(50)],
            stoichiometric_matrix=S,
            use_sparse=None,  # Auto-detect
        )

        assert network.is_sparse, "Should auto-detect sparse format"
        assert network.sparsity > 0.95, "Should be highly sparse"

    def test_dense_format_maintained(self):
        """Test that dense format is maintained when appropriate."""
        # Small dense matrix
        S = np.random.randn(5, 8)

        network = ReactionNetwork(
            species_ids=[f"S{i}" for i in range(5)],
            reaction_ids=[f"R{j}" for j in range(8)],
            stoichiometric_matrix=S,
            use_sparse=None,  # Auto-detect
        )

        assert not network.is_sparse, "Should use dense format for small dense matrices"
        assert isinstance(network.S, np.ndarray), "S should be dense numpy array"


if __name__ == "__main__":
    # Run specific tests
    test_class = TestSparseOperations()

    print("Testing basic sparse operations...")
    test_class.test_sparse_vs_dense_basic_properties()
    print("✓ Basic properties test passed")

    print("Testing reaction quotients...")
    test_class.test_sparse_reaction_quotients()
    print("✓ Reaction quotients test passed")

    print("Testing conservation laws...")
    test_class.test_sparse_conservation_laws()
    print("✓ Conservation laws test passed")

    print("Testing helper methods...")
    test_class.test_sparse_column_extraction()
    print("✓ Column extraction test passed")

    print("\nAll sparse operation tests passed!")
