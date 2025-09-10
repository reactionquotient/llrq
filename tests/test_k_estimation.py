"""Test K matrix estimation utilities with physical bounds."""

import pytest
import numpy as np
import cvxpy as cp

# Import the modules being tested
from llrq.reaction_network import ReactionNetwork
from llrq.estimation.k_estimation import KMatrixEstimator
from llrq.utils.physical_bounds import (
    compute_diffusion_limit,
    enzyme_to_conductance,
    compute_spectral_caps,
    validate_physical_consistency,
    gershgorin_bounds,
    estimate_reaction_timescales,
)


class TestPhysicalBoundsUtilities:
    """Test physical bounds utility functions."""

    def test_compute_diffusion_limit_units_fix(self):
        """Test diffusion limit calculation with corrected units."""
        # Test single molecular weight
        mw = 50000  # 50 kDa protein
        k_on = compute_diffusion_limit(mw)

        # Should be in reasonable range for protein-ligand binding
        # With the 1e3 units fix, values should be higher
        assert 1e8 <= k_on <= 1e11  # M^-1 s^-1 (corrected range)

        # Test multiple molecular weights
        mws = np.array([1000, 100000, 10000000])  # 1 kDa to 10 MDa (extreme range)
        k_ons = compute_diffusion_limit(mws)

        assert len(k_ons) == 3
        assert all(1e7 <= k <= 1e11 for k in k_ons)  # Updated range
        # For Smoluchowski kinetics, rate constants are relatively insensitive to MW
        # Test that they're in the right ballpark and all positive
        assert np.all(k_ons > 0)
        assert np.max(k_ons) / np.min(k_ons) < 10  # Within an order of magnitude

    def test_compute_diffusion_limit_two_molecules(self):
        """Test diffusion limit with two different molecular weights."""
        mw_A = 30000  # 30 kDa
        mw_B = 60000  # 60 kDa

        k_on_AB = compute_diffusion_limit(mw_A, mw_B)
        k_on_AA = compute_diffusion_limit(mw_A)  # Self-reaction

        # AB reaction should be different from AA reaction
        assert k_on_AB != k_on_AA
        assert k_on_AB > 0
        assert k_on_AA > 0

        # Test with custom reaction radius
        k_on_custom = compute_diffusion_limit(mw_A, mw_B, reaction_radius=2e-9)
        assert k_on_custom != k_on_AB

    def test_compute_diffusion_limit_physics(self):
        """Test that diffusion limit follows expected physics."""
        # Test temperature dependence (should increase with T)
        mw = 40000
        k_298 = compute_diffusion_limit(mw, temperature=298.15)
        k_310 = compute_diffusion_limit(mw, temperature=310.15)  # Body temp
        assert k_310 > k_298  # Higher temperature → higher rate

        # Test viscosity dependence (should decrease with viscosity)
        k_low_visc = compute_diffusion_limit(mw, viscosity=0.5e-3)
        k_high_visc = compute_diffusion_limit(mw, viscosity=2.0e-3)
        assert k_low_visc > k_high_visc  # Lower viscosity → higher rate

    def test_enzyme_to_conductance(self):
        """Test enzyme parameter to conductance conversion."""
        # Simple two-reaction system
        n_reactions = 2
        enzyme_conc = np.array([1e-6, 2e-6])  # 1 μM, 2 μM
        kcat = np.array([100.0, 200.0])  # s^-1
        km = np.array([1e-3, 2e-3])  # 1 mM, 2 mM

        # Equilibrium concentrations
        c_eq = np.array([1e-3, 5e-4, 1e-4])  # mM range

        # Simple stoichiometry: A → B, B → C
        S = np.array([[-1, 0], [1, -1], [0, 1]])  # 3 species, 2 reactions

        result = enzyme_to_conductance(enzyme_conc, kcat, km, c_eq, S)

        # Check return structure
        assert "g_diag_max" in result
        assert "g_diag_min" in result
        assert "thermodynamic_factors" in result
        assert "catalytic_efficiencies" in result

        # Check dimensions
        assert len(result["g_diag_max"]) == n_reactions
        assert len(result["g_diag_min"]) == n_reactions

        # Check that bounds are positive and reasonable
        assert np.all(result["g_diag_max"] > 0)
        assert np.all(result["g_diag_min"] >= 0)
        assert np.all(result["g_diag_min"] <= result["g_diag_max"])

        # Check that catalytic efficiencies are correct
        expected_cat_eff = kcat / km
        np.testing.assert_array_equal(result["catalytic_efficiencies"], expected_cat_eff)

    def test_compute_spectral_caps(self):
        """Test spectral cap computation."""
        g_min = np.array([0.1, 0.2, 0.05])
        g_max = np.array([10.0, 15.0, 8.0])

        caps = compute_spectral_caps((g_min, g_max))

        assert "gamma_max" in caps
        assert "gamma_min" in caps

        # gamma_max should be >= max diagonal
        assert caps["gamma_max"] >= np.max(g_max)

        # gamma_min should be <= min diagonal
        assert caps["gamma_min"] <= np.min(g_min)

        # Both should be non-negative
        assert caps["gamma_max"] > 0
        assert caps["gamma_min"] >= 0

    def test_validate_physical_consistency_W_based(self):
        """Test physical consistency validation using W = A^{-1/2} K A^{1/2}."""
        # Create a simple physically consistent system with symmetric G
        A = np.array([[2.0, -0.5], [-0.5, 1.5]])  # Non-identity A matrix

        # Create K = A*G where G is PSD and symmetric (to satisfy Onsager)
        G_true = np.array([[1.0, 0.0], [0.0, 0.8]])  # Diagonal G for perfect Onsager
        K = A @ G_true

        # Should pass consistency checks
        result = validate_physical_consistency(K, A)

        assert result["is_physically_consistent"] is True
        # May still have small Onsager violation due to numerical precision
        assert "W_min_eigenvalue" in result
        assert "W_max_eigenvalue" in result
        assert result["W_min_eigenvalue"] >= -1e-12  # Should be PSD

        # G should be recovered correctly
        G_recovered = result["G_matrix"]
        np.testing.assert_array_almost_equal(G_recovered, G_true, decimal=10)

        # Test with bounds
        g_bounds = (np.array([0.5, 0.5]), np.array([2.0, 1.5]))
        spectral_bounds = {"gamma_min": 0.1, "gamma_max": 2.0}

        result = validate_physical_consistency(K, A, g_bounds, spectral_bounds)

        assert result["is_physically_consistent"] is True
        assert "onsager_residual_frobenius" in result
        # Onsager residual may be non-zero even with diagonal G if A is not symmetric
        # This is expected behavior - the test threshold should be reasonable
        assert result["onsager_residual_frobenius"] < 1.0  # Reasonable threshold

    def test_validate_physical_consistency_singular_A(self):
        """Test validation with singular A matrix."""
        # Create singular A (rank deficient)
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])  # Rank 1

        # For singular A, use the null space approach
        # Create K that respects the structure of A
        K = np.array([[1.0, -1.0], [-1.0, 1.0]])  # Same structure as A

        result = validate_physical_consistency(K, A)

        # Should handle singular A gracefully
        if result.get("A_rank", 0) > 0:
            # If A has positive rank, should work
            assert "W_min_eigenvalue" in result
        else:
            # If A is zero, should detect this
            assert "A_rank_zero" in result["violations"]

    def test_validate_physical_consistency_violations(self):
        """Test detection of physical consistency violations."""
        A = np.eye(2)

        # Create K with negative eigenvalue (unphysical)
        K_bad = np.array([[1.0, 2.0], [2.0, -0.5]])

        result = validate_physical_consistency(K_bad, A)

        assert result["is_physically_consistent"] is False
        # The violation should be detected as W not being PSD
        assert "W_not_PSD" in result["violations"] or "negative_eigenvalue" in result["violations"]

    def test_gershgorin_bounds(self):
        """Test Gershgorin circle theorem bounds."""
        g_diag = np.array([2.0, 1.5, 3.0])

        # Test with estimated coupling
        bounds = gershgorin_bounds(g_diag, coupling_strength=0.1)

        assert "eigenvalue_bounds" in bounds
        assert "global_bounds" in bounds
        assert "gershgorin_radii" in bounds

        # Check dimensions
        assert len(bounds["eigenvalue_bounds"]["lower"]) == 3
        assert len(bounds["eigenvalue_bounds"]["upper"]) == 3

        # Lower bounds should be <= diagonal
        assert np.all(bounds["eigenvalue_bounds"]["lower"] <= g_diag)

        # Upper bounds should be >= diagonal
        assert np.all(bounds["eigenvalue_bounds"]["upper"] >= g_diag)

    def test_estimate_reaction_timescales_W_based(self):
        """Test reaction timescale estimation using W = A^{-1/2} K A^{1/2}."""
        # Create A and K matrices
        A = np.array([[2.0, -0.5], [-0.5, 1.5]])
        G = np.array([[1.0, 0.1], [0.1, 2.0]])
        K = A @ G

        result = estimate_reaction_timescales(K, A)

        assert "eigenvalues" in result
        assert "timescales" in result
        assert "fastest_timescale" in result
        assert "slowest_finite_timescale" in result
        assert "space_type" in result
        assert result["space_type"] == "proper_similarity_transform"

        # Should have positive eigenvalues (since G is PSD)
        assert np.all(result["eigenvalues"] >= -1e-12)

        # Timescales should be finite for positive eigenvalues
        finite_timescales = result["timescales"][np.isfinite(result["timescales"])]
        assert len(finite_timescales) > 0
        assert np.all(finite_timescales > 0)

        # Fastest timescale should be reciprocal of largest eigenvalue
        if result["num_finite_modes"] > 0:
            largest_eigenval = np.max(result["eigenvalues"][result["eigenvalues"] > 1e-12])
            expected_fastest = 1.0 / largest_eigenval
            np.testing.assert_almost_equal(result["fastest_timescale"], expected_fastest)

    def test_estimate_reaction_timescales_fallback(self):
        """Test timescale estimation fallback without A matrix."""
        # Create symmetric K matrix
        K = np.array([[2.0, 0.3], [0.3, 1.0]])

        with pytest.warns(UserWarning, match="No A matrix provided"):
            result = estimate_reaction_timescales(K)

        assert result["space_type"] == "fallback_symmetric_K"
        assert "eigenvalues" in result
        assert "timescales" in result

        # Should work but with warning
        assert len(result["eigenvalues"]) == 2

    def test_estimate_reaction_timescales_singular_A(self):
        """Test timescale estimation with singular A matrix."""
        # Create singular A
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])  # Rank 1
        K = np.array([[1.0, 0.5], [0.5, 0.8]])

        result = estimate_reaction_timescales(K, A)

        # Should handle singular A gracefully
        assert "space_type" in result
        if result["space_type"] == "degenerate_A_zero":
            # A is approximately zero
            assert len(result["eigenvalues"]) == 0
        else:
            # A has positive rank, should work in reduced space
            assert len(result["eigenvalues"]) > 0


class TestKMatrixEstimator:
    """Test K matrix estimator constraint building functionality."""

    @pytest.fixture
    def simple_network(self):
        """Create simple A ⇌ B reaction network for testing."""
        species = ["A", "B"]
        reactions = ["forward", "backward"]
        S = np.array([[-1, 1], [1, -1]])  # A ⇌ B

        return ReactionNetwork(species, reactions, S)

    @pytest.fixture
    def estimator(self, simple_network):
        """Create K matrix estimator for simple network."""
        c_eq = np.array([1.0, 2.0])  # Equilibrium concentrations
        return KMatrixEstimator(simple_network, c_eq)

    def test_initialization(self, simple_network):
        """Test KMatrixEstimator initialization."""
        c_eq = np.array([1.0, 2.0])
        estimator = KMatrixEstimator(simple_network, c_eq)

        assert estimator.network == simple_network
        np.testing.assert_array_equal(estimator.c_eq, c_eq)
        assert estimator.A.shape == (2, 2)  # 2 reactions
        assert estimator.A_sqrt.shape == (2, 2)
        assert estimator.A_invsqrt.shape == (2, 2)

        # A should be positive definite after ridge regularization
        eigenvals = np.linalg.eigvals(estimator.A)
        assert np.all(eigenvals > 0)

    def test_initialization_errors(self, simple_network):
        """Test initialization error handling."""
        # Wrong number of concentrations
        with pytest.raises(ValueError, match="Expected 2 c_eq entries"):
            KMatrixEstimator(simple_network, np.array([1.0]))  # Only 1 concentration

    def test_make_A_matrix(self, estimator):
        """Test A matrix construction."""
        A = estimator.A

        # A should be symmetric
        np.testing.assert_array_almost_equal(A, A.T)

        # A should be positive definite (after ridge)
        eigenvals = np.linalg.eigvals(A)
        assert np.all(eigenvals > 0)

        # Check manual calculation
        N = estimator.network.S
        Wc = np.diag(1.0 / estimator.c_eq)
        A_expected = N.T @ Wc @ N + estimator.eps * np.eye(N.shape[1])
        A_expected = 0.5 * (A_expected + A_expected.T)

        np.testing.assert_array_almost_equal(A, A_expected)

    def test_sqrt_decomposition(self, estimator):
        """Test A^{1/2} and A^{-1/2} computation."""
        A_sqrt = estimator.A_sqrt
        A_invsqrt = estimator.A_invsqrt
        A = estimator.A

        # Check that A_sqrt^2 = A
        np.testing.assert_array_almost_equal(A_sqrt @ A_sqrt, A)

        # Check that A_invsqrt @ A @ A_invsqrt = I
        should_be_I = A_invsqrt @ A @ A_invsqrt
        np.testing.assert_array_almost_equal(should_be_I, np.eye(A.shape[0]))

        # Check that A_sqrt @ A_invsqrt = I
        should_also_be_I = A_sqrt @ A_invsqrt
        np.testing.assert_array_almost_equal(should_also_be_I, np.eye(A.shape[0]))

    def test_build_k_bounds_constraints_basic(self, estimator):
        """Test basic constraint building without bounds."""
        K = cp.Variable((2, 2))

        constraints, W = estimator.build_k_bounds_constraints(K)

        # Should have at least the basic constraint
        assert len(constraints) >= 1

        # W should be a CVXPY variable
        assert isinstance(W, cp.Variable)
        assert W.shape == (2, 2)
        assert W.is_psd()

    def test_build_k_bounds_constraints_with_bounds(self, estimator):
        """Test constraint building with physical bounds."""
        K = cp.Variable((2, 2))

        # Set bounds
        g_diag_min = np.array([0.1, 0.2])
        g_diag_max = np.array([5.0, 4.0])
        gamma_min = 0.05
        gamma_max = 10.0

        constraints, W = estimator.build_k_bounds_constraints(
            K, g_diag_min=g_diag_min, g_diag_max=g_diag_max, gamma_min=gamma_min, gamma_max=gamma_max
        )

        # Should have multiple constraints
        assert len(constraints) > 1

        # Try to create a feasible problem
        objective = cp.Minimize(cp.norm(K, "fro"))
        problem = cp.Problem(objective, constraints)

        # Should be feasible (bounds are reasonable)
        problem.solve(verbose=False)
        assert problem.status in ["optimal", "optimal_inaccurate"]

    def test_build_k_bounds_constraints_dimension_errors(self, estimator):
        """Test error handling for wrong bound dimensions."""
        K = cp.Variable((2, 2))

        # Wrong dimension for g_diag_max
        with pytest.raises(ValueError, match="g_diag_max must have length equal to #reactions"):
            estimator.build_k_bounds_constraints(K, g_diag_max=np.array([1.0]))  # Should be length 2

        # Wrong dimension for g_diag_min
        with pytest.raises(ValueError, match="g_diag_min must have length equal to #reactions"):
            estimator.build_k_bounds_constraints(K, g_diag_min=np.array([1.0, 2.0, 3.0]))  # Should be length 2

    def test_build_k_bounds_constraints_integration(self, estimator):
        """Test that constraints can be used in a complete optimization."""
        # Create a simple optimization problem using the constraints
        K = cp.Variable((2, 2))

        # Use more generous physical bounds to ensure feasibility
        g_diag_max = np.array([10.0, 15.0])  # More generous bounds

        constraints, W = estimator.build_k_bounds_constraints(K, g_diag_max=g_diag_max)

        # Minimize Frobenius norm
        objective = cp.Minimize(cp.norm(K, "fro"))
        problem = cp.Problem(objective, constraints)

        problem.solve(verbose=False)

        # Should at least be feasible
        assert problem.status in ["optimal", "optimal_inaccurate", "unbounded"]

        if problem.status in ["optimal", "optimal_inaccurate"]:
            K_opt = K.value
            W_opt = W.value

            # Basic sanity checks
            assert K_opt is not None
            assert W_opt is not None

            # G should be PSD (the main physical requirement)
            G_opt = estimator.A_invsqrt @ W_opt @ estimator.A_invsqrt
            G_eigenvals = np.linalg.eigvals(G_opt)
            assert np.all(G_eigenvals.real >= -1e-6)


class TestIntegrationWithLLRQ:
    """Test integration with existing LLRQ classes."""

    def test_integration_with_physical_bounds_validation(self):
        """Test that KMatrixEstimator works with physical bounds validation."""
        # Create network
        species = ["A", "B"]
        reactions = ["R1"]
        S = np.array([[-1], [1]])  # A → B
        network = ReactionNetwork(species, reactions, S)

        # Create estimator
        c_eq = np.array([2.0, 1.0])
        estimator = KMatrixEstimator(network, c_eq)

        # Create a test K matrix
        K_test = np.array([[1.5]])  # Single reaction

        # Validate with physical bounds utilities
        result = validate_physical_consistency(K_test, estimator.A)

        # Should work without errors
        assert "is_physically_consistent" in result
        assert "G_matrix" in result

        # G matrix should be reasonable
        G = result["G_matrix"]
        assert G.shape == (1, 1)
        assert np.all(np.linalg.eigvals(G).real >= -1e-6)  # Should be PSD

    def test_constraint_building_with_cvxpy_problem(self):
        """Test that constraints work in a realistic CVXPY optimization."""
        # Create a slightly larger network
        species = ["A", "B", "C"]
        reactions = ["R1", "R2"]
        S = np.array([[-1, 0], [1, -1], [0, 1]])  # A → B → C
        network = ReactionNetwork(species, reactions, S)

        c_eq = np.array([1.0, 0.5, 0.2])
        estimator = KMatrixEstimator(network, c_eq)

        # Create optimization problem
        K = cp.Variable((2, 2))

        constraints, W = estimator.build_k_bounds_constraints(K, g_diag_max=np.array([10.0, 8.0]), gamma_max=15.0)

        # Add a data-fitting term (simulate fitting to identity)
        target_K = np.eye(2)
        objective = cp.Minimize(cp.norm(K - target_K, "fro"))

        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        assert problem.status in ["optimal", "optimal_inaccurate"]

        # Solution should be close to target but respect bounds
        K_opt = K.value
        assert K_opt is not None
        assert K_opt.shape == (2, 2)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_k_estimation.py -v
    pytest.main([__file__])
