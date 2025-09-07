"""
Comprehensive tests for thermodynamic accounting functionality.

Tests the Onsager conductance L, reaction forces, flux response matrix B,
and other thermodynamic accounting features added to ReactionNetwork.
"""

import os
import sys

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.reaction_network import ReactionNetwork


class TestOnsagerConductance:
    """Test Onsager conductance computation."""

    def test_simple_equilibrium_conductance(self):
        """Test A ⇌ B at equilibrium."""
        # Create A ⇌ B network
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # At equilibrium: forward_flux = reverse_flux
        concentrations = np.array([1.0, 2.0])  # K_eq = k+/k- = 2.0/1.0 = 2.0, so [B]/[A] = 2.0 is equilibrium
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])

        # Compute Onsager conductance
        result = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates, mode="equilibrium")

        # Check basic properties
        assert result["mode_used"] == "equilibrium"
        assert result["near_equilibrium"] == True
        assert result["L"].shape == (1, 1)

        # At equilibrium, L should be diagonal with L = 0.5*(f + r)
        forward_flux = result["forward_flux"][0]
        reverse_flux = result["reverse_flux"][0]
        expected_L = 0.5 * (forward_flux + reverse_flux)

        np.testing.assert_allclose(result["L"], [[expected_L]], rtol=1e-10)
        np.testing.assert_allclose(forward_flux, reverse_flux, rtol=1e-10)

    def test_nonequilibrium_local_conductance(self):
        """Test A ⇌ B away from equilibrium using local linearization."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Far from equilibrium concentrations
        concentrations = np.array([10.0, 0.1])  # [B]/[A] = 0.01, but K_eq = 2.0
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])

        result = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates, mode="local")

        # Check properties
        assert result["mode_used"] == "local"
        assert result["near_equilibrium"] == False
        assert result["L"].shape == (1, 1)

        # Net flux should be significant (not near equilibrium)
        net_flux_magnitude = np.abs(result["net_flux"][0])
        total_flux_magnitude = result["forward_flux"][0] + result["reverse_flux"][0]
        assert net_flux_magnitude > 0.1 * total_flux_magnitude

    def test_auto_mode_selection(self):
        """Test automatic mode selection based on equilibrium proximity."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])

        # Near equilibrium case
        concentrations_eq = np.array([1.0, 2.0])  # Close to equilibrium
        result_eq = network.compute_onsager_conductance(concentrations_eq, forward_rates, backward_rates, mode="auto")
        assert result_eq["mode_used"] == "equilibrium"

        # Far from equilibrium case
        concentrations_neq = np.array([10.0, 0.1])  # Far from equilibrium
        result_neq = network.compute_onsager_conductance(concentrations_neq, forward_rates, backward_rates, mode="auto")
        assert result_neq["mode_used"] == "local"

    def test_reciprocity_enforcement(self):
        """Test Onsager reciprocity (symmetry) enforcement."""
        # Create more complex network: A + B ⇌ C, C ⇌ D
        network = ReactionNetwork(["A", "B", "C", "D"], ["R1", "R2"], [[-1, 0], [-1, 0], [1, -1], [0, 1]])

        concentrations = np.array([1.0, 1.0, 0.5, 0.1])
        forward_rates = np.array([2.0, 1.0])
        backward_rates = np.array([1.0, 0.5])

        # Without reciprocity enforcement
        result_no_recip = network.compute_onsager_conductance(
            concentrations, forward_rates, backward_rates, mode="local", enforce_reciprocity=False
        )

        # With reciprocity enforcement
        result_recip = network.compute_onsager_conductance(
            concentrations, forward_rates, backward_rates, mode="local", enforce_reciprocity=True
        )

        # Check symmetry
        L_recip = result_recip["L"]
        np.testing.assert_allclose(L_recip, L_recip.T, rtol=1e-10)

        # Check positive semi-definiteness
        eigenvals = np.linalg.eigvals(L_recip)
        assert np.all(eigenvals >= -1e-10)

    def test_bimolecular_reaction_conductance(self):
        """Test A + B ⇌ C reaction."""
        network = ReactionNetwork(["A", "B", "C"], ["R1"], [[-1], [-1], [1]])

        concentrations = np.array([2.0, 3.0, 1.0])
        forward_rates = np.array([1.0])
        backward_rates = np.array([0.5])

        result = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates, mode="equilibrium")

        # Basic checks
        assert result["L"].shape == (1, 1)
        assert result["L"][0, 0] > 0  # Conductance should be positive

        # Check flux computation
        # Forward: k+ * [A] * [B] = 1.0 * 2.0 * 3.0 = 6.0
        # Reverse: k- * [C] = 0.5 * 1.0 = 0.5
        expected_forward = 6.0
        expected_reverse = 0.5

        np.testing.assert_allclose(result["forward_flux"][0], expected_forward)
        np.testing.assert_allclose(result["reverse_flux"][0], expected_reverse)

    def test_input_validation(self):
        """Test input validation for compute_onsager_conductance."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Valid inputs
        concentrations = np.array([1.0, 1.0])
        forward_rates = np.array([1.0])
        backward_rates = np.array([1.0])

        # Test dimension mismatches
        with pytest.raises(ValueError, match="Expected 2 concentrations"):
            network.compute_onsager_conductance([1.0], forward_rates, backward_rates)

        with pytest.raises(ValueError, match="Expected 1 forward rates"):
            network.compute_onsager_conductance(concentrations, [1.0, 2.0], backward_rates)

        with pytest.raises(ValueError, match="Expected 1 backward rates"):
            network.compute_onsager_conductance(concentrations, forward_rates, [1.0, 2.0])

        # Test negative/zero values
        with pytest.raises(ValueError, match="All concentrations must be positive"):
            network.compute_onsager_conductance([0.0, 1.0], forward_rates, backward_rates)

        with pytest.raises(ValueError, match="All forward rates must be positive"):
            network.compute_onsager_conductance(concentrations, [0.0], backward_rates)

        with pytest.raises(ValueError, match="All backward rates must be positive"):
            network.compute_onsager_conductance(concentrations, forward_rates, [-1.0])

        # Test invalid mode
        with pytest.raises(ValueError, match="Unknown mode"):
            network.compute_onsager_conductance(concentrations, forward_rates, backward_rates, mode="invalid")


class TestReactionForces:
    """Test reaction forces computation."""

    def test_simple_reaction_forces(self):
        """Test reaction forces for A ⇌ B."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        concentrations = np.array([1.0, 2.0])
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])  # K_eq = 2.0

        forces = network.compute_reaction_forces(concentrations, forward_rates, backward_rates)

        # Reaction force x = ln(Q) - ln(K_eq)
        # Q = [B]/[A] = 2.0/1.0 = 2.0
        # K_eq = k+/k- = 2.0/1.0 = 2.0
        # x = ln(2.0) - ln(2.0) = 0 (at equilibrium)

        np.testing.assert_allclose(forces, [0.0], atol=1e-10)

    def test_nonequilibrium_reaction_forces(self):
        """Test reaction forces away from equilibrium."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        concentrations = np.array([4.0, 2.0])  # Q = [B]/[A] = 0.5
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])  # K_eq = 2.0

        forces = network.compute_reaction_forces(concentrations, forward_rates, backward_rates)

        # x = ln(Q) - ln(K_eq) = ln(0.5) - ln(2.0) = ln(0.25) < 0
        expected = np.log(0.5) - np.log(2.0)
        np.testing.assert_allclose(forces, [expected])

    def test_bimolecular_reaction_forces(self):
        """Test forces for A + B ⇌ C."""
        network = ReactionNetwork(["A", "B", "C"], ["R1"], [[-1], [-1], [1]])

        concentrations = np.array([1.0, 2.0, 3.0])
        forward_rates = np.array([1.0])
        backward_rates = np.array([0.5])  # K_eq = 2.0

        forces = network.compute_reaction_forces(concentrations, forward_rates, backward_rates)

        # Q = [C]/([A]*[B]) = 3.0/(1.0*2.0) = 1.5
        # K_eq = 2.0
        # x = ln(1.5) - ln(2.0) = ln(0.75)
        expected = np.log(1.5) - np.log(2.0)
        np.testing.assert_allclose(forces, [expected])


class TestFluxResponseMatrix:
    """Test flux response matrix B(c) computation."""

    def test_simple_flux_response(self):
        """Test B(c) for A ⇌ B."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        concentrations = np.array([2.0, 4.0])
        B = network.compute_flux_response_matrix(concentrations)

        # B = S^T * diag(1/c) * S
        # S = [[-1], [1]], S^T = [[-1, 1]]
        # diag(1/c) = [[0.5, 0], [0, 0.25]]
        # B = [-1, 1] * [[0.5, 0], [0, 0.25]] * [[-1], [1]]
        #   = [-0.5, 0.25] * [[-1], [1]] = 0.5 + 0.25 = 0.75

        expected_B = np.array([[0.75]])
        np.testing.assert_allclose(B, expected_B)

    def test_two_reaction_flux_response(self):
        """Test B(c) for A ⇌ B ⇌ C."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], [[-1, 0], [1, -1], [0, 1]])

        concentrations = np.array([1.0, 2.0, 4.0])
        B = network.compute_flux_response_matrix(concentrations)

        # Should be 2x2 matrix
        assert B.shape == (2, 2)

        # B should be symmetric for this network structure
        np.testing.assert_allclose(B, B.T, rtol=1e-10)

    def test_flux_response_input_validation(self):
        """Test input validation for flux response matrix."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Wrong number of concentrations
        with pytest.raises(ValueError, match="Expected 2 concentrations"):
            network.compute_flux_response_matrix([1.0])

        # Zero concentration
        with pytest.raises(ValueError, match="All concentrations must be positive"):
            network.compute_flux_response_matrix([0.0, 1.0])


class TestLinearRelaxationMatrix:
    """Test linear relaxation matrix K = BL computation."""

    def test_simple_linear_relaxation(self):
        """Test K(c) for A ⇌ B."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        concentrations = np.array([1.0, 2.0])
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])

        result = network.compute_linear_relaxation_matrix(concentrations, forward_rates, backward_rates, mode="equilibrium")

        # Check structure
        assert "K" in result
        assert "B" in result
        assert "L" in result
        assert "onsager_info" in result

        K = result["K"]
        B = result["B"]
        L = result["L"]

        # Check dimensions
        assert K.shape == (1, 1)
        assert B.shape == (1, 1)
        assert L.shape == (1, 1)

        # Check K = B @ L
        expected_K = B @ L
        np.testing.assert_allclose(K, expected_K)

    def test_multipoint_linear_relaxation(self):
        """Test K(c) for different concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])
        forward_rates = np.array([1.0])
        backward_rates = np.array([1.0])

        # Test at multiple concentration points
        test_points = [[1.0, 1.0], [0.5, 2.0], [2.0, 0.5], [10.0, 0.1]]

        for conc in test_points:
            result = network.compute_linear_relaxation_matrix(conc, forward_rates, backward_rates)

            # K should always be positive definite near equilibrium
            K = result["K"]
            eigenvals = np.linalg.eigvals(K)
            assert np.all(eigenvals.real >= -1e-10), f"K not PSD at {conc}: eigenvals = {eigenvals}"


class TestDetailedBalance:
    """Test detailed balance checking."""

    def test_detailed_balance_at_equilibrium(self):
        """Test detailed balance check at equilibrium."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        # Set up equilibrium concentrations
        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])  # K_eq = 2.0
        concentrations = np.array([1.0, 2.0])  # [B]/[A] = 2.0 = K_eq

        balance = network.check_detailed_balance(concentrations, forward_rates, backward_rates)

        # Should satisfy detailed balance
        assert balance["detailed_balance"] == True
        assert balance["max_imbalance"] < 1e-10

        # Forward and reverse fluxes should be equal
        np.testing.assert_allclose(balance["forward_flux"], balance["reverse_flux"], rtol=1e-10)

        # Q/K ratio should be 1
        np.testing.assert_allclose(balance["quotient_ratio"], [1.0], rtol=1e-10)

    def test_detailed_balance_away_from_equilibrium(self):
        """Test detailed balance check away from equilibrium."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        forward_rates = np.array([2.0])
        backward_rates = np.array([1.0])  # K_eq = 2.0
        concentrations = np.array([4.0, 2.0])  # [B]/[A] = 0.5 ≠ K_eq

        balance = network.check_detailed_balance(concentrations, forward_rates, backward_rates)

        # Should not satisfy detailed balance
        assert balance["detailed_balance"] == False
        assert balance["max_imbalance"] > 1e-6

        # Q/K ratio should not be 1
        assert not np.allclose(balance["quotient_ratio"], [1.0], rtol=1e-6)

    def test_bimolecular_detailed_balance(self):
        """Test detailed balance for A + B ⇌ C."""
        network = ReactionNetwork(["A", "B", "C"], ["R1"], [[-1], [-1], [1]])

        forward_rates = np.array([1.0])
        backward_rates = np.array([0.5])  # K_eq = 2.0

        # At equilibrium: [C]/([A]*[B]) = K_eq = 2.0
        concentrations = np.array([1.0, 1.0, 2.0])  # Q = 2.0/1.0 = 2.0 = K_eq

        balance = network.check_detailed_balance(concentrations, forward_rates, backward_rates)

        assert balance["detailed_balance"] == True
        np.testing.assert_allclose(balance["quotient_ratio"], [1.0], rtol=1e-10)


class TestThermodynamicIntegration:
    """Test integration with existing ReactionNetwork functionality."""

    def test_integration_with_equilibrium_computation(self):
        """Test that computed equilibrium satisfies detailed balance."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        forward_rates = np.array([3.0])
        backward_rates = np.array([1.5])
        initial_concentrations = np.array([2.0, 1.0])

        # Compute equilibrium
        c_eq, _ = network.compute_equilibrium(forward_rates, backward_rates, initial_concentrations)

        # Check detailed balance at equilibrium
        balance = network.check_detailed_balance(c_eq, forward_rates, backward_rates)
        assert balance["detailed_balance"] == True

        # Check that Onsager conductance recognizes equilibrium
        onsager = network.compute_onsager_conductance(c_eq, forward_rates, backward_rates, mode="auto")
        assert onsager["near_equilibrium"] == True
        assert onsager["mode_used"] == "equilibrium"

    def test_thermodynamic_consistency_check(self):
        """Test thermodynamic consistency across different methods."""
        # Create a more complex network
        network = ReactionNetwork(
            ["A", "B", "C"],
            ["R1", "R2"],
            [[-1, 1], [1, -1], [0, 0]],  # A ⇌ B, B ⇌ A (redundant for testing)
        )

        concentrations = np.array([1.0, 2.0, 0.5])
        forward_rates = np.array([2.0, 1.0])
        backward_rates = np.array([1.0, 2.0])

        # All methods should work consistently
        forces = network.compute_reaction_forces(concentrations, forward_rates, backward_rates)
        onsager = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates)
        relaxation = network.compute_linear_relaxation_matrix(concentrations, forward_rates, backward_rates)
        balance = network.check_detailed_balance(concentrations, forward_rates, backward_rates)

        # Basic consistency checks
        assert len(forces) == network.n_reactions
        assert onsager["L"].shape == (network.n_reactions, network.n_reactions)
        assert relaxation["K"].shape == (network.n_reactions, network.n_reactions)
        assert len(balance["forward_flux"]) == network.n_reactions


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_concentrations(self):
        """Test behavior with very small concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        concentrations = np.array([1e-10, 1e-8])
        forward_rates = np.array([1.0])
        backward_rates = np.array([1.0])

        # Should not crash
        onsager = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates)
        forces = network.compute_reaction_forces(concentrations, forward_rates, backward_rates)

        # Results should be finite
        assert np.all(np.isfinite(onsager["L"]))
        assert np.all(np.isfinite(forces))

    def test_large_rate_ratios(self):
        """Test behavior with very large rate constant ratios."""
        network = ReactionNetwork(["A", "B"], ["R1"], [[-1], [1]])

        concentrations = np.array([1.0, 1.0])
        forward_rates = np.array([1e6])
        backward_rates = np.array([1e-6])  # Very large K_eq

        # Should not crash
        onsager = network.compute_onsager_conductance(concentrations, forward_rates, backward_rates)
        balance = network.check_detailed_balance(concentrations, forward_rates, backward_rates)

        # Results should be finite
        assert np.all(np.isfinite(onsager["L"]))
        assert np.isfinite(balance["max_imbalance"])

    def test_single_species_network(self):
        """Test degenerate case with single species (no reactions possible)."""
        # This should be caught during network initialization
        with pytest.raises(ValueError):
            ReactionNetwork(["A"], [], np.array([]))
