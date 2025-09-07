"""
Comprehensive tests for ReactionNetwork class.

Tests all core functionality of the ReactionNetwork class including
reaction quotient calculations, conservation laws, stoichiometry methods,
and network analysis features.
"""

import os
import sys

import numpy as np
import pytest

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.reaction_network import ReactionNetwork


class TestReactionNetworkBasics:
    """Test basic ReactionNetwork functionality."""

    def test_initialization_valid(self):
        """Test valid network initialization."""
        species_ids = ["A", "B", "C"]
        reaction_ids = ["R1", "R2"]
        S = np.array([[-1, 0], [1, -1], [0, 1]])  # A  # B  # C

        network = ReactionNetwork(species_ids, reaction_ids, S)

        assert network.species_ids == species_ids
        assert network.reaction_ids == reaction_ids
        assert np.array_equal(network.S, S)
        assert network.n_species == 3
        assert network.n_reactions == 2

    def test_initialization_dimension_mismatch(self):
        """Test initialization with mismatched dimensions."""
        species_ids = ["A", "B"]  # 2 species
        reaction_ids = ["R1"]  # 1 reaction
        S = np.array([[-1], [1], [0]])  # 3 rows - mismatch!

        with pytest.raises(ValueError, match="Stoichiometric matrix has 3 rows.*2 species"):
            ReactionNetwork(species_ids, reaction_ids, S)

        # Test column mismatch
        S = np.array([[-1, 0], [1, -1]])  # 2 columns - mismatch!
        with pytest.raises(ValueError, match="Stoichiometric matrix has 2 columns.*1 reactions"):
            ReactionNetwork(species_ids, reaction_ids, S)

    def test_species_reaction_mappings(self):
        """Test species and reaction index mappings."""
        species_ids = ["X", "Y", "Z"]
        reaction_ids = ["forward", "backward"]
        S = np.array([[-1, 0], [1, -1], [0, 1]])

        network = ReactionNetwork(species_ids, reaction_ids, S)

        assert network.species_to_idx == {"X": 0, "Y": 1, "Z": 2}
        assert network.reaction_to_idx == {"forward": 0, "backward": 1}

    def test_initial_concentrations_default(self):
        """Test initial concentrations with no species info."""
        network = self._create_simple_network()
        c0 = network.get_initial_concentrations()

        assert np.array_equal(c0, np.zeros(2))

    def test_initial_concentrations_with_info(self):
        """Test initial concentrations from species info."""
        species_info = {"A": {"initial_concentration": 2.0}, "B": {"initial_concentration": 1.5}}

        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]), species_info=species_info)

        c0 = network.get_initial_concentrations()
        expected = np.array([2.0, 1.5])
        assert np.array_equal(c0, expected)

    def _create_simple_network(self):
        """Helper to create simple A ⇌ B network."""
        return ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))


class TestReactionQuotients:
    """Test reaction quotient calculations."""

    def test_single_reaction_quotient(self):
        """Test A ⇌ B reaction quotient: Q = [B]/[A]."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        concentrations = np.array([2.0, 4.0])
        Q = network.compute_reaction_quotients(concentrations)

        expected = np.array([4.0 / 2.0])  # [B]/[A]
        assert np.allclose(Q, expected)

    def test_bimolecular_reaction_quotient(self):
        """Test A + B ⇌ C reaction quotient: Q = [C]/([A]*[B])."""
        network = ReactionNetwork(["A", "B", "C"], ["R1"], np.array([[-1], [-1], [1]]))

        concentrations = np.array([1.0, 2.0, 6.0])
        Q = network.compute_reaction_quotients(concentrations)

        expected = np.array([6.0 / (1.0 * 2.0)])  # [C]/([A]*[B])
        assert np.allclose(Q, expected)

    def test_complex_stoichiometry(self):
        """Test reaction with complex stoichiometry: 2A + B ⇌ 3C."""
        network = ReactionNetwork(["A", "B", "C"], ["R1"], np.array([[-2], [-1], [3]]))

        concentrations = np.array([2.0, 3.0, 4.0])
        Q = network.compute_reaction_quotients(concentrations)

        # Q = [C]^3 / ([A]^2 * [B]) = 4^3 / (2^2 * 3) = 64/12 = 16/3
        expected = np.array([4.0**3 / (2.0**2 * 3.0)])
        assert np.allclose(Q, expected)

    def test_multiple_reactions(self):
        """Test multiple reactions: A ⇌ B ⇌ C."""
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        concentrations = np.array([1.0, 2.0, 4.0])
        Q = network.compute_reaction_quotients(concentrations)

        expected = np.array([2.0 / 1.0, 4.0 / 2.0])  # [B]/[A], [C]/[B]
        assert np.allclose(Q, expected)

    def test_zero_concentration_handling(self):
        """Test handling of zero concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        concentrations = np.array([0.0, 1.0])
        Q = network.compute_reaction_quotients(concentrations)

        # Should not be NaN or Inf
        assert np.isfinite(Q).all()

    def test_single_reaction_quotient_by_id(self):
        """Test computing quotient for specific reaction ID."""
        network = ReactionNetwork(["A", "B", "C"], ["forward", "reverse"], np.array([[-1, 0], [1, -1], [0, 1]]))

        concentrations = np.array([1.0, 2.0, 4.0])

        Q_forward = network.compute_single_reaction_quotient("forward", concentrations)
        Q_reverse = network.compute_single_reaction_quotient("reverse", concentrations)

        assert np.isclose(Q_forward, 2.0)  # [B]/[A]
        assert np.isclose(Q_reverse, 2.0)  # [C]/[B]

    def test_invalid_reaction_id(self):
        """Test error for invalid reaction ID."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))
        concentrations = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Reaction 'invalid' not found"):
            network.compute_single_reaction_quotient("invalid", concentrations)

    def test_wrong_concentration_count(self):
        """Test error for wrong number of concentrations."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        with pytest.raises(ValueError, match="Expected 2 concentrations.*got 1"):
            network.compute_reaction_quotients([1.0])


class TestStoichiometry:
    """Test stoichiometry-related methods."""

    def test_reaction_stoichiometry_simple(self):
        """Test getting reactants and products for simple reaction."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        reactants, products = network.get_reaction_stoichiometry("R1")

        assert reactants == {"A": 1.0}
        assert products == {"B": 1.0}

    def test_reaction_stoichiometry_complex(self):
        """Test getting reactants and products for complex reaction."""
        # 2A + 3B ⇌ C + 4D
        network = ReactionNetwork(["A", "B", "C", "D"], ["R1"], np.array([[-2], [-3], [1], [4]]))

        reactants, products = network.get_reaction_stoichiometry("R1")

        assert reactants == {"A": 2.0, "B": 3.0}
        assert products == {"C": 1.0, "D": 4.0}

    def test_reaction_equation_simple(self):
        """Test reaction equation string generation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        equation = network.get_reaction_equation("R1")
        assert "A" in equation and "B" in equation
        assert "→" in equation or "⇌" in equation

    def test_reaction_equation_complex(self):
        """Test complex reaction equation."""
        # 2A + B ⇌ 3C
        network = ReactionNetwork(["A", "B", "C"], ["R1"], np.array([[-2], [-1], [3]]))

        equation = network.get_reaction_equation("R1")
        assert "2 A" in equation
        assert "B" in equation
        assert "3 C" in equation

    def test_reaction_equation_with_reversibility(self):
        """Test equation with reversibility info."""
        reaction_info = [{"id": "R1", "reversible": True}]
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]), reaction_info=reaction_info)

        equation = network.get_reaction_equation("R1")
        assert "⇌" in equation

    def test_empty_reactants_products(self):
        """Test reactions with empty reactants or products."""
        # ∅ → A (creation reaction)
        network = ReactionNetwork(["A"], ["R1"], np.array([[1]]))

        equation = network.get_reaction_equation("R1")
        assert "∅" in equation


class TestConservationLaws:
    """Test conservation law detection and computation."""

    def test_conservation_law_simple(self):
        """Test network with simple conservation law."""
        # A → B has conservation law A + B = constant
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        C = network.find_conservation_laws()
        assert C.shape[0] == 1  # One conservation law

        # Check that it's the total mass conservation [1,1] (normalized)
        expected = np.array([1, 1]) / np.linalg.norm([1, 1])
        assert np.allclose(np.abs(C[0]), expected, atol=1e-10)

        # Verify it's actually a conservation law: C @ S = 0
        assert np.allclose(C @ network.S, 0, atol=1e-10)

    def test_single_conservation_law(self):
        """Test network with one conservation law."""
        # A ⇌ B ⇌ C (closed cycle, A + B + C = constant)
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        C = network.find_conservation_laws()
        assert C.shape[0] == 1  # One conservation law

        # Check that it's the total mass conservation [1,1,1] (normalized)
        expected = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
        assert np.allclose(np.abs(C[0]), expected, atol=1e-10)

        # Verify it's actually a conservation law: C @ S = 0
        assert np.allclose(C @ network.S, 0, atol=1e-10)

    def test_multiple_conservation_laws(self):
        """Test network with multiple conservation laws."""
        # Two separate A ⇌ B and C ⇌ D
        network = ReactionNetwork(["A", "B", "C", "D"], ["R1", "R2"], np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))

        C = network.find_conservation_laws()
        assert C.shape[0] == 2  # Two conservation laws

    def test_conserved_quantities(self):
        """Test computation of conserved quantities."""
        # A ⇌ B with conservation A + B = constant
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        concentrations = np.array([2.0, 3.0])
        conserved = network.compute_conserved_quantities(concentrations)

        assert conserved.size == 1  # Should have one conserved quantity
        # Since conservation matrix is normalized [1,1]/sqrt(2),
        # conserved quantity is (A + B) / sqrt(2)
        expected = (2.0 + 3.0) / np.sqrt(2)
        assert np.isclose(conserved[0], expected, rtol=1e-10)

    def test_conserved_quantities_simple(self):
        """Test computation of conserved quantities."""
        # A → B has conservation law A + B = constant
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        concentrations = np.array([2.0, 3.0])
        conserved = network.compute_conserved_quantities(concentrations)

        assert conserved.size == 1
        # The conserved quantity should be approximately A + B
        # Since C is normalized [1,1]/sqrt(2), the conserved quantity is (A+B)/sqrt(2)
        expected = (2.0 + 3.0) / np.sqrt(2)
        assert np.isclose(conserved[0], expected, rtol=1e-10)


class TestNetworkAnalysis:
    """Test network analysis methods."""

    def test_independent_reactions_full_rank(self):
        """Test finding independent reactions for full rank system."""
        # A → B → C (linearly independent)
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2"], np.array([[-1, 0], [1, -1], [0, 1]]))

        independent = network.get_independent_reactions()
        assert len(independent) == 2  # Both reactions are independent

    def test_independent_reactions_rank_deficient(self):
        """Test finding independent reactions for rank deficient system."""
        # Create dependent reactions
        network = ReactionNetwork(["A", "B", "C"], ["R1", "R2", "R3"], np.array([[-1, 0, -1], [1, -1, 1], [0, 1, 0]]))

        independent = network.get_independent_reactions()
        assert len(independent) < 3  # Some reactions are dependent

    def test_network_summary(self):
        """Test network summary generation."""
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        summary = network.summary()

        assert "Species: 2" in summary
        assert "Reactions: 1" in summary
        assert "R1" in summary

    def test_summary_with_conservation(self):
        """Test summary including conservation laws."""
        # Create network with conservation
        network = ReactionNetwork(["A", "B"], ["R1"], np.array([[-1], [1]]))

        summary = network.summary()

        # Should mention conservation laws (or lack thereof)
        assert "Conservation" in summary


class TestFromSBMLData:
    """Test creation from SBML data."""

    def test_from_sbml_data_basic(self):
        """Test creating network from SBML-like data."""
        sbml_data = {
            "species_ids": ["A", "B"],
            "reaction_ids": ["R1"],
            "stoichiometric_matrix": np.array([[-1], [1]]),
            "species": {"A": {"initial_concentration": 1.0}},
            "reactions": [{"id": "R1", "reversible": True}],
            "parameters": {},
        }

        network = ReactionNetwork.from_sbml_data(sbml_data)

        assert network.species_ids == ["A", "B"]
        assert network.reaction_ids == ["R1"]
        assert np.array_equal(network.S, np.array([[-1], [1]]))
        assert network.species_info == sbml_data["species"]
        assert network.reaction_info == sbml_data["reactions"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
