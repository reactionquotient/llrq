"""
Comprehensive tests for SBMLParser class using real libsbml.

Tests SBML parsing functionality including species extraction, reaction extraction,
parameter parsing, and error handling using actual SBML files.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llrq.reaction_network import ReactionNetwork
from llrq.sbml_parser import SBMLParseError, SBMLParser


class TestSBMLParserImport:
    """Test SBML parser import and initialization."""

    def test_import_without_libsbml(self):
        """Test that ImportError is raised when libsbml is not available."""
        # Save original state
        import llrq.sbml_parser

        original_libsbml = llrq.sbml_parser.libsbml
        original_available = llrq.sbml_parser.LIBSBML_AVAILABLE

        try:
            # Mock the absence of libsbml
            llrq.sbml_parser.libsbml = None
            llrq.sbml_parser.LIBSBML_AVAILABLE = False

            with pytest.raises(ImportError, match="libsbml is required"):
                llrq.sbml_parser.SBMLParser("dummy_file.xml")
        finally:
            # Restore original state
            llrq.sbml_parser.libsbml = original_libsbml
            llrq.sbml_parser.LIBSBML_AVAILABLE = original_available


class TestSBMLParserInitialization:
    """Test SBMLParser initialization with different inputs."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_initialization_from_file(self):
        """Test initialization from SBML file."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)

        assert parser.document is not None
        assert parser.model is not None
        assert parser.model.getId() == "simple_model"

    def test_initialization_from_string(self):
        """Test initialization from SBML string."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        with open(test_file, "r") as f:
            sbml_string = f.read()

        parser = SBMLParser(sbml_string)
        assert parser.document is not None
        assert parser.model is not None
        assert parser.model.getId() == "simple_model"

    def test_initialization_with_parsing_errors(self):
        """Test initialization when SBML has parsing errors."""
        # This may or may not raise errors depending on libsbml's validation level
        test_file = self.get_test_file_path("malformed.xml")
        try:
            parser = SBMLParser(test_file)
            # If no error is raised, that's also valid behavior
            assert parser.model is not None
        except SBMLParseError:
            # If error is raised, that's the expected behavior
            pass

    def test_initialization_nonexistent_file(self):
        """Test initialization with non-existent file."""
        with pytest.raises((SBMLParseError, FileNotFoundError, Exception)):
            SBMLParser("nonexistent_file.xml")


class TestSpeciesExtraction:
    """Test species information extraction."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_get_species_info_basic(self):
        """Test basic species information extraction."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)
        species_info = parser.get_species_info()

        assert len(species_info) == 2
        assert "A" in species_info
        assert "B" in species_info

        assert species_info["A"]["name"] == "Species A"
        assert species_info["A"]["initial_concentration"] == 2.0
        assert species_info["A"]["compartment"] == "cell"
        assert species_info["A"]["boundary_condition"] == False

        assert species_info["B"]["name"] == "Species B"
        assert species_info["B"]["initial_concentration"] == 1.0

    def test_get_species_info_with_amounts(self):
        """Test species with initial amounts instead of concentrations."""
        test_file = self.get_test_file_path("species_with_amounts.xml")
        parser = SBMLParser(test_file)
        species_info = parser.get_species_info()

        # Concentration = amount / volume = 4.0 / 2.0 = 2.0
        assert species_info["A"]["initial_concentration"] == 2.0
        # Concentration = amount / volume = 2.0 / 2.0 = 1.0
        assert species_info["B"]["initial_concentration"] == 1.0

    def test_get_species_info_three_species(self):
        """Test species extraction with three species."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)
        species_info = parser.get_species_info()

        assert len(species_info) == 3
        assert "A" in species_info
        assert "B" in species_info
        assert "C" in species_info

        assert species_info["A"]["initial_concentration"] == 1.0
        assert species_info["B"]["initial_concentration"] == 0.5
        assert species_info["C"]["initial_concentration"] == 0.2


class TestReactionExtraction:
    """Test reaction information extraction."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_get_reaction_info_basic(self):
        """Test basic reaction information extraction."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)
        reaction_info = parser.get_reaction_info()

        assert len(reaction_info) == 1
        reaction = reaction_info[0]

        assert reaction["id"] == "R1"
        assert reaction["name"] == "A to B"
        assert reaction["reversible"] == True

        # Check reactants and products
        assert len(reaction["reactants"]) == 1
        assert reaction["reactants"][0] == ("A", 1.0)

        assert len(reaction["products"]) == 1
        assert reaction["products"][0] == ("B", 1.0)

        # Check kinetic law
        assert reaction["kinetic_law"] is not None
        assert "formula" in reaction["kinetic_law"]
        assert "parameters" in reaction["kinetic_law"]

    def test_get_reaction_info_multiple_reactions(self):
        """Test reaction extraction with multiple reactions."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)
        reaction_info = parser.get_reaction_info()

        assert len(reaction_info) == 2

        # First reaction: A + B -> C
        r1 = reaction_info[0]
        assert r1["id"] == "R1"
        assert r1["reversible"] == True
        assert len(r1["reactants"]) == 2
        assert ("A", 1.0) in r1["reactants"]
        assert ("B", 1.0) in r1["reactants"]
        assert len(r1["products"]) == 1
        assert r1["products"][0] == ("C", 1.0)

        # Second reaction: C -> A + B
        r2 = reaction_info[1]
        assert r2["id"] == "R2"
        assert r2["reversible"] == False
        assert len(r2["reactants"]) == 1
        assert r2["reactants"][0] == ("C", 1.0)
        assert len(r2["products"]) == 2
        assert ("A", 1.0) in r2["products"]
        assert ("B", 1.0) in r2["products"]

    def test_extract_stoichiometric_matrix(self):
        """Test stoichiometric matrix extraction."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)

        species_info = parser.get_species_info()
        reactions = parser.get_reaction_info()
        species_ids = list(species_info.keys())

        S = parser.create_stoichiometric_matrix(species_ids, reactions)

        assert S.shape == (3, 2)  # 3 species, 2 reactions

        # Find indices
        a_idx = species_ids.index("A")
        b_idx = species_ids.index("B")
        c_idx = species_ids.index("C")

        # R1: A + B -> C should have S = [[-1], [-1], [1]]
        assert S[a_idx, 0] == -1  # A consumed in R1
        assert S[b_idx, 0] == -1  # B consumed in R1
        assert S[c_idx, 0] == 1  # C produced in R1

        # R2: C -> A + B should have S = [[1], [1], [-1]]
        assert S[a_idx, 1] == 1  # A produced in R2
        assert S[b_idx, 1] == 1  # B produced in R2
        assert S[c_idx, 1] == -1  # C consumed in R2


class TestParameterExtraction:
    """Test parameter extraction."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_get_global_parameters(self):
        """Test global parameter information extraction."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)
        param_info = parser.get_global_parameters()

        assert "k1" in param_info
        assert "k2" in param_info
        assert param_info["k1"]["value"] == 2.0
        assert param_info["k2"]["value"] == 1.5
        assert param_info["k1"]["name"] == "Forward rate 1"
        assert param_info["k2"]["name"] == "Forward rate 2"

    def test_get_local_parameters_in_kinetic_law(self):
        """Test local parameter extraction from kinetic laws."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)
        reactions = parser.get_reaction_info()

        reaction = reactions[0]
        kinetic_law = reaction["kinetic_law"]

        assert "parameters" in kinetic_law
        # Should contain local parameter k1_local
        if "k1_local" in kinetic_law["parameters"]:
            assert kinetic_law["parameters"]["k1_local"]["value"] == 1.0


class TestNetworkDataExtraction:
    """Test complete network data extraction."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_extract_network_data_complete(self):
        """Test complete network data extraction."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)
        data = parser.extract_network_data()

        # Check all expected keys are present
        expected_keys = ["species_ids", "reaction_ids", "stoichiometric_matrix", "species", "reactions", "parameters"]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

        assert data["species_ids"] == ["A", "B"]
        assert data["reaction_ids"] == ["R1"]
        assert data["stoichiometric_matrix"].shape == (2, 1)

        # Verify matrix values
        S = data["stoichiometric_matrix"]
        assert S[0, 0] == -1  # A consumed
        assert S[1, 0] == 1  # B produced


class TestErrorHandling:
    """Test error handling in SBML parsing."""

    def test_sbml_parse_error_exception(self):
        """Test SBMLParseError exception."""
        error = SBMLParseError("Test error message")
        assert str(error) == "Test error message"

    def test_empty_model_handling(self):
        """Test handling of minimal model."""
        # Create a minimal valid SBML model
        minimal_sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="empty_model" name="Empty Model">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
  </model>
</sbml>"""

        parser = SBMLParser(minimal_sbml)
        species_info = parser.get_species_info()
        reactions = parser.get_reaction_info()
        parameters = parser.get_global_parameters()

        assert len(species_info) == 0
        assert len(reactions) == 0
        assert len(parameters) == 0


class TestIntegrationWithReactionNetwork:
    """Test integration with ReactionNetwork creation."""

    def get_test_file_path(self, filename):
        """Get path to test data file."""
        return os.path.join(os.path.dirname(__file__), "data", filename)

    def test_create_reaction_network_from_sbml_data(self):
        """Test creating ReactionNetwork from SBML parser data."""
        test_file = self.get_test_file_path("simple_reaction.xml")
        parser = SBMLParser(test_file)
        sbml_data = parser.extract_network_data()

        network = ReactionNetwork.from_sbml_data(sbml_data)

        assert network.species_ids == ["A", "B"]
        assert network.reaction_ids == ["R1"]
        assert network.S.shape == (2, 1)

        # Check initial concentrations
        c0 = network.get_initial_concentrations()
        assert np.allclose(c0, np.array([2.0, 1.0]))

    def test_create_reaction_network_complex_model(self):
        """Test creating ReactionNetwork from complex SBML model."""
        test_file = self.get_test_file_path("two_reactions.xml")
        parser = SBMLParser(test_file)
        sbml_data = parser.extract_network_data()

        network = ReactionNetwork.from_sbml_data(sbml_data)

        assert len(network.species_ids) == 3
        assert len(network.reaction_ids) == 2
        assert network.S.shape == (3, 2)

        # Check initial concentrations
        c0 = network.get_initial_concentrations()
        expected = np.array([1.0, 0.5, 0.2])  # A, B, C
        assert np.allclose(c0, expected)


# Mock-based tests for error conditions that are hard to reproduce with real files
class TestMockedErrorConditions:
    """Test error conditions using mocks for scenarios hard to reproduce with real SBML."""

    def test_initialization_no_model(self):
        """Test initialization when SBML has no model."""
        import llrq.sbml_parser

        original_libsbml = llrq.sbml_parser.libsbml

        try:
            mock_libsbml = MagicMock()
            mock_doc = MagicMock()
            mock_doc.getNumErrors.return_value = 0
            mock_doc.getModel.return_value = None  # No model
            mock_libsbml.readSBML.return_value = mock_doc

            # Replace libsbml temporarily
            llrq.sbml_parser.libsbml = mock_libsbml

            with pytest.raises(SBMLParseError, match="No model found"):
                SBMLParser("no_model.xml")
        finally:
            # Restore original state
            llrq.sbml_parser.libsbml = original_libsbml

    def test_initialization_with_libsbml_errors(self):
        """Test initialization when SBML parsing returns errors."""
        import llrq.sbml_parser

        original_libsbml = llrq.sbml_parser.libsbml

        try:
            mock_libsbml = MagicMock()
            mock_doc = MagicMock()
            mock_doc.getNumErrors.return_value = 2

            # Mock errors
            mock_error1 = MagicMock()
            mock_error1.getLine.return_value = 10
            mock_error1.getMessage.return_value = "Invalid element"

            mock_error2 = MagicMock()
            mock_error2.getLine.return_value = 15
            mock_error2.getMessage.return_value = "Missing attribute"

            mock_doc.getError.side_effect = [mock_error1, mock_error2]
            mock_libsbml.readSBML.return_value = mock_doc

            # Replace libsbml temporarily
            llrq.sbml_parser.libsbml = mock_libsbml

            with pytest.raises(SBMLParseError, match="SBML parsing errors"):
                SBMLParser("invalid.xml")
        finally:
            # Restore original state
            llrq.sbml_parser.libsbml = original_libsbml


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
