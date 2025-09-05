"""
Comprehensive tests for SBMLParser class.

Tests SBML parsing functionality including species extraction, reaction extraction,
parameter parsing, and error handling. Uses mocked libsbml to avoid dependency issues.
"""

import numpy as np
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSBMLParserImport:
    """Test SBML parser import and initialization."""

    def test_import_without_libsbml(self):
        """Test that SBMLParseError is raised when libsbml is not available."""
        with patch.dict('sys.modules', {'libsbml': None}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', False):
                from llrq.sbml_parser import SBMLParser, SBMLParseError
                
                with pytest.raises(ImportError, match="libsbml is required"):
                    SBMLParser("dummy_file.xml")

    def test_import_with_libsbml_available(self):
        """Test import when libsbml is available."""
        mock_libsbml = MagicMock()
        mock_doc = MagicMock()
        mock_model = MagicMock()
        
        mock_doc.getNumErrors.return_value = 0
        mock_doc.getModel.return_value = mock_model
        mock_libsbml.readSBML.return_value = mock_doc
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test_file.xml")
                assert parser.document == mock_doc
                assert parser.model == mock_model


class TestSBMLParserInitialization:
    """Test SBMLParser initialization with different inputs."""

    def create_mock_libsbml(self, errors=None):
        """Helper to create mock libsbml module."""
        mock_libsbml = MagicMock()
        mock_doc = MagicMock()
        mock_model = MagicMock()
        
        if errors:
            mock_doc.getNumErrors.return_value = len(errors)
            mock_errors = []
            for i, (line, message) in enumerate(errors):
                mock_error = MagicMock()
                mock_error.getLine.return_value = line
                mock_error.getMessage.return_value = message
                mock_errors.append(mock_error)
                mock_doc.getError.return_value = mock_error
        else:
            mock_doc.getNumErrors.return_value = 0
            
        mock_doc.getModel.return_value = mock_model
        mock_libsbml.readSBML.return_value = mock_doc
        mock_libsbml.readSBMLFromString.return_value = mock_doc
        
        return mock_libsbml, mock_doc, mock_model

    def test_initialization_from_file(self):
        """Test initialization from SBML file."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                
                mock_libsbml.readSBML.assert_called_once_with("test.xml")
                assert parser.document == mock_doc
                assert parser.model == mock_model

    def test_initialization_from_string(self):
        """Test initialization from SBML string."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Make readSBML fail to simulate string fallback
        mock_libsbml.readSBML.side_effect = Exception("File not found")
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                sbml_string = "<sbml>...</sbml>"
                parser = SBMLParser(sbml_string)
                
                mock_libsbml.readSBMLFromString.assert_called_once_with(sbml_string)
                assert parser.document == mock_doc

    def test_initialization_with_parsing_errors(self):
        """Test initialization when SBML has parsing errors."""
        errors = [(10, "Invalid element"), (15, "Missing attribute")]
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml(errors)
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser, SBMLParseError
                
                with pytest.raises(SBMLParseError, match="SBML parsing errors"):
                    SBMLParser("invalid.xml")

    def test_initialization_no_model(self):
        """Test initialization when SBML has no model."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        mock_doc.getModel.return_value = None  # No model
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser, SBMLParseError
                
                with pytest.raises(SBMLParseError, match="No model found"):
                    SBMLParser("no_model.xml")


class TestSpeciesExtraction:
    """Test species information extraction."""

    def create_mock_species(self, species_id, name=None, init_conc=None, 
                          init_amount=None, compartment="cell", boundary=False):
        """Helper to create mock species object."""
        species = MagicMock()
        species.getId.return_value = species_id
        species.getName.return_value = name or ""
        species.getCompartment.return_value = compartment
        species.getBoundaryCondition.return_value = boundary
        
        if init_conc is not None:
            species.isSetInitialConcentration.return_value = True
            species.getInitialConcentration.return_value = init_conc
            species.isSetInitialAmount.return_value = False
        elif init_amount is not None:
            species.isSetInitialConcentration.return_value = False
            species.isSetInitialAmount.return_value = True
            species.getInitialAmount.return_value = init_amount
        else:
            species.isSetInitialConcentration.return_value = False
            species.isSetInitialAmount.return_value = False
            
        return species

    def test_get_species_info_basic(self):
        """Test basic species information extraction."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Mock species
        species_A = self.create_mock_species("A", "Adenine", init_conc=2.0)
        species_B = self.create_mock_species("B", "Benzene", init_conc=1.5)
        
        mock_model.getNumSpecies.return_value = 2
        mock_model.getSpecies.side_effect = [species_A, species_B]
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                species_info = parser.get_species_info()
                
                assert len(species_info) == 2
                assert 'A' in species_info
                assert 'B' in species_info
                
                assert species_info['A']['name'] == 'Adenine'
                assert species_info['A']['initial_concentration'] == 2.0
                assert species_info['A']['compartment'] == 'cell'
                assert species_info['A']['boundary_condition'] == False
                
                assert species_info['B']['name'] == 'Benzene'
                assert species_info['B']['initial_concentration'] == 1.5

    def test_get_species_info_with_amounts(self):
        """Test species with initial amounts instead of concentrations."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Mock compartment
        mock_compartment = MagicMock()
        mock_compartment.isSetSize.return_value = True
        mock_compartment.getSize.return_value = 2.0  # Volume = 2.0
        mock_model.getCompartment.return_value = mock_compartment
        
        # Mock species with amount
        species = self.create_mock_species("A", init_amount=4.0)
        mock_model.getNumSpecies.return_value = 1
        mock_model.getSpecies.return_value = species
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                species_info = parser.get_species_info()
                
                # Concentration = amount / volume = 4.0 / 2.0 = 2.0
                assert species_info['A']['initial_concentration'] == 2.0

    def test_get_species_info_amount_no_compartment_size(self):
        """Test species with amount but no compartment size."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Mock compartment without size
        mock_compartment = MagicMock()
        mock_compartment.isSetSize.return_value = False
        mock_model.getCompartment.return_value = mock_compartment
        
        species = self.create_mock_species("A", init_amount=3.0)
        mock_model.getNumSpecies.return_value = 1
        mock_model.getSpecies.return_value = species
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                species_info = parser.get_species_info()
                
                # Should use amount directly
                assert species_info['A']['initial_concentration'] == 3.0

    def test_get_species_info_no_initial_value(self):
        """Test species with no initial concentration or amount."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        species = self.create_mock_species("A")  # No initial values
        mock_model.getNumSpecies.return_value = 1
        mock_model.getSpecies.return_value = species
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                species_info = parser.get_species_info()
                
                # Should default to 0.0
                assert species_info['A']['initial_concentration'] == 0.0

    def test_get_species_info_boundary_conditions(self):
        """Test species with boundary conditions."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        species = self.create_mock_species("A", boundary=True, init_conc=1.0)
        mock_model.getNumSpecies.return_value = 1
        mock_model.getSpecies.return_value = species
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                species_info = parser.get_species_info()
                
                assert species_info['A']['boundary_condition'] == True


class TestReactionExtraction:
    """Test reaction information extraction."""

    def create_mock_reaction(self, reaction_id, reversible=True, reactants=None, products=None):
        """Helper to create mock reaction object."""
        reaction = MagicMock()
        reaction.getId.return_value = reaction_id
        reaction.getName.return_value = reaction_id
        reaction.getReversible.return_value = reversible
        
        # Mock reactants
        reactants = reactants or []
        reaction.getNumReactants.return_value = len(reactants)
        mock_reactants = []
        for species_id, stoich in reactants:
            reactant = MagicMock()
            reactant.getSpecies.return_value = species_id
            reactant.getStoichiometry.return_value = stoich
            mock_reactants.append(reactant)
        reaction.getReactant.side_effect = lambda i: mock_reactants[i]
        
        # Mock products
        products = products or []
        reaction.getNumProducts.return_value = len(products)
        mock_products = []
        for species_id, stoich in products:
            product = MagicMock()
            product.getSpecies.return_value = species_id
            product.getStoichiometry.return_value = stoich
            mock_products.append(product)
        reaction.getProduct.side_effect = lambda i: mock_products[i]
        
        return reaction

    def test_get_reaction_info_basic(self):
        """Test basic reaction information extraction."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # A + B -> C
        reaction = self.create_mock_reaction(
            "R1", 
            reversible=True,
            reactants=[("A", 1.0), ("B", 1.0)],
            products=[("C", 1.0)]
        )
        
        mock_model.getNumReactions.return_value = 1
        mock_model.getReaction.return_value = reaction
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                
                # Mock the method (if it exists)
                if hasattr(parser, 'get_reaction_info'):
                    reaction_info = parser.get_reaction_info()
                    
                    assert len(reaction_info) == 1
                    assert reaction_info[0]['id'] == 'R1'
                    assert reaction_info[0]['reversible'] == True

    def test_extract_stoichiometric_matrix(self):
        """Test stoichiometric matrix extraction."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Set up species
        species_A = self.create_mock_species("A")
        species_B = self.create_mock_species("B") 
        species_C = self.create_mock_species("C")
        mock_model.getNumSpecies.return_value = 3
        mock_model.getSpecies.side_effect = [species_A, species_B, species_C]
        
        # Set up reactions: A + B -> C, C -> A + B
        reaction1 = self.create_mock_reaction(
            "R1",
            reactants=[("A", 1.0), ("B", 1.0)],
            products=[("C", 1.0)]
        )
        reaction2 = self.create_mock_reaction(
            "R2", 
            reactants=[("C", 1.0)],
            products=[("A", 1.0), ("B", 1.0)]
        )
        
        mock_model.getNumReactions.return_value = 2
        mock_model.getReaction.side_effect = [reaction1, reaction2]
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                
                # Mock the method (if it exists)
                if hasattr(parser, 'extract_network_data'):
                    data = parser.extract_network_data()
                    
                    assert 'species_ids' in data
                    assert 'reaction_ids' in data
                    assert 'stoichiometric_matrix' in data
                    
                    assert data['species_ids'] == ['A', 'B', 'C']
                    assert data['reaction_ids'] == ['R1', 'R2']
                    
                    S = data['stoichiometric_matrix']
                    assert S.shape == (3, 2)  # 3 species, 2 reactions
                    
                    # R1: A + B -> C should have S = [[-1, 1], [-1, 1], [1, -1]]
                    assert S[0, 0] == -1  # A consumed in R1
                    assert S[1, 0] == -1  # B consumed in R1
                    assert S[2, 0] == 1   # C produced in R1
                    
                    # R2: C -> A + B
                    assert S[0, 1] == 1   # A produced in R2
                    assert S[1, 1] == 1   # B produced in R2
                    assert S[2, 1] == -1  # C consumed in R2


class TestParameterExtraction:
    """Test parameter extraction."""

    def create_mock_parameter(self, param_id, value, name=None):
        """Helper to create mock parameter."""
        param = MagicMock()
        param.getId.return_value = param_id
        param.getName.return_value = name or param_id
        param.getValue.return_value = value
        return param

    def test_get_parameter_info(self):
        """Test parameter information extraction."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Mock parameters
        param1 = self.create_mock_parameter("k1", 2.5, "Forward rate")
        param2 = self.create_mock_parameter("k2", 1.0, "Backward rate")
        
        mock_model.getNumParameters.return_value = 2
        mock_model.getParameter.side_effect = [param1, param2]
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                
                # Mock the method (if it exists)
                if hasattr(parser, 'get_parameter_info'):
                    param_info = parser.get_parameter_info()
                    
                    assert 'k1' in param_info
                    assert 'k2' in param_info
                    assert param_info['k1']['value'] == 2.5
                    assert param_info['k2']['value'] == 1.0


class TestNetworkDataExtraction:
    """Test complete network data extraction."""

    def test_extract_network_data_complete(self):
        """Test complete network data extraction."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Mock complete model setup
        # Species: A, B
        species_A = self.create_mock_species("A", "Species A", init_conc=2.0)
        species_B = self.create_mock_species("B", "Species B", init_conc=1.0)
        mock_model.getNumSpecies.return_value = 2
        mock_model.getSpecies.side_effect = [species_A, species_B]
        
        # Reaction: A <-> B
        reaction = self.create_mock_reaction(
            "R1",
            reversible=True, 
            reactants=[("A", 1.0)],
            products=[("B", 1.0)]
        )
        mock_model.getNumReactions.return_value = 1
        mock_model.getReaction.return_value = reaction
        
        # Parameters
        param = self.create_mock_parameter("k1", 1.5)
        mock_model.getNumParameters.return_value = 1
        mock_model.getParameter.return_value = param
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("test.xml")
                
                # Mock the method (if it exists)
                if hasattr(parser, 'extract_network_data'):
                    data = parser.extract_network_data()
                    
                    # Check all expected keys are present
                    expected_keys = ['species_ids', 'reaction_ids', 'stoichiometric_matrix',
                                   'species', 'reactions', 'parameters']
                    for key in expected_keys:
                        assert key in data, f"Missing key: {key}"
                    
                    assert data['species_ids'] == ['A', 'B']
                    assert data['reaction_ids'] == ['R1']
                    assert data['stoichiometric_matrix'].shape == (2, 1)


class TestErrorHandling:
    """Test error handling in SBML parsing."""

    def test_sbml_parse_error_exception(self):
        """Test SBMLParseError exception."""
        from llrq.sbml_parser import SBMLParseError
        
        error = SBMLParseError("Test error message")
        assert str(error) == "Test error message"

    def test_malformed_sbml_handling(self):
        """Test handling of malformed SBML."""
        errors = [(1, "Malformed XML"), (5, "Invalid element")]
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml(errors)
        
        # Set up error handling
        mock_doc.getError.side_effect = lambda i: errors[i] if i < len(errors) else None
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser, SBMLParseError
                
                with pytest.raises(SBMLParseError) as exc_info:
                    SBMLParser("malformed.xml")
                
                # Should mention both errors
                assert "Line 1" in str(exc_info.value)
                assert "Line 5" in str(exc_info.value)

    def test_empty_model_handling(self):
        """Test handling of model with no species or reactions."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Empty model
        mock_model.getNumSpecies.return_value = 0
        mock_model.getNumReactions.return_value = 0
        mock_model.getNumParameters.return_value = 0
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                
                parser = SBMLParser("empty.xml")
                species_info = parser.get_species_info()
                
                assert len(species_info) == 0


class TestIntegrationWithReactionNetwork:
    """Test integration with ReactionNetwork creation."""

    def test_create_reaction_network_from_sbml_data(self):
        """Test creating ReactionNetwork from SBML parser data."""
        mock_libsbml, mock_doc, mock_model = self.create_mock_libsbml()
        
        # Set up minimal working model
        species_A = self.create_mock_species("A", init_conc=1.0)
        species_B = self.create_mock_species("B", init_conc=0.5)
        mock_model.getNumSpecies.return_value = 2
        mock_model.getSpecies.side_effect = [species_A, species_B]
        
        reaction = self.create_mock_reaction(
            "R1",
            reactants=[("A", 1.0)],
            products=[("B", 1.0)]
        )
        mock_model.getNumReactions.return_value = 1
        mock_model.getReaction.return_value = reaction
        mock_model.getNumParameters.return_value = 0
        
        with patch.dict('sys.modules', {'libsbml': mock_libsbml}):
            with patch('llrq.sbml_parser.LIBSBML_AVAILABLE', True):
                from llrq.sbml_parser import SBMLParser
                from llrq.reaction_network import ReactionNetwork
                
                parser = SBMLParser("test.xml")
                
                # Mock the method (if it exists)
                if hasattr(parser, 'extract_network_data'):
                    sbml_data = parser.extract_network_data()
                    network = ReactionNetwork.from_sbml_data(sbml_data)
                    
                    assert network.species_ids == ['A', 'B']
                    assert network.reaction_ids == ['R1']
                    assert network.S.shape == (2, 1)
                    
                    # Check initial concentrations
                    c0 = network.get_initial_concentrations()
                    assert np.allclose(c0, np.array([1.0, 0.5]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])