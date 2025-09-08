"""Tests for YAML model parser."""

import numpy as np
import pytest
import tempfile
import yaml
from unittest.mock import patch

from llrq.yaml_parser import YAMLModelParser, YAMLParseError, load_yaml_model


class TestYAMLModelParser:
    """Test YAML model parsing functionality."""

    def create_test_yaml_content(self):
        """Create test YAML content mimicking yeast-GEM structure."""
        yaml_data = [
            {"metaData": {"id": "test_model", "name": "Test Model", "version": "1.0"}},
            {
                "metabolites": [
                    {
                        "id": "s_001",
                        "name": "Glucose",
                        "compartment": "c",
                        "formula": "C6H12O6",
                        "charge": 0,
                        "deltaG": -915.0,  # kJ/mol
                        "smiles": "OCC1OC(O)C(O)C(O)C1O",
                        "annotation": {"kegg.compound": "C00031"},
                    },
                    {
                        "id": "s_002",
                        "name": "ATP",
                        "compartment": "c",
                        "formula": "C10H16N5O13P3",
                        "charge": -4,
                        "deltaG": -2292.0,  # kJ/mol
                        "annotation": {"kegg.compound": "C00002"},
                    },
                    {
                        "id": "s_003",
                        "name": "Missing thermodynamics",
                        "compartment": "c",
                        "formula": "C1H1",
                        "charge": 0,
                        "deltaG": 10000000.0,  # Placeholder value
                    },
                ]
            },
            {
                "reactions": [
                    {
                        "id": "r_001",
                        "name": "Glucose transport",
                        "metabolites": {"s_001": -1, "s_002": 1},  # s_001 -> s_002
                        "lower_bound": 0,
                        "upper_bound": 1000,
                        "gene_reaction_rule": "gene1 or gene2",
                        "subsystem": ["Transport"],
                        "eccodes": ["1.1.1.1"],
                        "annotation": {"kegg.reaction": "R00001"},
                    },
                    {
                        "id": "r_002",
                        "name": "Reaction with missing metabolite",
                        "metabolites": {"s_002": -1, "s_003": 1, "s_missing": 1},
                        "lower_bound": -1000,
                        "upper_bound": 1000,
                        "gene_reaction_rule": "",
                        "subsystem": [],
                        "eccodes": [],
                    },
                ]
            },
        ]
        return yaml.dump(yaml_data, default_flow_style=False)

    def test_init_from_string(self):
        """Test initialization from YAML string."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        assert "metabolites" in parser.parsed_data
        assert "reactions" in parser.parsed_data
        assert "metaData" in parser.parsed_data

    def test_init_from_file(self):
        """Test initialization from YAML file."""
        yaml_content = self.create_test_yaml_content()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name

        try:
            parser = YAMLModelParser(temp_file)
            assert "metabolites" in parser.parsed_data
        finally:
            import os

            os.unlink(temp_file)

    def test_init_invalid_yaml(self):
        """Test initialization with invalid YAML."""
        with pytest.raises(YAMLParseError):
            YAMLModelParser("invalid: yaml: content: [")

    def test_init_missing_sections(self):
        """Test initialization with missing required sections."""
        # Missing metabolites section
        yaml_data = [{"reactions": []}]
        yaml_content = yaml.dump(yaml_data)

        with pytest.raises(YAMLParseError, match="No 'metabolites' section"):
            YAMLModelParser(yaml_content)

        # Missing reactions section
        yaml_data = [{"metabolites": []}]
        yaml_content = yaml.dump(yaml_data)

        with pytest.raises(YAMLParseError, match="No 'reactions' section"):
            YAMLModelParser(yaml_content)

    def test_get_metadata(self):
        """Test metadata extraction."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        metadata = parser.get_metadata()
        assert metadata["id"] == "test_model"
        assert metadata["name"] == "Test Model"
        assert metadata["version"] == "1.0"

    def test_get_species_info(self):
        """Test species information extraction."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        species_info = parser.get_species_info()

        assert "s_001" in species_info
        assert "s_002" in species_info
        assert "s_003" in species_info

        # Check glucose data
        glucose = species_info["s_001"]
        assert glucose["name"] == "Glucose"
        assert glucose["compartment"] == "c"
        assert glucose["formula"] == "C6H12O6"
        assert glucose["charge"] == 0
        assert glucose["delta_g"] == -915.0
        assert glucose["smiles"] == "OCC1OC(O)C(O)C(O)C1O"
        assert "kegg.compound" in glucose["annotation"]

        # Check missing thermodynamics
        missing = species_info["s_003"]
        assert missing["delta_g"] == 10000000.0

    def test_get_reaction_info(self):
        """Test reaction information extraction."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        reaction_info = parser.get_reaction_info()

        assert len(reaction_info) == 2

        # Check first reaction
        rxn1 = reaction_info[0]
        assert rxn1["id"] == "r_001"
        assert rxn1["name"] == "Glucose transport"
        assert rxn1["metabolites"] == {"s_001": -1, "s_002": 1}
        assert rxn1["reversible"] == False  # lower_bound = 0
        assert rxn1["gene_reaction_rule"] == "gene1 or gene2"
        assert rxn1["subsystem"] == ["Transport"]
        assert rxn1["eccodes"] == ["1.1.1.1"]

        # Check reactants and products
        assert ("s_001", 1) in rxn1["reactants"]
        assert ("s_002", 1) in rxn1["products"]

        # Check second reaction (reversible)
        rxn2 = reaction_info[1]
        assert rxn2["reversible"] == True  # lower_bound < 0

    def test_compute_equilibrium_constants(self):
        """Test equilibrium constant computation."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        keq_array, info = parser.compute_equilibrium_constants(verbose=False)

        assert len(keq_array) == 2

        # First reaction: s_001 -> s_002
        # ΔG°_rxn = (-2292) - (-915) = -1377 kJ/mol -> large Keq
        assert keq_array[0] > 1e200  # Very favorable reaction

        # Second reaction has missing metabolite -> default Keq = 1
        assert keq_array[1] == pytest.approx(1.0)

        # Check info
        assert info["n_reactions"] == 2
        assert info["reactions_with_complete_data"] == 1
        assert info["reactions_with_no_data"] == 1
        assert info["coverage_complete"] == 0.5

    def test_compute_equilibrium_constants_with_options(self):
        """Test equilibrium constant computation with different options."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        # Test with different default Keq
        keq_array, info = parser.compute_equilibrium_constants(default_keq=5.0, verbose=False)
        assert keq_array[1] == 5.0  # Reaction with missing data

        # Test with different temperature
        keq_298, _ = parser.compute_equilibrium_constants(temperature=298.15)
        keq_373, _ = parser.compute_equilibrium_constants(temperature=373.15)

        # Higher temperature -> smaller Keq for favorable reaction
        assert keq_373[0] < keq_298[0]

    def test_create_stoichiometric_matrix_dense(self):
        """Test dense stoichiometric matrix creation."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        species_info = parser.get_species_info()
        reaction_info = parser.get_reaction_info()
        species_ids = list(species_info.keys())

        S = parser.create_stoichiometric_matrix(species_ids, reaction_info, use_sparse=False)

        assert S.shape == (3, 2)  # 3 species, 2 reactions
        assert isinstance(S, np.ndarray)

        # Check stoichiometry for first reaction: s_001 -> s_002
        s001_idx = species_ids.index("s_001")
        s002_idx = species_ids.index("s_002")

        assert S[s001_idx, 0] == -1  # s_001 consumed
        assert S[s002_idx, 0] == 1  # s_002 produced

    def test_create_stoichiometric_matrix_sparse(self):
        """Test sparse stoichiometric matrix creation."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        species_info = parser.get_species_info()
        reaction_info = parser.get_reaction_info()
        species_ids = list(species_info.keys())

        S = parser.create_stoichiometric_matrix(species_ids, reaction_info, use_sparse=True)

        assert S.shape == (3, 2)
        from scipy import sparse

        assert sparse.issparse(S)

        # Convert to dense for checking values
        S_dense = S.toarray()
        s001_idx = species_ids.index("s_001")
        s002_idx = species_ids.index("s_002")

        assert S_dense[s001_idx, 0] == -1
        assert S_dense[s002_idx, 0] == 1

    def test_parse_complete(self):
        """Test complete parsing workflow."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        result = parser.parse(compute_keq=True, verbose=False)

        # Check all required fields
        required_fields = [
            "metadata",
            "species",
            "reactions",
            "species_ids",
            "reaction_ids",
            "stoichiometric_matrix",
            "equilibrium_constants",
            "keq_info",
        ]
        for field in required_fields:
            assert field in result

        # Check sizes
        assert len(result["species_ids"]) == 3
        assert len(result["reaction_ids"]) == 2
        assert result["stoichiometric_matrix"].shape == (3, 2)
        assert len(result["equilibrium_constants"]) == 2

        # Check metadata
        assert result["metadata"]["id"] == "test_model"

    def test_parse_without_keq(self):
        """Test parsing without equilibrium constant computation."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        result = parser.parse(compute_keq=False)

        assert "equilibrium_constants" not in result
        assert "keq_info" not in result

    @patch("builtins.print")  # Mock print to test verbose output
    def test_parse_verbose(self, mock_print):
        """Test verbose parsing output."""
        yaml_content = self.create_test_yaml_content()
        parser = YAMLModelParser(yaml_content)

        parser.parse(verbose=True)

        # Should have printed thermodynamic summary
        mock_print.assert_called()

        # Check if thermodynamic summary was printed
        call_args = [call.args[0] for call in mock_print.call_args_list]
        summary_printed = any("THERMODYNAMIC DATA SUMMARY" in str(arg) for arg in call_args)
        assert summary_printed


class TestLoadYamlModel:
    """Test convenience function for loading YAML models."""

    def create_test_yaml_file(self):
        """Create temporary YAML file for testing."""
        yaml_data = [
            {"metaData": {"id": "test", "name": "Test Model"}},
            {"metabolites": [{"id": "s_001", "name": "A", "compartment": "c", "deltaG": -100.0}]},
            {"reactions": [{"id": "r_001", "name": "Test reaction", "metabolites": {"s_001": -1}}]},
        ]

        yaml_content = yaml.dump(yaml_data, default_flow_style=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            return f.name

    def test_load_yaml_model_basic(self):
        """Test basic YAML model loading."""
        temp_file = self.create_test_yaml_file()

        try:
            result = load_yaml_model(temp_file, verbose=False)

            assert "species_ids" in result
            assert "reaction_ids" in result
            assert result["metadata"]["id"] == "test"
        finally:
            import os

            os.unlink(temp_file)

    def test_load_yaml_model_with_options(self):
        """Test YAML model loading with options."""
        temp_file = self.create_test_yaml_file()

        try:
            result = load_yaml_model(
                temp_file,
                compute_keq=True,
                temperature=310.15,  # Body temperature
                verbose=False,
            )

            assert "equilibrium_constants" in result
            assert result["keq_info"]["temperature_K"] == 310.15
        finally:
            import os

            os.unlink(temp_file)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_yaml_import_not_available(self):
        """Test behavior when PyYAML is not available."""
        with patch("llrq.yaml_parser.YAML_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyYAML is required"):
                YAMLModelParser("test")

    def test_empty_metabolites_list(self):
        """Test with empty metabolites list."""
        yaml_data = [{"metabolites": []}, {"reactions": []}]
        yaml_content = yaml.dump(yaml_data)

        parser = YAMLModelParser(yaml_content)
        species_info = parser.get_species_info()
        assert len(species_info) == 0

    def test_empty_reactions_list(self):
        """Test with empty reactions list."""
        yaml_data = [{"metabolites": [{"id": "s_001", "name": "Test"}]}, {"reactions": []}]
        yaml_content = yaml.dump(yaml_data)

        parser = YAMLModelParser(yaml_content)
        reaction_info = parser.get_reaction_info()
        assert len(reaction_info) == 0

    def test_missing_metabolite_id(self):
        """Test metabolite without ID."""
        yaml_data = [{"metabolites": [{"name": "No ID metabolite"}]}, {"reactions": []}]
        yaml_content = yaml.dump(yaml_data)

        parser = YAMLModelParser(yaml_content)
        species_info = parser.get_species_info()
        assert len(species_info) == 0  # Should skip metabolites without ID

    def test_missing_reaction_id(self):
        """Test reaction without ID."""
        yaml_data = [{"metabolites": []}, {"reactions": [{"name": "No ID reaction"}]}]
        yaml_content = yaml.dump(yaml_data)

        parser = YAMLModelParser(yaml_content)
        reaction_info = parser.get_reaction_info()
        assert len(reaction_info) == 0  # Should skip reactions without ID

    def test_malformed_data_structure(self):
        """Test malformed data structure."""
        # Non-dictionary items in lists should be skipped
        yaml_data = [{"metabolites": ["string_instead_of_dict", {"id": "s_001", "name": "Valid"}]}, {"reactions": []}]
        yaml_content = yaml.dump(yaml_data)

        parser = YAMLModelParser(yaml_content)
        species_info = parser.get_species_info()
        assert len(species_info) == 1  # Should only get the valid metabolite
        assert "s_001" in species_info


if __name__ == "__main__":
    pytest.main([__file__])
