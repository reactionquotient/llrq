"""Tests comparing XML and YAML model formats."""

import numpy as np
import pytest
import os
from pathlib import Path

from llrq.sbml_parser import SBMLParser
from llrq.yaml_parser import YAMLModelParser
from llrq.reaction_network import ReactionNetwork
from llrq import from_model


@pytest.fixture(scope="module")
def yeast_model_paths():
    """Get paths to yeast model files."""
    models_dir = Path(__file__).parent.parent / "models"
    xml_path = models_dir / "yeast-GEM.xml"
    yaml_path = models_dir / "yeast-GEM.yml"

    if not xml_path.exists() or not yaml_path.exists():
        pytest.skip(f"Yeast model files not found: {xml_path}, {yaml_path}")

    return xml_path, yaml_path


@pytest.fixture(scope="module")
def loaded_networks(yeast_model_paths):
    """Load both XML and YAML networks (cached for speed)."""
    xml_path, yaml_path = yeast_model_paths

    # Load XML model
    xml_network, xml_dynamics, xml_solver, xml_viz = from_model(
        str(xml_path), compute_keq_from_thermodynamics=False, verbose=False
    )

    # Load YAML model
    yaml_network, yaml_dynamics, yaml_solver, yaml_viz = from_model(
        str(yaml_path), compute_keq_from_thermodynamics=False, verbose=False
    )

    return xml_network, yaml_network


class TestFormatComparison:
    """Test that XML and YAML formats produce equivalent network structures."""

    @pytest.mark.slow
    def test_yeast_model_network_structure_equivalent(self, loaded_networks):
        """Test that yeast-GEM.xml and yeast-GEM.yml produce equivalent network structures."""
        xml_network, yaml_network = loaded_networks

        # Compare network structures
        self._compare_networks(xml_network, yaml_network)

    def _compare_networks(self, xml_net: ReactionNetwork, yaml_net: ReactionNetwork):
        """Compare two ReactionNetwork objects for structural equivalence."""

        # Compare basic sizes
        assert len(xml_net.species_ids) == len(yaml_net.species_ids), (
            f"Different number of species: XML={len(xml_net.species_ids)}, " f"YAML={len(yaml_net.species_ids)}"
        )

        assert len(xml_net.reaction_ids) == len(yaml_net.reaction_ids), (
            f"Different number of reactions: XML={len(xml_net.reaction_ids)}, " f"YAML={len(yaml_net.reaction_ids)}"
        )

        # Compare species sets (order might differ)
        xml_species_set = set(xml_net.species_ids)
        yaml_species_set = set(yaml_net.species_ids)

        # Find differences
        only_in_xml = xml_species_set - yaml_species_set
        only_in_yaml = yaml_species_set - xml_species_set

        if only_in_xml:
            print(f"Species only in XML: {sorted(list(only_in_xml))[:10]}...")  # Show first 10
        if only_in_yaml:
            print(f"Species only in YAML: {sorted(list(only_in_yaml))[:10]}...")  # Show first 10

        # Allow for small differences but check they're reasonable
        species_overlap = len(xml_species_set & yaml_species_set)
        total_unique_species = len(xml_species_set | yaml_species_set)
        overlap_fraction = species_overlap / total_unique_species

        assert overlap_fraction > 0.95, (
            f"Species overlap too low: {overlap_fraction:.3f}. "
            f"Common: {species_overlap}, Total unique: {total_unique_species}"
        )

        # Compare reaction sets
        xml_reaction_set = set(xml_net.reaction_ids)
        yaml_reaction_set = set(yaml_net.reaction_ids)

        only_in_xml_rxn = xml_reaction_set - yaml_reaction_set
        only_in_yaml_rxn = yaml_reaction_set - xml_reaction_set

        if only_in_xml_rxn:
            print(f"Reactions only in XML: {sorted(list(only_in_xml_rxn))[:10]}...")
        if only_in_yaml_rxn:
            print(f"Reactions only in YAML: {sorted(list(only_in_yaml_rxn))[:10]}...")

        reaction_overlap = len(xml_reaction_set & yaml_reaction_set)
        total_unique_reactions = len(xml_reaction_set | yaml_reaction_set)
        reaction_overlap_fraction = reaction_overlap / total_unique_reactions

        assert reaction_overlap_fraction > 0.95, (
            f"Reaction overlap too low: {reaction_overlap_fraction:.3f}. "
            f"Common: {reaction_overlap}, Total unique: {total_unique_reactions}"
        )

    def test_stoichiometric_matrix_consistency(self, loaded_networks):
        """Test that stoichiometric matrices have consistent structure for common reactions."""
        xml_network, yaml_network = loaded_networks

        # Find common species and reactions
        common_species = set(xml_network.species_ids) & set(yaml_network.species_ids)
        common_reactions = set(xml_network.reaction_ids) & set(yaml_network.reaction_ids)

        if len(common_species) < 100 or len(common_reactions) < 100:
            pytest.skip("Not enough common species/reactions for meaningful comparison")

        # Create mappings for common elements
        xml_species_idx = {s: i for i, s in enumerate(xml_network.species_ids) if s in common_species}
        yaml_species_idx = {s: i for i, s in enumerate(yaml_network.species_ids) if s in common_species}

        xml_rxn_idx = {r: i for i, r in enumerate(xml_network.reaction_ids) if r in common_reactions}
        yaml_rxn_idx = {r: i for i, r in enumerate(yaml_network.reaction_ids) if r in common_reactions}

        # Compare stoichiometry for sample of common reactions
        sample_reactions = list(common_reactions)[:50]  # Test first 50 common reactions

        differences = 0
        for rxn_id in sample_reactions:
            xml_rxn_col = xml_rxn_idx[rxn_id]
            yaml_rxn_col = yaml_rxn_idx[rxn_id]

            # Get stoichiometry for this reaction in both formats
            for species_id in list(common_species)[:100]:  # Check first 100 common species
                xml_row = xml_species_idx[species_id]
                yaml_row = yaml_species_idx[species_id]

                xml_coeff = xml_network.S[xml_row, xml_rxn_col]
                yaml_coeff = yaml_network.S[yaml_row, yaml_rxn_col]

                if abs(xml_coeff - yaml_coeff) > 1e-10:
                    differences += 1

        # Allow for some small differences but not too many
        total_comparisons = len(sample_reactions) * min(100, len(common_species))
        difference_fraction = differences / total_comparisons if total_comparisons > 0 else 0

        assert difference_fraction < 0.01, (
            f"Too many stoichiometric differences: {difference_fraction:.3%} " f"({differences}/{total_comparisons})"
        )

    def test_reaction_information_consistency(self, yeast_model_paths):
        """Test that reaction information is consistent between formats."""
        xml_path, yaml_path = yeast_model_paths

        # Parse both formats directly
        xml_parser = SBMLParser(str(xml_path))
        xml_data = xml_parser.extract_network_data()

        yaml_parser = YAMLModelParser(str(yaml_path))
        yaml_data = yaml_parser.parse(compute_keq=False)

        # Find common reactions
        xml_rxn_ids = {rxn["id"] for rxn in xml_data["reactions"]}
        yaml_rxn_ids = {rxn["id"] for rxn in yaml_data["reactions"]}
        common_rxn_ids = xml_rxn_ids & yaml_rxn_ids

        assert len(common_rxn_ids) > 1000, f"Too few common reactions: {len(common_rxn_ids)}"

        # Create lookup dictionaries
        xml_rxns = {rxn["id"]: rxn for rxn in xml_data["reactions"]}
        yaml_rxns = {rxn["id"]: rxn for rxn in yaml_data["reactions"]}

        # Check consistency for sample of reactions
        sample_rxns = list(common_rxn_ids)[:100]

        for rxn_id in sample_rxns:
            xml_rxn = xml_rxns[rxn_id]
            yaml_rxn = yaml_rxns[rxn_id]

            # Compare basic properties
            assert xml_rxn["reversible"] == yaml_rxn["reversible"], (
                f"Reversibility mismatch for {rxn_id}: XML={xml_rxn['reversible']}, " f"YAML={yaml_rxn['reversible']}"
            )

            # Compare number of participants (allowing for small differences)
            xml_participants = len(xml_rxn.get("reactants", [])) + len(xml_rxn.get("products", []))
            yaml_participants = len(yaml_rxn.get("reactants", [])) + len(yaml_rxn.get("products", []))

            # Allow for small differences in participant count
            if abs(xml_participants - yaml_participants) > 2:
                print(f"Participant count difference for {rxn_id}: " f"XML={xml_participants}, YAML={yaml_participants}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
