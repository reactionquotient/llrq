"""YAML parser for log-linear reaction quotient dynamics.

This module provides functionality to parse YAML models (like yeast-GEM.yml)
and extract reaction network information with thermodynamic data needed
for log-linear dynamics.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
import yaml  # type: ignore

from .utils.thermodynamics import metabolite_delta_g_to_reaction_keq


class YAMLParseError(Exception):
    """Exception raised when YAML parsing fails."""

    pass


class YAMLModelParser:
    """Parser for YAML models to extract reaction network information.

    This class parses YAML files (like yeast-GEM.yml) and extracts the
    information needed to construct log-linear reaction quotient dynamics:
    - Species and their thermodynamic properties (ΔG° values)
    - Reactions and stoichiometry
    - Computed equilibrium constants from thermodynamic data
    """

    def __init__(self, yaml_file: str):
        """Initialize parser with YAML file.

        Args:
            yaml_file: Path to YAML file or YAML string content
        """

        # Load YAML data
        try:
            # Check if it's YAML content or file path
            if (
                "\n" in yaml_file
                or yaml_file.strip().startswith("---")
                or yaml_file.strip().startswith("!!omap")
                or yaml_file.strip().startswith("-")
            ):
                # Input is YAML string content
                self.data = yaml.safe_load(yaml_file)
            else:
                # Input is a file path
                with open(yaml_file, "r") as f:
                    self.data = yaml.safe_load(f)
        except Exception as e:
            raise YAMLParseError(f"Failed to load YAML file: {e}")

        if not isinstance(self.data, list):
            raise YAMLParseError("Expected YAML data to be a list (!!omap format)")

        # Parse the ordered map structure
        self.parsed_data = {}
        for item in self.data:
            if isinstance(item, dict):
                self.parsed_data.update(item)
            elif hasattr(item, "items"):  # Handle OrderedDict from !!omap
                for key, value in item.items():
                    self.parsed_data[key] = value
            elif isinstance(item, tuple) and len(item) == 2:  # Handle tuple from !!omap
                key, value = item
                self.parsed_data[key] = value

        # Validate required sections
        if "metabolites" not in self.parsed_data:
            raise YAMLParseError("No 'metabolites' section found in YAML")
        if "reactions" not in self.parsed_data:
            raise YAMLParseError("No 'reactions' section found in YAML")

    def get_metadata(self) -> Dict[str, Any]:
        """Extract model metadata.

        Returns:
            Dictionary with model metadata (id, name, version, etc.)
        """
        return self.parsed_data.get("metaData", {})

    def get_species_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract species information from YAML model.

        Returns:
            Dictionary mapping species ID to species information:
            - 'name': species name
            - 'compartment': compartment ID
            - 'formula': chemical formula
            - 'charge': electrical charge
            - 'delta_g': Gibbs free energy of formation (kJ/mol)
            - 'smiles': SMILES string (if available)
            - 'annotation': external database annotations
        """
        species_info = {}
        metabolites = self.parsed_data.get("metabolites", [])

        for metabolite in metabolites:
            # Handle different YAML formats: dict, OrderedDict, or list of tuples (!!omap)
            metabolite_data = {}
            if isinstance(metabolite, dict):
                metabolite_data = metabolite
            elif hasattr(metabolite, "items"):  # OrderedDict from !!omap
                for key, value in metabolite.items():
                    metabolite_data[key] = value
            elif isinstance(metabolite, list):  # List of tuples from !!omap
                for item in metabolite:
                    if isinstance(item, tuple) and len(item) == 2:
                        key, value = item
                        metabolite_data[key] = value
            else:
                continue

            # Get species ID
            species_id = metabolite_data.get("id")
            if not species_id:
                continue

            # Extract information
            species_data = {
                "name": metabolite_data.get("name", species_id),
                "compartment": metabolite_data.get("compartment", "c"),
                "formula": metabolite_data.get("formula", ""),
                "charge": metabolite_data.get("charge", 0),
                "smiles": metabolite_data.get("smiles", ""),
                "annotation": metabolite_data.get("annotation", {}),
                "initial_concentration": 0.0,  # Not typically specified in yeast-GEM
                "boundary_condition": False,
            }

            # Extract ΔG° value if available
            if "deltaG" in metabolite_data:
                species_data["delta_g"] = float(metabolite_data["deltaG"])
            else:
                species_data["delta_g"] = None

            species_info[species_id] = species_data

        return species_info

    def get_reaction_info(self) -> List[Dict[str, Any]]:
        """Extract reaction information from YAML model.

        Returns:
            List of reaction dictionaries, each containing:
            - 'id': reaction identifier
            - 'name': human readable name
            - 'metabolites': stoichiometric coefficients {species_id: coeff}
            - 'lower_bound': minimum flux bound
            - 'upper_bound': maximum flux bound
            - 'gene_reaction_rule': gene association rule
            - 'subsystem': metabolic subsystem(s)
            - 'ec_codes': enzyme commission codes
            - 'annotation': external database annotations
            - 'reversible': whether reaction is reversible
            - 'reactants': list of (species_id, stoichiometry) for reactants
            - 'products': list of (species_id, stoichiometry) for products
        """
        reaction_info = []
        reactions = self.parsed_data.get("reactions", [])

        for reaction in reactions:
            # Handle different YAML formats: dict, OrderedDict, or list of tuples (!!omap)
            reaction_data = {}
            if isinstance(reaction, dict):
                reaction_data = reaction
            elif hasattr(reaction, "items"):  # OrderedDict from !!omap
                for key, value in reaction.items():
                    reaction_data[key] = value
            elif isinstance(reaction, list):  # List of tuples from !!omap
                for item in reaction:
                    if isinstance(item, tuple) and len(item) == 2:
                        key, value = item
                        reaction_data[key] = value
            else:
                continue

            # Get reaction ID
            rxn_id = reaction_data.get("id")
            if not rxn_id:
                continue

            # Get metabolites and stoichiometry
            metabolites = reaction_data.get("metabolites", {})

            # Handle different metabolites formats
            if hasattr(metabolites, "items"):
                metabolites = dict(metabolites.items())
            elif isinstance(metabolites, list):  # List of tuples from !!omap
                metabolites_dict = {}
                for item in metabolites:
                    if isinstance(item, tuple) and len(item) == 2:
                        species_id, coeff = item
                        metabolites_dict[species_id] = float(coeff)  # Convert to float
                metabolites = metabolites_dict

            # Separate into reactants and products
            reactants = []
            products = []
            for species_id, coeff in metabolites.items():
                # Convert coefficient to float (may be string from YAML)
                coeff = float(coeff)
                if coeff < 0:
                    reactants.append((species_id, abs(coeff)))
                elif coeff > 0:
                    products.append((species_id, coeff))

            # Determine if reversible (check bounds)
            lower_bound = reaction_data.get("lower_bound", 0)
            upper_bound = reaction_data.get("upper_bound", 1000)
            reversible = lower_bound < 0

            # Extract additional information
            rxn_data = {
                "id": rxn_id,
                "name": reaction_data.get("name", rxn_id),
                "metabolites": metabolites,
                "reactants": reactants,
                "products": products,
                "reversible": reversible,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "gene_reaction_rule": reaction_data.get("gene_reaction_rule", ""),
                "subsystem": reaction_data.get("subsystem", []),
                "eccodes": reaction_data.get("eccodes", []),
                "annotation": reaction_data.get("annotation", {}),
                "kinetic_law": None,  # YAML format doesn't include kinetic laws
            }

            reaction_info.append(rxn_data)

        return reaction_info

    def compute_equilibrium_constants(
        self, temperature: float = 298.15, missing_value: float = 10000000.0, default_keq: float = 1.0, verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Compute equilibrium constants for all reactions using thermodynamic data.

        Args:
            temperature: Temperature in Kelvin for Keq computation
            missing_value: Placeholder value indicating missing ΔG° data
            default_keq: Default Keq for reactions with incomplete data
            verbose: Whether to print warnings about missing data

        Returns:
            Tuple of (keq_array, info):
            - keq_array: Array of equilibrium constants for each reaction
            - info: Dictionary with statistics about thermodynamic data coverage
        """
        # Get species thermodynamic data
        species_info = self.get_species_info()
        metabolite_delta_g = {}

        for species_id, info in species_info.items():
            if info["delta_g"] is not None:
                metabolite_delta_g[species_id] = info["delta_g"]

        # Get reaction information
        reactions = self.get_reaction_info()

        # Compute Keq values using thermodynamic utilities
        keq_array, info = metabolite_delta_g_to_reaction_keq(
            metabolite_delta_g=metabolite_delta_g,
            reactions=reactions,
            T=temperature,
            delta_g_units="kJ/mol",  # yeast-GEM uses kJ/mol
            missing_value=missing_value,
            default_keq=default_keq,
            verbose=verbose,
        )

        return keq_array, info

    def create_stoichiometric_matrix(
        self, species_ids: List[str], reactions: List[Dict[str, Any]], use_sparse: Optional[bool] = None
    ) -> Union[np.ndarray, sparse.spmatrix]:
        """Create stoichiometric matrix from reaction information.

        Args:
            species_ids: List of species IDs (rows of matrix)
            reactions: List of reaction dictionaries
            use_sparse: Force sparse (True) or dense (False) format. If None, auto-detect

        Returns:
            Stoichiometric matrix S where S[i,j] is stoichiometry of
            species i in reaction j (products positive, reactants negative)
        """
        n_species = len(species_ids)
        n_reactions = len(reactions)

        species_to_idx = {species_id: i for i, species_id in enumerate(species_ids)}

        # Count non-zero entries to decide on sparse vs dense
        if use_sparse is None:
            non_zero_count = 0
            for reaction in reactions:
                non_zero_count += len(reaction["metabolites"])

            total_entries = n_species * n_reactions
            sparsity = 1 - (non_zero_count / total_entries) if total_entries > 0 else 0
            use_sparse = sparsity > 0.95

        if use_sparse:
            # Build sparse matrix using COO format
            row_indices = []
            col_indices = []
            data = []

            for j, reaction in enumerate(reactions):
                metabolites = reaction["metabolites"]
                for species_id, stoich in metabolites.items():
                    if species_id in species_to_idx:
                        i = species_to_idx[species_id]
                        row_indices.append(i)
                        col_indices.append(j)
                        data.append(float(stoich))  # Convert to float, already has correct sign

            return sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_species, n_reactions), dtype=float).tocsr()

        else:
            # Build dense matrix
            S = np.zeros((n_species, n_reactions), dtype=float)

            for j, reaction in enumerate(reactions):
                metabolites = reaction["metabolites"]
                for species_id, stoich in metabolites.items():
                    if species_id in species_to_idx:
                        i = species_to_idx[species_id]
                        S[i, j] = float(stoich)  # Convert to float, already has correct sign

            return S

    def parse(
        self,
        compute_keq: bool = True,
        temperature: float = 298.15,
        missing_value: float = 10000000.0,
        default_keq: float = 1.0,
        use_sparse: Optional[bool] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Parse complete YAML model and return structured data.

        Args:
            compute_keq: Whether to compute equilibrium constants from ΔG° data
            temperature: Temperature for Keq computation (K)
            missing_value: Placeholder for missing ΔG° data
            default_keq: Default Keq for incomplete thermodynamic data
            use_sparse: Use sparse matrices for stoichiometry
            verbose: Print warnings about missing data

        Returns:
            Dictionary containing:
            - 'metadata': model metadata
            - 'species': species information dictionary
            - 'reactions': list of reaction dictionaries
            - 'species_ids': ordered list of species IDs
            - 'reaction_ids': ordered list of reaction IDs
            - 'stoichiometric_matrix': stoichiometric matrix S
            - 'equilibrium_constants': Keq array (if compute_keq=True)
            - 'keq_info': thermodynamic data statistics (if compute_keq=True)
        """
        # Extract basic information
        metadata = self.get_metadata()
        species_info = self.get_species_info()
        reaction_info = self.get_reaction_info()

        # Create ordered lists
        species_ids = list(species_info.keys())
        reaction_ids = [rxn["id"] for rxn in reaction_info]

        # Create stoichiometric matrix
        stoich_matrix = self.create_stoichiometric_matrix(species_ids, reaction_info, use_sparse)

        # Prepare result
        result = {
            "metadata": metadata,
            "species": species_info,
            "reactions": reaction_info,
            "species_ids": species_ids,
            "reaction_ids": reaction_ids,
            "stoichiometric_matrix": stoich_matrix,
        }

        # Compute equilibrium constants if requested
        if compute_keq:
            keq_array, keq_info = self.compute_equilibrium_constants(
                temperature=temperature, missing_value=missing_value, default_keq=default_keq, verbose=verbose
            )
            result["equilibrium_constants"] = keq_array
            result["keq_info"] = keq_info

            # Print summary if verbose
            if verbose:
                self._print_thermodynamic_summary(keq_info)

        return result

    def _print_thermodynamic_summary(self, keq_info: Dict):
        """Print summary of thermodynamic data coverage."""
        print("\n" + "=" * 60)
        print("THERMODYNAMIC DATA SUMMARY")
        print("=" * 60)
        print(f"Total reactions: {keq_info['n_reactions']:,}")
        print(
            f"Reactions with complete thermodynamic data: {keq_info['reactions_with_complete_data']:,} "
            f"({keq_info['coverage_complete']:.1%})"
        )
        print(f"Reactions with partial thermodynamic data: {keq_info['reactions_with_partial_data']:,}")
        print(
            f"Reactions using default Keq = {keq_info['default_keq_used']}: {keq_info['reactions_with_no_data']:,} "
            f"({(1-keq_info['coverage_partial']):.1%})"
        )
        print(f"Unique metabolites missing ΔG° data: {keq_info['unique_missing_metabolites']:,}")
        print(f"Temperature used: {keq_info['temperature_K']} K")
        print("=" * 60)


def load_yaml_model(
    yaml_file: str, compute_keq: bool = True, temperature: float = 298.15, verbose: bool = False, **kwargs
) -> Dict[str, Any]:
    """Convenience function to load and parse a YAML model.

    Args:
        yaml_file: Path to YAML file
        compute_keq: Whether to compute equilibrium constants
        temperature: Temperature for Keq computation (K)
        verbose: Print parsing information
        **kwargs: Additional arguments passed to parser.parse()

    Returns:
        Parsed model data dictionary

    Examples:
        >>> # Load yeast-GEM with thermodynamic data
        >>> model_data = load_yaml_model('models/yeast-GEM.yml', verbose=True)
        >>> print(f"Loaded {len(model_data['species_ids'])} species")
        >>> print(f"Computed Keq for {len(model_data['reaction_ids'])} reactions")
    """
    parser = YAMLModelParser(yaml_file)
    return parser.parse(compute_keq=compute_keq, temperature=temperature, verbose=verbose, **kwargs)
