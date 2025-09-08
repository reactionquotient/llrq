"""Genome-scale model utilities for LLRQ analysis.

This module provides specialized tools and utilities for working with
large-scale metabolic models, including efficiency optimizations,
model subsetting, and compartment-aware analysis.
"""

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy import sparse
import time

from .sbml_parser import SBMLParser, SBMLParseError
from .reaction_network import ReactionNetwork
from .llrq_dynamics import LLRQDynamics


class GenomeScaleAnalyzer:
    """Analyzer for genome-scale metabolic models with performance optimizations.

    This class provides methods for efficient analysis of large-scale models:
    - Streaming SBML parsing for very large files
    - Model subsetting and pathway extraction
    - Compartment-aware analysis
    - Numerical stability checks
    - Performance monitoring
    """

    def __init__(self, sbml_file: str, lazy_load: bool = True):
        """Initialize genome-scale analyzer.

        Args:
            sbml_file: Path to SBML file
            lazy_load: If True, delay loading detailed information until needed

        Raises:
            FileNotFoundError: If SBML file does not exist
        """
        import os

        if not os.path.exists(sbml_file):
            raise FileNotFoundError(f"SBML file not found: {sbml_file}")

        self.sbml_file = sbml_file
        self.lazy_load = lazy_load
        self._parser: Optional[SBMLParser] = None
        self._network_data: Optional[Dict[str, Any]] = None
        self._network: Optional[ReactionNetwork] = None

        # Performance metrics
        self.performance_metrics: Dict[str, float] = {}

        # Model statistics
        self._model_stats: Optional[Dict[str, Any]] = None

        if not lazy_load:
            self._load_all()

    def _load_all(self):
        """Load all model data immediately."""
        start_time = time.time()
        self._parser = SBMLParser(self.sbml_file)
        self._network_data = self._parser.extract_network_data()
        self.performance_metrics["full_load_time"] = time.time() - start_time

    @property
    def parser(self) -> SBMLParser:
        """Get SBML parser, loading if needed."""
        if self._parser is None:
            start_time = time.time()
            self._parser = SBMLParser(self.sbml_file)
            self.performance_metrics["parser_load_time"] = time.time() - start_time
        return self._parser

    @property
    def network_data(self) -> Dict[str, Any]:
        """Get network data, loading if needed."""
        if self._network_data is None:
            start_time = time.time()
            self._network_data = self.parser.extract_network_data()
            self.performance_metrics["network_data_time"] = time.time() - start_time
        return self._network_data

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        if self._model_stats is None:
            data = self.network_data

            # Basic counts
            n_species = len(data["species_ids"])
            n_reactions = len(data["reaction_ids"])

            # Reaction analysis
            reactions = data["reactions"]
            reversible_count = sum(1 for r in reactions if r["reversible"])
            with_bounds_count = sum(1 for r in reactions if r.get("fbc_bounds"))
            with_genes_count = sum(1 for r in reactions if r.get("gene_association"))

            # Species analysis
            species_info = data["species"]
            compartments = set(s["compartment"] for s in species_info.values())
            boundary_species = sum(1 for s in species_info.values() if s["boundary_condition"])

            # Matrix properties
            S = data["stoichiometric_matrix"]
            if sparse.issparse(S):
                non_zeros = S.nnz
                sparsity = 1 - (non_zeros / (S.shape[0] * S.shape[1]))
                matrix_type = "sparse"
                memory_mb = S.data.nbytes / (1024 * 1024)
            else:
                non_zeros = np.count_nonzero(S)
                sparsity = 1 - (non_zeros / S.size)
                matrix_type = "dense"
                memory_mb = S.nbytes / (1024 * 1024)

            # FBC information
            fbc_objectives = data.get("fbc_objectives", {})
            gene_products = data.get("gene_products", {})

            self._model_stats = {
                # Basic structure
                "n_species": n_species,
                "n_reactions": n_reactions,
                "n_compartments": len(compartments),
                "compartments": sorted(compartments),
                # Reaction properties
                "reversible_reactions": reversible_count,
                "irreversible_reactions": n_reactions - reversible_count,
                "reactions_with_bounds": with_bounds_count,
                "reactions_with_genes": with_genes_count,
                # Species properties
                "boundary_species": boundary_species,
                "internal_species": n_species - boundary_species,
                # Matrix properties
                "matrix_type": matrix_type,
                "non_zero_elements": int(non_zeros),
                "sparsity": sparsity,
                "matrix_memory_mb": memory_mb,
                # FBC properties
                "n_objectives": len(fbc_objectives.get("objectives", [])),
                "active_objective": fbc_objectives.get("active_objective"),
                "n_gene_products": len(gene_products),
                # Performance metrics
                "performance": self.performance_metrics.copy(),
            }

        return self._model_stats

    def create_network(self, use_sparse: Optional[bool] = None) -> ReactionNetwork:
        """Create ReactionNetwork with optimal settings for genome-scale models."""
        # Check if we need to create/recreate network
        recreate_network = self._network is None or (use_sparse is not None and self._network.is_sparse != use_sparse)

        if recreate_network:
            start_time = time.time()
            data = self.network_data

            # Auto-detect sparse usage if not specified
            if use_sparse is None:
                S = data["stoichiometric_matrix"]
                if sparse.issparse(S):
                    use_sparse = True
                else:
                    sparsity = 1 - (np.count_nonzero(S) / S.size)
                    use_sparse = sparsity > 0.95

            self._network = ReactionNetwork(
                species_ids=data["species_ids"],
                reaction_ids=data["reaction_ids"],
                stoichiometric_matrix=data["stoichiometric_matrix"],
                species_info=data["species"],
                reaction_info=data["reactions"],
                parameters=data["parameters"],
                use_sparse=use_sparse,
            )

            self.performance_metrics["network_creation_time"] = time.time() - start_time

        assert self._network is not None
        return self._network

    def extract_compartment_submodel(self, compartment_ids: Union[str, List[str]]) -> "GenomeScaleAnalyzer":
        """Extract submodel containing only specified compartments.

        Args:
            compartment_ids: Single compartment ID or list of compartment IDs

        Returns:
            New GenomeScaleAnalyzer with submodel
        """
        if isinstance(compartment_ids, str):
            compartment_ids = [compartment_ids]

        data = self.network_data
        species_info = data["species"]

        # Filter species by compartment
        filtered_species = {
            species_id: info for species_id, info in species_info.items() if info["compartment"] in compartment_ids
        }

        if not filtered_species:
            raise ValueError(f"No species found in compartments: {compartment_ids}")

        # Create submodel
        return self._create_submodel_from_species(list(filtered_species.keys()))

    def extract_pathway_submodel(self, reaction_ids: List[str], include_connected: bool = True) -> "GenomeScaleAnalyzer":
        """Extract submodel containing specified reactions and their species.

        Args:
            reaction_ids: List of reaction IDs to include
            include_connected: If True, include reactions connected to the same species

        Returns:
            New GenomeScaleAnalyzer with submodel
        """
        data = self.network_data
        reactions = data["reactions"]

        # Find reactions to include
        reaction_map = {r["id"]: r for r in reactions}
        selected_reactions = []
        involved_species = set()

        # Add specified reactions
        for rxn_id in reaction_ids:
            if rxn_id in reaction_map:
                rxn = reaction_map[rxn_id]
                selected_reactions.append(rxn)
                # Collect species
                for species_id, _ in rxn["reactants"] + rxn["products"]:
                    involved_species.add(species_id)

        if include_connected:
            # Add reactions that involve any of the collected species
            for rxn in reactions:
                if rxn["id"] not in reaction_ids:
                    rxn_species = set()
                    for species_id, _ in rxn["reactants"] + rxn["products"]:
                        rxn_species.add(species_id)

                    # If this reaction shares species with our pathway
                    if rxn_species & involved_species:
                        selected_reactions.append(rxn)
                        involved_species.update(rxn_species)

        # Create submodel
        return self._create_submodel_from_reactions(selected_reactions)

    def _create_submodel_from_species(self, species_ids: List[str]) -> "GenomeScaleAnalyzer":
        """Create submodel from specified species."""
        data = self.network_data

        # Filter reactions that involve these species
        selected_reactions = []
        for rxn in data["reactions"]:
            rxn_species = set()
            for species_id, _ in rxn["reactants"] + rxn["products"]:
                rxn_species.add(species_id)

            if rxn_species & set(species_ids):
                selected_reactions.append(rxn)

        return self._create_submodel_from_reactions(selected_reactions)

    def _create_submodel_from_reactions(self, reactions: List[Dict[str, Any]]) -> "GenomeScaleAnalyzer":
        """Create submodel from specified reactions."""
        if not reactions:
            raise ValueError("No reactions selected for submodel")

        # Collect all species involved in these reactions
        involved_species = set()
        for rxn in reactions:
            for species_id, _ in rxn["reactants"] + rxn["products"]:
                involved_species.add(species_id)

        species_ids = sorted(involved_species)
        reaction_ids = [r["id"] for r in reactions]

        # Create filtered species info
        data = self.network_data
        species_info = {species_id: data["species"][species_id] for species_id in species_ids if species_id in data["species"]}

        # Create new stoichiometric matrix
        parser = SBMLParser.__new__(SBMLParser)  # Create without initialization
        S_sub = parser.create_stoichiometric_matrix(species_ids, reactions, use_sparse=True)

        # Create submodel data
        submodel_data = {
            "species": species_info,
            "reactions": reactions,
            "parameters": data["parameters"],  # Keep all parameters
            "stoichiometric_matrix": S_sub,
            "species_ids": species_ids,
            "reaction_ids": reaction_ids,
            "fbc_objectives": data.get("fbc_objectives", {}),  # Keep objectives
            "gene_products": data.get("gene_products", {}),  # Keep gene products
        }

        # Create new analyzer with submodel data
        analyzer = GenomeScaleAnalyzer.__new__(GenomeScaleAnalyzer)
        analyzer.sbml_file = f"{self.sbml_file}_submodel"
        analyzer.lazy_load = False
        analyzer._parser = None
        analyzer._network_data = submodel_data
        analyzer._network = None
        analyzer.performance_metrics = {}
        analyzer._model_stats = None

        return analyzer

    def check_numerical_stability(self) -> Dict[str, Any]:
        """Check for potential numerical stability issues."""
        data = self.network_data
        S = data["stoichiometric_matrix"]

        # Convert to dense for analysis if needed
        if sparse.issparse(S):
            S_dense = S.toarray()
        else:
            S_dense = S

        # Check stoichiometric coefficients
        max_coeff = np.max(np.abs(S_dense))
        min_nonzero = np.min(np.abs(S_dense[S_dense != 0])) if np.any(S_dense != 0) else np.inf
        coeff_range = max_coeff / min_nonzero if min_nonzero > 0 else np.inf

        # Check for very large/small coefficients
        large_coeffs = np.sum(np.abs(S_dense) > 1000)
        small_coeffs = np.sum((np.abs(S_dense) > 0) & (np.abs(S_dense) < 1e-6))

        # Check condition number (expensive for large matrices)
        condition_number = None
        try:
            if min(S.shape) < 1000:  # Only for smaller matrices
                condition_number = np.linalg.cond(S_dense)
        except:
            pass

        # Check for isolated species
        species_degrees = np.sum(np.abs(S_dense), axis=1)
        isolated_species = np.sum(species_degrees == 0)

        # Check for isolated reactions
        reaction_degrees = np.sum(np.abs(S_dense), axis=0)
        empty_reactions = np.sum(reaction_degrees == 0)

        # Check reaction bounds for extreme values
        extreme_bounds = 0
        unbounded_reactions = 0

        for rxn in data["reactions"]:
            bounds = rxn.get("fbc_bounds")
            if bounds:
                lower, upper = bounds
                if lower is not None and abs(lower) > 1e6:
                    extreme_bounds += 1
                if upper is not None and abs(upper) > 1e6:
                    extreme_bounds += 1
                if lower is None and upper is None:
                    unbounded_reactions += 1

        return {
            "max_stoichiometric_coeff": float(max_coeff),
            "min_nonzero_stoichiometric_coeff": float(min_nonzero),
            "stoichiometric_range": float(coeff_range),
            "large_coefficients": int(large_coeffs),
            "small_coefficients": int(small_coeffs),
            "condition_number": float(condition_number) if condition_number else None,
            "isolated_species": int(isolated_species),
            "empty_reactions": int(empty_reactions),
            "extreme_flux_bounds": int(extreme_bounds),
            "unbounded_reactions": int(unbounded_reactions),
            "stability_warnings": self._generate_stability_warnings(
                {
                    "max_coeff": max_coeff,
                    "coeff_range": coeff_range,
                    "large_coeffs": large_coeffs,
                    "small_coeffs": small_coeffs,
                    "isolated_species": isolated_species,
                    "empty_reactions": empty_reactions,
                    "condition_number": condition_number,
                }
            ),
        }

    def _generate_stability_warnings(self, stats: Dict) -> List[str]:
        """Generate numerical stability warnings."""
        warnings = []

        if stats["max_coeff"] > 1000:
            warnings.append(f"Large stoichiometric coefficients detected (max: {stats['max_coeff']:.1f})")

        if stats["coeff_range"] > 1e6:
            warnings.append(f"Very wide range of stoichiometric coefficients ({stats['coeff_range']:.1e})")

        if stats["large_coeffs"] > 0:
            warnings.append(f"{stats['large_coeffs']} stoichiometric coefficients > 1000")

        if stats["small_coeffs"] > 0:
            warnings.append(f"{stats['small_coeffs']} very small stoichiometric coefficients < 1e-6")

        if stats["isolated_species"] > 0:
            warnings.append(f"{stats['isolated_species']} species not involved in any reaction")

        if stats["empty_reactions"] > 0:
            warnings.append(f"{stats['empty_reactions']} reactions with no participants")

        if stats["condition_number"] and stats["condition_number"] > 1e12:
            warnings.append(f"Poorly conditioned stoichiometric matrix (cond = {stats['condition_number']:.1e})")

        return warnings

    def get_compartment_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze model by compartments."""
        data = self.network_data
        species_info = data["species"]
        reactions = data["reactions"]

        # Group species by compartment
        compartments: Dict[str, Dict[str, Any]] = {}
        for species_id, info in species_info.items():
            comp = info["compartment"]
            if comp not in compartments:
                compartments[comp] = {"species": [], "reactions": [], "internal_reactions": [], "transport_reactions": []}
            compartments[comp]["species"].append(species_id)

        # Analyze reactions by compartment involvement
        for rxn in reactions:
            involved_compartments = set()
            for species_id, _ in rxn["reactants"] + rxn["products"]:
                if species_id in species_info:
                    involved_compartments.add(species_info[species_id]["compartment"])

            if len(involved_compartments) == 1:
                # Internal reaction
                comp = list(involved_compartments)[0]
                if comp in compartments:
                    compartments[comp]["reactions"].append(rxn["id"])
                    compartments[comp]["internal_reactions"].append(rxn["id"])
            else:
                # Transport reaction
                for comp in involved_compartments:
                    if comp in compartments:
                        compartments[comp]["reactions"].append(rxn["id"])
                        compartments[comp]["transport_reactions"].append(rxn["id"])

        # Add summary statistics
        for comp, data in compartments.items():
            data["n_species"] = len(data["species"])
            data["n_reactions"] = len(set(data["reactions"]))
            data["n_internal_reactions"] = len(data["internal_reactions"])
            data["n_transport_reactions"] = len(data["transport_reactions"])

        return compartments

    def print_summary(self):
        """Print a comprehensive model summary."""
        stats = self.get_model_statistics()

        print("Genome-Scale Model Analysis Summary")
        print("=" * 50)

        print(f"Basic Structure:")
        print(f"  Species: {stats['n_species']:,}")
        print(f"  Reactions: {stats['n_reactions']:,}")
        print(f"  Compartments: {stats['n_compartments']}")
        if len(stats["compartments"]) <= 20:
            print(f"    {', '.join(stats['compartments'])}")

        print(f"\\nReaction Properties:")
        print(
            f"  Reversible: {stats['reversible_reactions']:,} ({100*stats['reversible_reactions']/stats['n_reactions']:.1f}%)"
        )
        print(
            f"  Irreversible: {stats['irreversible_reactions']:,} ({100*stats['irreversible_reactions']/stats['n_reactions']:.1f}%)"
        )
        print(f"  With flux bounds: {stats['reactions_with_bounds']:,}")
        print(f"  With gene associations: {stats['reactions_with_genes']:,}")

        print(f"\\nMatrix Properties:")
        print(f"  Format: {stats['matrix_type']}")
        print(f"  Non-zero elements: {stats['non_zero_elements']:,}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")
        print(f"  Memory usage: {stats['matrix_memory_mb']:.2f} MB")

        if stats["n_objectives"] > 0:
            print(f"\\nFBC Information:")
            print(f"  Objectives: {stats['n_objectives']}")
            print(f"  Active objective: {stats['active_objective']}")
            print(f"  Gene products: {stats['n_gene_products']:,}")

        if stats["performance"]:
            print(f"\\nPerformance:")
            for key, value in stats["performance"].items():
                print(f"  {key.replace('_', ' ').title()}: {value:.3f}s")

        # Numerical stability check
        stability = self.check_numerical_stability()
        if stability["stability_warnings"]:
            print(f"\\nStability Warnings:")
            for warning in stability["stability_warnings"]:
                print(f"  ⚠ {warning}")
        else:
            print(f"\\n✓ No major numerical stability issues detected")


def load_genome_scale_model(sbml_file: str, lazy_load: bool = True) -> GenomeScaleAnalyzer:
    """Convenient function to load a genome-scale model.

    Args:
        sbml_file: Path to SBML file
        lazy_load: Whether to delay loading until needed

    Returns:
        GenomeScaleAnalyzer instance
    """
    return GenomeScaleAnalyzer(sbml_file, lazy_load=lazy_load)


def compare_model_sizes(models: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Compare statistics across multiple models.

    Args:
        models: Dictionary mapping model names to SBML file paths

    Returns:
        Dictionary with model statistics for comparison
    """
    comparison = {}

    for name, sbml_file in models.items():
        try:
            analyzer = GenomeScaleAnalyzer(sbml_file, lazy_load=False)
            stats = analyzer.get_model_statistics()
            comparison[name] = stats
        except Exception as e:
            comparison[name] = {"error": str(e)}

    return comparison
