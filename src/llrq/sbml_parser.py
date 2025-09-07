"""SBML parser for log-linear reaction quotient dynamics.

This module provides functionality to parse SBML models and extract
reaction network information needed for log-linear dynamics.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import libsbml

    LIBSBML_AVAILABLE = True
except ImportError:
    LIBSBML_AVAILABLE = False
    libsbml = None


class SBMLParseError(Exception):
    """Exception raised when SBML parsing fails."""

    pass


class SBMLParser:
    """Parser for SBML models to extract reaction network information.

    This class parses SBML files and extracts the information needed
    to construct log-linear reaction quotient dynamics:
    - Species and their initial concentrations
    - Reactions and stoichiometry
    - Kinetic parameters and equilibrium constants
    """

    def __init__(self, sbml_file: str):
        """Initialize parser with SBML file.

        Args:
            sbml_file: Path to SBML file or SBML string content
        """
        if not LIBSBML_AVAILABLE:
            raise ImportError("libsbml is required for SBML parsing. " "Install with: pip install python-libsbml")

        # Check if input looks like SBML content (starts with <?xml) or is a file path
        if sbml_file.strip().startswith("<?xml"):
            # Input is SBML string content
            self.document = libsbml.readSBMLFromString(sbml_file)
        else:
            # Input is a file path
            try:
                self.document = libsbml.readSBML(sbml_file)
            except:
                # If file reading fails, try as string
                self.document = libsbml.readSBMLFromString(sbml_file)

        if self.document.getNumErrors() > 0:
            errors = []
            for i in range(self.document.getNumErrors()):
                error = self.document.getError(i)
                errors.append(f"Line {error.getLine()}: {error.getMessage()}")
            raise SBMLParseError(f"SBML parsing errors:\n" + "\n".join(errors))

        self.model = self.document.getModel()
        if not self.model:
            raise SBMLParseError("No model found in SBML file")

    def get_species_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract species information from SBML model.

        Returns:
            Dictionary mapping species ID to species information:
            - 'name': species name
            - 'initial_concentration': initial concentration
            - 'compartment': compartment ID
            - 'boundary_condition': whether species is boundary condition
        """
        species_info = {}

        for i in range(self.model.getNumSpecies()):
            species = self.model.getSpecies(i)
            species_id = species.getId()

            # Get initial concentration
            if species.isSetInitialConcentration():
                init_conc = species.getInitialConcentration()
            elif species.isSetInitialAmount():
                # Convert amount to concentration if compartment size is known
                compartment = self.model.getCompartment(species.getCompartment())
                if compartment and compartment.isSetSize():
                    init_conc = species.getInitialAmount() / compartment.getSize()
                else:
                    init_conc = species.getInitialAmount()
            else:
                init_conc = 0.0

            species_info[species_id] = {
                "name": species.getName() or species_id,
                "initial_concentration": init_conc,
                "compartment": species.getCompartment(),
                "boundary_condition": species.getBoundaryCondition(),
            }

        return species_info

    def get_reaction_info(self) -> List[Dict[str, Any]]:
        """Extract reaction information from SBML model.

        Returns:
            List of dictionaries, each containing:
            - 'id': reaction ID
            - 'name': reaction name
            - 'reactants': list of (species_id, stoichiometry) tuples
            - 'products': list of (species_id, stoichiometry) tuples
            - 'reversible': whether reaction is reversible
            - 'kinetic_law': kinetic law information if available
        """
        reactions = []

        for i in range(self.model.getNumReactions()):
            reaction = self.model.getReaction(i)
            reaction_id = reaction.getId()

            # Extract reactants
            reactants = []
            for j in range(reaction.getNumReactants()):
                reactant = reaction.getReactant(j)
                species_id = reactant.getSpecies()
                stoich = reactant.getStoichiometry()
                reactants.append((species_id, stoich))

            # Extract products
            products = []
            for j in range(reaction.getNumProducts()):
                product = reaction.getProduct(j)
                species_id = product.getSpecies()
                stoich = product.getStoichiometry()
                products.append((species_id, stoich))

            # Extract kinetic law
            kinetic_info = None
            if reaction.isSetKineticLaw():
                kinetic_law = reaction.getKineticLaw()
                kinetic_info = self._parse_kinetic_law(kinetic_law)

            reactions.append(
                {
                    "id": reaction_id,
                    "name": reaction.getName() or reaction_id,
                    "reactants": reactants,
                    "products": products,
                    "reversible": reaction.getReversible(),
                    "kinetic_law": kinetic_info,
                }
            )

        return reactions

    def _parse_kinetic_law(self, kinetic_law) -> Dict[str, Any]:
        """Parse kinetic law to extract parameters.

        Args:
            kinetic_law: libsbml KineticLaw object

        Returns:
            Dictionary with kinetic law information
        """
        kinetic_info = {"formula": kinetic_law.getFormula() if kinetic_law.isSetFormula() else None, "parameters": {}}

        # Extract local parameters
        for i in range(kinetic_law.getNumParameters()):
            param = kinetic_law.getParameter(i)
            param_id = param.getId()
            value = param.getValue() if param.isSetValue() else None
            kinetic_info["parameters"][param_id] = {"value": value, "units": param.getUnits() if param.isSetUnits() else None}

        return kinetic_info

    def get_global_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Extract global parameters from SBML model.

        Returns:
            Dictionary mapping parameter ID to parameter information
        """
        parameters = {}

        for i in range(self.model.getNumParameters()):
            param = self.model.getParameter(i)
            param_id = param.getId()

            parameters[param_id] = {
                "name": param.getName() or param_id,
                "value": param.getValue() if param.isSetValue() else None,
                "units": param.getUnits() if param.isSetUnits() else None,
                "constant": param.getConstant(),
            }

        return parameters

    def create_stoichiometric_matrix(self, species_ids: List[str], reactions: List[Dict[str, Any]]) -> np.ndarray:
        """Create stoichiometric matrix from reaction information.

        Args:
            species_ids: List of species IDs (rows of matrix)
            reactions: List of reaction dictionaries

        Returns:
            Stoichiometric matrix S where S[i,j] is stoichiometry of
            species i in reaction j (products positive, reactants negative)
        """
        n_species = len(species_ids)
        n_reactions = len(reactions)

        S = np.zeros((n_species, n_reactions))

        species_to_idx = {species_id: i for i, species_id in enumerate(species_ids)}

        for j, reaction in enumerate(reactions):
            # Reactants (negative stoichiometry)
            for species_id, stoich in reaction["reactants"]:
                if species_id in species_to_idx:
                    i = species_to_idx[species_id]
                    S[i, j] -= stoich

            # Products (positive stoichiometry)
            for species_id, stoich in reaction["products"]:
                if species_id in species_to_idx:
                    i = species_to_idx[species_id]
                    S[i, j] += stoich

        return S

    def extract_network_data(self) -> Dict[str, Any]:
        """Extract all network data needed for log-linear dynamics.

        Returns:
            Dictionary containing:
            - 'species': species information
            - 'reactions': reaction information
            - 'parameters': global parameters
            - 'stoichiometric_matrix': stoichiometric matrix
            - 'species_ids': ordered list of species IDs
            - 'reaction_ids': ordered list of reaction IDs
        """
        species_info = self.get_species_info()
        reactions = self.get_reaction_info()
        parameters = self.get_global_parameters()

        species_ids = list(species_info.keys())
        reaction_ids = [r["id"] for r in reactions]

        S = self.create_stoichiometric_matrix(species_ids, reactions)

        return {
            "species": species_info,
            "reactions": reactions,
            "parameters": parameters,
            "stoichiometric_matrix": S,
            "species_ids": species_ids,
            "reaction_ids": reaction_ids,
        }
