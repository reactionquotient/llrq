"""Reaction network representation for log-linear reaction quotient dynamics.

This module provides the ReactionNetwork class that represents chemical
reaction networks and computes reaction quotients.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings


class ReactionNetwork:
    """Representation of a chemical reaction network.
    
    This class stores reaction network information and provides methods
    to compute reaction quotients and other network properties needed
    for log-linear dynamics.
    """
    
    def __init__(self, 
                 species_ids: List[str],
                 reaction_ids: List[str],
                 stoichiometric_matrix: np.ndarray,
                 species_info: Optional[Dict[str, Dict[str, Any]]] = None,
                 reaction_info: Optional[List[Dict[str, Any]]] = None,
                 parameters: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize reaction network.
        
        Args:
            species_ids: List of species identifiers
            reaction_ids: List of reaction identifiers  
            stoichiometric_matrix: Stoichiometric matrix S (species x reactions)
            species_info: Additional species information from SBML
            reaction_info: Additional reaction information from SBML
            parameters: Global parameters from SBML
        """
        self.species_ids = species_ids
        self.reaction_ids = reaction_ids
        self.S = np.array(stoichiometric_matrix, dtype=float)
        self.species_info = species_info or {}
        self.reaction_info = reaction_info or []
        self.parameters = parameters or {}
        
        # Validate dimensions
        if self.S.shape[0] != len(species_ids):
            raise ValueError(f"Stoichiometric matrix has {self.S.shape[0]} rows "
                           f"but {len(species_ids)} species provided")
        if self.S.shape[1] != len(reaction_ids):
            raise ValueError(f"Stoichiometric matrix has {self.S.shape[1]} columns "
                           f"but {len(reaction_ids)} reactions provided")
        
        # Create index mappings
        self.species_to_idx = {species_id: i for i, species_id in enumerate(species_ids)}
        self.reaction_to_idx = {reaction_id: i for i, reaction_id in enumerate(reaction_ids)}
        
        # Cache properties
        self._conservation_matrix = None
        self._null_space = None
    
    @property
    def n_species(self) -> int:
        """Number of species."""
        return len(self.species_ids)
    
    @property
    def n_reactions(self) -> int:
        """Number of reactions."""
        return len(self.reaction_ids)
    
    def get_initial_concentrations(self) -> np.ndarray:
        """Get initial concentrations from species info.
        
        Returns:
            Array of initial concentrations in species order
        """
        concentrations = np.zeros(self.n_species)
        
        for i, species_id in enumerate(self.species_ids):
            if species_id in self.species_info:
                conc = self.species_info[species_id].get('initial_concentration', 0.0)
                concentrations[i] = conc
        
        return concentrations
    
    def compute_reaction_quotients(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute reaction quotients for all reactions.
        
        Args:
            concentrations: Current species concentrations
            
        Returns:
            Array of reaction quotients Q_j = ∏_i [X_i]^S_ij
        """
        if len(concentrations) != self.n_species:
            raise ValueError(f"Expected {self.n_species} concentrations, "
                           f"got {len(concentrations)}")
        
        # Avoid log(0) by adding small epsilon
        eps = 1e-12
        safe_conc = np.maximum(concentrations, eps)
        
        # Q_j = ∏_i [X_i]^S_ij = exp(∑_i S_ij * ln[X_i])
        log_Q = self.S.T @ np.log(safe_conc)  # Shape: (n_reactions,)
        Q = np.exp(log_Q)
        
        return Q
    
    def compute_single_reaction_quotient(self, reaction_id: str, 
                                       concentrations: np.ndarray) -> float:
        """Compute reaction quotient for a single reaction.
        
        Args:
            reaction_id: Identifier for the reaction
            concentrations: Current species concentrations
            
        Returns:
            Reaction quotient for the specified reaction
        """
        if reaction_id not in self.reaction_to_idx:
            raise ValueError(f"Reaction '{reaction_id}' not found")
        
        j = self.reaction_to_idx[reaction_id]
        Q_all = self.compute_reaction_quotients(concentrations)
        
        return Q_all[j]
    
    def get_reaction_stoichiometry(self, reaction_id: str) -> Tuple[Dict[str, float], 
                                                                  Dict[str, float]]:
        """Get reactants and products with stoichiometries for a reaction.
        
        Args:
            reaction_id: Identifier for the reaction
            
        Returns:
            Tuple of (reactants_dict, products_dict) where keys are species IDs
            and values are stoichiometric coefficients
        """
        if reaction_id not in self.reaction_to_idx:
            raise ValueError(f"Reaction '{reaction_id}' not found")
        
        j = self.reaction_to_idx[reaction_id]
        reactants = {}
        products = {}
        
        for i, species_id in enumerate(self.species_ids):
            stoich = self.S[i, j]
            if stoich < 0:
                reactants[species_id] = -stoich  # Store as positive
            elif stoich > 0:
                products[species_id] = stoich
        
        return reactants, products
    
    def find_conservation_laws(self, tol: float = 1e-10) -> np.ndarray:
        """Find conservation laws (null space of stoichiometric matrix).
        
        Args:
            tol: Tolerance for determining rank
            
        Returns:
            Conservation matrix C where C @ concentrations = constants
        """
        if self._conservation_matrix is None:
            # Find left null space of S (rows that sum to zero when multiplied by S)
            # This is equivalent to right null space of S.T
            U, s, Vt = np.linalg.svd(self.S.T, full_matrices=True)
            rank = np.sum(s > tol)
            
            if rank < self.n_species:
                # Conservation laws exist
                self._conservation_matrix = Vt[rank:, :]
            else:
                # No conservation laws
                self._conservation_matrix = np.zeros((0, self.n_species))
        
        return self._conservation_matrix
    
    def compute_conserved_quantities(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute values of conserved quantities.
        
        Args:
            concentrations: Species concentrations
            
        Returns:
            Array of conserved quantity values
        """
        C = self.find_conservation_laws()
        if C.shape[0] == 0:
            return np.array([])
        
        return C @ concentrations
    
    def get_independent_reactions(self, tol: float = 1e-10) -> List[int]:
        """Find indices of linearly independent reactions.
        
        Args:
            tol: Tolerance for determining rank
            
        Returns:
            List of reaction indices that form an independent set
        """
        U, s, Vt = np.linalg.svd(self.S, full_matrices=False)
        rank = np.sum(s > tol)
        
        # Find which columns (reactions) are independent
        from scipy.linalg import qr
        Q, R, P = qr(self.S, pivoting=True, mode='economic')
        independent_indices = P[:rank].tolist()
        
        return independent_indices
    
    def get_reaction_equation(self, reaction_id: str) -> str:
        """Get human-readable equation for a reaction.
        
        Args:
            reaction_id: Identifier for the reaction
            
        Returns:
            String representation like "A + 2B -> C + D"
        """
        reactants, products = self.get_reaction_stoichiometry(reaction_id)
        
        # Format reactants
        reactant_terms = []
        for species_id, stoich in reactants.items():
            if stoich == 1:
                reactant_terms.append(species_id)
            else:
                reactant_terms.append(f"{stoich:.0f} {species_id}")
        
        # Format products  
        product_terms = []
        for species_id, stoich in products.items():
            if stoich == 1:
                product_terms.append(species_id)
            else:
                product_terms.append(f"{stoich:.0f} {species_id}")
        
        reactant_str = " + ".join(reactant_terms) if reactant_terms else "∅"
        product_str = " + ".join(product_terms) if product_terms else "∅"
        
        # Check if reversible
        reversible = False
        if reaction_id in [r['id'] for r in self.reaction_info]:
            reaction = next(r for r in self.reaction_info if r['id'] == reaction_id)
            reversible = reaction.get('reversible', False)
        
        arrow = " ⇌ " if reversible else " → "
        
        return f"{reactant_str}{arrow}{product_str}"
    
    def summary(self) -> str:
        """Generate summary of the reaction network.
        
        Returns:
            String summary of network properties
        """
        lines = [
            f"Reaction Network Summary",
            f"=======================",
            f"Species: {self.n_species}",
            f"Reactions: {self.n_reactions}",
        ]
        
        # Conservation laws
        C = self.find_conservation_laws()
        if C.shape[0] > 0:
            lines.append(f"Conservation laws: {C.shape[0]}")
        else:
            lines.append("Conservation laws: None")
        
        # List reactions
        lines.append("\nReactions:")
        for reaction_id in self.reaction_ids:
            equation = self.get_reaction_equation(reaction_id)
            lines.append(f"  {reaction_id}: {equation}")
        
        return "\n".join(lines)
    
    @classmethod
    def from_sbml_data(cls, sbml_data: Dict[str, Any]) -> 'ReactionNetwork':
        """Create ReactionNetwork from SBML parser data.
        
        Args:
            sbml_data: Dictionary from SBMLParser.extract_network_data()
            
        Returns:
            ReactionNetwork instance
        """
        return cls(
            species_ids=sbml_data['species_ids'],
            reaction_ids=sbml_data['reaction_ids'],
            stoichiometric_matrix=sbml_data['stoichiometric_matrix'],
            species_info=sbml_data['species'],
            reaction_info=sbml_data['reactions'],
            parameters=sbml_data['parameters']
        )