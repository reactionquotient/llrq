"""Reaction network representation for log-linear reaction quotient dynamics.

This module provides the ReactionNetwork class that represents chemical
reaction networks and computes reaction quotients.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from scipy.linalg import pinv


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
        # Validate input arguments
        if len(species_ids) == 0:
            raise ValueError("Cannot create reaction network with empty species list")
        if len(reaction_ids) == 0:
            raise ValueError("Cannot create reaction network with empty reaction list")
        
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
        concentrations = np.asarray(concentrations)
        
        if len(concentrations) != self.n_species:
            raise ValueError(f"Expected {self.n_species} concentrations, "
                           f"got {len(concentrations)}")
        
        # Validate for NaN and infinite values
        if np.any(np.isnan(concentrations)):
            raise ValueError("Concentration values cannot be NaN")
        if np.any(np.isinf(concentrations)):
            raise ValueError("Concentration values cannot be infinite")
        
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
        """Find conservation laws (left null space of stoichiometric matrix).
        
        Conservation laws are linear combinations w of species concentrations
        that remain constant over time: w @ c(t) = constant.
        These satisfy: w @ S = 0 (left null space of S).
        
        Args:
            tol: Tolerance for determining rank
            
        Returns:
            Conservation matrix C where each row is a conservation law vector
        """
        if self._conservation_matrix is None:
            # Find left null space of S using SVD
            U, s, Vt = np.linalg.svd(self.S.T, full_matrices=True)
            rank = np.sum(s > tol)
            
            if rank < self.n_species:
                # Conservation laws exist
                # The left null space of S is the right null space of S.T
                conservation_vectors = Vt[rank:, :]
                
                # Normalize and clean up small numerical errors
                clean_vectors = []
                for vec in conservation_vectors:
                    # Normalize
                    vec_normalized = vec / np.linalg.norm(vec)
                    
                    # Try to make it have nice integer coefficients if possible
                    # Scale to make largest absolute component 1
                    max_coeff = np.max(np.abs(vec_normalized))
                    if max_coeff > tol:
                        vec_scaled = vec_normalized / max_coeff
                        
                        # Check if coefficients are close to integers
                        if np.allclose(vec_scaled, np.round(vec_scaled), atol=1e-6):
                            vec_clean = np.round(vec_scaled)
                            # Renormalize to unit vector
                            vec_clean = vec_clean / np.linalg.norm(vec_clean)
                            clean_vectors.append(vec_clean)
                        else:
                            clean_vectors.append(vec_normalized)
                
                if clean_vectors:
                    self._conservation_matrix = np.array(clean_vectors)
                else:
                    self._conservation_matrix = np.zeros((0, self.n_species))
            else:
                # No conservation laws (full rank)
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
    
    def compute_dynamics_matrix(self,
                               equilibrium_point: np.ndarray,
                               forward_rates: np.ndarray,
                               backward_rates: np.ndarray,
                               mode: str = 'equilibrium',
                               reduce_to_image: bool = True,
                               enforce_symmetry: bool = False) -> Dict[str, Any]:
        """Compute dynamics matrix K from mass action network.
        
        Implements the algorithm from Diamond (2025) "Log-Linear Reaction Quotient 
        Dynamics" for computing the dynamics matrix K that governs:
        
            d/dt ln Q = -K ln(Q/Keq) + u(t)
        
        where Q are reaction quotients, Keq equilibrium constants, u(t) external drives.
        
        Algorithm modes:
        
        **Equilibrium mode** (near thermodynamic equilibrium):
        1. Compute flux coefficients: φⱼ = kⱼ⁺ (c*)^νⱼʳᵉᵃᶜ = kⱼ⁻ (c*)^νⱼᵖʳᵒᵈ
        2. Form Φ = Diag(φ), D* = Diag(c*)
        3. Return K = S^T (D*)⁻¹ S Φ
        
        **Nonequilibrium mode** (general steady state):
        1. Evaluate Jacobian Jᵤ = ∂v/∂u at c* (u = ln c, v = reaction rates)
        2. Build R = D* S (S^T D* S)⁻¹
        3. Return K = -S^T (D*)⁻¹ S Jᵤ R
        
        Optional enhancements:
        - **Basis reduction**: Projects to Im(S^T) to eliminate conservation null space
        - **Symmetry enforcement**: Ensures K is symmetric positive definite
        
        Physical interpretation:
        - K captures how reaction quotient deviations relax back to equilibrium
        - Eigenvalues of K give relaxation timescales (1/λ)
        - Off-diagonal coupling shows how reactions influence each other
        
        Args:
            equilibrium_point: Equilibrium/steady-state concentrations c* [species]
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            mode: 'equilibrium' for near thermodynamic equilibrium,
                  'nonequilibrium' for general steady state
            reduce_to_image: Whether to reduce to Im(S^T) basis (recommended)
            enforce_symmetry: Whether to symmetrize K (for detailed balance systems)
            
        Returns:
            Dictionary containing:
            - 'K': Full dynamics matrix [reactions × reactions]
            - 'K_reduced': Reduced matrix [rank(S^T) × rank(S^T)] (if reduce_to_image=True)
            - 'basis': Orthonormal basis B for Im(S^T) [reactions × rank(S^T)] (if reduce_to_image=True)
            - 'phi': Flux coefficients [reactions] (equilibrium mode only)
            - 'eigenanalysis': {'eigenvalues', 'eigenvectors', 'is_stable'}
            
        Examples:
            Simple A ⇌ B reaction:
            >>> network = ReactionNetwork(['A', 'B'], ['R1'], [[-1], [1]])
            >>> result = network.compute_dynamics_matrix(
            ...     equilibrium_point=[1.0, 2.0],
            ...     forward_rates=[2.0],
            ...     backward_rates=[1.0],
            ...     mode='equilibrium'
            ... )
            >>> print(f"K = {result['K']}, stable = {result['eigenanalysis']['is_stable']}")
            
        References:
            Diamond, S. (2025). "Log-Linear Reaction Quotient Dynamics"
        """
        if len(equilibrium_point) != self.n_species:
            raise ValueError(f"Expected {self.n_species} equilibrium concentrations, "
                           f"got {len(equilibrium_point)}")
        if len(forward_rates) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} forward rates, "
                           f"got {len(forward_rates)}")
        if len(backward_rates) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} backward rates, "
                           f"got {len(backward_rates)}")
        
        c_star = np.array(equilibrium_point)
        k_plus = np.array(forward_rates)
        k_minus = np.array(backward_rates)
        
        # Form D* = Diag(c*)
        D_star = np.diag(c_star)
        D_star_inv = np.diag(1.0 / np.maximum(c_star, 1e-12))
        
        if mode == 'equilibrium':
            K = self._compute_equilibrium_dynamics_matrix(
                c_star, k_plus, k_minus, D_star_inv)
            phi = self._compute_flux_coefficients(c_star, k_plus, k_minus)
            result = {'K': K, 'phi': phi}
        elif mode == 'nonequilibrium':
            K = self._compute_nonequilibrium_dynamics_matrix(
                c_star, k_plus, k_minus, D_star, D_star_inv)
            result = {'K': K}
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'equilibrium' or 'nonequilibrium'.")
        
        # Optional basis reduction
        if reduce_to_image:
            K_reduced, basis = self._reduce_to_image_space(K)
            result['K_reduced'] = K_reduced
            result['basis'] = basis
        
        # Optional symmetry enforcement
        if enforce_symmetry:
            if 'K_reduced' in result:
                result['K_reduced'] = self._enforce_symmetry(result['K_reduced'])
            else:
                result['K'] = self._enforce_symmetry(result['K'])
        
        # Eigenanalysis
        K_analysis = result.get('K_reduced', result['K'])
        eigenvals, eigenvecs = np.linalg.eig(K_analysis)
        result['eigenanalysis'] = {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'is_stable': np.all(eigenvals.real >= -1e-12)
        }
        
        return result
    
    def _compute_flux_coefficients(self, c_star: np.ndarray, 
                                 k_plus: np.ndarray, k_minus: np.ndarray) -> np.ndarray:
        """Compute flux coefficients φ_j = k_j^+ (c*)^(ν_j^reac) = k_j^- (c*)^(ν_j^prod)."""
        phi = np.zeros(self.n_reactions)
        
        for j in range(self.n_reactions):
            # Get reactant and product stoichiometries
            nu_reac = np.maximum(-self.S[:, j], 0)  # Reactant coefficients (positive)
            nu_prod = np.maximum(self.S[:, j], 0)   # Product coefficients (positive)
            
            # Compute φ_j = k_j^+ * ∏(c_i*)^(ν_ij^reac)
            phi_forward = k_plus[j] * np.prod(c_star ** nu_reac)
            
            # Should equal k_j^- * ∏(c_i*)^(ν_ij^prod)
            phi_backward = k_minus[j] * np.prod(c_star ** nu_prod)
            
            # Use average for robustness
            phi[j] = 0.5 * (phi_forward + phi_backward)
        
        return phi
    
    def _compute_equilibrium_dynamics_matrix(self, c_star: np.ndarray,
                                           k_plus: np.ndarray, k_minus: np.ndarray,
                                           D_star_inv: np.ndarray) -> np.ndarray:
        """Compute K = S^T D*^(-1) S Φ for equilibrium case."""
        phi = self._compute_flux_coefficients(c_star, k_plus, k_minus)
        Phi = np.diag(phi)
        
        # K = S^T D*^(-1) S Φ
        K = self.S.T @ D_star_inv @ self.S @ Phi
        
        return K
    
    def _compute_nonequilibrium_dynamics_matrix(self, c_star: np.ndarray,
                                              k_plus: np.ndarray, k_minus: np.ndarray,
                                              D_star: np.ndarray, D_star_inv: np.ndarray) -> np.ndarray:
        """Compute K = -S^T D*^(-1) S J_u R for nonequilibrium case."""
        # Compute Jacobian J_u = ∂v/∂u at c*
        J_u = self._compute_flux_jacobian(c_star, k_plus, k_minus)
        
        # Compute R = D* S (S^T D* S)^(-1)
        STS_D = self.S.T @ D_star @ self.S
        try:
            STS_D_inv = np.linalg.inv(STS_D)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            STS_D_inv = pinv(STS_D)
            warnings.warn("S^T D* S is singular, using pseudoinverse")
        
        R = D_star @ self.S @ STS_D_inv
        
        # K = -S^T D*^(-1) S J_u R
        K = -self.S.T @ D_star_inv @ self.S @ J_u @ R
        
        return K
    
    def _compute_flux_jacobian(self, c_star: np.ndarray,
                             k_plus: np.ndarray, k_minus: np.ndarray) -> np.ndarray:
        """Compute Jacobian J_u = ∂v/∂u of reaction fluxes."""
        n_species = len(c_star)
        J_u = np.zeros((self.n_reactions, n_species))
        
        for j in range(self.n_reactions):
            for i in range(n_species):
                # ∂v_j/∂u_i where u_i = ln(c_i), v_j is net flux of reaction j
                
                # Reactant contribution: -ν_ij^reac * k_j^+ * c_i * ∏(c_k)^(ν_kj^reac)
                nu_reac_i = max(-self.S[i, j], 0)
                if nu_reac_i > 0:
                    nu_reac = np.maximum(-self.S[:, j], 0)
                    forward_flux = k_plus[j] * np.prod(c_star ** nu_reac)
                    J_u[j, i] -= nu_reac_i * forward_flux
                
                # Product contribution: +ν_ij^prod * k_j^- * c_i * ∏(c_k)^(ν_kj^prod)
                nu_prod_i = max(self.S[i, j], 0)
                if nu_prod_i > 0:
                    nu_prod = np.maximum(self.S[:, j], 0)
                    backward_flux = k_minus[j] * np.prod(c_star ** nu_prod)
                    J_u[j, i] += nu_prod_i * backward_flux
        
        return J_u
    
    def _reduce_to_image_space(self, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce matrix K to Im(S^T) basis."""
        # Find orthonormal basis for Im(S^T)
        U, s, Vt = np.linalg.svd(self.S.T, full_matrices=False)
        rank = np.sum(s > 1e-12)
        
        if rank == 0:
            warnings.warn("S^T has zero rank, returning original matrix")
            return K, np.eye(K.shape[0])
        
        # Basis matrix B (columns are basis vectors)
        B = U[:, :rank]
        
        # Reduced matrix K_red = B^T K B
        K_reduced = B.T @ K @ B
        
        return K_reduced, B
    
    def _enforce_symmetry(self, K: np.ndarray) -> np.ndarray:
        """Enforce symmetry and positive stability."""
        # Symmetrize
        K_sym = 0.5 * (K + K.T)
        
        # Ensure positive definite by shifting negative eigenvalues
        eigenvals, eigenvecs = np.linalg.eigh(K_sym)
        min_eigenval = np.min(eigenvals)
        
        if min_eigenval < 1e-12:
            shift = 1e-10 - min_eigenval
            eigenvals += shift
            K_sym = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return K_sym

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