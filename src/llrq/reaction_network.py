"""Reaction network representation for log-linear reaction quotient dynamics.

This module provides the ReactionNetwork class that represents chemical
reaction networks and computes reaction quotients.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import lstsq, pinv, svd


class ReactionNetwork:
    """Representation of a chemical reaction network.

    This class stores reaction network information and provides methods
    to compute reaction quotients and other network properties needed
    for log-linear dynamics.
    """

    def __init__(
        self,
        species_ids: List[str],
        reaction_ids: List[str],
        stoichiometric_matrix: np.ndarray,
        species_info: Optional[Dict[str, Dict[str, Any]]] = None,
        reaction_info: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
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
            raise ValueError(f"Stoichiometric matrix has {self.S.shape[0]} rows " f"but {len(species_ids)} species provided")
        if self.S.shape[1] != len(reaction_ids):
            raise ValueError(
                f"Stoichiometric matrix has {self.S.shape[1]} columns " f"but {len(reaction_ids)} reactions provided"
            )

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

    def get_reactant_stoichiometry_matrix(self) -> np.ndarray:
        """Extract reactant stoichiometry matrix A from stoichiometric matrix S.

        For reaction j, A[:,j] contains positive stoichiometric coefficients
        for reactants (corresponding to negative entries in S).

        Returns:
            Reactant stoichiometry matrix A (n_species × n_reactions)
        """
        return np.maximum(-self.S, 0)

    def get_product_stoichiometry_matrix(self) -> np.ndarray:
        """Extract product stoichiometry matrix B from stoichiometric matrix S.

        For reaction j, B[:,j] contains positive stoichiometric coefficients
        for products (corresponding to positive entries in S).

        Returns:
            Product stoichiometry matrix B (n_species × n_reactions)
        """
        return np.maximum(self.S, 0)

    def _nullspace(self, M: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
        """Right nullspace of M (columns span ker(M))."""
        U, s, Vt = svd(M, full_matrices=True)
        rank = (s > rtol * s.max()).sum()
        return Vt[rank:].T  # shape: cols = nullity

    def _left_nullspace(self, M: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
        """Left nullspace of M: columns L with L^T M = 0 (i.e., ker(M^T))."""
        return self._nullspace(M.T, rtol)

    def get_initial_concentrations(self) -> np.ndarray:
        """Get initial concentrations from species info.

        Returns:
            Array of initial concentrations in species order
        """
        concentrations = np.zeros(self.n_species)

        for i, species_id in enumerate(self.species_ids):
            if species_id in self.species_info:
                conc = self.species_info[species_id].get("initial_concentration", 0.0)
                concentrations[i] = conc

        return concentrations

    def is_at_equilibrium(
        self, concentrations: np.ndarray, forward_rates: np.ndarray, backward_rates: np.ndarray, tol: float = 1e-6
    ) -> bool:
        """Check if concentrations satisfy detailed balance (equilibrium condition).

        Tests whether Q_j(c) ≈ K_j for all reactions j, where:
        - Q_j = reaction quotient for reaction j
        - K_j = equilibrium constant = k_j⁺/k_j⁻

        Args:
            concentrations: Species concentrations to test
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            tol: Relative tolerance for equilibrium test

        Returns:
            True if concentrations are at equilibrium within tolerance
        """
        try:
            # Compute reaction quotients at given concentrations
            Q = self.compute_reaction_quotients(concentrations)

            # Compute equilibrium constants
            K = forward_rates / backward_rates

            # Check detailed balance: Q ≈ K for all reactions
            return np.allclose(Q, K, rtol=tol, atol=1e-12)
        except (ValueError, ZeroDivisionError):
            # If any computation fails, assume not at equilibrium
            return False

    def compute_equilibrium(
        self,
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        initial_concentrations: Optional[np.ndarray] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute equilibrium concentrations for mass action network.

        Uses the detailed balance equilibrium algorithm to find equilibrium
        concentrations that satisfy:
        1. Detailed balance: Q_j(c*) = K_j for all reactions j
        2. Conservation laws: conserved quantities match initial totals

        Args:
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            initial_concentrations: Initial concentrations for conservation [species]
                                  If None, uses get_initial_concentrations()
            tol: Convergence tolerance on conservation residual
            max_iter: Maximum Newton iterations

        Returns:
            Tuple of (equilibrium_concentrations, info_dict) where info contains
            diagnostic information including convergence status

        Raises:
            ValueError: If rate constants violate thermodynamic consistency
        """
        # Input validation
        k_plus = np.asarray(forward_rates, dtype=float)
        k_minus = np.asarray(backward_rates, dtype=float)

        if len(k_plus) != self.n_reactions or len(k_minus) != self.n_reactions:
            raise ValueError(
                f"Expected {self.n_reactions} rate constants, " f"got {len(k_plus)} forward and {len(k_minus)} backward"
            )

        if not np.all(k_plus > 0) or not np.all(k_minus > 0):
            raise ValueError("All rate constants must be positive for equilibrium computation")

        # Get initial concentrations for conservation laws
        if initial_concentrations is None:
            c0 = self.get_initial_concentrations()
        else:
            c0 = np.asarray(initial_concentrations, dtype=float)
            if len(c0) != self.n_species:
                raise ValueError(f"Expected {self.n_species} initial concentrations, " f"got {len(c0)}")

        # Extract A, B matrices from stoichiometric matrix
        A = self.get_reactant_stoichiometry_matrix()
        B = self.get_product_stoichiometry_matrix()

        # Net stoichiometry and equilibrium constants
        N = B - A  # This is the same as self.S
        lnK = np.log(k_plus / k_minus)

        # 1) Solve N^T x = ln K for x = ln c (particular solution)
        # This encodes detailed balance: Q(c) = K
        NT = N.T
        x_p, _, _, _ = lstsq(NT, lnK)  # particular solution
        resid = np.linalg.norm(NT @ x_p - lnK)

        # Check thermodynamic consistency (Wegscheider conditions)
        if resid > 1e-8:
            raise ValueError(
                f"Inconsistent equilibrium constants (||N^T x - lnK||={resid:.2e}). "
                "Check Wegscheider conditions / rate constants."
            )

        # 2) Add general solution: x = x_p + Z y, where Z spans ker(N^T)
        Z = self._nullspace(NT)
        p = Z.shape[1]  # Number of conserved moieties

        # If no conserved moieties, equilibrium is unique
        if p == 0:
            c_star = np.exp(x_p)
            return c_star, {"iterations": 0, "conservation_residual": 0.0, "n_conserved": 0, "thermodynamic_check": resid}

        # 3) Enforce conservation laws using initial totals
        # L^T c is conserved, where L spans left nullspace of N
        L = self._left_nullspace(N)
        m = L.T @ c0  # Target conserved totals

        # Solve g(y) = L^T exp(x_p + Z y) - m = 0 via Newton's method
        y = np.zeros(p)
        for it in range(max_iter):
            x = x_p + Z @ y
            c = np.exp(x)
            g = L.T @ c - m
            g_norm = np.linalg.norm(g, ord=2)

            if g_norm < tol:
                return c, {
                    "iterations": it,
                    "conservation_residual": float(g_norm),
                    "n_conserved": p,
                    "thermodynamic_check": resid,
                }

            # Jacobian: J = d/dy [L^T exp(x_p + Z y)] = L^T diag(c) Z
            J = L.T @ (c[:, None] * Z)

            # Solve J Δy = -g
            try:
                dy, _, _, _ = lstsq(J, -g)
            except Exception:
                dy = -g  # Fallback step

            # Backtracking line search to ensure progress and positivity
            step = 1.0
            for _ in range(20):
                y_trial = y + step * dy
                x_trial = x_p + Z @ y_trial
                c_trial = np.exp(x_trial)
                g_trial = L.T @ c_trial - m
                if np.linalg.norm(g_trial) < g_norm * (1 - 1e-4 * step):
                    y = y_trial
                    break
                step *= 0.5
            else:
                # Could not improve; return best found
                return c, {
                    "iterations": it + 1,
                    "conservation_residual": float(g_norm),
                    "n_conserved": p,
                    "thermodynamic_check": resid,
                    "warning": "Newton iteration stalled",
                }

        # If we exit loop without converging
        final_c = np.exp(x_p + Z @ y)
        final_g_norm = np.linalg.norm(L.T @ final_c - m)
        return final_c, {
            "iterations": max_iter,
            "conservation_residual": float(final_g_norm),
            "n_conserved": p,
            "thermodynamic_check": resid,
            "warning": "Max iterations reached",
        }

    def compute_reaction_quotients(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute reaction quotients for all reactions.

        Args:
            concentrations: Current species concentrations

        Returns:
            Array of reaction quotients Q_j = ∏_i [X_i]^S_ij
        """
        concentrations = np.asarray(concentrations)

        if len(concentrations) != self.n_species:
            raise ValueError(f"Expected {self.n_species} concentrations, " f"got {len(concentrations)}")

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

    def compute_single_reaction_quotient(self, reaction_id: str, concentrations: np.ndarray) -> float:
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

    def get_reaction_stoichiometry(self, reaction_id: str) -> Tuple[Dict[str, float], Dict[str, float]]:
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

        Q, R, P = qr(self.S, pivoting=True, mode="economic")
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
        if reaction_id in [r["id"] for r in self.reaction_info]:
            reaction = next(r for r in self.reaction_info if r["id"] == reaction_id)
            reversible = reaction.get("reversible", False)

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

    def compute_dynamics_matrix(
        self,
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        initial_concentrations: Optional[np.ndarray] = None,
        mode: str = "equilibrium",
        reduce_to_image: bool = True,
        enforce_symmetry: bool = False,
    ) -> Dict[str, Any]:
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
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            initial_concentrations: Species concentrations [species]
                                  If at equilibrium, used directly; otherwise equilibrium is computed
                                  If None, uses get_initial_concentrations() from species info
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
            ...     forward_rates=[2.0],
            ...     backward_rates=[1.0],
            ...     initial_concentrations=[1.0, 2.0],
            ...     mode='equilibrium'
            ... )
            >>> print(f"K = {result['K']}, stable = {result['eigenanalysis']['is_stable']}")

        References:
            Diamond, S. (2025). "Log-Linear Reaction Quotient Dynamics"
        """
        # Input validation
        k_plus = np.array(forward_rates)
        k_minus = np.array(backward_rates)

        if len(k_plus) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} forward rates, " f"got {len(k_plus)}")
        if len(k_minus) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} backward rates, " f"got {len(k_minus)}")

        # Get or compute equilibrium point
        if initial_concentrations is None:
            initial_concentrations = self.get_initial_concentrations()
        else:
            initial_concentrations = np.array(initial_concentrations)
            if len(initial_concentrations) != self.n_species:
                raise ValueError(f"Expected {self.n_species} initial concentrations, " f"got {len(initial_concentrations)}")

        # Check if already at equilibrium
        if self.is_at_equilibrium(initial_concentrations, k_plus, k_minus):
            # Use provided concentrations as equilibrium
            c_star = initial_concentrations
            equilibrium_info = {"already_at_equilibrium": True, "iterations": 0}
            result_has_equilibrium_info = True
        else:
            # Compute equilibrium from initial concentrations
            c_star, equilibrium_info = self.compute_equilibrium(k_plus, k_minus, initial_concentrations)
            equilibrium_info["already_at_equilibrium"] = False
            result_has_equilibrium_info = True

        # Form D* = Diag(c*)
        D_star = np.diag(c_star)
        D_star_inv = np.diag(1.0 / np.maximum(c_star, 1e-12))

        if mode == "equilibrium":
            K = self._compute_equilibrium_dynamics_matrix(c_star, k_plus, k_minus, D_star_inv)
            phi = self._compute_flux_coefficients(c_star, k_plus, k_minus)
            result = {"K": K, "phi": phi}
        elif mode == "nonequilibrium":
            K = self._compute_nonequilibrium_dynamics_matrix(c_star, k_plus, k_minus, D_star, D_star_inv)
            result = {"K": K}
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'equilibrium' or 'nonequilibrium'.")

        # Optional basis reduction
        if reduce_to_image:
            K_reduced, basis = self._reduce_to_image_space(K)
            result["K_reduced"] = K_reduced
            result["basis"] = basis

        # Optional symmetry enforcement
        if enforce_symmetry:
            if "K_reduced" in result:
                result["K_reduced"] = self._enforce_symmetry(result["K_reduced"])
            else:
                result["K"] = self._enforce_symmetry(result["K"])

        # Eigenanalysis
        K_analysis = result.get("K_reduced", result["K"])
        eigenvals, eigenvecs = np.linalg.eig(K_analysis)
        result["eigenanalysis"] = {
            "eigenvalues": eigenvals,
            "eigenvectors": eigenvecs,
            "is_stable": np.all(eigenvals.real >= -1e-12),
        }

        # Add equilibrium computation info if computed automatically
        if result_has_equilibrium_info:
            result["equilibrium_info"] = equilibrium_info
            result["equilibrium_point"] = c_star

        return result

    def _compute_flux_coefficients(self, c_star: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray) -> np.ndarray:
        """Compute flux coefficients φ_j = k_j^+ (c*)^(ν_j^reac) = k_j^- (c*)^(ν_j^prod)."""
        phi = np.zeros(self.n_reactions)

        for j in range(self.n_reactions):
            # Get reactant and product stoichiometries
            nu_reac = np.maximum(-self.S[:, j], 0)  # Reactant coefficients (positive)
            nu_prod = np.maximum(self.S[:, j], 0)  # Product coefficients (positive)

            # Compute φ_j = k_j^+ * ∏(c_i*)^(ν_ij^reac)
            phi_forward = k_plus[j] * np.prod(c_star**nu_reac)

            # Should equal k_j^- * ∏(c_i*)^(ν_ij^prod)
            phi_backward = k_minus[j] * np.prod(c_star**nu_prod)

            # Use average for robustness
            phi[j] = 0.5 * (phi_forward + phi_backward)

        return phi

    def _compute_equilibrium_dynamics_matrix(
        self, c_star: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray, D_star_inv: np.ndarray
    ) -> np.ndarray:
        """Compute K = S^T D*^(-1) S Φ for equilibrium case."""
        phi = self._compute_flux_coefficients(c_star, k_plus, k_minus)
        Phi = np.diag(phi)

        # K = S^T D*^(-1) S Φ
        K = self.S.T @ D_star_inv @ self.S @ Phi

        return K

    def _compute_nonequilibrium_dynamics_matrix(
        self, c_star: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray, D_star: np.ndarray, D_star_inv: np.ndarray
    ) -> np.ndarray:
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

    def _compute_flux_jacobian(self, c_star: np.ndarray, k_plus: np.ndarray, k_minus: np.ndarray) -> np.ndarray:
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
                    forward_flux = k_plus[j] * np.prod(c_star**nu_reac)
                    J_u[j, i] -= nu_reac_i * forward_flux

                # Product contribution: +ν_ij^prod * k_j^- * c_i * ∏(c_k)^(ν_kj^prod)
                nu_prod_i = max(self.S[i, j], 0)
                if nu_prod_i > 0:
                    nu_prod = np.maximum(self.S[:, j], 0)
                    backward_flux = k_minus[j] * np.prod(c_star**nu_prod)
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

    def _compute_flux_parts(
        self, concentrations: np.ndarray, forward_rates: np.ndarray, backward_rates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute forward, reverse, and net fluxes for mass action kinetics.

        Args:
            concentrations: Species concentrations [species]
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]

        Returns:
            Tuple of (forward_flux, reverse_flux, net_flux) arrays [reactions]
        """
        concentrations = np.asarray(concentrations)
        forward_rates = np.asarray(forward_rates)
        backward_rates = np.asarray(backward_rates)

        # Safe logarithm to avoid log(0)
        ln_c = np.log(np.maximum(concentrations, 1e-300))

        # Get reactant and product stoichiometry matrices
        A = self.get_reactant_stoichiometry_matrix()  # reactants (positive coeffs)
        B = self.get_product_stoichiometry_matrix()  # products (positive coeffs)

        # Forward flux: f_j = k_j⁺ * ∏(c_i^A_ij)
        forward_flux = forward_rates * np.exp(A.T @ ln_c)

        # Reverse flux: r_j = k_j⁻ * ∏(c_i^B_ij)
        reverse_flux = backward_rates * np.exp(B.T @ ln_c)

        # Net flux
        net_flux = forward_flux - reverse_flux

        return forward_flux, reverse_flux, net_flux

    def compute_onsager_conductance(
        self,
        concentrations: np.ndarray,
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        mode: str = "auto",
        equilibrium_tol: float = 1e-8,
        enforce_reciprocity: bool = True,
        clip_eigenvalues: float = 1e-12,
    ) -> Dict[str, Any]:
        """Compute Onsager conductance matrix L for thermodynamic accounting.

        The Onsager conductance L maps reaction forces to fluxes in the linear regime:
        J ≈ -L x, where x = S^T ln(c) - ln(K_eq) are reaction forces.

        This implements the thermodynamic accounting framework for mass action networks,
        providing the conductance matrix that characterizes linear response near
        equilibrium or local linearization away from equilibrium.

        Modes:
        - "equilibrium": L = diag(v*) where v* ≈ 0.5(f + r) are equilibrium fluxes.
                        Exact at detailed balance, best near equilibrium.
        - "local": Local linearization L = -A @ pinv(S^T) where A = ∂J/∂(ln c).
                  Valid for general non-equilibrium states.
        - "auto": Automatically choose based on proximity to equilibrium.

        Args:
            concentrations: Current species concentrations [species]
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            mode: Computation mode ("equilibrium", "local", or "auto")
            equilibrium_tol: Relative tolerance for equilibrium detection
            enforce_reciprocity: Whether to symmetrize and ensure positive definiteness
            clip_eigenvalues: Minimum eigenvalue after reciprocity enforcement

        Returns:
            Dictionary containing:
            - 'L': Onsager conductance matrix [reactions × reactions]
            - 'forward_flux': Forward reaction fluxes [reactions]
            - 'reverse_flux': Reverse reaction fluxes [reactions]
            - 'net_flux': Net reaction fluxes [reactions]
            - 'mode_used': Actual computation mode used
            - 'near_equilibrium': Whether system is near equilibrium
            - 'reaction_forces': Current reaction forces x = S^T ln(c) - ln(K_eq)

        Examples:
            Simple A ⇌ B reaction:
            >>> network = ReactionNetwork(['A', 'B'], ['R1'], [[-1], [1]])
            >>> result = network.compute_onsager_conductance(
            ...     concentrations=[1.0, 2.0],
            ...     forward_rates=[2.0],
            ...     backward_rates=[1.0]
            ... )
            >>> print(f"L = {result['L']}, near_eq = {result['near_equilibrium']}")

        References:
            Diamond, S. (2025). "Log-Linear Reaction Quotient Dynamics"
        """
        # Input validation
        concentrations = np.asarray(concentrations, dtype=float)
        forward_rates = np.asarray(forward_rates, dtype=float)
        backward_rates = np.asarray(backward_rates, dtype=float)

        if len(concentrations) != self.n_species:
            raise ValueError(f"Expected {self.n_species} concentrations, got {len(concentrations)}")
        if len(forward_rates) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} forward rates, got {len(forward_rates)}")
        if len(backward_rates) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} backward rates, got {len(backward_rates)}")

        if not np.all(concentrations > 0):
            raise ValueError("All concentrations must be positive")
        if not np.all(forward_rates > 0):
            raise ValueError("All forward rates must be positive")
        if not np.all(backward_rates > 0):
            raise ValueError("All backward rates must be positive")

        # Compute flux components
        forward_flux, reverse_flux, net_flux = self._compute_flux_parts(concentrations, forward_rates, backward_rates)

        # Check if near equilibrium
        flux_balance = np.linalg.norm(net_flux)
        flux_magnitude = np.linalg.norm(forward_flux + reverse_flux) + 1e-30
        near_equilibrium = flux_balance <= equilibrium_tol * flux_magnitude

        # Choose computation mode
        if mode == "auto":
            mode_used = "equilibrium" if near_equilibrium else "local"
        elif mode in ["equilibrium", "local"]:
            mode_used = mode
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'equilibrium', 'local', or 'auto'.")

        # Compute Onsager conductance matrix
        if mode_used == "equilibrium":
            L = self._compute_onsager_equilibrium(forward_flux, reverse_flux)
        else:  # mode_used == "local"
            L = self._compute_onsager_local(concentrations, forward_rates, backward_rates, forward_flux, reverse_flux)

        # Optional reciprocity enforcement
        if enforce_reciprocity:
            L = self._enforce_onsager_reciprocity(L, clip_eigenvalues)

        # Compute reaction forces
        reaction_forces = self.compute_reaction_forces(concentrations, forward_rates, backward_rates)

        return {
            "L": L,
            "forward_flux": forward_flux,
            "reverse_flux": reverse_flux,
            "net_flux": net_flux,
            "mode_used": mode_used,
            "near_equilibrium": near_equilibrium,
            "reaction_forces": reaction_forces,
        }

    def _compute_onsager_equilibrium(self, forward_flux: np.ndarray, reverse_flux: np.ndarray) -> np.ndarray:
        """Compute equilibrium Onsager conductance L = diag(v*).

        At detailed balance, the conductance is diagonal with v* = f = r.
        For near-equilibrium states, we approximate v* ≈ 0.5(f + r).

        Args:
            forward_flux: Forward reaction fluxes [reactions]
            reverse_flux: Reverse reaction fluxes [reactions]

        Returns:
            Diagonal Onsager conductance matrix [reactions × reactions]
        """
        equilibrium_flux = 0.5 * (forward_flux + reverse_flux)
        return np.diag(equilibrium_flux)

    def _compute_onsager_local(
        self,
        concentrations: np.ndarray,
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        forward_flux: np.ndarray,
        reverse_flux: np.ndarray,
    ) -> np.ndarray:
        """Compute local linearization Onsager conductance L = -A @ pinv(S^T).

        Uses local linear response: J ≈ A * δ(ln c) where A = ∂J/∂(ln c).
        Since δ(ln c) = pinv(S^T) * δx for reaction forces x, we get L = -A @ pinv(S^T).

        Args:
            concentrations: Species concentrations [species]
            forward_rates: Forward rate constants [reactions]
            backward_rates: Backward rate constants [reactions]
            forward_flux: Forward reaction fluxes [reactions]
            reverse_flux: Reverse reaction fluxes [reactions]

        Returns:
            Onsager conductance matrix from local linearization [reactions × reactions]
        """
        # Compute flux Jacobian A = ∂J/∂(ln c)
        A = self._compute_flux_jacobian_ln(concentrations, forward_rates, backward_rates, forward_flux, reverse_flux)

        # Moore-Penrose pseudoinverse of S^T
        S_T_pinv = np.linalg.pinv(self.S.T)

        # L = -A @ pinv(S^T)
        L = -A @ S_T_pinv

        return L

    def _compute_flux_jacobian_ln(
        self,
        concentrations: np.ndarray,
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        forward_flux: np.ndarray,
        reverse_flux: np.ndarray,
    ) -> np.ndarray:
        """Compute flux Jacobian A = ∂J/∂(ln c) for local linearization.

        For mass action kinetics:
        ∂J_j/∂(ln c_i) = A_ij * f_j - B_ij * r_j

        where A_ij, B_ij are reactant/product stoichiometric coefficients.

        Args:
            concentrations: Species concentrations [species]
            forward_rates: Forward rate constants [reactions]
            backward_rates: Backward rate constants [reactions]
            forward_flux: Forward reaction fluxes [reactions]
            reverse_flux: Reverse reaction fluxes [reactions]

        Returns:
            Flux Jacobian matrix A [reactions × species]
        """
        # Get stoichiometry matrices
        A_matrix = self.get_reactant_stoichiometry_matrix()  # [species × reactions]
        B_matrix = self.get_product_stoichiometry_matrix()  # [species × reactions]

        # A = diag(f) @ A^T - diag(r) @ B^T
        # Shape: [reactions × species]
        jacobian = np.diag(forward_flux) @ A_matrix.T - np.diag(reverse_flux) @ B_matrix.T

        return jacobian

    def _enforce_onsager_reciprocity(self, L: np.ndarray, clip_eigenvalues: float = 1e-12) -> np.ndarray:
        """Enforce Onsager reciprocity (symmetry) and positive semi-definiteness.

        Symmetrizes the matrix and clips negative eigenvalues to ensure
        the matrix is positive semi-definite, as required by thermodynamics.

        Args:
            L: Input conductance matrix [reactions × reactions]
            clip_eigenvalues: Minimum eigenvalue after clipping

        Returns:
            Symmetrized positive semi-definite matrix [reactions × reactions]
        """
        # Enforce symmetry (Onsager reciprocity)
        L_symmetric = 0.5 * (L + L.T)

        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(L_symmetric)

        # Clip negative eigenvalues to ensure positive semi-definiteness
        eigenvals_clipped = np.maximum(eigenvals, clip_eigenvalues)

        # Reconstruct matrix
        L_psd = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T

        return L_psd

    def compute_reaction_forces(
        self, concentrations: np.ndarray, forward_rates: np.ndarray, backward_rates: np.ndarray
    ) -> np.ndarray:
        """Compute reaction forces x = S^T ln(c) - ln(K_eq).

        Reaction forces represent thermodynamic driving forces that push
        reactions away from equilibrium. At equilibrium, all forces are zero.

        Args:
            concentrations: Current species concentrations [species]
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]

        Returns:
            Reaction forces x [reactions]
        """
        concentrations = np.asarray(concentrations)

        # Compute ln(Q) = S^T ln(c) where Q are reaction quotients
        ln_c = np.log(np.maximum(concentrations, 1e-300))
        ln_Q = self.S.T @ ln_c

        # Compute ln(K_eq) = ln(k⁺/k⁻)
        ln_K_eq = np.log(forward_rates / backward_rates)

        # Reaction forces: x = ln(Q) - ln(K_eq) = ln(Q/K_eq)
        reaction_forces = ln_Q - ln_K_eq

        return reaction_forces

    def compute_flux_response_matrix(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute flux response matrix B(c) = S^T diag(1/c) S.

        This matrix appears in the relation: dx/dt = B(c) J
        where x = S^T ln(c) - ln(K_eq) are reaction forces and J are net fluxes.

        Args:
            concentrations: Species concentrations [species]

        Returns:
            Flux response matrix B [reactions × reactions]
        """
        concentrations = np.asarray(concentrations)

        if len(concentrations) != self.n_species:
            raise ValueError(f"Expected {self.n_species} concentrations, got {len(concentrations)}")
        if not np.all(concentrations > 0):
            raise ValueError("All concentrations must be positive")

        # Inverse concentration diagonal matrix
        inv_c_diag = np.diag(1.0 / concentrations)

        # B(c) = S^T diag(1/c) S
        B = self.S.T @ inv_c_diag @ self.S

        return B

    def compute_linear_relaxation_matrix(
        self, concentrations: np.ndarray, forward_rates: np.ndarray, backward_rates: np.ndarray, **onsager_kwargs
    ) -> Dict[str, Any]:
        """Compute linear relaxation matrix K(c) ≈ B(c) L(c).

        In the linear regime near equilibrium, reaction force dynamics follow:
        dx/dt ≈ -K(c) x + external_drives

        where K = B(c) L(c) combines flux response B and Onsager conductance L.

        Args:
            concentrations: Species concentrations [species]
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            **onsager_kwargs: Additional arguments passed to compute_onsager_conductance

        Returns:
            Dictionary containing:
            - 'K': Linear relaxation matrix [reactions × reactions]
            - 'B': Flux response matrix [reactions × reactions]
            - 'L': Onsager conductance matrix [reactions × reactions]
            - 'onsager_info': Full result from compute_onsager_conductance
        """
        # Compute Onsager conductance
        onsager_result = self.compute_onsager_conductance(concentrations, forward_rates, backward_rates, **onsager_kwargs)
        L = onsager_result["L"]

        # Compute flux response matrix
        B = self.compute_flux_response_matrix(concentrations)

        # Linear relaxation matrix
        K = B @ L

        return {
            "K": K,
            "B": B,
            "L": L,
            "onsager_info": onsager_result,
        }

    def check_detailed_balance(
        self, concentrations: np.ndarray, forward_rates: np.ndarray, backward_rates: np.ndarray, tol: float = 1e-10
    ) -> Dict[str, Any]:
        """Check if system satisfies detailed balance condition.

        Detailed balance requires that at equilibrium, forward and reverse
        fluxes are equal for each reaction: f_j = r_j for all j.

        This is equivalent to: Q_j = K_j where Q_j are reaction quotients
        and K_j = k_j⁺/k_j⁻ are equilibrium constants.

        Args:
            concentrations: Species concentrations [species]
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            tol: Tolerance for balance check

        Returns:
            Dictionary containing:
            - 'detailed_balance': Whether detailed balance is satisfied
            - 'forward_flux': Forward reaction fluxes [reactions]
            - 'reverse_flux': Reverse reaction fluxes [reactions]
            - 'flux_ratio': Ratio f_j/r_j for each reaction [reactions]
            - 'reaction_quotients': Current reaction quotients Q_j [reactions]
            - 'equilibrium_constants': Equilibrium constants K_j [reactions]
            - 'quotient_ratio': Ratio Q_j/K_j for each reaction [reactions]
            - 'max_imbalance': Maximum relative imbalance across reactions
        """
        # Compute fluxes
        forward_flux, reverse_flux, _ = self._compute_flux_parts(concentrations, forward_rates, backward_rates)

        # Compute reaction quotients and equilibrium constants
        Q = self.compute_reaction_quotients(concentrations)
        K = forward_rates / backward_rates

        # Check balance conditions
        flux_ratio = np.divide(forward_flux, reverse_flux, out=np.ones_like(forward_flux), where=reverse_flux != 0)
        quotient_ratio = np.divide(Q, K, out=np.ones_like(Q), where=K != 0)

        # Maximum imbalance (should be ~1 at detailed balance)
        max_flux_imbalance = np.max(np.abs(flux_ratio - 1))
        max_quotient_imbalance = np.max(np.abs(quotient_ratio - 1))
        max_imbalance = max(max_flux_imbalance, max_quotient_imbalance)

        # Overall detailed balance condition
        detailed_balance = max_imbalance <= tol

        return {
            "detailed_balance": detailed_balance,
            "forward_flux": forward_flux,
            "reverse_flux": reverse_flux,
            "flux_ratio": flux_ratio,
            "reaction_quotients": Q,
            "equilibrium_constants": K,
            "quotient_ratio": quotient_ratio,
            "max_imbalance": max_imbalance,
        }

    @classmethod
    def from_sbml_data(cls, sbml_data: Dict[str, Any]) -> "ReactionNetwork":
        """Create ReactionNetwork from SBML parser data.

        Args:
            sbml_data: Dictionary from SBMLParser.extract_network_data()

        Returns:
            ReactionNetwork instance
        """
        return cls(
            species_ids=sbml_data["species_ids"],
            reaction_ids=sbml_data["reaction_ids"],
            stoichiometric_matrix=sbml_data["stoichiometric_matrix"],
            species_info=sbml_data["species"],
            reaction_info=sbml_data["reactions"],
            parameters=sbml_data["parameters"],
        )
