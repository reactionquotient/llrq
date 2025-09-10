"""Physical bounds utilities for K matrix constraints.

This module provides helper functions for computing physically meaningful bounds
on the conductance matrix G (and hence relaxation matrix K = A*G) based on:

1. Diffusion limitations: k_on ∈ [10^6, 10^9] M^-1 s^-1
2. Enzyme concentrations: G_ii ~ (k_cat/K_m) * [E_i] * thermodynamic_factor
3. Spectral constraints: eigenvalue bounds from physical considerations
4. Thermodynamic consistency: detailed balance and Onsager relations

These bounds are used in the convex optimization framework for K estimation.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.constants import k as k_boltzmann, N_A

# Physical constants
DEFAULT_K_ON_RANGE = (1e6, 1e9)  # M^-1 s^-1, typical protein-ligand binding
DEFAULT_TEMPERATURE = 298.15  # K (25°C)


def compute_diffusion_limit(
    molecular_weights_A: Union[float, np.ndarray],
    molecular_weights_B: Optional[Union[float, np.ndarray]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    viscosity: float = 1e-3,  # Pa⋅s (water at 25°C)
    reaction_radius: Optional[float] = None,  # m (auto-computed if None)
) -> np.ndarray:
    """Compute diffusion-limited association rate constants.

    Uses the Smoluchowski equation for diffusion-limited reactions:
    k_on = 4π * (D_A + D_B) * a * N_A

    where D_A, D_B are diffusion coefficients and a is the reaction radius.

    Args:
        molecular_weights_A: Molecular weight(s) of species A in g/mol
        molecular_weights_B: Molecular weight(s) of species B in g/mol.
                            If None, assumes B = A (self-reaction)
        temperature: Temperature in K
        viscosity: Solution viscosity in Pa⋅s
        reaction_radius: Reaction radius in m. If None, uses sum of hydrodynamic radii

    Returns:
        Diffusion-limited rate constants in M^-1 s^-1
    """
    mA = np.asarray(molecular_weights_A, dtype=float)
    mB = mA if molecular_weights_B is None else np.asarray(molecular_weights_B, dtype=float)

    # Stokes-Einstein relation: D = kT / (6πηr)
    # Approximate molecular radius: r ≈ 0.66 * (MW/1000)^(1/3) nm for proteins
    rA = 0.66e-9 * (mA / 1000.0) ** (1 / 3)  # m
    rB = 0.66e-9 * (mB / 1000.0) ** (1 / 3)  # m

    # Diffusion coefficients
    D_A = k_boltzmann * temperature / (6 * np.pi * viscosity * rA)
    D_B = k_boltzmann * temperature / (6 * np.pi * viscosity * rB)
    D_rel = D_A + D_B

    # Reaction radius: sum of hydrodynamic radii if not specified
    a = (rA + rB) if reaction_radius is None else reaction_radius

    # Smoluchowski rate constant
    k_on_SI = 4 * np.pi * D_rel * a * N_A  # m^3 mol^-1 s^-1
    k_on = k_on_SI * 1e3  # Convert to M^-1 s^-1 (L mol^-1 s^-1)

    return k_on


def enzyme_to_conductance(
    enzyme_concentrations: np.ndarray,
    kcat_values: np.ndarray,
    km_values: np.ndarray,
    equilibrium_concentrations: np.ndarray,
    stoichiometry_matrix: np.ndarray,
    thermodynamic_factors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Convert enzyme parameters to conductance bounds.

    For enzyme-catalyzed reactions, the conductance is approximately:
    G_ii ≈ (k_cat/K_m) * [E_i] * f_thermo

    where f_thermo is a heuristic thermodynamic factor accounting for substrate/product
    concentrations relative to K_m values.

    Args:
        enzyme_concentrations: Enzyme concentrations [E_i] in M
        kcat_values: Turnover numbers k_cat in s^-1
        km_values: Michaelis constants K_m in M (single value per reaction)
        equilibrium_concentrations: Substrate/product concentrations in M
        stoichiometry_matrix: Stoichiometry matrix [species × reactions]
        thermodynamic_factors: Optional pre-computed factors (default: estimate heuristically)

    Returns:
        Dictionary with:
        - g_diag_max: Upper bounds on diag(G)
        - g_diag_min: Lower bounds on diag(G)
        - thermodynamic_factors: Computed thermodynamic factors
        - catalytic_efficiencies: k_cat/K_m values
    """
    enzyme_concentrations = np.asarray(enzyme_concentrations)
    kcat_values = np.asarray(kcat_values)
    km_values = np.asarray(km_values)
    equilibrium_concentrations = np.asarray(equilibrium_concentrations)

    n_reactions = len(enzyme_concentrations)

    if len(kcat_values) != n_reactions:
        raise ValueError("kcat_values must match number of reactions")
    if len(km_values) != n_reactions:
        raise ValueError("km_values must match number of reactions")

    # Catalytic efficiencies
    cat_efficiencies = kcat_values / km_values  # M^-1 s^-1

    # Estimate thermodynamic factors if not provided
    if thermodynamic_factors is None:
        thermodynamic_factors = _estimate_thermodynamic_factors_heuristic(
            equilibrium_concentrations, stoichiometry_matrix, km_values
        )

    # Conductance bounds
    g_diag_max = cat_efficiencies * enzyme_concentrations * thermodynamic_factors

    # Lower bounds: assume enzymes can be "turned off" but not completely
    # Use 1% of maximum as a reasonable lower bound
    g_diag_min = 0.01 * g_diag_max

    return {
        "g_diag_max": g_diag_max,
        "g_diag_min": g_diag_min,
        "thermodynamic_factors": thermodynamic_factors,
        "catalytic_efficiencies": cat_efficiencies,
    }


def _estimate_thermodynamic_factors_heuristic(
    concentrations: np.ndarray, stoichiometry_matrix: np.ndarray, km_values: np.ndarray
) -> np.ndarray:
    """Estimate thermodynamic factors heuristically for enzyme kinetics.

    WARNING: This is a rough heuristic that uses a single K_m per reaction
    and only considers reactant species. In reality, K_m values are
    substrate-specific and can vary by orders of magnitude.

    The heuristic computes:
    f_thermo,j ≈ geometric_mean over reactants of (c_s / K_m,j)^|ν_sj|

    This should be replaced with proper substrate-specific K_m values
    or experimental measurements when available.

    Args:
        concentrations: Species concentrations
        stoichiometry_matrix: [species × reactions]
        km_values: Single K_m value for each reaction (heuristic)

    Returns:
        Thermodynamic factors for each reaction (heavily clamped)
    """
    warnings.warn(
        "Using heuristic thermodynamic factors with single Km per reaction. "
        "Consider using substrate-specific Km values or experimental data.",
        UserWarning,
    )

    n_reactions = stoichiometry_matrix.shape[1]
    factors = np.ones(n_reactions)

    for j in range(n_reactions):
        stoich_j = stoichiometry_matrix[:, j]

        # Find reactant species (negative stoichiometry)
        reactants = np.where(stoich_j < 0)[0]

        if len(reactants) == 0:
            continue

        # Use geometric mean of concentration/K_m ratios for reactants only
        # Weight by stoichiometric coefficient magnitude
        log_factor = 0.0
        total_weight = 0.0

        for s in reactants:
            weight = abs(stoich_j[s])
            conc_ratio = concentrations[s] / km_values[j]
            conc_ratio = max(conc_ratio, 1e-6)  # Avoid log(0)

            log_factor += weight * np.log(conc_ratio)
            total_weight += weight

        if total_weight > 0:
            factors[j] = np.exp(log_factor / total_weight)

        # Aggressively clamp to reasonable range [0.01, 100]
        # This is essential given the heuristic nature
        factors[j] = np.clip(factors[j], 0.01, 100.0)

    return factors


def compute_spectral_caps(
    g_diag_bounds: Tuple[np.ndarray, np.ndarray], coupling_strength: float = 0.1, safety_factor: float = 1.2
) -> Dict[str, float]:
    """Compute spectral bounds γ_min, γ_max from diagonal bounds.

    Conservative bounds based on Gershgorin circle theorem:
    - γ_max ≤ max_i(g_diag_max_i) * safety_factor * (1 + coupling_strength)
    - γ_min ≥ max(0, min_i(g_diag_min_i) / safety_factor * (1 - coupling_strength))

    Args:
        g_diag_bounds: Tuple of (g_diag_min, g_diag_max) arrays
        coupling_strength: Expected strength of off-diagonal coupling (0-1)
        safety_factor: Safety factor for conservative bounds (>1)

    Returns:
        Dictionary with γ_min and γ_max bounds for eigenvalues of G
    """
    g_diag_min, g_diag_max = g_diag_bounds

    # Conservative spectral bounds
    gamma_max = float(np.max(g_diag_max)) * safety_factor
    gamma_min = max(0.0, float(np.min(g_diag_min)) / safety_factor)

    # Account for coupling effects
    # Off-diagonal terms can increase largest eigenvalue
    gamma_max *= 1 + coupling_strength

    # Off-diagonal terms can decrease smallest eigenvalue
    # Ensure gamma_min never goes negative
    if gamma_min > 0:
        gamma_min *= max(0.01, 1 - coupling_strength)  # Floor at 1% to avoid negative

    return {"gamma_max": gamma_max, "gamma_min": gamma_min}


def validate_physical_consistency(
    K_matrix: np.ndarray,
    A_matrix: np.ndarray,
    g_diag_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    spectral_bounds: Optional[Dict[str, float]] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Validate that K matrix satisfies physical consistency constraints.

    Uses the correct similarity variable W = A^{-1/2} K A^{1/2} to check:
    1. W ≽ 0 (equivalent to existence of G ≽ 0 with K = A*G)
    2. Diagonal bounds on G = A^{-1/2} W A^{-1/2}
    3. Spectral bounds on G
    4. Onsager/detailed balance: ||A*G - (A*G)^T||_F ≈ 0

    Args:
        K_matrix: Relaxation matrix to validate
        A_matrix: Stoichiometric matrix A = N^T diag(1/c_eq) N
        g_diag_bounds: Optional (g_min, g_max) diagonal bounds on G
        spectral_bounds: Optional dict with γ_min, γ_max bounds on eigenvalues of G
        tolerance: Numerical tolerance for constraint violations

    Returns:
        Dictionary with validation results and violation magnitudes
    """
    # Symmetrize A for numerical stability
    A = 0.5 * (A_matrix + A_matrix.T)

    # Eigendecomposition of A to compute A^{1/2} and A^{-1/2}
    lam, U = np.linalg.eigh(A)
    tol_A = max(1e-12, 1e-12 * np.max(lam))
    keep = lam > tol_A

    if not np.any(keep):
        return {
            "is_physically_consistent": False,
            "violations": {"A_rank_zero": "A matrix is approximately zero"},
            "A_rank": 0,
        }

    # Compute A^{1/2} and A^{-1/2} using kept eigenvalues
    A_half = (U[:, keep] * np.sqrt(lam[keep])) @ U[:, keep].T
    A_inhalf = (U[:, keep] * (1 / np.sqrt(lam[keep]))) @ U[:, keep].T

    # Compute similarity variable W = A^{-1/2} K A^{1/2}
    W = A_inhalf @ K_matrix @ A_half
    W_sym = 0.5 * (W + W.T)  # Symmetrize for numerical stability

    # Check if W is PSD
    eigW = np.linalg.eigvalsh(W_sym)
    min_W_eig = float(np.min(eigW))
    max_W_eig = float(np.max(eigW))

    violations: Dict[str, float] = {}
    results = {
        "is_physically_consistent": True,
        "violations": violations,
        "W_min_eigenvalue": min_W_eig,
        "W_max_eigenvalue": max_W_eig,
        "A_rank": int(np.sum(keep)),
    }

    # Check 1: W ≽ 0 (PSD condition)
    if min_W_eig < -tolerance:
        results["is_physically_consistent"] = False
        violations["W_not_PSD"] = abs(min_W_eig)

    # Compute G = A^{-1/2} W A^{-1/2} for further checks
    G = A_inhalf @ W_sym @ A_inhalf
    G_sym = 0.5 * (G + G.T)  # Ensure symmetry
    g_diagonal = np.diag(G_sym)
    G_eigenvals = np.linalg.eigvalsh(G_sym)

    results.update({"G_matrix": G_sym, "G_diagonal": g_diagonal, "G_eigenvalues": G_eigenvals})

    # Check 2: Diagonal bounds on G
    if g_diag_bounds is not None:
        g_diag_min, g_diag_max = g_diag_bounds

        # Lower bound violations
        lower_violations = np.maximum(0.0, g_diag_min - g_diagonal)
        if np.any(lower_violations > tolerance):
            results["is_physically_consistent"] = False
            violations["diagonal_lower"] = float(np.max(lower_violations))

        # Upper bound violations
        upper_violations = np.maximum(0.0, g_diagonal - g_diag_max)
        if np.any(upper_violations > tolerance):
            results["is_physically_consistent"] = False
            violations["diagonal_upper"] = float(np.max(upper_violations))

        results["diagonal_violations"] = {"lower": lower_violations, "upper": upper_violations}

    # Check 3: Spectral bounds on G
    if spectral_bounds is not None:
        G_min_eig = float(np.min(G_eigenvals))
        G_max_eig = float(np.max(G_eigenvals))

        if "gamma_max" in spectral_bounds:
            gamma_max = spectral_bounds["gamma_max"]
            if G_max_eig > gamma_max + tolerance:
                results["is_physically_consistent"] = False
                violations["spectral_upper"] = G_max_eig - gamma_max

        if "gamma_min" in spectral_bounds:
            gamma_min = spectral_bounds["gamma_min"]
            if G_min_eig < gamma_min - tolerance:
                results["is_physically_consistent"] = False
                violations["spectral_lower"] = gamma_min - G_min_eig

        results["G_min_eigenvalue"] = G_min_eig
        results["G_max_eigenvalue"] = G_max_eig

    # Check 4: Onsager/detailed balance symmetry
    # For K = A*G with symmetric G, we should have A*G = (A*G)^T
    AG = A @ G_sym
    onsager_residual = np.linalg.norm(AG - AG.T, ord="fro")
    results["onsager_residual_frobenius"] = float(onsager_residual)

    # Large Onsager residual indicates violation of detailed balance
    onsager_threshold = tolerance * np.linalg.norm(AG, ord="fro")
    if onsager_residual > onsager_threshold:
        violations["onsager_symmetry"] = float(onsager_residual)
        # Note: Don't mark as inconsistent since this might be expected for some systems

    return results


def gershgorin_bounds(
    g_diagonal: np.ndarray, coupling_matrix: Optional[np.ndarray] = None, coupling_strength: float = 0.1
) -> Dict[str, Any]:
    """Compute eigenvalue bounds from Gershgorin circle theorem.

    For a conductance matrix G with diagonal g_ii and off-diagonal coupling,
    the Gershgorin theorem gives bounds on eigenvalues of G (not K):

    λ_i ∈ [g_ii - R_i, g_ii + R_i]

    where R_i = Σ_{j≠i} |G_ij| is the i-th row sum of off-diagonal elements.

    Note: These bounds apply to eigenvalues of G, not directly to K.

    Args:
        g_diagonal: Diagonal elements of G
        coupling_matrix: Full G matrix (if available) or None
        coupling_strength: Estimated coupling strength if matrix not available

    Returns:
        Dictionary with eigenvalue bounds and radius estimates for G matrix
    """
    n = len(g_diagonal)

    if coupling_matrix is not None:
        # Compute exact Gershgorin radii
        radii = np.sum(np.abs(coupling_matrix), axis=1) - np.abs(g_diagonal)
    else:
        # Estimate radii from coupling strength
        # Assume uniform coupling: R_i ≈ coupling_strength * |g_ii| * (n-1)
        radii = coupling_strength * np.abs(g_diagonal) * (n - 1)

    # Eigenvalue bounds for G matrix
    lambda_lower = g_diagonal - radii
    lambda_upper = g_diagonal + radii

    # Global bounds
    global_lower = np.min(lambda_lower)
    global_upper = np.max(lambda_upper)

    return {
        "eigenvalue_bounds": {"lower": lambda_lower, "upper": lambda_upper},
        "global_bounds": {"lambda_min_bound": global_lower, "lambda_max_bound": global_upper},
        "gershgorin_radii": radii,
    }


def estimate_reaction_timescales(
    K_matrix: np.ndarray, A_matrix: Optional[np.ndarray] = None, tolerance: float = 1e-12
) -> Dict[str, Any]:
    """Estimate reaction timescales from eigenvalues of W = A^{-1/2} K A^{1/2}.

    The relaxation timescales are τ_i = 1/λ_i where λ_i are the eigenvalues
    of W (the similarity transform that makes the problem well-conditioned).

    Args:
        K_matrix: Relaxation matrix
        A_matrix: Stoichiometric matrix for proper similarity transform.
                 If None, assumes A ≈ I (less robust)
        tolerance: Threshold for considering eigenvalues as zero

    Returns:
        Dictionary with timescales and eigenvalue analysis
    """
    if A_matrix is None:
        # Fallback: symmetrize K and use directly (not ideal)
        warnings.warn("No A matrix provided. Using symmetrized K directly, " "which may be numerically unstable.", UserWarning)
        W = 0.5 * (K_matrix + K_matrix.T)
        space_type = "fallback_symmetric_K"
    else:
        # Proper approach: use W = A^{-1/2} K A^{1/2}
        A = 0.5 * (A_matrix + A_matrix.T)  # Symmetrize
        lam, U = np.linalg.eigh(A)
        tol_A = max(tolerance, tolerance * np.max(lam))
        keep = lam > tol_A

        if not np.any(keep):
            # A is approximately zero - degenerate case
            return {
                "eigenvalues": np.array([]),
                "timescales": np.array([]),
                "fastest_timescale": np.inf,
                "slowest_finite_timescale": np.inf,
                "num_zero_modes": 0,
                "space_type": "degenerate_A_zero",
            }

        A_half = (U[:, keep] * np.sqrt(lam[keep])) @ U[:, keep].T
        A_inhalf = (U[:, keep] * (1 / np.sqrt(lam[keep]))) @ U[:, keep].T
        W = A_inhalf @ K_matrix @ A_half
        W = 0.5 * (W + W.T)  # Symmetrize
        space_type = "proper_similarity_transform"

    # Compute eigenvalues of symmetric W
    eigenvals = np.linalg.eigvalsh(W)  # Real, sorted ascending

    # Compute timescales (avoid division by zero)
    positive_mask = eigenvals > tolerance
    timescales = np.full_like(eigenvals, np.inf, dtype=float)
    timescales[positive_mask] = 1.0 / eigenvals[positive_mask]

    # Statistics
    finite_timescales = timescales[np.isfinite(timescales)]

    return {
        "eigenvalues": eigenvals,
        "timescales": timescales,
        "fastest_timescale": np.min(finite_timescales) if len(finite_timescales) > 0 else np.inf,
        "slowest_finite_timescale": np.max(finite_timescales) if len(finite_timescales) > 0 else np.inf,
        "num_zero_modes": int(np.sum(~positive_mask)),
        "num_finite_modes": int(np.sum(positive_mask)),
        "space_type": space_type,
    }
