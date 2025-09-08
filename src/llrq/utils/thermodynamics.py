"""Thermodynamic utilities for LLRQ framework.

This module provides functions for converting between Gibbs free energy values
and equilibrium constants, which are essential for accurate thermodynamic
modeling in the LLRQ framework.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union


# Physical constants
R_JOULE_MOL_K = 8.314462618  # Universal gas constant in J/(mol·K)
R_KCAL_MOL_K = R_JOULE_MOL_K / 4184.0  # Universal gas constant in kcal/(mol·K)


def delta_g_to_keq(delta_g: Union[float, np.ndarray], T: float = 298.15, units: str = "kJ/mol") -> Union[float, np.ndarray]:
    """Convert Gibbs free energy change to equilibrium constant.

    Uses the fundamental thermodynamic relationship:
    ΔG° = -RT ln(Keq)  =>  Keq = exp(-ΔG°/RT)

    Args:
        delta_g: Gibbs free energy change (ΔG°)
        T: Temperature in Kelvin (default: 298.15 K = 25°C)
        units: Units of delta_g ('kJ/mol', 'kcal/mol', 'J/mol')

    Returns:
        Equilibrium constant(s) Keq (dimensionless)

    Examples:
        >>> # Favorable reaction: ΔG° = -10 kJ/mol
        >>> keq = delta_g_to_keq(-10.0)
        >>> print(f"Keq = {keq:.2f}")  # Keq ≈ 55

        >>> # Unfavorable reaction: ΔG° = +10 kJ/mol
        >>> keq = delta_g_to_keq(10.0)
        >>> print(f"Keq = {keq:.4f}")  # Keq ≈ 0.018
    """
    delta_g = np.asarray(delta_g, dtype=float)

    # Convert to J/mol if needed
    if units.lower() == "kj/mol":
        delta_g_joules = delta_g * 1000.0
    elif units.lower() == "kcal/mol":
        delta_g_joules = delta_g * 4184.0
    elif units.lower() == "j/mol":
        delta_g_joules = delta_g
    else:
        raise ValueError(f"Unsupported units: {units}. Use 'kJ/mol', 'kcal/mol', or 'J/mol'")

    # Compute Keq = exp(-ΔG°/RT)
    keq = np.exp(-delta_g_joules / (R_JOULE_MOL_K * T))

    # Return scalar if input was scalar
    if np.isscalar(delta_g):
        return float(keq)
    else:
        return keq


def keq_to_delta_g(keq: Union[float, np.ndarray], T: float = 298.15, units: str = "kJ/mol") -> Union[float, np.ndarray]:
    """Convert equilibrium constant to Gibbs free energy change.

    Uses the fundamental thermodynamic relationship:
    ΔG° = -RT ln(Keq)

    Args:
        keq: Equilibrium constant(s) (dimensionless, must be > 0)
        T: Temperature in Kelvin (default: 298.15 K = 25°C)
        units: Desired units for ΔG° ('kJ/mol', 'kcal/mol', 'J/mol')

    Returns:
        Gibbs free energy change(s) ΔG°

    Examples:
        >>> # Large Keq = 100 => negative ΔG° (favorable)
        >>> dg = keq_to_delta_g(100.0)
        >>> print(f"ΔG° = {dg:.2f} kJ/mol")  # ΔG° ≈ -11.4 kJ/mol

        >>> # Small Keq = 0.01 => positive ΔG° (unfavorable)
        >>> dg = keq_to_delta_g(0.01)
        >>> print(f"ΔG° = {dg:.2f} kJ/mol")  # ΔG° ≈ +11.4 kJ/mol
    """
    keq = np.asarray(keq, dtype=float)

    # Check for positive values
    if np.any(keq <= 0):
        raise ValueError("Equilibrium constants must be positive")

    # Compute ΔG° = -RT ln(Keq) in J/mol
    delta_g_joules = -R_JOULE_MOL_K * T * np.log(keq)

    # Convert units if needed
    if units.lower() == "kj/mol":
        delta_g = delta_g_joules / 1000.0
    elif units.lower() == "kcal/mol":
        delta_g = delta_g_joules / 4184.0
    elif units.lower() == "j/mol":
        delta_g = delta_g_joules
    else:
        raise ValueError(f"Unsupported units: {units}. Use 'kJ/mol', 'kcal/mol', or 'J/mol'")

    # Return scalar if input was scalar
    if np.isscalar(keq):
        return float(delta_g)
    else:
        return delta_g


def compute_reaction_delta_g(
    metabolite_delta_g: Dict[str, float],
    stoichiometry: Dict[str, float],
    missing_value: float = 10000000.0,
    default_delta_g: Optional[float] = None,
) -> Tuple[Optional[float], List[str]]:
    """Compute reaction ΔG° from metabolite formation energies.

    For a reaction: aA + bB -> cC + dD
    The reaction ΔG° is: ΔG°_rxn = (c·ΔG°_C + d·ΔG°_D) - (a·ΔG°_A + b·ΔG°_B)

    Args:
        metabolite_delta_g: Dictionary mapping metabolite IDs to formation ΔG° values
        stoichiometry: Dictionary mapping metabolite IDs to stoichiometric coefficients
                      (negative for reactants, positive for products)
        missing_value: Value indicating missing/unknown ΔG° (default: 10000000.0)
        default_delta_g: Default ΔG° to use for missing metabolites (None = skip reaction)

    Returns:
        Tuple of (reaction_delta_g, missing_metabolites):
        - reaction_delta_g: ΔG° for the reaction (None if incomplete data)
        - missing_metabolites: List of metabolite IDs with missing ΔG° data

    Examples:
        >>> # A -> B reaction where A has ΔG° = -100 kJ/mol, B has ΔG° = -120 kJ/mol
        >>> metabolite_dg = {'A': -100.0, 'B': -120.0}
        >>> stoich = {'A': -1, 'B': 1}  # A consumed, B produced
        >>> rxn_dg, missing = compute_reaction_delta_g(metabolite_dg, stoich)
        >>> print(f"Reaction ΔG° = {rxn_dg} kJ/mol")  # -20 kJ/mol (favorable)
    """
    missing_metabolites = []
    reactant_sum = 0.0
    product_sum = 0.0

    for metabolite_id, coeff in stoichiometry.items():
        if metabolite_id not in metabolite_delta_g:
            missing_metabolites.append(metabolite_id)
            continue

        delta_g_val = metabolite_delta_g[metabolite_id]

        # Check if this is a missing/placeholder value
        if abs(delta_g_val - missing_value) < 1e-6:
            missing_metabolites.append(metabolite_id)
            continue

        # Add to appropriate sum based on stoichiometry
        if coeff < 0:  # Reactant
            reactant_sum += abs(coeff) * delta_g_val
        else:  # Product
            product_sum += coeff * delta_g_val

    # If we have missing metabolites and no default, return None
    if missing_metabolites and default_delta_g is None:
        return None, missing_metabolites

    # Use default value for missing metabolites if provided
    if missing_metabolites and default_delta_g is not None:
        for metabolite_id, coeff in stoichiometry.items():
            if metabolite_id in missing_metabolites:
                if coeff < 0:  # Reactant
                    reactant_sum += abs(coeff) * default_delta_g
                else:  # Product
                    product_sum += coeff * default_delta_g

    # Compute reaction ΔG° = products - reactants
    reaction_delta_g = product_sum - reactant_sum

    return reaction_delta_g, missing_metabolites


def metabolite_delta_g_to_reaction_keq(
    metabolite_delta_g: Dict[str, float],
    reactions: List[Dict],
    T: float = 298.15,
    delta_g_units: str = "kJ/mol",
    missing_value: float = 10000000.0,
    default_keq: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Convert metabolite ΔG° values to reaction equilibrium constants.

    This function computes Keq for all reactions in a network by:
    1. Computing reaction ΔG° from metabolite formation energies
    2. Converting reaction ΔG° to Keq using thermodynamic relations

    Args:
        metabolite_delta_g: Dictionary of metabolite formation ΔG° values
        reactions: List of reaction dictionaries with 'metabolites' field
        T: Temperature in Kelvin
        delta_g_units: Units of ΔG° values ('kJ/mol', 'kcal/mol', 'J/mol')
        missing_value: Placeholder value for missing ΔG° data
        default_keq: Default Keq for reactions with incomplete thermodynamic data
        verbose: Whether to print warnings about missing data

    Returns:
        Tuple of (keq_array, info):
        - keq_array: Array of equilibrium constants for each reaction
        - info: Dictionary with statistics about thermodynamic data coverage

    Examples:
        >>> reactions = [
        ...     {'id': 'R1', 'metabolites': {'A': -1, 'B': 1}},
        ...     {'id': 'R2', 'metabolites': {'B': -1, 'C': 1}}
        ... ]
        >>> metabolite_dg = {'A': -100, 'B': -120, 'C': -90}
        >>> keq_values, info = metabolite_delta_g_to_reaction_keq(metabolite_dg, reactions)
    """
    n_reactions = len(reactions)
    keq_array = np.ones(n_reactions) * default_keq

    # Statistics tracking
    reactions_with_data = 0
    reactions_with_partial_data = 0
    reactions_with_no_data = 0
    total_missing_metabolites = set()

    for i, reaction in enumerate(reactions):
        reaction_id = reaction.get("id", f"R{i}")
        stoichiometry = reaction["metabolites"]

        # Compute reaction ΔG°
        rxn_delta_g, missing_metabolites = compute_reaction_delta_g(metabolite_delta_g, stoichiometry, missing_value)

        if rxn_delta_g is not None:
            # Convert to Keq
            try:
                keq_array[i] = delta_g_to_keq(rxn_delta_g, T, delta_g_units)

                if missing_metabolites:
                    reactions_with_partial_data += 1
                    if verbose:
                        warnings.warn(
                            f"Reaction {reaction_id}: Missing ΔG° for {missing_metabolites}, "
                            f"but computed Keq = {keq_array[i]:.3f}"
                        )
                else:
                    reactions_with_data += 1

            except (OverflowError, ValueError) as e:
                if verbose:
                    warnings.warn(f"Reaction {reaction_id}: Error computing Keq from ΔG° = {rxn_delta_g}: {e}")
                keq_array[i] = default_keq
                reactions_with_no_data += 1
        else:
            # No thermodynamic data available
            keq_array[i] = default_keq
            reactions_with_no_data += 1
            total_missing_metabolites.update(missing_metabolites)

            if verbose and missing_metabolites:
                warnings.warn(
                    f"Reaction {reaction_id}: Missing ΔG° for {missing_metabolites}, " f"using default Keq = {default_keq}"
                )

    # Compile information
    info = {
        "n_reactions": n_reactions,
        "reactions_with_complete_data": reactions_with_data,
        "reactions_with_partial_data": reactions_with_partial_data,
        "reactions_with_no_data": reactions_with_no_data,
        "coverage_complete": reactions_with_data / n_reactions if n_reactions > 0 else 0,
        "coverage_partial": (reactions_with_data + reactions_with_partial_data) / n_reactions if n_reactions > 0 else 0,
        "unique_missing_metabolites": len(total_missing_metabolites),
        "missing_metabolite_ids": sorted(total_missing_metabolites),
        "default_keq_used": default_keq,
        "temperature_K": T,
    }

    return keq_array, info


def validate_thermodynamic_consistency(
    keq_values: np.ndarray, stoichiometric_matrix: np.ndarray, rtol: float = 1e-6
) -> Tuple[bool, float]:
    """Check thermodynamic consistency of equilibrium constants.

    For thermodynamically consistent Keq values, the Wegscheider conditions
    must be satisfied: for any cycle in the reaction network, the product
    of forward Keq values equals the product of reverse Keq values.

    This is equivalent to checking that ln(Keq) is in the row space of
    the stoichiometric matrix (i.e., satisfies conservation laws).

    Args:
        keq_values: Array of equilibrium constants
        stoichiometric_matrix: Stoichiometric matrix (species × reactions)
        rtol: Relative tolerance for consistency check

    Returns:
        Tuple of (is_consistent, max_violation):
        - is_consistent: True if thermodynamically consistent
        - max_violation: Maximum violation of Wegscheider conditions

    Note:
        This function checks necessary but not sufficient conditions for
        thermodynamic consistency. Full consistency requires checking
        all reaction cycles, which is computationally expensive for
        large networks.
    """
    ln_keq = np.log(keq_values)

    # Check if ln(Keq) satisfies the conservation constraints
    # This means ln(Keq) should be orthogonal to the left nullspace of S
    from .equilibrium_utils import _left_nullspace

    # Get conservation matrix (left nullspace of stoichiometric matrix)
    L = _left_nullspace(stoichiometric_matrix)

    if L.shape[1] == 0:
        # No conservation laws, so no constraints on Keq
        return True, 0.0

    # Check violation: L^T * ln(Keq) should be close to zero
    violations = L.T @ ln_keq
    max_violation = np.max(np.abs(violations))

    is_consistent = max_violation < rtol * np.max(np.abs(ln_keq))

    return is_consistent, max_violation
