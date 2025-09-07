"""
LLRQ to Mass Action Control Mapping

This module implements the mathematically correct mapping from LLRQ control inputs
to mass action rate constant modifications.

Key insight: LLRQ control u is additive in log-space, corresponding to shifting
ln(Keq) in mass action kinetics. This is achieved by asymmetric rate modifications.
"""

from typing import Tuple

import numpy as np


def apply_llrq_drive_to_rates(
    kf_base: np.ndarray, kr_base: np.ndarray, B: np.ndarray, K_red: np.ndarray, u_red: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert LLRQ reduced control to mass action rate modifications.

    The mathematical chain:
    1. u_red → Δ = K_red^(-1) @ u_red  (solve linear system)
    2. Δ → δ = B @ Δ                   (expand to full reaction space)
    3. δ → (kf', kr') via asymmetric exponential modification

    This implements: ln(Keq_new) = ln(Keq_old) + δ

    Args:
        kf_base: Base forward rate constants (r,)
        kr_base: Base reverse rate constants (r,)
        B: Basis matrix for Im(S^T) from LLRQ solver (r x rankS)
        K_red: Reduced relaxation matrix (rankS x rankS)
        u_red: Reduced control input (rankS,)

    Returns:
        (kf_new, kr_new): Modified rate constants
    """
    # Step 1: Solve K_red @ Δ = u_red for Δ
    Delta = np.linalg.lstsq(K_red, u_red, rcond=None)[0]

    # Step 2: Expand to full reaction space
    delta = B @ Delta

    # Step 3: Asymmetric rate modification
    # This shifts ln(Keq) by δ: Keq_new = Keq_old * exp(δ)
    kf_new = kf_base * np.exp(+0.5 * delta)
    kr_new = kr_base * np.exp(-0.5 * delta)

    return kf_new, kr_new


def compute_equilibrium_shift(kf_base: np.ndarray, kr_base: np.ndarray, kf_new: np.ndarray, kr_new: np.ndarray) -> np.ndarray:
    """Compute the equilibrium constant shift from rate modifications.

    Args:
        kf_base, kr_base: Original rate constants
        kf_new, kr_new: Modified rate constants

    Returns:
        delta: The shift in ln(Keq) for each reaction
    """
    Keq_base = kf_base / kr_base
    Keq_new = kf_new / kr_new

    return np.log(Keq_new / Keq_base)


def validate_llrq_control_mapping(
    B: np.ndarray, K_red: np.ndarray, kf_base: np.ndarray, kr_base: np.ndarray, u_red: np.ndarray
) -> dict:
    """Validate the LLRQ control mapping with diagnostic information.

    Returns diagnostic information about the control mapping including
    equilibrium shifts and mathematical consistency checks.

    Args:
        B: Basis matrix (r x rankS)
        K_red: Reduced relaxation matrix (rankS x rankS)
        kf_base, kr_base: Base rate constants (r,)
        u_red: Reduced control input (rankS,)

    Returns:
        Dictionary with diagnostic information
    """
    # Apply the mapping
    kf_new, kr_new = apply_llrq_drive_to_rates(kf_base, kr_base, B, K_red, u_red)

    # Compute shifts
    delta_computed = compute_equilibrium_shift(kf_base, kr_base, kf_new, kr_new)

    # Expected shift from LLRQ theory
    Delta = np.linalg.lstsq(K_red, u_red, rcond=None)[0]
    delta_expected = B @ Delta

    # Check consistency
    max_error = np.max(np.abs(delta_computed - delta_expected))

    # Base equilibrium constants
    Keq_base = kf_base / kr_base
    Keq_new = kf_new / kr_new

    return {
        "u_red": u_red,
        "Delta": Delta,
        "delta_expected": delta_expected,
        "delta_computed": delta_computed,
        "max_error": max_error,
        "Keq_base": Keq_base,
        "Keq_new": Keq_new,
        "Keq_ratio": Keq_new / Keq_base,
        "kf_base": kf_base,
        "kr_base": kr_base,
        "kf_new": kf_new,
        "kr_new": kr_new,
        "consistent": max_error < 1e-12,
    }


def symmetric_rate_scaling(kf_base: np.ndarray, kr_base: np.ndarray, speed_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply symmetric scaling to both forward and reverse rates.

    This changes the kinetic speed without affecting equilibrium constants.
    Can be combined with asymmetric shifts for full control.

    Args:
        kf_base, kr_base: Base rate constants
        speed_factor: Multiplicative factor (>1 speeds up, <1 slows down)

    Returns:
        (kf_scaled, kr_scaled): Scaled rate constants
    """
    kf_scaled = kf_base * speed_factor
    kr_scaled = kr_base * speed_factor

    return kf_scaled, kr_scaled


def combined_llrq_and_speed_control(
    kf_base: np.ndarray,
    kr_base: np.ndarray,
    B: np.ndarray,
    K_red: np.ndarray,
    u_red: np.ndarray,
    speed_factors: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply both LLRQ control and speed scaling.

    This combines:
    1. Asymmetric shifts for equilibrium control (LLRQ)
    2. Symmetric scaling for kinetic speed control

    Args:
        kf_base, kr_base: Base rate constants
        B, K_red: LLRQ matrices
        u_red: Reduced control input
        speed_factors: Optional speed scaling per reaction (default: no scaling)

    Returns:
        (kf_final, kr_final): Final modified rate constants
    """
    # Step 1: Apply LLRQ control (asymmetric)
    kf_llrq, kr_llrq = apply_llrq_drive_to_rates(kf_base, kr_base, B, K_red, u_red)

    # Step 2: Apply speed scaling (symmetric) if provided
    if speed_factors is not None:
        kf_final = kf_llrq * speed_factors
        kr_final = kr_llrq * speed_factors
    else:
        kf_final = kf_llrq
        kr_final = kr_llrq

    return kf_final, kr_final
