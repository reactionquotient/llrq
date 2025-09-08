"""Utilities for computing species concentrations from reaction quotients.

This module provides shared functions for reconstructing species concentrations
from reaction quotients while enforcing conservation laws.
"""

import warnings
from typing import Optional

import numpy as np
from numpy.linalg import solve
from scipy.linalg import lstsq, null_space
from scipy.optimize import fsolve


def _reconstruct_concentrations_fsolve(
    Q_t: np.ndarray, c0: np.ndarray, network, B: np.ndarray, lnKeq_consistent: np.ndarray
) -> np.ndarray:
    """Reconstruct concentrations using fsolve-based algorithm.

    This function solves the nonlinear system:
    1. Conservation laws: C @ c = cons0
    2. Reduced quotient constraints: y = B^T @ (S^T @ ln(c) - lnKeq)
    """
    # Get conservation matrix and conserved quantities
    C = network.find_conservation_laws()  # (n_cons, n_species)
    cons0 = network.compute_conserved_quantities(c0)  # (n_cons,)

    S = network.S  # (n_species, n_reactions)
    n_species = network.n_species
    n_times = len(Q_t) if Q_t.ndim > 1 else 1

    # Ensure Q_t is 2D
    if Q_t.ndim == 1:
        Q_t = Q_t[None, :]

    c_t = np.zeros((n_times, n_species))
    c_guess = np.maximum(c0, 1e-9)

    for i in range(n_times):
        # Target reduced state: y = B^T @ (ln(Q) - lnKeq)
        y_target = B.T @ (np.log(Q_t[i]) - lnKeq_consistent)

        def residual(u):
            """Residual function for nonlinear solve.

            Variables: u = ln(c) (log concentrations)
            Constraints:
            1. Conservation: C @ exp(u) = cons0
            2. Quotient: y_target = B^T @ (S^T @ u - lnKeq)
            """
            c = np.exp(u)

            # Conservation residuals
            r1 = C @ c - cons0

            # Quotient residuals
            r2 = y_target - (B.T @ (S.T @ u - lnKeq_consistent))

            return np.concatenate([r1, r2])

        try:
            # Initial guess in log space
            u0 = np.log(np.maximum(c_guess, 1e-12))

            # Solve nonlinear system
            u_sol = fsolve(residual, u0, xtol=1e-9, maxfev=2000)

            # Convert back to concentrations
            c_sol = np.maximum(np.exp(u_sol), 1e-12)
            c_t[i] = c_sol
            c_guess = c_sol  # Use as next initial guess

        except Exception as e:
            warnings.warn(f"Concentration solve failed at time point {i}: {e}; using previous guess")
            c_t[i] = c_guess

    # Return scalar case as 1D array
    if n_times == 1:
        return c_t[0]

    return c_t


def _reconstruct_concentrations_convex(
    Q_t: np.ndarray,
    c0: np.ndarray,
    network,
    B: np.ndarray,
    lnKeq_consistent: np.ndarray,
    tol: float = 1e-10,
    max_newton: int = 50,
) -> np.ndarray:
    C = network.find_conservation_laws()
    S = network.S
    n_species = network.n_species

    # Basis for conservation subspace (ker S^T)
    L = null_space(S.T)  # (n_species, m)
    m = L.shape[1]

    # Ensure Q_t is (T, R)
    Q = Q_t[None, :] if Q_t.ndim == 1 else Q_t
    n_times = Q.shape[0]
    c_t = np.zeros((n_times, n_species))

    # Totals in the SAME basis as L
    cons0_L = L.T @ c0  # <-- crucial: use L, not C

    # Handle projection with non-orthonormal B
    BTB = B.T @ B
    B_orthonormal = np.allclose(BTB, np.eye(BTB.shape[0]), atol=1e-8)

    for i in range(n_times):
        x_meas = np.log(np.maximum(Q[i], 1e-300)) - lnKeq_consistent
        z = (B.T @ x_meas) if B_orthonormal else solve(BTB, B.T @ x_meas)
        x_star = B @ z

        # Solve S^T u0 = x_star (consistent after projection)
        u0 = lstsq(S.T, x_star)[0]
        c_base = np.exp(np.clip(u0, -700, 700))

        if m == 0:
            c_t[i] = np.maximum(c_base, 1e-12)
            continue

        # Newton on alpha for: minimize sum_i c_base_i exp((L alpha)_i) - cons0_L^T alpha
        alpha = np.zeros(m)
        for _ in range(max_newton):
            LA = L @ alpha
            cA = c_base * np.exp(np.clip(LA, -700, 700))
            g = L.T @ cA - cons0_L  # gradient
            if np.linalg.norm(g, np.inf) < tol:
                break
            H = L.T @ (cA[:, None] * L)  # SPD Hessian
            # Cholesky solve (falls back to tiny damping if needed)
            try:
                delta = solve(H, g)
            except np.linalg.LinAlgError:
                delta = solve(H + 1e-12 * np.eye(m), g)

            # Backtracking to ensure residual decreases
            t = 1.0
            base_norm = np.linalg.norm(g, np.inf)
            while t > 1e-8:
                LA_t = L @ (alpha - t * delta)
                cA_t = c_base * np.exp(np.clip(LA_t, -700, 700))
                if np.linalg.norm(L.T @ cA_t - cons0_L, np.inf) < base_norm:
                    break
                t *= 0.5
            alpha -= t * delta

        c_t[i] = np.maximum(c_base * np.exp(L @ alpha), 1e-12)

    return c_t[0] if n_times == 1 else c_t


def compute_concentrations_from_quotients(
    Q_t: np.ndarray,
    c0: np.ndarray,
    network,
    B: np.ndarray,
    lnKeq_consistent: np.ndarray,
    enforce_conservation: bool = True,
    algorithm: str = "fsolve",
) -> Optional[np.ndarray]:
    """Reconstruct concentrations from reaction quotients using conservation laws.

    Args:
        Q_t: Reaction quotients at multiple time points (n_times, n_reactions)
        c0: Initial concentrations (n_species,)
        network: ReactionNetwork object with S matrix and conservation methods
        B: LLRQ basis matrix (n_reactions, rankS)
        lnKeq_consistent: Consistent equilibrium constants (n_reactions,)
        enforce_conservation: Whether to enforce conservation laws
        algorithm: Algorithm to use ("convex" or "fsolve")

    Returns:
        Concentrations array (n_times, n_species) or None if not enforcing conservation
    """
    if not enforce_conservation:
        return None

    # Get conservation matrix and conserved quantities
    C = network.find_conservation_laws()  # (n_cons, n_species)

    if C.shape[0] == 0:
        warnings.warn("No conservation laws found, cannot compute concentrations")
        return None

    if algorithm == "convex":
        return _reconstruct_concentrations_convex(Q_t, c0, network, B, lnKeq_consistent)
    elif algorithm == "fsolve":
        return _reconstruct_concentrations_fsolve(Q_t, c0, network, B, lnKeq_consistent)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'convex' or 'fsolve'")


def apply_state_disturbance_to_concentrations(
    c_current: np.ndarray,
    d_reduced: np.ndarray,
    network,
    B: np.ndarray,
    lnKeq_consistent: np.ndarray,
    algorithm: str = "scipy",
) -> np.ndarray:
    """Apply reduced state disturbance and compute resulting concentrations.

    This applies a disturbance d_reduced to the current reduced state y,
    then reconstructs concentrations that satisfy conservation laws.

    Args:
        c_current: Current concentrations (n_species,)
        d_reduced: Disturbance in reduced coordinates (rankS,)
        network: ReactionNetwork object
        B: LLRQ basis matrix (n_reactions, rankS)
        lnKeq_consistent: Consistent equilibrium constants (n_reactions,)

    Returns:
        New concentrations after disturbance (n_species,)
    """
    # Get current reaction quotients and reduced state
    Q_current = network.compute_reaction_quotients(c_current)
    y_current = B.T @ (np.log(Q_current) - lnKeq_consistent)

    # Apply disturbance
    y_new = y_current + d_reduced

    # Convert back to reaction quotients
    ln_Q_new = lnKeq_consistent + B @ y_new
    Q_new = np.exp(ln_Q_new)

    # Reconstruct concentrations using conservation laws
    c_new = compute_concentrations_from_quotients(
        Q_new, c_current, network, B, lnKeq_consistent, enforce_conservation=True, algorithm=algorithm
    )

    if c_new is None:
        warnings.warn("Failed to apply state disturbance - conservation reconstruction failed")
        return c_current

    return c_new
