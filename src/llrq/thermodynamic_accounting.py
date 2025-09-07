"""Thermodynamic accounting for entropy production in reaction networks.

This module provides tools for computing entropy production rates and energy balance
diagnostics in log-linear reaction quotient dynamics and mass action kinetics.

Based on the framework described in Diamond (2025) "Log-Linear Reaction Quotient Dynamics".
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from .reaction_network import ReactionNetwork


@dataclass
class AccountingResult:
    """Result of entropy production calculation."""

    sigma_time: np.ndarray  # instantaneous entropy-prod rate (shape T,)
    sigma_total: float  # integral over the trajectory (scalar)
    notes: str = ""  # optional comment (e.g., warning about quasi-ss)


@dataclass
class DualAccountingResult:
    """Result of dual entropy accounting (from x and u) plus energy balance."""

    from_x: AccountingResult
    from_u: AccountingResult
    balance: dict  # keys: V_dot, P_relax, P_ctrl, residual, and their integrals


class ThermodynamicAccountant:
    """Thermodynamic accounting for reaction network dynamics.

    Computes entropy production rates, energy balance diagnostics, and related
    thermodynamic quantities from trajectory data.
    """

    def __init__(self, network: ReactionNetwork, onsager_conductance: Optional[np.ndarray] = None):
        """Initialize thermodynamic accountant.

        Args:
            network: Reaction network
            onsager_conductance: Pre-computed Onsager conductance matrix L.
                If None, will be computed when needed from trajectory data.
        """
        self.network = network
        self.L = onsager_conductance
        self.n_reactions = network.n_reactions

    def _quad_time_series(self, t: np.ndarray, Y: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute q(t) = diag(Y(t) @ M @ Y(t)^T) and its time integral via trapezoid.

        Args:
            t: Time points (T,)
            Y: Trajectories (T, m)
            M: Symmetric matrix (m, m)

        Returns:
            Tuple of (q_t (T,), integral float)
        """
        q_t = np.einsum("ti,ij,tj->t", Y, M, Y, optimize=True)  # T-long vector
        integral = np.trapz(q_t, t)
        return q_t, integral

    def _sym_psd(self, M: np.ndarray, eps: Optional[float] = 0.0) -> np.ndarray:
        """Symmetrize and (optionally) clip small negative eigvals to eps."""
        Ms = 0.5 * (M + M.T)
        if eps is None or eps <= 0:
            return Ms
        w, V = np.linalg.eigh(Ms)
        w = np.maximum(w, eps)
        return (V * w) @ V.T

    def entropy_from_x(
        self, t: np.ndarray, x: np.ndarray, L: Optional[np.ndarray] = None, scale: float = 1.0, psd_clip: float = 0.0
    ) -> AccountingResult:
        """Entropy production Σ = ∫ sigma(t) dt with sigma(t) = scale * x(t)^T L x(t).

        Args:
            t: Time points (T,) - nonuniform OK
            x: Reaction-force coordinates (T, m) - your paper's x = ln(Q/K_eq)
            L: Onsager conductance (m, m) - if None, uses self.L
            scale: Multiply by R (or k_B) to get physical units if desired
            psd_clip: Clip eigenvalues of L to this value to ensure PSD

        Returns:
            AccountingResult with entropy production rate and total
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available. Provide L or set during initialization.")
            L = self.L

        Ls = self._sym_psd(np.asarray(L), eps=psd_clip)
        sigma_t, Sigma = self._quad_time_series(t, np.asarray(x), Ls)
        return AccountingResult(sigma_time=scale * sigma_t, sigma_total=scale * Sigma)

    def entropy_from_u(
        self,
        t: np.ndarray,
        u: np.ndarray,
        K: np.ndarray,
        L: Optional[np.ndarray] = None,
        scale: float = 1.0,
        psd_clip: float = 0.0,
    ) -> AccountingResult:
        """Quasi-steady approximation: x ≈ K^{-1} u ⇒ sigma(t) ≈ scale * u^T M u.

        With M = K^{-T} L K^{-1}. Good when u varies slowly relative to K's time scales.

        Args:
            t: Time points (T,)
            u: External drive trajectories (T, m)
            K: Relaxation matrix from dot{x} = -K x + u (m, m)
            L: Onsager conductance (m, m) - if None, uses self.L
            scale: Physical scale factor
            psd_clip: Clip eigenvalues to ensure PSD

        Returns:
            AccountingResult with quasi-steady entropy estimate
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available. Provide L or set during initialization.")
            L = self.L

        K = np.asarray(K)
        L = np.asarray(L)

        # Build M = K^{-T} L K^{-1} using solves (no explicit inverses)
        Z = np.linalg.solve(K.T, L)  # Z = K^{-T} L
        M = np.linalg.solve(K, Z.T).T  # M = Z K^{-1}
        M = self._sym_psd(M, eps=psd_clip)

        sigma_t, Sigma = self._quad_time_series(t, np.asarray(u), M)
        note = "quasi-steady: x ≈ K^{-1} u used"
        return AccountingResult(sigma_time=scale * sigma_t, sigma_total=scale * Sigma, notes=note)

    def entropy_from_xu(
        self,
        t: np.ndarray,
        x: np.ndarray,
        u: np.ndarray,
        K: np.ndarray,
        L: Optional[np.ndarray] = None,
        scale: float = 1.0,
        psd_clip: float = 0.0,
    ) -> DualAccountingResult:
        """Returns entropy from x(t), the quasi-steady estimate from u(t), and a
        model-space power balance check based on dot{x} = -K x + u:

            d/dt (1/2 ||x||^2) = - x^T K x + x^T u

        The balance is not "entropy" but is a useful diagnostic for sampling/noise.

        Args:
            t: Time points (T,)
            x: Reaction force trajectories (T, m)
            u: External drive trajectories (T, m)
            K: Relaxation matrix (m, m)
            L: Onsager conductance (m, m) - if None, uses self.L
            scale: Physical scale factor
            psd_clip: Clip eigenvalues to ensure PSD

        Returns:
            DualAccountingResult with entropy from both methods plus energy balance
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available. Provide L or set during initialization.")
            L = self.L

        x = np.asarray(x)
        u = np.asarray(u)
        K = np.asarray(K)
        L = np.asarray(L)

        # 1) entropy from x
        res_x = self.entropy_from_x(t, x, L, scale=scale, psd_clip=psd_clip)
        # 2) entropy from u (quasi-steady)
        res_u = self.entropy_from_u(t, u, K, L, scale=scale, psd_clip=psd_clip)

        # 3) power balance diagnostics in model coordinates
        Ksym = self._sym_psd(K, eps=0.0)  # use symmetric part in the quadratic
        P_relax_t = np.einsum("ti,ij,tj->t", x, Ksym, x, optimize=True)  # >= 0 if Ksym ≽ 0
        P_ctrl_t = np.einsum("ti,ti->t", x, u)  # injection by control
        V_t = 0.5 * np.einsum("ti,ti->t", x, x)
        # numerical derivative of V by finite differences matching t
        dVdt_t = np.gradient(V_t, t, edge_order=2)

        residual_t = dVdt_t - (-P_relax_t + P_ctrl_t)  # should be ~0 in clean data
        integ = lambda y: float(np.trapz(y, t))

        balance = dict(
            V_dot_time=dVdt_t,
            V_dot_total=integ(dVdt_t),
            P_relax_time=P_relax_t,
            P_relax_total=integ(P_relax_t),
            P_ctrl_time=P_ctrl_t,
            P_ctrl_total=integ(P_ctrl_t),
            residual_time=residual_t,
            residual_total=integ(residual_t),
            comment="Residual reflects discretization/noise; exact identity holds for continuous-time model.",
        )

        return DualAccountingResult(from_x=res_x, from_u=res_u, balance=balance)

    def from_solution(
        self,
        solution: Dict[str, Any],
        forward_rates: Optional[np.ndarray] = None,
        backward_rates: Optional[np.ndarray] = None,
        concentrations: Optional[np.ndarray] = None,
        scale: float = 1.0,
        compute_onsager: bool = True,
        mode: str = "auto",
    ) -> Union[AccountingResult, DualAccountingResult]:
        """Compute entropy production from LLRQSolver solution results.

        Args:
            solution: Dictionary returned by LLRQSolver.solve()
            forward_rates: Forward rate constants for Onsager conductance computation
            backward_rates: Backward rate constants for Onsager conductance computation
            concentrations: Reference concentrations for Onsager computation (if None, uses initial)
            scale: Physical scale factor (e.g., kB*T)
            compute_onsager: Whether to compute Onsager conductance if not provided
            mode: Onsager computation mode ('auto', 'equilibrium', 'local')

        Returns:
            AccountingResult if only reaction forces available, DualAccountingResult if drives also available
        """
        if "time" not in solution:
            raise ValueError("Solution must contain 'time' field")

        t = solution["time"]

        # Extract reaction forces if available
        if "log_deviations" in solution:
            x = solution["log_deviations"]
        else:
            raise ValueError("Solution must contain 'log_deviations' for entropy computation")

        # Compute or use provided Onsager conductance
        L = self.L
        if L is None and compute_onsager:
            if forward_rates is None or backward_rates is None:
                raise ValueError("forward_rates and backward_rates required for Onsager conductance computation")

            # Use provided concentrations or initial concentrations from solution
            if concentrations is None:
                if "initial_concentrations" in solution:
                    concentrations = solution["initial_concentrations"]
                else:
                    raise ValueError("concentrations or solution['initial_concentrations'] required")

            onsager_result = self.network.compute_onsager_conductance(concentrations, forward_rates, backward_rates, mode=mode)
            L = onsager_result["L"]

        if L is None:
            raise ValueError("No Onsager conductance available. Provide L or enable compute_onsager.")

        # Check if external drives are available for dual accounting
        if hasattr(solution, "external_drives") or "external_drives" in solution:
            # Extract external drives and relaxation matrix for dual accounting
            u = solution.get("external_drives")
            K = solution.get("relaxation_matrix")

            if u is not None and K is not None:
                return self.entropy_from_xu(t, x, u, K, L, scale=scale)

        # Fall back to entropy from reaction forces only
        return self.entropy_from_x(t, x, L, scale=scale)

    def compute_onsager_conductance(
        self, concentrations: np.ndarray, forward_rates: np.ndarray, backward_rates: np.ndarray, mode: str = "auto"
    ) -> np.ndarray:
        """Compute and cache Onsager conductance matrix.

        Args:
            concentrations: Species concentrations
            forward_rates: Forward rate constants
            backward_rates: Backward rate constants
            mode: Computation mode ('auto', 'equilibrium', 'local')

        Returns:
            Onsager conductance matrix L
        """
        result = self.network.compute_onsager_conductance(concentrations, forward_rates, backward_rates, mode=mode)
        self.L = result["L"]
        return self.L
