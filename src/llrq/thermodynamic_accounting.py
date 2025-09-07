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

    # =============================================================================
    # Frequency-Domain Entropy Calculations
    # =============================================================================

    @staticmethod
    def _freqs(N: int, dt: float) -> np.ndarray:
        """Angular frequencies matching np.fft.fft bins (two-sided)."""
        return 2 * np.pi * np.fft.fftfreq(N, d=dt)

    def entropy_from_x_freq(
        self, x_t: np.ndarray, dt: float, L: Optional[np.ndarray] = None, scale: float = 1.0
    ) -> tuple[float, np.ndarray]:
        """Approximate Σ_x = ∫ x(t)^T L x(t) dt using FFT (two-sided sum).

        Based on Parseval's theorem, the time-domain integral equals the frequency-domain
        spectral sum: ∫ x^T L x dt = (1/2π) ∫ X(ω)^H L X(ω) dω

        Args:
            x_t: Time series trajectories (N, m) where N is time points, m is reactions
            dt: Time step size
            L: Onsager conductance matrix (m, m). If None, uses self.L
            scale: Scale factor (e.g., k_B*T for physical units)

        Returns:
            Tuple of (Sigma_total, per_bin_contrib) where per_bin_contrib has shape (N,)
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available. Provide L or set during initialization.")
            L = self.L

        N, m = x_t.shape
        X = np.fft.fft(x_t, axis=0)  # (N, m) - FFT along time axis

        # Symmetrize L to ensure positive semidefinite
        Ls = 0.5 * (L + L.T)

        # Per-bin quadratic: X_k^H L X_k for each frequency bin k
        quad = np.einsum("ki,ij,kj->k", np.conj(X), Ls, X, optimize=True)

        # Total entropy using correct DFT Parseval relation
        # For DFT: ∫ x^T L x dt ≈ (dt/N) Σ X[k]^H L X[k]
        Sigma = (quad.real).sum() * dt / N

        return float(scale * Sigma), scale * quad.real * dt / N

    def entropy_from_u_freq(
        self, u_t: np.ndarray, dt: float, K: np.ndarray, L: Optional[np.ndarray] = None, scale: float = 1.0
    ) -> tuple[float, np.ndarray]:
        """Approximate Σ_u = (1/2π) ∫ U(ω)^H G(ω)^H L G(ω) U(ω) dω via FFT sum.

        This computes entropy production from control inputs using the frequency-domain
        entropy kernel H_u(ω) = G(iω)^H L G(iω) where G(iω) = (K + iωI)^{-1}.

        Args:
            u_t: Control input time series (N, m)
            dt: Time step size
            K: Relaxation matrix (m, m) from ẋ = -Kx + u
            L: Onsager conductance matrix (m, m). If None, uses self.L
            scale: Scale factor

        Returns:
            Tuple of (Sigma_total, per_bin_contrib) where per_bin_contrib has shape (N,)
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available. Provide L or set during initialization.")
            L = self.L

        N, m = u_t.shape
        U = np.fft.fft(u_t, axis=0)
        w = self._freqs(N, dt)

        Sigma_bins = np.empty(N, dtype=float)
        Ls = 0.5 * (L + L.T)  # ensure symmetric/PSD

        for k, wk in enumerate(w):
            A = K + 1j * wk * np.eye(m)
            G = np.linalg.solve(A, np.eye(m))  # (K + iωI)^{-1}
            Hu = np.conj(G.T) @ Ls @ G  # G^H L G
            Sigma_bins[k] = np.real(np.conj(U[k]).T @ Hu @ U[k])

        Sigma = Sigma_bins.sum() * dt / N
        return float(scale * Sigma), scale * Sigma_bins * dt / N

    def map_xref_to_u(self, Xref: np.ndarray, dt: float, K: np.ndarray) -> np.ndarray:
        """Given desired state spectrum Xref[k] (FFT grid), return required control spectrum.

        For each frequency bin: U[k] = (K + i ω_k I) Xref[k]
        This inverts the transfer function to find the control needed for a target state spectrum.

        Args:
            Xref: Desired state spectrum on FFT bins (N, m) complex array
            dt: Time step size
            K: Relaxation matrix (m, m)

        Returns:
            Required control spectrum U of same shape as Xref
        """
        N, m = Xref.shape
        w = self._freqs(N, dt)
        U = np.empty_like(Xref)

        for k, wk in enumerate(w):
            # U[k] = (K + i ω_k I) Xref[k]
            U[k] = (K + 1j * wk * np.eye(m)) @ Xref[k]

        return U

    def compute_entropy_spectrum(
        self, u_t: np.ndarray, dt: float, K: np.ndarray, L: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Compute detailed entropy spectrum analysis.

        Args:
            u_t: Control input time series (N, m)
            dt: Time step size
            K: Relaxation matrix (m, m)
            L: Onsager conductance matrix (m, m). If None, uses self.L

        Returns:
            Dictionary containing:
            - frequencies: Angular frequencies (N,)
            - entropy_spectrum: Per-frequency entropy contributions (N,)
            - entropy_kernel_trace: Trace of H_u(ω) at each frequency (N,)
            - control_spectrum_power: ||U(ω)||² at each frequency (N,)
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available.")
            L = self.L

        N, m = u_t.shape
        U = np.fft.fft(u_t, axis=0)
        frequencies = self._freqs(N, dt)

        entropy_spectrum = np.empty(N)
        entropy_kernel_trace = np.empty(N)
        control_spectrum_power = np.empty(N)

        for k, wk in enumerate(frequencies):
            # Transfer function
            G = np.linalg.inv(K + 1j * wk * np.eye(m))

            # Entropy kernel
            Hu = np.conj(G.T) @ L @ G

            # Spectral quantities
            entropy_spectrum[k] = np.real(np.conj(U[k]).T @ Hu @ U[k])
            entropy_kernel_trace[k] = np.real(np.trace(Hu))
            control_spectrum_power[k] = np.real(np.conj(U[k]).T @ U[k])

        return {
            "frequencies": frequencies,
            "entropy_spectrum": entropy_spectrum * dt / N,  # <-- fix: multiply, not divide
            "entropy_kernel_trace": entropy_kernel_trace,
            "control_spectrum_power": control_spectrum_power,  # optional: leave unnormalized
        }

    def validate_parseval_entropy(
        self, x_t: np.ndarray, dt: float, L: Optional[np.ndarray] = None, scale: float = 1.0
    ) -> Dict[str, float]:
        """Validate Parseval's theorem for entropy: time-domain vs frequency-domain should match.

        Args:
            x_t: State time series (N, m)
            dt: Time step size
            L: Onsager conductance matrix (m, m). If None, uses self.L
            scale: Scale factor

        Returns:
            Dictionary with 'time_domain', 'frequency_domain', and 'relative_error' keys
        """
        if L is None:
            if self.L is None:
                raise ValueError("No Onsager conductance matrix available.")
            L = self.L

        # Time-domain calculation
        t = np.arange(len(x_t)) * dt
        time_result = self.entropy_from_x(t, x_t, L, scale)
        time_domain_entropy = time_result.sigma_total

        # Frequency-domain calculation
        freq_domain_entropy, _ = self.entropy_from_x_freq(x_t, dt, L, scale)

        # Relative error
        if abs(time_domain_entropy) > 1e-12:
            relative_error = abs(freq_domain_entropy - time_domain_entropy) / abs(time_domain_entropy)
        else:
            relative_error = abs(freq_domain_entropy - time_domain_entropy)

        return {
            "time_domain": float(time_domain_entropy),
            "frequency_domain": float(freq_domain_entropy),
            "relative_error": float(relative_error),
        }
