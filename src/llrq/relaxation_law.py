from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from .utils.concentration_utils import compute_concentrations_from_quotients

if TYPE_CHECKING:
    from .llrq_dynamics import LLRQDynamics
    from .reaction_network import ReactionNetwork


class RelaxationLaw:
    """Interface for reaction-force relaxation laws."""

    is_linear: bool = True

    def attach(self, dynamics: "LLRQDynamics") -> None:
        """Attach to a dynamics instance (allows subclasses to cache info)."""
        self._dynamics = dynamics

    def evaluate(self, dynamics: "LLRQDynamics", t: float, x: np.ndarray) -> np.ndarray:
        """Compute time derivative dx/dt given current state."""
        raise NotImplementedError


class LinearRelaxationLaw(RelaxationLaw):
    """Default linear relaxation law: dx/dt = -K x + u(t)."""

    def evaluate(self, dynamics: "LLRQDynamics", t: float, x: np.ndarray) -> np.ndarray:
        u = dynamics.external_drive(t)
        return -dynamics.K @ x + u


class AcceleratedRelaxationLaw(RelaxationLaw):
    """Modal accelerated relaxation with selectable variant.

    Variants:
    - 'exp':    g_i(y_i) = -k_i (1 + β_i e^{|y_i|}) y_i
    - 'sinh':   g_i(y_i) = -a_i y_i - b_i sinh(y_i)

    Both reduce to linear near zero by matching the local slope to λ_i (eigenvalue
    of the symmetric part). The 'sinh' variant often better matches mass-action tails.
    """

    is_linear = False

    def __init__(
        self,
        basis: np.ndarray,
        modal_matrix: np.ndarray,
        k: np.ndarray,
        beta: np.ndarray,
        skew: np.ndarray,
        projector: np.ndarray,
        ln_keq_consistent: np.ndarray,
        exp_clip: float = 80.0,
        variant: str = "exp",
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> None:
        self._B = np.array(basis, dtype=float)
        self._V = np.array(modal_matrix, dtype=float)
        self._k = np.array(k, dtype=float)
        self._beta = np.array(beta, dtype=float)
        self._skew = np.array(skew, dtype=float)
        self._P = np.array(projector, dtype=float)
        self._lnKeq_consistent = np.array(ln_keq_consistent, dtype=float)
        self._rank = self._B.shape[1]
        self._exp_clip = float(exp_clip)
        self._variant = variant
        self._a = None if a is None else np.array(a, dtype=float)
        self._b = None if b is None else np.array(b, dtype=float)

        if self._rank != len(self._k):
            raise ValueError("Modal parameter dimensions do not match basis rank")
        if self._V.shape != (self._rank, self._rank):
            raise ValueError("Modal matrix must be square with size equal to basis rank")
        if self._skew.shape != (self._rank, self._rank):
            raise ValueError("Skew matrix dimensions must match basis rank")

    def evaluate(self, dynamics: "LLRQDynamics", t: float, x: np.ndarray) -> np.ndarray:
        u = dynamics.external_drive(t)
        if self._rank == 0:
            return u

        y = self._B.T @ x
        y_modal = self._V.T @ y

        if self._variant == "sinh":
            a = self._a if self._a is not None else np.zeros_like(self._k)
            b = self._b if self._b is not None else np.zeros_like(self._k)
            damping_modal = a * y_modal + b * np.sinh(y_modal)
            damping = self._V @ damping_modal
        else:  # exp variant
            abs_modal = np.clip(np.abs(y_modal), a_min=None, a_max=self._exp_clip)
            mu = self._k * (1.0 + self._beta * np.exp(abs_modal))
            damping = self._V @ (mu * y_modal)

        dy = -self._skew @ y - damping + self._B.T @ u
        dx = self._B @ dy + (u - self._P @ u)

        return dx

    @classmethod
    def from_dynamics(
        cls,
        dynamics: "LLRQDynamics",
        xi_star: float = 1.0,
        delta: float = 1e-3,
        exp_clip: float = 80.0,
        variant: str = "exp",
    ) -> "AcceleratedRelaxationLaw":
        if xi_star <= 0:
            raise ValueError("xi_star must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")

        mass_info = dynamics.get_mass_action_info()
        if mass_info is None:
            raise ValueError(
                "Accelerated relaxation requires mass action metadata. "
                "Use LLRQDynamics.from_mass_action() or set_mass_action_parameters() first."
            )

        dynamics_data = mass_info.get("dynamics_data")
        if dynamics_data is None:
            raise ValueError("Mass action metadata missing dynamics data; cannot build accelerated law")

        network = dynamics.network

        basis = dynamics_data.get("basis")
        if basis is None:
            basis = cls._compute_basis(network)
        K_reduced = dynamics_data.get("K_reduced")
        if K_reduced is None:
            K_reduced = basis.T @ dynamics.K @ basis

        skew = 0.5 * (K_reduced - K_reduced.T)
        symmetric = 0.5 * (K_reduced + K_reduced.T)

        eigenvals, modal_matrix = np.linalg.eigh(symmetric)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        modal_matrix = modal_matrix[:, idx]
        eigenvals = np.maximum(eigenvals, 1e-12)

        projector = basis @ basis.T
        lnKeq = np.log(np.maximum(dynamics.Keq, 1e-300))
        lnKeq_consistent = projector @ lnKeq

        equilibrium_c = mass_info.get("equilibrium_point")
        if equilibrium_c is None:
            raise ValueError("Mass action metadata missing equilibrium concentrations")

        forward_rates = np.array(mass_info.get("forward_rates"), dtype=float)
        backward_rates = np.array(mass_info.get("backward_rates"), dtype=float)

        k = np.zeros_like(eigenvals)
        beta = np.zeros_like(eigenvals)
        a = np.zeros_like(eigenvals)
        b = np.zeros_like(eigenvals)

        for i, lam in enumerate(eigenvals):
            local_slope = lam
            slope_far = cls._estimate_modal_slope(
                network,
                forward_rates,
                backward_rates,
                lnKeq_consistent,
                basis,
                modal_matrix,
                i,
                xi_star,
                delta,
                equilibrium_c,
            )

            if variant == "sinh":
                # Fit a_i, b_i from s0 = a+b, s* = a + b cosh(xi*)
                if slope_far is None or not np.isfinite(slope_far) or slope_far <= local_slope * (1.0 + 1e-6):
                    a[i] = local_slope
                    b[i] = 0.0
                    continue
                denom = math.cosh(xi_star) - 1.0
                if abs(denom) < 1e-12:
                    a[i] = local_slope
                    b[i] = 0.0
                    continue
                b_i = (slope_far - local_slope) / denom
                a_i = local_slope - b_i
                if not np.isfinite(b_i) or b_i < 0:
                    b_i = 0.0
                if not np.isfinite(a_i) or a_i < 0:
                    a_i = max(local_slope - b_i, 0.0)
                a[i] = a_i
                b[i] = b_i
            else:  # exp variant
                if slope_far is None or not np.isfinite(slope_far) or slope_far <= local_slope * (1.0 + 1e-6):
                    beta[i] = 0.0
                    k[i] = local_slope
                    continue
                denom = local_slope * math.exp(xi_star) * (1.0 + xi_star) - slope_far
                if abs(denom) < 1e-12:
                    beta[i] = 0.0
                    k[i] = local_slope
                    continue
                beta_i = (slope_far - local_slope) / denom
                if not np.isfinite(beta_i) or beta_i <= 0:
                    beta[i] = 0.0
                    k[i] = local_slope
                    continue
                beta[i] = beta_i
                k[i] = local_slope / (1.0 + beta_i)

        if variant == "sinh":
            return cls(
                basis,
                modal_matrix,
                k=np.zeros_like(k),
                beta=np.zeros_like(beta),
                skew=skew,
                projector=projector,
                ln_keq_consistent=lnKeq_consistent,
                exp_clip=exp_clip,
                variant="sinh",
                a=a,
                b=b,
            )
        else:
            return cls(basis, modal_matrix, k, beta, skew, projector, lnKeq_consistent, exp_clip=exp_clip, variant="exp")

    @staticmethod
    def _compute_basis(network: "ReactionNetwork") -> np.ndarray:
        S = network.S
        if hasattr(S, "toarray"):
            S_mat = S.toarray().astype(float)
        else:
            S_mat = np.array(S, dtype=float)

        U, s, _ = np.linalg.svd(S_mat.T, full_matrices=False)
        tol = max(S_mat.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        rank = int(np.sum(s > tol))
        if rank == 0:
            return np.zeros((S_mat.shape[1], 0))
        return U[:, :rank]

    @classmethod
    def _estimate_modal_slope(
        cls,
        network: "ReactionNetwork",
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        lnKeq_consistent: np.ndarray,
        basis: np.ndarray,
        modal_matrix: np.ndarray,
        mode_index: int,
        xi_star: float,
        delta: float,
        equilibrium_c: np.ndarray,
    ) -> Optional[float]:
        try:
            g_plus = cls._modal_drift(
                network,
                forward_rates,
                backward_rates,
                lnKeq_consistent,
                basis,
                modal_matrix,
                mode_index,
                xi_star + delta,
                equilibrium_c,
            )
            g_minus = cls._modal_drift(
                network,
                forward_rates,
                backward_rates,
                lnKeq_consistent,
                basis,
                modal_matrix,
                mode_index,
                xi_star - delta,
                equilibrium_c,
            )
        except ValueError:
            return None

        derivative = (g_plus - g_minus) / (2.0 * delta)
        return -derivative

    @classmethod
    def _modal_drift(
        cls,
        network: "ReactionNetwork",
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        lnKeq_consistent: np.ndarray,
        basis: np.ndarray,
        modal_matrix: np.ndarray,
        mode_index: int,
        xi: float,
        equilibrium_c: np.ndarray,
    ) -> float:
        rank = basis.shape[1]
        if rank == 0:
            return 0.0

        modal_vec = modal_matrix[:, mode_index]
        y = modal_vec * xi
        x = basis @ y
        lnQ = lnKeq_consistent + x
        Q = np.exp(lnQ)

        concentrations = cls._recover_concentrations(network, Q, equilibrium_c, basis, lnKeq_consistent)
        if concentrations is None:
            raise ValueError("Failed to recover concentrations for modal perturbation")

        forward_flux, reverse_flux, net_flux = network._compute_flux_parts(concentrations, forward_rates, backward_rates)
        flux_response = network.compute_flux_response_matrix(concentrations)
        dxdt = flux_response @ net_flux
        dydt = basis.T @ dxdt

        modal_rate = float(modal_vec.T @ dydt)
        return modal_rate

    @staticmethod
    def _recover_concentrations(
        network: "ReactionNetwork",
        Q: np.ndarray,
        c_ref: np.ndarray,
        basis: np.ndarray,
        lnKeq_consistent: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            conc = compute_concentrations_from_quotients(
                Q,
                c_ref,
                network,
                basis,
                lnKeq_consistent,
                enforce_conservation=True,
            )
            if conc is not None:
                conc = np.array(conc, dtype=float)
                if conc.ndim > 1:
                    conc = conc[0]
                if np.all(np.isfinite(conc)) and np.all(conc > 0):
                    return conc
        except Exception:
            pass

        lnQ = np.log(np.maximum(Q, 1e-300))
        S = network.S
        S_T = S.T.toarray().astype(float) if hasattr(S, "toarray") else np.array(S.T, dtype=float)
        try:
            u, *_ = np.linalg.lstsq(S_T, lnQ, rcond=None)
        except np.linalg.LinAlgError:
            return None
        conc = np.exp(np.clip(u, -50, 50))
        if not np.all(np.isfinite(conc)) or not np.all(conc > 0):
            return None
        return conc
