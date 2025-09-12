"""Moiety dynamics for LLRQ systems with analytical solutions.

This module implements block-triangular decomposition of LLRQ dynamics
into reaction quotient (ratio) and moiety total (level) dynamics,
with analytical propagation and correct concentration reconstruction.

Key equations:
    ẋ = -K x + B_x u_x         (reaction quotient logs)
    ẏ = A_y y + g_y            (moiety totals)

where x = ln(Q/Keq) and y = L^T c with L^T S = 0.

Features:
- Analytical solutions via matrix exponential (no numerical integration)
- Correct reconstruction via Newton iteration on moiety coordinates
- Support for CSTR and moiety-respecting removal operators
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy.linalg import expm
from .llrq_dynamics import LLRQDynamics
from .reaction_network import ReactionNetwork


def analytical_lti_response(
    A: np.ndarray, B: Optional[np.ndarray], x0: np.ndarray, u: Optional[np.ndarray], t: np.ndarray
) -> np.ndarray:
    """Analytical response of ẋ = Ax + Bu for constant u.

    Uses the block matrix exponential trick to handle singular A:

        M = [[A, Bu],
             [0,  0]]

    Then [x(t); 1] = exp(Mt) @ [x0; 1], and x(t) is the top block.

    Args:
        A: System matrix (n x n)
        B: Input matrix (n x m), can be None
        x0: Initial condition (n,)
        u: Constant input (m,), can be None
        t: Time points

    Returns:
        State trajectory x(t) at each time point (len(t) x n)
    """
    n = A.shape[0]
    out = np.zeros((len(t), n))

    if u is None or B is None:
        # Homogeneous system: x(t) = exp(At) x0
        for i, ti in enumerate(t):
            out[i] = expm(A * ti) @ x0
    else:
        # Non-homogeneous system with constant input
        u = np.asarray(u)
        if u.ndim == 0:
            u = u.item()  # Convert scalar array to scalar
        Bu = B @ u if B.ndim == 2 and hasattr(u, "__len__") else B.flatten() * u

        # Augmented system matrix
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = Bu
        # Last row is zeros (keeps augmented state constant)

        # Augmented initial condition
        z0 = np.zeros(n + 1)
        z0[:n] = x0
        z0[n] = 1.0

        for i, ti in enumerate(t):
            exp_Mt = expm(M * ti)
            z_t = exp_Mt @ z0
            out[i] = z_t[:n]

    return out


@dataclass
class BlockTriangularSystem:
    """Block-triangular LTI system for LLRQ with moiety decomposition.

    Dynamics:
        ẋ = A_x x + B_x u_x
        ẏ = A_y y + g_y

    No coupling between x and y blocks.
    """

    A_x: np.ndarray  # Reaction quotient dynamics matrix
    B_x: Optional[np.ndarray]  # Reaction quotient input matrix
    A_y: np.ndarray  # Moiety dynamics matrix
    g_y: Optional[np.ndarray] = None  # Constant moiety input

    def simulate(
        self, t: np.ndarray, x0: np.ndarray, y0: np.ndarray, u_x: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical simulation of block-triangular system.

        Args:
            t: Time points
            x0: Initial reaction quotient logs
            y0: Initial moiety totals
            u_x: Constant control input for x-block

        Returns:
            (X, Y): State trajectories for x and y blocks
        """
        # x-block response
        X = analytical_lti_response(self.A_x, self.B_x, x0, u_x, t)

        # y-block response (homogeneous part)
        Y = analytical_lti_response(self.A_y, None, y0, None, t)

        # Add particular solution for constant g_y
        if self.g_y is not None and np.any(self.g_y != 0):
            I = np.eye(self.A_y.shape[0])
            Y_particular = analytical_lti_response(self.A_y, I, np.zeros_like(y0), self.g_y, t)
            Y = Y + Y_particular

        return X, Y


class MoietyDynamics(LLRQDynamics):
    """LLRQ dynamics with moiety decomposition and analytical solutions.

    Provides:
    - Block-triangular decomposition into x (quotients) and y (totals)
    - Analytical propagation via matrix exponential
    - Correct concentration reconstruction via Newton iteration
    - Support for CSTR and moiety-respecting removal
    """

    def __init__(
        self,
        network: ReactionNetwork,
        K: Union[float, np.ndarray],
        equilibrium_constants: Optional[np.ndarray] = None,
        external_drive: Optional[Callable[[float], np.ndarray]] = None,
    ):
        """Initialize moiety dynamics.

        Args:
            network: Reaction network with stoichiometry
            K: Relaxation rate matrix (positive semidefinite)
            equilibrium_constants: Equilibrium constants (defaults to ones)
            external_drive: External drive function u(t)
        """
        # Convert K to relaxation_matrix format
        if np.isscalar(K):
            relaxation_matrix = np.array(K) * np.eye(network.n_reactions)  # type: ignore
        else:
            relaxation_matrix = np.array(K)

        # Ensure equilibrium_constants is array or None
        if equilibrium_constants is not None:
            equilibrium_constants = np.array(equilibrium_constants)
            if equilibrium_constants.ndim == 0:
                equilibrium_constants = np.array([equilibrium_constants])

        super().__init__(network, equilibrium_constants, relaxation_matrix, external_drive)  # type: ignore

        # Compute moiety matrix L (rows span left nullspace of S)
        self.L = self._compute_moiety_matrix()
        self.n_moieties = self.L.shape[0]

        # Nullspace basis for reconstruction
        self.N = self.L.T  # columns span ker(S^T)

        # Set up log equilibrium constants
        self.lnKeq = np.log(self.Keq)

        # x-block matrices
        if np.isscalar(K):
            self.A_x = -np.array(K) * np.eye(self.n_reactions)  # type: ignore
        else:
            self.A_x = -np.array(K)
        self.B_x = np.eye(self.n_reactions)

        # y-block matrices (default: closed system)
        self.A_y = np.zeros((self.n_moieties, self.n_moieties))
        self.g_y = np.zeros(self.n_moieties) if self.n_moieties > 0 else None

    def _compute_moiety_matrix(self) -> np.ndarray:
        """Compute moiety matrix L such that L @ S = 0.

        Returns:
            Matrix L where rows are moiety conservation laws
        """
        # Use existing conservation law finder
        L = self.network.find_conservation_laws()
        if L.shape[0] == 0:
            # No conservation laws
            return np.zeros((0, self.network.n_species))
        return L

    def configure_cstr(self, dilution_rate: float, inlet_composition: Optional[np.ndarray] = None) -> "MoietyDynamics":
        """Configure for CSTR with uniform dilution (exactly decoupled).

        Dynamics: ẏ = -D*y + D*y_in where y_in = L @ c_in

        Args:
            dilution_rate: D = F_out/V
            inlet_composition: Species concentrations in inlet

        Returns:
            Self for chaining
        """
        D = dilution_rate
        self.A_y = -D * np.eye(self.n_moieties)

        if inlet_composition is not None:
            c_in = np.asarray(inlet_composition).reshape(self.network.n_species)
            y_in = self.L @ c_in
            self.g_y = D * y_in
        else:
            self.g_y = np.zeros(self.n_moieties)

        return self

    def configure_moiety_respecting_removal(
        self, removal_matrix: np.ndarray, inlet_totals: Optional[np.ndarray] = None
    ) -> "MoietyDynamics":
        """Configure moiety-respecting removal operator.

        For removal matrix R, requires L @ R = A_y @ L for some A_y.
        Then ẏ = -A_y*y + inlet_y

        Args:
            removal_matrix: Species removal operator R
            inlet_totals: Moiety totals in inlet stream

        Returns:
            Self for chaining

        Raises:
            ValueError: If removal is not moiety-respecting
        """
        R = np.asarray(removal_matrix)

        if self.n_moieties == 0:
            return self

        # Compute A_y = L @ R @ L^+
        L_pinv = np.linalg.pinv(self.L)
        A_y = (self.L @ R) @ L_pinv

        # Verify moiety-respecting property
        if not np.allclose(self.L @ R, A_y @ self.L, atol=1e-10):
            raise ValueError("Removal operator is not moiety-respecting")

        self.A_y = -A_y

        if inlet_totals is not None:
            self.g_y = np.asarray(inlet_totals).reshape(self.n_moieties)
        else:
            self.g_y = np.zeros(self.n_moieties)

        return self

    def get_block_system(self) -> BlockTriangularSystem:
        """Get block-triangular system for analytical simulation.

        Returns:
            BlockTriangularSystem with decoupled x and y dynamics
        """
        g_y = self.g_y if self.g_y is not None and self.g_y.size > 0 else None
        return BlockTriangularSystem(self.A_x, self.B_x, self.A_y, g_y)

    def reconstruct_concentrations(
        self, x: np.ndarray, y: np.ndarray, newton_tol: float = 1e-10, newton_max_iter: int = 50
    ) -> np.ndarray:
        """Reconstruct concentrations from quotients and moiety totals.

        Solves:
            S^T ln(c) = ln(Q) = lnKeq + x
            L^T c = y

        Algorithm:
        1. Find one solution u0 to S^T u = ln(Q)
        2. Parameterize general solution as u = u0 + N*alpha where N = L^T
        3. Use Newton's method to find alpha such that L^T exp(u) = y

        Args:
            x: Reaction quotient logs ln(Q/Keq)
            y: Moiety totals L^T c
            newton_tol: Convergence tolerance for Newton iteration
            newton_max_iter: Maximum Newton iterations

        Returns:
            Species concentrations

        Raises:
            RuntimeError: If Newton iteration doesn't converge
        """
        x = np.asarray(x).reshape(self.n_reactions)
        y = np.asarray(y).reshape(self.n_moieties)

        # Target quotient logs
        lnQ = self.lnKeq + x

        # Step 1: Find one solution to S^T u = lnQ
        S_T = self.network.S.T
        u0, _, _, _ = np.linalg.lstsq(S_T, lnQ, rcond=None)

        if self.n_moieties == 0:
            # No conservation laws to enforce
            return np.exp(u0)

        # Steps 2-3: Newton iteration on alpha
        N = self.N  # L^T, shape (n_species, n_moieties)
        alpha = np.zeros(self.n_moieties)

        for iteration in range(newton_max_iter):
            # Current u and concentrations
            u = u0 + N @ alpha
            c = np.exp(u)

            # Residual: L^T c - y
            residual = self.L @ c - y

            if np.linalg.norm(residual) < newton_tol:
                return c

            # Jacobian: J = d(L^T exp(u))/d(alpha) = L^T diag(c) N = N^T diag(c) N
            J = N.T @ (c[:, None] * N)

            # Newton step
            try:
                delta_alpha = np.linalg.solve(J, residual)
            except np.linalg.LinAlgError:
                # Singular Jacobian - use pseudoinverse
                delta_alpha = np.linalg.pinv(J) @ residual

            alpha -= delta_alpha

        raise RuntimeError(f"Newton iteration did not converge after {newton_max_iter} iterations")

    def simulate_analytical(
        self,
        t: np.ndarray,
        initial_concentrations: np.ndarray,
        u_x: Optional[np.ndarray] = None,
        return_concentrations: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Simulate dynamics analytically using matrix exponential.

        Args:
            t: Time points
            initial_concentrations: Initial species concentrations
            u_x: Constant control input for reaction quotients
            return_concentrations: Whether to reconstruct concentrations

        Returns:
            Dictionary with:
                - 'time': Time points
                - 'x': Reaction quotient log trajectories
                - 'y': Moiety total trajectories
                - 'Q': Reaction quotient trajectories
                - 'concentrations': Species concentrations (if requested)
        """
        c0 = np.asarray(initial_concentrations)

        # Initial conditions
        Q0 = self.network.compute_reaction_quotients(c0)
        x0 = np.log(Q0 / self.Keq)
        y0 = self.L @ c0 if self.n_moieties > 0 else np.array([])

        # Get block system and simulate
        sys = self.get_block_system()
        X, Y = sys.simulate(t, x0, y0, u_x)

        # Compute reaction quotients
        Q = self.Keq * np.exp(X)

        results = {"time": t, "x": X, "y": Y, "Q": Q}

        # Reconstruct concentrations if requested
        if return_concentrations and self.n_moieties > 0:
            n_times = len(t)
            C = np.zeros((n_times, self.network.n_species))

            for i in range(n_times):
                C[i] = self.reconstruct_concentrations(X[i], Y[i])

            results["concentrations"] = C

        return results
