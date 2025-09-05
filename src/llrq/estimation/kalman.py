import numpy as np
from typing import Optional, Tuple
import warnings
from scipy.linalg import solve_continuous_are, expm

Array = np.ndarray


def _as_psd(mat_or_scalar: Optional[Array], n: int, default: float) -> Array:
    """Make (n x n) PSD from scalar or matrix."""
    if mat_or_scalar is None:
        return default * np.eye(n)
    M = np.array(mat_or_scalar, dtype=float)
    if M.ndim == 0:
        return float(M) * np.eye(n)
    if M.shape != (n, n):
        raise ValueError(f"matrix must be {n}x{n}, got {M.shape}")
    return M


def reduce_y_from_Q(B: Array, lnKeq: Array, Q: Array) -> Array:
    """
    Map reaction quotients Q (length r) to reduced measurement y (length r_S):
      y = B^T ( ln(Q) - ln(Keq) )
    """
    return B.T @ (np.log(Q) - lnKeq)


def reduce_y_from_x(B: Array, x: Array) -> Array:
    """
    Map log-deviations x (length r) to reduced measurement y (length r_S):
      y = B^T x
    """
    return B.T @ x


class ReducedKalmanFilterCT:
    """
    Continuous-time steady-state Kalman filter on the reduced LLRQ model.

    Plant (reduced coordinates):
        dot y = A y + u_red(t) + d(t),   (process noise with PSD W)
        z     = C y + v(t),              (measurement noise with PSD V)

    The steady-state Kalman gain solves the dual CARE:
        A P + P A^T - P C^T V^{-1} C P + W = 0
        L = P C^T V^{-1}

    The 'step' method advances one Euler step of dt:
        yhat <- yhat + dt * (A yhat + u_red + L (z - C yhat))
        (optionally add d(t) externally to u_red if you like)
    """

    def __init__(self, A: Array, C: Optional[Array] = None,
                 W: Optional[Array] = None, V: Optional[Array] = None):
        A = np.array(A, dtype=float)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("A must be square")
        self.A = A
        self.C = np.eye(n) if C is None else np.array(C, dtype=float)
        if self.C.shape != (n, n):
            raise ValueError("C must be n x n (use an explicit output map otherwise)")
        self.W = _as_psd(W, n, default=1e-8)
        self.V = _as_psd(V, n, default=1e-4)

        # Solve steady-state filter Riccati (dual CARE)
        try:
            P = solve_continuous_are(self.A.T, self.C.T, self.W, self.V)
            # Robust solve for V^{-1}
            try:
                Vinv = np.linalg.inv(self.V)
            except np.linalg.LinAlgError:
                Vinv = np.linalg.pinv(self.V)
            self.L = P @ self.C.T @ Vinv
        except Exception as e:
            warnings.warn(f"CARE solve failed ({e}); using small-gain heuristic.")
            self.L = 1e-3 * np.eye(n)

        self.yhat = np.zeros(n)

    @classmethod
    def from_solver(cls, solver, W: Optional[Array] = None, V: Optional[Array] = None):
        """
        Build from an LLRQSolver (uses reduced dynamics A = -K_red).
        """
        B = solver._B
        K_red = B.T @ solver.dynamics.K @ B
        A = -K_red
        return cls(A=A, C=None, W=W, V=V)

    def reset(self, y0: Optional[Array] = None):
        self.yhat = np.zeros_like(self.yhat) if y0 is None else np.array(y0, dtype=float)

    def step(self, z: Array, dt: float, u_red: Optional[Array] = None) -> Array:
        """
        Advance the steady-state CT filter by one Euler step of size dt.
        Args:
            z: reduced measurement (n,)
            dt: step size
            u_red: known reduced input (n,), e.g. B_red uhat + projected exogenous drive
        Returns:
            yhat (n,)
        """
        z = np.array(z, dtype=float)
        n = self.A.shape[0]
        if z.shape != (n,):
            raise ValueError(f"z must be shape ({n},)")
        if u_red is None:
            u_red = np.zeros(n)
        # Predictor-corrector (explicit Euler with correction term)
        innov = z - self.C @ self.yhat
        self.yhat = self.yhat + dt * (self.A @ self.yhat + u_red + self.L @ innov)
        return self.yhat.copy()


class ReducedKalmanFilterDT:
    """
    Discrete-time reduced Kalman filter.

    Model:
        y_{k+1} = A_d y_k + B_d u_k + w_k,   w_k ~ N(0, Qd)
        z_k     = C   y_k + v_k,             v_k ~ N(0, Rd)

    Uses Joseph-form covariance update for numerical PSD safety.
    """

    def __init__(self, A_d: Array, B_d: Optional[Array] = None, C: Optional[Array] = None,
                 Qd: Optional[Array] = None, Rd: Optional[Array] = None):
        A_d = np.array(A_d, dtype=float)
        n = A_d.shape[0]
        if A_d.shape != (n, n):
            raise ValueError("A_d must be square")
        self.A_d = A_d
        self.B_d = np.zeros((n, n)) if B_d is None else np.array(B_d, dtype=float)
        self.C   = np.eye(n) if C   is None else np.array(C,   dtype=float)
        self.Qd  = _as_psd(Qd, n, default=1e-6)
        self.Rd  = _as_psd(Rd, n, default=1e-3)
        self.yhat = np.zeros(n)
        self.P    = 1e-2 * np.eye(n)   # initial covariance

    @classmethod
    def from_solver(cls, solver, dt: float,
                    Qc: Optional[Array] = None, Rd: Optional[Array] = None,
                    discretization: str = "euler"):
        """
        Build a DT filter from solver by discretizing A = -K_red.
        - 'euler': A_d = I + A*dt, Qd ≈ Qc*dt  (good for small dt)
        - 'exact': A_d = expm(A*dt), Qd  ≈ Qc*dt (simple PSD approx; swap if you implement Van Loan)
        """
        B = solver._B
        A = -(B.T @ solver.dynamics.K @ B)
        n = A.shape[0]
        if discretization == "exact":
            A_d = expm(A * dt)
        elif discretization == "euler":
            A_d = np.eye(n) + A * dt
        else:
            raise ValueError("discretization must be 'euler' or 'exact'")
        Qd = _as_psd(Qc, n, default=1e-6) * dt
        return cls(A_d=A_d, B_d=None, C=None, Qd=Qd, Rd=Rd)

    def reset(self, y0: Optional[Array] = None, P0: Optional[Array] = None):
        self.yhat = np.zeros_like(self.yhat) if y0 is None else np.array(y0, dtype=float)
        if P0 is not None:
            P0 = np.array(P0, dtype=float)
            if P0.shape != self.P.shape:
                raise ValueError("P0 has wrong shape")
            self.P = P0

    def predict(self, u: Optional[Array] = None):
        u = np.zeros(self.A_d.shape[0]) if u is None else np.array(u, dtype=float)
        self.yhat = self.A_d @ self.yhat + (self.B_d @ u if self.B_d.size else 0.0)
        self.P = self.A_d @ self.P @ self.A_d.T + self.Qd

    def update(self, z: Array):
        z = np.array(z, dtype=float)
        S = self.C @ self.P @ self.C.T + self.Rd
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.C.T @ S_inv
        innov = z - self.C @ self.yhat
        self.yhat = self.yhat + K @ innov
        I = np.eye(self.P.shape[0])
        # Joseph form
        self.P = (I - K @ self.C) @ self.P @ (I - K @ self.C).T + K @ self.Rd @ K.T

    def step(self, z: Array, u: Optional[Array] = None):
        self.predict(u)
        self.update(z)
        return self.yhat.copy()

