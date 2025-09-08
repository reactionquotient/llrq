import numpy as np
from scipy.linalg import solve_continuous_are


class ReducedKalmanFilterCT:
    """
    Continuous-time Kalman filter for the reduced LLRQ model.
    Estimates y from measurements z with dynamics: dy/dt = A*y + B*u + w, z = C*y + v
    """

    def __init__(self, A, C, W, V, B=None):
        """
        Args:
            A: System matrix (rankS x rankS)
            C: Measurement matrix (rankS x rankS, typically identity)
            W: Process noise covariance (rankS x rankS)
            V: Measurement noise covariance (rankS x rankS)
            B: Input matrix (rankS x m) - optional
        """
        self.A = np.array(A, dtype=float)
        self.C = np.array(C, dtype=float)
        self.W = np.array(W, dtype=float)
        self.V = np.array(V, dtype=float)
        self.B = np.array(B, dtype=float) if B is not None else None

        self.n = self.A.shape[0]  # state dimension
        self.m = self.C.shape[0]  # measurement dimension

        # Solve steady-state Riccati equation for P
        try:
            self.P = solve_continuous_are(self.A.T, self.C.T, self.W, self.V)
        except:
            # Fallback to identity if Riccati fails
            self.P = np.eye(self.n)

        # Kalman gain
        self.K = self.P @ self.C.T @ np.linalg.pinv(self.V)

        # Current estimate
        self.y_hat = np.zeros(self.n)

    @classmethod
    def from_solver(cls, solver, W, V, C=None):
        """
        Create Kalman filter from LLRQ solver.

        Args:
            solver: LLRQSolver instance
            W: Process noise covariance
            V: Measurement noise covariance
            C: Measurement matrix (defaults to identity)
        """
        B = solver._B
        K_red = B.T @ solver.dynamics.K @ B
        A = -K_red

        rankS = solver._rankS
        if C is None:
            C = np.eye(rankS)

        return cls(A=A, C=C, W=W, V=V)

    def reset(self, y0=None):
        """Reset filter with initial estimate y0."""
        if y0 is None:
            self.y_hat = np.zeros(self.n)
        else:
            self.y_hat = np.array(y0, dtype=float)

    def step(self, z, dt, u_red=None):
        """
        Perform one Kalman filter step.

        Args:
            z: measurement vector (m,)
            dt: time step
            u_red: control input in reduced space (optional)

        Returns:
            y_hat: updated state estimate
        """
        # Predict
        dydt = self.A @ self.y_hat
        if u_red is not None and self.B is not None:
            dydt += self.B @ u_red

        # Simple Euler integration for prediction
        y_pred = self.y_hat + dt * dydt

        # Update with measurement
        innovation = z - self.C @ y_pred
        self.y_hat = y_pred + self.K @ innovation

        return self.y_hat.copy()


def reduce_y_from_Q(B, lnKeq_consistent, Q_meas):
    """
    Convert measured reaction quotients Q to reduced state y.

    Args:
        B: Basis matrix (r x rankS)
        lnKeq_consistent: log equilibrium constants (r,)
        Q_meas: measured reaction quotients (r,)

    Returns:
        y: reduced state vector (rankS,)
    """
    # Convert Q to log-space deviation from equilibrium
    x = np.log(Q_meas) - lnKeq_consistent

    # Project to reduced space
    y = B.T @ x

    return y
