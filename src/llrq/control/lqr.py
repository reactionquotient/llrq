import numpy as np
import warnings
from scipy.linalg import solve_continuous_are

class LQRController:
    """
    LQR for the reduced LLRQ model.
    Controls y in dot y = A y + B uhat + d(t), then maps to full u_full = G uhat.
    """
    def __init__(self, solver, controlled_reactions, Q=None, R=None,
                 integral=False, Ki_weight=1.0):
        """
        Args:
            solver: LLRQSolver (already built, with _B, _rankS, dynamics.K, etc.)
            controlled_reactions: list of reaction IDs or indices that are actuated
            Q, R: LQR weight matrices (defaults: I)
            integral: add integral action on y-y_ref
            Ki_weight: weight on integral states in Q_aug (if integral=True)
        """
        self.solver = solver
        self.Basis = solver._B
        self.rankS = solver._rankS

        # Reduced plant matrices
        K = solver.dynamics.K
        self.K_red = self.Basis.T @ K @ self.Basis
        self.A = -self.K_red

        # Build selection G (r x m)
        r = solver.dynamics.n_reactions
        idx = []
        for rid in controlled_reactions:
            if isinstance(rid, str):
                idx.append(solver.network.reaction_to_idx[rid])
            else:
                idx.append(int(rid))
        m = len(idx)
        G = np.zeros((r, m))
        for j, k in enumerate(idx):
            G[k, j] = 1.0
        self.G = G

        # Reduced input matrix B = B^T G
        self.B = self.Basis.T @ self.G
        if np.linalg.matrix_rank(self.B) < min(self.rankS, m):
            warnings.warn("B_red appears rank-deficient; the selected reactions may not fully control y.")

        # Weights
        self.Q = np.eye(self.rankS) if Q is None else np.array(Q, float)
        self.R = np.eye(m) if R is None else np.array(R, float)

        self.integral = integral
        self.ny = self.rankS
        self.m = m

        if not integral:
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            # uhat = -F_y (y - y_ref)
            self.F_y = np.linalg.solve(self.R, self.B.T @ P)
            self.F_eta = None
        else:
            # Augment with integral states: z = [y; eta], eta_dot = y - y_ref
            A_aug = np.block([
                [self.A,               np.zeros((self.ny, self.ny))],
                [np.eye(self.ny),      np.zeros((self.ny, self.ny))]
            ])
            B_aug = np.vstack([self.B, np.zeros((self.ny, self.m))])
            Q_aug = np.block([
                [self.Q,                         np.zeros((self.ny, self.ny))],
                [np.zeros((self.ny, self.ny)),   Ki_weight*np.eye(self.ny)]
            ])
            P = solve_continuous_are(A_aug, B_aug, Q_aug, self.R)
            K_aug = np.linalg.solve(self.R, B_aug.T @ P)  # (m x 2ny)
            self.F_y   = K_aug[:, :self.ny]
            self.F_eta = K_aug[:, self.ny:]
            self._eta = np.zeros(self.ny)
            self._last_t = None

    def reset(self):
        if self.integral:
            self._eta[:] = 0.0
            self._last_t = None

    def u_full(self, t, y, y_ref, uhat_bounds=None):
        """
        Returns full-space control u_full = G * uhat.
        Args:
            t: time
            y: reduced state (rankS,)
            y_ref: desired reduced setpoint (rankS,)
            uhat_bounds: (umin, umax) tuple for clipping in R^m (optional)
        """
        e = y - y_ref
        if self.integral:
            if self._last_t is None:
                self._last_t = t
            dt = max(0.0, t - self._last_t)
            self._eta = self._eta + dt*e
            self._last_t = t
            uhat = - self.F_y @ e - self.F_eta @ self._eta
        else:
            uhat = - self.F_y @ e

        if uhat_bounds is not None:
            umin, umax = uhat_bounds
            uhat = np.minimum(np.maximum(uhat, umin), umax)

        return self.G @ uhat  # map to full r-dim force

