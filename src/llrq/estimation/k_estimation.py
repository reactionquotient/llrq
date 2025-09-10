import numpy as np
import cvxpy as cp


class KMatrixEstimator:
    def __init__(self, network, equilibrium_concentrations, eps=1e-9):
        self.network = network
        self.c_eq = np.asarray(equilibrium_concentrations, dtype=float).reshape(-1)
        self.eps = eps

        if len(self.c_eq) != network.n_species:
            raise ValueError(f"Expected {network.n_species} c_eq entries, got {len(self.c_eq)}")

        self.A = self._make_A_matrix()
        self.A_sqrt, self.A_invsqrt = self._sqrt_and_invsqrt(self.A)

    def _make_A_matrix(self):
        # Expect network.S shape [n_species x n_reactions]
        N = np.asarray(self.network.S, dtype=float)
        if N.shape[0] != len(self.c_eq):
            raise ValueError("S row-count must match length of c_eq")
        Wc = np.diag(1.0 / self.c_eq)  # diag(1/c_eq)
        A = N.T @ Wc @ N
        A = 0.5 * (A + A.T)  # symmetrize
        A += self.eps * np.eye(A.shape[0])  # ridge for SPD
        return A

    def _sqrt_and_invsqrt(self, A):
        lam, U = np.linalg.eigh(A)
        lam = np.clip(lam, 1e-12, None)  # floor tiny/negative eigenvalues
        A_sqrt = (U * np.sqrt(lam)) @ U.T
        A_invsqrt = (U * (1.0 / np.sqrt(lam))) @ U.T
        return A_sqrt, A_invsqrt

    def build_k_bounds_constraints(self, K, g_diag_min=None, g_diag_max=None, gamma_min=None, gamma_max=None):
        r = self.A.shape[0]
        W = cp.Variable((r, r), PSD=True)  # similarity variable, PSD => symmetric

        cons = [W == self.A_invsqrt @ K @ self.A_sqrt]  # ties K to W (⇒ K = A^(1/2) W A^(-1/2))

        # Spectral caps/floors on G via W ⪯ γ_max A and γ_min A ⪯ W
        if gamma_max is not None:
            cons += [W << gamma_max * self.A]
        if gamma_min is not None and gamma_min > 0:
            cons += [gamma_min * self.A << W]

        # Diagonal bounds on G = A^{-1/2} W A^{-1/2}
        if g_diag_max is not None:
            g_diag_max = np.asarray(g_diag_max, dtype=float).reshape(-1)
            if g_diag_max.size != r:
                raise ValueError("g_diag_max must have length equal to #reactions")
            cons += [cp.diag(self.A_invsqrt @ W @ self.A_invsqrt) <= g_diag_max]
            if gamma_max is None:
                cons += [W << float(np.max(g_diag_max)) * self.A]

        if g_diag_min is not None:
            g_diag_min = np.asarray(g_diag_min, dtype=float).reshape(-1)
            if g_diag_min.size != r:
                raise ValueError("g_diag_min must have length equal to #reactions")
            cons += [cp.diag(self.A_invsqrt @ W @ self.A_invsqrt) >= g_diag_min]
            if gamma_min is None and np.min(g_diag_min) > 0:
                cons += [float(np.min(g_diag_min)) * self.A << W]

        return cons, W
