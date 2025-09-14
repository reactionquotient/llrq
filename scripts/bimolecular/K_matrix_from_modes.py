import numpy as np


def orthogonal_with_flat_first_row(m: int) -> np.ndarray:
    """
    Build an orthogonal U whose first row is constant 1/sqrt(m).
    Deterministic: uses QR with a fixed 'seed' matrix.
    """
    u = np.ones(m) / np.sqrt(m)  # desired first row
    A = np.zeros((m, m))
    A[:, 0] = u  # first column ~ u (as a column)
    # fill remaining columns with canonical basis (shifted), ensures determinism
    for j in range(1, m):
        e = np.zeros(m)
        e[j] = 1.0
        A[:, j] = e
    # QR with column pivoting off for determinism
    Q, _ = np.linalg.qr(A, mode="reduced")
    # Now Q has first column ~ u; we want U whose FIRST ROW is u.
    # So take U := Q^T. Then U[0, :] = u^T (up to sign; fix sign if needed).
    U = Q.T
    # ensure the first row is +u (not -u) for consistency
    if np.dot(U[0], u) < 0:
        U[0] *= -1.0
        U[1:] *= -1.0  # keep orthogonality
    return U


def build_symmetric_K_and_x0(lambdas, amps):
    """
    Given modal rates (lambdas) and fitted weights (amps) for x1(t)=sum a_i e^{-λ_i t},
    return symmetric K and an initial state x0 so that the first state x1(t) reproduces
    the mixture exactly under dx/dt = -K x.
    """
    lam = np.asarray(lambdas, float).ravel()
    a = np.asarray(amps, float).ravel()
    assert lam.shape == a.shape and lam.ndim == 1, "lambdas and amps must be 1D, same length"
    m = lam.size

    U = orthogonal_with_flat_first_row(m)  # orthogonal, flat first row
    K = (U * lam) @ U.T  # U diag(lam) U^T  (broadcast multiply)

    # coefficients in eigenbasis so that x1(t)=sum a_i e^{-λ_i t}
    alpha = np.sqrt(m) * a
    x0 = U @ alpha
    return K, x0, U


def x1_of_t(t, K, x0):
    """Compute the observed coordinate x1(t) = e1^T x(t) for given K, x0."""
    # use eigen-decomposition for efficiency/numerics
    lam, U = np.linalg.eigh(K)  # K = U diag(lam) U^T (U orthonormal)
    alpha = U.T @ x0
    # e1^T U = first row of U^T = first column of U^T? Careful: we need first row of U^T, i.e. U[0,:].
    frow = U[0, :]  # shape (m,), equals 1/sqrt(m) if U has flat first row
    t = np.asarray(t, float)
    return (frow * (alpha * np.exp(-np.outer(t, lam)))).sum(axis=1)


# -------- Example usage --------
if __name__ == "__main__":
    # Suppose you fitted m modes:
    lams = np.array([7.96e2, 5.83, 0.9])  # rates (s^-1)
    amps = np.array([0.75, 0.22, 0.03])  # weights in x1(t)
    K, x0, U = build_symmetric_K_and_x0(lams, amps)

    # Verify x1(t) matches the mixture
    t = np.linspace(0, 0.5, 2000)
    x1_direct = (amps * np.exp(-np.outer(t, lams))).sum(axis=1)
    x1_model = x1_of_t(t, K, x0)
    print("max abs error:", np.max(np.abs(x1_direct - x1_model)))

    # Recover ln Q1 by adding the (fitted) intercept b1:
    b1 = -2.0  # ln Keq_1 (from your fit)
    lnQ1 = b1 + x1_model
    Q1 = np.exp(lnQ1)
