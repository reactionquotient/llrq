# tests/test_kalman_reduced.py
import numpy as np
import pytest

from llrq.estimation.kalman import (
    reduce_y_from_Q, reduce_y_from_x,
    ReducedKalmanFilterCT, ReducedKalmanFilterDT
)

# ----------------------------
# Utilities
# ----------------------------

def ortho_cols(mat, tol=1e-12):
    """QR-orthonormalize columns for a clean reduced basis B."""
    Q, _ = np.linalg.qr(mat)
    # Handle potential sign flips deterministically
    for i in range(Q.shape[1]):
        if Q[0, i] < 0:
            Q[:, i] *= -1
    return Q

rng = np.random.default_rng(12345)


# ----------------------------
# Mapping helpers
# ----------------------------

def test_reduce_helpers_match_definitions():
    r, rS = 5, 3
    B = ortho_cols(rng.standard_normal((r, rS)))
    x = rng.standard_normal(r)
    lnKeq = rng.standard_normal(r)
    Q = np.exp(lnKeq + x)  # so that ln(Q) - lnKeq = x

    y_from_x = reduce_y_from_x(B, x)
    y_from_Q = reduce_y_from_Q(B, lnKeq, Q)

    assert y_from_x.shape == (rS,)
    assert y_from_Q.shape == (rS,)
    np.testing.assert_allclose(y_from_x, B.T @ x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(y_from_Q, B.T @ x, rtol=0, atol=1e-12)


# ----------------------------
# Continuous-time steady-state KF
# ----------------------------

def test_ct_kalman_converges_on_noisy_measurements():
    # Stable reduced dynamics: dot y = A y + w, z = y + v
    A = np.array([[-0.8, 0.2],
                  [ 0.0, -0.5]], dtype=float)
    n = A.shape[0]
    W = 1e-6 * np.eye(n)         # process PSD
    V = (0.05 ** 2) * np.eye(n)  # measurement PSD (5% noise in "units" of y)

    kf = ReducedKalmanFilterCT(A=A, C=None, W=W, V=V)
    y_true = rng.standard_normal(n) * 0.5  # random start
    kf.reset(y0=np.zeros(n))

    dt = 0.01
    T = 6.0
    steps = int(T / dt)

    errs = []
    for k in range(steps):
        # propagate truth with small process noise
        w = rng.multivariate_normal(np.zeros(n), W*dt)
        y_true = y_true + dt * (A @ y_true) + w

        # noisy measurement
        v = rng.multivariate_normal(np.zeros(n), V)
        z = y_true + v

        yhat = kf.step(z=z, dt=dt, u_red=np.zeros(n))
        errs.append(np.linalg.norm(yhat - y_true))

    # Error should drop significantly and be small at the end
    assert errs[-1] < 0.25, f"final error too large: {errs[-1]:.3f}"
    # Median of last 25% less than half of first 25%
    q = steps // 4
    assert (np.median(errs[-q:]) < 0.5 * np.median(errs[:q]))


# ----------------------------
# Discrete-time KF: PSD + convergence
# ----------------------------

def test_dt_kalman_psd_and_converges():
    # Start from a stable CT A and discretize with small dt
    A = np.array([[-1.0, 0.3],
                  [ 0.0, -0.6]], dtype=float)
    n = A.shape[0]
    dt = 0.02
    A_d = np.eye(n) + A * dt  # Euler discretization (stable for small dt)
    Qc = 5e-6 * np.eye(n)
    Qd = Qc * dt
    Rd = (0.04 ** 2) * np.eye(n)

    kfd = ReducedKalmanFilterDT(A_d=A_d, B_d=None, C=None, Qd=Qd, Rd=Rd)
    kfd.reset(y0=np.zeros(n))

    y_true = rng.standard_normal(n)
    errs = []

    for _ in range(800):
        # truth
        w = rng.multivariate_normal(np.zeros(n), Qd)
        y_true = A_d @ y_true + w
        # measurement
        v = rng.multivariate_normal(np.zeros(n), Rd)
        z = y_true + v

        yhat = kfd.step(z=z)
        errs.append(np.linalg.norm(yhat - y_true))

        # P should remain PSD (Joseph form)
        evals = np.linalg.eigvalsh(kfd.P)
        assert evals.min() > -1e-10, "Covariance lost PSD"

    assert errs[-1] < 0.3
    q = len(errs) // 4
    assert (np.median(errs[-q:]) < 0.6 * np.median(errs[:q]))


# ----------------------------
# From-solver constructor sanity
# ----------------------------

def test_ct_from_solver_uses_reduced_model():
    # Build a "full" K with rank deficiency (cycles) and verify A = -B^T K B
    r, rS = 4, 2
    B = ortho_cols(rng.standard_normal((r, rS)))
    K_red = np.diag([1.2, 0.7])             # SPD on reduced space
    # Full K that is singular in the nullspace complement
    K_full = B @ K_red @ B.T                 # rank 2 in 4-dim

    class _Dyn:
        def __init__(self, K): self.K = K

    class _Solver:
        def __init__(self, B, K):
            self._B = B
            self._rankS = B.shape[1]
            self.dynamics = _Dyn(K)

    solver = _Solver(B=B, K=K_full)
    kf = ReducedKalmanFilterCT.from_solver(solver, W=1e-6*np.eye(rS), V=1e-3*np.eye(rS))

    # A computed inside should be -K_red (within numerical tolerance)
    A_expected = -K_red
    np.testing.assert_allclose(kf.A, A_expected, rtol=1e-12, atol=1e-12)
    assert kf.yhat.shape == (rS,)

