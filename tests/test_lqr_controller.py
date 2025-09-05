# tests/test_lqr_controller.py
import numpy as np
import pytest

from llrq.control.lqr import LQRController


def _orthonormal_cols(M):
    # deterministic sign convention for stability in CI
    Q, _ = np.linalg.qr(M)
    for i in range(Q.shape[1]):
        if Q[0, i] < 0:
            Q[:, i] *= -1
    return Q


def _make_mock_solver(r=3, rS=2, seed=7):
    rng = np.random.default_rng(seed)
    B = _orthonormal_cols(rng.standard_normal((r, rS)))   # basis for Im(S^T)
    # Positive-definite reduced K (diagonal for clarity)
    K_red = np.diag([1.2, 0.7][:rS])
    K_full = B @ K_red @ B.T                              # rank rS in r-dim

    class _Dyn:
        def __init__(self, K):
            self.K = K
            self.n_reactions = K.shape[0]

    class _Net:
        def __init__(self):
            self.reaction_to_idx = {"R1": 0, "R2": 1, "R3": 2}

    class _Solver:
        def __init__(self, B, K):
            self._B = B
            self._rankS = B.shape[1]
            self.dynamics = _Dyn(K)
            self.network = _Net()

    return _Solver(B, K_full), B, K_red


def _simulate_closed_loop(controller, B, K_red, y0, y_ref, T=8.0, dt=0.01, disturbance=None, uhat_bounds=None):
    """Integrate reduced dynamics: ẏ = -K_red y + Bᵀ u_full + d(t)"""
    A = -K_red
    t = 0.0
    y = y0.copy()
    ys, us = [], []
    for _ in range(int(T / dt)):
        d = np.zeros_like(y)
        if disturbance is not None:
            d = disturbance(t)
        u_full = controller.u_full(t, y, y_ref, uhat_bounds=uhat_bounds)
        u_red = B.T @ u_full
        y = y + dt * (A @ y + u_red + d)
        ys.append(y.copy())
        us.append(u_full.copy())
        t += dt
    return np.array(ys), np.array(us)


def test_constructs_and_wires_reduced_matrices():
    solver, B, K_red = _make_mock_solver()
    ctrl = LQRController(solver, controlled_reactions=["R1", "R3"], Q=np.eye(solver._rankS), R=0.1*np.eye(2))
    # A inside controller should be -K_red (within tolerance)
    np.testing.assert_allclose(ctrl.A, -K_red, rtol=1e-12, atol=1e-12)
    # B (reduced input) should be Bᵀ G; rank should be full (here = rS=2)
    assert hasattr(ctrl, "B")
    assert ctrl.B.shape == (solver._rankS, 2)
    assert np.linalg.matrix_rank(ctrl.B) == solver._rankS


def test_lqr_with_integral_tracks_setpoint_under_disturbance():
    solver, B, K_red = _make_mock_solver()
    # Two actuators: R1 and R3
    ctrl = LQRController(solver,
                         controlled_reactions=["R1", "R3"],
                         Q=np.eye(solver._rankS),
                         R=0.05*np.eye(2),
                         integral=True,
                         Ki_weight=0.2)
    ctrl.reset()

    # Nonzero target (driven steady state)
    y_ref = np.array([0.5, -0.25])[:solver._rankS]
    y0 = np.array([0.3, 0.2])[:solver._rankS]

    # Constant disturbance + small sinusoid
    def d(t):
        return np.array([0.05, -0.02])[:solver._rankS] + 0.03*np.array([np.sin(0.7*t), 0.3*np.sin(1.3*t)])[:solver._rankS]

    ys, us = _simulate_closed_loop(ctrl, B, K_red, y0, y_ref, T=10.0, dt=0.01, disturbance=d)

    # Final tracking error should be small with integral action
    err_final = np.linalg.norm(ys[-1] - y_ref)
    assert err_final < 0.03, f"Integral LQR failed to track setpoint: final error {err_final:.3f}"

    # Inputs should only appear on selected reactions (R1, R3), R2 ≈ 0
    # (allow tiny numerical wiggle)
    idx_R2 = 1
    assert np.max(np.abs(us[:, idx_R2])) < 1e-8


def test_lqr_without_integral_exhibits_steady_state_error_under_disturbance():
    solver, B, K_red = _make_mock_solver()
    ctrl = LQRController(solver,
                         controlled_reactions=["R1", "R3"],
                         Q=np.eye(solver._rankS),
                         R=0.05*np.eye(2),
                         integral=False)
    ctrl.reset()

    y_ref = np.array([0.5, -0.25])[:solver._rankS]
    y0 = np.array([0.3, 0.2])[:solver._rankS]

    def d(t):
        return np.array([0.05, -0.02])[:solver._rankS]

    ys, _ = _simulate_closed_loop(ctrl, B, K_red, y0, y_ref, T=10.0, dt=0.01, disturbance=d)
    err_final = np.linalg.norm(ys[-1] - y_ref)
    # Without integral action, there should remain a noticeable bias
    assert err_final > 0.05, f"Expected nonzero steady-state error without integral, got {err_final:.3f}"


def test_input_saturation_and_state_boundedness():
    solver, B, K_red = _make_mock_solver()
    ctrl = LQRController(solver,
                         controlled_reactions=["R1", "R3"],
                         Q=1.0*np.eye(solver._rankS),
                         R=0.02*np.eye(2),
                         integral=True,
                         Ki_weight=0.2)
    ctrl.reset()

    # Tight input bounds to force saturation
    uhat_bounds = (np.array([-0.2, -0.2]), np.array([0.2, 0.2]))

    # Aggressive target to trigger sustained saturation
    y_ref = np.array([1.2, -0.9])[:solver._rankS]
    y0 = np.array([0.0, 0.0])[:solver._rankS]

    ys, us = _simulate_closed_loop(ctrl, B, K_red, y0, y_ref, T=8.0, dt=0.01,
                                   disturbance=None, uhat_bounds=uhat_bounds)

    # Most inputs on actuated channels should be clamped at the bounds for a while
    idxs = [0, 2]  # R1 and R3
    clamped_counts = sum(np.isclose(np.abs(us[:, j]), 0.2, atol=1e-6).sum() for j in idxs)
    assert clamped_counts > 0.15 * us.shape[0] * 2, "Expected significant saturation on actuated reactions"

    # State stays bounded (no windup-induced blow-up)
    y_norms = np.linalg.norm(ys, axis=1)
    assert np.max(y_norms) < 5.0, "State exploded under saturation; anti-windup likely broken"

    # Unactuated reaction index (R2) should remain ~0 input
    assert np.max(np.abs(us[:, 1])) < 1e-8

