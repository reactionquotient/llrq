# tests/test_time_varying_drives.py
import numpy as np
import pytest
from scipy.linalg import expm

from llrq.ops.time_varying import exact_const_solution, foh_discretize, simulate_zoh, zoh_discretize

rng = np.random.default_rng(2025)


def _stable_A(n=2, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lambdas = -0.5 - 1.2 * rng.random(n)  # negative real parts
    return Q @ np.diag(lambdas) @ np.linalg.inv(Q)


def test_zoh_matches_exact_constant_drive():
    A = _stable_A(3, seed=1)
    B = rng.standard_normal((3, 2))
    y0 = rng.standard_normal(3)
    u0 = rng.standard_normal(2)
    T = 2.75

    # Closed form
    y_exact = exact_const_solution(A, B, y0, u0, T)

    # ZOH with one big step equals exact for constant input
    Ad, Bd = zoh_discretize(A, B, T)
    y_zoh_one = Ad @ y0 + Bd @ u0
    np.testing.assert_allclose(y_zoh_one, y_exact, rtol=1e-12, atol=1e-12)

    # And many small ZOH steps give the same result
    t_grid = np.linspace(0.0, T, 50)
    y_many = simulate_zoh(A, B, lambda t: u0, t_grid, y0)[-1]
    np.testing.assert_allclose(y_many, y_exact, rtol=1e-9, atol=1e-9)


def test_foh_ramp_matches_high_res_reference_and_beats_zoh():
    A = _stable_A(2, seed=2)
    B = np.array([[0.0], [1.0]])  # SISO
    y0 = np.array([0.2, -0.1])
    T = 5.0
    dt_coarse = 0.1
    dt_ref = 0.001  # high-resolution reference via ZOH stepping

    # Ramp input: u(t) = u0 + a*t
    u0 = 0.3
    a = -0.08

    def u_of_t(t):
        return np.array([u0 + a * t])

    # Reference: very fine ZOH stepping of the true ramp
    t_ref = np.arange(0.0, T + 1e-12, dt_ref)
    y_ref = simulate_zoh(A, B, u_of_t, t_ref, y0)[-1]

    # Coarse FOH using exact FOH discretization
    Ad, B0, B1 = foh_discretize(A, B, dt_coarse)
    y_foh = y0.copy()
    t = 0.0
    N = int(np.round(T / dt_coarse))
    for k in range(N):
        uk = u_of_t(t)
        uk1 = u_of_t(t + dt_coarse)
        y_foh = Ad @ y_foh + (B0 @ uk + B1 @ (uk1 - uk)).reshape(-1)
        t += dt_coarse

    # Coarse ZOH (hold u at left endpoint)
    t_coarse = np.arange(0.0, T + 1e-12, dt_coarse)
    y_zoh_coarse = simulate_zoh(A, B, u_of_t, t_coarse, y0)[-1]

    # FOH should be closer to the fine reference than coarse ZOH
    err_foh = np.linalg.norm(y_foh - y_ref)
    err_zoh = np.linalg.norm(y_zoh_coarse - y_ref)
    assert err_foh < err_zoh, f"FOH error {err_foh:.3e} not < ZOH error {err_zoh:.3e}"
    assert err_foh < 5e-3, "FOH not accurate enough vs. high-res reference"


def test_shapes_and_basic_stability():
    A = _stable_A(4, seed=3)
    B = rng.standard_normal((4, 3))
    dt = 0.2
    Ad, Bd = zoh_discretize(A, B, dt)
    assert Ad.shape == (4, 4)
    assert Bd.shape == (4, 3)

    # discrete eigenvalues should lie inside unit circle for stable A (small dt)
    radii = np.abs(np.linalg.eigvals(Ad))
    assert np.max(radii) < 1.0
