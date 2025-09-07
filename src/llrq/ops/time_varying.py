# src/llrq/ops/time_varying.py
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm


def _asarray_2d(M: ArrayLike) -> np.ndarray:
    A = np.array(M, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square 2D matrix")
    return A


def exact_const_solution(A: ArrayLike, B: ArrayLike, y0: ArrayLike, u0: ArrayLike, t: float) -> np.ndarray:
    """
    Exact solution for constant input u(t) = u0 on [0, t]:
        y(t) = e^{A t} y0 + (∫_0^t e^{A τ} dτ) B u0
             = e^{A t} y0 + X B u0,      with A X = e^{A t} - I
    Implemented via a linear solve (no explicit matrix inverse).
    """
    A = _asarray_2d(A)
    B = np.array(B, dtype=float)
    y0 = np.array(y0, dtype=float).reshape(-1)
    u0 = np.array(u0, dtype=float).reshape(-1)
    Et = expm(A * t)
    rhs = Et - np.eye(A.shape[0])
    X = np.linalg.solve(A, rhs)  # robust even for very slow modes
    return Et @ y0 + (X @ B) @ u0


def zoh_discretize(A: ArrayLike, B: ArrayLike, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact ZOH discretization for y' = A y + B u, with u held constant on [kΔ, (k+1)Δ):
        y_{k+1} = A_d y_k + B_d u_k
        A_d = e^{A Δ}
        B_d = (∫_0^Δ e^{A τ} dτ) B  = X B,  with A X = A_d - I
    """
    A = _asarray_2d(A)
    B = np.array(B, dtype=float)
    Ad = expm(A * dt)
    X = np.linalg.solve(A, Ad - np.eye(A.shape[0]))
    Bd = X @ B
    return Ad, Bd


def foh_discretize(A: ArrayLike, B: ArrayLike, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact FOH (first-order hold) for y' = A y + B u with u(t) linearly interpolated on [kΔ, (k+1)Δ):
        u(t) ≈ u_k + (t/Δ)(u_{k+1} - u_k)
    Returns (A_d, B0, B1) for the update
        y_{k+1} = A_d y_k + B0 u_k + B1 (u_{k+1} - u_k).
    Note: When u_{k+1} = u_k, this reduces to the ZOH update with B_d = B0.
    Implemented via a (n+2m) x (n+2m) Van-Loan block exponential.
    """
    A = _asarray_2d(A)
    B = np.array(B, dtype=float)
    n, m = A.shape[0], B.shape[1]
    Znn = np.zeros((n, n))
    Znm = np.zeros((n, m))
    Zmn = np.zeros((m, n))
    Zmm = np.zeros((m, m))
    Im = np.eye(m)

    # M = [[A, B, 0],
    #      [0, 0, I],
    #      [0, 0, 0]]
    M = np.block([[A, B, Znm], [Zmn, Zmm, Im], [Zmn, Zmm, Zmm]])
    E = expm(M * dt)
    Ad = E[:n, :n]  # (n x n)
    B0 = E[:n, n : n + m]  # (n x m)
    B1 = E[:n, n + m : n + 2 * m]  # (n x m)
    return Ad, B0, B1


def simulate_zoh(A: ArrayLike, B: ArrayLike, u_of_t, t_grid: np.ndarray, y0: ArrayLike) -> np.ndarray:
    """
    Integrate y' = A y + B u(t) using exact ZOH across arbitrary time steps in t_grid.
    u_of_t is sampled at the LEFT endpoint and held over the interval.
    Returns Y with shape (len(t_grid), n).
    """
    A = _asarray_2d(A)
    B = np.array(B, dtype=float)
    y = np.array(y0, dtype=float).reshape(-1)
    Y = np.zeros((len(t_grid), y.size))
    Y[0] = y
    for k in range(1, len(t_grid)):
        dt = float(t_grid[k] - t_grid[k - 1])
        Ad, Bd = zoh_discretize(A, B, dt)
        u_k = np.array(u_of_t(float(t_grid[k - 1]))).reshape(-1)
        y = Ad @ y + Bd @ u_k
        Y[k] = y
    return Y
