import numpy as np
from scipy.linalg import expm


def step_const(A, y0, u0, t):
    """Exact solution for constant u(t)=u0 on [0,t]: y(t)=e^{At}y0 + A^{-1}(e^{At}-I)u0.
    Implemented without explicit inverse via solve(A, ...)."""
    A = np.array(A, float)
    y0 = np.array(y0, float)
    u0 = np.array(u0, float)
    Et = expm(A * t)
    rhs = Et - np.eye(A.shape[0])
    # Solve A X = rhs for X, then y = Et y0 + X u0
    X = np.linalg.solve(A, rhs)
    return Et @ y0 + X @ u0


def zoh_discretize(A, dt):
    """Exact ZOH discretization: (A_d, B_d) with u held constant on [kΔ,(k+1)Δ)."""
    A = np.array(A, float)
    n = A.shape[0]
    Ad = expm(A * dt)
    X = np.linalg.solve(A, Ad - np.eye(n))  # A X = Ad - I
    Bd = X  # because input matrix is identity in reduced LLRQ
    return Ad, Bd


def foh_discretize(A, dt):
    """Exact FOH (first-order hold) for identity input matrix.
    y_{k+1} = A_d y_k + B0 u_k + B1 u_{k+1}.
    Uses a 3x3 block Van-Loan exponential."""
    A = np.array(A, float)
    n = A.shape[0]
    Z = np.zeros_like(A)
    # Build block matrix:
    # M = [[A, I, 0],
    #      [0, 0, I],
    #      [0, 0, 0]]  (all n×n blocks)
    M = np.block([[A, np.eye(n), Z], [Z, Z, np.eye(n)], [Z, Z, Z]])
    E = expm(M * dt)
    # Partition E:
    # E = [[A_d,  B0,  *],
    #      [0,    I,   B1],
    #      [0,    0,    I ]]
    Ad = E[:n, :n]
    B0 = E[:n, n : 2 * n]
    B1 = E[n : 2 * n, 2 * n : 3 * n]  # Note: this block sits in the (2,3) position
    return Ad, B0, B1


def convolve_lti(A, t_grid, u_of_t, y0):
    """Generic convolution y(t_k) = e^{At_k} y0 + ∫_0^{t_k} e^{A(t_k-s)} u(s) ds
    using piecewise-constant ZOH between samples of u_of_t(t).
    - A: (n,n)
    - t_grid: increasing 1D array [t0=0, t1, ..., tN]
    - u_of_t: callable returning u(t) shape (n,)
    - y0: initial condition (n,)
    Returns: Y (N+1,n)
    """
    A = np.array(A, float)
    y = np.array(y0, float)
    Y = np.zeros((len(t_grid), len(y)))
    Y[0] = y
    for k in range(1, len(t_grid)):
        dt = t_grid[k] - t_grid[k - 1]
        Ad, Bd = zoh_discretize(A, dt)
        u_km = u_of_t(t_grid[k - 1])  # ZOH: hold previous value
        y = Ad @ y + Bd @ u_km
        Y[k] = y
    return Y
