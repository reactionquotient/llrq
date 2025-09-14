# lnQ_loglinear_fit_cvx.py
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

try:
    import cvxpy as cp
except Exception as e:
    raise RuntimeError("cvxpy is required. Install with `pip install cvxpy osqp`.") from e


@dataclass
class LogLinearFit:
    lambdas: np.ndarray  # (m,) grid of rates used
    w: np.ndarray  # (m,) nonnegative weights
    b: float  # intercept = ln K_eq
    s: int  # +1 if y decreases, -1 if increases
    active_idx: np.ndarray  # indices of active (non-tiny) weights
    r2: float  # R^2 on the provided samples
    y_hat: np.ndarray  # fitted y at sample times
    # one valid realization:
    K: np.ndarray  # (r,r) diag with active lambdas
    c: np.ndarray  # (r,) all ones
    z0: np.ndarray  # (r,) so x(t)=c^T e^{-Kt} z0

    def K_eq(self) -> float:
        return float(np.exp(self.b))


def _second_diff_matrix(m: int) -> np.ndarray:
    # (m-2) x m second-difference operator
    if m < 3:
        return np.zeros((0, m))
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _design_matrix(t: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    # stabilize by subtracting t[0]
    t0 = float(t[0])
    return np.exp(-np.outer(t - t0, lambdas))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_lnQ_loglinear_cvx(
    t: Sequence[float],
    y: Sequence[float],
    m: int = 60,
    lambda_min: Optional[float] = None,
    lambda_max: Optional[float] = None,
    direction: Optional[int] = None,
    alpha: float = 1e-3,  # smoothness on Dw (Tikhonov)
    beta: float = 1e-6,  # slow-rate penalty ∑ w_i / λ_i (helps separate b)
    gamma_l1: float = 1e-4,  # L1 penalty on w for sparsity (set 0 to disable)
    cardinality: Optional[int] = None,  # max number of nonzero weights (set None to disable)
    prune_tol: float = 1e-10,  # drop tiny weights (relative to max)
    solver: str = "OSQP",  # "OSQP" (QP) or "SCS"
    solver_kwargs: Optional[dict] = None,
) -> LogLinearFit:
    """
    Fit y(t) ≈ b + s ∑_i w_i e^{-λ_i t}, with w_i ≥ 0 (nonnegative mixture).
      - direction: +1 for decreasing y over window, -1 for increasing; if None, inferred
      - alpha:  ||D w||_2^2 smoothing on weights (promotes smooth spectrum)
      - beta:   ∑ (w_i / λ_i) slow-end penalty so b doesn't get replaced by ultra-slow mass
      - gamma_l1: γ ||w||_1 sparsity (promotes a few active rates)
      - prune_tol: remove weights ≤ prune_tol * max(w)
    Returns one valid diagonal realization (K, c, z0) and metrics.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    assert t.ndim == 1 and y.ndim == 1 and t.size == y.size and t.size >= 3, "t and y must be 1D of equal length ≥ 3"

    # Direction sign s
    if direction is None:
        s = 1 if (y[0] - y[-1]) > 0 else -1
        if np.isclose(y[0], y[-1]):  # tie-breaker
            s = 1
    else:
        s = int(np.sign(direction)) or 1

    # Log-spaced rate grid based on window
    T = max(t[-1] - t[0], 1e-12)
    if lambda_min is None:
        lambda_min = max(1e-12, 0.5 / T)  # slowest time ~ 2T
    if lambda_max is None:
        dtu = np.diff(np.unique(t))
        dt = np.min(dtu) if dtu.size else max(T / 100.0, 1e-6)
        lambda_max = max(10.0 / dt, 50.0 / T)  # a couple fast decades

    lambdas = np.geomspace(lambda_min, lambda_max, int(m))
    Phi = _design_matrix(t, lambdas)  # n x m
    A = s * Phi
    ones = np.ones((t.size, 1))

    # Regularizers
    D = _second_diff_matrix(m)  # (m-2) x m
    invlam = 1.0 / lambdas  # (m,)

    # CVXPY variables
    w = cp.Variable(m, nonneg=True)
    b = cp.Variable()

    # Objective terms
    residual = A @ w + b * ones.squeeze() - y
    obj = cp.sum_squares(residual)
    constraints = []

    if alpha > 0 and D.size > 0:
        obj += alpha * cp.sum_squares(D @ w)
    if beta > 0:
        obj += beta * cp.sum(cp.multiply(invlam, w))
    if gamma_l1 > 1:  # don't need norm1 because w >= 0
        obj += gamma_l1 * cp.sum(w)
    if cardinality is not None and cardinality > 0:
        # add cardinality constraint to help with sparsity
        binary_var = cp.Variable(m, boolean=True)  # big-M binary
        constraints.append(w <= binary_var * 1000)  # big-M
        constraints.append(cp.sum(binary_var) <= cardinality)

    prob = cp.Problem(cp.Minimize(obj), constraints)  # only w >= 0 constraint is in variable domain

    # Solve
    if solver_kwargs is None:
        solver_kwargs = {}
    try:
        prob.solve(solver=solver, **solver_kwargs)
    except Exception:
        # fallback
        prob.solve(solver="SCS", **solver_kwargs)

    if w.value is None or b.value is None:
        raise RuntimeError("CVXPY failed to find a solution.")

    w_val = np.asarray(w.value, float).clip(min=0.0)
    b_val = float(b.value)

    # Prune small weights
    wmax = max(1.0, float(np.max(w_val)))
    active = np.where(w_val > prune_tol * wmax)[0]
    if active.size == 0:
        active = np.array([int(np.argmax(w_val))])

    w_pruned = w_val[active]
    lambdas_pruned = lambdas[active]

    # Fitted curve at sample times (using only active modes for reporting)
    y_hat = b_val + s * (_design_matrix(t, lambdas_pruned) @ w_pruned)

    # Metrics
    r2 = _r2(y, y_hat)

    # One valid realization
    K = np.diag(lambdas_pruned)
    c = np.ones(active.size)
    z0 = s * w_pruned

    return LogLinearFit(
        lambdas=lambdas,
        w=w_val,
        b=b_val,
        s=s,
        active_idx=active,
        r2=float(r2),
        y_hat=y_hat,
        K=K,
        c=c,
        z0=z0,
    )


# ---- Demo ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    n = 400
    t = np.linspace(0.0, 10.0, n)
    # ground truth: b + sum w_i exp(-lambda_i t)
    true_b = -2.0
    true_lam = np.array([0.25, 1.3, 4.0])
    true_w = np.array([0.9, 0.25, 0.07])
    y_clean = true_b + (np.exp(-np.outer(t, true_lam)) @ true_w)
    y = y_clean + 0.01 * rng.normal(size=n)

    fit = fit_lnQ_loglinear_cvx(
        t,
        y,
        m=100,
        alpha=3e-3,
        beta=1e-6,
        gamma_l1=2e-3,  # turn this up ↗ for sparser weights
        prune_tol=1e-5,
        solver="OSQP",
        solver_kwargs=dict(eps_abs=1e-6, eps_rel=1e-6, max_iter=20000),
    )

    print("R^2:", fit.r2)
    print("ln K_eq (b):", fit.b, "  K_eq:", np.exp(fit.b))
    print(f"Active modes: {fit.active_idx.size} / {fit.lambdas.size}")
    print("Active lambdas:", np.round(np.diag(fit.K), 6))

    plt.figure(figsize=(9, 5))
    plt.plot(t, y, "o", ms=3, label="data")
    plt.plot(t, fit.y_hat, "-", lw=2, label="cvx fit")
    plt.axhline(fit.b, linestyle=":", label="b = ln K_eq")
    plt.xlabel("t")
    plt.ylabel("ln Q(t)")
    plt.title("CVX fit with L1 sparsity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
