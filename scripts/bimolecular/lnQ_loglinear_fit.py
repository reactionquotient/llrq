import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence
from scipy.optimize import lsq_linear


@dataclass
class LogLinearFit:
    lambdas: np.ndarray  # (m,) grid of rates used
    w: np.ndarray  # (m,) nonnegative weights (post-pruning)
    b: float  # intercept = ln K_eq
    s: int  # direction sign (+1 if y decreases, -1 if increases)
    active_idx: np.ndarray  # indices of active lambdas after pruning
    r2: float  # R^2 of the fit on provided samples
    y_hat: np.ndarray  # fitted y on the provided samples
    # One valid realization of the hidden-state model:
    K: np.ndarray  # (r,r) diag matrix with active lambdas
    c: np.ndarray  # (r,) output row-vector for x=c^T z
    z0: np.ndarray  # (r,) initial condition so that x(t)=c^T exp(-Kt) z0

    # convenience:
    def K_eq(self) -> float:
        return float(np.exp(self.b))


def _second_diff_matrix(m: int) -> np.ndarray:
    """(m-2) x m second-difference operator"""
    if m < 3:
        return np.zeros((0, m))
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _design_matrix(t: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    # Use time relative to the first sample for numerical stability.
    t0 = float(t[0])
    return np.exp(-np.outer(t - t0, lambdas))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_lnQ_loglinear(
    t: Sequence[float],
    y: Sequence[float],
    m: int = 60,
    lambda_min: Optional[float] = None,
    lambda_max: Optional[float] = None,
    direction: Optional[int] = None,
    alpha: float = 1e-3,  # smoothing on second-differences of w
    beta: float = 1e-6,  # small penalty on slow rates: sum (w_i / lambda_i)
    prune_tol: float = 1e-10,  # drop tiny weights
) -> LogLinearFit:
    """
    Fit y(t) = b + s * sum_i w_i exp(-lambda_i t) with w_i >= 0.
      - t, y: data arrays (monotone y is assumed but not enforced)
      - m: number of grid rates
      - lambda_min / lambda_max: if None, chosen from data window
      - direction: +1 for decreasing y, -1 for increasing y; if None, inferred
      - alpha: Tikhonov smoothing on second differences Dw
      - beta: small penalty to discourage ultra-slow weight (helps separate b)
      - prune_tol: zero-out tiny weights
    Returns a LogLinearFit containing one valid (K, c, z0) realization.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    assert t.ndim == 1 and y.ndim == 1 and t.size == y.size and t.size >= 3, "t and y must be 1D of equal length >= 3"

    # Direction s: +1 if y decreases over window, -1 if increases
    if direction is None:
        s = 1 if (y[0] - y[-1]) > 0 else -1
        if np.isclose(y[0], y[-1]):
            s = 1  # default
    else:
        s = int(np.sign(direction)) or 1

    # Time scales -> rate grid
    T = t[-1] - t[0]
    if lambda_min is None:
        lambda_min = max(1e-12, 0.5 / max(T, 1e-12))  # slowest time ~ 2T
    if lambda_max is None:
        dt_unique = np.diff(np.unique(t))
        dt = np.min(dt_unique) if dt_unique.size > 0 else max(T / 100, 1e-6)
        lambda_max = max(10.0 / max(dt, 1e-12), 50.0 / max(T, 1e-12))  # fast decade(s)

    lambdas = np.geomspace(lambda_min, lambda_max, m)
    Phi = _design_matrix(t, lambdas)  # n x m
    A_data = s * Phi
    ones = np.ones((t.size, 1))

    # Regularization:
    D = _second_diff_matrix(m)  # (m-2) x m
    S = np.diag(1.0 / lambdas)  # m x m
    A_reg_smooth = np.sqrt(alpha) * D  # (m-2) x m
    A_reg_slow = np.sqrt(beta) * S  # m x m

    # Build augmented LS: [A_data  1; sqrt(alpha)D 0; sqrt(beta)S 0] [w; b] ~ [y; 0; 0]
    A_top = np.hstack([A_data, ones])  # n x (m+1)
    A_mid = np.hstack([A_reg_smooth, np.zeros((A_reg_smooth.shape[0], 1))])
    A_bot = np.hstack([A_reg_slow, np.zeros((A_reg_slow.shape[0], 1))])
    A_aug = np.vstack([A_top, A_mid, A_bot])
    y_aug = np.concatenate([y, np.zeros(A_mid.shape[0] + A_bot.shape[0])])

    # Solve bounded LS: w >= 0, b free.
    lb = np.concatenate([np.zeros(m), [-np.inf]])
    ub = np.concatenate([np.full(m, np.inf), [np.inf]])
    sol = lsq_linear(A_aug, y_aug, bounds=(lb, ub), lsmr_tol="auto", verbose=0)

    theta = sol.x
    w = theta[:m]
    b = float(theta[-1])

    # Prune tiny weights to get a compact realization
    active = np.where(w > prune_tol * max(1.0, np.max(w)))[0]
    if active.size == 0:
        # force at least the largest weight to remain
        active = np.array([int(np.argmax(w))])

    w_pruned = w[active]
    lambdas_pruned = lambdas[active]

    # Recompute fit on provided samples (no regularizers in prediction)
    y_hat = b + s * (_design_matrix(t, lambdas_pruned) @ w_pruned)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # One valid (K,c,z0) realization:
    K = np.diag(lambdas_pruned)
    c = np.ones(active.size)  # x(t) = c^T z(t)
    z0 = s * w_pruned  # so that x(t) = sum w_i e^{-lambda_i t}

    return LogLinearFit(
        lambdas=lambdas,
        w=w,
        b=b,
        s=s,
        active_idx=active,
        r2=float(r2),
        y_hat=y_hat,
        K=K,
        c=c,
        z0=z0,
    )


# ---------- Demo ----------
def _demo():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = 400
    t = np.linspace(0.0, 10.0, n)
    # synth: decreasing ln Q = b + sum w_i e^{-lambda_i t}
    true_b = -2.0
    true_lam = np.array([0.3, 2.0])
    true_w = np.array([0.8, 0.2])
    y_clean = true_b + (np.exp(-np.outer(t, true_lam)) @ true_w)
    y = y_clean + 0.01 * rng.normal(size=n)

    fit = fit_lnQ_loglinear(t, y, m=80, alpha=3e-3, beta=1e-6, prune_tol=1e-6)

    print("---- Fit summary ----")
    print(f"R^2         : {fit.r2:.6f}")
    print(f"ln K_eq (b) : {fit.b:.6f}   -> K_eq = {np.exp(fit.b):.6f}")
    print(f"Active modes: {fit.active_idx.size} / {fit.lambdas.size}")
    print(f"Active lambdas:", fit.K.diagonal())

    plt.figure(figsize=(9, 5))
    plt.plot(t, y, "o", ms=3, label="data")
    plt.plot(t, fit.y_hat, "-", label="fit")
    plt.axhline(fit.b, linestyle=":", label="b = ln K_eq")
    plt.xlabel("t")
    plt.ylabel("ln Q(t)")
    plt.title("Log-linear mixture fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    _demo()
