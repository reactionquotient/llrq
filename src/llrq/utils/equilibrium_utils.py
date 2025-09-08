import numpy as np
from numpy.linalg import lstsq, norm
from scipy.linalg import svd


def _nullspace(M, rtol=1e-12):
    """Right nullspace of M (columns span ker(M))."""
    U, s, Vt = svd(M, full_matrices=True)
    rank = (s > rtol * s.max()).sum()
    return Vt[rank:].T  # shape: cols = nullity


def _left_nullspace(M, rtol=1e-12):
    """Left nullspace of M: columns L with L^T M = 0 (i.e., ker(M^T))."""
    return _nullspace(M.T, rtol)


def equilibrium_mass_action_detailed_balance(A, B, kf, kr, c0, tol=1e-10, max_iter=100):
    """
    Compute equilibrium concentrations for a reversible mass-action network.

    Parameters
    ----------
    A, B : (n, r) arrays
        Stoichiometry of reactants/products for each reaction (columns).
    kf, kr : (r,) arrays
        Forward and reverse rate constants (>0).
    c0 : (n,) array
        Initial concentrations (used to compute conserved totals).
    tol : float
        Convergence tolerance on the conservation residual.
    max_iter : int
        Max Newton iterations for solving conservation equations.

    Returns
    -------
    c_star : (n,) array
        Positive equilibrium concentrations.
    info : dict
        Diagnostic information.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    kf = np.asarray(kf, dtype=float)
    kr = np.asarray(kr, dtype=float)
    c0 = np.asarray(c0, dtype=float)

    n, r = A.shape
    assert B.shape == (n, r), "A and B must be n×r."
    assert kf.shape == (r,) and kr.shape == (r,), "kf, kr must be length r."
    if not np.all(kf > 0) or not np.all(kr > 0):
        raise ValueError("This solver assumes all reactions are reversible (kf, kr > 0).")

    # Net stoichiometry and equilibrium constants
    N = B - A  # shape (n, r)
    lnK = np.log(kf / kr)  # shape (r,)

    # 1) Solve N^T x = ln K for x = ln c (particular solution).
    #    (This encodes N^T ln c = ln K  <=>  Q(c)=K, i.e., detailed balance. :contentReference[oaicite:1]{index=1})
    #    Use least-squares, then check consistency (Wegscheider conditions).
    NT = N.T  # shape (r, n)
    x_p, *_ = lstsq(NT, lnK, rcond=None)  # particular solution
    resid = norm(NT @ x_p - lnK)

    # If residual isn't tiny, the supplied K's violate thermodynamic consistency.
    if resid > 1e-8:
        raise ValueError(
            f"Inconsistent equilibrium constants (||N^T x - lnK||={resid:.2e}). "
            "Check Wegscheider conditions / rate constants."
        )

    # 2) Add the general solution of N^T x = lnK: x = x_p + Z y,  where columns of Z span ker(N^T).
    Z = _nullspace(NT)  # shape (n, p), p = n - rank(N)
    p = Z.shape[1]

    # If p == 0, no conserved moieties: unique ln c.
    if p == 0:
        c_star = np.exp(x_p)
        return c_star, {"iterations": 0, "conservation_residual": 0.0, "p": 0}

    # 3) Enforce conservation laws using initial totals.
    #    Columns of L span the left nullspace of N: L^T N = 0 ⇒ L^T c is conserved.
    L = _left_nullspace(N)  # shape (n, p)
    # Project to the same subspace dimension as Z; bases may differ by invertible p×p transform.
    # We’ll use the L we computed to write constraints L^T exp(x_p + Z y) = L^T c0.
    m = L.T @ c0

    # Solve g(y) = L^T exp(x_p + Z y) - m = 0 for y ∈ R^p via damped Newton in y.
    y = np.zeros(p)
    for it in range(max_iter):
        x = x_p + Z @ y
        c = np.exp(x)
        g = L.T @ c - m  # shape (p,)
        g_norm = norm(g, ord=2)
        if g_norm < tol:
            return c, {"iterations": it, "conservation_residual": float(g_norm), "p": p}

        # Jacobian: J = d/dy [L^T exp(x_p + Z y)] = L^T diag(c) Z  (p×p)
        J = L.T @ (c[:, None] * Z)

        # Solve J Δy = -g  (use least squares in case J is ill-conditioned)
        try:
            dy, *_ = lstsq(J, -g, rcond=None)
        except Exception:
            # Very rare; fall back to pseudo-step
            dy = -g

        # Backtracking line search to ensure progress & positivity (positivity is automatic in log-space)
        step = 1.0
        for _ in range(20):
            y_trial = y + step * dy
            x_trial = x_p + Z @ y_trial
            c_trial = np.exp(x_trial)
            g_trial = L.T @ c_trial - m
            if norm(g_trial) < g_norm * (1 - 1e-4 * step):
                y = y_trial
                break
            step *= 0.5
        else:
            # Could not improve; stop and report best found.
            return c, {"iterations": it + 1, "conservation_residual": float(g_norm), "p": p}

    # If we exit loop without converging:
    return np.exp(x_p + Z @ y), {
        "iterations": max_iter,
        "conservation_residual": float(norm(L.T @ np.exp(x_p + Z @ y) - m)),
        "p": p,
        "warning": "Max iterations reached",
    }


# ---------- Convenience: build A, B from a reaction list ----------
def build_stoichiometry(species, reactions):
    """
    species: list of species names, length n.
    reactions: list of dicts, each like:
        {"reactants": {"A": 1, "B": 2}, "products": {"C": 1}}
    Returns A, B (n×r) following species order.
    """
    n = len(species)
    r = len(reactions)
    idx = {s: i for i, s in enumerate(species)}
    A = np.zeros((n, r))
    B = np.zeros((n, r))
    for j, rxn in enumerate(reactions):
        for s, coeff in rxn.get("reactants", {}).items():
            A[idx[s], j] = coeff
        for s, coeff in rxn.get("products", {}).items():
            B[idx[s], j] = coeff
    return A, B
