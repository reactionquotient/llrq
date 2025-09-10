# pip install cvxpy numpy
import numpy as np
import cvxpy as cp


def make_A(N, c_eq, eps=1e-9):
    """
    Build A = N^T diag(1/c_eq) N. If A is only PSD (redundant reactions),
    add a tiny ridge 'eps' to make it numerically PD so that A^{-1/2} exists.
    """
    c_eq = np.asarray(c_eq).reshape(-1)
    Wc = np.diag(1.0 / c_eq)  # diag(1/c_eq)
    A = N.T @ Wc @ N
    # Symmetrize & ridge for numerical stability
    A = 0.5 * (A + A.T)
    A += eps * np.eye(A.shape[0])
    return A


def sqrt_and_invsqrt(A, rtol=1e-12):
    """
    Return (A^{1/2}, A^{-1/2}) from an eigen-decomposition. Assumes A SPD.
    """
    lam, U = np.linalg.eigh(A)
    lam = np.maximum(lam, 0.0)
    if lam.min() <= 0:
        raise ValueError("A must be SPD; try adding a small ridge.")
    A_sqrt = (U * np.sqrt(lam)) @ U.T
    A_invsqrt = (U * (1.0 / np.sqrt(lam))) @ U.T
    return A_sqrt, A_invsqrt


def K_bounds_constraints(
    K,
    N,
    c_eq,
    g_diag_min=None,  # vector lower bounds on diag(G) (optional)
    g_diag_max=None,  # vector upper bounds on diag(G) (diffusion caps)
    gamma_min=None,  # scalar lower bound on lambda_min(G)  (optional)
    gamma_max=None,  # scalar upper bound on lambda_max(G)  (e.g., max diag cap)
    eps=1e-9,
):
    """
    Build convex constraints on K that ensure:  exists G ⪰ 0 with K = A G
    and the chosen physical bounds on G (diagonal caps, spectral caps).

    Returns:
      constraints (list of cvxpy constraints), and the helper symmetric variable W.
    """
    # Constants
    A = make_A(N, c_eq, eps=eps)
    A = 0.5 * (A + A.T)  # ensure exact symmetry
    A_sqrt, A_invsqrt = sqrt_and_invsqrt(A)

    r = A.shape[0]
    # Helper symmetric variable W = A^{-1/2} K A^{1/2}  (must be PSD & symmetric)
    W = cp.Variable((r, r), symmetric=True)

    constraints = []

    # Linear relation tying K to W :  W == A^{-1/2} K A^{1/2}
    constraints += [W == A_invsqrt @ K @ A_sqrt]

    # PSD (existence of some G ⪰ 0):  W ⪰ 0
    constraints += [W >> 0]

    # ----- Optional: spectral bounds on G via LMIs on W -----
    # If G ⪯ γ_max I   =>  A^{1/2} G A^{1/2} ⪯ γ_max A   =>  W ⪯ γ_max A
    if gamma_max is not None:
        constraints += [W << gamma_max * A]

    # If γ_min I ⪯ G   =>  γ_min A ⪯ A^{1/2} G A^{1/2}   =>  γ_min A ⪯ W
    if gamma_min is not None and gamma_min > 0:
        constraints += [gamma_min * A << W]

    # ----- Optional: elementwise diagonal bounds on G -----
    # G = A^{-1/2} W A^{-1/2}.  Diagonal bounds are linear in W:
    if g_diag_max is not None:
        g_diag_max = np.asarray(g_diag_max).reshape(-1)
        assert g_diag_max.shape[0] == r
        constraints += [cp.diag(A_invsqrt @ W @ A_invsqrt) <= g_diag_max]

        # a safe spectral cap consistent with diag caps (conservative):
        # lambda_max(G) <= max_i g_diag_max[i]
        if gamma_max is None:
            constraints += [W << float(np.max(g_diag_max)) * A]

    if g_diag_min is not None:
        g_diag_min = np.asarray(g_diag_min).reshape(-1)
        assert g_diag_min.shape[0] == r
        constraints += [cp.diag(A_invsqrt @ W @ A_invsqrt) >= g_diag_min]

        # a safe spectral floor consistent with diag mins (conservative):
        if gamma_min is None and np.min(g_diag_min) > 0:
            constraints += [float(np.min(g_diag_min)) * A << W]

    # Return constraints and W (handy if you want to regularize/penalize it)
    return constraints, W, A


# ----------------------- Example usage -----------------------
if __name__ == "__main__":
    # Toy sizes (replace with your real N and c_eq):
    m, r = 12, 8  # m species, r reactions
    np.random.seed(0)
    # Example stoichiometry: sparse-ish integers
    N = np.random.randint(-2, 3, size=(m, r)).astype(float)
    # Ensure some reactant/product structure (optional polish)
    # Equilibrium concentrations (in M); pick reasonable magnitudes
    c_eq = 1e-4 + 1e-3 * np.random.rand(m)

    # Physical caps from diffusion & enzyme abundance (per-reaction):
    #   g_diag_max[i] ~ k_on[i] * [E]_i * (stoich thermo factor)
    g_diag_max = 10.0 * np.ones(r)  # s^-1 (example cap)
    g_diag_min = 0.0 * np.ones(r)  # allow zero if enzyme can be "off"
    # Scalar spectral caps (optional; conservative from g_diag_max/min):
    gamma_max = float(np.max(g_diag_max))
    gamma_min = 0.0

    # Decision variable K
    K = cp.Variable((r, r))

    # Build constraints that encode: ∃ G ⪰ 0 with K = A G and bounds on G
    constraints, W, A = K_bounds_constraints(
        K, N, c_eq, g_diag_min=g_diag_min, g_diag_max=g_diag_max, gamma_min=gamma_min, gamma_max=gamma_max, eps=1e-9
    )

    # (Optional) Add any additional structural priors on K (e.g., sparsity mask)
    # Example: encourage small off-diagonals via objective
    obj = cp.Minimize(1e-3 * cp.norm1(K - cp.diag(cp.diag(K))))

    # Solve a dummy feasibility/regularization problem
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)  # Uncomment to run if CVXPY/SCS is available
    print("Status:", prob.status)
    print("Example K (not unique):\n", K.value)
