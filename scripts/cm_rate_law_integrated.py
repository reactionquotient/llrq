# Re-run generator (state was reset)

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
import json

OUT_DIR = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CMParams:
    u: float
    kM_A: float
    kM_B: float
    kM_C: float
    k_plus: float
    k_minus: float


def rate_cm(a, b, c, p: CMParams):
    a_p = a / p.kM_A
    b_p = b / p.kM_B
    c_p = c / p.kM_C
    num = p.k_plus * a_p * b_p - p.k_minus * (c_p**2)
    den = (1.0 + a_p) * (1.0 + b_p) + (1.0 + c_p) ** 2 - 1.0
    return p.u * num / den


def rhs(t, y, p: CMParams):
    A, B, C = y
    v = rate_cm(A, B, C, p)
    return [-v, -v, 2.0 * v]


def simulate(t_span, y0, params: CMParams, t_eval=None, rtol=1e-8, atol=1e-10):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(lambda t, y: rhs(t, y, params), t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol)
    A, B, C = sol.y
    v = rate_cm(A, B, C, params)
    return sol.t, A, B, C, v


def loguniform(rng, low, high):
    return np.exp(rng.uniform(np.log(low), np.log(high)))


def sample_params(seed=20250912):
    rng = np.random.default_rng(seed)
    kM_A = loguniform(rng, 1e-3, 1e1)
    kM_B = loguniform(rng, 1e-3, 1e1)
    kM_C = loguniform(rng, 1e-3, 1e1)
    Keq = loguniform(rng, 1e-3, 1e3)
    kV = loguniform(rng, 1e-1, 1e3)
    u = loguniform(rng, 1e-3, 1.0)
    sum_n_ln_kM = (-np.log(kM_A)) + (-np.log(kM_B)) + (2 * np.log(kM_C))
    ln_kplus = np.log(kV) + 0.5 * (np.log(Keq) - sum_n_ln_kM)
    ln_kminus = np.log(kV) - 0.5 * (np.log(Keq) - sum_n_ln_kM)
    k_plus, k_minus = np.exp(ln_kplus), np.exp(ln_kminus)
    return CMParams(float(u), float(kM_A), float(kM_B), float(kM_C), float(k_plus), float(k_minus))


# Parameters & experiments
params = sample_params()

# Shorter time range to focus on dynamics (equilibrium reached by t~10)
t_eval = np.linspace(0.0, 2.5, 200)
expts = {
    "forward": dict(y0=[1.0 * params.kM_A, 1.0 * params.kM_B, 0.0]),
    "reverse": dict(y0=[0.05 * params.kM_A, 0.05 * params.kM_B, 1.0 * params.kM_C]),
}

csv_paths = {}
for name, spec in expts.items():
    t, A, B, C, v = simulate((t_eval[0], t_eval[-1]), spec["y0"], params, t_eval=t_eval)
    df = pd.DataFrame({"t": t, "A": A, "B": B, "C": C, "v": v, "exp_name": name})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"cm_timeseries_{name}_{ts}.csv"
    df.to_csv(path, index=False)
    csv_paths[name] = path.as_posix()
    print(f"\nPreview: {name} timeseries")
    print(df.head(15))

# Quick plot for forward experiment
t, A, B, C, v = simulate((t_eval[0], t_eval[-1]), expts["forward"]["y0"], params, t_eval=t_eval)
plt.figure()
plt.plot(t, A, label="A")
plt.plot(t, B, label="B")
plt.plot(t, C, label="C")
plt.xlabel("time")
plt.ylabel("concentration")
plt.title("A + B <-> 2C (forward experiment)")
plt.legend()
plt.tight_layout()
plt.show()

param_path = OUT_DIR / "cm_params_timeseries.json"
with open(param_path, "w") as f:
    json.dump(params.__dict__, f, indent=2)

csv_paths, param_path.as_posix()

# ============================================================================
# LLRQ FITTING SECTION
# ============================================================================


def compute_reaction_quotient(A, B, C):
    """Compute Q = C^2 / (A * B) for reaction A + B <-> 2C"""
    eps = 1e-10
    # Handle case where C is zero (use small positive value for Q)
    if np.isscalar(C):
        if C < eps:
            return eps
        return C**2 / ((A + eps) * (B + eps))
    else:
        # Vectorized version
        Q = C**2 / ((A + eps) * (B + eps))
        Q[C < eps] = eps
        return Q


def llrq_model(t, ln_Q0, K, ln_Keq):
    """Simple LLRQ model: ln(Q(t)) = ln(Keq) + (ln(Q0) - ln(Keq)) * exp(-K*t)"""
    return ln_Keq + (ln_Q0 - ln_Keq) * np.exp(-K * t)


def fit_llrq(t, A, B, C):
    """Fit LLRQ model: d/dt ln(Q) = -K * (ln(Q) - ln(Keq))

    Parameters to fit:
    - K: relaxation rate
    - ln_Keq: log equilibrium constant (since true Keq unknown)

    Returns:
    - K_fit: fitted relaxation rate
    - Keq_fit: fitted equilibrium constant
    - ln_Q_fit: fitted ln(Q) values
    - r_squared: R^2 goodness of fit
    """
    # Compute reaction quotients (now handles C=0 properly)
    Q = compute_reaction_quotient(A, B, C)
    ln_Q = np.log(Q)

    # Initial guesses
    ln_Q0 = ln_Q[0]
    ln_Keq_guess = ln_Q[-1]  # Final value as equilibrium estimate
    K_guess = 0.1

    # Define model for curve_fit
    def model(t, K, ln_Keq):
        return llrq_model(t, ln_Q0, K, ln_Keq)

    try:
        # Fit the model using ALL data points
        popt, pcov = curve_fit(model, t, ln_Q, p0=[K_guess, ln_Keq_guess], maxfev=5000)
        K_fit, ln_Keq_fit = popt

        # Compute fitted values
        ln_Q_fit = model(t, K_fit, ln_Keq_fit)

        # Compute R-squared
        ss_res = np.sum((ln_Q - ln_Q_fit) ** 2)
        ss_tot = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return K_fit, np.exp(ln_Keq_fit), ln_Q_fit, r_squared
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None, None


# Fit LLRQ to forward experiment
print("\n" + "=" * 60)
print("LLRQ FITTING ANALYSIS")
print("=" * 60)

t_forward, A_forward, B_forward, C_forward, v_forward = simulate(
    (t_eval[0], t_eval[-1]), expts["forward"]["y0"], params, t_eval=t_eval
)

K_fit, Keq_fit, ln_Q_fit, r2 = fit_llrq(t_forward, A_forward, B_forward, C_forward)

if K_fit is not None:
    print(f"\nForward experiment LLRQ fit:")
    print(f"  Fitted K (relaxation rate): {K_fit:.4f}")
    print(f"  Fitted Keq: {Keq_fit:.4e}")
    print(f"  R-squared: {r2:.4f}")

    # Compute true Keq from CM parameters
    true_Keq = (params.k_plus / params.k_minus) * (params.kM_C**2) / (params.kM_A * params.kM_B)
    print(f"  True Keq (from CM params): {true_Keq:.4e}")
    print(f"  Keq ratio (fitted/true): {Keq_fit/true_Keq:.4f}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Concentrations over time
    ax = axes[0, 0]
    ax.plot(t_forward, A_forward, "b-", label="A", linewidth=2)
    ax.plot(t_forward, B_forward, "g-", label="B", linewidth=2)
    ax.plot(t_forward, C_forward, "r-", label="C", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title("CM Dynamics: Concentrations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Reaction quotient Q over time
    ax = axes[0, 1]
    Q = compute_reaction_quotient(A_forward, B_forward, C_forward)
    ax.plot(t_forward, Q, "k-", linewidth=2, label="Q(t)")
    ax.axhline(y=Keq_fit, color="r", linestyle="--", label=f"Fitted Keq = {Keq_fit:.2e}")
    ax.axhline(y=true_Keq, color="b", linestyle=":", label=f"True Keq = {true_Keq:.2e}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reaction Quotient Q")
    ax.set_title("Reaction Quotient Evolution")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: ln(Q) and LLRQ fit
    ax = axes[1, 0]
    ln_Q = np.log(Q)
    ax.plot(t_forward, ln_Q, "ko", markersize=3, alpha=0.5, label="Data")
    ax.plot(t_forward, ln_Q_fit, "r-", linewidth=2, label=f"LLRQ fit (K={K_fit:.3f})")
    ax.set_xlabel("Time")
    ax.set_ylabel("ln(Q)")
    ax.set_title(f"LLRQ Fit: ln(Q) vs Time (R² = {r2:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals = ln_Q - ln_Q_fit
    ax.plot(t_forward, residuals, "b-", linewidth=1, alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual (ln(Q) - fit)")
    ax.set_title("LLRQ Fit Residuals")
    ax.grid(True, alpha=0.3)

    # Add RMSE to residual plot
    rmse = np.sqrt(np.mean(residuals**2))
    ax.text(
        0.02,
        0.98,
        f"RMSE = {rmse:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle("CM Dynamics with LLRQ Fitting Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save figure
    fig_path = OUT_DIR / "cm_llrq_fitting_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved analysis plot to {fig_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The LLRQ model fits the ln(Q) evolution with a single exponential")
    print(f"relaxation rate K = {K_fit:.4f} and equilibrium constant Keq = {Keq_fit:.4e}.")
    print(f"The fit quality (R² = {r2:.4f}) indicates how well the CM dynamics")
    print("follow LLRQ-like behavior for the reaction quotient evolution.")
else:
    print("LLRQ fitting failed.")
