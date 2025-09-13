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
import sys
import os

# Add LLRQ package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import llrq

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
    "forward": dict(y0=[1.0 * params.kM_A, 1.0 * params.kM_B, 0.1 * params.kM_C]),
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
# LLRQ FITTING AND COMPARISON SECTION
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


def fit_llrq_least_squares(t, A, B, C, true_Keq):
    """Fit LLRQ model using least squares on integrated form.

    The LLRQ model is: ln(Q(t)) = ln(Keq) + (ln(Q0) - ln(Keq)) * exp(-K*t)
    We minimize the squared error between observed ln(Q) and model prediction.

    Returns:
    - K_fit: fitted relaxation rate
    - Keq_used: equilibrium constant used
    - ln_Q_fit: fitted ln(Q) values
    - r_squared: R^2 goodness of fit
    """
    # Compute reaction quotients
    Q = compute_reaction_quotient(A, B, C)
    ln_Q = np.log(Q)
    ln_Keq = np.log(true_Keq)
    ln_Q0 = ln_Q[0]

    # For the model ln(Q(t)) = ln(Keq) + (ln(Q0) - ln(Keq)) * exp(-K*t)
    # Rearranging: (ln(Q(t)) - ln(Keq)) = (ln(Q0) - ln(Keq)) * exp(-K*t)
    # Let y(t) = ln(Q(t)) - ln(Keq) and y0 = ln(Q0) - ln(Keq)
    # Then: y(t) = y0 * exp(-K*t)
    # Taking log: ln(|y(t)|) = ln(|y0|) - K*t

    # This is linear in K, so we can use linear least squares
    y = ln_Q - ln_Keq  # Deviation from equilibrium in log space
    y0 = y[0]

    # Only use points where y is significantly different from 0 (away from equilibrium)
    # and has the same sign as y0 (no oscillations through equilibrium)
    mask = (np.abs(y) > 1e-4) & (np.sign(y) == np.sign(y0))

    if np.sum(mask) < 5:  # If too few valid points, use all non-zero points
        mask = np.abs(y) > 1e-10

    # Set up linear regression: ln(|y|) = ln(|y0|) - K*t
    t_masked = t[mask]
    y_masked = y[mask]

    # Build design matrix for linear regression
    X = np.column_stack([np.ones_like(t_masked), -t_masked])
    Y = np.log(np.abs(y_masked))

    try:
        # Solve least squares problem
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        K_fit = coeffs[1]  # The coefficient of -t

        # Ensure K is positive (physical constraint)
        K_fit = abs(K_fit)

        # Compute fitted values using the exponential model
        ln_Q_fit = llrq_model(t, ln_Q0, K_fit, ln_Keq)

        # Compute R-squared
        ss_res = np.sum((ln_Q - ln_Q_fit) ** 2)
        ss_tot = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return K_fit, true_Keq, ln_Q_fit, r_squared

    except Exception as e:
        print(f"Least squares fitting failed: {e}")
        # Fall back to curve_fit method
        return fit_llrq_curve_fit(t, A, B, C, true_Keq)


def fit_llrq_parameter_sweep(t, A, B, C, true_Keq, fit_on="concentration"):
    """Fit K using parameter sweep to find global minimum.

    Args:
        fit_on: 'concentration' to fit on concentration profiles, 'lnQ' to fit on ln(Q)
    """
    Q = compute_reaction_quotient(A, B, C)
    ln_Q = np.log(Q)
    ln_Q0 = ln_Q[0]
    ln_Keq = np.log(true_Keq)

    # Try a wide range of K values
    K_values = np.logspace(-2, 3, 500)  # From 0.01 to 1000
    errors_lnQ = []
    errors_conc = []

    # Prepare initial conditions for LLRQ simulation
    initial_conc_dict = {"A": A[0], "B": B[0], "C": C[0]}

    print(f"  Fitting based on {fit_on} profiles...")

    for K in K_values:
        # Error in ln(Q)
        ln_Q_pred = llrq_model(t, ln_Q0, K, ln_Keq)
        error_lnQ = np.sum((ln_Q - ln_Q_pred) ** 2)
        errors_lnQ.append(error_lnQ)

        # Error in concentrations
        try:
            llrq_result = simulate_llrq_concentrations(t, initial_conc_dict, K, true_Keq)
            # Compute concentration errors
            A_err = np.sum((A - llrq_result["A"]) ** 2)
            B_err = np.sum((B - llrq_result["B"]) ** 2)
            C_err = np.sum((C - llrq_result["C"]) ** 2)
            error_conc = A_err + B_err + C_err
            errors_conc.append(error_conc)
        except:
            # If LLRQ simulation fails, assign large error
            errors_conc.append(1e10)

    # Choose which error to minimize
    if fit_on == "concentration":
        errors = errors_conc
        error_type = "concentration"
    else:
        errors = errors_lnQ
        error_type = "ln(Q)"

    # Find the K with minimum error
    best_idx = np.argmin(errors)
    K_fit = K_values[best_idx]

    # Compute fitted values with best K
    ln_Q_fit = llrq_model(t, ln_Q0, K_fit, ln_Keq)

    # Compute R-squared for ln(Q)
    ss_res_lnQ = np.sum((ln_Q - ln_Q_fit) ** 2)
    ss_tot_lnQ = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
    r_squared_lnQ = 1 - (ss_res_lnQ / ss_tot_lnQ) if ss_tot_lnQ > 0 else 0

    # Compute R-squared for concentrations
    try:
        llrq_result = simulate_llrq_concentrations(t, initial_conc_dict, K_fit, true_Keq)

        # R² for individual species
        ss_res_A = np.sum((A - llrq_result["A"]) ** 2)
        ss_tot_A = np.sum((A - np.mean(A)) ** 2)
        r_squared_A = 1 - (ss_res_A / ss_tot_A) if ss_tot_A > 0 else 0

        ss_res_B = np.sum((B - llrq_result["B"]) ** 2)
        ss_tot_B = np.sum((B - np.mean(B)) ** 2)
        r_squared_B = 1 - (ss_res_B / ss_tot_B) if ss_tot_B > 0 else 0

        ss_res_C = np.sum((C - llrq_result["C"]) ** 2)
        ss_tot_C = np.sum((C - np.mean(C)) ** 2)
        r_squared_C = 1 - (ss_res_C / ss_tot_C) if ss_tot_C > 0 else 0

        # Overall concentration R²
        r_squared_conc = (r_squared_A + r_squared_B + r_squared_C) / 3

    except:
        r_squared_A = r_squared_B = r_squared_C = r_squared_conc = -999

    # Print diagnostic info
    print(f"\nParameter sweep results:")
    print(f"  Best K found: {K_fit:.4f}")
    print(f"  Minimum {error_type} error: {errors[best_idx]:.6f}")
    print(f"  K range searched: [{K_values[0]:.2f}, {K_values[-1]:.2f}]")
    print(f"  R² for ln(Q): {r_squared_lnQ:.4f}")
    if r_squared_conc > -999:
        print(f"  R² for concentrations: {r_squared_conc:.4f}")
        print(f"    A: {r_squared_A:.4f}, B: {r_squared_B:.4f}, C: {r_squared_C:.4f}")

    # Optional: show if we're at boundary of search range
    if best_idx == 0:
        print("  WARNING: Best K at lower boundary - consider expanding search range")
    elif best_idx == len(K_values) - 1:
        print("  WARNING: Best K at upper boundary - consider expanding search range")

    # Create a plot showing the error landscape
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.semilogx(K_values, errors_lnQ, "b-", linewidth=1, label="ln(Q) error")
    if fit_on == "lnQ":
        plt.axvline(K_fit, color="r", linestyle="--", label=f"Best K = {K_fit:.2f}")
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    plt.title("ln(Q) Error vs K")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.semilogx(K_values, errors_conc, "g-", linewidth=1, label="Concentration error")
    if fit_on == "concentration":
        plt.axvline(K_fit, color="r", linestyle="--", label=f"Best K = {K_fit:.2f}")
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    plt.title("Concentration Error vs K")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Zoom in around the minimum for the chosen error type
    plt.subplot(1, 3, 3)
    zoom_range = 50  # points around minimum
    start_idx = max(0, best_idx - zoom_range)
    end_idx = min(len(K_values), best_idx + zoom_range + 1)
    plt.plot(K_values[start_idx:end_idx], errors[start_idx:end_idx], "r-", linewidth=2)
    plt.axvline(K_fit, color="k", linestyle="--", label=f"Best K = {K_fit:.2f}")
    plt.xlabel("K")
    plt.ylabel(f"Sum of Squared Errors ({error_type})")
    plt.title(f"Zoomed View Around Minimum ({error_type})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"parameter_sweep_K_{error_type}.png", dpi=150)
    print(f"  Saved parameter sweep plot to {OUT_DIR / f'parameter_sweep_K_{error_type}.png'}")
    plt.show()

    # Return both R² values in a dict
    result_dict = {
        "K_fit": K_fit,
        "Keq_used": true_Keq,
        "ln_Q_fit": ln_Q_fit,
        "r_squared_lnQ": r_squared_lnQ,
        "r_squared_conc": r_squared_conc,
        "r_squared_A": r_squared_A,
        "r_squared_B": r_squared_B,
        "r_squared_C": r_squared_C,
    }

    return result_dict


def fit_llrq_curve_fit(t, A, B, C, true_Keq):
    """Original curve_fit method as fallback."""
    Q = compute_reaction_quotient(A, B, C)
    ln_Q = np.log(Q)
    ln_Q0 = ln_Q[0]
    ln_Keq = np.log(true_Keq)

    def model(t, K):
        return llrq_model(t, ln_Q0, K, ln_Keq)

    try:
        popt, pcov = curve_fit(model, t, ln_Q, p0=[0.1], maxfev=5000)
        K_fit = popt[0]
        ln_Q_fit = model(t, K_fit)

        ss_res = np.sum((ln_Q - ln_Q_fit) ** 2)
        ss_tot = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return K_fit, true_Keq, ln_Q_fit, r_squared
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return None, None, None, None


def fit_llrq(t, A, B, C, true_Keq=None, fit_on="concentration"):
    """Fit LLRQ model: d/dt ln(Q) = -K * (ln(Q) - ln(Keq))

    Uses parameter sweep to find global minimum.
    """
    if true_Keq is not None:
        return fit_llrq_parameter_sweep(t, A, B, C, true_Keq, fit_on=fit_on)
    else:
        # For backward compatibility - fit both K and Keq
        Q = compute_reaction_quotient(A, B, C)
        ln_Q = np.log(Q)
        ln_Q0 = ln_Q[0]
        ln_Keq_guess = ln_Q[-1]
        K_guess = 0.1

        def model(t, K, ln_Keq):
            return llrq_model(t, ln_Q0, K, ln_Keq)

        try:
            popt, pcov = curve_fit(model, t, ln_Q, p0=[K_guess, ln_Keq_guess], maxfev=5000)
            K_fit, ln_Keq_fit = popt
            ln_Q_fit = model(t, K_fit, ln_Keq_fit)

            ss_res = np.sum((ln_Q - ln_Q_fit) ** 2)
            ss_tot = np.sum((ln_Q - np.mean(ln_Q)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return K_fit, np.exp(ln_Keq_fit), ln_Q_fit, r_squared
        except Exception as e:
            print(f"Fitting failed: {e}")
            return None, None, None, None


def simulate_llrq_concentrations_direct(t_eval, initial_concentrations, K_fit, Keq_fit):
    """Simulate concentrations directly from the LLRQ exponential model.

    For A + B <-> 2C, we have Q = C^2/(A*B) and the LLRQ model:
    ln(Q(t)) = ln(Keq) + (ln(Q0) - ln(Keq)) * exp(-K*t)

    We need to solve for A(t), B(t), C(t) given Q(t) and conservation laws.
    """
    A0 = initial_concentrations["A"]
    B0 = initial_concentrations["B"]
    C0 = initial_concentrations["C"]

    # Initial reaction quotient
    Q0 = compute_reaction_quotient(A0, B0, C0)
    ln_Q0 = np.log(Q0)
    ln_Keq = np.log(Keq_fit)

    # Evolve ln(Q) according to LLRQ model
    ln_Q_t = llrq_model(t_eval, ln_Q0, K_fit, ln_Keq)
    Q_t = np.exp(ln_Q_t)

    # For A + B <-> 2C, conservation laws are:
    # A0 + B0 = A(t) + B(t) + C(t)  (total atom balance)
    # Also, if we assume A and B change by equal amounts: A(t) = A0 - ξ, B(t) = B0 - ξ, C(t) = C0 + 2ξ
    # where ξ is the extent of reaction

    # Given Q(t) = C(t)^2 / (A(t) * B(t)) and the conservation laws:
    # Q(t) = (C0 + 2ξ)^2 / ((A0 - ξ)(B0 - ξ))

    A_t = np.zeros_like(t_eval)
    B_t = np.zeros_like(t_eval)
    C_t = np.zeros_like(t_eval)

    for i, Q in enumerate(Q_t):
        # Solve: Q = (C0 + 2ξ)^2 / ((A0 - ξ)(B0 - ξ))
        # Rearranging: Q*(A0 - ξ)(B0 - ξ) = (C0 + 2ξ)^2
        # This is a quadratic in ξ

        # Q*(A0*B0 - (A0+B0)*ξ + ξ^2) = C0^2 + 4*C0*ξ + 4*ξ^2
        # Q*A0*B0 - Q*(A0+B0)*ξ + Q*ξ^2 = C0^2 + 4*C0*ξ + 4*ξ^2
        # (Q - 4)*ξ^2 - (Q*(A0+B0) + 4*C0)*ξ + (Q*A0*B0 - C0^2) = 0

        a_coeff = Q - 4
        b_coeff = -(Q * (A0 + B0) + 4 * C0)
        c_coeff = Q * A0 * B0 - C0**2

        if abs(a_coeff) < 1e-12:  # Linear case
            if abs(b_coeff) > 1e-12:
                xi = -c_coeff / b_coeff
            else:
                xi = 0  # No change
        else:
            # Quadratic case
            discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
            if discriminant >= 0:
                xi1 = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
                xi2 = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)

                # Choose the physically meaningful root (concentrations should be positive)
                A1, B1, C1 = A0 - xi1, B0 - xi1, C0 + 2 * xi1
                A2, B2, C2 = A0 - xi2, B0 - xi2, C0 + 2 * xi2

                if A1 >= 0 and B1 >= 0 and C1 >= 0:
                    xi = xi1
                elif A2 >= 0 and B2 >= 0 and C2 >= 0:
                    xi = xi2
                else:
                    # Neither root gives positive concentrations, use the closer one to initial
                    xi = xi1 if abs(xi1) < abs(xi2) else xi2
            else:
                xi = 0  # No real solution, no change

        A_t[i] = max(0, A0 - xi)  # Ensure non-negative
        B_t[i] = max(0, B0 - xi)
        C_t[i] = max(0, C0 + 2 * xi)

    return {"t": t_eval, "A": A_t, "B": B_t, "C": C_t}


def simulate_llrq_concentrations(t_eval, initial_concentrations, K_fit, Keq_fit):
    """Wrapper that uses direct simulation for consistency."""
    return simulate_llrq_concentrations_direct(t_eval, initial_concentrations, K_fit, Keq_fit)


# Fit LLRQ to forward experiment and compare with true CM dynamics
print("\n" + "=" * 60)
print("LLRQ FITTING AND CONCENTRATION COMPARISON")
print("=" * 60)

# Simulate true CM dynamics
t_forward, A_forward, B_forward, C_forward, v_forward = simulate(
    (t_eval[0], t_eval[-1]), expts["forward"]["y0"], params, t_eval=t_eval
)

# Compute true Keq from CM parameters
true_Keq = (params.k_plus / params.k_minus) * (params.kM_C**2) / (params.kM_A * params.kM_B)
print(f"\nTrue Keq (from CM params): {true_Keq:.4e}")

# Fit LLRQ model to CM data using true Keq (optimizing for concentrations)
fit_result = fit_llrq(t_forward, A_forward, B_forward, C_forward, true_Keq=true_Keq, fit_on="concentration")

if fit_result is not None:
    K_fit = fit_result["K_fit"]
    Keq_used = fit_result["Keq_used"]
    ln_Q_fit = fit_result["ln_Q_fit"]
    r2_lnQ = fit_result["r_squared_lnQ"]
    r2_conc = fit_result["r_squared_conc"]

    print(f"\nLLRQ fit with true Keq (optimized for concentrations):")
    print(f"  Fitted K (relaxation rate): {K_fit:.4f}")
    print(f"  Using true Keq: {Keq_used:.4e}")
    print(f"  R² for ln(Q): {r2_lnQ:.4f}")
    print(f"  R² for concentrations: {r2_conc:.4f}")

    # Simulate LLRQ concentrations using fitted parameters
    initial_conc_dict = {"A": expts["forward"]["y0"][0], "B": expts["forward"]["y0"][1], "C": expts["forward"]["y0"][2]}

    print("\nSimulating LLRQ concentrations...")
    llrq_result = simulate_llrq_concentrations(t_forward, initial_conc_dict, K_fit, Keq_used)

    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 10))

    # First row: Concentration comparisons
    axes_conc = [plt.subplot(2, 3, i + 1) for i in range(3)]

    # Plot A concentrations
    ax = axes_conc[0]
    ax.plot(t_forward, A_forward, "b-", label="CM (true)", linewidth=2)
    ax.plot(llrq_result["t"], llrq_result["A"], "b--", label="LLRQ (predicted)", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title("Species A")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot B concentrations
    ax = axes_conc[1]
    ax.plot(t_forward, B_forward, "g-", label="CM (true)", linewidth=2)
    ax.plot(llrq_result["t"], llrq_result["B"], "g--", label="LLRQ (predicted)", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title("Species B")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot C concentrations
    ax = axes_conc[2]
    ax.plot(t_forward, C_forward, "r-", label="CM (true)", linewidth=2)
    ax.plot(llrq_result["t"], llrq_result["C"], "r--", label="LLRQ (predicted)", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title("Species C")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Second row: Reaction quotient comparisons
    # Compute Q values
    Q_true = compute_reaction_quotient(A_forward, B_forward, C_forward)
    Q_pred = compute_reaction_quotient(llrq_result["A"], llrq_result["B"], llrq_result["C"])

    # Plot Q on linear scale
    ax = plt.subplot(2, 3, 4)
    ax.plot(t_forward, Q_true, "k-", label="CM (true)", linewidth=2)
    ax.plot(llrq_result["t"], Q_pred, "k--", label="LLRQ (predicted)", linewidth=2, alpha=0.7)
    ax.axhline(y=Keq_used, color="r", linestyle=":", label=f"Keq = {Keq_used:.2e}", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Q")
    ax.set_title("Reaction Quotient (Linear Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Q on log scale
    ax = plt.subplot(2, 3, 5)
    ax.semilogy(t_forward, Q_true, "k-", label="CM (true)", linewidth=2)
    ax.semilogy(llrq_result["t"], Q_pred, "k--", label="LLRQ (predicted)", linewidth=2, alpha=0.7)
    ax.axhline(y=Keq_used, color="r", linestyle=":", label=f"Keq = {Keq_used:.2e}", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Q (log scale)")
    ax.set_title("Reaction Quotient (Log Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot ln(Q) with fit
    ax = plt.subplot(2, 3, 6)
    ax.plot(t_forward, np.log(Q_true), "ko", markersize=3, alpha=0.5, label="CM data")
    ax.plot(t_forward, ln_Q_fit, "r-", linewidth=2, label=f"LLRQ fit (K={K_fit:.2f})")
    ax.axhline(y=np.log(Keq_used), color="g", linestyle=":", label=f"ln(Keq)", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("ln(Q)")
    ax.set_title(f"ln(Q) Fit (R² = {r2_lnQ:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle(
        f"CM vs LLRQ: Concentrations and Reaction Quotient (K={K_fit:.2f}, Keq={Keq_used:.2e})", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    # Save figure
    fig_path = OUT_DIR / "cm_vs_llrq_full_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {fig_path}")

    plt.show()

    # Compute and display errors
    print("\n" + "=" * 60)
    print("CONCENTRATION PREDICTION ERRORS")
    print("=" * 60)

    # Calculate RMS errors
    rms_error_A = np.sqrt(np.mean((A_forward - llrq_result["A"]) ** 2))
    rms_error_B = np.sqrt(np.mean((B_forward - llrq_result["B"]) ** 2))
    rms_error_C = np.sqrt(np.mean((C_forward - llrq_result["C"]) ** 2))

    # Calculate relative errors at final time
    rel_error_A_final = abs(A_forward[-1] - llrq_result["A"][-1]) / A_forward[-1] * 100
    rel_error_B_final = abs(B_forward[-1] - llrq_result["B"][-1]) / B_forward[-1] * 100
    rel_error_C_final = abs(C_forward[-1] - llrq_result["C"][-1]) / C_forward[-1] * 100

    print(f"RMS Errors:")
    print(f"  Species A: {rms_error_A:.6f}")
    print(f"  Species B: {rms_error_B:.6f}")
    print(f"  Species C: {rms_error_C:.6f}")
    print(f"\nRelative Errors at t={t_forward[-1]:.1f}:")
    print(f"  Species A: {rel_error_A_final:.2f}%")
    print(f"  Species B: {rel_error_B_final:.2f}%")
    print(f"  Species C: {rel_error_C_final:.2f}%")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("The LLRQ model approximates the CM dynamics by modeling the evolution")
    print(f"of ln(Q) with a single relaxation rate K = {K_fit:.4f}.")
    print(
        f"The concentration predictions show {'good' if max(rms_error_A, rms_error_B, rms_error_C) < 0.01 else 'reasonable'} agreement"
    )
    print("with the true CM dynamics, demonstrating the LLRQ approximation quality.")
else:
    print("LLRQ fitting failed.")
