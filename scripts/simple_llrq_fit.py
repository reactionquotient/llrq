#!/usr/bin/env python
"""
Simple LLRQ fitting script - stripped down version.
Fits velocities using Wiener and Linear models with least squares.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def generate_simple_data(n_points=100, noise=0.0, seed=42):
    """Generate simple synthetic velocity data."""
    np.random.seed(seed)

    # Simple reaction A + B <-> 2C
    # Generate concentrations
    A = np.logspace(-2, 1, n_points)  # 0.01 to 10
    B = np.ones(n_points) * 1.0  # Fixed at 1
    C = np.logspace(-3, 0, n_points)  # 0.001 to 1

    # True parameters for velocity generation
    # Using a simple form: v = k_plus * A * B - k_minus * C^2
    k_plus = 1.0
    k_minus = 0.5

    # Compute velocities
    v_true = k_plus * A * B - k_minus * C**2

    # Add noise if requested
    if noise > 0:
        v_obs = v_true + np.random.normal(0, noise, n_points)
    else:
        v_obs = v_true

    return A, B, C, v_obs, v_true


def wiener_model(params, A, B, C):
    """Wiener model: v = α(1 - e^x) where x = θ0 + θA*ln(A) + θB*ln(B) + θC*ln(C)"""
    log_alpha, theta0, thetaA, thetaB, thetaC = params

    logA = np.log(A)
    logB = np.log(B)
    logC = np.log(C)

    x = theta0 + thetaA * logA + thetaB * logB + thetaC * logC
    alpha = np.exp(log_alpha)

    return alpha * (1.0 - np.exp(x))


def linear_model(params, A, B, C):
    """Linear model: v = β * x where x = θ0 + θA*ln(A) + θB*ln(B) + θC*ln(C)"""
    beta, theta0, thetaA, thetaB, thetaC = params

    logA = np.log(A)
    logB = np.log(B)
    logC = np.log(C)

    x = theta0 + thetaA * logA + thetaB * logB + thetaC * logC

    return beta * x


def fit_model(A, B, C, v, model_func, p0):
    """Fit model using least squares."""

    def residuals(params):
        return v - model_func(params, A, B, C)

    result = least_squares(residuals, p0, method="lm")
    return result


def main():
    # Generate data
    print("Generating synthetic data...")
    A, B, C, v_obs, v_true = generate_simple_data(n_points=100, noise=0.0)

    print(f"Data ranges:")
    print(f"  A: [{A.min():.3f}, {A.max():.3f}]")
    print(f"  B: [{B.min():.3f}, {B.max():.3f}]")
    print(f"  C: [{C.min():.3f}, {C.max():.3f}]")
    print(f"  v: [{v_obs.min():.3f}, {v_obs.max():.3f}]")

    # Fit Wiener model
    print("\nFitting Wiener model...")
    p0_wiener = [np.log(1.0), 0.0, -1.0, -1.0, 2.0]  # Initial guess
    result_wiener = fit_model(A, B, C, v_obs, wiener_model, p0_wiener)

    v_pred_wiener = wiener_model(result_wiener.x, A, B, C)
    rmse_wiener = np.sqrt(np.mean((v_obs - v_pred_wiener) ** 2))
    r2_wiener = 1 - np.sum((v_obs - v_pred_wiener) ** 2) / np.sum((v_obs - v_obs.mean()) ** 2)

    print(f"  Converged: {result_wiener.success}")
    print(f"  RMSE: {rmse_wiener:.6f}")
    print(f"  R²: {r2_wiener:.6f}")
    print(f"  Parameters:")
    param_names = ["log_alpha", "theta0", "thetaA", "thetaB", "thetaC"]
    for name, val in zip(param_names, result_wiener.x):
        print(f"    {name}: {val:.6f}")

    # Fit Linear model
    print("\nFitting Linear model...")
    p0_linear = [1.0, 0.0, -1.0, -1.0, 2.0]  # Initial guess
    result_linear = fit_model(A, B, C, v_obs, linear_model, p0_linear)

    v_pred_linear = linear_model(result_linear.x, A, B, C)
    rmse_linear = np.sqrt(np.mean((v_obs - v_pred_linear) ** 2))
    r2_linear = 1 - np.sum((v_obs - v_pred_linear) ** 2) / np.sum((v_obs - v_obs.mean()) ** 2)

    print(f"  Converged: {result_linear.success}")
    print(f"  RMSE: {rmse_linear:.6f}")
    print(f"  R²: {r2_linear:.6f}")
    print(f"  Parameters:")
    param_names = ["beta", "theta0", "thetaA", "thetaB", "thetaC"]
    for name, val in zip(param_names, result_linear.x):
        print(f"    {name}: {val:.6f}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # True vs predicted - Wiener
    axes[0].scatter(v_obs, v_pred_wiener, alpha=0.6)
    lims = [min(v_obs.min(), v_pred_wiener.min()), max(v_obs.max(), v_pred_wiener.max())]
    axes[0].plot(lims, lims, "r--", alpha=0.5)
    axes[0].set_xlabel("Observed v")
    axes[0].set_ylabel("Predicted v")
    axes[0].set_title(f"Wiener Model\nRMSE={rmse_wiener:.4f}, R²={r2_wiener:.4f}")
    axes[0].grid(True, alpha=0.3)

    # True vs predicted - Linear
    axes[1].scatter(v_obs, v_pred_linear, alpha=0.6)
    lims = [min(v_obs.min(), v_pred_linear.min()), max(v_obs.max(), v_pred_linear.max())]
    axes[1].plot(lims, lims, "r--", alpha=0.5)
    axes[1].set_xlabel("Observed v")
    axes[1].set_ylabel("Predicted v")
    axes[1].set_title(f"Linear Model\nRMSE={rmse_linear:.4f}, R²={r2_linear:.4f}")
    axes[1].grid(True, alpha=0.3)

    # Velocity curves vs A
    sort_idx = np.argsort(A)
    axes[2].plot(A[sort_idx], v_obs[sort_idx], "ko-", label="Data", markersize=4)
    axes[2].plot(A[sort_idx], v_pred_wiener[sort_idx], "b-", label="Wiener", linewidth=2)
    axes[2].plot(A[sort_idx], v_pred_linear[sort_idx], "r-", label="Linear", linewidth=2)
    axes[2].set_xlabel("[A]")
    axes[2].set_ylabel("Velocity")
    axes[2].set_xscale("log")
    axes[2].set_title("Velocity vs [A]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/simple_llrq_fit.png", dpi=150)
    plt.show()

    print("\nPlot saved to output/simple_llrq_fit.png")


if __name__ == "__main__":
    main()
