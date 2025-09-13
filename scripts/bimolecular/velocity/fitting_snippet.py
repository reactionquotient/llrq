#!/usr/bin/env python3
"""
Velocity Fitting for Bimolecular Reactions using CM-Compatible Model

This module implements a compact fitter for reaction velocities using the
CM-compatible model: v = Lambda(A,B,C) * (1 - Q/Keq), where:
- Lambda(A,B,C) = Vmax * (A/(KA+A)) * (B/(KB+B)) / (1 + a1*C + a2*C^2)
- Q = C^2/(A*B) is the reaction quotient
- Positive parameters are enforced via exponential transformations

The fitter jointly processes forward (varying A at fixed B, C=0) and
reverse (varying C at fixed low A,B) datasets to estimate kinetic parameters.
"""

import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Constants
DEFAULT_REL_WEIGHT_REVERSE = 4.0  # Weight reverse data more heavily due to fewer points
DEFAULT_REG_STRENGTH = 1e-3  # L2 regularization strength for log parameters
DEFAULT_MAX_NFEV = 100000  # Maximum function evaluations for optimizer
EPS_CONCENTRATION = 1e-12  # Small epsilon to prevent division by zero

# File paths
DATA_DIR = Path(__file__).parent.parent.parent / "output"  # Use local output directory
SYNTHETIC_DATA_PATH = Path(__file__).parent / "synthetic_data.py"


def load_synthetic_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load synthetic forward and reverse datasets from the synthetic_data module.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (forward_df, reverse_df)

    Raises:
        ImportError: If synthetic_data.py cannot be loaded
        AttributeError: If required dataframes are not found in the module
    """
    if not SYNTHETIC_DATA_PATH.exists():
        raise FileNotFoundError(f"Synthetic data module not found at {SYNTHETIC_DATA_PATH}")

    spec = importlib.util.spec_from_file_location("synthetic_data", SYNTHETIC_DATA_PATH)
    syn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(syn_module)

    if not hasattr(syn_module, "forward_df") or not hasattr(syn_module, "reverse_df"):
        raise AttributeError("synthetic_data module must define 'forward_df' and 'reverse_df'")

    return syn_module.forward_df.copy(), syn_module.reverse_df.copy()


def compute_reaction_quotient(A: np.ndarray, B: np.ndarray, C: np.ndarray, eps: float = EPS_CONCENTRATION) -> np.ndarray:
    """
    Compute reaction quotient Q = C^2/(A*B) for the reaction A + B ⇌ 2C.

    Parameters:
        A, B, C: Concentration arrays (mM)
        eps: Small value to prevent division by zero

    Returns:
        np.ndarray: Reaction quotient values
    """
    AB = np.maximum(A * B, eps)
    return (C * C + eps) / AB


def compute_lambda_amplitude(theta_pos: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Compute the Lambda amplitude function for the CM-compatible rate law.

    Lambda(A,B,C) = Vmax * (A/(KA+A)) * (B/(KB+B)) / (1 + a1*C + a2*C^2)

    Parameters:
        theta_pos: Log-transformed positive parameters [ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2]
        A, B, C: Concentration arrays (mM)

    Returns:
        np.ndarray: Lambda amplitude values
    """
    ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2 = theta_pos

    # Transform back to positive parameters
    Vmax = np.exp(ln_Vmax)
    KA = np.exp(ln_KA)
    KB = np.exp(ln_KB)
    a1 = np.exp(ln_a1)
    a2 = np.exp(ln_a2)

    # Compute Michaelis-Menten terms
    numerator = Vmax * (A / (KA + A)) * (B / (KB + B))

    # Compute product inhibition denominator
    denominator = 1.0 + a1 * C + a2 * C * C

    return numerator / denominator


def predict_velocity(theta: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Predict reaction velocity using the CM-compatible model.

    v = Lambda(A,B,C) * (1 - Q/Keq)

    Parameters:
        theta: Full parameter vector [ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2, ln_Keq]
        A, B, C: Concentration arrays (mM)

    Returns:
        np.ndarray: Predicted velocity values
    """
    ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2, ln_Keq = theta

    # Extract equilibrium constant
    Keq = np.exp(ln_Keq)

    # Compute amplitude and thermodynamic factor
    lambda_amp = compute_lambda_amplitude([ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2], A, B, C)
    Q = compute_reaction_quotient(A, B, C)
    thermodynamic_factor = 1.0 - Q / Keq

    return lambda_amp * thermodynamic_factor


def compute_residuals(
    theta: np.ndarray,
    data: pd.DataFrame,
    reverse_weight: float = DEFAULT_REL_WEIGHT_REVERSE,
    reg_strength: float = DEFAULT_REG_STRENGTH,
) -> np.ndarray:
    """
    Compute weighted residuals for parameter optimization.

    Parameters:
        theta: Parameter vector [ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2, ln_Keq]
        data: Combined dataset with columns ['A_mM', 'B_mM', 'C_mM', 'v_obs', 'split']
        reverse_weight: Relative weight for reverse data points
        reg_strength: L2 regularization strength

    Returns:
        np.ndarray: Weighted residuals including regularization terms
    """
    # Predict velocities
    v_pred = predict_velocity(theta, data["A_mM"].values, data["B_mM"].values, data["C_mM"].values)
    v_obs = data["v_obs"].values

    # Apply differential weighting (reverse data typically has fewer points)
    weights = np.where(data["split"].values == "reverse", reverse_weight, 1.0)
    weighted_residuals = weights * (v_pred - v_obs)

    # Add L2 regularization to keep parameters on reasonable scales
    regularization = reg_strength * np.array(theta)

    return np.concatenate([weighted_residuals, regularization])


def initialize_parameters(data: pd.DataFrame) -> np.ndarray:
    """
    Initialize parameter estimates using data-driven heuristics.

    Parameters:
        data: Combined dataset

    Returns:
        np.ndarray: Initial parameter vector [ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2, ln_Keq]
    """
    # Estimate Vmax from 95th percentile of absolute velocities
    ln_Vmax0 = np.log(max(1e-6, data["v_obs"].abs().quantile(0.95)))

    # Estimate Michaelis constants from median concentrations
    pos_A = data["A_mM"][data["A_mM"] > 0]
    pos_B = data["B_mM"][data["B_mM"] > 0]
    ln_KA0 = np.log(max(1e-6, pos_A.median() if len(pos_A) > 0 else 1e-2))
    ln_KB0 = np.log(max(1e-6, pos_B.median() if len(pos_B) > 0 else 1e-2))

    # Initialize product inhibition terms with weak effects
    ln_a1_0 = np.log(0.1)
    ln_a2_0 = np.log(0.1)

    # Estimate equilibrium constant from reverse data where C is substantial
    reverse_data = data[data["split"] == "reverse"]
    if len(reverse_data) > 0:
        Q_reverse = compute_reaction_quotient(
            reverse_data["A_mM"].values, reverse_data["B_mM"].values, reverse_data["C_mM"].values
        )
        Keq0 = np.median(Q_reverse[Q_reverse > 0])
    else:
        Keq0 = 1.0

    ln_Keq0 = np.log(max(Keq0, 1e-9))

    return np.array([ln_Vmax0, ln_KA0, ln_KB0, ln_a1_0, ln_a2_0, ln_Keq0])


def fit_velocity_model(
    data: pd.DataFrame,
    reverse_weight: float = DEFAULT_REL_WEIGHT_REVERSE,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    max_nfev: int = DEFAULT_MAX_NFEV,
    theta0: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit the CM-compatible velocity model to experimental data.

    Parameters:
        data: Combined dataset with forward and reverse experiments
        reverse_weight: Relative weight for reverse data points
        reg_strength: L2 regularization strength
        max_nfev: Maximum function evaluations
        theta0: Initial parameter guess (auto-generated if None)

    Returns:
        Dict containing fitted parameters, optimization result, and diagnostics
    """
    # Initialize parameters if not provided
    if theta0 is None:
        theta0 = initialize_parameters(data)

    # Define residual function
    residual_func = lambda theta: compute_residuals(theta, data, reverse_weight, reg_strength)

    # Perform optimization
    result = least_squares(residual_func, theta0, max_nfev=max_nfev)

    # Extract fitted parameters
    theta_hat = result.x
    ln_Vmax, ln_KA, ln_KB, ln_a1, ln_a2, ln_Keq = theta_hat
    fitted_params = {
        "Vmax": np.exp(ln_Vmax),
        "KA": np.exp(ln_KA),
        "KB": np.exp(ln_KB),
        "a1": np.exp(ln_a1),
        "a2": np.exp(ln_a2),
        "Keq": np.exp(ln_Keq),
    }

    # Compute predictions and diagnostics
    data_with_pred = data.copy()
    data_with_pred["v_pred"] = predict_velocity(theta_hat, data["A_mM"].values, data["B_mM"].values, data["C_mM"].values)
    data_with_pred["residual"] = data_with_pred["v_pred"] - data_with_pred["v_obs"]

    # Compute split-specific RMS errors
    forward_resid = data_with_pred.loc[data_with_pred["split"] == "forward", "residual"]
    reverse_resid = data_with_pred.loc[data_with_pred["split"] == "reverse", "residual"]

    return {
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "fitted_params": fitted_params,
        "rms_forward": float(np.sqrt(np.mean(forward_resid**2))) if len(forward_resid) > 0 else 0.0,
        "rms_reverse": float(np.sqrt(np.mean(reverse_resid**2))) if len(reverse_resid) > 0 else 0.0,
        "data_with_predictions": data_with_pred,
        "optimization_result": result,
    }


def create_diagnostic_plots(fit_result: Dict[str, Any], output_dir: Path = DATA_DIR) -> None:
    """
    Create diagnostic plots showing model fit quality.

    Parameters:
        fit_result: Dictionary returned by fit_velocity_model()
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)
    data = fit_result["data_with_predictions"]

    # Forward direction plots: v vs A for each B series
    plt.figure(figsize=(10, 6))
    forward_data = data.query("split == 'forward'")

    for series_id, group in forward_data.groupby("series_id"):
        # Sort by A concentration for clean lines
        group_sorted = group.sort_values("A_mM")
        plt.scatter(group_sorted["A_mM"], group_sorted["v_obs"], s=20, alpha=0.7, label=f"{series_id} (obs)")
        plt.plot(group_sorted["A_mM"], group_sorted["v_pred"], "--", label=f"{series_id} (pred)")

    plt.xscale("log")
    plt.xlabel("A concentration (mM)")
    plt.ylabel("Velocity")
    plt.title("Forward Direction: Velocity vs [A] at Fixed [B] (C=0)\nΛ(A,B,C) · (1 - Q/Keq) Model Fit")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_fit_forward.png", dpi=150)
    plt.close()

    # Reverse direction plot: v vs C
    plt.figure(figsize=(8, 6))
    reverse_data = data.query("split == 'reverse'").sort_values("C_mM")

    plt.scatter(reverse_data["C_mM"], reverse_data["v_obs"], s=30, label="Reverse (observed)", color="red", alpha=0.7)
    plt.plot(reverse_data["C_mM"], reverse_data["v_pred"], "r--", label="Reverse (predicted)", linewidth=2)

    plt.xscale("log")
    plt.xlabel("C concentration (mM)")
    plt.ylabel("Velocity")
    plt.title("Reverse Direction: Velocity vs [C] at Fixed Low [A],[B]\nΛ(A,B,C) · (1 - Q/Keq) Model Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_fit_reverse.png", dpi=150)
    plt.close()


def save_fit_summary(fit_result: Dict[str, Any], output_dir: Path = DATA_DIR) -> None:
    """
    Save fitting results to a JSON summary file.

    Parameters:
        fit_result: Dictionary returned by fit_velocity_model()
        output_dir: Directory to save summary
    """
    output_dir.mkdir(exist_ok=True)

    # Create summary without non-serializable objects
    summary = {
        "success": fit_result["success"],
        "message": fit_result["message"],
        "cost": fit_result["cost"],
        "fitted_params": fit_result["fitted_params"],
        "rms_forward": fit_result["rms_forward"],
        "rms_reverse": fit_result["rms_reverse"],
        "n_forward_points": int(
            len(fit_result["data_with_predictions"][fit_result["data_with_predictions"]["split"] == "forward"])
        ),
        "n_reverse_points": int(
            len(fit_result["data_with_predictions"][fit_result["data_with_predictions"]["split"] == "reverse"])
        ),
    }

    with open(output_dir / "velocity_fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> Dict[str, Any]:
    """
    Main execution function: load data, fit model, create plots, and save results.

    Returns:
        Dict: Fitting results summary
    """
    try:
        # Load synthetic datasets
        print("Loading synthetic data...")
        forward_df, reverse_df = load_synthetic_data()

        # Combine datasets with split labels
        combined_data = pd.concat([forward_df.assign(split="forward"), reverse_df.assign(split="reverse")], ignore_index=True)

        print(f"Loaded {len(forward_df)} forward and {len(reverse_df)} reverse data points")

        # Fit the velocity model
        print("Fitting CM-compatible velocity model...")
        fit_result = fit_velocity_model(combined_data)

        if fit_result["success"]:
            print("✓ Optimization converged successfully")
            print(f"Final cost: {fit_result['cost']:.2e}")
            print(f"Forward RMS error: {fit_result['rms_forward']:.2e}")
            print(f"Reverse RMS error: {fit_result['rms_reverse']:.2e}")

            # Display fitted parameters
            print("\nFitted Parameters:")
            for param, value in fit_result["fitted_params"].items():
                print(f"  {param}: {value:.3e}")
        else:
            print(f"✗ Optimization failed: {fit_result['message']}")

        # Create diagnostic plots
        print("Creating diagnostic plots...")
        create_diagnostic_plots(fit_result)

        # Save summary
        print("Saving fit summary...")
        save_fit_summary(fit_result)

        print(f"Results saved to {DATA_DIR}")
        return fit_result

    except Exception as e:
        print(f"Error during fitting: {e}")
        raise


if __name__ == "__main__":
    result = main()
