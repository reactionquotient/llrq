#!/usr/bin/env python
"""
Improved LLRQ-based fitting with better handling of CM-like data.

This version includes:
1. Better initial parameter estimation
2. Saturating Wiener model option
3. Robust fitting with outlier detection
4. Cross-validation
"""

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares, differential_evolution
from sklearn.model_selection import KFold


@dataclass
class FitResult:
    """Container for fitting results."""

    params: np.ndarray
    param_names: List[str]
    param_std: Optional[np.ndarray]
    residuals: np.ndarray
    rmse: float
    r2: float
    aic: float
    bic: float
    n_params: int
    n_data: int
    converged: bool
    message: str
    model_name: str
    cv_rmse: Optional[float] = None
    cv_r2: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "params": {name: float(val) for name, val in zip(self.param_names, self.params)},
            "param_std": {name: float(val) for name, val in zip(self.param_names, self.param_std)}
            if self.param_std is not None
            else None,
            "metrics": {
                "rmse": float(self.rmse),
                "r2": float(self.r2),
                "aic": float(self.aic),
                "bic": float(self.bic),
                "cv_rmse": float(self.cv_rmse) if self.cv_rmse else None,
                "cv_r2": float(self.cv_r2) if self.cv_r2 else None,
            },
            "fit_info": {
                "n_params": self.n_params,
                "n_data": self.n_data,
                "converged": self.converged,
                "message": self.message,
            },
        }


class SaturatingWienerModel:
    """
    Saturating Wiener model that can handle CM-like kinetics.

    v = vmax * (1 - exp(x)) / (1 + K_sat * (1 - exp(x)))

    where x = ln(Q/Keq)
    """

    def __init__(self, constrained: bool = False):
        self.constrained = constrained
        self.fitted = False
        self.result: Optional[FitResult] = None

    def get_param_names(self) -> List[str]:
        if self.constrained:
            # vmax, K_sat, w0 (where w0 = -ln(Keq))
            return ["log_vmax", "log_Ksat", "w0"]
        else:
            # vmax, K_sat, and theta coefficients
            return ["log_vmax", "log_Ksat", "theta0", "thetaA", "thetaB", "thetaC"]

    def predict(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict velocities."""
        eps = 1e-12
        logA = np.log(np.maximum(A, eps))
        logB = np.log(np.maximum(B, eps))
        logC = np.log(np.maximum(C, eps))

        if self.constrained:
            log_vmax, log_Ksat, w0 = params
            # Stoichiometric constraint: x = -ln[A] - ln[B] + 2ln[C] + w0
            x = w0 - logA - logB + 2.0 * logC
        else:
            log_vmax, log_Ksat, theta0, thetaA, thetaB, thetaC = params
            x = theta0 + thetaA * logA + thetaB * logB + thetaC * logC

        vmax = np.exp(log_vmax)
        Ksat = np.exp(log_Ksat)

        # Saturating form
        driving = 1.0 - np.exp(x)
        v = vmax * driving / (1.0 + Ksat * np.abs(driving))

        return v

    def get_initial_params(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Smart initial parameter estimation."""
        # Estimate vmax from high concentration data
        high_conc_mask = (A > np.percentile(A, 75)) | (B > np.percentile(B, 75))
        if high_conc_mask.sum() > 10:
            vmax_est = np.percentile(np.abs(y[high_conc_mask]), 90)
        else:
            vmax_est = np.percentile(np.abs(y), 90)

        # Estimate Ksat from half-saturation
        Ksat_est = 0.1  # Start with moderate saturation

        if self.constrained:
            # Estimate w0 from equilibrium-like points
            low_v_mask = np.abs(y) < np.percentile(np.abs(y), 25)
            if low_v_mask.sum() > 5:
                # At equilibrium, Q ≈ Keq, so x ≈ 0
                # Try to find w0 that makes x ≈ 0 for low velocity points
                eps = 1e-12
                logA_low = np.log(np.maximum(A[low_v_mask], eps))
                logB_low = np.log(np.maximum(B[low_v_mask], eps))
                logC_low = np.log(np.maximum(C[low_v_mask], eps))
                w0_est = np.mean(logA_low + logB_low - 2 * logC_low)
            else:
                w0_est = 0.0

            return np.array([np.log(vmax_est), np.log(Ksat_est), w0_est])
        else:
            return np.array([np.log(vmax_est), np.log(Ksat_est), 0.0, -1.0, -1.0, 2.0])

    def fit(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        method: str = "global",
        loss: str = "soft_l1",
        verbose: bool = False,
    ) -> FitResult:
        """
        Fit model using either local or global optimization.
        """
        # Get initial guess
        p0 = self.get_initial_params(A, B, C, y)

        if method == "global":
            # Use differential evolution for global optimization
            if self.constrained:
                bounds = [
                    (-2, 10),  # log_vmax
                    (-10, 5),  # log_Ksat
                    (-10, 10),  # w0
                ]
            else:
                bounds = [
                    (-2, 10),  # log_vmax
                    (-10, 5),  # log_Ksat
                    (-10, 10),  # theta0
                    (-5, 5),  # thetaA
                    (-5, 5),  # thetaB
                    (-5, 5),  # thetaC
                ]

            def objective(params):
                y_pred = self.predict(A, B, C, params)
                if loss == "soft_l1":
                    # Soft L1 loss
                    residuals = y - y_pred
                    return np.sum(2 * (np.sqrt(1 + residuals**2) - 1))
                else:
                    return np.sum((y - y_pred) ** 2)

            result_de = differential_evolution(objective, bounds, seed=42, maxiter=500, polish=True, workers=1)
            params = result_de.x
            converged = result_de.success
            message = str(result_de.message)

        else:
            # Local optimization with least squares
            def residuals(params):
                return y - self.predict(A, B, C, params)

            result_ls = least_squares(residuals, p0, method="trf", loss=loss, max_nfev=1000)
            params = result_ls.x
            converged = result_ls.success
            message = result_ls.message

        # Compute statistics
        y_pred = self.predict(A, B, C, params)
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))

        # Information criteria
        n = len(y)
        k = len(params)
        log_likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Cross-validation
        cv_rmse, cv_r2 = self._cross_validate(A, B, C, y, params)

        self.result = FitResult(
            params=params,
            param_names=self.get_param_names(),
            param_std=None,  # Not computed for global optimization
            residuals=residuals,
            rmse=rmse,
            r2=r2,
            aic=aic,
            bic=bic,
            n_params=k,
            n_data=n,
            converged=converged,
            message=message,
            model_name=f"SaturatingWiener_{'constrained' if self.constrained else 'unconstrained'}",
            cv_rmse=cv_rmse,
            cv_r2=cv_r2,
        )

        self.fitted = True
        return self.result

    def _cross_validate(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray, params: np.ndarray, n_folds: int = 5
    ) -> Tuple[float, float]:
        """Perform k-fold cross-validation."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        rmse_scores = []
        r2_scores = []

        for train_idx, test_idx in kf.split(A):
            # Split data
            A_train, A_test = A[train_idx], A[test_idx]
            B_train, B_test = B[train_idx], B[test_idx]
            C_train, C_test = C[train_idx], C[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit on training data
            temp_model = SaturatingWienerModel(constrained=self.constrained)
            temp_result = temp_model.fit(A_train, B_train, C_train, y_train, method="local", verbose=False)

            # Predict on test data
            y_pred = temp_model.predict(A_test, B_test, C_test, temp_result.params)

            # Compute metrics
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            rmse_scores.append(rmse)
            r2_scores.append(r2)

        return np.mean(rmse_scores), np.mean(r2_scores)


class ImprovedLinearModel:
    """
    Improved linear model with better parameterization.

    v = sum_i(beta_i * x_i) where x_i are log-transformed features
    """

    def __init__(self, include_interactions: bool = False):
        self.include_interactions = include_interactions
        self.fitted = False
        self.result: Optional[FitResult] = None

    def get_param_names(self) -> List[str]:
        names = ["beta0", "beta_logA", "beta_logB", "beta_logC"]
        if self.include_interactions:
            names.extend(["beta_logA_logB", "beta_logA_logC", "beta_logB_logC"])
        return names

    def _build_features(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Build feature matrix."""
        eps = 1e-12
        logA = np.log(np.maximum(A, eps))
        logB = np.log(np.maximum(B, eps))
        logC = np.log(np.maximum(C, eps))

        features = [np.ones_like(A), logA, logB, logC]

        if self.include_interactions:
            features.extend([logA * logB, logA * logC, logB * logC])

        return np.column_stack(features)

    def predict(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict velocities."""
        X = self._build_features(A, B, C)
        return X @ params

    def fit(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray, regularization: float = 0.01) -> FitResult:
        """
        Fit using regularized least squares.
        """
        X = self._build_features(A, B, C)

        # Ridge regression
        XtX = X.T @ X
        Xty = X.T @ y
        n_features = X.shape[1]

        # Add regularization
        reg_matrix = regularization * np.eye(n_features)
        reg_matrix[0, 0] = 0  # Don't regularize intercept

        # Solve normal equations
        params = np.linalg.solve(XtX + reg_matrix, Xty)

        # Compute statistics
        y_pred = X @ params
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))

        # Information criteria
        n = len(y)
        k = len(params)
        # Account for regularization in effective degrees of freedom
        df_effective = np.trace(X @ np.linalg.inv(XtX + reg_matrix) @ X.T)
        log_likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = 2 * df_effective - 2 * log_likelihood
        bic = df_effective * np.log(n) - 2 * log_likelihood

        # Parameter uncertainties
        try:
            sigma2 = ss_res / (n - k)
            cov = sigma2 * np.linalg.inv(XtX + reg_matrix)
            param_std = np.sqrt(np.diag(cov))
        except:
            param_std = None

        self.result = FitResult(
            params=params,
            param_names=self.get_param_names(),
            param_std=param_std,
            residuals=residuals,
            rmse=rmse,
            r2=r2,
            aic=aic,
            bic=bic,
            n_params=k,
            n_data=n,
            converged=True,
            message="Ridge regression",
            model_name=f"LinearRegularized_{'with_interactions' if self.include_interactions else 'no_interactions'}",
        )

        self.fitted = True
        return self.result


def fit_all_improved_models(df: pd.DataFrame, verbose: bool = True) -> Dict[str, FitResult]:
    """Fit all improved model variants."""
    # Extract data
    A = df["A_mM"].values
    B = df["B_mM"].values
    C = df["C_mM"].values
    y = df["v_obs"].values

    results = {}

    # Saturating Wiener models
    for constrained in [False, True]:
        name = f"SaturatingWiener_{'constrained' if constrained else 'unconstrained'}"
        if verbose:
            print(f"\nFitting {name}...")

        model = SaturatingWienerModel(constrained=constrained)
        result = model.fit(A, B, C, y, method="global", verbose=verbose)
        results[name] = result

        if verbose:
            print(f"  Converged: {result.converged}")
            print(f"  RMSE: {result.rmse:.4f}")
            print(f"  R²: {result.r2:.4f}")
            print(f"  CV-RMSE: {result.cv_rmse:.4f}" if result.cv_rmse else "  CV-RMSE: N/A")
            print(f"  CV-R²: {result.cv_r2:.4f}" if result.cv_r2 else "  CV-R²: N/A")

    # Linear models with regularization
    for interactions in [False, True]:
        name = f"LinearRegularized_{'with_interactions' if interactions else 'no_interactions'}"
        if verbose:
            print(f"\nFitting {name}...")

        model = ImprovedLinearModel(include_interactions=interactions)
        result = model.fit(A, B, C, y, regularization=0.01)
        results[name] = result

        if verbose:
            print(f"  RMSE: {result.rmse:.4f}")
            print(f"  R²: {result.r2:.4f}")

    return results


def create_improved_diagnostic_plots(df: pd.DataFrame, results: Dict[str, FitResult], output_dir: Path, show: bool = True):
    """Create improved diagnostic plots."""
    # Extract data
    A = df["A_mM"].values
    B = df["B_mM"].values
    C = df["C_mM"].values
    y = df["v_obs"].values

    fig = plt.figure(figsize=(16, 10))

    n_models = len(results)
    n_cols = min(n_models, 4)
    n_rows = (n_models + n_cols - 1) // n_cols

    for i, (name, result) in enumerate(results.items(), 1):
        # Recreate model for predictions
        if "SaturatingWiener" in name:
            model = SaturatingWienerModel(constrained="constrained" in name)
        else:
            model = ImprovedLinearModel(include_interactions="with_interactions" in name)

        y_pred = model.predict(A, B, C, result.params)

        # Observed vs Predicted
        ax = plt.subplot(n_rows, n_cols, i)
        ax.scatter(y, y_pred, alpha=0.5, s=10)

        # Add diagonal line
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, "k-", alpha=0.5, lw=1)

        ax.set_xlabel("Observed v")
        ax.set_ylabel("Predicted v")
        title = f"{name}\nR²={result.r2:.3f}, RMSE={result.rmse:.2f}"
        if result.cv_r2 is not None:
            title += f"\nCV-R²={result.cv_r2:.3f}"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"llrq_improved_diagnostics_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    # Create residual analysis plot
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Select best model for detailed analysis
    best_model_name = min(results.keys(), key=lambda k: results[k].aic)
    best_result = results[best_model_name]

    if "SaturatingWiener" in best_model_name:
        best_model = SaturatingWienerModel(constrained="constrained" in best_model_name)
    else:
        best_model = ImprovedLinearModel(include_interactions="with_interactions" in best_model_name)

    y_pred_best = best_model.predict(A, B, C, best_result.params)
    residuals = y - y_pred_best

    # Residuals vs fitted
    axes[0, 0].scatter(y_pred_best, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title(f"Residuals vs Fitted ({best_model_name})")
    axes[0, 0].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q Plot")
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].grid(True, alpha=0.3)

    # Scale-location plot
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 1].scatter(y_pred_best, np.sqrt(np.abs(standardized_residuals)), alpha=0.5, s=10)
    axes[1, 1].set_xlabel("Fitted values")
    axes[1, 1].set_ylabel("√|Standardized residuals|")
    axes[1, 1].set_title("Scale-Location Plot")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save residual analysis
    fig2_path = output_dir / f"llrq_residual_analysis_{timestamp}.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig_path, fig2_path


def main():
    """Main execution function."""
    # Setup
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Improved LLRQ Fitting for CM-like Data")
    print("=" * 60)

    # Load data
    print("\nLoading synthetic data...")
    fwd_df = pd.read_csv(sorted(output_dir.glob("synthetic_forward_*.csv"))[-1])
    rev_df = pd.read_csv(sorted(output_dir.glob("synthetic_reverse_*.csv"))[-1])
    df = pd.concat([fwd_df, rev_df], ignore_index=True)
    print(f"Loaded {len(df)} data points ({len(fwd_df)} forward, {len(rev_df)} reverse)")

    # Load true parameters if available
    param_files = sorted(output_dir.glob("cm_params_*.json"))
    if param_files:
        with open(param_files[-1], "r") as f:
            true_params = json.load(f)
        print("\nTrue CM parameters:")
        for key in ["kM_A", "kM_B", "kM_C", "keq", "kV"]:
            if key in true_params:
                print(f"  {key}: {true_params[key]:.6f}")
    else:
        true_params = None

    # Fit improved models
    results = fit_all_improved_models(df, verbose=True)

    # Create comparison table
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    comparison_data = []
    for name, result in results.items():
        row = {
            "Model": name,
            "n_params": result.n_params,
            "RMSE": f"{result.rmse:.3f}",
            "R²": f"{result.r2:.3f}",
            "CV-R²": f"{result.cv_r2:.3f}" if result.cv_r2 else "N/A",
            "AIC": f"{result.aic:.1f}",
            "BIC": f"{result.bic:.1f}",
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("AIC")
    print(comparison_df.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_dict = {name: result.to_dict() for name, result in results.items()}
    results_dict["data_info"] = {
        "n_total": len(df),
        "n_forward": len(fwd_df),
        "n_reverse": len(rev_df),
        "v_range": [float(df["v_obs"].min()), float(df["v_obs"].max())],
    }
    if true_params:
        results_dict["true_params"] = true_params

    results_path = output_dir / f"llrq_improved_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    fig_paths = create_improved_diagnostic_plots(df, results, output_dir, show=False)
    print(f"Saved diagnostic plots to: {fig_paths[0]}")
    print(f"Saved residual analysis to: {fig_paths[1]}")

    # Best model analysis
    print("\n" + "=" * 60)
    print("Best Model Analysis")
    print("=" * 60)
    best_name = comparison_df.iloc[0]["Model"]
    best_result = results[best_name]
    print(f"Best model: {best_name}")
    print(f"Parameters:")
    for pname, pval in zip(best_result.param_names, best_result.params):
        if best_result.param_std is not None and len(best_result.param_std) == len(best_result.params):
            idx = best_result.param_names.index(pname)
            print(f"  {pname}: {pval:.6f} ± {best_result.param_std[idx]:.6f}")
        else:
            print(f"  {pname}: {pval:.6f}")

    # Extract equilibrium constant if applicable
    if "constrained" in best_name and "w0" in best_result.param_names:
        idx = best_result.param_names.index("w0")
        w0 = best_result.params[idx]
        Keq_estimate = np.exp(-w0)
        print(f"\nEstimated Keq: {Keq_estimate:.6f}")
        if true_params and "keq" in true_params:
            print(f"True Keq: {true_params['keq']:.6f}")
            rel_error = abs(Keq_estimate - true_params["keq"]) / true_params["keq"] * 100
            print(f"Relative error: {rel_error:.2f}%")

    print("\n" + "=" * 60)
    print("Fitting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
