#!/usr/bin/env python
"""
LLRQ-based fitting using proper least squares optimization.

This script implements both Wiener (1-e^x) and Linear (x) models
for fitting reaction velocity data in the LLRQ framework.
"""

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import logsumexp


@dataclass
class FitResult:
    """Container for fitting results."""

    params: np.ndarray
    param_names: List[str]
    param_std: Optional[np.ndarray]
    residuals: np.ndarray
    jacobian: Optional[np.ndarray]
    rmse: float
    r2: float
    aic: float
    bic: float
    n_params: int
    n_data: int
    converged: bool
    message: str
    model_name: str
    constrained: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "constrained": self.constrained,
            "params": {name: float(val) for name, val in zip(self.param_names, self.params)},
            "param_std": {name: float(val) for name, val in zip(self.param_names, self.param_std)}
            if self.param_std is not None
            else None,
            "metrics": {
                "rmse": float(self.rmse),
                "r2": float(self.r2),
                "aic": float(self.aic),
                "bic": float(self.bic),
            },
            "fit_info": {
                "n_params": self.n_params,
                "n_data": self.n_data,
                "converged": self.converged,
                "message": self.message,
            },
        }


class LLRQModel(ABC):
    """Abstract base class for LLRQ models."""

    def __init__(self, constrained: bool = False):
        """
        Initialize model.

        Args:
            constrained: Whether to enforce stoichiometric constraints
        """
        self.constrained = constrained
        self.fitted = False
        self.result: Optional[FitResult] = None

    @abstractmethod
    def predict(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict velocities given concentrations and parameters."""
        pass

    @abstractmethod
    def get_initial_params(self, y: np.ndarray) -> np.ndarray:
        """Get initial parameter guess."""
        pass

    @abstractmethod
    def get_param_names(self) -> List[str]:
        """Get parameter names."""
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for optimization."""
        pass

    def residuals(self, params: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute residuals for least squares."""
        y_pred = self.predict(A, B, C, params)
        return y - y_pred

    def jacobian(self, params: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix numerically."""
        eps = 1e-8
        n_params = len(params)
        n_data = len(y)
        jac = np.zeros((n_data, n_params))

        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            y_plus = self.predict(A, B, C, params_plus)

            params_minus = params.copy()
            params_minus[i] -= eps
            y_minus = self.predict(A, B, C, params_minus)

            jac[:, i] = -(y_plus - y_minus) / (2 * eps)

        return jac

    def fit(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray, method: str = "trf", loss: str = "linear", **kwargs
    ) -> FitResult:
        """
        Fit model to data using least squares.

        Args:
            A, B, C: Concentration arrays
            y: Observed velocities
            method: Optimization method ('trf', 'dogbox', 'lm')
            loss: Loss function ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan')
            **kwargs: Additional arguments for least_squares

        Returns:
            FitResult object
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        A = np.maximum(A, eps)
        B = np.maximum(B, eps)
        C = np.maximum(C, eps)

        # Get initial parameters
        p0 = self.get_initial_params(y)

        # Get bounds
        bounds = self.get_bounds()

        # Run optimization
        result = least_squares(
            self.residuals, p0, args=(A, B, C, y), method=method, loss=loss, bounds=bounds, jac=self.jacobian, **kwargs
        )

        # Extract parameters and compute statistics
        params = result.x
        residuals = result.fun
        jacobian = result.jac

        # Compute covariance and standard errors if converged
        param_std = None
        if result.success and jacobian is not None:
            try:
                # Covariance matrix: (J'J)^{-1} * sigma^2
                JtJ = jacobian.T @ jacobian
                sigma2 = np.sum(residuals**2) / (len(y) - len(params))
                cov = np.linalg.inv(JtJ) * sigma2
                param_std = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                warnings.warn("Could not compute parameter uncertainties (singular matrix)")

        # Compute metrics
        y_pred = self.predict(A, B, C, params)
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

        # Store result
        self.result = FitResult(
            params=params,
            param_names=self.get_param_names(),
            param_std=param_std,
            residuals=residuals,
            jacobian=jacobian,
            rmse=rmse,
            r2=r2,
            aic=aic,
            bic=bic,
            n_params=k,
            n_data=n,
            converged=result.success,
            message=result.message,
            model_name=self.__class__.__name__,
            constrained=self.constrained,
        )

        self.fitted = True
        return self.result

    def predict_fitted(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Predict using fitted parameters."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        return self.predict(A, B, C, self.result.params)


class WienerModel(LLRQModel):
    """Wiener model: v = α(1 - e^x) where x = ln(Q/Keq)."""

    def get_param_names(self) -> List[str]:
        if self.constrained:
            return ["log_alpha", "w0"]
        else:
            return ["log_alpha", "theta0", "thetaA", "thetaB", "thetaC"]

    def get_initial_params(self, y: np.ndarray) -> np.ndarray:
        alpha0 = max(np.std(y), 1e-6)
        if self.constrained:
            return np.array([np.log(alpha0), 0.0])
        else:
            return np.array([np.log(alpha0), 0.0, -0.5, -0.5, 1.0])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.constrained:
            # log_alpha can be any value, w0 can be any value
            return ([-np.inf, -np.inf], [np.inf, np.inf])
        else:
            # All parameters unconstrained
            return ([-np.inf] * 5, [np.inf] * 5)

    def predict(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, params: np.ndarray) -> np.ndarray:
        logA = np.log(A)
        logB = np.log(B)
        logC = np.log(C)

        if self.constrained:
            # Stoichiometry-constrained: x = ln(Q/Keq) = -ln[A] - ln[B] + 2ln[C] + w0
            log_alpha, w0 = params
            x = w0 - logA - logB + 2.0 * logC
        else:
            # Unconstrained
            log_alpha, theta0, thetaA, thetaB, thetaC = params
            x = theta0 + thetaA * logA + thetaB * logB + thetaC * logC

        alpha = np.exp(log_alpha)
        return alpha * (1.0 - np.exp(x))


class LinearModel(LLRQModel):
    """Linear model: v = βx where x = ln(Q/Keq)."""

    def get_param_names(self) -> List[str]:
        if self.constrained:
            return ["beta", "w0"]
        else:
            return ["beta", "theta0", "thetaA", "thetaB", "thetaC"]

    def get_initial_params(self, y: np.ndarray) -> np.ndarray:
        beta0 = np.std(y) / 2.0
        if self.constrained:
            return np.array([beta0, 0.0])
        else:
            return np.array([beta0, 0.0, -0.5, -0.5, 1.0])

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.constrained:
            # All parameters unconstrained for flexibility
            return ([-np.inf, -np.inf], [np.inf, np.inf])
        else:
            # All parameters unconstrained
            return ([-np.inf] * 5, [np.inf] * 5)

    def predict(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, params: np.ndarray) -> np.ndarray:
        logA = np.log(A)
        logB = np.log(B)
        logC = np.log(C)

        if self.constrained:
            # Stoichiometry-constrained: x = ln(Q/Keq) = -ln[A] - ln[B] + 2ln[C] + w0
            beta, w0 = params
            x = w0 - logA - logB + 2.0 * logC
        else:
            # Unconstrained
            beta, theta0, thetaA, thetaB, thetaC = params
            x = theta0 + thetaA * logA + thetaB * logB + thetaC * logC

        return beta * x


def load_or_generate_data(output_dir: Path) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Load existing synthetic data or use the existing generator."""
    # Look for most recent synthetic data
    fwd_files = sorted(output_dir.glob("synthetic_forward_*.csv"))
    rev_files = sorted(output_dir.glob("synthetic_reverse_*.csv"))

    if fwd_files and rev_files:
        # Load most recent
        fwd_df = pd.read_csv(fwd_files[-1])
        rev_df = pd.read_csv(rev_files[-1])
        df = pd.concat([fwd_df, rev_df], ignore_index=True)

        # Try to load corresponding parameters
        params = None
        param_files = sorted(output_dir.glob("cm_params_*.json"))
        if param_files:
            with open(param_files[-1], "r") as f:
                params = json.load(f)

        return df, params
    else:
        raise FileNotFoundError("No synthetic data found. Please run synthetic_data_with_fitting.py first.")


def fit_all_models(df: pd.DataFrame, verbose: bool = True) -> Dict[str, FitResult]:
    """Fit all model variants to the data."""
    # Extract data
    A = df["A_mM"].values
    B = df["B_mM"].values
    C = df["C_mM"].values
    y = df["v_obs"].values

    # Models to fit
    models = {
        "Wiener_unconstrained": WienerModel(constrained=False),
        "Wiener_constrained": WienerModel(constrained=True),
        "Linear_unconstrained": LinearModel(constrained=False),
        "Linear_constrained": LinearModel(constrained=True),
    }

    results = {}
    for name, model in models.items():
        if verbose:
            print(f"\nFitting {name}...")

        result = model.fit(A, B, C, y, method="trf", loss="soft_l1")
        results[name] = result

        if verbose:
            print(f"  Converged: {result.converged}")
            print(f"  RMSE: {result.rmse:.6f}")
            print(f"  R²: {result.r2:.6f}")
            print(f"  AIC: {result.aic:.2f}")
            print(f"  BIC: {result.bic:.2f}")
            print(f"  Parameters:")
            for pname, pval in zip(result.param_names, result.params):
                if result.param_std is not None:
                    pstd = result.param_std[result.param_names.index(pname)]
                    print(f"    {pname}: {pval:.6f} ± {pstd:.6f}")
                else:
                    print(f"    {pname}: {pval:.6f}")

    return results


def create_diagnostic_plots(df: pd.DataFrame, results: Dict[str, FitResult], output_dir: Path, show: bool = True):
    """Create diagnostic plots for model comparison."""

    # Extract data
    A = df["A_mM"].values
    B = df["B_mM"].values
    C = df["C_mM"].values
    y = df["v_obs"].values

    # Add epsilon to avoid log(0)
    eps = 1e-12
    A = np.maximum(A, eps)
    B = np.maximum(B, eps)
    C = np.maximum(C, eps)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Observed vs Predicted for all models
    for i, (name, result) in enumerate(results.items(), 1):
        ax = plt.subplot(3, 4, i)

        # Recreate model to get predictions
        if "Wiener" in name:
            model = WienerModel(constrained=result.constrained)
        else:
            model = LinearModel(constrained=result.constrained)

        y_pred = model.predict(A, B, C, result.params)

        ax.scatter(y, y_pred, alpha=0.5, s=10)

        # Add diagonal line
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, "k-", alpha=0.5, lw=1)

        ax.set_xlabel("Observed v")
        ax.set_ylabel("Predicted v")
        ax.set_title(f"{name}\nR²={result.r2:.4f}, RMSE={result.rmse:.4f}")
        ax.grid(True, alpha=0.3)

    # 2. Residual plots
    for i, (name, result) in enumerate(results.items(), 1):
        ax = plt.subplot(3, 4, 4 + i)

        if "Wiener" in name:
            model = WienerModel(constrained=result.constrained)
        else:
            model = LinearModel(constrained=result.constrained)

        y_pred = model.predict(A, B, C, result.params)
        residuals = y - y_pred

        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Predicted v")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{name} Residuals")
        ax.grid(True, alpha=0.3)

    # 3. Q-Q plots for residuals
    for i, (name, result) in enumerate(results.items(), 1):
        ax = plt.subplot(3, 4, 8 + i)

        if "Wiener" in name:
            model = WienerModel(constrained=result.constrained)
        else:
            model = LinearModel(constrained=result.constrained)

        y_pred = model.predict(A, B, C, result.params)
        residuals = y - y_pred

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{name} Q-Q Plot")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"llrq_fit_diagnostics_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig_path


def create_comparison_table(results: Dict[str, FitResult]) -> pd.DataFrame:
    """Create comparison table of all models."""
    rows = []
    for name, result in results.items():
        row = {
            "Model": name,
            "Constrained": result.constrained,
            "n_params": result.n_params,
            "RMSE": result.rmse,
            "R²": result.r2,
            "AIC": result.aic,
            "BIC": result.bic,
            "Converged": result.converged,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("AIC")
    return df


def main():
    """Main execution function."""
    # Setup
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("LLRQ-based Fitting with Least Squares")
    print("=" * 60)

    # Load data
    print("\nLoading synthetic data...")
    df, true_params = load_or_generate_data(output_dir)
    print(f"Loaded {len(df)} data points")

    if true_params:
        print("\nTrue parameters (CM model):")
        for key in ["kM_A", "kM_B", "kM_C", "keq", "kV"]:
            if key in true_params:
                print(f"  {key}: {true_params[key]:.6f}")

    # Fit all models
    results = fit_all_models(df, verbose=True)

    # Create comparison table
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    comparison_df = create_comparison_table(results)
    print(comparison_df.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    results_dict = {name: result.to_dict() for name, result in results.items()}
    results_dict["data_info"] = {
        "n_points": len(df),
        "n_forward": len(df[df["protocol"] == "forward_A_var"]),
        "n_reverse": len(df[df["protocol"] == "reverse_C_var"]),
    }
    if true_params:
        results_dict["true_params"] = true_params

    results_path = output_dir / f"llrq_fit_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Save comparison table
    table_path = output_dir / f"llrq_fit_comparison_{timestamp}.csv"
    comparison_df.to_csv(table_path, index=False)
    print(f"Saved comparison table to: {table_path}")

    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    fig_path = create_diagnostic_plots(df, results, output_dir, show=True)
    print(f"Saved diagnostic plots to: {fig_path}")

    # Best model analysis
    print("\n" + "=" * 60)
    print("Best Model Analysis (by AIC)")
    print("=" * 60)
    best_name = comparison_df.iloc[0]["Model"]
    best_result = results[best_name]
    print(f"Best model: {best_name}")
    print(f"AIC: {best_result.aic:.2f}")
    print(f"BIC: {best_result.bic:.2f}")
    print(f"R²: {best_result.r2:.6f}")
    print(f"RMSE: {best_result.rmse:.6f}")

    if best_result.constrained and "Wiener" in best_name:
        # Extract equilibrium constant estimate
        w0 = best_result.params[1]
        Keq_estimate = np.exp(-w0)
        print(f"\nEstimated Keq: {Keq_estimate:.6f}")
        if true_params and "keq" in true_params:
            print(f"True Keq: {true_params['keq']:.6f}")
            print(f"Relative error: {abs(Keq_estimate - true_params['keq'])/true_params['keq']*100:.2f}%")

    print("\n" + "=" * 60)
    print("Fitting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
