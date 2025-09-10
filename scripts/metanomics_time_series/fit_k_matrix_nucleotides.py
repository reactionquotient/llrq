#!/usr/bin/env python3
"""
LLRQ K Matrix Fitting for Nucleotide Time Series Data

This script implements log-linear reaction quotient (LLRQ) theory to fit the K matrix
from experimental nucleotide concentration time series data using regularized least squares.

The approach:
1. Define nucleotide reaction network (ATP ⇌ ADP ⇌ AMP)
2. Compute log-deviations from equilibrium: x = ln(c/c_eq)
3. Project to reaction space: y = N^T x
4. Fit dynamics: dy/dt = -K*y + u(t) using regularized least squares
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NucleotideKFitter:
    """Fits LLRQ K matrix for nucleotide dynamics using regularized least squares."""

    def __init__(self):
        # Define stoichiometry matrix for nucleotide reactions
        # R1: ATP -> ADP + Pi
        # R2: ADP -> AMP + Pi
        self.N = np.array(
            [
                [-1, 0],  # ATP
                [1, -1],  # ADP
                [0, 1],  # AMP
            ]
        )
        self.species_names = ["ATP", "ADP", "AMP"]
        self.reaction_names = ["ATP->ADP", "ADP->AMP"]

    def load_data(self, csv_path, pulse_id="pulse_1"):
        """Load nucleotide time series data."""
        df = pd.read_csv(csv_path)
        pulse_data = df[df.dataset == pulse_id].sort_values("time_s")

        # Extract concentrations
        times = pulse_data["time_s"].values
        conc_matrix = np.vstack(
            [pulse_data["ATP_mean"].values, pulse_data["ADP_mean"].values, pulse_data["AMP_mean"].values]
        ).T

        # Remove NaN rows
        mask = ~np.isnan(conc_matrix).any(axis=1)
        self.t_raw = times[mask]
        self.c_raw = conc_matrix[mask]

        print(f"Loaded data: {len(self.t_raw)} time points")
        return self

    def preprocess_data(self, t_min=0, equilibrium_method="last_points", n_eq_points=3):
        """Preprocess data for LLRQ fitting."""
        # Focus on post-pulse relaxation dynamics
        post_mask = self.t_raw >= t_min
        self.t = self.t_raw[post_mask]
        self.c = self.c_raw[post_mask]

        # Estimate equilibrium concentrations
        if equilibrium_method == "last_points":
            self.c_eq = self.c[-n_eq_points:].mean(axis=0)
        elif equilibrium_method == "mean":
            self.c_eq = self.c.mean(axis=0)
        else:
            raise ValueError(f"Unknown equilibrium method: {equilibrium_method}")

        # Compute log-deviations from equilibrium
        self.x = np.log(self.c / self.c_eq)

        # Project to reaction space
        self.y = self.x @ self.N  # Shape: (n_times, n_reactions)

        print(f"Preprocessed data:")
        print(f"  Time range: {self.t.min():.1f} to {self.t.max():.1f} seconds")
        print(f"  Equilibrium: ATP={self.c_eq[0]:.3f}, ADP={self.c_eq[1]:.3f}, AMP={self.c_eq[2]:.3f}")
        print(f"  Log-deviation range: [{self.x.min():.3f}, {self.x.max():.3f}]")

        return self

    def estimate_derivatives(self, method="finite_diff", smoothing_factor=None):
        """Estimate time derivatives using various methods."""
        self.dy_dt = np.zeros_like(self.y)

        if method == "finite_diff":
            # Use centered finite differences with irregular spacing
            for i in range(self.y.shape[1]):
                # Forward difference for first point
                self.dy_dt[0, i] = (self.y[1, i] - self.y[0, i]) / (self.t[1] - self.t[0])

                # Centered differences for interior points
                for j in range(1, len(self.t) - 1):
                    dt_left = self.t[j] - self.t[j - 1]
                    dt_right = self.t[j + 1] - self.t[j]
                    dt_total = self.t[j + 1] - self.t[j - 1]

                    # Weighted finite difference for irregular grid
                    self.dy_dt[j, i] = (
                        self.y[j + 1, i] * dt_left**2
                        - self.y[j - 1, i] * dt_right**2
                        + self.y[j, i] * (dt_right**2 - dt_left**2)
                    ) / (dt_left * dt_right * dt_total)

                # Backward difference for last point
                self.dy_dt[-1, i] = (self.y[-1, i] - self.y[-2, i]) / (self.t[-1] - self.t[-2])

        elif method == "spline":
            if smoothing_factor is None:
                # More conservative smoothing
                data_range = np.ptp(self.y, axis=0).mean()
                smoothing_factor = data_range * len(self.t) * 0.5  # Increased smoothing

            for i in range(self.y.shape[1]):
                try:
                    spline = UnivariateSpline(self.t, self.y[:, i], s=smoothing_factor)
                    self.dy_dt[:, i] = spline.derivative()(self.t)
                except Exception as e:
                    print(f"Warning: Spline fitting failed for reaction {i}: {e}")
                    # Fallback to finite differences
                    method = "finite_diff"
                    return self.estimate_derivatives(method="finite_diff")
        else:
            raise ValueError(f"Unknown derivative method: {method}")

        print(f"Derivative estimation ({method}):")
        print(f"  Mean |dy/dt|: {np.abs(self.dy_dt).mean():.4f}")
        print(f"  Max |dy/dt|: {np.abs(self.dy_dt).max():.4f}")
        if method == "spline" and smoothing_factor is not None:
            print(f"  Smoothing factor: {smoothing_factor:.4f}")

        return self

    def fit_k_matrix(self, alpha=1e-6, drive_model="zero"):
        """Fit K matrix using regularized least squares.

        Solves: dy/dt = -K*y + u(t)

        Parameters:
        - alpha: Ridge regularization parameter
        - drive_model: 'zero' (u=0) or 'exponential' (u=u0*exp(-t/tau))
        """
        n_times, n_reactions = self.y.shape

        if drive_model == "zero":
            # Simple case: dy/dt = -K*y
            # Reshape for sklearn: each time point is a sample
            X = -self.y  # Shape: (n_times, n_reactions)
            Y = self.dy_dt  # Shape: (n_times, n_reactions)

            # Fit separate ridge regression for each reaction
            self.K = np.zeros((n_reactions, n_reactions))
            self.fit_scores = []

            for i in range(n_reactions):
                ridge = Ridge(alpha=alpha, fit_intercept=False)
                ridge.fit(X, Y[:, i])
                self.K[i, :] = ridge.coef_
                self.fit_scores.append(ridge.score(X, Y[:, i]))

        elif drive_model == "exponential":
            # More complex: dy/dt = -K*y + u0*exp(-t/tau)
            # Need to estimate u0 and tau along with K
            raise NotImplementedError("Exponential drive model not yet implemented")
        else:
            raise ValueError(f"Unknown drive model: {drive_model}")

        # Compute predicted dynamics
        self.dy_dt_pred = -self.y @ self.K.T

        # Fit quality metrics
        residuals = self.dy_dt - self.dy_dt_pred
        self.rmse = np.sqrt(np.mean(residuals**2))
        self.r2_scores = np.array(self.fit_scores)

        print(f"K matrix fitting results:")
        print(f"  RMSE: {self.rmse:.5f}")
        print(f"  R² scores: {self.r2_scores}")
        print(f"  Regularization α: {alpha:.2e}")

        return self

    def cross_validate_regularization(self, alpha_range=None, n_splits=3):
        """Cross-validate regularization parameter using time series splits."""
        if alpha_range is None:
            alpha_range = np.logspace(-8, -2, 20)

        cv_scores = []
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for alpha in alpha_range:
            scores = []
            for train_idx, val_idx in tscv.split(self.y):
                # Fit on training data
                X_train = -self.y[train_idx]
                Y_train = self.dy_dt[train_idx]
                X_val = -self.y[val_idx]
                Y_val = self.dy_dt[val_idx]

                # Fit and validate each reaction
                val_scores = []
                for i in range(self.y.shape[1]):
                    ridge = Ridge(alpha=alpha, fit_intercept=False)
                    ridge.fit(X_train, Y_train[:, i])
                    val_score = ridge.score(X_val, Y_val[:, i])
                    val_scores.append(val_score)

                scores.append(np.mean(val_scores))
            cv_scores.append(np.mean(scores))

        # Find best alpha
        best_idx = np.argmax(cv_scores)
        self.best_alpha = alpha_range[best_idx]
        self.cv_scores = cv_scores
        self.alpha_range = alpha_range

        print(f"Cross-validation results:")
        print(f"  Best α: {self.best_alpha:.2e}")
        print(f"  Best CV score: {cv_scores[best_idx]:.4f}")

        return self

    def analyze_k_matrix(self):
        """Analyze properties of fitted K matrix."""
        # Eigenvalue analysis
        eigenvals, eigenvecs = np.linalg.eig(self.K)
        self.eigenvals = eigenvals
        self.eigenvecs = eigenvecs

        # Relaxation timescales
        real_eigenvals = eigenvals.real
        positive_eigenvals = real_eigenvals[real_eigenvals > 1e-10]
        self.timescales = 1.0 / positive_eigenvals if len(positive_eigenvals) > 0 else []

        print(f"\\nK matrix analysis:")
        print(f"  Shape: {self.K.shape}")
        print(f"  Eigenvalues: {eigenvals}")
        print(f"  Timescales: {self.timescales} seconds")

        # Check for physical consistency
        is_stable = np.all(eigenvals.real > -1e-10)
        print(f"  Stable (eigenvals ≥ 0): {is_stable}")

        return self

    def plot_results(self, figsize=(15, 10)):
        """Create comprehensive plots of fitting results."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 1. Concentration trajectories
        ax = axes[0, 0]
        for i, name in enumerate(self.species_names):
            ax.plot(self.t, self.c[:, i], "o-", label=f"{name}", alpha=0.7)
            ax.axhline(self.c_eq[i], linestyle="--", alpha=0.5)
        ax.axvline(0, color="red", linestyle=":", alpha=0.5, label="Pulse")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration (μmol/gDW)")
        ax.set_title("Nucleotide Concentrations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Log-deviations
        ax = axes[0, 1]
        for i, name in enumerate(self.species_names):
            ax.plot(self.t, self.x[:, i], "o-", label=f"{name}", alpha=0.7)
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ln(c/c_eq)")
        ax.set_title("Log-Deviations from Equilibrium")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Reaction quotient dynamics
        ax = axes[0, 2]
        for i, name in enumerate(self.reaction_names):
            ax.plot(self.t, self.y[:, i], "o-", label=f"{name}", alpha=0.7)
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ln(Q/K_eq)")
        ax.set_title("Reaction Quotient Log-Deviations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Derivative fitting
        ax = axes[1, 0]
        for i, name in enumerate(self.reaction_names):
            ax.plot(self.t, self.dy_dt[:, i], "o", label=f"{name} (data)", alpha=0.7)
            ax.plot(self.t, self.dy_dt_pred[:, i], "-", label=f"{name} (fit)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("d(ln Q)/dt")
        ax.set_title("Derivative Fitting")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Cross-validation curve (if available)
        ax = axes[1, 1]
        if hasattr(self, "cv_scores"):
            ax.semilogx(self.alpha_range, self.cv_scores, "o-")
            ax.axvline(self.best_alpha, color="red", linestyle="--", label=f"Best α = {self.best_alpha:.2e}")
            ax.set_xlabel("Regularization α")
            ax.set_ylabel("CV Score")
            ax.set_title("Cross-Validation")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Cross-validation\\nnot performed", ha="center", va="center", transform=ax.transAxes)

        # 6. K matrix heatmap
        ax = axes[1, 2]
        im = ax.imshow(self.K, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(len(self.reaction_names)))
        ax.set_yticks(range(len(self.reaction_names)))
        ax.set_xticklabels(self.reaction_names, rotation=45)
        ax.set_yticklabels(self.reaction_names)
        ax.set_title("K Matrix")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Add text annotations to heatmap
        for i in range(len(self.reaction_names)):
            for j in range(len(self.reaction_names)):
                ax.text(j, i, f"{self.K[i,j]:.3f}", ha="center", va="center", color="white", weight="bold")

        plt.tight_layout()
        return fig

    def print_summary(self):
        """Print comprehensive summary of fitting results."""
        print("\\n" + "=" * 60)
        print("LLRQ K MATRIX FITTING SUMMARY")
        print("=" * 60)

        print(f"\\nData Summary:")
        print(f"  Time points: {len(self.t)}")
        print(f"  Time range: {self.t.min():.1f} to {self.t.max():.1f} seconds")
        print(f"  Species: {', '.join(self.species_names)}")
        print(f"  Reactions: {', '.join(self.reaction_names)}")

        print(f"\\nEquilibrium Concentrations (μmol/gDW):")
        for i, name in enumerate(self.species_names):
            print(f"  {name}: {self.c_eq[i]:.3f}")

        print(f"\\nFitted K Matrix:")
        print(f"  Shape: {self.K.shape}")
        for i, row_name in enumerate(self.reaction_names):
            row_str = "  " + row_name + ": ["
            row_str += ", ".join([f"{self.K[i,j]:8.5f}" for j in range(self.K.shape[1])])
            row_str += "]"
            print(row_str)

        print(f"\\nEigenvalue Analysis:")
        for i, (val, tscale) in enumerate(zip(self.eigenvals, self.timescales)):
            print(f"  λ{i+1}: {val:.5f} → τ = {tscale:.1f} seconds")

        print(f"\\nFit Quality:")
        print(f"  RMSE: {self.rmse:.5f}")
        for i, (name, score) in enumerate(zip(self.reaction_names, self.r2_scores)):
            print(f"  R² ({name}): {score:.4f}")

        if hasattr(self, "best_alpha"):
            print(f"\\nRegularization:")
            print(f"  Best α: {self.best_alpha:.2e}")


def main():
    """Main function to run K matrix fitting analysis."""
    print("LLRQ K Matrix Fitting for Nucleotide Dynamics")
    print("=" * 50)

    # Initialize fitter
    fitter = NucleotideKFitter()

    # Load and process data
    csv_path = "nucleotides_timeseries.csv"
    fitter.load_data(csv_path, pulse_id="pulse_1")
    fitter.preprocess_data(t_min=0)

    # Try finite differences first (more stable for noisy data)
    fitter.estimate_derivatives(method="finite_diff")

    # Cross-validate regularization parameter
    fitter.cross_validate_regularization(alpha_range=np.logspace(-8, -2, 15))

    # Fit K matrix with best regularization
    fitter.fit_k_matrix(alpha=fitter.best_alpha)

    # Analyze results
    fitter.analyze_k_matrix()

    # Print summary
    fitter.print_summary()

    # Create plots
    fig = fitter.plot_results()
    plt.savefig("nucleotide_k_matrix_fit.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\\nAnalysis complete! Results saved to 'nucleotide_k_matrix_fit.png'")

    return fitter


if __name__ == "__main__":
    fitter = main()
