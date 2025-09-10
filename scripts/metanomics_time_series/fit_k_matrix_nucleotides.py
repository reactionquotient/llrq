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
from scipy.signal import savgol_filter
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

    def estimate_derivatives(self, method="savgol", window_pts=7, poly=3, smoothing_factor=None):
        """Estimate time derivatives using various methods."""

        if method == "savgol":
            # Regrid to uniform time spacing to make SavGol legit
            t_uniform = np.linspace(self.t.min(), self.t.max(), len(self.t))
            self.y = np.column_stack([np.interp(t_uniform, self.t, self.y[:, i]) for i in range(self.y.shape[1])])
            self.t = t_uniform
            # Filter returns smoothed signal; derivative=True gives poly derivative; need dt scale
            dt = self.t[1] - self.t[0]
            self.dy_dt = np.column_stack(
                [
                    savgol_filter(self.y[:, i], window_length=window_pts, polyorder=poly, deriv=1, delta=dt, mode="interp")
                    for i in range(self.y.shape[1])
                ]
            )

            print(f"Derivative estimation ({method}):")
            print(f"  Window points: {window_pts}, Polynomial order: {poly}")
            print(f"  Uniform time step: {dt:.2f} seconds")

        elif method == "finite_diff":
            self.dy_dt = np.zeros_like(self.y)
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
            self.dy_dt = np.zeros_like(self.y)
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
                    return self.estimate_derivatives(method="finite_diff")

        else:
            raise ValueError(f"Unknown derivative method: {method}")

        print(f"  Mean |dy/dt|: {np.abs(self.dy_dt).mean():.4f}")
        print(f"  Max |dy/dt|: {np.abs(self.dy_dt).max():.4f}")
        if method == "spline" and smoothing_factor is not None:
            print(f"  Smoothing factor: {smoothing_factor:.4f}")

        return self

    def fit_k_matrix(self, alpha=1e-6, with_bias=True):
        """
        Fit symmetric K via ridge:
          dy/dt = -K y + b,   K = K.T
        Solve for params theta = [k11, k12, k22, (b1,b2 optional)]

        Parameters:
        - alpha: Ridge regularization parameter
        - with_bias: Whether to include bias terms b1, b2
        """
        Y = self.dy_dt  # (T, 2)
        Y1, Y2 = Y[:, 0], Y[:, 1]
        y1, y2 = self.y[:, 0], self.y[:, 1]

        # Build design for symmetric K:
        #   d1 = -(k11*y1 + k12*y2) + b1
        #   d2 = -(k12*y1 + k22*y2) + b2
        A1 = np.column_stack([-y1, -y2, 0.0 * np.ones_like(y1)])  # -> k11,k12,k22
        A2 = np.column_stack([0.0 * np.ones_like(y1), -y1, -y2])
        A = np.vstack([A1, A2])  # (2T, 3)
        rhs = np.concatenate([Y1, Y2])

        if with_bias:
            B1 = np.column_stack([np.ones_like(y1), np.zeros_like(y1)])  # b1
            B2 = np.column_stack([np.zeros_like(y1), np.ones_like(y1)])  # b2
            B = np.vstack([B1, B2])  # (2T, 2)
            A_full = np.hstack([A, B])  # (2T, 5)
            # Ridge normal equations
            reg = alpha * np.eye(A_full.shape[1])
            theta = np.linalg.solve(A_full.T @ A_full + reg, A_full.T @ rhs)
            k11, k12, k22, b1, b2 = theta
        else:
            reg = alpha * np.eye(3)
            theta = np.linalg.solve(A.T @ A + reg, A.T @ rhs)
            k11, k12, k22 = theta
            b1 = b2 = 0.0

        self.K = np.array([[k11, k12], [k12, k22]])
        self.b = np.array([b1, b2])

        # Predictions and metrics
        self.dy_dt_pred = -self.y @ self.K.T + self.b
        resid = self.dy_dt - self.dy_dt_pred
        self.rmse = np.sqrt(np.mean(resid**2))
        self.fit_scores = np.array(
            [
                1 - np.var(resid[:, 0]) / np.var(self.dy_dt[:, 0]),
                1 - np.var(resid[:, 1]) / np.var(self.dy_dt[:, 1]),
            ]
        )

        print(f"Symmetric K matrix fitting results:")
        print(f"  RMSE: {self.rmse:.5f}")
        print(f"  R² scores: {self.fit_scores}")
        print(f"  Regularization α: {alpha:.2e}")
        print(f"  With bias: {with_bias}")
        if with_bias:
            print(f"  Bias terms: [{b1:.5f}, {b2:.5f}]")

        return self

    def cross_validate_regularization(self, alpha_range=None, n_splits=3, with_bias=True):
        """Cross-validate regularization parameter using time series splits with symmetric K."""
        if alpha_range is None:
            alpha_range = np.logspace(-8, -2, 20)

        cv_scores = []
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for alpha in alpha_range:
            scores = []
            for train_idx, val_idx in tscv.split(self.y):
                # Get training and validation data
                y_train = self.y[train_idx]
                dy_dt_train = self.dy_dt[train_idx]
                y_val = self.y[val_idx]
                dy_dt_val = self.dy_dt[val_idx]

                # Fit symmetric K model on training data
                Y_train = dy_dt_train
                Y1_train, Y2_train = Y_train[:, 0], Y_train[:, 1]
                y1_train, y2_train = y_train[:, 0], y_train[:, 1]

                # Build design matrix for symmetric K
                A1 = np.column_stack([-y1_train, -y2_train, 0.0 * np.ones_like(y1_train)])
                A2 = np.column_stack([0.0 * np.ones_like(y1_train), -y1_train, -y2_train])
                A = np.vstack([A1, A2])
                rhs = np.concatenate([Y1_train, Y2_train])

                if with_bias:
                    B1 = np.column_stack([np.ones_like(y1_train), np.zeros_like(y1_train)])
                    B2 = np.column_stack([np.zeros_like(y1_train), np.ones_like(y1_train)])
                    B = np.vstack([B1, B2])
                    A_full = np.hstack([A, B])
                    reg = alpha * np.eye(A_full.shape[1])
                    theta = np.linalg.solve(A_full.T @ A_full + reg, A_full.T @ rhs)
                    k11, k12, k22, b1, b2 = theta
                else:
                    reg = alpha * np.eye(3)
                    theta = np.linalg.solve(A.T @ A + reg, A.T @ rhs)
                    k11, k12, k22 = theta
                    b1 = b2 = 0.0

                K_cv = np.array([[k11, k12], [k12, k22]])
                b_cv = np.array([b1, b2])

                # Predict on validation data
                dy_dt_pred = -y_val @ K_cv.T + b_cv

                # Compute validation score (R² for each reaction)
                val_scores = []
                for i in range(2):
                    ss_res = np.sum((dy_dt_val[:, i] - dy_dt_pred[:, i]) ** 2)
                    ss_tot = np.sum((dy_dt_val[:, i] - np.mean(dy_dt_val[:, i])) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                    val_scores.append(r2)

                scores.append(np.mean(val_scores))
            cv_scores.append(np.mean(scores))

        # Find best alpha
        best_idx = np.argmax(cv_scores)
        self.best_alpha = alpha_range[best_idx]
        self.cv_scores = cv_scores
        self.alpha_range = alpha_range

        print(f"Cross-validation results (symmetric K):")
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
        colors = ["orange", "purple"]
        for i, name in enumerate(self.reaction_names):
            color = colors[i] if i < len(colors) else None
            ax.plot(self.t, self.dy_dt[:, i], "o", label=f"{name} (data)", alpha=0.7, color=color)
            line = ax.plot(self.t, self.dy_dt_pred[:, i], "-", label=f"{name} (fit)", color=color)
            # Show bias if present
            if hasattr(self, "b") and abs(self.b[i]) > 1e-6:
                ax.axhline(self.b[i], linestyle=":", alpha=0.5, color=color, label=f"bias={self.b[i]:.4f}")
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
        for i, (name, score) in enumerate(zip(self.reaction_names, self.fit_scores)):
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

    # Use Savitzky-Golay filter for smoother derivatives (default method)
    fitter.estimate_derivatives(method="savgol", window_pts=7, poly=3)

    # Cross-validate regularization parameter with symmetric K
    fitter.cross_validate_regularization(alpha_range=np.logspace(-8, -2, 15), with_bias=True)

    # Fit symmetric K matrix with bias term and best regularization
    fitter.fit_k_matrix(alpha=fitter.best_alpha, with_bias=True)

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
