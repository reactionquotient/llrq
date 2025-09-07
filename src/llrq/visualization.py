"""Visualization tools for log-linear reaction quotient dynamics.

This module provides plotting functions for analyzing and visualizing
the dynamics of chemical reaction networks using the log-linear framework.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .llrq_dynamics import LLRQDynamics
from .solver import LLRQSolver


class LLRQVisualizer:
    """Visualization tools for LLRQ dynamics."""

    def __init__(self, solver: LLRQSolver):
        """Initialize visualizer.

        Args:
            solver: LLRQSolver instance
        """
        self.solver = solver
        self.dynamics = solver.dynamics
        self.network = solver.network

        # Set up default plotting style
        self._setup_style()

    def _setup_style(self):
        """Set up default matplotlib style."""
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            try:
                plt.style.use("seaborn-darkgrid")
            except OSError:
                plt.style.use("default")

    def plot_dynamics(
        self,
        solution: Dict[str, Any],
        species_to_plot: Optional[List[str]] = None,
        reactions_to_plot: Optional[List[str]] = None,
        log_scale: bool = False,
        compare_mass_action: bool = False,
        figsize: Tuple[float, float] = (12, 8),
    ) -> plt.Figure:
        """Plot concentration and reaction quotient dynamics.

        Args:
            solution: Solution dictionary from LLRQSolver.solve()
            species_to_plot: Species to include in plot (default: all)
            reactions_to_plot: Reactions to include in plot (default: all)
            log_scale: Use log scale for y-axis
            compare_mass_action: Compare with mass action kinetics
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Log-Linear Reaction Quotient Dynamics", fontsize=16)

        t = solution["time"]

        # Panel A: Species concentrations
        ax = axes[0, 0]
        if solution["concentrations"] is not None:
            c_t = solution["concentrations"]
            species_list = species_to_plot or self.network.species_ids[: min(6, len(self.network.species_ids))]

            for species_id in species_list:
                if species_id in self.network.species_to_idx:
                    i = self.network.species_to_idx[species_id]
                    ax.plot(t, c_t[:, i], label=species_id, linewidth=2)

            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.set_title("(A) Species Concentrations")
            ax.legend()
            if log_scale:
                ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "Concentrations not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("(A) Species Concentrations")

        ax.grid(True, alpha=0.3)

        # Panel B: Reaction quotients
        ax = axes[0, 1]
        Q_t = solution["reaction_quotients"]
        reaction_list = reactions_to_plot or self.network.reaction_ids[: min(4, len(self.network.reaction_ids))]

        for reaction_id in reaction_list:
            if reaction_id in self.network.reaction_to_idx:
                j = self.network.reaction_to_idx[reaction_id]
                ax.plot(t, Q_t[:, j], label=f"Q({reaction_id})", linewidth=2)

                # Add equilibrium line
                Keq_j = self.dynamics.Keq[j]
                ax.axhline(
                    y=Keq_j,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    label=f"K_eq({reaction_id})" if len(reaction_list) == 1 else None,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Reaction Quotient")
        ax.set_title("(B) Reaction Quotients")
        ax.legend()
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Panel C: Log deviations
        ax = axes[1, 0]
        x_t = solution["log_deviations"]

        for i, reaction_id in enumerate(reaction_list):
            if reaction_id in self.network.reaction_to_idx:
                j = self.network.reaction_to_idx[reaction_id]
                ax.plot(t, x_t[:, j], label=f"ln(Q/K_eq) {reaction_id}", linewidth=2)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("Log Deviation from Equilibrium")
        ax.set_title("(C) Log-Space Dynamics")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel D: Phase portrait (for 2D systems) or eigenmode analysis
        ax = axes[1, 1]

        if len(reaction_list) >= 2 and self.dynamics.n_reactions >= 2:
            # Phase portrait
            j1 = self.network.reaction_to_idx[reaction_list[0]]
            j2 = self.network.reaction_to_idx[reaction_list[1]]

            ax.plot(x_t[:, j1], x_t[:, j2], "b-", linewidth=2, alpha=0.7)
            ax.plot(x_t[0, j1], x_t[0, j2], "go", markersize=8, label="Start")
            ax.plot(x_t[-1, j1], x_t[-1, j2], "ro", markersize=8, label="End")
            ax.plot(0, 0, "k*", markersize=10, label="Equilibrium")

            ax.set_xlabel(f"ln(Q/K_eq) {reaction_list[0]}")
            ax.set_ylabel(f"ln(Q/K_eq) {reaction_list[1]}")
            ax.set_title("(D) Phase Portrait")
            ax.legend()
        else:
            # Eigenmode analysis
            eigen_info = self.dynamics.compute_eigenanalysis()

            ax.text(0.1, 0.9, "Eigenvalue Analysis:", transform=ax.transAxes, fontweight="bold")

            y_pos = 0.8
            for i, (eigenval, timescale) in enumerate(zip(eigen_info["eigenvalues"][:3], eigen_info["timescales"][:3])):
                if np.abs(eigenval.imag) < 1e-10:
                    text = f"λ_{i+1} = {eigenval.real:.3f}"
                else:
                    text = f"λ_{i+1} = {eigenval.real:.3f} ± {abs(eigenval.imag):.3f}i"

                ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=10)
                y_pos -= 0.15

            if eigen_info["is_stable"]:
                ax.text(0.1, 0.4, "✓ System is stable", transform=ax.transAxes, color="green", fontsize=12)
            else:
                ax.text(0.1, 0.4, "✗ System is unstable", transform=ax.transAxes, color="red", fontsize=12)

            if eigen_info["has_oscillations"]:
                ax.text(0.1, 0.25, "~ Oscillatory behavior", transform=ax.transAxes, color="blue", fontsize=12)

            ax.set_title("(D) Eigenanalysis")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_single_reaction(
        self,
        reaction_id: str,
        initial_concentrations: Dict[str, float],
        t_span: Union[Tuple[float, float], np.ndarray],
        external_drive: Optional[Callable] = None,
        compare_mass_action: bool = True,
        figsize: Tuple[float, float] = (12, 5),
    ) -> plt.Figure:
        """Plot dynamics for a single reaction.

        Args:
            reaction_id: Reaction to analyze
            initial_concentrations: Initial species concentrations
            t_span: Time span for simulation
            external_drive: External drive function u(t)
            compare_mass_action: Compare with mass action kinetics
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        # Solve LLRQ dynamics
        solution = self.solver.solve_single_reaction(reaction_id, initial_concentrations, t_span, external_drive)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        t = solution["time"]
        Q_llrq = solution["reaction_quotient"]
        concentrations = solution["concentrations"]

        # Get reaction equation for title
        equation = self.network.get_reaction_equation(reaction_id)
        fig.suptitle(f"Single Reaction Dynamics: {equation}", fontsize=14)

        # Panel A: Concentrations
        ax = axes[0]

        if concentrations:
            for species_id, conc_t in concentrations.items():
                ax.plot(t, conc_t, label=f"[{species_id}]", linewidth=2.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title("(A) Species Concentrations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel B: Reaction quotient
        ax = axes[1]

        ax.plot(t, Q_llrq, "b-", linewidth=2.5, label="Log-linear Q(t)")

        # Add equilibrium line
        Keq = solution["equilibrium_constant"]
        ax.axhline(y=Keq, color="b", linestyle="--", alpha=0.5, label="K_eq")

        # Compare with mass action if requested
        if compare_mass_action and concentrations:
            # This would require implementing mass action comparison
            # For now, just show the log-linear result
            pass

        ax.set_xlabel("Time")
        ax.set_ylabel("Reaction Quotient Q")
        ax.set_title("(B) Reaction Quotient Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add parameter info
        k = solution["relaxation_rate"]
        textstr = f"k = {k:.3f}\\nK_eq = {Keq:.3f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

        plt.tight_layout()
        return fig

    def plot_mass_action_comparison(
        self,
        solution: Dict[str, Any],
        mass_action_solution: Optional[Dict[str, Any]] = None,
        reaction_id: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6),
    ) -> plt.Figure:
        """Compare log-linear and mass action dynamics.

        Args:
            solution: LLRQ solution
            mass_action_solution: Mass action solution (if available)
            reaction_id: Specific reaction to compare
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("Log-Linear vs Mass Action Comparison", fontsize=14)

        t = solution["time"]

        if reaction_id:
            if reaction_id not in self.network.reaction_to_idx:
                raise ValueError(f"Reaction '{reaction_id}' not found")

            j = self.network.reaction_to_idx[reaction_id]
            Q_llrq = solution["reaction_quotients"][:, j]

            # Panel A: Reaction quotients
            ax = axes[0]
            ax.plot(t, Q_llrq, "b-", linewidth=2.5, label="Log-linear")

            if mass_action_solution:
                Q_ma = mass_action_solution["reaction_quotients"][:, j]
                ax.plot(t, Q_ma, "r--", linewidth=2, label="Mass action")

            Keq = self.dynamics.Keq[j]
            ax.axhline(y=Keq, color="gray", linestyle=":", alpha=0.7, label="K_eq")

            ax.set_xlabel("Time")
            ax.set_ylabel("Reaction Quotient")
            ax.set_title(f"(A) Q({reaction_id})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Panel B: Log space
            ax = axes[1]
            x_llrq = solution["log_deviations"][:, j]
            ax.plot(t, x_llrq, "b-", linewidth=2.5, label="Log-linear")

            if mass_action_solution:
                Q_ma = mass_action_solution["reaction_quotients"][:, j]
                x_ma = np.log(Q_ma) - np.log(Keq)
                ax.plot(t, x_ma, "r--", linewidth=2, label="Mass action")

            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.7)
            ax.set_xlabel("Time")
            ax.set_ylabel("ln(Q/K_eq)")
            ax.set_title("(B) Log Deviations")
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            # Plot first reaction by default
            if self.dynamics.n_reactions > 0:
                return self.plot_mass_action_comparison(solution, mass_action_solution, self.network.reaction_ids[0], figsize)

        plt.tight_layout()
        return fig

    def plot_external_drive_response(
        self,
        initial_conditions: Dict[str, float],
        drive_functions: List[Callable],
        drive_labels: List[str],
        t_span: Tuple[float, float] = (0, 10),
        figsize: Tuple[float, float] = (12, 8),
    ) -> plt.Figure:
        """Plot system response to different external drives.

        Args:
            initial_conditions: Initial concentrations
            drive_functions: List of external drive functions
            drive_labels: Labels for each drive function
            t_span: Time span for simulation
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Response to External Drives", fontsize=16)

        colors = ["blue", "red", "green", "orange", "purple"]

        t_eval = np.linspace(t_span[0], t_span[1], 1000)

        # Plot drive functions
        ax = axes[0, 0]
        for i, (drive_func, label) in enumerate(zip(drive_functions, drive_labels)):
            u_vals = [drive_func(t) for t in t_eval]
            # For multi-reaction case, plot first component
            if hasattr(u_vals[0], "__len__"):
                u_vals = [u[0] for u in u_vals]
            ax.plot(t_eval, u_vals, color=colors[i % len(colors)], label=label, linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("External Drive u(t)")
        ax.set_title("(A) Drive Functions")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Simulate responses
        solutions = []
        for drive_func in drive_functions:
            # Set external drive
            original_drive = self.dynamics.external_drive
            self.dynamics.external_drive = drive_func

            try:
                sol = self.solver.solve(initial_conditions, t_span, method="numerical")
                solutions.append(sol)
            except Exception as e:
                warnings.warn(f"Failed to solve with drive function: {e}")
                # Create a failure result instead of None
                solutions.append({"success": False, "error": str(e)})
            finally:
                # Restore original drive
                self.dynamics.external_drive = original_drive

        # Plot reaction quotient responses
        ax = axes[0, 1]
        for i, (sol, label) in enumerate(zip(solutions, drive_labels)):
            if sol and sol["success"]:
                Q_t = sol["reaction_quotients"]
                # Plot first reaction quotient
                ax.plot(sol["time"], Q_t[:, 0], color=colors[i % len(colors)], label=label, linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Reaction Quotient")
        ax.set_title("(B) Q(t) Response")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot log deviations
        ax = axes[1, 0]
        for i, (sol, label) in enumerate(zip(solutions, drive_labels)):
            if sol and sol["success"]:
                x_t = sol["log_deviations"]
                ax.plot(sol["time"], x_t[:, 0], color=colors[i % len(colors)], label=label, linewidth=2)

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("ln(Q/K_eq)")
        ax.set_title("(C) Log-Space Response")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot concentrations (if available)
        ax = axes[1, 1]
        for i, (sol, label) in enumerate(zip(solutions, drive_labels)):
            if sol and sol["success"] and sol["concentrations"] is not None:
                c_t = sol["concentrations"]
                # Plot first species
                ax.plot(
                    sol["time"],
                    c_t[:, 0],
                    color=colors[i % len(colors)],
                    label=f"{label} [{self.network.species_ids[0]}]",
                    linewidth=2,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title("(D) Concentration Response")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300, format: str = "png"):
        """Save figure to file.

        Args:
            fig: matplotlib Figure to save
            filename: Output filename
            dpi: Resolution for raster formats
            format: Output format ('png', 'pdf', 'svg', etc.)
        """
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches="tight")
        print(f"Figure saved to {filename}")
