"""LLRQ control strategies.

This module provides control algorithms based on LLRQ theory,
separated from the dynamics simulation.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .solver import LLRQSolver


class LLRQController:
    """Controller for LLRQ systems.

    Computes control inputs based on LLRQ theory without simulating dynamics.
    Can be used with either linear LLRQ simulation or mass action simulation.
    """

    def __init__(self, solver: LLRQSolver, controlled_reactions: Optional[list] = None):
        """Initialize LLRQ controller.

        Args:
            solver: LLRQSolver with computed basis matrices
            controlled_reactions: List of reaction IDs or indices to control.
                                If None, controls all reactions.
        """
        self.solver = solver
        self.network = solver.network
        self.B = solver._B  # Basis for Im(S^T)
        self.rankS = solver._rankS

        # Reduced dynamics matrix
        K = solver.dynamics.K
        self.K_red = self.B.T @ K @ self.B
        self.A = -self.K_red

        # Set up control actuation
        if controlled_reactions is None:
            controlled_reactions = list(range(len(self.network.reaction_ids)))

        self.controlled_reactions = controlled_reactions
        self._setup_actuation_matrix()

    def _setup_actuation_matrix(self):
        """Set up actuation matrix G mapping control inputs to reactions."""
        r = len(self.network.reaction_ids)

        # Convert reaction IDs to indices
        idx = []
        for rid in self.controlled_reactions:
            if isinstance(rid, str):
                idx.append(self.network.reaction_to_idx[rid])
            else:
                idx.append(int(rid))

        self.controlled_indices = idx
        m = len(idx)

        # Build selection matrix G (r x m)
        self.G = np.zeros((r, m))
        for j, k in enumerate(idx):
            self.G[k, j] = 1.0

        # Reduced input matrix
        self.B_red = self.B.T @ self.G

        # Check controllability
        if np.linalg.matrix_rank(self.B_red) < min(self.rankS, m):
            import warnings

            warnings.warn(
                "Reduced input matrix appears rank-deficient. " "Selected reactions may not fully control the system."
            )

    def compute_steady_state_control(self, y_target: np.ndarray) -> np.ndarray:
        """Compute analytical steady-state control for target reduced state.

        At steady state: 0 = A*y + B*u, so u = -pinv(B)*A*y_target

        Args:
            y_target: Target reduced state (rankS,)

        Returns:
            Steady-state control input (m,)
        """
        # u_red = -A @ y_target (in reduced space)
        u_red_ss = -self.A @ y_target

        # Map to control space: u = pinv(B_red) @ u_red
        u_ss = np.linalg.pinv(self.B_red) @ u_red_ss

        return u_ss

    def compute_feedback_control(self, y_current: np.ndarray, y_target: np.ndarray, feedback_gain: float = 1.0) -> np.ndarray:
        """Compute proportional feedback control.

        Args:
            y_current: Current reduced state
            y_target: Target reduced state
            feedback_gain: Proportional gain

        Returns:
            Feedback control input
        """
        error = y_current - y_target
        u_feedback = -feedback_gain * np.linalg.pinv(self.B_red) @ error
        return u_feedback

    def compute_control(
        self, Q_current: np.ndarray, Q_target: np.ndarray, feedback_gain: float = 1.0, feedforward: bool = True
    ) -> np.ndarray:
        """Compute total control input from reaction quotients.

        Args:
            Q_current: Current reaction quotients
            Q_target: Target reaction quotients
            feedback_gain: Proportional feedback gain
            feedforward: Include steady-state feedforward

        Returns:
            Total control input
        """
        # Convert Q to reduced coordinates
        y_current = self.reaction_quotients_to_reduced_state(Q_current)
        y_target = self.reaction_quotients_to_reduced_state(Q_target)

        u_total = np.zeros(len(self.controlled_reactions))

        # Feedforward (steady-state) control
        if feedforward:
            u_ff = self.compute_steady_state_control(y_target)
            u_total += u_ff

        # Feedback control
        u_fb = self.compute_feedback_control(y_current, y_target, feedback_gain)
        u_total += u_fb

        return u_total

    def reaction_quotients_to_reduced_state(self, Q: np.ndarray) -> np.ndarray:
        """Convert reaction quotients to reduced state coordinates.

        Args:
            Q: Reaction quotients (r,)

        Returns:
            Reduced state y (rankS,)
        """
        # x = ln(Q/Keq)
        x = np.log(Q) - self.solver._lnKeq_consistent

        # Project to reduced space: y = B^T @ P @ x
        x_projected = self.solver._P @ x
        y = self.B.T @ x_projected

        return y

    def reduced_state_to_reaction_quotients(self, y: np.ndarray) -> np.ndarray:
        """Convert reduced state to reaction quotients.

        Args:
            y: Reduced state (rankS,)

        Returns:
            Reaction quotients Q (r,)
        """
        # x = B @ y (in projected space)
        x = self.B @ y

        # Q = Keq * exp(x)
        Q = self.solver._Keq_consistent * np.exp(x)

        return Q

    def compute_target_concentrations(self, Q_target: np.ndarray, initial_concentrations: np.ndarray) -> np.ndarray:
        """Compute species concentrations corresponding to target reaction quotients.

        Args:
            Q_target: Target reaction quotients
            initial_concentrations: Initial concentrations for constraint

        Returns:
            Target concentrations
        """
        return self.solver._compute_concentrations_from_reduced(
            Q_target[None, :], initial_concentrations, enforce_conservation=True
        )

    def compute_control_entropy_metric(self, L: np.ndarray) -> np.ndarray:
        """Compute control entropy metric matrix M = K^{-T} L K^{-1}.

        This matrix defines the entropy cost metric σ_u(u) = u^T M u in the
        quasi-steady approximation where x ≈ K^{-1} u.

        Args:
            L: Onsager conductance matrix (r x r) where r is number of reactions

        Returns:
            Control entropy metric matrix M (r x r)
        """
        K = self.solver.dynamics.K

        # Compute M = K^{-T} L K^{-1} using solves to avoid explicit inverses
        Z = np.linalg.solve(K.T, L)  # Z = K^{-T} L
        M = np.linalg.solve(K, Z.T).T  # M = Z K^{-1}

        # Ensure symmetry (should already be symmetric but numerical errors)
        M = 0.5 * (M + M.T)

        return M

    def compute_steady_state_entropy_rate(self, u: np.ndarray, M: np.ndarray) -> float:
        """Compute steady-state entropy production rate σ_u = u^T M u.

        Args:
            u: Control input (r,)
            M: Control entropy metric matrix (r x r)

        Returns:
            Entropy production rate
        """
        return float(u.T @ M @ u)

    def compute_entropy_aware_steady_state_control(
        self,
        x_target: np.ndarray,
        L: np.ndarray,
        entropy_weight: float = 1.0,
        controlled_reactions_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute steady-state control that trades off target tracking vs entropy production,
        without forming any explicit matrix inverses.

        Original problem:
            min_u  ||K^{-1} u - x_target||^2 + λ * u^T M u
        with M = K^{-T} L K^{-1}.  At steady state: K x = u.

        Strategy:
        • If optimizing over all reactions, use u = K x:
                min_x ||x - x_target||^2 + λ (Kx)^T M (Kx)
            ⇒ (I + λ K^T M K) x = x_target, then u = K x.

        • If optimizing over controlled reactions u = G u_c, avoid K^{-1} by
            computing A_eff = solve(K, G) (i.e., K A_eff = G) and solving the
            normal equations (A_eff^T A_eff + λ M_eff) u_c = A_eff^T x_target
            where M_eff = G^T M G.
        """
        K: np.ndarray = self.solver.dynamics.K

        # Control-entropy metric (implementation should avoid explicit inverses internally).
        M: np.ndarray = self.compute_control_entropy_metric(L)
        # Improve numerical symmetry
        M = 0.5 * (M + M.T)

        if controlled_reactions_only:
            # u = G u_c (optimize only controlled reactions)
            G: np.ndarray = self.G  # (r x m), column selector for controlled reactions

            # A_eff ≡ K^{-1} G computed via solve(K, G) (no explicit inverse)
            try:
                A_eff = np.linalg.solve(K, G)  # (r x m)
            except np.linalg.LinAlgError:
                # Least-squares fallback if K is singular/ill-conditioned
                A_eff, *_ = np.linalg.lstsq(K, G, rcond=None)

            M_eff = G.T @ M @ G  # (m x m)

            # Normal equations: (A_eff^T A_eff + λ M_eff) u_c = A_eff^T x_target
            H = A_eff.T @ A_eff + entropy_weight * M_eff
            H = 0.5 * (H + H.T)  # symmetrize
            b = A_eff.T @ x_target

            try:
                u_controlled = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                # Gentle Tikhonov jitter + final lstsq fallback
                jitter = 1e-9 * np.eye(H.shape[0], dtype=H.dtype)
                try:
                    u_controlled = np.linalg.solve(H + jitter, b)
                except np.linalg.LinAlgError:
                    u_controlled, *_ = np.linalg.lstsq(H, b, rcond=None)

            u_optimal = G @ u_controlled

            # Achieved steady state x from K x = u (avoid using A_eff again)
            try:
                x_achieved = np.linalg.solve(K, u_optimal)
            except np.linalg.LinAlgError:
                x_achieved, *_ = np.linalg.lstsq(K, u_optimal, rcond=None)

        else:
            # Optimize over all reactions via x-variable (u = K x)
            I = np.eye(K.shape[0], dtype=K.dtype)
            S = I + entropy_weight * (K.T @ M @ K)
            S = 0.5 * (S + S.T)  # enforce symmetry for stability

            # Solve for x directly; then u = K x
            try:
                x_achieved = np.linalg.solve(S, x_target)
            except np.linalg.LinAlgError:
                jitter = 1e-9 * np.eye(S.shape[0], dtype=S.dtype)
                try:
                    x_achieved = np.linalg.solve(S + jitter, x_target)
                except np.linalg.LinAlgError:
                    x_achieved, *_ = np.linalg.lstsq(S, x_target, rcond=None)

            u_optimal = K @ x_achieved

        # Metrics
        tracking_error = float(np.linalg.norm(x_achieved - x_target) ** 2)
        # Keep using your helper for consistency (it may include physical constants/scaling)
        entropy_rate = self.compute_steady_state_entropy_rate(u_optimal, M)
        total_cost = tracking_error + entropy_weight * float(entropy_rate)

        return {
            "u_optimal": u_optimal,
            "x_achieved": x_achieved,
            "tracking_error": tracking_error,
            "entropy_rate": float(entropy_rate),
            "total_cost": total_cost,
            "entropy_weight": entropy_weight,
            "controlled_reactions_only": controlled_reactions_only,
        }


class AdaptiveController(LLRQController):
    """LLRQ controller with adaptive/learning capabilities."""

    def __init__(self, solver: LLRQSolver, controlled_reactions: Optional[list] = None, adaptation_rate: float = 0.1):
        """Initialize adaptive controller.

        Args:
            solver: LLRQSolver
            controlled_reactions: Controlled reaction list
            adaptation_rate: Learning rate for parameter updates
        """
        super().__init__(solver, controlled_reactions)
        self.adaptation_rate = adaptation_rate
        self.estimated_disturbance = np.zeros(self.rankS)

    def update_disturbance_estimate(self, y_current: np.ndarray, y_dot_measured: np.ndarray, u_applied: np.ndarray):
        """Update estimated constant disturbance based on measurements.

        Args:
            y_current: Current reduced state
            y_dot_measured: Measured state derivative
            u_applied: Applied control input
        """
        # Expected dynamics: y_dot = A*y + B*u + d
        u_red = self.B_red @ u_applied
        expected_ydot = self.A @ y_current + u_red

        # Estimate disturbance: d = y_dot_measured - expected_ydot
        disturbance_measurement = y_dot_measured - expected_ydot

        # Update estimate with learning rate
        self.estimated_disturbance += self.adaptation_rate * (disturbance_measurement - self.estimated_disturbance)

    def compute_adaptive_control(self, Q_current: np.ndarray, Q_target: np.ndarray, feedback_gain: float = 1.0) -> np.ndarray:
        """Compute control with disturbance compensation.

        Args:
            Q_current: Current reaction quotients
            Q_target: Target reaction quotients
            feedback_gain: Feedback gain

        Returns:
            Control input with disturbance compensation
        """
        # Base control
        u_base = self.compute_control(Q_current, Q_target, feedback_gain)

        # Disturbance compensation
        u_disturbance = -np.linalg.pinv(self.B_red) @ self.estimated_disturbance

        return u_base + u_disturbance


def design_lqr_controller(solver: LLRQSolver, Q_weight: float = 1.0, R_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Design LQR controller for LLRQ system.

    Args:
        solver: LLRQSolver
        Q_weight: State cost weight
        R_weight: Control cost weight

    Returns:
        (K_lqr, P) where K_lqr is feedback gain and P is cost matrix
    """
    from scipy.linalg import solve_continuous_are

    # Get reduced system matrices
    B = solver._B
    K = solver.dynamics.K
    A = -(B.T @ K @ B)  # Reduced A matrix

    # Assume all reactions controllable for simplicity
    B_ctrl = np.eye(A.shape[0])  # Full control authority

    # Cost matrices
    Q_cost = Q_weight * np.eye(A.shape[0])
    R_cost = R_weight * np.eye(B_ctrl.shape[1])

    # Solve Riccati equation
    P = solve_continuous_are(A, B_ctrl, Q_cost, R_cost)

    # LQR gain
    K_lqr = np.linalg.inv(R_cost) @ B_ctrl.T @ P

    return K_lqr, P


class ControlledSimulation:
    """Unified interface for controlled simulations.

    This class provides a high-level API for running controlled simulations
    that encapsulates the workflow from linear_vs_mass_action.py:
    1. Setup reaction with initial concentrations
    2. Choose target point
    3. Compute static control input to reach target
    4. Simulate controlled dynamics
    """

    def __init__(self, solver, controller: Optional[LLRQController] = None):
        """Initialize controlled simulation.

        Args:
            solver: LLRQSolver instance
            controller: LLRQController instance. If None, creates default controller.
        """
        self.solver = solver
        self.controller = controller or LLRQController(solver)
        self.network = solver.network

    @classmethod
    def from_mass_action(
        cls, network, forward_rates, backward_rates, initial_concentrations, controlled_reactions=None, **kwargs
    ):
        """Create controlled simulation from mass action parameters.

        Args:
            network: ReactionNetwork
            forward_rates: Forward rate constants
            backward_rates: Backward rate constants
            initial_concentrations: Initial concentrations for equilibrium computation
            controlled_reactions: List of reactions to control
            **kwargs: Additional arguments passed to LLRQDynamics.from_mass_action

        Returns:
            ControlledSimulation instance
        """
        from .llrq_dynamics import LLRQDynamics
        from .solver import LLRQSolver

        # Create dynamics
        dynamics = LLRQDynamics.from_mass_action(
            network=network,
            forward_rates=forward_rates,
            backward_rates=backward_rates,
            initial_concentrations=initial_concentrations,
            **kwargs,
        )

        # Create solver and controller
        solver = LLRQSolver(dynamics)
        controller = LLRQController(solver, controlled_reactions)

        return cls(solver, controller)

    def simulate_to_target(
        self,
        initial_concentrations: Union[Dict[str, float], np.ndarray],
        target_state: Union[Dict[str, float], np.ndarray, str],
        t_span: Union[Tuple[float, float], np.ndarray],
        method: str = "linear",
        feedback_gain: float = 1.0,
        disturbance_function: Optional[Callable[[float], np.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate controlled dynamics to reach target concentrations.

        Args:
            initial_concentrations: Initial species concentrations (dict or array)
            target_state: Target concentrations (dict or array) or 'equilibrium'.
                         Must satisfy same conservation laws as initial concentrations.
            t_span: Time span for simulation
            method: Simulation method ('linear' or 'mass_action')
            feedback_gain: Proportional feedback gain
            disturbance_function: Optional disturbance function f(t) -> disturbance
            **kwargs: Additional simulation arguments

        Returns:
            Simulation results dictionary
        """
        return self.solver.solve_with_control(
            initial_conditions=initial_concentrations,
            target_state=target_state,
            t_span=t_span,
            controlled_reactions=self.controller.controlled_reactions,
            method=method,
            feedback_gain=feedback_gain,
            disturbance_function=disturbance_function,
            **kwargs,
        )

    def compare_methods(
        self,
        initial_concentrations: Union[Dict[str, float], np.ndarray],
        target_state: Union[Dict[str, float], np.ndarray, str],
        t_span: Union[Tuple[float, float], np.ndarray],
        feedback_gain: float = 1.0,
        disturbance_function: Optional[Callable[[float], np.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compare linear LLRQ vs mass action control performance.

        Args:
            initial_concentrations: Initial species concentrations (dict or array)
            target_state: Target concentrations (dict or array) or 'equilibrium'.
                         Must satisfy same conservation laws as initial concentrations.
            t_span: Time span for simulation
            feedback_gain: Proportional feedback gain
            disturbance_function: Optional disturbance function f(t) -> disturbance
            **kwargs: Additional simulation arguments

        Returns:
            Dictionary with both 'linear_result' and 'mass_action_result' keys
        """
        return self.solver.solve_with_control(
            initial_conditions=initial_concentrations,
            target_state=target_state,
            t_span=t_span,
            controlled_reactions=self.controller.controlled_reactions,
            compare_methods=True,
            feedback_gain=feedback_gain,
            disturbance_function=disturbance_function,
            **kwargs,
        )

    def analyze_performance(self, result: Dict[str, Any], target_state: Union[Dict[str, float], np.ndarray]) -> Dict[str, Any]:
        """Analyze control performance metrics.

        Args:
            result: Simulation result from simulate_to_target or compare_methods
            target_state: Target state for comparison

        Returns:
            Performance metrics dictionary
        """
        metrics = {}

        # Parse target state
        if isinstance(target_state, dict):
            c_target = self.solver._parse_initial_dict(target_state)
            Q_target = self.network.compute_reaction_quotients(c_target)
            y_target = self.controller.reaction_quotients_to_reduced_state(Q_target)
        elif isinstance(target_state, np.ndarray):
            if len(target_state) == self.solver._rankS:
                y_target = target_state
            else:
                # Assume concentrations or reaction quotients
                if len(target_state) == len(self.network.species_ids):
                    Q_target = self.network.compute_reaction_quotients(target_state)
                else:
                    Q_target = target_state
                y_target = self.controller.reaction_quotients_to_reduced_state(Q_target)
        else:
            raise ValueError("Cannot analyze performance for string target states")

        # Analyze single result
        if "linear_result" not in result and "mass_action_result" not in result:
            # Single simulation result
            metrics = self._compute_single_performance(result, y_target)
        else:
            # Comparison results
            if "linear_result" in result:
                metrics["linear"] = self._compute_single_performance(result["linear_result"], y_target)
            if "mass_action_result" in result:
                metrics["mass_action"] = self._compute_single_performance(result["mass_action_result"], y_target)

            # Cross-comparison metrics
            if "linear_result" in result and "mass_action_result" in result:
                linear_final = result["linear_result"]["reduced_state"][-1]
                mass_action_final = result["mass_action_result"]["reduced_state"][-1]
                metrics["method_difference"] = {
                    "final_state_error": float(np.linalg.norm(linear_final - mass_action_final)),
                    "max_trajectory_difference": float(
                        np.max(
                            np.linalg.norm(
                                result["linear_result"]["reduced_state"] - result["mass_action_result"]["reduced_state"],
                                axis=1,
                            )
                        )
                    ),
                }

        return metrics

    def _compute_single_performance(self, result: Dict[str, Any], y_target: np.ndarray) -> Dict[str, Any]:
        """Compute performance metrics for a single simulation result."""
        y_traj = result["reduced_state"]

        # Tracking errors
        errors = np.linalg.norm(y_traj - y_target, axis=1)

        # Control effort
        if "control_inputs" in result:
            control_effort = np.sum(np.abs(result["control_inputs"]), axis=1)
            total_control_effort = float(np.trapz(control_effort, result["time"]))
        else:
            total_control_effort = None

        return {
            "final_error": float(errors[-1]),
            "max_error": float(np.max(errors)),
            "rms_error": float(np.sqrt(np.mean(errors**2))),
            "settling_time": self._compute_settling_time(result["time"], errors),
            "total_control_effort": total_control_effort,
            "steady_state_achieved": bool(errors[-1] < 0.05),  # 5% tolerance
        }

    def _compute_settling_time(self, time: np.ndarray, errors: np.ndarray, tolerance: float = 0.05) -> Optional[float]:
        """Compute settling time (time to reach and stay within tolerance)."""
        within_tolerance = errors <= tolerance
        if not np.any(within_tolerance):
            return None

        # Find last time point outside tolerance
        last_violation_idx = None
        for i in reversed(range(len(within_tolerance))):
            if not within_tolerance[i]:
                last_violation_idx = i
                break

        if last_violation_idx is None:
            # Always within tolerance
            return float(time[0])
        elif last_violation_idx == len(time) - 1:
            # Never settled
            return None
        else:
            return float(time[last_violation_idx + 1])

    def plot_comparison(
        self,
        comparison_result: Dict[str, Any],
        target_state: Union[Dict[str, float], np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot comparison between linear and mass action results.

        Args:
            comparison_result: Result from compare_methods()
            target_state: Target state for reference lines
            save_path: Optional path to save plot

        Returns:
            Path to saved plot if save_path provided, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

        # Extract results
        if "linear_result" not in comparison_result or "mass_action_result" not in comparison_result:
            raise ValueError("comparison_result must contain both linear_result and mass_action_result")

        linear_result = comparison_result["linear_result"]
        mass_action_result = comparison_result["mass_action_result"]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Linear LLRQ vs Mass Action Control Comparison", fontsize=16)

        time = linear_result["time"]

        # Plot 1: Reduced state trajectories
        ax = axes[0, 0]
        rankS = linear_result["reduced_state"].shape[1]
        for i in range(min(rankS, 2)):  # Plot first two components
            ax.plot(time, linear_result["reduced_state"][:, i], "b-", linewidth=2, label=f"Linear y{i+1}" if i == 0 else "")
            ax.plot(
                time,
                mass_action_result["reduced_state"][:, i],
                "r--",
                linewidth=2,
                label=f"Mass Action y{i+1}" if i == 0 else "",
            )

        # Target reference if provided
        if target_state is not None:
            try:
                y_target = self._parse_target_for_plotting(target_state)
                for i in range(min(len(y_target), 2)):
                    ax.axhline(y_target[i], color="k", linestyle=":", alpha=0.7, label="Target" if i == 0 else "")
            except:
                pass  # Skip target plotting if parsing fails

        ax.set_xlabel("Time")
        ax.set_ylabel("Reduced State")
        ax.set_title("Reduced State Trajectories")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Concentration trajectories (first 3 species)
        ax = axes[0, 1]
        n_species_plot = min(3, len(self.network.species_ids))
        colors = ["blue", "green", "orange"]
        for i in range(n_species_plot):
            species_id = self.network.species_ids[i]
            ax.plot(
                time, linear_result["concentrations"][:, i], "-", color=colors[i], linewidth=2, label=f"Linear [{species_id}]"
            )
            ax.plot(
                time,
                mass_action_result["concentrations"][:, i],
                "--",
                color=colors[i],
                linewidth=2,
                label=f"Mass Action [{species_id}]",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        ax.set_title("Species Concentrations")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Control inputs (if available)
        ax = axes[1, 0]
        if "control_inputs" in linear_result and "control_inputs" in mass_action_result:
            n_controls = min(2, linear_result["control_inputs"].shape[1])
            for i in range(n_controls):
                reaction_id = self.controller.controlled_reactions[i]
                if isinstance(reaction_id, int):
                    reaction_id = self.network.reaction_ids[reaction_id]
                ax.plot(
                    time,
                    linear_result["control_inputs"][:, i],
                    "b-",
                    linewidth=2,
                    label=f"Linear {reaction_id}" if i == 0 else "",
                )
                ax.plot(
                    time,
                    mass_action_result["control_inputs"][:, i],
                    "r--",
                    linewidth=2,
                    label=f"Mass Action {reaction_id}" if i == 0 else "",
                )
            ax.set_ylabel("Control Input")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Control inputs not available", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Time")
        ax.set_title("Control Signals")
        ax.grid(True, alpha=0.3)

        # Plot 4: Error comparison
        ax = axes[1, 1]
        if target_state is not None:
            try:
                y_target = self._parse_target_for_plotting(target_state)
                linear_errors = np.linalg.norm(linear_result["reduced_state"] - y_target, axis=1)
                mass_action_errors = np.linalg.norm(mass_action_result["reduced_state"] - y_target, axis=1)

                ax.plot(time, linear_errors, "b-", linewidth=2, label="Linear LLRQ")
                ax.plot(time, mass_action_errors, "r--", linewidth=2, label="Mass Action")
                ax.set_ylabel("Tracking Error")
                ax.legend()
            except:
                ax.text(0.5, 0.5, "Error computation failed", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No target state provided", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Time")
        ax.set_title("Tracking Error")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    def _parse_target_for_plotting(self, target_state):
        """Parse target state for plotting purposes."""
        if isinstance(target_state, dict):
            c_target = self.solver._parse_initial_dict(target_state)
            Q_target = self.network.compute_reaction_quotients(c_target)
            return self.controller.reaction_quotients_to_reduced_state(Q_target)
        elif isinstance(target_state, np.ndarray):
            if len(target_state) == self.solver._rankS:
                return target_state
            elif len(target_state) == len(self.network.species_ids):
                Q_target = self.network.compute_reaction_quotients(target_state)
                return self.controller.reaction_quotients_to_reduced_state(Q_target)
            elif len(target_state) == len(self.network.reaction_ids):
                return self.controller.reaction_quotients_to_reduced_state(target_state)
        raise ValueError("Cannot parse target state for plotting")
