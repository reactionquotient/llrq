"""Moiety-based control for LLRQ systems.

This module provides decoupled control design for block-triangular LLRQ systems,
enabling separate control of reaction quotients (ratios) and moiety totals (levels).
"""

from typing import Dict, Optional, Tuple, Union, Callable, Any
import numpy as np
from scipy.linalg import solve_continuous_are, solve_lyapunov
from .moiety_dynamics import MoietyDynamics
from .open_system_network import OpenSystemNetwork


class MoietyController:
    """Decoupled controller for moiety dynamics.

    Designs separate controllers for:
    - x-block: Reaction quotient logs (controls ratios/composition)
    - y-block: Moiety totals (controls levels/amounts)

    Enables superposition when actuators affect both blocks.
    """

    def __init__(
        self,
        moiety_dynamics: MoietyDynamics,
        controlled_reactions: Optional[list] = None,
        controlled_flows: Optional[list] = None,
    ):
        """Initialize moiety controller.

        Args:
            moiety_dynamics: MoietyDynamics system
            controlled_reactions: Reaction indices/IDs for x-block control
            controlled_flows: Flow parameters for y-block control
        """
        self.dynamics = moiety_dynamics
        self.network = moiety_dynamics.network

        # Get block-triangular system
        self.system = moiety_dynamics.get_block_system()

        # x-block (reaction quotients)
        self.A_x = self.system.A_x
        self.B_x = self.system.B_x
        self.n_reactions = self.A_x.shape[0]

        # y-block (moiety totals)
        self.A_y = self.system.A_y
        self.n_moieties = self.A_y.shape[0]

        # Setup actuation
        self.controlled_reactions = controlled_reactions or list(range(self.n_reactions))
        self.controlled_flows = controlled_flows or []

        self._setup_control_matrices()

        # Controller gains (to be computed)
        self.K_x: Optional[np.ndarray] = None  # x-block feedback gain
        self.K_y: Optional[np.ndarray] = None  # y-block feedback gain
        self.K_ff_x: Optional[np.ndarray] = None  # x-block feedforward gain
        self.K_ff_y: Optional[np.ndarray] = None  # y-block feedforward gain

    def _setup_control_matrices(self):
        """Setup control input matrices for each block."""
        # x-block control matrix (select which reactions to control)
        if self.controlled_reactions and isinstance(self.controlled_reactions[0], str):
            # Convert reaction IDs to indices
            indices = [self.network.reaction_to_idx[rid] for rid in self.controlled_reactions]
        else:
            indices = list(self.controlled_reactions)

        self.B_x_ctrl = np.zeros((self.n_reactions, len(indices)))
        for i, idx in enumerate(indices):
            self.B_x_ctrl[idx, i] = 1.0

        # y-block control matrix (flow inputs)
        if self.n_moieties > 0:
            # For now, assume we can control all moiety inlets
            self.B_y_ctrl = np.eye(self.n_moieties)
        else:
            self.B_y_ctrl = np.zeros((0, 0))

    def design_lqr_x(self, Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> np.ndarray:
        """Design LQR controller for x-block (reaction quotients).

        Solves: min ∫ (x^T Q x + u^T R u) dt
        subject to: ẋ = A_x x + B_x u

        Args:
            Q: State weighting matrix (defaults to identity)
            R: Control weighting matrix (defaults to identity)

        Returns:
            Feedback gain K_x such that u = -K_x @ x
        """
        if Q is None:
            Q = np.eye(self.n_reactions)
        if R is None:
            R = np.eye(self.B_x_ctrl.shape[1])

        try:
            # Solve algebraic Riccati equation
            P = solve_continuous_are(self.A_x, self.B_x_ctrl, Q, R)
            self.K_x = np.linalg.solve(R, self.B_x_ctrl.T @ P)
        except Exception as e:
            import warnings

            warnings.warn(f"LQR design failed for x-block: {e}. Using pole placement.")
            # Fallback to simple proportional control
            self.K_x = np.linalg.pinv(self.B_x_ctrl) @ np.eye(self.n_reactions)

        return self.K_x

    def design_lqr_y(self, Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Design LQR controller for y-block (moiety totals).

        Args:
            Q: State weighting matrix
            R: Control weighting matrix

        Returns:
            Feedback gain K_y or None if no moieties
        """
        if self.n_moieties == 0:
            return None

        if Q is None:
            Q = np.eye(self.n_moieties)
        if R is None:
            R = np.eye(self.B_y_ctrl.shape[1])

        try:
            # Handle time-varying A_y
            if callable(self.A_y):
                # Use A_y at t=0 for design
                A_y_nom = self.A_y(0.0)
            else:
                A_y_nom = self.A_y

            P = solve_continuous_are(A_y_nom, self.B_y_ctrl, Q, R)
            self.K_y = np.linalg.solve(R, self.B_y_ctrl.T @ P)
        except Exception as e:
            import warnings

            warnings.warn(f"LQR design failed for y-block: {e}. Using proportional control.")
            self.K_y = np.linalg.pinv(self.B_y_ctrl) @ np.eye(self.n_moieties)

        return self.K_y

    def design_feedforward_x(self, x_ref: np.ndarray) -> np.ndarray:
        """Design feedforward controller for x-block steady-state tracking.

        At steady state: 0 = A_x @ x_ref + B_x @ u_ff
        So: u_ff = -pinv(B_x) @ A_x @ x_ref

        Args:
            x_ref: Reference reaction quotient logs

        Returns:
            Feedforward control u_ff
        """
        u_ss = -np.linalg.pinv(self.B_x_ctrl) @ self.A_x @ x_ref
        self.K_ff_x = u_ss
        return u_ss

    def design_feedforward_y(self, y_ref: np.ndarray) -> Optional[np.ndarray]:
        """Design feedforward controller for y-block steady-state tracking.

        Args:
            y_ref: Reference moiety totals

        Returns:
            Feedforward control or None if no moieties
        """
        if self.n_moieties == 0:
            return None

        # Handle time-varying A_y
        if callable(self.A_y):
            A_y_nom = self.A_y(0.0)
        else:
            A_y_nom = self.A_y

        # Steady state: 0 = A_y @ y_ref + B_y @ u_ff + g_y
        g_y = self.system.g_y if self.system.g_y is not None else np.zeros(self.n_moieties)

        u_ss = -np.linalg.pinv(self.B_y_ctrl) @ (A_y_nom @ y_ref + g_y)
        self.K_ff_y = u_ss
        return u_ss

    def compute_control(
        self,
        x_current: np.ndarray,
        y_current: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        y_ref: Optional[np.ndarray] = None,
        t: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Compute control inputs for both blocks.

        Args:
            x_current: Current reaction quotient logs
            y_current: Current moiety totals
            x_ref: Reference reaction quotient logs
            y_ref: Reference moiety totals
            t: Current time

        Returns:
            Dictionary with 'u_x' (reaction control) and 'u_y' (flow control)
        """
        control: Dict[str, np.ndarray] = {}

        # x-block control
        if self.K_x is not None:
            x_error = x_current - (x_ref if x_ref is not None else np.zeros_like(x_current))
            u_x_fb = -self.K_x @ x_error
            u_x_ff = self.K_ff_x if self.K_ff_x is not None else 0.0
            control["u_x"] = u_x_fb + u_x_ff
        else:
            control["u_x"] = np.zeros(len(self.controlled_reactions))

        # y-block control
        if self.n_moieties > 0 and y_current is not None and self.K_y is not None:
            y_error = y_current - (y_ref if y_ref is not None else np.zeros_like(y_current))
            u_y_fb = -self.K_y @ y_error
            u_y_ff = self.K_ff_y if self.K_ff_y is not None else np.zeros(self.n_moieties)
            control["u_y"] = u_y_fb + u_y_ff
        else:
            control["u_y"] = np.zeros(self.n_moieties) if self.n_moieties > 0 else np.array([])

        return control

    def design_pole_placement_x(self, desired_poles: np.ndarray) -> np.ndarray:
        """Design pole placement controller for x-block.

        Args:
            desired_poles: Desired closed-loop poles

        Returns:
            Feedback gain K_x
        """
        from scipy.signal import place_poles

        try:
            result = place_poles(self.A_x, self.B_x_ctrl, desired_poles)
            self.K_x = result.gain_matrix
        except Exception as e:
            import warnings

            warnings.warn(f"Pole placement failed: {e}. Using LQR instead.")
            self.K_x = self.design_lqr_x()

        return self.K_x

    def analyze_controllability(self) -> Dict[str, Any]:
        """Analyze controllability of both blocks.

        Returns:
            Dictionary with controllability analysis
        """
        analysis: Dict[str, Any] = {}

        # x-block controllability
        n_x = self.A_x.shape[0]
        m_x = self.B_x_ctrl.shape[1]

        # Controllability matrix
        C_x = self.B_x_ctrl.copy()
        A_power = np.eye(n_x)
        for i in range(1, n_x):
            A_power = A_power @ self.A_x
            C_x = np.hstack([C_x, A_power @ self.B_x_ctrl])

        rank_x = np.linalg.matrix_rank(C_x)
        analysis["x_block"] = {"controllable": rank_x == n_x, "controllability_rank": rank_x, "max_rank": n_x}

        # y-block controllability
        if self.n_moieties > 0:
            n_y = self.A_y.shape[0]

            # Handle time-varying A_y
            if callable(self.A_y):
                A_y_nom = self.A_y(0.0)
            else:
                A_y_nom = self.A_y

            C_y = self.B_y_ctrl.copy()
            A_power = np.eye(n_y)
            for i in range(1, n_y):
                A_power = A_power @ A_y_nom
                C_y = np.hstack([C_y, A_power @ self.B_y_ctrl])

            rank_y = np.linalg.matrix_rank(C_y)
            analysis["y_block"] = {"controllable": rank_y == n_y, "controllability_rank": rank_y, "max_rank": n_y}
        else:
            analysis["y_block"] = {
                "controllable": True,  # Trivially controllable (no states)
                "controllability_rank": 0,
                "max_rank": 0,
            }

        return analysis

    def compute_closed_loop_poles(self) -> Dict[str, np.ndarray]:
        """Compute closed-loop poles for designed controllers.

        Returns:
            Dictionary with poles for each block
        """
        poles: Dict[str, np.ndarray] = {}

        # x-block closed-loop poles
        if self.K_x is not None:
            A_cl_x = self.A_x - self.B_x_ctrl @ self.K_x
            poles["x_block"] = np.linalg.eigvals(A_cl_x)

        # y-block closed-loop poles
        if self.K_y is not None and self.n_moieties > 0:
            # Handle time-varying A_y
            if callable(self.A_y):
                A_y_nom = self.A_y(0.0)
            else:
                A_y_nom = self.A_y

            A_cl_y = A_y_nom - self.B_y_ctrl @ self.K_y
            poles["y_block"] = np.linalg.eigvals(A_cl_y)

        return poles

    def simulate_closed_loop(
        self,
        t: np.ndarray,
        x0: np.ndarray,
        y0: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        y_ref: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """Simulate closed-loop response.

        Args:
            t: Time points
            x0: Initial reaction quotient logs
            y0: Initial moiety totals
            x_ref: Reference reaction quotient logs
            y_ref: Reference moiety totals

        Returns:
            Closed-loop simulation results
        """
        n_times = len(t)
        dt = t[1] - t[0] if len(t) > 1 else 0.01

        # Initialize trajectories
        x_traj = np.zeros((n_times, self.n_reactions))
        x_traj[0] = x0

        if self.n_moieties > 0 and y0 is not None:
            y_traj = np.zeros((n_times, self.n_moieties))
            y_traj[0] = y0
        else:
            y_traj = np.zeros((n_times, 0))

        u_x_traj = np.zeros((n_times, len(self.controlled_reactions)))
        u_y_traj = np.zeros((n_times, self.n_moieties)) if self.n_moieties > 0 else np.zeros((n_times, 0))

        # Simulate
        for i in range(n_times - 1):
            # Compute control
            y_current = y_traj[i] if y_traj.shape[1] > 0 else None
            control = self.compute_control(x_traj[i], y_current, x_ref, y_ref, t[i])

            u_x_traj[i] = control["u_x"]
            if self.n_moieties > 0:
                u_y_traj[i] = control["u_y"]

            # x-block dynamics
            if self.K_x is not None:
                A_cl_x = self.A_x - self.B_x_ctrl @ self.K_x
                dx = A_cl_x @ x_traj[i]
                if x_ref is not None and self.K_ff_x is not None:
                    dx += self.B_x_ctrl @ self.K_ff_x
            else:
                dx = self.A_x @ x_traj[i]

            x_traj[i + 1] = x_traj[i] + dt * dx

            # y-block dynamics
            if self.n_moieties > 0:
                if callable(self.A_y):
                    A_y_t = self.A_y(t[i])
                else:
                    A_y_t = self.A_y

                if self.K_y is not None:
                    A_cl_y = A_y_t - self.B_y_ctrl @ self.K_y
                    dy = A_cl_y @ y_traj[i]
                else:
                    dy = A_y_t @ y_traj[i]

                # Add constant terms
                if self.system.g_y is not None:
                    dy += self.system.g_y

                if y_ref is not None and self.K_ff_y is not None:
                    dy += self.B_y_ctrl @ self.K_ff_y

                y_traj[i + 1] = y_traj[i] + dt * dy

        # Final control inputs
        y_final = y_traj[-1] if y_traj.shape[1] > 0 else None
        control_final = self.compute_control(x_traj[-1], y_final, x_ref, y_ref, t[-1])
        u_x_traj[-1] = control_final["u_x"]
        if self.n_moieties > 0:
            u_y_traj[-1] = control_final["u_y"]

        return {
            "time": t,
            "x": x_traj,
            "y": y_traj,
            "u_x": u_x_traj,
            "u_y": u_y_traj,
            "x_ref": np.tile(x_ref, (n_times, 1)) if x_ref is not None else None,
            "y_ref": np.tile(y_ref, (n_times, 1)) if y_ref is not None else None,
        }
