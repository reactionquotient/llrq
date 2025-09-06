"""LLRQ control strategies.

This module provides control algorithms based on LLRQ theory,
separated from the dynamics simulation.
"""

import numpy as np
from typing import Optional, Union, Callable, Tuple
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
            warnings.warn("Reduced input matrix appears rank-deficient. "
                         "Selected reactions may not fully control the system.")
    
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
    
    def compute_feedback_control(self, y_current: np.ndarray, y_target: np.ndarray, 
                               feedback_gain: float = 1.0) -> np.ndarray:
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
    
    def compute_control(self, Q_current: np.ndarray, Q_target: np.ndarray,
                       feedback_gain: float = 1.0, 
                       feedforward: bool = True) -> np.ndarray:
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
    
    def compute_target_concentrations(self, Q_target: np.ndarray, 
                                    initial_concentrations: np.ndarray) -> np.ndarray:
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


class AdaptiveController(LLRQController):
    """LLRQ controller with adaptive/learning capabilities."""
    
    def __init__(self, solver: LLRQSolver, controlled_reactions: Optional[list] = None,
                 adaptation_rate: float = 0.1):
        """Initialize adaptive controller.
        
        Args:
            solver: LLRQSolver 
            controlled_reactions: Controlled reaction list
            adaptation_rate: Learning rate for parameter updates
        """
        super().__init__(solver, controlled_reactions)
        self.adaptation_rate = adaptation_rate
        self.estimated_disturbance = np.zeros(self.rankS)
        
    def update_disturbance_estimate(self, y_current: np.ndarray, y_dot_measured: np.ndarray,
                                  u_applied: np.ndarray):
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
        self.estimated_disturbance += self.adaptation_rate * (
            disturbance_measurement - self.estimated_disturbance
        )
    
    def compute_adaptive_control(self, Q_current: np.ndarray, Q_target: np.ndarray,
                               feedback_gain: float = 1.0) -> np.ndarray:
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


def design_lqr_controller(solver: LLRQSolver, Q_weight: float = 1.0, 
                         R_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
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