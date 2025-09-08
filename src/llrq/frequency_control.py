"""Frequency-space control for LLRQ systems.

This module provides frequency-domain control methods for LLRQ systems,
enabling design of sinusoidal inputs to achieve target periodic steady states.

Mathematical Foundation:
For system ẋ = -Kx + Bu(t) with u(t) = Re{U e^(iωt)}:
- Frequency response: H(iω) = (K + iωI)^(-1)B
- Steady state: x_ss(t) = Re{H(iω)U e^(iωt)}
- Optimal control: U = (H*WH + λI)^(-1)H*WX* (weighted least squares + regularization)
"""

from typing import Optional, Tuple

import numpy as np


class FrequencySpaceController:
    """Controller for designing sinusoidal inputs in frequency space.

    This class provides methods for:
    1. Computing frequency response matrices
    2. Designing optimal sinusoidal control inputs
    3. Evaluating periodic steady states
    """

    def __init__(self, K: np.ndarray, B: np.ndarray):
        """Initialize frequency-space controller.

        Args:
            K: System matrix (n x n) from ẋ = -Kx + Bu
            B: Input matrix (n x m) from ẋ = -Kx + Bu
        """
        self.K = K
        self.B = B
        self.n_states = K.shape[0]
        self.n_controls = B.shape[1]

        # Validate dimensions
        assert K.shape == (self.n_states, self.n_states), "K must be square"
        assert B.shape == (self.n_states, self.n_controls), "B dimensions inconsistent with K"

        # Check symmetry (warning only, as some applications may have non-symmetric K)
        if not np.allclose(K, K.T):
            import warnings

            warnings.warn("K is not symmetric - this may indicate the system is not from LLRQ theory")

    @classmethod
    def from_llrq_solver(cls, solver, controlled_reactions=None):
        """Create frequency controller from LLRQ solver.

        This factory method creates a FrequencySpaceController using the reduced
        system matrices from an LLRQSolver, with optional reaction selection.

        Args:
            solver: LLRQSolver instance with computed basis matrices
            controlled_reactions: List of reaction IDs or indices to control.
                                If None, controls all reactions.
                                Can be strings (reaction IDs) or integers (indices).

        Returns:
            FrequencySpaceController instance configured for the specified reactions

        Example:
            # Control all reactions
            freq_controller = FrequencySpaceController.from_llrq_solver(solver)

            # Control specific reactions
            freq_controller = FrequencySpaceController.from_llrq_solver(
                solver, controlled_reactions=["R1", "R3"]
            )
        """
        K_red, B_red = solver.get_reduced_system_matrices(controlled_reactions)
        return cls(K_red, B_red)

    def compute_frequency_response(self, omega: float) -> np.ndarray:
        """Compute frequency response matrix H(iω) = (K + iωI)^(-1)B.

        Args:
            omega: Frequency in rad/s

        Returns:
            H: Complex frequency response matrix (n x m)
        """
        # Form complex matrix K + iωI
        K_complex = self.K + 1j * omega * np.eye(self.n_states)

        # Compute H(iω) = (K + iωI)^(-1)B
        try:
            H = np.linalg.solve(K_complex, self.B)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            H = np.linalg.pinv(K_complex) @ self.B

        return H

    def design_sinusoidal_control(
        self, X_target: np.ndarray, omega: float, W: Optional[np.ndarray] = None, lam: float = 0.01
    ) -> np.ndarray:
        """Design optimal sinusoidal control U to achieve target complex amplitude.

        Solves: min_U ||H(iω)U - X*||²_W + λ||U||²
        Solution: U = (H*WH + λI)^(-1)H*WX*

        Args:
            X_target: Target complex amplitude vector (n,)
            omega: Frequency in rad/s
            W: Weighting matrix (n x n). If None, uses identity.
            lam: Regularization parameter

        Returns:
            U: Optimal complex control amplitude vector (m,)
        """
        # Get frequency response
        H = self.compute_frequency_response(omega)

        # Default weighting matrix
        if W is None:
            W = np.eye(self.n_states)

        # Validate dimensions
        assert X_target.shape == (self.n_states,), f"X_target must be ({self.n_states},)"
        assert W.shape == (self.n_states, self.n_states), f"W must be ({self.n_states}, {self.n_states})"

        # Compute optimal control: U = (H*WH + λI)^(-1)H*WX*
        H_conj = np.conj(H.T)  # H* (conjugate transpose)

        # Form Gram matrix: H*WH + λI
        gram_matrix = H_conj @ W @ H + lam * np.eye(self.n_controls)

        # Right hand side: H*WX*
        rhs = H_conj @ W @ X_target

        # Solve for U
        try:
            U = np.linalg.solve(gram_matrix, rhs)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            U = np.linalg.pinv(gram_matrix) @ rhs

        return U

    def evaluate_steady_state(self, U: np.ndarray, omega: float, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate periodic steady state x_ss(t) = Re{H(iω)U e^(iωt)}.

        Args:
            U: Complex control amplitude vector (m,)
            omega: Frequency in rad/s
            t: Time points (nt,)

        Returns:
            x_ss: Steady-state trajectory (nt x n)
            u_real: Real control signal u(t) = Re{U e^(iωt)} (nt x m)
        """
        # Get frequency response
        H = self.compute_frequency_response(omega)

        # Complex amplitude for state: X = H(iω)U
        X_achieved = H @ U  # (n,)

        # Time-domain signals
        nt = len(t)
        x_ss = np.zeros((nt, self.n_states))
        u_real = np.zeros((nt, self.n_controls))

        for i, t_val in enumerate(t):
            # x_ss(t) = Re{X e^(iωt)}
            x_ss[i, :] = np.real(X_achieved * np.exp(1j * omega * t_val))

            # u(t) = Re{U e^(iωt)}
            u_real[i, :] = np.real(U * np.exp(1j * omega * t_val))

        return x_ss, u_real

    def compute_tracking_error(
        self, U: np.ndarray, X_target: np.ndarray, omega: float, W: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """Compute tracking error for achieved vs target amplitudes.

        Args:
            U: Control amplitude vector (m,)
            X_target: Target complex amplitude (n,)
            omega: Frequency in rad/s
            W: Weighting matrix (n x n). If None, uses identity.

        Returns:
            error_norm: Weighted norm of tracking error
            X_achieved: Achieved complex amplitude (n,)
        """
        # Get frequency response and achieved amplitude
        H = self.compute_frequency_response(omega)
        X_achieved = H @ U

        # Compute error
        error = X_achieved - X_target

        # Apply weighting
        if W is None:
            W = np.eye(self.n_states)

        # Weighted error norm: ||error||_W = sqrt(error* W error)
        error_norm = np.sqrt(np.real(np.conj(error) @ W @ error))

        return error_norm, X_achieved

    def frequency_sweep(self, omega_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute frequency response over a range of frequencies.

        Args:
            omega_range: Array of frequencies (nf,)

        Returns:
            magnitude: Magnitude of H(iω) (nf x n x m)
            phase: Phase of H(iω) in degrees (nf x n x m)
        """
        nf = len(omega_range)
        magnitude = np.zeros((nf, self.n_states, self.n_controls))
        phase = np.zeros((nf, self.n_states, self.n_controls))

        for i, omega in enumerate(omega_range):
            H = self.compute_frequency_response(omega)
            magnitude[i, :, :] = np.abs(H)
            phase[i, :, :] = np.angle(H, deg=True)

        return magnitude, phase

    # =============================================================================
    # Entropy-Aware Frequency Control
    # =============================================================================

    def compute_entropy_kernel(self, omega: float, L: np.ndarray) -> np.ndarray:
        """Compute frequency-domain entropy kernel H_u(ω) = G(iω)^H L G(iω).

        This kernel defines the entropy cost of control inputs in frequency domain:
        σ_u(ω) = U(ω)^H H_u(ω) U(ω)

        Args:
            omega: Frequency in rad/s
            L: Onsager conductance matrix (n_states x n_states)

        Returns:
            H_u: Entropy kernel matrix (n_controls x n_controls)
        """
        # Get frequency response G(iω) = (K + iωI)^{-1}B
        G = self.compute_frequency_response(omega)

        # Compute entropy kernel H_u(ω) = G^H L G
        H_u = np.conj(G.T) @ L @ G

        return H_u

    def compute_sinusoidal_entropy_rate(self, U: np.ndarray, omega: float, L: np.ndarray) -> float:
        """Compute time-averaged entropy production rate for sinusoidal control.

        For u(t) = Re{U e^(iωt)}, the time-averaged entropy rate is:
        σ̄ = (1/2) Re{U^H H_u(ω) U}

        Args:
            U: Complex control amplitude vector (n_controls,)
            omega: Frequency in rad/s
            L: Onsager conductance matrix (n_states x n_states)

        Returns:
            Time-averaged entropy production rate
        """
        H_u = self.compute_entropy_kernel(omega, L)
        entropy_rate = 0.5 * np.real(np.conj(U).T @ H_u @ U)
        return float(entropy_rate)

    def design_entropy_aware_sinusoidal_control(
        self, X_target: np.ndarray, omega: float, L: np.ndarray, entropy_weight: float = 1.0, W: Optional[np.ndarray] = None
    ) -> dict:
        """Design sinusoidal control with entropy-tracking tradeoff.

        Solves the optimization problem:
        min_U  ||H(iω)U - X*||²_W + λ * Re{U^H H_u(ω) U}

        Where the entropy cost is the time-averaged production rate for sinusoidal signals.

        Args:
            X_target: Target complex state amplitude (n_states,)
            omega: Frequency in rad/s
            L: Onsager conductance matrix (n_states x n_states)
            entropy_weight: Weight λ on entropy cost vs tracking error
            W: State weighting matrix (n_states x n_states). If None, uses identity.

        Returns:
            Dictionary containing:
            - U_optimal: Optimal complex control amplitude
            - X_achieved: Achieved complex state amplitude
            - tracking_error: Weighted tracking error
            - entropy_rate: Time-averaged entropy production rate
            - total_cost: Combined cost function value
        """
        if W is None:
            W = np.eye(self.n_states)

        # Get frequency response and entropy kernel
        H = self.compute_frequency_response(omega)
        H_u = self.compute_entropy_kernel(omega, L)

        # Solve: (H^H W H + λ H_u) U = H^H W X*
        # Note: For sinusoidal signals, entropy weight is applied to full H_u, not 0.5*H_u,
        # because the optimization includes the (1/2) factor in the cost function
        gram_matrix = np.conj(H.T) @ W @ H + entropy_weight * 0.5 * H_u  # Include the 1/2 factor
        rhs = np.conj(H.T) @ W @ X_target

        # Solve for optimal control
        try:
            U_optimal = np.linalg.solve(gram_matrix, rhs)
        except np.linalg.LinAlgError:
            U_optimal = np.linalg.pinv(gram_matrix) @ rhs

        # Compute achieved state and performance metrics
        X_achieved = H @ U_optimal
        tracking_error = np.real(np.conj(X_achieved - X_target).T @ W @ (X_achieved - X_target))
        entropy_rate = self.compute_sinusoidal_entropy_rate(U_optimal, omega, L)
        total_cost = float(tracking_error + entropy_weight * entropy_rate)

        return {
            "U_optimal": U_optimal,
            "X_achieved": X_achieved,
            "tracking_error": float(tracking_error),
            "entropy_rate": entropy_rate,
            "total_cost": total_cost,
            "entropy_weight": entropy_weight,
            "omega": omega,
        }

    def analyze_frequency_entropy_tradeoff(
        self,
        X_target: np.ndarray,
        omega_range: np.ndarray,
        L: np.ndarray,
        entropy_weights: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> dict:
        """Analyze entropy-tracking tradeoff across frequencies and entropy weights.

        Args:
            X_target: Target complex state amplitude (n_states,)
            omega_range: Array of frequencies to analyze (nf,)
            L: Onsager conductance matrix (n_states x n_states)
            entropy_weights: Array of entropy weight parameters (nw,)
            W: State weighting matrix. If None, uses identity.

        Returns:
            Dictionary containing analysis results with keys:
            - omega_range, entropy_weights: Input parameter arrays
            - tracking_errors: (nf x nw) array of tracking errors
            - entropy_rates: (nf x nw) array of entropy production rates
            - total_costs: (nf x nw) array of total costs
            - control_amplitudes: (nf x nw) array of control effort ||U||
        """
        nf, nw = len(omega_range), len(entropy_weights)

        # Initialize result arrays
        tracking_errors = np.zeros((nf, nw))
        entropy_rates = np.zeros((nf, nw))
        total_costs = np.zeros((nf, nw))
        control_amplitudes = np.zeros((nf, nw))

        # Compute for each frequency and entropy weight combination
        for i, omega in enumerate(omega_range):
            for j, lam in enumerate(entropy_weights):
                result = self.design_entropy_aware_sinusoidal_control(X_target, omega, L, entropy_weight=lam, W=W)

                tracking_errors[i, j] = result["tracking_error"]
                entropy_rates[i, j] = result["entropy_rate"]
                total_costs[i, j] = result["total_cost"]
                control_amplitudes[i, j] = np.linalg.norm(result["U_optimal"])

        return {
            "omega_range": omega_range,
            "entropy_weights": entropy_weights,
            "tracking_errors": tracking_errors,
            "entropy_rates": entropy_rates,
            "total_costs": total_costs,
            "control_amplitudes": control_amplitudes,
        }

    def compute_entropy_kernel_spectrum(self, omega_range: np.ndarray, L: np.ndarray) -> dict:
        """Compute entropy kernel properties across frequency range.

        Args:
            omega_range: Array of frequencies (nf,)
            L: Onsager conductance matrix (n_states x n_states)

        Returns:
            Dictionary with frequency-dependent entropy kernel properties:
            - frequencies: Input frequency array
            - kernel_trace: Trace of H_u(ω) at each frequency
            - kernel_determinant: Determinant of H_u(ω) at each frequency
            - kernel_condition: Condition number of H_u(ω) at each frequency
        """
        nf = len(omega_range)

        kernel_trace = np.zeros(nf)
        kernel_determinant = np.zeros(nf)
        kernel_condition = np.zeros(nf)

        for i, omega in enumerate(omega_range):
            H_u = self.compute_entropy_kernel(omega, L)

            kernel_trace[i] = np.real(np.trace(H_u))
            kernel_determinant[i] = np.real(np.linalg.det(H_u))

            # Condition number (ratio of largest to smallest eigenvalue)
            eigenvals = np.linalg.eigvals(H_u)
            eigenvals_real = np.real(eigenvals)
            eigenvals_positive = eigenvals_real[eigenvals_real > 1e-12]  # Remove near-zero eigenvals

            if len(eigenvals_positive) > 0:
                kernel_condition[i] = np.max(eigenvals_positive) / np.min(eigenvals_positive)
            else:
                kernel_condition[i] = np.inf

        return {
            "frequencies": omega_range,
            "kernel_trace": kernel_trace,
            "kernel_determinant": kernel_determinant,
            "kernel_condition": kernel_condition,
        }
