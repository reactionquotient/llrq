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
