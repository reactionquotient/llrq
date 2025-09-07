"""Tests for frequency-space control functionality."""

import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.frequency_control import FrequencySpaceController


class TestFrequencySpaceController:
    """Test suite for FrequencySpaceController."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple 2x2 system
        self.K_simple = np.array([[1.0, 0.0], [0.0, 2.0]])  # Diagonal
        self.B_simple = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity

        # Coupled system
        self.K_coupled = np.array([[1.5, 0.3], [0.3, 2.0]])
        self.B_coupled = np.array([[1.0, 0.5], [0.0, 1.0]])

        # Scalar system (for analytical validation)
        self.K_scalar = np.array([[1.0]])
        self.B_scalar = np.array([[1.0]])

    def test_initialization(self):
        """Test controller initialization."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        assert controller.n_states == 2
        assert controller.n_controls == 2
        assert np.allclose(controller.K, self.K_simple)
        assert np.allclose(controller.B, self.B_simple)

    def test_initialization_errors(self):
        """Test initialization with invalid inputs."""
        # Non-square K
        with pytest.raises(AssertionError):
            FrequencySpaceController(np.array([[1, 2, 3]]), self.B_simple)

        # Inconsistent dimensions
        with pytest.raises(AssertionError):
            FrequencySpaceController(self.K_simple, np.array([[1], [2], [3]]))

        # Non-symmetric K (warning in real use, but should still work)
        K_nonsymm = np.array([[1.0, 0.5], [0.2, 2.0]])
        # This should not fail in constructor, just issue warning
        FrequencySpaceController(K_nonsymm, self.B_simple)

    def test_frequency_response_scalar(self):
        """Test frequency response for scalar system (analytical comparison)."""
        controller = FrequencySpaceController(self.K_scalar, self.B_scalar)

        k = self.K_scalar[0, 0]  # 1.0
        omega = 2.0

        H = controller.compute_frequency_response(omega)

        # Analytical: H(iω) = 1/(k + iω) = (k - iω)/(k² + ω²)
        expected_H = 1.0 / (k + 1j * omega)

        assert H.shape == (1, 1)
        assert np.allclose(H[0, 0], expected_H)

        # Check magnitude and phase
        expected_mag = 1.0 / np.sqrt(k**2 + omega**2)
        expected_phase = -np.arctan2(omega, k)

        assert np.allclose(np.abs(H[0, 0]), expected_mag)
        assert np.allclose(np.angle(H[0, 0]), expected_phase)

    def test_frequency_response_diagonal(self):
        """Test frequency response for diagonal system."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        omega = 1.5
        H = controller.compute_frequency_response(omega)

        assert H.shape == (2, 2)

        # For diagonal K and B=I: H[i,j] = δ[i,j] / (K[i,i] + iω)
        expected_H = np.zeros((2, 2), dtype=complex)
        expected_H[0, 0] = 1.0 / (self.K_simple[0, 0] + 1j * omega)
        expected_H[1, 1] = 1.0 / (self.K_simple[1, 1] + 1j * omega)

        assert np.allclose(H, expected_H)

    def test_frequency_response_coupled(self):
        """Test frequency response for coupled system."""
        controller = FrequencySpaceController(self.K_coupled, self.B_coupled)

        omega = 1.0
        H = controller.compute_frequency_response(omega)

        assert H.shape == (2, 2)

        # Verify by matrix inversion
        K_complex = self.K_coupled + 1j * omega * np.eye(2)
        expected_H = np.linalg.solve(K_complex, self.B_coupled)

        assert np.allclose(H, expected_H)

    def test_sinusoidal_control_design_scalar(self):
        """Test control design for scalar system."""
        controller = FrequencySpaceController(self.K_scalar, self.B_scalar)

        omega = 2.0
        X_target = np.array([0.5 + 0.3j])  # Target complex amplitude

        U_opt = controller.design_sinusoidal_control(X_target, omega, lam=1e-6)

        assert U_opt.shape == (1,)
        assert np.iscomplexobj(U_opt)

        # Verify optimality: should achieve target exactly for scalar case
        H = controller.compute_frequency_response(omega)
        X_achieved = H @ U_opt

        assert np.allclose(X_achieved, X_target, atol=1e-10)

    def test_sinusoidal_control_design_overdetermined(self):
        """Test control design for overdetermined system (more states than controls)."""
        # 3 states, 2 controls
        K = np.diag([1.0, 1.5, 2.0])
        B = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        controller = FrequencySpaceController(K, B)

        omega = 1.0
        X_target = np.array([0.2, 0.3j, 0.1 - 0.1j])

        U_opt = controller.design_sinusoidal_control(X_target, omega, lam=0.01)

        assert U_opt.shape == (2,)

        # Check that solution minimizes weighted error
        error, X_achieved = controller.compute_tracking_error(U_opt, X_target, omega)

        # Should be a reasonable approximation
        assert error < 1.0

    def test_sinusoidal_control_with_weighting(self):
        """Test control design with non-identity weighting matrix."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        omega = 1.0
        X_target = np.array([1.0, 0.5j])

        # Weight first state more heavily
        W = np.diag([10.0, 1.0])

        U_weighted = controller.design_sinusoidal_control(X_target, omega, W=W, lam=0.01)
        U_uniform = controller.design_sinusoidal_control(X_target, omega, W=None, lam=0.01)

        # Check that solutions are different
        assert not np.allclose(U_weighted, U_uniform)

        # Check tracking errors
        error_weighted, X_w = controller.compute_tracking_error(U_weighted, X_target, omega, W=W)
        error_uniform, X_u = controller.compute_tracking_error(U_uniform, X_target, omega, W=W)

        # Weighted solution should perform better with weighted metric
        assert error_weighted <= error_uniform + 1e-10

    def test_evaluate_steady_state(self):
        """Test steady-state evaluation."""
        controller = FrequencySpaceController(self.K_scalar, self.B_scalar)

        omega = 1.0
        U = np.array([1.0 + 0.5j])
        t = np.linspace(0, 2 * np.pi / omega, 100)

        x_ss, u_real = controller.evaluate_steady_state(U, omega, t)

        assert x_ss.shape == (100, 1)
        assert u_real.shape == (100, 1)

        # Check that signals are sinusoidal
        # u(t) = Re{U e^(iωt)} = |U|cos(ωt + arg(U))
        U_mag = np.abs(U[0])
        U_phase = np.angle(U[0])
        u_expected = U_mag * np.cos(omega * t + U_phase)

        assert np.allclose(u_real[:, 0], u_expected)

        # Check that x_ss is also sinusoidal with correct frequency
        # Should have same frequency as input
        fft_x = np.fft.fft(x_ss[:, 0])
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft_x[1 : len(fft_x) // 2])) + 1  # Skip DC
        peak_freq = abs(freqs[peak_idx])
        expected_freq = omega / (2 * np.pi)

        assert np.abs(peak_freq - expected_freq) < 0.1  # Allow some numerical error

    def test_tracking_error_computation(self):
        """Test tracking error computation."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        omega = 1.0
        X_target = np.array([1.0, 0.5j])

        # Perfect control (should give zero error)
        H = controller.compute_frequency_response(omega)
        U_perfect = np.linalg.pinv(H) @ X_target

        error, X_achieved = controller.compute_tracking_error(U_perfect, X_target, omega)

        assert np.allclose(X_achieved, X_target)
        assert error < 1e-10

    def test_frequency_sweep(self):
        """Test frequency sweep functionality."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        omega_range = np.logspace(-1, 1, 20)  # 0.1 to 10 rad/s

        magnitude, phase = controller.frequency_sweep(omega_range)

        assert magnitude.shape == (20, 2, 2)
        assert phase.shape == (20, 2, 2)

        # Check that magnitude and phase are real
        assert np.all(np.isreal(magnitude))
        assert np.all(np.isreal(phase))

        # Check that magnitude is positive
        assert np.all(magnitude >= 0)

        # Check that phase is in reasonable range
        assert np.all(np.abs(phase) <= 180)

        # For diagonal system, off-diagonal elements should be small
        assert np.allclose(magnitude[:, 0, 1], 0, atol=1e-10)
        assert np.allclose(magnitude[:, 1, 0], 0, atol=1e-10)

    def test_regularization_effect(self):
        """Test effect of regularization parameter."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        omega = 1.0
        X_target = np.array([1.0, 0.5j])

        # Test different regularization values
        lam_small = 1e-6
        lam_large = 1.0

        U_small = controller.design_sinusoidal_control(X_target, omega, lam=lam_small)
        U_large = controller.design_sinusoidal_control(X_target, omega, lam=lam_large)

        # Larger regularization should give smaller control magnitude
        assert np.linalg.norm(U_large) <= np.linalg.norm(U_small)

        # But larger regularization should give worse tracking
        error_small, _ = controller.compute_tracking_error(U_small, X_target, omega)
        error_large, _ = controller.compute_tracking_error(U_large, X_target, omega)

        assert error_small <= error_large + 1e-10

    def test_edge_cases(self):
        """Test edge cases and robustness."""
        controller = FrequencySpaceController(self.K_simple, self.B_simple)

        # Zero frequency
        omega = 0.0
        H = controller.compute_frequency_response(omega)
        expected_H = np.linalg.inv(self.K_simple) @ self.B_simple
        assert np.allclose(H, expected_H)

        # Very high frequency - should approach zero
        omega = 1e6
        H = controller.compute_frequency_response(omega)
        # At high frequency, |H| ≈ |B|/ω
        assert np.all(np.abs(H) < 1e-5)

        # Zero target
        omega = 1.0
        X_target = np.zeros(2, dtype=complex)
        U_opt = controller.design_sinusoidal_control(X_target, omega)

        # Should give small control (regularization effect)
        assert np.linalg.norm(U_opt) < 1.0


@pytest.mark.parametrize("omega", [0.5, 1.0, 2.0, 5.0])
def test_analytical_validation_simple_reaction(omega):
    """Test against analytical solution from snippet.py for simple A⇌B reaction."""
    # System parameters matching snippet.py
    k = 1.0  # relaxation rate
    K = np.array([[k]])
    B = np.array([[1.0]])

    controller = FrequencySpaceController(K, B)

    # Analytical frequency response: H(iω) = 1/(k + iω)
    H_analytical = 1.0 / (k + 1j * omega)
    H_computed = controller.compute_frequency_response(omega)[0, 0]

    assert np.allclose(H_computed, H_analytical)

    # Test magnitude and phase
    mag_analytical = 1.0 / np.sqrt(k**2 + omega**2)
    phase_analytical = -np.arctan2(omega, k)

    assert np.allclose(np.abs(H_computed), mag_analytical)
    assert np.allclose(np.angle(H_computed), phase_analytical)

    # Test control design
    target_amplitude = 0.3
    X_target = np.array([target_amplitude])

    U_opt = controller.design_sinusoidal_control(X_target, omega, lam=1e-10)

    # For scalar case with small regularization, should get exact solution
    U_analytical = X_target[0] / H_analytical
    assert np.allclose(U_opt[0], U_analytical, rtol=1e-6)
