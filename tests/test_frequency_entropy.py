"""
Tests for frequency-domain entropy-aware control functionality.

Tests the new FFT-based entropy methods in ThermodynamicAccountant and
entropy-aware control methods in FrequencySpaceController.
"""

import pytest
import numpy as np
from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.frequency_control import FrequencySpaceController
from llrq.thermodynamic_accounting import ThermodynamicAccountant


@pytest.fixture
def simple_network():
    """Simple A ⇌ B reaction network."""
    return ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])


@pytest.fixture
def simple_system(simple_network):
    """Simple A ⇌ B LLRQ system with controller and accountant."""
    forward_rates = np.array([2.0])
    backward_rates = np.array([1.0])
    initial_concentrations = np.array([1.0, 1.0])

    dynamics = LLRQDynamics.from_mass_action(simple_network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)
    freq_controller = FrequencySpaceController.from_llrq_solver(solver, controlled_reactions=[0])

    accountant = ThermodynamicAccountant(simple_network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

    return {
        "network": simple_network,
        "dynamics": dynamics,
        "solver": solver,
        "freq_controller": freq_controller,
        "accountant": accountant,
        "L": L,
        "K": dynamics.K,
    }


@pytest.fixture
def chain_system():
    """A ⇌ B ⇌ C reaction chain system."""
    network = ReactionNetwork(
        species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
    )

    forward_rates = np.array([2.0, 1.5])
    backward_rates = np.array([1.0, 0.8])
    initial_concentrations = np.array([2.0, 1.0, 0.5])

    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)
    freq_controller = FrequencySpaceController.from_llrq_solver(solver, controlled_reactions=[0, 1])

    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

    return {
        "network": network,
        "freq_controller": freq_controller,
        "accountant": accountant,
        "L": L,
        "K": dynamics.K,
    }


class TestFFTEntropyMethods:
    """Test FFT-based entropy calculation methods."""

    def test_entropy_from_x_freq_parseval(self, simple_system):
        """Test that FFT entropy calculation satisfies Parseval's theorem."""
        # Generate test state trajectory
        T = 5.0
        dt = 0.02
        t = np.arange(0, T, dt)
        omega_test = 1.5
        x_t = 0.3 * np.cos(omega_test * t + 0.2)
        x_t = x_t.reshape(-1, 1)

        accountant = simple_system["accountant"]
        L = simple_system["L"]

        # Time-domain entropy
        time_result = accountant.entropy_from_x(t, x_t, L)

        # Frequency-domain entropy
        freq_entropy, _ = accountant.entropy_from_x_freq(x_t, dt, L)

        # Should match within numerical integration tolerance (trapz vs rectangular rule)
        relative_error = abs(freq_entropy - time_result.sigma_total) / abs(time_result.sigma_total)
        assert relative_error < 0.005, f"Parseval's theorem violated, error: {relative_error}"

    def test_entropy_from_u_freq_consistency(self, simple_system):
        """Test consistency of control entropy calculation."""
        # Generate slowly-varying control signal for quasi-steady approximation validity
        T = 10.0
        dt = 0.02
        t = np.arange(0, T, dt)
        u_t = 0.3 * np.sin(0.2 * t) + 0.1 * np.cos(0.5 * t)  # Low frequencies for quasi-steady validity
        u_t = u_t.reshape(-1, 1)

        accountant = simple_system["accountant"]
        K = simple_system["K"]
        L = simple_system["L"]

        # FFT-based calculation (exact)
        entropy_u_fft, _ = accountant.entropy_from_u_freq(u_t, dt, K, L)

        # Time-domain calculation (quasi-steady approximation)
        entropy_u_time = accountant.entropy_from_u(t, u_t, K, L)

        # Should be reasonably close when quasi-steady approximation is valid
        relative_error = abs(entropy_u_fft - entropy_u_time.sigma_total) / abs(entropy_u_time.sigma_total)
        assert relative_error < 0.05, f"Inconsistent control entropy calculation, error: {relative_error}"

    def test_freqs_helper_function(self, simple_system):
        """Test frequency grid generation."""
        accountant = simple_system["accountant"]

        N = 16
        dt = 0.1
        freqs = accountant._freqs(N, dt)

        # Check properties
        assert len(freqs) == N
        assert freqs[0] == 0.0  # DC component
        # For FFT frequency grid, not exactly symmetric due to Nyquist handling

        # Check against numpy
        expected_freqs = 2 * np.pi * np.fft.fftfreq(N, d=dt)
        assert np.allclose(freqs, expected_freqs)

    def test_map_xref_to_u(self, simple_system):
        """Test mapping from desired state spectrum to control spectrum."""
        accountant = simple_system["accountant"]
        K = simple_system["K"]

        N = 8
        dt = 0.1
        # Create test state spectrum
        Xref = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)

        # Map to control spectrum
        U = accountant.map_xref_to_u(Xref, dt, K)

        assert U.shape == Xref.shape
        assert U.dtype == np.complex128

        # Verify relationship for a few frequency bins
        freqs = accountant._freqs(N, dt)
        for k in [0, 1, N // 2]:  # Test DC, positive freq, Nyquist
            expected_U_k = (K + 1j * freqs[k] * np.eye(K.shape[0])) @ Xref[k]
            assert np.allclose(U[k], expected_U_k), f"Incorrect mapping at frequency bin {k}"

    def test_validate_parseval_entropy(self, simple_system):
        """Test Parseval validation method."""
        # Generate simple harmonic state
        T = 6.0
        dt = 0.02
        t = np.arange(0, T, dt)
        x_t = 0.4 * np.cos(1.0 * t) + 0.2 * np.sin(3.0 * t)
        x_t = x_t.reshape(-1, 1)

        accountant = simple_system["accountant"]
        L = simple_system["L"]

        validation = accountant.validate_parseval_entropy(x_t, dt, L)

        assert "time_domain" in validation
        assert "frequency_domain" in validation
        assert "relative_error" in validation

        # Error should be small (accounts for trapz vs rectangular integration differences)
        assert validation["relative_error"] < 0.005, f"Parseval validation failed: {validation['relative_error']}"


class TestEntropyKernel:
    """Test entropy kernel computation and properties."""

    def test_entropy_kernel_hermitian(self, simple_system):
        """Test that entropy kernel is Hermitian."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        omega = 2.0
        H_u = freq_controller.compute_entropy_kernel(omega, L)

        # Should be Hermitian: H_u = H_u^H
        assert np.allclose(H_u, np.conj(H_u.T)), "Entropy kernel should be Hermitian"

    def test_entropy_kernel_positive_semidefinite(self, simple_system):
        """Test that entropy kernel is positive semidefinite."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        # Test at various frequencies
        for omega in [0.0, 0.5, 2.0, 10.0]:
            H_u = freq_controller.compute_entropy_kernel(omega, L)
            eigenvals = np.linalg.eigvals(H_u)
            real_eigenvals = np.real(eigenvals)

            assert np.all(real_eigenvals >= -1e-10), f"Entropy kernel not PSD at ω={omega}"

    def test_entropy_kernel_frequency_scaling(self, simple_system):
        """Test entropy kernel behavior at extreme frequencies."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]
        K = simple_system["K"]

        # At DC (ω=0): H_u(0) = B^T K^{-T} L K^{-1} B
        H_u_dc = freq_controller.compute_entropy_kernel(0.0, L)
        B = freq_controller.B
        K_inv = np.linalg.inv(K)
        expected_dc = B.T @ K_inv.T @ L @ K_inv @ B
        assert np.allclose(H_u_dc, expected_dc), "Incorrect DC entropy kernel"

        # At high frequency: H_u(ω) ≈ (1/ω²) B^T L B for large ω
        omega_high = 100.0
        H_u_high = freq_controller.compute_entropy_kernel(omega_high, L)
        expected_high_approx = (1 / omega_high**2) * B.T @ L @ B

        # Rough approximation check (within factor of 2)
        ratio = np.trace(H_u_high) / np.trace(expected_high_approx)
        assert 0.5 < ratio < 2.0, f"High-frequency scaling incorrect: {ratio}"


class TestSinusoidalEntropyControl:
    """Test sinusoidal entropy-aware control methods."""

    def test_sinusoidal_entropy_rate_positive(self, simple_system):
        """Test that sinusoidal entropy rate is non-negative."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        # Test various control amplitudes and frequencies
        test_cases = [
            (np.array([1.0 + 0.5j]), 1.0),
            (np.array([0.3 - 0.8j]), 0.1),
            (np.array([2.0 + 1.2j]), 5.0),
        ]

        for U, omega in test_cases:
            entropy_rate = freq_controller.compute_sinusoidal_entropy_rate(U, omega, L)
            assert entropy_rate >= 0, f"Negative entropy rate: {entropy_rate}"

    def test_sinusoidal_entropy_rate_scaling(self, simple_system):
        """Test quadratic scaling of entropy rate with amplitude."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        U_base = np.array([0.5 + 0.3j])
        omega = 1.5

        entropy_base = freq_controller.compute_sinusoidal_entropy_rate(U_base, omega, L)
        entropy_scaled = freq_controller.compute_sinusoidal_entropy_rate(2.0 * U_base, omega, L)

        # Should scale quadratically
        expected_ratio = 4.0
        actual_ratio = entropy_scaled / entropy_base
        assert abs(actual_ratio - expected_ratio) < 1e-10, f"Scaling not quadratic: {actual_ratio} vs {expected_ratio}"

    def test_entropy_aware_control_zero_weight(self, simple_system):
        """Test that zero entropy weight gives exact tracking."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        X_target = np.array([0.5 + 0.2j])
        omega = 2.0

        result = freq_controller.design_entropy_aware_sinusoidal_control(X_target, omega, L, entropy_weight=0.0)

        # Should achieve exact tracking
        assert np.allclose(result["X_achieved"], X_target, atol=1e-10), "Zero entropy weight should give exact tracking"
        assert np.allclose(result["tracking_error"], 0.0, atol=1e-10), "Tracking error should be zero"

    def test_entropy_aware_control_tradeoff_monotonic(self, simple_system):
        """Test that increasing entropy weight monotonically affects costs."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        X_target = np.array([0.8 + 0.1j])
        omega = 1.0
        entropy_weights = [0.1, 1.0, 10.0]

        results = []
        for lam in entropy_weights:
            result = freq_controller.design_entropy_aware_sinusoidal_control(X_target, omega, L, entropy_weight=lam)
            results.append(result)

        # Entropy rates should decrease with increasing weight
        assert results[1]["entropy_rate"] <= results[0]["entropy_rate"]
        assert results[2]["entropy_rate"] <= results[1]["entropy_rate"]

        # Tracking errors should increase with increasing weight
        assert results[1]["tracking_error"] >= results[0]["tracking_error"]
        assert results[2]["tracking_error"] >= results[1]["tracking_error"]

    def test_entropy_aware_control_cost_consistency(self, simple_system):
        """Test that total cost equals sum of components."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        X_target = np.array([0.4 - 0.3j])
        omega = 3.0
        entropy_weight = 2.5

        result = freq_controller.design_entropy_aware_sinusoidal_control(X_target, omega, L, entropy_weight=entropy_weight)

        expected_total = result["tracking_error"] + entropy_weight * result["entropy_rate"]
        assert np.allclose(
            result["total_cost"], expected_total
        ), f"Cost inconsistency: {result['total_cost']} vs {expected_total}"


class TestFrequencyEntropyAnalysis:
    """Test comprehensive frequency-entropy analysis methods."""

    def test_frequency_entropy_tradeoff_analysis(self, chain_system):
        """Test comprehensive tradeoff analysis across frequencies."""
        freq_controller = chain_system["freq_controller"]
        L = chain_system["L"]

        X_target = np.array([0.3 + 0.1j, -0.2 + 0.4j])
        omega_range = np.array([0.1, 1.0, 5.0])
        entropy_weights = np.array([0.1, 1.0, 10.0])

        analysis = freq_controller.analyze_frequency_entropy_tradeoff(X_target, omega_range, L, entropy_weights)

        # Check structure
        assert "tracking_errors" in analysis
        assert "entropy_rates" in analysis
        assert "total_costs" in analysis
        assert "control_amplitudes" in analysis

        # Check dimensions
        nf, nw = len(omega_range), len(entropy_weights)
        assert analysis["tracking_errors"].shape == (nf, nw)
        assert analysis["entropy_rates"].shape == (nf, nw)

        # Check monotonicity in entropy weight
        for i in range(nf):
            for j in range(nw - 1):
                # Entropy should decrease with increasing weight
                assert analysis["entropy_rates"][i, j + 1] <= analysis["entropy_rates"][i, j]
                # Tracking error should increase with increasing weight
                assert analysis["tracking_errors"][i, j + 1] >= analysis["tracking_errors"][i, j]

    def test_entropy_kernel_spectrum(self, chain_system):
        """Test entropy kernel spectrum analysis."""
        freq_controller = chain_system["freq_controller"]
        L = chain_system["L"]

        omega_range = np.logspace(-1, 1, 10)  # 0.1 to 10 rad/s
        spectrum = freq_controller.compute_entropy_kernel_spectrum(omega_range, L)

        # Check structure
        assert "kernel_trace" in spectrum
        assert "kernel_determinant" in spectrum
        assert "kernel_condition" in spectrum

        # Check properties
        assert len(spectrum["kernel_trace"]) == len(omega_range)
        assert np.all(spectrum["kernel_trace"] >= 0), "Kernel trace should be non-negative"
        assert np.all(spectrum["kernel_condition"] >= 1), "Condition number should be >= 1"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_frequency_handling(self, simple_system):
        """Test handling of zero frequency (DC)."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        # Should not crash at DC
        H_u = freq_controller.compute_entropy_kernel(0.0, L)
        assert np.isfinite(H_u).all(), "DC entropy kernel should be finite"

        # Entropy rate at DC
        U = np.array([1.0 + 0.0j])
        entropy_rate = freq_controller.compute_sinusoidal_entropy_rate(U, 0.0, L)
        assert np.isfinite(entropy_rate), "DC entropy rate should be finite"

    def test_high_frequency_stability(self, simple_system):
        """Test numerical stability at high frequencies."""
        freq_controller = simple_system["freq_controller"]
        L = simple_system["L"]

        omega_high = 1000.0
        H_u = freq_controller.compute_entropy_kernel(omega_high, L)
        assert np.isfinite(H_u).all(), "High frequency entropy kernel should be finite"

        # Control design should still work
        X_target = np.array([0.1 + 0.05j])
        result = freq_controller.design_entropy_aware_sinusoidal_control(X_target, omega_high, L)
        assert np.isfinite(result["U_optimal"]).all(), "High frequency control should be finite"

    def test_empty_inputs_handling(self, simple_system):
        """Test handling of edge case inputs."""
        accountant = simple_system["accountant"]
        L = simple_system["L"]

        # Empty time series should raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            empty_signal = np.empty((0, 1))
            accountant.entropy_from_x_freq(empty_signal, 0.1, L)

    def test_singular_onsager_matrix(self, simple_system):
        """Test behavior with singular Onsager matrix."""
        freq_controller = simple_system["freq_controller"]

        # Singular (zero) Onsager matrix
        L_singular = np.zeros_like(simple_system["L"])

        # Should not crash, entropy should be zero
        omega = 1.0
        H_u = freq_controller.compute_entropy_kernel(omega, L_singular)
        U_test = np.array([1.0 + 0.5j])
        entropy_rate = freq_controller.compute_sinusoidal_entropy_rate(U_test, omega, L_singular)

        assert np.allclose(entropy_rate, 0.0), "Singular Onsager should give zero entropy"


if __name__ == "__main__":
    pytest.main([__file__])
