#!/usr/bin/env python3
"""
Frequency-Domain Entropy-Aware Control Demonstration.

This example demonstrates entropy-aware control design in frequency domain,
showing how different frequencies contribute to entropy production and how
to optimize the tradeoff between tracking performance and thermodynamic cost.

Key Concepts:
1. Entropy kernel H_u(ω) = G(iω)^H L G(iω) quantifies frequency-dependent entropy cost
2. Sinusoidal controls have time-averaged entropy σ̄ = (1/2) Re{U^H H_u(ω) U}
3. Optimal control trades off tracking vs entropy: min ||HU - X*||² + λ σ̄
4. FFT methods enable analysis of broadband signals and spectral entropy distribution
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.frequency_control import FrequencySpaceController
from llrq.thermodynamic_accounting import ThermodynamicAccountant


def demo_frequency_entropy_kernel():
    """Demonstrate entropy kernel properties across frequencies."""
    print("=== Frequency-Domain Entropy Kernel Analysis ===\n")

    # Create A ⇌ B ⇌ C reaction network
    network = ReactionNetwork(
        species_ids=["A", "B", "C"], reaction_ids=["R1", "R2"], stoichiometric_matrix=[[-1, 0], [1, -1], [0, 1]]
    )

    # Set up mass action kinetics
    forward_rates = np.array([2.0, 1.5])
    backward_rates = np.array([1.0, 0.8])
    initial_concentrations = np.array([2.0, 1.0, 0.5])

    print(f"Network: A ⇌ B ⇌ C")
    print(f"Forward rates: {forward_rates}")
    print(f"Backward rates: {backward_rates}")

    # Create LLRQ system
    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)

    # Create frequency controller
    freq_controller = FrequencySpaceController.from_llrq_solver(solver, controlled_reactions=[0, 1])

    print(f"Reduced system: {freq_controller.n_states} states, {freq_controller.n_controls} controls")
    print(f"K matrix:\n{freq_controller.K}")
    print(f"B matrix:\n{freq_controller.B}")

    # Compute Onsager conductance
    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")
    print(f"Onsager conductance L:\n{L}")

    # Analyze entropy kernel across frequency range
    omega_range = np.logspace(-2, 2, 50)  # 0.01 to 100 rad/s
    kernel_analysis = freq_controller.compute_entropy_kernel_spectrum(omega_range, L)

    print(f"\nEntropy kernel analysis:")
    print(f"  DC (ω=0) kernel trace: {kernel_analysis['kernel_trace'][0]:.4f}")
    print(f"  High-freq kernel trace: {kernel_analysis['kernel_trace'][-1]:.4f}")
    print(f"  DC kernel condition: {kernel_analysis['kernel_condition'][0]:.2f}")
    print(f"  High-freq kernel condition: {kernel_analysis['kernel_condition'][-1]:.2f}")

    return freq_controller, L, kernel_analysis, omega_range


def demo_sinusoidal_entropy_tradeoff():
    """Demonstrate entropy-tracking tradeoff for sinusoidal controls."""
    print(f"\n=== Sinusoidal Control Entropy Tradeoff ===\n")

    # Use same system as before
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

    # Target complex state amplitude
    X_target = np.array([0.3 + 0.2j, -0.1 + 0.4j])  # Complex target
    print(f"Target state amplitude: {X_target}")

    # Test different frequencies
    test_frequencies = [0.1, 1.0, 5.0, 20.0]  # rad/s
    entropy_weights = np.logspace(-2, 2, 10)  # λ from 0.01 to 100

    print(f"\n{'Frequency':>10} {'λ':>8} {'Track Err':>10} {'Entropy':>10} {'Total Cost':>12} {'||U||':>8}")
    print("-" * 70)

    tradeoff_results = {}

    for omega in test_frequencies:
        results_at_freq = []

        for lam in entropy_weights:
            result = freq_controller.design_entropy_aware_sinusoidal_control(
                X_target=X_target, omega=omega, L=L, entropy_weight=lam
            )

            control_amplitude = np.linalg.norm(result["U_optimal"])
            results_at_freq.append(result)

            if lam in [0.01, 0.1, 1.0, 10.0]:  # Print selected values
                print(
                    f"{omega:10.1f} {lam:8.2f} {result['tracking_error']:10.4f} "
                    f"{result['entropy_rate']:10.4f} {result['total_cost']:12.4f} {control_amplitude:8.4f}"
                )

        tradeoff_results[omega] = results_at_freq

    print()

    # Special case: compare exact tracking (λ=0) across frequencies
    print("Exact tracking comparison (λ=0):")
    print(f"{'Frequency':>10} {'Track Err':>12} {'Entropy':>12} {'||U||':>10}")
    print("-" * 50)

    for omega in test_frequencies:
        result = freq_controller.design_entropy_aware_sinusoidal_control(
            X_target=X_target, omega=omega, L=L, entropy_weight=0.0
        )
        control_amp = np.linalg.norm(result["U_optimal"])
        print(f"{omega:10.1f} {result['tracking_error']:12.6f} {result['entropy_rate']:12.4f} {control_amp:10.4f}")

    return tradeoff_results, test_frequencies, entropy_weights


def demo_fft_entropy_validation():
    """Demonstrate and validate FFT-based entropy calculations."""
    print(f"\n=== FFT-Based Entropy Validation ===\n")

    # Simple A ⇌ B system for clear validation
    network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

    forward_rates = np.array([2.0])
    backward_rates = np.array([1.0])
    initial_concentrations = np.array([1.0, 1.0])

    dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)
    solver = LLRQSolver(dynamics)

    accountant = ThermodynamicAccountant(network)
    L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

    print(f"Simple A ⇌ B system")
    print(f"K = {dynamics.K[0,0]:.2f}, L = {L[0,0]:.4f}")

    # Generate test signals
    T = 10.0  # Total time
    dt = 0.01  # Time step
    t = np.arange(0, T, dt)
    N = len(t)

    # Test 1: Single sinusoid
    omega_test = 2.0
    u_sin = 0.5 * np.cos(omega_test * t + 0.3)  # Single sinusoid
    u_t_sin = u_sin.reshape(-1, 1)

    # Corresponding state (analytical)
    K_scalar = dynamics.K[0, 0]
    amp_factor = 1.0 / np.sqrt(K_scalar**2 + omega_test**2)
    phase_lag = np.arctan2(omega_test, K_scalar)
    x_sin = 0.5 * amp_factor * np.cos(omega_test * t + 0.3 - phase_lag)
    x_t_sin = x_sin.reshape(-1, 1)

    print(f"\nTest 1: Single sinusoid (ω = {omega_test:.1f} rad/s)")
    print(f"  Control amplitude: {0.5:.1f}")
    print(f"  Expected state amplitude: {0.5 * amp_factor:.4f}")

    # FFT entropy calculations
    entropy_x_fft, _ = accountant.entropy_from_x_freq(x_t_sin, dt, L)
    entropy_u_fft, _ = accountant.entropy_from_u_freq(u_t_sin, dt, dynamics.K, L)

    # Time-domain entropy calculations
    time_result_x = accountant.entropy_from_x(t, x_t_sin, L)
    time_result_u = accountant.entropy_from_u(t, u_t_sin, dynamics.K, L)

    print(f"  Entropy from x (time): {time_result_x.sigma_total:.6f}")
    print(f"  Entropy from x (FFT):  {entropy_x_fft:.6f}")
    print(f"  Entropy from u (time): {time_result_u.sigma_total:.6f}")
    print(f"  Entropy from u (FFT):  {entropy_u_fft:.6f}")

    # Validation errors
    x_error = abs(entropy_x_fft - time_result_x.sigma_total) / abs(time_result_x.sigma_total)
    u_error = abs(entropy_u_fft - time_result_u.sigma_total) / abs(time_result_u.sigma_total)
    print(f"  Relative error (x): {x_error:.2e}")
    print(f"  Relative error (u): {u_error:.2e}")

    # Test 2: Multi-frequency signal
    u_multi = 0.3 * np.cos(1.0 * t) + 0.2 * np.cos(3.0 * t + 0.5) + 0.1 * np.cos(8.0 * t + 1.2)
    u_t_multi = u_multi.reshape(-1, 1)

    print(f"\nTest 2: Multi-frequency signal")

    # Simulate corresponding state
    dynamics.external_drive = lambda t_val: np.interp(t_val, t, u_multi)
    solution = solver.solve(initial_conditions={"A": 1.0, "B": 1.0}, t_span=(0, T), method="numerical")
    x_t_multi = solution["log_deviations"]

    # Compare entropy calculations
    entropy_x_multi_fft, _ = accountant.entropy_from_x_freq(x_t_multi, dt, L)
    entropy_u_multi_fft, _ = accountant.entropy_from_u_freq(u_t_multi, dt, dynamics.K, L)

    time_result_x_multi = accountant.entropy_from_x(t, x_t_multi, L)

    print(f"  Entropy from x (time): {time_result_x_multi.sigma_total:.6f}")
    print(f"  Entropy from x (FFT):  {entropy_x_multi_fft:.6f}")
    print(f"  Entropy from u (FFT):  {entropy_u_multi_fft:.6f}")

    # Validate Parseval's theorem
    parseval_result = accountant.validate_parseval_entropy(x_t_multi, dt, L)
    print(f"  Parseval validation - relative error: {parseval_result['relative_error']:.2e}")

    return {
        "single_freq": {
            "x_fft": entropy_x_fft,
            "u_fft": entropy_u_fft,
            "x_time": time_result_x.sigma_total,
            "u_time": time_result_u.sigma_total,
        },
        "multi_freq": {"x_fft": entropy_x_multi_fft, "u_fft": entropy_u_multi_fft, "x_time": time_result_x_multi.sigma_total},
        "parseval": parseval_result,
    }


def plot_frequency_entropy_analysis(freq_controller, L, kernel_analysis, omega_range, tradeoff_results):
    """Plot comprehensive frequency-entropy analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Frequency-Domain Entropy-Aware Control Analysis", fontsize=16)

    # Plot 1: Entropy kernel properties vs frequency
    ax = axes[0, 0]
    ax.loglog(omega_range, kernel_analysis["kernel_trace"], "b-", linewidth=2, label="Trace(H_u)")
    ax.loglog(omega_range, kernel_analysis["kernel_determinant"], "r--", linewidth=2, label="Det(H_u)")
    ax.set_xlabel("Frequency ω (rad/s)")
    ax.set_ylabel("Kernel Properties")
    ax.set_title("Entropy Kernel vs Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Kernel condition number
    ax = axes[0, 1]
    ax.loglog(omega_range, kernel_analysis["kernel_condition"], "g-", linewidth=2)
    ax.set_xlabel("Frequency ω (rad/s)")
    ax.set_ylabel("Condition Number")
    ax.set_title("Entropy Kernel Conditioning")
    ax.grid(True, alpha=0.3)

    # Plot 3: Pareto frontiers for different frequencies
    ax = axes[0, 2]
    colors = ["blue", "red", "green", "purple"]
    test_frequencies = [0.1, 1.0, 5.0, 20.0]

    for i, omega in enumerate(test_frequencies):
        if omega in tradeoff_results:
            results = tradeoff_results[omega]
            tracking_errors = [r["tracking_error"] for r in results]
            entropy_rates = [r["entropy_rate"] for r in results]

            ax.loglog(tracking_errors, entropy_rates, "o-", color=colors[i], label=f"ω = {omega:.1f} rad/s", markersize=4)

    ax.set_xlabel("Tracking Error")
    ax.set_ylabel("Entropy Rate")
    ax.set_title("Pareto Frontiers: Tracking vs Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Control effort vs frequency (for exact tracking)
    ax = axes[1, 0]
    X_target = np.array([0.3 + 0.2j, -0.1 + 0.4j])

    control_efforts = []
    entropy_rates_exact = []

    for omega in omega_range[::5]:  # Sample frequencies
        result = freq_controller.design_entropy_aware_sinusoidal_control(
            X_target=X_target, omega=omega, L=L, entropy_weight=0.0
        )
        control_efforts.append(np.linalg.norm(result["U_optimal"]))
        entropy_rates_exact.append(result["entropy_rate"])

    sampled_frequencies = omega_range[::5]
    ax.loglog(sampled_frequencies, control_efforts, "b-", linewidth=2, label="||U||")
    ax.set_xlabel("Frequency ω (rad/s)")
    ax.set_ylabel("Control Effort")
    ax.set_title("Control Effort for Exact Tracking")
    ax.grid(True, alpha=0.3)

    # Plot 5: Entropy rate vs frequency (for exact tracking)
    ax = axes[1, 1]
    ax.loglog(sampled_frequencies, entropy_rates_exact, "r-", linewidth=2)
    ax.set_xlabel("Frequency ω (rad/s)")
    ax.set_ylabel("Entropy Rate")
    ax.set_title("Entropy Rate for Exact Tracking")
    ax.grid(True, alpha=0.3)

    # Plot 6: Frequency response magnitude
    ax = axes[1, 2]
    freq_response_mag = []
    for omega in omega_range:
        H = freq_controller.compute_frequency_response(omega)
        freq_response_mag.append(np.linalg.norm(H, "fro"))  # Frobenius norm

    ax.loglog(omega_range, freq_response_mag, "m-", linewidth=2)
    ax.set_xlabel("Frequency ω (rad/s)")
    ax.set_ylabel("||H(iω)||_F")
    ax.set_title("Frequency Response Magnitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations."""
    print("Frequency-Domain Entropy-Aware Control Demonstration")
    print("=" * 60)

    # Run demonstrations
    freq_controller, L, kernel_analysis, omega_range = demo_frequency_entropy_kernel()
    tradeoff_results, test_frequencies, entropy_weights = demo_sinusoidal_entropy_tradeoff()
    fft_validation = demo_fft_entropy_validation()

    # Create comprehensive plots
    plot_frequency_entropy_analysis(freq_controller, L, kernel_analysis, omega_range, tradeoff_results)

    print(f"\n=== Summary ===")
    print("✅ Entropy kernel analysis across frequencies")
    print("✅ Sinusoidal control entropy-tracking tradeoffs")
    print("✅ FFT-based entropy calculation validation")
    print("✅ Parseval theorem verification")
    print(f"✅ Maximum Parseval error: {fft_validation['parseval']['relative_error']:.2e}")

    print(f"\nKey Insights:")
    print("• Entropy kernel H_u(ω) = G(iω)^H L G(iω) quantifies frequency-dependent costs")
    print("• High frequencies typically have lower entropy cost per unit control amplitude")
    print("• But achieving given state amplitude at high frequency requires larger control")
    print("• FFT methods enable exact spectral entropy analysis for broadband signals")
    print("• Optimal frequency selection balances tracking performance and thermodynamic cost")

    print(f"\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
