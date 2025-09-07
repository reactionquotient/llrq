#!/usr/bin/env python3
"""
Frequency-Space Control for LLRQ Systems

This example demonstrates open-loop sinusoidal control to achieve target
periodic steady states in LLRQ systems.

Key Concepts:
- Design u(t) = Re{U e^(iωt)} to achieve target periodic state x_ss(t)
- Use frequency response H(iω) = (K + iωI)^(-1)B
- Optimal control: U = (H*WH + λI)^(-1)H*WX*

Examples:
1. Simple A ⇌ B reaction (validates against snippet.py analytical solution)
2. 3-cycle network with controlled oscillations
3. Frequency sweep analysis
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np

import llrq
from llrq.frequency_control import FrequencySpaceController


def example_1_simple_reaction():
    """Example 1: Simple A ⇌ B reaction - validate against analytical solution."""
    print("=" * 60)
    print("Example 1: Simple A ⇌ B Reaction (Analytical Validation)")
    print("=" * 60)

    # Create simple A ⇌ B system (same as snippet.py)
    network, dynamics, solver, visualizer = llrq.simple_reaction(
        reactant_species="A",
        product_species="B",
        equilibrium_constant=2.0,  # Keq
        relaxation_rate=1.0,  # k
        initial_concentrations={"A": 1.0, "B": 0.1},
    )

    print(f"System: {network.summary()}")

    # Extract system matrices (in reduced coordinates)
    K_red = solver._B.T @ dynamics.K @ solver._B  # Reduced relaxation matrix
    B_red = solver._B.T @ np.eye(len(network.reaction_ids))  # Reduced input matrix

    print(f"Reduced system dimensions: {K_red.shape[0]} states, {B_red.shape[1]} controls")
    print(f"K_red = {K_red}")
    print(f"B_red = {B_red}")

    # Create frequency controller
    freq_controller = FrequencySpaceController(K_red, B_red)

    # Problem parameters (matching snippet.py)
    omega = 2.0  # frequency (rad/s)
    U_analytical = 0.6  # control amplitude from snippet.py

    # Compute analytical steady-state amplitude (from snippet.py formula)
    k_scalar = K_red[0, 0]  # For scalar case, K_red is 1x1
    amp_analytical = U_analytical / np.hypot(k_scalar, omega)  # U / sqrt(k^2 + ω^2)
    phase_lag = np.arctan2(omega, k_scalar)  # atan2(ω, k)

    print(f"\nAnalytical solution (from snippet.py):")
    print(f"  Control amplitude: {U_analytical}")
    print(f"  State amplitude: {amp_analytical:.6f}")
    print(f"  Phase lag: {phase_lag:.6f} rad = {np.degrees(phase_lag):.2f}°")

    # Test frequency response computation
    H = freq_controller.compute_frequency_response(omega)
    print(f"\nFrequency response H(iω): {H}")
    print(f"  |H|: {np.abs(H[0,0]):.6f}")
    print(f"  ∠H: {np.angle(H[0,0]):.6f} rad = {np.degrees(np.angle(H[0,0])):.2f}°")

    # Verify: |H| should equal 1/sqrt(k^2 + ω^2) and ∠H should equal -atan2(ω,k)
    expected_mag = 1.0 / np.hypot(k_scalar, omega)
    expected_phase = -np.arctan2(omega, k_scalar)  # Note: negative for phase lag

    print(f"\nValidation:")
    print(f"  Expected |H|: {expected_mag:.6f}, Got: {np.abs(H[0,0]):.6f}")
    print(f"  Expected ∠H: {expected_phase:.6f} rad, Got: {np.angle(H[0,0]):.6f} rad")
    print(f"  Magnitude error: {abs(np.abs(H[0,0]) - expected_mag):.2e}")
    print(f"  Phase error: {abs(np.angle(H[0,0]) - expected_phase):.2e}")

    # Forward problem: given control U, what state amplitude do we get?
    U_control = np.array([U_analytical], dtype=complex)
    X_achieved = H @ U_control

    print(f"\nForward problem (given control, find state):")
    print(f"  Control amplitude: {np.abs(U_control[0]):.6f}")
    print(f"  Achieved state amplitude: {np.abs(X_achieved[0]):.6f}")
    print(f"  Expected state amplitude: {amp_analytical:.6f}")
    print(f"  Error: {abs(np.abs(X_achieved[0]) - amp_analytical):.2e}")

    # Inverse problem: given target state amplitude, design control
    target_amplitude = 0.4  # Target state amplitude
    target_phase = 0.5  # Target phase (rad)
    X_target = target_amplitude * np.exp(1j * target_phase)

    print(f"\nInverse problem (given target state, design control):")
    print(f"  Target state amplitude: {target_amplitude}")
    print(f"  Target phase: {target_phase:.3f} rad = {np.degrees(target_phase):.1f}°")

    # Design optimal control
    U_optimal = freq_controller.design_sinusoidal_control(
        X_target=np.array([X_target]),
        omega=omega,
        lam=1e-6,  # Small regularization
    )

    print(f"  Designed control amplitude: {np.abs(U_optimal[0]):.6f}")
    print(f"  Designed control phase: {np.angle(U_optimal[0]):.6f} rad = {np.degrees(np.angle(U_optimal[0])):.1f}°")

    # Verify achieved vs target
    error_norm, X_achieved_opt = freq_controller.compute_tracking_error(U_optimal, np.array([X_target]), omega)

    print(f"  Achieved state amplitude: {np.abs(X_achieved_opt[0]):.6f}")
    print(f"  Achieved phase: {np.angle(X_achieved_opt[0]):.6f} rad = {np.degrees(np.angle(X_achieved_opt[0])):.1f}°")
    print(f"  Tracking error: {error_norm:.2e}")

    # Time-domain simulation
    print(f"\nTime-domain simulation:")
    t = np.linspace(0, 10, 1000)
    x_ss, u_real = freq_controller.evaluate_steady_state(U_optimal, omega, t)

    print(f"  Simulated for {len(t)} time points from {t[0]:.1f} to {t[-1]:.1f} s")
    print(f"  State oscillation amplitude: {np.max(x_ss[:, 0]) - np.min(x_ss[:, 0]):.6f}")
    print(f"  Control oscillation amplitude: {np.max(u_real[:, 0]) - np.min(u_real[:, 0]):.6f}")

    # Convert back to concentrations for physical interpretation
    Keq = dynamics.Keq[0]
    Q_ss = Keq * np.exp(x_ss[:, 0])  # Q(t) = Keq * exp(x(t))
    C_tot = 1.1  # A0 + B0 from initial conditions
    A_ss = C_tot / (1.0 + Q_ss)
    B_ss = C_tot * Q_ss / (1.0 + Q_ss)

    print(f"  [A] oscillation: {np.min(A_ss):.4f} to {np.max(A_ss):.4f}")
    print(f"  [B] oscillation: {np.min(B_ss):.4f} to {np.max(B_ss):.4f}")

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot state trajectory
    axes[0, 0].plot(t, x_ss[:, 0], "b-", linewidth=2, label="x(t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x = ln(Q/Keq)")
    axes[0, 0].set_title("State Trajectory (Log-Quotient)")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Plot control signal
    axes[0, 1].plot(t, u_real[:, 0], "r-", linewidth=2, label="u(t)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Control Input")
    axes[0, 1].set_title("Control Signal")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Plot concentrations
    axes[1, 0].plot(t, A_ss, "g-", linewidth=2, label="[A](t)")
    axes[1, 0].plot(t, B_ss, "orange", linewidth=2, label="[B](t)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Concentration (M)")
    axes[1, 0].set_title("Physical Concentrations")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Phase portrait
    axes[1, 1].plot(A_ss, B_ss, "purple", linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel("[A] (M)")
    axes[1, 1].set_ylabel("[B] (M)")
    axes[1, 1].set_title("Phase Portrait")
    axes[1, 1].grid(True)
    axes[1, 1].axis("equal")

    plt.tight_layout()
    plt.show()

    return freq_controller, U_optimal, X_target


def example_2_cycle_network():
    """Example 2: 3-cycle network with controlled oscillations."""
    print("\n" + "=" * 60)
    print("Example 2: 3-Cycle Network Oscillations")
    print("=" * 60)

    # Create 3-cycle network (from linear_vs_mass_action_simple.py)
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2", "R3"]

    S = np.array(
        [
            [-1, 0, 1],  # A: -1 in R1, 0 in R2, +1 in R3
            [1, -1, 0],  # B: +1 in R1, -1 in R2, 0 in R3
            [0, 1, -1],  # C: 0 in R1, +1 in R2, -1 in R3
        ]
    )

    species_info = {
        "A": {"name": "A", "initial_concentration": 2.0, "compartment": "cell", "boundary_condition": False},
        "B": {"name": "B", "initial_concentration": 0.2, "compartment": "cell", "boundary_condition": False},
        "C": {"name": "C", "initial_concentration": 0.1, "compartment": "cell", "boundary_condition": False},
    }

    reaction_info = [
        {"id": "R1", "name": "A ⇌ B", "reactants": [("A", 1.0)], "products": [("B", 1.0)], "reversible": True},
        {"id": "R2", "name": "B ⇌ C", "reactants": [("B", 1.0)], "products": [("C", 1.0)], "reversible": True},
        {"id": "R3", "name": "C ⇌ A", "reactants": [("C", 1.0)], "products": [("A", 1.0)], "reversible": True},
    ]

    network = llrq.ReactionNetwork(species_ids, reaction_ids, S, species_info, reaction_info)

    # Set up thermodynamically consistent parameters
    forward_rates = [3.0, 1.0, 3.0]
    backward_rates = [1.5, 2.0, 3.0]
    equilibrium_constants = np.array(forward_rates) / np.array(backward_rates)
    relaxation_matrix = np.diag([2.0, 1.5, 2.5])

    print(f"Network: {len(network.species_ids)} species, {len(network.reaction_ids)} reactions")
    print(f"Equilibrium constants: {equilibrium_constants}")
    print(f"Product of Keq: {np.prod(equilibrium_constants):.3f} (should be 1.0 for cycles)")

    # Create dynamics and solver
    dynamics = llrq.LLRQDynamics(
        network=network, equilibrium_constants=equilibrium_constants, relaxation_matrix=relaxation_matrix
    )

    solver = llrq.LLRQSolver(dynamics)
    print(f"Reduced system dimensions: {solver._rankS}")

    # Get reduced system matrices
    K_red = solver._B.T @ dynamics.K @ solver._B
    B_red = solver._B.T @ np.eye(len(network.reaction_ids))

    print(f"K_red shape: {K_red.shape}")
    print(f"B_red shape: {B_red.shape}")

    # Create frequency controller
    freq_controller = FrequencySpaceController(K_red, B_red)

    # Design oscillatory control
    omega = 1.5  # rad/s

    # Target: oscillations with specific amplitudes in each reduced state
    n_reduced = K_red.shape[0]
    target_amplitudes = [0.3, 0.2][:n_reduced]  # Adjust based on actual dimensions
    target_phases = [0.0, np.pi / 4][:n_reduced]

    X_target = np.array([amp * np.exp(1j * phase) for amp, phase in zip(target_amplitudes, target_phases)])

    print(f"\nFrequency control design:")
    print(f"  Frequency: {omega} rad/s ({omega/(2*np.pi):.2f} Hz)")
    print(f"  Target amplitudes: {target_amplitudes}")
    print(f"  Target phases: {[np.degrees(p) for p in target_phases]} degrees")

    # Design optimal control (control all reactions)
    U_optimal = freq_controller.design_sinusoidal_control(X_target=X_target, omega=omega, lam=0.01)

    print(f"  Control amplitudes: {np.abs(U_optimal)}")
    print(f"  Control phases: {[np.degrees(np.angle(u)) for u in U_optimal]} degrees")

    # Check tracking performance
    error_norm, X_achieved = freq_controller.compute_tracking_error(U_optimal, X_target, omega)

    print(f"  Achieved amplitudes: {np.abs(X_achieved)}")
    print(f"  Tracking error: {error_norm:.2e}")

    # Time-domain simulation
    t = np.linspace(0, 15, 1500)
    x_ss, u_real = freq_controller.evaluate_steady_state(U_optimal, omega, t)

    # Convert to concentration coordinates (approximate)
    # This is a simplification - full conversion requires solving the conservation laws
    print(f"\nTime-domain simulation:")
    print(f"  State oscillation ranges:")
    for i in range(x_ss.shape[1]):
        state_range = np.max(x_ss[:, i]) - np.min(x_ss[:, i])
        print(f"    State {i+1}: {state_range:.4f}")

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot reduced states
    for i in range(x_ss.shape[1]):
        axes[0, 0].plot(t, x_ss[:, i], linewidth=2, label=f"x_{i+1}(t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Reduced State")
    axes[0, 0].set_title("Reduced State Trajectories")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Plot control signals
    for i in range(u_real.shape[1]):
        axes[0, 1].plot(t, u_real[:, i], linewidth=2, label=f"u_{i+1}(t)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Control Input")
    axes[0, 1].set_title("Control Signals")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Phase portrait of reduced states (if 2D)
    if x_ss.shape[1] >= 2:
        axes[1, 0].plot(x_ss[:, 0], x_ss[:, 1], "purple", linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel("x₁")
        axes[1, 0].set_ylabel("x₂")
        axes[1, 0].set_title("Phase Portrait (Reduced States)")
        axes[1, 0].grid(True)
        axes[1, 0].axis("equal")
    else:
        axes[1, 0].text(0.5, 0.5, "Phase portrait\n(need ≥2 states)", ha="center", va="center", transform=axes[1, 0].transAxes)

    # Control phase portrait
    if u_real.shape[1] >= 2:
        axes[1, 1].plot(u_real[:, 0], u_real[:, 1], "red", linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel("u₁")
        axes[1, 1].set_ylabel("u₂")
        axes[1, 1].set_title("Control Phase Portrait")
        axes[1, 1].grid(True)
        axes[1, 1].axis("equal")
    else:
        axes[1, 1].plot(t, u_real[:, 0], "red", linewidth=2)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Control")
        axes[1, 1].set_title("Control Signal")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    return freq_controller, network, dynamics


def example_3_frequency_sweep():
    """Example 3: Frequency sweep analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Frequency Sweep Analysis")
    print("=" * 60)

    # Use simple 2x2 system for clear visualization
    K = np.array([[1.5, 0.2], [0.2, 2.0]])  # Coupled system
    B = np.array([[1.0, 0.0], [0.0, 1.0]])  # Independent controls

    freq_controller = FrequencySpaceController(K, B)

    # Frequency range
    omega_range = np.logspace(-1, 1, 100)  # 0.1 to 10 rad/s

    print(f"System matrices:")
    print(f"K = \n{K}")
    print(f"B = \n{B}")
    print(f"Frequency range: {omega_range[0]:.2f} to {omega_range[-1]:.2f} rad/s")

    # Compute frequency response
    magnitude, phase = freq_controller.frequency_sweep(omega_range)

    # Create Bode plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Magnitude plots
    for i in range(2):  # states
        for j in range(2):  # controls
            axes[0, j].semilogx(omega_range, 20 * np.log10(magnitude[:, i, j]), linewidth=2, label=f"State {i+1}")

    for j in range(2):
        axes[0, j].set_xlabel("Frequency (rad/s)")
        axes[0, j].set_ylabel("Magnitude (dB)")
        axes[0, j].set_title(f"Magnitude: Control {j+1} → States")
        axes[0, j].grid(True)
        axes[0, j].legend()

    # Phase plots
    for i in range(2):  # states
        for j in range(2):  # controls
            axes[1, j].semilogx(omega_range, phase[:, i, j], linewidth=2, label=f"State {i+1}")

    for j in range(2):
        axes[1, j].set_xlabel("Frequency (rad/s)")
        axes[1, j].set_ylabel("Phase (degrees)")
        axes[1, j].set_title(f"Phase: Control {j+1} → States")
        axes[1, j].grid(True)
        axes[1, j].legend()

    plt.tight_layout()
    plt.show()

    # Test control design at different frequencies
    print(f"\nControl design at different frequencies:")
    test_frequencies = [0.5, 1.0, 2.0, 5.0]
    X_target = np.array([1.0, 0.5j])  # Complex target

    for omega in test_frequencies:
        U_opt = freq_controller.design_sinusoidal_control(X_target, omega, lam=0.01)
        error, X_achieved = freq_controller.compute_tracking_error(U_opt, X_target, omega)

        print(f"  ω = {omega:4.1f} rad/s: |U| = {np.abs(U_opt)}, error = {error:.2e}")


if __name__ == "__main__":
    print("Frequency-Space Control for LLRQ Systems")
    print("=========================================")

    # Run examples
    try:
        example_1_simple_reaction()
        example_2_cycle_network()
        example_3_frequency_sweep()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback

        traceback.print_exc()
