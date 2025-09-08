#!/usr/bin/env python3
"""
Frequency-Space Control with Full Dynamics Simulation

This example demonstrates the complete workflow of frequency-space control:
1. Design sinusoidal control using FrequencySpaceController
2. Simulate actual system dynamics under this control
3. Compare linear LLRQ vs mass action dynamics
4. Analyze how well the target periodic steady state is achieved

This complements frequency_space_control.py by showing actual dynamics simulation
rather than just analytical steady-state evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

import llrq
from llrq.frequency_control import FrequencySpaceController


def create_3cycle_system():
    """Create the same 3-cycle network used in other examples."""
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2", "R3"]

    # Stoichiometric matrix
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

    # Thermodynamically consistent parameters (Wegscheider condition)
    forward_rates = [3.0, 1.0, 3.0]
    backward_rates = [1.5, 2.0, 3.0]
    equilibrium_constants = np.array(forward_rates) / np.array(backward_rates)
    relaxation_matrix = np.diag([2.0, 1.5, 2.5])

    print(f"Equilibrium constants: {equilibrium_constants}")
    print(f"Wegscheider check: ∏Keq = {np.prod(equilibrium_constants):.3f} (should be 1.0)")

    # Create dynamics from mass action rates for compatibility
    initial_concentrations = np.array([2.0, 0.2, 0.1])  # A, B, C
    dynamics = llrq.LLRQDynamics.from_mass_action(
        network=network,
        forward_rates=forward_rates,
        backward_rates=backward_rates,
        initial_concentrations=initial_concentrations,
    )

    solver = llrq.LLRQSolver(dynamics)

    return network, dynamics, solver


def design_periodic_control(solver, controlled_reactions, omega, target_amplitudes, target_phases):
    """Design sinusoidal control to achieve target periodic steady state."""
    print(f"\nDesigning frequency-space control:")
    print(f"  Frequency: {omega} rad/s ({omega/(2*np.pi):.3f} Hz)")
    print(f"  Controlled reactions: {controlled_reactions}")
    print(f"  Target amplitudes: {target_amplitudes}")
    print(f"  Target phases: {[f'{np.degrees(p):.1f}' for p in target_phases]}°")

    # Create frequency controller
    freq_controller = FrequencySpaceController.from_llrq_solver(solver, controlled_reactions=controlled_reactions)

    print(f"  System dimensions: {freq_controller.n_states} states, {freq_controller.n_controls} controls")

    # Build target complex amplitudes
    X_target = np.array([amp * np.exp(1j * phase) for amp, phase in zip(target_amplitudes, target_phases)])

    # Design optimal control
    U_optimal = freq_controller.design_sinusoidal_control(
        X_target=X_target,
        omega=omega,
        lam=0.01,  # Small regularization
    )

    # Check tracking performance
    error_norm, X_achieved = freq_controller.compute_tracking_error(U_optimal, X_target, omega)

    print(f"  Designed control amplitudes: {np.abs(U_optimal)}")
    print(f"  Designed control phases: {[f'{np.degrees(np.angle(u)):.1f}' for u in U_optimal]}°")
    print(f"  Achieved amplitudes: {np.abs(X_achieved)}")
    print(f"  Tracking error: {error_norm:.2e}")

    return freq_controller, U_optimal, X_target, X_achieved


def create_control_function(U_optimal, omega, controlled_reactions, all_reactions):
    """Create time-varying control function for simulation."""
    # Map controlled reactions to all reactions
    control_map = {}
    for i, rid in enumerate(controlled_reactions):
        if isinstance(rid, str):
            idx = all_reactions.index(rid)
        else:
            idx = rid
        control_map[idx] = i

    def control_function(t):
        """Control function u(t) = Re{U e^(iωt)}."""
        u_full = np.zeros(len(all_reactions))
        u_controlled = np.real(U_optimal * np.exp(1j * omega * t))

        for reaction_idx, control_idx in control_map.items():
            u_full[reaction_idx] = u_controlled[control_idx]

        return u_full

    def controlled_signal_function(t):
        """Return only the controlled reaction signals."""
        return np.real(U_optimal * np.exp(1j * omega * t))

    return control_function, controlled_signal_function


def simulate_with_control(
    solver,
    control_function,
    initial_concentrations,
    t_span,
    method="linear",
    controlled_signal_function=None,
    freq_controller=None,
):
    """Simulate dynamics with time-varying control."""
    print(f"  Running {method} simulation...")

    if method == "linear":
        # Create external drive function for LLRQ dynamics
        def external_drive(t):
            return control_function(t)

        # Temporarily set the external drive
        old_drive = solver.dynamics.external_drive
        solver.dynamics.external_drive = external_drive

        try:
            result = solver.solve(
                initial_conditions=initial_concentrations,
                t_span=t_span,
                n_points=2000,
                method="numerical",  # Use numerical for time-varying control
            )
            result["method"] = "Linear LLRQ"
            result["control_signals"] = np.array([control_function(t) for t in result["time"]])
        finally:
            # Restore original drive
            solver.dynamics.external_drive = old_drive

    elif method == "mass_action":
        try:
            from llrq.mass_action_simulator import MassActionSimulator

            # Create mass action simulator from LLRQ dynamics
            simulator = MassActionSimulator.from_llrq_dynamics(solver.dynamics, solver.network)

            # Create time points
            if isinstance(t_span, tuple):
                t_eval = np.linspace(t_span[0], t_span[1], 2000)
            else:
                t_eval = t_span

            # Adapt control function for mass action simulator
            # MassActionSimulator expects f(t, Q) -> (u_red, u_total)
            def mass_action_control(t, Q):
                u_total = control_function(t)

                # Convert controlled signal to reduced space using B matrix
                if controlled_signal_function is not None and freq_controller is not None:
                    u_controlled = controlled_signal_function(t)
                    u_red = freq_controller.B @ u_controlled
                else:
                    # Fallback to zero control if no proper mapping available
                    u_red = np.zeros(solver._rankS)

                return u_red, u_total

            # Simulate with control
            result = simulator.simulate(t_eval, mass_action_control)
            result["method"] = "Mass Action"

        except ImportError:
            print("    Mass action simulation requires tellurium")
            return None
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def analyze_periodic_steady_state(result, omega, target_X, start_time_ratio=0.7):
    """Analyze the periodic steady state portion of the simulation."""
    if result is None:
        return None

    t = result["time"]

    # Extract steady-state portion (last 30% of simulation)
    start_idx = int(len(t) * start_time_ratio)
    t_ss = t[start_idx:]

    # Get reduced state trajectory if available
    if "reduced_state" in result:
        x_ss = result["reduced_state"][start_idx:]
    else:
        # Convert concentrations to reduced state (approximate)
        # This is a simplification - would need proper conversion
        x_ss = None

    # Analyze frequency content using FFT
    analysis = {}

    if x_ss is not None:
        dt = t_ss[1] - t_ss[0]

        for i in range(x_ss.shape[1]):
            # Remove mean and apply FFT
            x_i = x_ss[:, i] - np.mean(x_ss[:, i])
            X_fft = fft(x_i)
            freqs = fftfreq(len(x_i), dt)

            # Find peak at target frequency
            target_freq = omega / (2 * np.pi)
            freq_idx = np.argmin(np.abs(freqs - target_freq))

            # Extract amplitude and phase
            amplitude = 2 * np.abs(X_fft[freq_idx]) / len(x_i)
            phase = np.angle(X_fft[freq_idx])

            analysis[f"state_{i+1}"] = {
                "amplitude": amplitude,
                "phase": phase,
                "target_amplitude": np.abs(target_X[i]) if i < len(target_X) else 0,
                "target_phase": np.angle(target_X[i]) if i < len(target_X) else 0,
            }

    analysis["method"] = result["method"]
    return analysis


def plot_comparison_results(linear_result, mass_action_result, control_function, omega, t_plot_max=None):
    """Generate comprehensive comparison plots."""
    print(f"\nGenerating plots...")

    # Time vector for plotting
    if t_plot_max is None:
        t_plot_max = min(
            linear_result["time"][-1], mass_action_result["time"][-1] if mass_action_result else linear_result["time"][-1]
        )

    # Create subplot grid
    fig = plt.figure(figsize=(16, 12))

    # 1. Concentrations over time
    for i, species in enumerate(["A", "B", "C"]):
        ax = plt.subplot(3, 4, i + 1)

        # Linear LLRQ
        t_linear = linear_result["time"]
        mask_linear = t_linear <= t_plot_max
        c_linear = linear_result["concentrations"][mask_linear, i]
        ax.plot(t_linear[mask_linear], c_linear, "b-", linewidth=2, label="Linear LLRQ")

        # Mass action (if available)
        if mass_action_result:
            t_mass = mass_action_result["time"]
            mask_mass = t_mass <= t_plot_max
            c_mass = mass_action_result["concentrations"][mask_mass, i]
            ax.plot(t_mass[mask_mass], c_mass, "r--", linewidth=2, label="Mass Action")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"[{species}] (M)")
        ax.set_title(f"Species {species}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Control signals
    ax = plt.subplot(3, 4, 4)
    t_ctrl = linear_result["time"]
    mask_ctrl = t_ctrl <= t_plot_max
    ctrl_signals = linear_result.get("control_signals", np.array([control_function(t) for t in t_ctrl]))

    for i, rid in enumerate(["R1", "R2", "R3"]):
        if i < ctrl_signals.shape[1] and np.any(ctrl_signals[mask_ctrl, i] != 0):
            ax.plot(t_ctrl[mask_ctrl], ctrl_signals[mask_ctrl, i], linewidth=2, label=f"{rid}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Signals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Phase portraits (concentration space)
    ax = plt.subplot(3, 4, 5)
    A_linear = linear_result["concentrations"][mask_linear, 0]
    B_linear = linear_result["concentrations"][mask_linear, 1]
    ax.plot(A_linear, B_linear, "b-", linewidth=2, alpha=0.7, label="Linear LLRQ")

    if mass_action_result:
        A_mass = mass_action_result["concentrations"][mask_mass, 0]
        B_mass = mass_action_result["concentrations"][mask_mass, 1]
        ax.plot(A_mass, B_mass, "r--", linewidth=2, alpha=0.7, label="Mass Action")

    ax.set_xlabel("[A] (M)")
    ax.set_ylabel("[B] (M)")
    ax.set_title("Phase Portrait: A vs B")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Another phase portrait
    ax = plt.subplot(3, 4, 6)
    B_linear = linear_result["concentrations"][mask_linear, 1]
    C_linear = linear_result["concentrations"][mask_linear, 2]
    ax.plot(B_linear, C_linear, "b-", linewidth=2, alpha=0.7, label="Linear LLRQ")

    if mass_action_result:
        B_mass = mass_action_result["concentrations"][mask_mass, 1]
        C_mass = mass_action_result["concentrations"][mask_mass, 2]
        ax.plot(B_mass, C_mass, "r--", linewidth=2, alpha=0.7, label="Mass Action")

    ax.set_xlabel("[B] (M)")
    ax.set_ylabel("[C] (M)")
    ax.set_title("Phase Portrait: B vs C")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Steady-state oscillations (zoomed view)
    steady_start = max(0, int(0.7 * len(t_linear[mask_linear])))
    ax = plt.subplot(3, 4, 7)

    t_zoom = t_linear[mask_linear][steady_start:]
    A_zoom = A_linear[steady_start:]
    ax.plot(t_zoom, A_zoom, "b-", linewidth=2, label="Linear LLRQ")

    if mass_action_result:
        steady_start_mass = max(0, int(0.7 * len(t_mass[mask_mass])))
        t_zoom_mass = t_mass[mask_mass][steady_start_mass:]
        A_zoom_mass = mass_action_result["concentrations"][mask_mass, 0][steady_start_mass:]
        ax.plot(t_zoom_mass, A_zoom_mass, "r--", linewidth=2, label="Mass Action")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("[A] (M)")
    ax.set_title("Steady-State Oscillations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Frequency spectrum
    if "reduced_state" in linear_result:
        ax = plt.subplot(3, 4, 8)

        # Get steady-state portion
        x_steady = linear_result["reduced_state"][steady_start:, 0]  # First reduced state
        dt = t_linear[1] - t_linear[0]

        # Compute FFT
        X_fft = fft(x_steady - np.mean(x_steady))
        freqs = fftfreq(len(x_steady), dt)

        # Plot only positive frequencies
        pos_mask = freqs > 0
        ax.semilogy(freqs[pos_mask], np.abs(X_fft[pos_mask]), "b-", linewidth=2)

        # Mark target frequency
        target_freq = omega / (2 * np.pi)
        ax.axvline(target_freq, color="red", linestyle="--", alpha=0.7, label=f"Target: {target_freq:.2f} Hz")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Frequency Spectrum (State 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2 * target_freq)

    plt.tight_layout()
    plt.show()

    return fig


def run_frequency_space_simulation():
    """Main simulation function."""
    print("=" * 70)
    print("Frequency-Space Control with Full Dynamics Simulation")
    print("=" * 70)

    # 1. Setup system
    print("\n1. Setting up 3-cycle reaction network...")
    network, dynamics, solver = create_3cycle_system()
    print(f"   ✓ Network: {len(network.species_ids)} species, {len(network.reaction_ids)} reactions")
    print(f"   ✓ Reduced system: {solver._rankS} states")

    # 2. Design control
    print("\n2. Designing frequency-space control...")
    controlled_reactions = ["R1", "R3"]  # Control reactions 1 and 3
    omega = 1.5  # rad/s - periodic frequency
    target_amplitudes = [0.25, 0.15]  # Target oscillation amplitudes
    target_phases = [0.0, np.pi / 3]  # Target phase shifts

    freq_controller, U_optimal, X_target, X_achieved = design_periodic_control(
        solver, controlled_reactions, omega, target_amplitudes, target_phases
    )

    # 3. Create control function
    control_function, controlled_signal_function = create_control_function(
        U_optimal, omega, controlled_reactions, network.reaction_ids
    )

    # 4. Run simulations
    print("\n3. Running dynamics simulations...")
    initial_concentrations = {"A": 2.0, "B": 0.2, "C": 0.1}
    t_span = (0.0, 20.0)  # 20 seconds - enough for several periods and transients

    # Linear LLRQ simulation
    linear_result = simulate_with_control(solver, control_function, initial_concentrations, t_span, method="linear")

    # Mass action simulation
    mass_action_result = simulate_with_control(
        solver,
        control_function,
        initial_concentrations,
        t_span,
        method="mass_action",
        controlled_signal_function=controlled_signal_function,
        freq_controller=freq_controller,
    )

    # 5. Analysis
    print("\n4. Analyzing periodic steady states...")

    linear_analysis = analyze_periodic_steady_state(linear_result, omega, X_target)
    mass_action_analysis = analyze_periodic_steady_state(mass_action_result, omega, X_target)

    # Print analysis results
    if linear_analysis:
        print(f"   Linear LLRQ Analysis:")
        for key, data in linear_analysis.items():
            if key.startswith("state_"):
                print(
                    f"     {key}: amplitude {data['amplitude']:.4f} (target {data['target_amplitude']:.4f}), "
                    f"phase {np.degrees(data['phase']):.1f}° (target {np.degrees(data['target_phase']):.1f}°)"
                )

    if mass_action_analysis:
        print(f"   Mass Action Analysis:")
        for key, data in mass_action_analysis.items():
            if key.startswith("state_"):
                print(
                    f"     {key}: amplitude {data['amplitude']:.4f} (target {data['target_amplitude']:.4f}), "
                    f"phase {np.degrees(data['phase']):.1f}° (target {np.degrees(data['target_phase']):.1f}°)"
                )

    # 6. Generate plots
    print("\n5. Generating comparison plots...")
    fig = plot_comparison_results(linear_result, mass_action_result, control_function, omega)

    print("\n" + "=" * 70)
    print("🎉 Simulation completed successfully!")
    print("=" * 70)

    return {
        "linear_result": linear_result,
        "mass_action_result": mass_action_result,
        "control_function": control_function,
        "freq_controller": freq_controller,
        "target_X": X_target,
        "achieved_X": X_achieved,
        "linear_analysis": linear_analysis,
        "mass_action_analysis": mass_action_analysis,
    }


if __name__ == "__main__":
    results = run_frequency_space_simulation()
