#!/usr/bin/env python3
"""
CSTR Example with Moiety Dynamics

Demonstrates the block-triangular decomposition for A ⇌ B in a CSTR:
- x = ln([B]/[A]): reaction quotient log (controls composition ratio)
- y = [A] + [B]: total moiety (controls concentration level)

With uniform dilution, the dynamics decouple completely:
  ẋ = -k*x + u_x         (composition control)
  ẏ = -D*y + D*y_in      (level control via flow)

This example shows:
1. Analytical solution using matrix exponential
2. Decoupled control design for quotients vs totals
3. Step responses and steady-state tracking
4. Comparison with numerical integration
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq.open_system_network import create_cstr_network
from llrq.moiety_dynamics import MoietyDynamics
from llrq.moiety_controller import MoietyController
from llrq.open_system_solver import OpenSystemSolver


def main():
    print("=== CSTR Moiety Dynamics Example ===\n")

    # System setup: A ⇌ B
    species_ids = ["A", "B"]
    reaction_ids = ["R1"]
    S = np.array(
        [
            [-1.0],  # A coefficient in R1
            [1.0],
        ]
    )  # B coefficient in R1

    # CSTR parameters
    k = 0.6  # Reaction rate constant
    D = 0.25  # Dilution rate (F_out/V)
    c_in = {"A": 1.0, "B": 0.0}  # Pure A in inlet

    # Create CSTR network
    network = create_cstr_network(
        species_ids=species_ids, reaction_ids=reaction_ids, stoichiometric_matrix=S, dilution_rate=D, inlet_composition=c_in
    )

    print(f"Network: {network}")
    print(f"Is moiety-respecting: {network.is_moiety_respecting_flow()}")

    # Create moiety dynamics
    moiety_dynamics = MoietyDynamics(
        network=network,
        K=k,  # Single relaxation rate
        equilibrium_constants=np.array([1.0]),  # Keq = 1
    )

    # Configure CSTR flow dynamics
    moiety_dynamics.configure_cstr(dilution_rate=D, inlet_composition=np.array([1.0, 0.0]))

    # Display system matrices
    system = moiety_dynamics.get_block_system()
    print(f"\nBlock-triangular system:")
    print(f"A_x (quotient dynamics): \n{system.A_x}")
    print(f"A_y (moiety dynamics): \n{system.A_y}")
    print(f"g_y (moiety inlet): {system.g_y}")
    print(f"Is decoupled: {system.is_decoupled if hasattr(system, 'is_decoupled') else 'N/A'}")

    # Simulation parameters
    t_final = 20.0
    dt = 0.05
    t = np.arange(0, t_final + dt, dt)

    # Initial conditions
    c0 = np.array([0.8, 0.4])  # [A] = 0.8 M, [B] = 0.4 M

    print(f"\nInitial concentrations: [A] = {c0[0]:.2f}, [B] = {c0[1]:.2f}")
    print(f"Initial total: {c0.sum():.2f}")
    print(f"Initial ratio [B]/[A]: {c0[1]/c0[0]:.2f}")

    # === 1. Open-loop response ===
    print("\n1. Open-loop response (no control)...")

    solver = OpenSystemSolver(moiety_dynamics, network)

    result_openloop = solver.solve_analytical(initial_conditions=c0, t_span=t)

    # === 2. Step response in composition control ===
    print("2. Step response in quotient control...")

    # Apply step in u_x at t=0 (composition control)
    u_x_step = np.array([0.5])  # Drive toward more B

    result_step = solver.solve_analytical(initial_conditions=c0, t_span=t, u_x=u_x_step)

    # === 3. Controller design ===
    print("3. Designing decoupled controllers...")

    controller = MoietyController(moiety_dynamics)

    # Design LQR controllers for both blocks
    K_x = controller.design_lqr_x(
        Q=np.array([[1.0]]),  # Penalize quotient error
        R=np.array([[0.1]]),  # Control effort penalty
    )

    K_y = controller.design_lqr_y(
        Q=np.array([[1.0]]),  # Penalize total error
        R=np.array([[0.1]]),  # Control effort penalty
    )

    print(f"x-block LQR gain: {K_x}")
    print(f"y-block LQR gain: {K_y}")

    # Set reference targets
    x_ref = np.array([0.8])  # Target ln([B]/[A]) = 0.8 → [B]/[A] ≈ 2.23
    y_ref = np.array([2.0])  # Target total = 2.0 M

    # Design feedforward controllers
    controller.design_feedforward_x(x_ref)
    controller.design_feedforward_y(y_ref)

    print(f"Reference quotient log: {x_ref[0]:.2f} (ratio [B]/[A] = {np.exp(x_ref[0]):.2f})")
    print(f"Reference total: {y_ref[0]:.2f} M")

    # === 4. Closed-loop simulation ===
    print("4. Closed-loop tracking simulation...")

    # Get initial x and y
    Q0 = network.compute_reaction_quotients(c0)
    x0 = np.log(Q0 / moiety_dynamics.Keq)
    L = network.find_conservation_laws()
    y0 = L @ c0

    result_closedloop = controller.simulate_closed_loop(t=t, x0=x0, y0=y0, x_ref=x_ref, y_ref=y_ref)

    # === 5. Analysis and Visualization ===
    print("5. Creating visualizations...")

    # Controllability analysis
    controllability = controller.analyze_controllability()
    print(f"\nControllability analysis:")
    print(f"  x-block controllable: {controllability['x_block']['controllable']}")
    print(f"  y-block controllable: {controllability['y_block']['controllable']}")

    # Closed-loop poles
    poles = controller.compute_closed_loop_poles()
    print(f"Closed-loop poles:")
    if "x_block" in poles:
        print(f"  x-block: {poles['x_block']}")
    if "y_block" in poles:
        print(f"  y-block: {poles['y_block']}")

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("CSTR Moiety Dynamics: A ⇌ B with Decoupled Control", fontsize=14)

    # Plot 1: Concentrations
    ax = axes[0, 0]
    ax.plot(t, result_openloop["concentrations"][:, 0], "b-", label="[A] open-loop", linewidth=2)
    ax.plot(t, result_openloop["concentrations"][:, 1], "r-", label="[B] open-loop", linewidth=2)
    ax.plot(t, result_step["concentrations"][:, 0], "b--", label="[A] with step", alpha=0.7)
    ax.plot(t, result_step["concentrations"][:, 1], "r--", label="[B] with step", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration (M)")
    ax.set_title("Species Concentrations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Reaction quotient logs (x)
    ax = axes[0, 1]
    ax.plot(t, result_openloop["x"][:, 0], "g-", label="Open-loop", linewidth=2)
    ax.plot(t, result_step["x"][:, 0], "g--", label="With step", linewidth=2)
    if "x" in result_closedloop:
        ax.plot(t, result_closedloop["x"][:, 0], "purple", label="Closed-loop", linewidth=2)
        if x_ref is not None:
            ax.axhline(x_ref[0], color="k", linestyle=":", label="Reference")
    ax.set_xlabel("Time")
    ax.set_ylabel("x = ln([B]/[A])")
    ax.set_title("Reaction Quotient Log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Moiety totals (y)
    ax = axes[0, 2]
    ax.plot(t, result_openloop["y"][:, 0], "orange", label="Open-loop", linewidth=2)
    ax.plot(t, result_step["y"][:, 0], "orange", linestyle="--", label="With step", alpha=0.7)
    if "y" in result_closedloop:
        ax.plot(t, result_closedloop["y"][:, 0], "brown", label="Closed-loop", linewidth=2)
        if y_ref is not None:
            ax.axhline(y_ref[0], color="k", linestyle=":", label="Reference")
    ax.set_xlabel("Time")
    ax.set_ylabel("y = [A] + [B]")
    ax.set_title("Moiety Total")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Phase portrait (quotient vs total)
    ax = axes[1, 0]
    ax.plot(
        result_openloop["x"][:, 0],
        result_openloop["y"][:, 0],
        "g-",
        label="Open-loop",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=50,
    )
    ax.plot(result_step["x"][:, 0], result_step["y"][:, 0], "g--", label="With step", linewidth=2, alpha=0.7)
    if "x" in result_closedloop and "y" in result_closedloop:
        ax.plot(result_closedloop["x"][:, 0], result_closedloop["y"][:, 0], "purple", label="Closed-loop", linewidth=2)
        if x_ref is not None and y_ref is not None:
            ax.plot(x_ref[0], y_ref[0], "ks", markersize=8, label="Target")
    ax.set_xlabel("x = ln([B]/[A])")
    ax.set_ylabel("y = [A] + [B]")
    ax.set_title("Phase Portrait")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Control inputs
    ax = axes[1, 1]
    if "u_x" in result_closedloop:
        ax.plot(t, result_closedloop["u_x"][:, 0], "blue", label="u_x (quotient)", linewidth=2)
    if "u_y" in result_closedloop and result_closedloop["u_y"].shape[1] > 0:
        ax.plot(t, result_closedloop["u_y"][:, 0], "red", label="u_y (total)", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Control Input")
    ax.set_title("Control Effort")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Tracking errors
    ax = axes[1, 2]
    if "x" in result_closedloop and x_ref is not None:
        x_error = result_closedloop["x"][:, 0] - x_ref[0]
        ax.plot(t, x_error, "blue", label="x error", linewidth=2)
    if "y" in result_closedloop and y_ref is not None:
        y_error = result_closedloop["y"][:, 0] - y_ref[0]
        ax.plot(t, y_error, "red", label="y error", linewidth=2)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Tracking Error")
    ax.set_title("Reference Tracking")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # === 6. Summary ===
    print("\n=== Summary ===")

    # Final states
    c_final_ol = result_openloop["concentrations"][-1]
    c_final_step = result_step["concentrations"][-1]

    print(f"Final concentrations (open-loop): [A] = {c_final_ol[0]:.3f}, [B] = {c_final_ol[1]:.3f}")
    print(f"Final ratio (open-loop): [B]/[A] = {c_final_ol[1]/c_final_ol[0]:.3f}")
    print(f"Final total (open-loop): {c_final_ol.sum():.3f}")

    print(f"Final concentrations (step): [A] = {c_final_step[0]:.3f}, [B] = {c_final_step[1]:.3f}")
    print(f"Final ratio (step): [B]/[A] = {c_final_step[1]/c_final_step[0]:.3f}")
    print(f"Final total (step): {c_final_step.sum():.3f}")

    if "x" in result_closedloop and "y" in result_closedloop:
        x_final_cl = result_closedloop["x"][-1, 0]
        y_final_cl = result_closedloop["y"][-1, 0]
        print(f"Final quotient log (closed-loop): {x_final_cl:.3f} (target: {x_ref[0]:.3f})")
        print(f"Final total (closed-loop): {y_final_cl:.3f} (target: {y_ref[0]:.3f})")

    print(f"\nKey insights:")
    print(f"- x-block (quotient) is controlled by reaction kinetics and energy drives")
    print(f"- y-block (total) is controlled independently by flow rates and inlet composition")
    print(f"- CSTR with uniform dilution enables exact decoupling")
    print(f"- Separate LQR designs optimize composition vs level control")


if __name__ == "__main__":
    main()
