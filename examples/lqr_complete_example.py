#!/usr/bin/env python3
"""
Complete LQR Control Example for LLRQ Dynamics

This example demonstrates Linear Quadratic Regulator (LQR) control
applied to a reaction network using the log-linear reaction quotient framework.

The example shows:
1. Setting up a multi-reaction enzymatic cascade network
2. Configuring an LQR controller with selected actuated reactions
3. Tracking a desired equilibrium state
4. Comparing controlled vs uncontrolled dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import sys
import os

# Add parent directory to path to import llrq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llrq.reaction_network import ReactionNetwork
from src.llrq.llrq_dynamics import LLRQDynamics
from src.llrq.solver import LLRQSolver
from src.llrq.control.lqr import LQRController


def create_enzymatic_cascade_network():
    """
    Create a simple enzymatic cascade network:
    
    A ⇌ B (R1: catalyzed by enzyme E1)
    B ⇌ C (R2: catalyzed by enzyme E2)
    C ⇌ D (R3: catalyzed by enzyme E3)
    
    We'll control reactions R1 and R3 (first and last in cascade)
    """
    # Define species and reactions
    species_ids = ['A', 'B', 'C', 'D']
    reaction_ids = ['R1', 'R2', 'R3']
    
    # Stoichiometric matrix S (species x reactions)
    # R1: A -> B
    # R2: B -> C  
    # R3: C -> D
    S = np.array([
        [-1,  0,  0],  # A
        [ 1, -1,  0],  # B
        [ 0,  1, -1],  # C
        [ 0,  0,  1],  # D
    ])
    
    # Create network
    network = ReactionNetwork(
        species_ids=species_ids,
        reaction_ids=reaction_ids,
        stoichiometric_matrix=S
    )
    
    return network


def main():
    print("=" * 60)
    print("LQR Control of Enzymatic Cascade using LLRQ Framework")
    print("=" * 60)
    
    # Create the reaction network
    network = create_enzymatic_cascade_network()
    print(f"\nNetwork created with {network.n_species} species and {network.n_reactions} reactions")
    print(f"Species: {network.species_ids}")
    print(f"Reactions: {network.reaction_ids}")
    
    # Set equilibrium constants (favor forward direction)
    Keq = np.array([2.0, 1.5, 3.0])  # Equilibrium constants for R1, R2, R3
    
    # Set relaxation matrix (diagonal with different time scales)
    K = np.diag([1.0, 0.8, 1.2])  # Different relaxation rates
    
    # Create dynamics object
    dynamics = LLRQDynamics(
        network=network,
        equilibrium_constants=Keq,
        relaxation_matrix=K
    )
    
    # Create solver
    solver = LLRQSolver(dynamics)
    print(f"\nReduced system dimension: {solver._rankS}")
    
    # Set initial concentrations (far from equilibrium)
    initial_conditions = {
        'A': 2.0,
        'B': 0.1,
        'C': 0.1,
        'D': 0.1
    }
    
    # ============= Uncontrolled Simulation =============
    print("\n" + "=" * 40)
    print("Running uncontrolled simulation...")
    
    uncontrolled_result = solver.solve(
        initial_conditions=initial_conditions,
        t_span=(0.0, 30.0),
        n_points=1000,
        method='numerical'
    )
    
    # ============= LQR Controller Setup =============
    print("\n" + "=" * 40)
    print("Setting up LQR controller...")
    
    # Select reactions R1 and R3 as actuated (can add external drive)
    controlled_reactions = ['R1', 'R3']
    print(f"Controlled reactions: {controlled_reactions}")
    
    # LQR weight matrices
    Q = 1.0 * np.eye(solver._rankS)  # State cost
    R = 0.1 * np.eye(2)  # Control effort cost (2 controlled reactions)
    
    # Create LQR controller WITHOUT integral action for stability
    # (integral action can cause numerical issues with some systems)
    controller = LQRController(
        solver=solver,
        controlled_reactions=controlled_reactions,
        Q=Q,
        R=R,
        integral=False  # Disable integral action for now
    )
    
    # Define reference in reduced coordinates (track equilibrium)
    y_ref = np.zeros(solver._rankS)  # Equilibrium in log-deviation coordinates
    
    # ============= Controlled Simulation =============
    print("\nRunning controlled simulation with LQR...")
    
    controlled_result = solver.simulate_closed_loop(
        initial_conditions=initial_conditions,
        t_span=(0.0, 30.0),
        controller=controller,
        y_ref=y_ref,
        n_points=1000
    )
    
    # ============= Visualization =============
    print("\n" + "=" * 40)
    print("Generating plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('LQR Control of Enzymatic Cascade', fontsize=14, fontweight='bold')
    
    # Time arrays
    t_uncontrolled = uncontrolled_result['time']
    t_controlled = controlled_result['time']
    
    # Plot 1: Species concentrations (uncontrolled)
    ax = axes[0, 0]
    for i, species in enumerate(network.species_ids):
        ax.plot(t_uncontrolled, uncontrolled_result['concentrations'][:, i], 
                label=species, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title('Uncontrolled Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Species concentrations (controlled)
    ax = axes[0, 1]
    for i, species in enumerate(network.species_ids):
        ax.plot(t_controlled, controlled_result['concentrations'][:, i], 
                label=species, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title('LQR Controlled Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reaction quotients (uncontrolled)
    ax = axes[1, 0]
    for i, rxn in enumerate(network.reaction_ids):
        Q = uncontrolled_result['reaction_quotients'][:, i]
        ax.plot(t_uncontrolled, Q, label=rxn, linewidth=2)
        ax.axhline(y=Keq[i], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Reaction Quotient Q')
    ax.set_title('Reaction Quotients (Uncontrolled)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reaction quotients (controlled)
    ax = axes[1, 1]
    for i, rxn in enumerate(network.reaction_ids):
        Q = controlled_result['reaction_quotients'][:, i]
        ax.plot(t_controlled, Q, label=rxn, linewidth=2)
        ax.axhline(y=Keq[i], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Reaction Quotient Q')
    ax.set_title('Reaction Quotients (Controlled)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Log deviations from equilibrium
    ax = axes[2, 0]
    x_uncontrolled = uncontrolled_result['log_deviations']
    x_controlled = controlled_result['log_deviations']
    
    # Compute norm of log deviations
    norm_uncontrolled = np.linalg.norm(x_uncontrolled, axis=1)
    norm_controlled = np.linalg.norm(x_controlled, axis=1)
    
    ax.semilogy(t_uncontrolled, norm_uncontrolled, 'b-', label='Uncontrolled', linewidth=2)
    ax.semilogy(t_controlled, norm_controlled, 'r-', label='LQR Controlled', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('||ln(Q/Keq)||')
    ax.set_title('Distance from Equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Control inputs (reconstructed)
    ax = axes[2, 1]
    
    # Reconstruct control inputs from the controlled simulation
    # This is approximate since we don't store control signals directly
    dt = t_controlled[1] - t_controlled[0]
    y_controlled = controlled_result['log_deviations'] @ solver._B.T
    
    # Approximate control by finite differences
    dy_dt = np.gradient(y_controlled, dt, axis=0)
    A = -solver._B.T @ dynamics.K @ solver._B
    
    # u ≈ dy/dt - A*y (in reduced space)
    u_approx = dy_dt - (y_controlled @ A.T)
    
    # Project to controlled subspace
    if u_approx.shape[0] > 1:
        u_ctrl = u_approx @ solver._B @ controller.G
        ax.plot(t_controlled[1:], u_ctrl[1:, 0], label=f'u_{controlled_reactions[0]}', linewidth=2)
        ax.plot(t_controlled[1:], u_ctrl[1:, 1], label=f'u_{controlled_reactions[1]}', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Control Input')
    ax.set_title('Approximate Control Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ============= Performance Metrics =============
    print("\n" + "=" * 40)
    print("Performance Comparison:")
    
    # Time to reach near equilibrium (within 5% of equilibrium)
    threshold = 0.05
    
    # Find settling time for uncontrolled
    norm_uncontrolled_relative = norm_uncontrolled / norm_uncontrolled[0]
    idx_uncontrolled = np.where(norm_uncontrolled_relative < threshold)[0]
    t_settle_uncontrolled = t_uncontrolled[idx_uncontrolled[0]] if len(idx_uncontrolled) > 0 else np.inf
    
    # Find settling time for controlled
    norm_controlled_relative = norm_controlled / norm_controlled[0]
    idx_controlled = np.where(norm_controlled_relative < threshold)[0]
    t_settle_controlled = t_controlled[idx_controlled[0]] if len(idx_controlled) > 0 else np.inf
    
    print(f"\nSettling time (to 5% of initial deviation):")
    print(f"  Uncontrolled: {t_settle_uncontrolled:.2f} time units")
    print(f"  LQR Controlled: {t_settle_controlled:.2f} time units")
    print(f"  Improvement: {(1 - t_settle_controlled/t_settle_uncontrolled)*100:.1f}%")
    
    # Final equilibrium error
    final_error_uncontrolled = norm_uncontrolled[-1]
    final_error_controlled = norm_controlled[-1]
    
    print(f"\nFinal equilibrium error ||ln(Q/Keq)||:")
    print(f"  Uncontrolled: {final_error_uncontrolled:.2e}")
    print(f"  LQR Controlled: {final_error_controlled:.2e}")
    
    print("\n" + "=" * 60)
    print("LQR control example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()