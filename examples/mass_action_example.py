#!/usr/bin/env python3
"""
Mass action dynamics example for the LLRQ package.

This example demonstrates how to:
1. Create a mass action reaction network
2. Compute the dynamics matrix K using the Diamond (2025) algorithm
3. Solve the resulting log-linear dynamics
4. Compare equilibrium and non-equilibrium approaches
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llrq import ReactionNetwork, LLRQDynamics


def example_enzymatic_reaction():
    """Example: Simple enzymatic reaction E + S ⇌ ES → E + P."""
    print("Example 1: Enzymatic Reaction Network")
    print("=" * 40)
    
    # Network: E + S ⇌ ES → E + P
    # Reactions: R1: E + S ⇌ ES, R2: ES → E + P
    species_ids = ['E', 'S', 'ES', 'P']
    reaction_ids = ['R1_forward', 'R1_backward', 'R2']
    
    S = np.array([
        [-1, 1, 1],   # E: consumed in R1f, produced in R1b and R2
        [-1, 1, 0],   # S: consumed in R1f, produced in R1b
        [1, -1, -1],  # ES: produced in R1f, consumed in R1b and R2
        [0, 0, 1]     # P: produced in R2
    ])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    print("Network summary:")
    print(network.summary())
    
    # Set parameters for steady state
    c_star = np.array([1.0, 2.0, 0.1, 0.5])  # Concentrations at operating point
    k_plus = np.array([2.0, 1.0, 5.0])       # Forward rates
    k_minus = np.array([1.0, 2.0, 1e-6])     # Backward rates (R2 nearly irreversible)
    
    print(f"\nOperating point concentrations: {dict(zip(species_ids, c_star))}")
    print(f"Forward rates: {dict(zip(reaction_ids, k_plus))}")
    print(f"Backward rates: {dict(zip(reaction_ids, k_minus))}")
    
    # Compute dynamics matrix (equilibrium approximation)
    print("\n--- Equilibrium Mode ---")
    try:
        dynamics_eq = LLRQDynamics.from_mass_action(
            network=network,
            equilibrium_point=c_star,
            forward_rates=k_plus,
            backward_rates=k_minus,
            mode='equilibrium',
            reduce_basis=True,
            enforce_symmetry=True
        )
        
        mass_action_info = dynamics_eq.get_mass_action_info()
        dynamics_data = mass_action_info['dynamics_data']
        
        print(f"Dynamics matrix K shape: {dynamics_eq.K.shape}")
        print(f"K matrix:\n{dynamics_eq.K}")
        
        eigenanalysis = dynamics_data['eigenanalysis']
        print(f"System stable: {eigenanalysis['is_stable']}")
        print(f"Eigenvalues: {eigenanalysis['eigenvalues']}")
        
        if 'K_reduced' in dynamics_data:
            print(f"Reduced to basis dimension: {dynamics_data['K_reduced'].shape[0]}")
    
    except Exception as e:
        print(f"Equilibrium mode failed: {e}")
    
    # Compute dynamics matrix (non-equilibrium)
    print("\n--- Non-equilibrium Mode ---")
    try:
        dynamics_neq = LLRQDynamics.from_mass_action(
            network=network,
            equilibrium_point=c_star,
            forward_rates=k_plus,
            backward_rates=k_minus,
            mode='nonequilibrium',
            reduce_basis=True
        )
        
        mass_action_info_neq = dynamics_neq.get_mass_action_info()
        dynamics_data_neq = mass_action_info_neq['dynamics_data']
        
        print(f"Dynamics matrix K shape: {dynamics_neq.K.shape}")
        print(f"K matrix:\n{dynamics_neq.K}")
        
        eigenanalysis_neq = dynamics_data_neq['eigenanalysis']
        print(f"System stable: {eigenanalysis_neq['is_stable']}")
        print(f"Eigenvalues: {eigenanalysis_neq['eigenvalues']}")
        
    except Exception as e:
        print(f"Non-equilibrium mode failed: {e}")
    
    print()


def example_metabolic_pathway():
    """Example: Linear metabolic pathway A → B → C → D."""
    print("Example 2: Linear Metabolic Pathway")
    print("=" * 40)
    
    # Network: A ⇌ B ⇌ C ⇌ D (each step reversible)
    species_ids = ['A', 'B', 'C', 'D']
    reaction_ids = ['R1', 'R2', 'R3']
    
    S = np.array([
        [-1, 0, 0],   # A
        [1, -1, 0],   # B
        [0, 1, -1],   # C
        [0, 0, 1]     # D
    ])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    print("Network summary:")
    print(network.summary())
    
    # Set parameters
    c_star = np.array([2.0, 1.5, 1.0, 0.5])  # Steady state concentrations
    k_plus = np.array([1.0, 2.0, 1.5])       # Forward rates
    k_minus = np.array([0.5, 1.0, 0.8])      # Backward rates
    
    print(f"\nSteady state concentrations: {dict(zip(species_ids, c_star))}")
    print(f"Forward/backward rate ratios (Keq): {k_plus/k_minus}")
    
    # Create dynamics with external drive
    def metabolic_drive(t):
        """External drive simulating substrate input and product removal."""
        return np.array([
            0.5,    # R1: substrate A input
            0.0,    # R2: no drive
            -0.3    # R3: product D removal
        ])
    
    dynamics = LLRQDynamics.from_mass_action(
        network=network,
        equilibrium_point=c_star,
        forward_rates=k_plus,
        backward_rates=k_minus,
        mode='equilibrium',
        external_drive=metabolic_drive,
        reduce_basis=False,  # Keep full dimension for this example
        enforce_symmetry=False
    )
    
    print(f"\nDynamics matrix K:\n{dynamics.K}")
    print(f"Equilibrium constants: {dynamics.Keq}")
    
    # Eigenanalysis
    eigen_info = dynamics.compute_eigenanalysis()
    print(f"\nEigen-analysis:")
    print(f"Stable: {eigen_info['is_stable']}")
    print(f"Oscillations: {eigen_info['has_oscillations']}")
    print(f"Timescales: {eigen_info['timescales'].real[eigen_info['timescales'].real > 0]}")
    
    # Simulate dynamics
    print("\nSimulating dynamics...")
    
    # Create solver  
    from llrq import LLRQSolver
    solver = LLRQSolver(dynamics)
    
    # Initial concentrations (away from equilibrium)
    c0 = c_star * np.array([1.2, 0.8, 1.1, 0.9])
    
    # Solve dynamics
    result = solver.solve(
        initial_conditions=c0,
        t_span=(0.0, 10.0),
        n_points=200,
        method='numerical'
    )
    
    print(f"Initial concentrations: {c0}")
    print(f"Final concentrations: {result['concentrations'][-1]}")
    print(f"Final reaction quotients: {result['reaction_quotients'][-1]}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = result['time']
    
    # Plot concentrations
    ax = axes[0, 0]
    for i, species in enumerate(species_ids):
        ax.plot(t, result['concentrations'][:, i], label=species, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title('Species Concentrations with External Drive')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot reaction quotients vs equilibrium
    ax = axes[0, 1]
    for i, rxn in enumerate(reaction_ids):
        ax.plot(t, result['reaction_quotients'][:, i], label=rxn, linewidth=2)
        ax.axhline(y=dynamics.Keq[i], color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Reaction Quotient Q')
    ax.set_title('Reaction Quotients (dashed = equilibrium)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot log deviations
    ax = axes[1, 0]
    log_devs = result['log_deviations']
    for i, rxn in enumerate(reaction_ids):
        ax.plot(t, log_devs[:, i], label=f'ln(Q_{i+1}/Keq_{i+1})', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('ln(Q/Keq)')
    ax.set_title('Log Deviations from Equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot external drive
    ax = axes[1, 1]
    drive_vals = np.array([metabolic_drive(ti) for ti in t])
    for i, rxn in enumerate(reaction_ids):
        ax.plot(t, drive_vals[:, i], label=f'u_{i+1}({rxn})', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('External Drive u(t)')
    ax.set_title('External Drive Signals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('LLRQ Dynamics: Linear Metabolic Pathway', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print()


def example_oscillatory_network():
    """Example: Network that can exhibit oscillations."""
    print("Example 3: Potentially Oscillatory Network")  
    print("=" * 40)
    
    # Simplified version of oscillatory biochemical network
    # A ⇌ B ⇌ C, with C inhibiting the first step
    species_ids = ['A', 'B', 'C']
    reaction_ids = ['R1', 'R2']
    
    S = np.array([
        [-1, 0],   # A
        [1, -1],   # B  
        [0, 1]     # C
    ])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    
    # Parameters that might lead to oscillations
    c_star = np.array([1.0, 0.8, 1.2])
    k_plus = np.array([3.0, 2.0])
    k_minus = np.array([1.0, 0.5])
    
    print(f"Concentrations: {dict(zip(species_ids, c_star))}")
    print(f"Rate constants ratio: {k_plus/k_minus}")
    
    # Test both modes
    for mode in ['equilibrium', 'nonequilibrium']:
        print(f"\n--- {mode.capitalize()} Mode ---")
        
        try:
            dynamics = LLRQDynamics.from_mass_action(
                network=network,
                equilibrium_point=c_star,
                forward_rates=k_plus,
                backward_rates=k_minus,
                mode=mode,
                reduce_basis=True
            )
            
            eigen_info = dynamics.compute_eigenanalysis()
            print(f"Eigenvalues: {eigen_info['eigenvalues']}")
            print(f"Has oscillations: {eigen_info['has_oscillations']}")
            print(f"Is stable: {eigen_info['is_stable']}")
            
            if eigen_info['has_oscillations']:
                # Find oscillatory modes
                oscillatory = np.abs(eigen_info['eigenvalues'].imag) > 1e-6
                if np.any(oscillatory):
                    freqs = eigen_info['eigenvalues'].imag[oscillatory] / (2*np.pi)
                    print(f"Oscillation frequencies: {freqs}")
        
        except Exception as e:
            print(f"Failed: {e}")
    
    print()


def main():
    """Run all examples."""
    print("LLRQ Package: Mass Action Dynamics Examples")
    print("=" * 50)
    print()
    
    try:
        example_enzymatic_reaction()
        example_metabolic_pathway()
        example_oscillatory_network()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()