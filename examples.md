---
layout: page
title: Examples
nav_order: 4
---

# Examples

Complete working examples demonstrating the LLRQ package capabilities.

## Interactive Tools

### [Mass Action Simulator & Comparison](mass-action-simulator.html)
Interactive comparison between LLRQ and traditional mass action kinetics with embedded visualizations and parameter exploration tools.

## Code Examples

## Simple Reaction Example

**File:** `examples/simple_example.py`

A complete example showing basic LLRQ usage with a single A ⇌ B reaction.

```python
#!/usr/bin/env python3
"""
Simple example demonstrating the LLRQ package.

This example shows how to:
1. Create a simple A ⇌ B reaction system
2. Solve the log-linear dynamics
3. Visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt
import llrq

def main():
    print("LLRQ Package Example: Simple A ⇌ B Reaction")
    print("=" * 50)
    
    # Create a simple reaction system
    network, dynamics, solver, visualizer = llrq.simple_reaction(
        reactant_species="A",
        product_species="B", 
        equilibrium_constant=2.0,
        relaxation_rate=1.0,
        initial_concentrations={"A": 1.0, "B": 0.1}
    )
    
    # Print network summary
    print("\nReaction Network:")
    print(network.summary())
    
    # Solve the dynamics
    solution = solver.solve(
        initial_conditions={"A": 1.0, "B": 0.1},
        t_span=(0, 10),
        method='analytical'
    )
    
    print(f"\nSolution successful: {solution['success']}")
    print(f"Method used: {solution['method']}")
    
    # Plot results
    fig = visualizer.plot_dynamics(solution)
    plt.show()
    
    # Demonstrate external drive
    print("\nExternal drive example:")
    
    def step_drive(t):
        return np.array([0.5 if t > 5 else 0.0])
    
    def oscillating_drive(t):
        return np.array([0.3 * np.sin(2*np.pi*t)])
    
    # Test step drive
    dynamics.external_drive = step_drive
    step_solution = solver.solve(
        initial_conditions={"A": 1.0, "B": 0.1},
        t_span=(0, 10),
        method='numerical'
    )
    
    # Test oscillating drive  
    dynamics.external_drive = oscillating_drive
    osc_solution = solver.solve(
        initial_conditions={"A": 1.0, "B": 0.1},
        t_span=(0, 10),
        method='numerical'
    )
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # No drive
    dynamics.external_drive = lambda t: np.array([0.0])
    no_drive_solution = solver.solve(
        initial_conditions={"A": 1.0, "B": 0.1},
        t_span=(0, 10)
    )
    
    solutions = [
        (no_drive_solution, "No Drive"),
        (step_solution, "Step Drive"),
        (osc_solution, "Oscillating Drive"),
        (solution, "Analytical")
    ]
    
    for i, (sol, title) in enumerate(solutions):
        ax = axes[i//2, i%2]
        t = sol['t']
        ax.plot(t, sol['concentrations']['A'], label='A', linewidth=2)
        ax.plot(t, sol['concentrations']['B'], label='B', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
```

**Key Features Demonstrated:**
- Basic network creation
- Analytical vs numerical solving
- External drive functions
- Visualization methods

---

## Mass Action Dynamics Example

**File:** `examples/mass_action_example.py`

Shows how to use LLRQ with traditional mass action kinetics parameters.

```python
#!/usr/bin/env python3
"""
Mass action dynamics example for the LLRQ package.

This example demonstrates:
1. Converting mass action parameters to LLRQ format
2. Handling enzymatic reactions
3. Comparing equilibrium vs non-equilibrium behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver, LLRQVisualizer

def enzymatic_reaction_example():
    """Example: Simple enzymatic reaction E + S ⇌ ES → E + P."""
    print("Enzymatic Reaction Network: E + S ⇌ ES → E + P")
    print("=" * 50)
    
    # Network topology
    species_ids = ['E', 'S', 'ES', 'P']
    reaction_ids = ['R1_forward', 'R1_backward', 'R2']
    
    # Stoichiometric matrix
    S = np.array([
        [-1,  1,  1],   # E: consumed in R1f, produced in R1b and R2
        [-1,  1,  0],   # S: consumed in R1f, produced in R1b
        [ 1, -1, -1],   # ES: produced in R1f, consumed in R1b and R2
        [ 0,  0,  1]    # P: produced in R2
    ])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    
    # Mass action parameters
    # R1: E + S ⇌ ES with kf1=2.0, kr1=1.0 → Keq1 = kf1/kr1 = 2.0
    # R2: ES → E + P with kf2=5.0, kr2=0.0 → Keq2 = ∞ (irreversible)
    
    kf = np.array([2.0, 1.0, 5.0])  # Forward rates
    kr = np.array([1.0, 2.0, 0.0])  # Backward rates
    
    # Convert to LLRQ parameters
    Keq = np.where(kr > 0, kf/kr, 1e6)  # Large Keq for irreversible reactions
    
    # LLRQ relaxation rates (Diamond 2025 algorithm)
    c_star = np.array([1.0, 2.0, 0.1, 0.5])  # Operating point concentrations
    
    # For single substrate reactions: k = kr * (1 + Keq)
    # For more complex stoichiometry, use the full algorithm
    K_diag = np.array([
        kr[0] * (1 + Keq[0]),  # R1 forward
        kf[1] * (1 + 1/Keq[0]),  # R1 backward  
        kf[2]  # R2 (irreversible)
    ])
    K = np.diag(K_diag)
    
    print(f"Equilibrium constants: {Keq}")
    print(f"Relaxation matrix diagonal: {K_diag}")
    
    # Create dynamics
    dynamics = LLRQDynamics(network, Keq, K)
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(network)
    
    # Initial conditions
    initial_conditions = {
        'E': 1.0,    # Total enzyme
        'S': 2.0,    # Substrate concentration
        'ES': 0.0,   # No complex initially
        'P': 0.0     # No product initially
    }
    
    # Solve the dynamics
    solution = solver.solve(
        initial_conditions=initial_conditions,
        t_span=(0, 3),
        method='numerical'
    )
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Species concentrations
    t = solution['t']
    concentrations = solution['concentrations']
    
    axes[0, 0].plot(t, concentrations['E'], label='E', linewidth=2)
    axes[0, 0].plot(t, concentrations['S'], label='S', linewidth=2)
    axes[0, 0].plot(t, concentrations['ES'], label='ES', linewidth=2)
    axes[0, 0].plot(t, concentrations['P'], label='P', linewidth=2)
    axes[0, 0].set_title('Species Concentrations')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Concentration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Conservation laws
    E_total = concentrations['E'] + concentrations['ES']
    mass_total = concentrations['S'] + concentrations['ES'] + concentrations['P']
    
    axes[0, 1].plot(t, E_total, label='Total Enzyme', linewidth=2)
    axes[0, 1].plot(t, mass_total, label='Total S/P', linewidth=2)
    axes[0, 1].set_title('Conservation Laws')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Conserved Quantities')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reaction quotients
    Q = solution['reaction_quotients']
    
    axes[1, 0].semilogy(t, Q[:, 0], label='Q₁ (forward)', linewidth=2)
    axes[1, 0].semilogy(t, Q[:, 1], label='Q₁ (backward)', linewidth=2)
    axes[1, 0].semilogy(t, Q[:, 2], label='Q₂', linewidth=2)
    axes[1, 0].axhline(y=Keq[0], color='red', linestyle='--', alpha=0.7, label='Keq₁')
    axes[1, 0].set_title('Reaction Quotients')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Reaction Quotient')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log deviations
    log_dev = solution['log_deviations']
    
    axes[1, 1].plot(t, log_dev[:, 0], label='ln(Q₁/Keq₁)', linewidth=2)
    axes[1, 1].plot(t, log_dev[:, 1], label='ln(Q₁⁻¹/Keq₁⁻¹)', linewidth=2)
    axes[1, 1].plot(t, log_dev[:, 2], label='ln(Q₂/Keq₂)', linewidth=2)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Equilibrium')
    axes[1, 1].set_title('Log Deviations from Equilibrium')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('ln(Q/Keq)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final state
    print("\nFinal state:")
    for species in species_ids:
        print(f"  {species}: {concentrations[species][-1]:.4f}")
    
    print(f"\nFinal reaction quotients:")
    for i, rxn in enumerate(reaction_ids):
        print(f"  {rxn}: Q={Q[-1, i]:.4f} (Keq={Keq[i]:.4f})")

if __name__ == "__main__":
    enzymatic_reaction_example()
```

**Key Features:**
- Mass action to LLRQ parameter conversion
- Multi-reaction networks
- Conservation law checking
- Log-scale visualization

---

## LQR Control Example

**File:** `examples/lqr_complete_example.py`

Demonstrates optimal control of chemical reaction networks using Linear Quadratic Regulation.

```python
#!/usr/bin/env python3
"""
Complete LQR control example for chemical reaction networks.

This example shows:
1. Setting up optimal control problems
2. Designing LQR controllers
3. Comparing controlled vs uncontrolled dynamics
4. State estimation with Kalman filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver, LLRQVisualizer

class LQRController:
    """Linear Quadratic Regulator for LLRQ dynamics."""
    
    def __init__(self, dynamics, Q, R):
        """
        Initialize LQR controller.
        
        Args:
            dynamics: LLRQDynamics instance
            Q: State cost matrix
            R: Control cost matrix
        """
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        
        # System matrices: dx/dt = -Kx + Bu
        self.A = -dynamics.K
        self.B = np.eye(dynamics.n_reactions)  # Direct control of each reaction
        
        # Solve Riccati equation
        self.P = solve_continuous_are(self.A, self.B, Q, R)
        self.gain = np.linalg.inv(R) @ self.B.T @ self.P
        
    def control_law(self, x, x_target=None):
        """Compute optimal control input."""
        if x_target is None:
            x_target = np.zeros_like(x)
        error = x - x_target
        return -self.gain @ error

def lqr_control_example():
    """Demonstrate LQR control of A + B ⇌ C reaction."""
    print("LQR Control Example: A + B ⇌ C")
    print("=" * 40)
    
    # Create reaction network: A + B ⇌ C
    species_ids = ['A', 'B', 'C']
    reaction_ids = ['forward', 'backward']
    
    S = np.array([
        [-1, 1],   # A
        [-1, 1],   # B  
        [ 2, -2]   # C (2 molecules produced/consumed)
    ])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    
    # Set up dynamics
    Keq = np.array([2.0, 0.5])  # Forward and backward equilibrium constants
    K = np.array([[1.5, 0.1],   # Coupled relaxation rates
                  [0.1, 1.0]])
    
    dynamics = LLRQDynamics(network, Keq, K)
    solver = LLRQSolver(dynamics)
    
    # Design LQR controller
    Q_weight = np.diag([1.0, 1.0])  # Equal weighting on both reactions
    R_weight = np.diag([0.1, 0.1])  # Control effort penalty
    
    controller = LQRController(dynamics, Q_weight, R_weight)
    
    print(f"LQR gain matrix:\n{controller.gain}")
    
    # Initial conditions
    initial_conditions = {'A': 2.0, 'B': 1.0, 'C': 0.1}
    
    # Target: drive system to specific reaction quotient ratios
    target_x = np.array([-0.5, 0.3])  # Target log deviations
    
    # Simulate uncontrolled system
    solution_uncontrolled = solver.solve(
        initial_conditions=initial_conditions,
        t_span=np.linspace(0, 5, 100),
        method='numerical'
    )
    
    # Simulate controlled system
    def controlled_dynamics(t, x):
        # Compute control input
        u = controller.control_law(x, target_x)
        # Apply dynamics with control
        return dynamics.dynamics(t, x) + u
    
    # Initial log deviations
    c0 = np.array([initial_conditions[species] for species in species_ids])
    Q0 = network.compute_reaction_quotients(c0)
    x0 = dynamics.compute_log_deviation(Q0)
    
    # Solve controlled system
    from scipy.integrate import odeint
    t_controlled = np.linspace(0, 5, 100)
    x_controlled = odeint(controlled_dynamics, x0, t_controlled)
    
    # Convert back to concentrations (simplified for visualization)
    Q_controlled = np.array([dynamics.compute_reaction_quotients(x) for x in x_controlled])
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Concentrations comparison
    t_unc = solution_uncontrolled['t']
    conc_unc = solution_uncontrolled['concentrations']
    
    for species in species_ids:
        axes[0, 0].plot(t_unc, conc_unc[species], '-', label=f'{species} (uncontrolled)', linewidth=2)
    
    axes[0, 0].set_title('Concentrations: Uncontrolled')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Concentration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log deviations comparison
    x_unc = solution_uncontrolled['log_deviations']
    
    axes[0, 1].plot(t_unc, x_unc[:, 0], '-', label='Forward (uncontrolled)', linewidth=2)
    axes[0, 1].plot(t_unc, x_unc[:, 1], '-', label='Backward (uncontrolled)', linewidth=2)
    axes[0, 1].plot(t_controlled, x_controlled[:, 0], '--', label='Forward (controlled)', linewidth=2)
    axes[0, 1].plot(t_controlled, x_controlled[:, 1], '--', label='Backward (controlled)', linewidth=2)
    axes[0, 1].axhline(y=target_x[0], color='red', linestyle=':', alpha=0.7, label='Target')
    axes[0, 1].axhline(y=target_x[1], color='red', linestyle=':', alpha=0.7)
    axes[0, 1].set_title('Log Deviations: Control Comparison')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('ln(Q/Keq)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Control inputs
    u_controlled = np.array([controller.control_law(x, target_x) for x in x_controlled])
    
    axes[1, 0].plot(t_controlled, u_controlled[:, 0], label='Forward control', linewidth=2)
    axes[1, 0].plot(t_controlled, u_controlled[:, 1], label='Backward control', linewidth=2)
    axes[1, 0].set_title('Optimal Control Inputs')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Control u(t)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cost comparison
    def compute_cost(x_traj, u_traj):
        cost = 0
        for i in range(len(x_traj)):
            x_err = x_traj[i] - target_x
            cost += x_err.T @ Q_weight @ x_err
            if i < len(u_traj):
                cost += u_traj[i].T @ R_weight @ u_traj[i]
        return cost
    
    cost_uncontrolled = compute_cost(x_unc, np.zeros_like(x_unc))
    cost_controlled = compute_cost(x_controlled, u_controlled)
    
    axes[1, 1].bar(['Uncontrolled', 'LQR Controlled'], 
                   [cost_uncontrolled, cost_controlled],
                   color=['red', 'blue'], alpha=0.7)
    axes[1, 1].set_title('Total Cost Comparison')
    axes[1, 1].set_ylabel('Quadratic Cost')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nCost reduction: {(cost_uncontrolled - cost_controlled)/cost_uncontrolled*100:.1f}%")
    print("LQR control example completed!")

if __name__ == "__main__":
    lqr_control_example()
```

**Key Features:**
- Optimal control design
- Riccati equation solution
- Control cost analysis
- Performance comparison

---

## Glycolysis Oscillations Example

**File:** `examples/glycolysis_example.py`

Models oscillatory behavior in glycolytic pathways.

```python
#!/usr/bin/env python3
"""
Glycolysis oscillation example using LLRQ framework.

Demonstrates:
1. Multi-step metabolic pathways
2. Oscillatory dynamics
3. Metabolic control analysis
4. ATP/ADP drive effects
"""

import numpy as np
import matplotlib.pyplot as plt
from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver, LLRQVisualizer

def glycolysis_oscillation_example():
    """Simplified glycolysis model showing oscillations."""
    print("Glycolysis Oscillation Example")
    print("=" * 35)
    
    # Simplified glycolysis network
    # Glucose → G6P → F6P → FBP → DHAP/GAP → PEP → Pyruvate
    species_ids = ['Glc', 'G6P', 'F6P', 'FBP', 'GAP', 'PEP', 'Pyr', 'ATP', 'ADP']
    reaction_ids = ['HK', 'PGI', 'PFK', 'ALDO', 'GAPDH', 'PK', 'ATPase']
    
    # Simplified stoichiometry (focusing on key regulatory steps)
    S = np.array([
        [-1,  0,  0,  0,  0,  0,  0],   # Glc
        [ 1, -1,  0,  0,  0,  0,  0],   # G6P
        [ 0,  1, -1,  0,  0,  0,  0],   # F6P  
        [ 0,  0,  1, -1,  0,  0,  0],   # FBP
        [ 0,  0,  0,  2, -1,  0,  0],   # GAP
        [ 0,  0,  0,  0,  1, -1,  0],   # PEP
        [ 0,  0,  0,  0,  0,  1,  0],   # Pyr
        [-1,  0, -1,  0,  1,  1, -1],   # ATP
        [ 1,  0,  1,  0, -1, -1,  1]    # ADP
    ])
    
    network = ReactionNetwork(species_ids, reaction_ids, S)
    print("Network created with", network.n_species, "species and", network.n_reactions, "reactions")
    
    # Set kinetic parameters (based on typical values)
    # Equilibrium constants
    Keq = np.array([
        1000.0,  # HK: highly favorable
        1.0,     # PGI: reversible
        100.0,   # PFK: favorable, key control point
        0.1,     # ALDO: unfavorable but driven
        10.0,    # GAPDH: coupled to ATP synthesis
        100.0,   # PK: highly favorable
        1000.0   # ATPase: ATP consumption
    ])
    
    # Relaxation matrix (includes allosteric effects)
    K = np.diag([5.0, 10.0, 2.0, 8.0, 6.0, 15.0, 3.0])
    
    # Add coupling for key regulatory interactions
    K[2, 6] = -0.5  # PFK inhibited by ATP consumption
    K[6, 2] = 0.3   # ATPase affected by PFK activity
    
    dynamics = LLRQDynamics(network, Keq, K)
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(network)
    
    # Define ATP/ADP drive (oscillating ATP demand)
    def atp_oscillation(t):
        """Oscillating ATP demand mimicking cellular work."""
        base_consumption = 1.0
        oscillatory = 0.5 * np.sin(0.5 * t)
        drive = np.zeros(len(reaction_ids))
        drive[6] = base_consumption + oscillatory  # ATPase
        return drive
    
    dynamics.external_drive = atp_oscillation
    
    # Initial conditions (typical cellular concentrations in mM)
    initial_conditions = {
        'Glc': 5.0,   # External glucose
        'G6P': 0.5,   # G6P
        'F6P': 0.2,   # F6P
        'FBP': 0.1,   # FBP
        'GAP': 0.05,  # GAP
        'PEP': 0.02,  # PEP
        'Pyr': 0.1,   # Pyruvate
        'ATP': 5.0,   # High ATP
        'ADP': 0.5    # Low ADP
    }
    
    # Solve for oscillatory behavior
    solution = solver.solve(
        initial_conditions=initial_conditions,
        t_span=(0, 20),  # Longer time to see oscillations
        method='numerical'
    )
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(15, 10))
    
    # Metabolite concentrations
    ax1 = plt.subplot(2, 3, 1)
    t = solution['t']
    key_metabolites = ['G6P', 'F6P', 'FBP', 'PEP']
    
    for metabolite in key_metabolites:
        ax1.plot(t, solution['concentrations'][metabolite], 
                label=metabolite, linewidth=2)
    ax1.set_title('Key Metabolite Concentrations')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Concentration (mM)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ATP/ADP ratio
    ax2 = plt.subplot(2, 3, 2)
    atp = solution['concentrations']['ATP']
    adp = solution['concentrations']['ADP']
    atp_adp_ratio = atp / (adp + 1e-6)  # Avoid division by zero
    
    ax2.plot(t, atp_adp_ratio, 'r-', linewidth=2, label='ATP/ADP')
    ax2.set_title('Energy Charge (ATP/ADP Ratio)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ATP/ADP Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Reaction fluxes (approximated from reaction quotients)
    ax3 = plt.subplot(2, 3, 3)
    Q = solution['reaction_quotients']
    
    # Key regulatory fluxes
    key_reactions = [0, 2, 5, 6]  # HK, PFK, PK, ATPase
    reaction_names = ['HK', 'PFK', 'PK', 'ATPase']
    
    for i, rxn_idx in enumerate(key_reactions):
        flux_approx = np.log(Q[:, rxn_idx] / Keq[rxn_idx])  # Proportional to flux
        ax3.plot(t, flux_approx, label=reaction_names[i], linewidth=2)
    
    ax3.set_title('Key Reaction Fluxes')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Log Flux (arbitrary units)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # External drive
    ax4 = plt.subplot(2, 3, 4)
    drive_values = np.array([atp_oscillation(time) for time in t])
    ax4.plot(t, drive_values[:, 6], 'k-', linewidth=2, label='ATP Demand')
    ax4.set_title('External ATP Demand')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Drive Strength')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Phase portrait (G6P vs ATP)
    ax5 = plt.subplot(2, 3, 5)
    g6p = solution['concentrations']['G6P']
    ax5.plot(g6p, atp, 'b-', linewidth=2, alpha=0.7)
    ax5.plot(g6p[0], atp[0], 'go', markersize=8, label='Start')
    ax5.plot(g6p[-1], atp[-1], 'ro', markersize=8, label='End')
    ax5.set_title('Phase Portrait: G6P vs ATP')
    ax5.set_xlabel('G6P Concentration (mM)')
    ax5.set_ylabel('ATP Concentration (mM)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Frequency analysis
    ax6 = plt.subplot(2, 3, 6)
    from scipy.fft import fft, fftfreq
    
    # FFT of ATP concentration
    dt = t[1] - t[0]
    atp_fft = fft(atp - np.mean(atp))
    freqs = fftfreq(len(atp), dt)
    
    # Plot power spectrum (positive frequencies only)
    pos_freqs = freqs[:len(freqs)//2]
    power = np.abs(atp_fft[:len(freqs)//2])
    
    ax6.semilogy(pos_freqs, power, 'r-', linewidth=2)
    ax6.set_title('ATP Oscillation Frequency Spectrum')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Power')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print(f"\nAnalysis Results:")
    print(f"Simulation time: {t[-1]:.1f} time units")
    print(f"ATP range: {np.min(atp):.3f} - {np.max(atp):.3f} mM")
    print(f"G6P range: {np.min(g6p):.3f} - {np.max(g6p):.3f} mM")
    print(f"ATP/ADP ratio range: {np.min(atp_adp_ratio):.2f} - {np.max(atp_adp_ratio):.2f}")
    
    # Find oscillation period
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(atp, height=np.mean(atp))
    if len(peaks) > 1:
        period = np.mean(np.diff(t[peaks]))
        frequency = 1.0 / period
        print(f"Oscillation period: {period:.2f} time units")
        print(f"Oscillation frequency: {frequency:.3f} Hz")
    
    print("Glycolysis oscillation example completed!")

if __name__ == "__main__":
    glycolysis_oscillation_example()
```

**Key Features:**
- Complex metabolic networks
- Oscillatory dynamics
- Energy charge analysis
- Frequency domain analysis

---

## Running the Examples

To run any example:

1. **Navigate to the examples directory:**
   ```bash
   cd examples
   ```

2. **Run with Python:**
   ```bash
   python simple_example.py
   ```

3. **Or use the tellurium environment:**
   ```bash
   source /path/to/anaconda3/etc/profile.d/conda.sh
   conda activate tellurium
   python mass_action_example.py
   ```

## Example Output

Each example produces:
- **Console output** with analysis results
- **Multiple plots** showing dynamics
- **Numerical summaries** of key properties

## Customization

You can easily modify the examples:
- **Change parameters**: Adjust equilibrium constants, relaxation rates
- **Add species**: Extend the reaction networks
- **Modify drives**: Create custom external perturbations
- **Try different initial conditions**: Explore parameter space

## Performance Notes

- **Simple examples** run in seconds
- **Complex networks** may take minutes for long simulations
- **Use analytical methods** when possible for speed
- **Consider stiff solvers** for fast relaxation timescales