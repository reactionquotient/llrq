# src/llrq/examples/linear_vs_mass_action.py
import os, io, base64, json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from llrq.reaction_network import ReactionNetwork
from llrq.llrq_dynamics import LLRQDynamics
from llrq.solver import LLRQSolver
from llrq.control import LLRQController

# Try to import mass action simulation (optional)
try:
    from llrq.mass_action_simulator import MassActionSimulator
    HAS_MASS_ACTION = True
except ImportError:
    HAS_MASS_ACTION = False


@dataclass
class ComparisonConfig:
    T: float = 60.0
    npoints: int = 2000
    y_ref: np.ndarray = None
    controlled_reactions: tuple = ("R1", "R3")
    feedback_gain: float = 2.0
    
    # Disturbances
    impulse_time: float = 20.0
    impulse_magnitude: np.ndarray = None
    sinus_amp: float = 0.01  # Small amplitude to test disturbance consistency
    
    # Mass action parameters (if available)
    rate_constants: dict = None


def _build_3cycle_network() -> ReactionNetwork:
    """Build the same 3-cycle network as the original example."""
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2", "R3"]
    
    # Stoichiometric matrix (3 species x 3 reactions)
    S = np.array([[-1, 0, 1],   # A: -1 in R1, 0 in R2, +1 in R3
                  [1, -1, 0],   # B: +1 in R1, -1 in R2, 0 in R3
                  [0, 1, -1]])  # C: 0 in R1, +1 in R2, -1 in R3
    
    # Species info with initial concentrations
    species_info = {
        "A": {"name": "A", "initial_concentration": 2.0, "compartment": "cell", "boundary_condition": False},
        "B": {"name": "B", "initial_concentration": 0.2, "compartment": "cell", "boundary_condition": False},
        "C": {"name": "C", "initial_concentration": 0.1, "compartment": "cell", "boundary_condition": False}
    }
    
    # Reaction info
    reaction_info = [
        {"id": "R1", "name": "A ⇌ B", "reactants": [("A", 1.0)], "products": [("B", 1.0)], "reversible": True},
        {"id": "R2", "name": "B ⇌ C", "reactants": [("B", 1.0)], "products": [("C", 1.0)], "reversible": True},
        {"id": "R3", "name": "C ⇌ A", "reactants": [("C", 1.0)], "products": [("A", 1.0)], "reversible": True}
    ]
    
    return ReactionNetwork(species_ids, reaction_ids, S, species_info, reaction_info)


def simulate_linear_dynamics(network, controller, cfg, t_eval):
    """Simulate using LLRQ linear approximation."""
    dt = t_eval[1] - t_eval[0]
    n = len(t_eval)
    
    # Get system matrices
    A = controller.A
    B_red = controller.B_red
    
    # Initial state
    c0 = controller.solver._parse_initial_dict({
        sid: info.get("initial_concentration", 0.0)
        for sid, info in network.species_info.items()
    })
    Q0 = network.compute_reaction_quotients(c0)
    y = controller.reaction_quotients_to_reduced_state(Q0)
    
    # Target
    if cfg.y_ref is None:
        cfg.y_ref = np.array([0.1, -0.05])
    
    # Storage
    Y = np.zeros((n, len(y)))
    U = np.zeros((n, len(controller.controlled_reactions)))
    Q_traj = np.zeros((n, len(network.reaction_ids)))
    
    # Disturbance function
    def disturbance(t):
        sinus = cfg.sinus_amp * np.array([np.sin(0.8*t), 0.3*np.sin(1.1*t)])[:len(y)]
        return sinus
    
    # Compute constant steady-state control (feedforward only)
    Q_target = controller.reduced_state_to_reaction_quotients(cfg.y_ref)
    u_const = controller.compute_steady_state_control(cfg.y_ref)
    
    # Simulate
    for i, t in enumerate(t_eval):
        # Use constant control (no feedback)
        u = u_const
        
        # Dynamics with disturbance
        u_red = B_red @ u
        d = disturbance(t)
        ydot = A @ y + u_red + d
        
        # Integrate
        y = y + dt * ydot
        
        # Log
        Y[i] = y
        U[i] = u
        Q_traj[i] = controller.reduced_state_to_reaction_quotients(y)
    
    # Compute concentrations
    C_traj = np.zeros((n, len(network.species_ids)))
    for i in range(n):
        C_traj[i] = controller.solver._compute_concentrations_from_reduced(
            Q_traj[i:i+1], c0, True
        )
    
    return {
        'time': t_eval,
        'y': Y,
        'u': U,
        'Q': Q_traj,
        'C': C_traj,
        'method': 'Linear LLRQ'
    }


def simulate_mass_action_dynamics(network, controller, cfg, t_eval):
    """Simulate using true mass action kinetics with corrected LLRQ control."""
    if not HAS_MASS_ACTION:
        return None
        
    # Use the same consistent rate constants as the linear system
    rate_constants_ma = {
        'R1': (3.0, 1.5),  # Keq = 2.0
        'R2': (1.0, 2.0),  # Keq = 0.5
        'R3': (3.0, 3.0)   # Keq = 1.0
    }
    
    # Create simulator with LLRQ control capability
    sim = MassActionSimulator(network, rate_constants_ma, 
                             B=controller.B, K_red=controller.K_red,
                             lnKeq_consistent=controller.solver._lnKeq_consistent)
    
    # Control function with constant feedforward only (same as linear)
    def control_function(t, Q_current):
        if cfg.y_ref is None:
            cfg.y_ref = np.array([0.1, -0.05])
        
        # Use same constant steady-state control as linear simulation
        u_total = controller.compute_steady_state_control(cfg.y_ref)
        
        # Convert full control to reduced control
        # The controller returns control for selected reactions, we need to convert to u_red
        G = controller.G  # Selection matrix (r x m)
        B = controller.B  # Basis matrix (r x rankS)
        
        # Full control vector (all reactions)
        u_full = np.zeros(len(network.reaction_ids))
        for i, idx in enumerate(controller.controlled_indices):
            u_full[idx] = u_total[i]
        
        # Convert to reduced space: u_red = B^T @ u_full
        u_red = B.T @ u_full
        
        # Return both u_red and u_total for proper tracking
        return u_red, u_total
    
    # Define disturbance functions
    def disturbance_function(t):
        # Sinusoidal disturbance (same as linear simulation)
        sinus = cfg.sinus_amp * np.array([np.sin(0.8*t), 0.3*np.sin(1.1*t)])[:controller.rankS]
        return sinus
    
    # Simulate with LLRQ control and proper state disturbances
    result = sim.simulate(
        t_eval, 
        control_function,
        disturbance_function=disturbance_function
    )
    result['method'] = 'Mass Action (Corrected)'
    
    # Add reduced state trajectory for comparison
    n = len(t_eval)
    Y = np.zeros((n, controller.rankS))
    for i in range(n):
        Y[i] = controller.reaction_quotients_to_reduced_state(result['reaction_quotients'][i])
    result['y'] = Y
    
    # Use proper control for plotting (u_total for controlled reactions)
    if 'u_total' in result:
        result['u'] = result['u_total']
    elif 'u_red' in result:
        # Fallback to u_red if u_total not available
        result['u'] = result['u_red']
    
    return result


def _png(fig) -> str:
    """Return a data: URI PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_and_run_comparison(out_dir: str = "llrq_linear_vs_mass_action"):
    """Compare linear LLRQ vs mass action simulation."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Build system with thermodynamically consistent parameters
    network = _build_3cycle_network()
    
    # Use thermodynamically consistent Keq values (must satisfy Keq1 × Keq2 × Keq3 = 1)
    Keq = np.array([2.0, 0.5, 1.0])  # 2.0 × 0.5 × 1.0 = 1.0 ✓
    
    # Define consistent rate constants
    rate_constants = {
        'R1': (3.0, 1.5),  # Keq = 2.0
        'R2': (1.0, 2.0),  # Keq = 0.5
        'R3': (3.0, 3.0)   # Keq = 1.0
    }
    
    # Proper K matrix (diagonal approximation: K[i,i] = kf[i] + kr[i])
    kf = np.array([rate_constants[rid][0] for rid in network.reaction_ids])
    kr = np.array([rate_constants[rid][1] for rid in network.reaction_ids])
    K = np.diag(kf + kr)
    
    dynamics = LLRQDynamics(network=network, equilibrium_constants=Keq, relaxation_matrix=K)
    solver = LLRQSolver(dynamics)
    
    # Controller
    controller = LLRQController(solver, controlled_reactions=["R1", "R3"])
    
    # Configuration
    cfg = ComparisonConfig()
    cfg.y_ref = np.array([0.1, -0.05])
    # cfg.impulse_magnitude = np.array([0.2, -0.1])  # Removed impulse disturbance
    
    # Time points
    t_eval = np.linspace(0, cfg.T, cfg.npoints)
    
    # Pass rate constants to config for consistency
    cfg.rate_constants = rate_constants
    
    # Run both simulations
    print("Running linear LLRQ simulation...")
    linear_result = simulate_linear_dynamics(network, controller, cfg, t_eval)
    
    mass_action_result = None
    if HAS_MASS_ACTION:
        print("Running mass action simulation...")
        mass_action_result = simulate_mass_action_dynamics(network, controller, cfg, t_eval)
    else:
        print("Mass action simulation not available")
    
    # Add concentration debugging
    c0 = np.array([2.0, 0.2, 0.1])
    Q_target = controller.reduced_state_to_reaction_quotients(cfg.y_ref)
    c_target = controller.compute_target_concentrations(Q_target, c0)
    
    print("\n=== CONCENTRATION ANALYSIS ===")
    print(f"Initial concentrations: {c0}")
    print(f"Target Q: {Q_target}")  
    print(f"Target concentrations: {c_target}")
    print(f"Target raw: {repr(c_target)}")
    print(f"Target type: {type(c_target)}")
    if hasattr(c_target, 'shape'):
        print(f"Target shape: {c_target.shape}")
        if len(c_target.shape) > 0:
            print(f"Target total mass: {np.sum(c_target):.6f}")
    else:
        print(f"Target is scalar: {c_target}")
    
    print(f"\nLinear simulation:")
    print(f"  C shape: {linear_result['C'].shape}")
    print(f"  Initial concentrations: {linear_result['C'][0]}")
    print(f"  Initial raw: {repr(linear_result['C'][0])}")
    print(f"  Initial individual: [{linear_result['C'][0][0]:.6f}, {linear_result['C'][0][1]:.6f}, {linear_result['C'][0][2]:.6f}]")
    print(f"  Final concentrations: {linear_result['C'][-1]}")
    print(f"  Final individual: [{linear_result['C'][-1][0]:.6f}, {linear_result['C'][-1][1]:.6f}, {linear_result['C'][-1][2]:.6f}]")
    print(f"  Final total mass: {np.sum(linear_result['C'][-1]):.6f}")
    
    if mass_action_result:
        print(f"\nMass action simulation:")
        print(f"  Initial concentrations: {mass_action_result['concentrations'][0]}")  
        print(f"  Final concentrations: {mass_action_result['concentrations'][-1]}")
        print(f"  Final total mass: {np.sum(mass_action_result['concentrations'][-1]):.6f}")
        
        # Concentration differences
        c_diff = mass_action_result['concentrations'][-1] - linear_result['C'][-1]
        print(f"  Concentration differences: {c_diff}")
        print(f"  Max absolute difference: {np.max(np.abs(c_diff)):.6f}")
        
        # Disturbance effect analysis
        c_target_array = np.array([c_target, c_target, c_target])  # Target for all time points
        linear_deviations = np.linalg.norm(linear_result['y'] - cfg.y_ref, axis=1)
        mass_deviations = np.linalg.norm(mass_action_result['y'] - cfg.y_ref, axis=1)
        
        print(f"\n=== DISTURBANCE EFFECT ANALYSIS ===")
        print(f"RMS deviation from target (linear): {np.sqrt(np.mean(linear_deviations**2)):.6f}")
        print(f"RMS deviation from target (mass action): {np.sqrt(np.mean(mass_deviations**2)):.6f}")
        print(f"Max deviation from target (linear): {np.max(linear_deviations):.6f}")
        print(f"Max deviation from target (mass action): {np.max(mass_deviations):.6f}")
    
    # Create plots
    figs = {}
    
    # Reduced state comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # y1 component
    ax1.plot(t_eval, linear_result['y'][:, 0], 'b-', linewidth=2, label='Linear LLRQ')
    if mass_action_result:
        ax1.plot(t_eval, mass_action_result['y'][:, 0], 'r--', linewidth=2, label='Mass Action')
    ax1.axhline(cfg.y_ref[0], color='black', linestyle=':', alpha=0.7, label='Target')
    # ax1.axvline(cfg.impulse_time, color='red', linestyle=':', alpha=0.5, label='Impulse')  # Removed impulse
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('y₁')
    ax1.set_title('Reduced State Component 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # y2 component  
    ax2.plot(t_eval, linear_result['y'][:, 1], 'b-', linewidth=2, label='Linear LLRQ')
    if mass_action_result:
        ax2.plot(t_eval, mass_action_result['y'][:, 1], 'r--', linewidth=2, label='Mass Action')
    ax2.axhline(cfg.y_ref[1], color='black', linestyle=':', alpha=0.7, label='Target')
    # ax2.axvline(cfg.impulse_time, color='red', linestyle=':', alpha=0.5, label='Impulse')  # Removed impulse
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('y₂')  
    ax2.set_title('Reduced State Component 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figs["reduced_state.png"] = _png(fig)
    
    # Concentration comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    species_names = ['A', 'B', 'C']
    colors = ['blue', 'green', 'orange']
    
    for i, (species, color) in enumerate(zip(species_names, colors)):
        ax = axes[i]
        ax.plot(t_eval, linear_result['C'][:, i], '-', color=color, linewidth=2, 
                label=f'Linear LLRQ')
        if mass_action_result:
            ax.plot(t_eval, mass_action_result['concentrations'][:, i], '--', 
                   color=color, linewidth=2, label=f'Mass Action')
        # ax.axvline(cfg.impulse_time, color='red', linestyle=':', alpha=0.5, 
        #           label='Impulse' if i == 0 else '')  # Removed impulse
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'[{species}]')
        ax.set_title(f'Species {species} Concentration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figs["concentrations.png"] = _png(fig)
    
    # Control signals comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    controlled_reactions = ["R1: A⇌B", "R3: C⇌A"]
    for i, rxn_name in enumerate(controlled_reactions):
        ax = ax1 if i == 0 else ax2
        ax.plot(t_eval, linear_result['u'][:, i], 'b-', linewidth=2, label='Linear LLRQ')
        if mass_action_result and 'u' in mass_action_result:
            ax.plot(t_eval, mass_action_result['u'][:, i], 'r--', linewidth=2, label='Mass Action')
        # ax.axvline(cfg.impulse_time, color='red', linestyle=':', alpha=0.5, label='Impulse')  # Removed impulse
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Input')
        ax.set_title(f'Control Signal: {rxn_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figs["control_signals.png"] = _png(fig)
    
    # Summary statistics
    linear_error = np.linalg.norm(linear_result['y'] - cfg.y_ref, axis=1)
    stats = {
        "linear_rms_error": float(np.sqrt(np.mean(linear_error**2))),
        "linear_final_error": float(linear_error[-1]),
        "has_mass_action": HAS_MASS_ACTION,
    }
    
    if mass_action_result:
        mass_action_error = np.linalg.norm(mass_action_result['y'] - cfg.y_ref, axis=1)
        stats.update({
            "mass_action_rms_error": float(np.sqrt(np.mean(mass_action_error**2))),
            "mass_action_final_error": float(mass_action_error[-1]),
            "max_difference": float(np.max(np.abs(linear_result['y'] - mass_action_result['y']))),
        })
    
    # HTML report
    mass_action_status = "Available" if HAS_MASS_ACTION else "Not Available (install roadrunner/tellurium)"
    
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LLRQ: Linear vs Mass Action Comparison</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0.3em 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin: 10px 0; }}
    img {{ max-width: 100%; height: auto; }}
    .status {{ padding: 8px; border-radius: 4px; }}
    .available {{ background: #d4edda; color: #155724; }}
    .unavailable {{ background: #f8d7da; color: #721c24; }}
  </style>
</head>
<body>
  <h1>LLRQ: Linear Approximation vs True Mass Action Dynamics</h1>
  
  <h2>What This Comparison Shows</h2>
  <div class="card">
    <p><b>Key Question:</b> How accurate is the LLRQ linear approximation compared to true mass action kinetics?</p>
    <ul>
      <li><b>Linear LLRQ:</b> Uses the approximation d/dt ln(Q) = -K(ln(Q) - ln(Keq)) + u</li>
      <li><b>Mass Action:</b> Simulates true nonlinear kinetics with roadrunner/tellurium</li>
      <li><b>Same Controller:</b> LLRQ control strategy applied to both systems</li>
      <li><b>Test:</b> Control performance and accuracy of linearization</li>
    </ul>
    <p><b>Mass Action Status:</b> <span class="status {'available' if HAS_MASS_ACTION else 'unavailable'}">{mass_action_status}</span></p>
  </div>

  <h2>Performance Summary</h2>
  <pre>{json.dumps(stats, indent=2)}</pre>

  <h2>1. Reduced State Comparison</h2>
  <div class="card">
    <p>The mathematical coordinates used for control. Should be identical if linearization is perfect.</p>
    <img src="{figs['reduced_state.png']}" />
  </div>

  <h2>2. Physical Concentrations</h2>
  <div class="card">
    <p>The actual species concentrations - what we really care about.</p>
    <img src="{figs['concentrations.png']}" />
  </div>

  <h2>3. Control Effort</h2>  
  <div class="card">
    <p>Control signals required to achieve the same performance.</p>
    <img src="{figs['control_signals.png']}" />
  </div>

  <h2>Key Insights</h2>
  <div class="card">
    {'<p><b>Mass action simulation available:</b> Compare the plots to see when and where the LLRQ linearization breaks down. Large differences indicate nonlinear effects that LLRQ cannot capture.</p>' if HAS_MASS_ACTION else '<p><b>Mass action simulation not available:</b> Install roadrunner or tellurium to enable comparison with true kinetic dynamics.</p>'}
    
    <p><b>Control robustness:</b> LLRQ control strategies can work even when applied to nonlinear systems, demonstrating the practical value of the approach.</p>
    
    <p><b>Validation framework:</b> This comparison provides a way to validate when LLRQ approximations are accurate enough for your application.</p>
  </div>

  <p style="color:#888;margin-top:24px">Generated by llrq.examples.linear_vs_mass_action</p>
</body>
</html>
"""
    
    out_html = os.path.join(out_dir, "comparison_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    
    return out_html


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    output_dir = "llrq_linear_vs_mass_action"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"Running LLRQ Linear vs Mass Action Comparison")
    print(f"Output directory: {output_dir}")
    
    # Run the comparison
    report_path = build_and_run_comparison(output_dir)
    
    print(f"\nComparison completed!")
    print(f"Report generated at: {report_path}")
    print(f"\nTo view the report, open: {os.path.abspath(report_path)}")