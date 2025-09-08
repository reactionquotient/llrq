---
layout: page
title: Tutorials
nav_order: 4
---

# Tutorials

Step-by-step guides to using the LLRQ package for common chemical reaction analysis tasks.

## Tutorial 1: Simple A ⇌ B Reaction

Learn the basics with a single reversible reaction.

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt
import llrq
```

### Step 1: Create the Reaction System

```python
# Create a simple A ⇌ B reaction
network, dynamics, solver, visualizer = llrq.simple_reaction(
    reactant_species="A",
    product_species="B",
    equilibrium_constant=2.0,      # At equilibrium: [B]/[A] = 2.0
    relaxation_rate=1.0,           # How fast system approaches equilibrium
    initial_concentrations={"A": 1.0, "B": 0.1}  # Starting concentrations
)
```

### Step 2: Examine the Network

```python
# Print network information
print("Network Summary:")
print(network.summary())

# Access key properties
print(f"Number of species: {network.n_species}")
print(f"Species names: {network.species_names}")
print(f"Stoichiometric matrix:\n{network.S}")
```

**Expected Output:**
```
Network Summary:
ReactionNetwork: 2 species, 1 reaction
Species: ['A', 'B']
Reactions: ['R1']
Stoichiometric matrix (2x1):
[[-1.]
 [ 1.]]
```

### Step 3: Solve the Dynamics

```python
# Solve using analytical method (exact solution)
solution = solver.solve(
    initial_conditions={"A": 1.0, "B": 0.1},
    t_span=(0, 5),  # Solve from t=0 to t=5
    method='analytical'
)

print(f"Solution successful: {solution['success']}")
print(f"Method used: {solution['method']}")
```

### Step 4: Analyze Results

```python
# Extract time points and concentrations
t = solution['t']
c_A = solution['concentrations']['A']
c_B = solution['concentrations']['B']
Q = solution['reaction_quotients'][0]  # Q = [B]/[A]

# Print equilibrium values
print(f"Final concentrations: A={c_A[-1]:.3f}, B={c_B[-1]:.3f}")
print(f"Final reaction quotient: Q={Q[-1]:.3f} (expected: 2.0)")
```

### Step 5: Visualize

```python
# Create publication-quality plots
fig = visualizer.plot_dynamics(solution)
plt.show()

# Or create custom plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot concentrations
ax1.plot(t, c_A, label='A', linewidth=2, color='blue')
ax1.plot(t, c_B, label='B', linewidth=2, color='red')
ax1.set_xlabel('Time')
ax1.set_ylabel('Concentration')
ax1.set_title('Species Concentrations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot reaction quotient
ax2.plot(t, Q, linewidth=2, color='green')
ax2.axhline(y=2.0, color='red', linestyle='--', label='K_eq = 2.0')
ax2.set_xlabel('Time')
ax2.set_ylabel('Reaction Quotient Q')
ax2.set_title('Approach to Equilibrium')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Tutorial 2: Multi-Reaction Networks

Learn to handle complex reaction networks with multiple reactions.

### Setup: Enzymatic Reaction

We'll model a simple enzyme kinetics system:
```
E + S ⇌ ES → E + P
```

### Step 1: Manual Network Construction

```python
# Define the network manually
species_ids = ['E', 'S', 'ES', 'P']
reaction_ids = ['R1_forward', 'R1_backward', 'R2']

# Stoichiometric matrix (species × reactions)
S = np.array([
    [-1,  1,  1],   # E: consumed in R1f, produced in R1b and R2
    [-1,  1,  0],   # S: consumed in R1f, produced in R1b
    [ 1, -1, -1],   # ES: produced in R1f, consumed in R1b and R2
    [ 0,  0,  1]    # P: produced in R2
])

# Create network
network = ReactionNetwork(species_ids, reaction_ids, S)
print("Network summary:")
print(network.summary())
```

### Step 2: Set Up Dynamics

```python
# Set equilibrium constants
# R1: E + S ⇌ ES has Keq1 = [ES]/([E][S])
# R2: ES → E + P is irreversible, set Keq2 large
Keq = np.array([0.1, 10.0, 1000.0])  # Forward, backward, irreversible

# Set relaxation matrix (diagonal for independent reactions)
K = np.diag([2.0, 2.0, 5.0])  # Fast irreversible step

# Create dynamics
dynamics = LLRQDynamics(network, Keq, K)
solver = LLRQSolver(dynamics)
visualizer = LLRQVisualizer(network)
```

### Step 3: Solve with Initial Conditions

```python
# Set initial concentrations
initial_conditions = {
    'E': 1.0,    # Enzyme concentration
    'S': 2.0,    # Substrate concentration
    'ES': 0.0,   # Start with no complex
    'P': 0.0     # No product initially
}

# Solve the system
solution = solver.solve(
    initial_conditions=initial_conditions,
    t_span=(0, 3),
    method='numerical'  # Use numerical for complex networks
)
```

### Step 4: Analyze Enzyme Kinetics

```python
# Extract results
t = solution['t']
concentrations = solution['concentrations']

# Plot all species
plt.figure(figsize=(10, 6))
for species in species_ids:
    plt.plot(t, concentrations[species], label=species, linewidth=2)

plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Enzyme Kinetics: E + S ⇌ ES → E + P')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check conservation laws
E_total = concentrations['E'] + concentrations['ES']
plt.figure(figsize=(8, 4))
plt.plot(t, E_total, label='Total Enzyme', linewidth=2)
plt.axhline(y=E_total[0], color='red', linestyle='--', label='Initial')
plt.xlabel('Time')
plt.ylabel('Total Enzyme Concentration')
plt.title('Conservation Law: E_total = [E] + [ES]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Tutorial 3: External Drives and Control

Learn to apply external drives and control chemical networks.

### Step 1: Time-Varying Drives

```python
# Start with simple A ⇌ B reaction
network, dynamics, solver, visualizer = llrq.simple_reaction(
    reactant_species="A", product_species="B",
    equilibrium_constant=2.0, relaxation_rate=1.0,
    initial_concentrations={"A": 1.0, "B": 0.1}
)

# Define external drive functions
def step_drive(t):
    """Step function drive."""
    return np.array([0.5 if t > 2 else 0.0])

def oscillating_drive(t):
    """Sinusoidal drive."""
    return np.array([0.3 * np.sin(2*np.pi*t)])

def ramp_drive(t):
    """Linear ramp drive."""
    return np.array([0.1 * t])
```

### Step 2: Compare Different Drives

```python
# Test each drive
drives = [
    (lambda t: np.array([0.0]), "No drive"),
    (step_drive, "Step drive"),
    (oscillating_drive, "Oscillating drive"),
    (ramp_drive, "Ramp drive")
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for i, (drive_func, label) in enumerate(drives):
    # Set the drive
    dynamics.external_drive = drive_func

    # Solve
    solution = solver.solve(
        initial_conditions={"A": 1.0, "B": 0.1},
        t_span=(0, 5),
        method='numerical'
    )

    # Plot
    t = solution['t']
    axes[i].plot(t, solution['concentrations']['A'], label='A', linewidth=2)
    axes[i].plot(t, solution['concentrations']['B'], label='B', linewidth=2)
    axes[i].set_title(label)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Concentration')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 3: LQR Optimal Control

```python
# Import control module
from llrq.control import LQRController

# Set up LQR controller
Q_weight = np.array([[1.0]])  # State cost
R_weight = np.array([[0.1]])  # Control cost
controller = LQRController(dynamics, Q_weight, R_weight)

# Set target state (equilibrium)
target_log_deviation = np.array([0.0])  # ln(Q/Keq) = 0 at equilibrium

# Solve with LQR control
controlled_solution = controller.solve_lqr(
    initial_conditions={"A": 1.0, "B": 0.1},
    target_state=target_log_deviation,
    t_span=(0, 5)
)

# Compare controlled vs uncontrolled
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Uncontrolled
dynamics.external_drive = lambda t: np.array([0.0])
uncontrolled = solver.solve(
    initial_conditions={"A": 1.0, "B": 0.1},
    t_span=(0, 5)
)

t = uncontrolled['t']
ax1.plot(t, uncontrolled['concentrations']['A'], label='A (uncontrolled)')
ax1.plot(t, uncontrolled['concentrations']['B'], label='B (uncontrolled)')
ax1.plot(controlled_solution['t'], controlled_solution['concentrations']['A'],
         '--', label='A (controlled)')
ax1.plot(controlled_solution['t'], controlled_solution['concentrations']['B'],
         '--', label='B (controlled)')
ax1.set_title('Concentration Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Control input
ax2.plot(controlled_solution['t'], controlled_solution['control_input'],
         linewidth=2, color='red')
ax2.set_title('Optimal Control Input')
ax2.set_xlabel('Time')
ax2.set_ylabel('Control u(t)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Tutorial 4: SBML Model Import

Learn to work with standard biological models.

### Step 1: Load SBML Model

```python
# Load an SBML model (you need an actual SBML file)
try:
    network, dynamics, solver, visualizer = llrq.from_sbml(
        sbml_file='examples/glycolysis.xml',  # Your SBML file
        equilibrium_constants=None,  # Will use defaults
        relaxation_matrix=None       # Will use identity matrix
    )

    print("SBML model loaded successfully!")
    print(network.summary())

except FileNotFoundError:
    print("SBML file not found. Creating synthetic example...")

    # Create a simple synthetic network instead
    species_ids = ['Glucose', 'G6P', 'F6P', 'ATP', 'ADP']
    reaction_ids = ['Hexokinase', 'PGI']

    S = np.array([
        [-1,  0],   # Glucose
        [ 1, -1],   # G6P
        [ 0,  1],   # F6P
        [-1,  0],   # ATP
        [ 1,  0]    # ADP
    ])

    network = ReactionNetwork(species_ids, reaction_ids, S)
    dynamics = LLRQDynamics(network)
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(network)
```

### Step 2: Set Biological Parameters

```python
# Set realistic metabolic parameters
# Equilibrium constants from literature
Keq = np.array([1000.0, 1.0])  # Hexokinase strongly favored, PGI reversible

# Relaxation rates based on enzyme concentrations
K = np.diag([10.0, 5.0])  # Fast hexokinase, moderate PGI

# Update dynamics
dynamics.Keq = Keq
dynamics.K = K
```

### Step 3: Simulate Metabolic Conditions

```python
# Normal conditions
initial_conditions = {
    'Glucose': 5.0,  # 5 mM glucose
    'G6P': 0.1,      # Low G6P
    'F6P': 0.05,     # Low F6P
    'ATP': 5.0,      # High ATP
    'ADP': 0.5       # Low ADP
}

# Solve under normal conditions
normal_solution = solver.solve(
    initial_conditions=initial_conditions,
    t_span=(0, 2)
)

# Simulate ATP depletion
def atp_depletion(t):
    """Simulate ATP consumption."""
    return np.array([-2.0, 0.0])  # Drain ATP via hexokinase

dynamics.external_drive = atp_depletion
depleted_solution = solver.solve(
    initial_conditions=initial_conditions,
    t_span=(0, 2)
)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Normal conditions
t1 = normal_solution['t']
for species in ['Glucose', 'G6P', 'F6P']:
    axes[0].plot(t1, normal_solution['concentrations'][species],
                 label=species, linewidth=2)
axes[0].set_title('Normal Conditions')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Concentration (mM)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ATP depletion
t2 = depleted_solution['t']
for species in ['Glucose', 'G6P', 'F6P']:
    axes[1].plot(t2, depleted_solution['concentrations'][species],
                 '--', label=species, linewidth=2)
axes[1].set_title('ATP Depletion')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Concentration (mM)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Tutorial 5: Control Optimization with CVXPY

Learn to design optimal control strategies using convex optimization.

### Setup

```python
import llrq
import numpy as np
import matplotlib.pyplot as plt
from llrq import CVXController, CVXObjectives, CVXConstraints
```

### Step 1: Create Controlled System

```python
# Create a two-reaction system
network = llrq.ReactionNetwork()
network.add_reaction_from_string("R1", "A <-> B", k_forward=2.0, k_backward=1.0)
network.add_reaction_from_string("R2", "B <-> C", k_forward=1.5, k_backward=0.8)

dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create CVX controller controlling first reaction only
controller = CVXController(solver, controlled_reactions=["R1"])
```

### Step 2: Sparse Control Design

Design control that minimizes both tracking error and control sparsity:

```python
# Define sparse control objective
objective = CVXObjectives.sparse_control(
    sparsity_weight=0.1,    # Penalty on control magnitude (L1 norm)
    tracking_weight=1.0     # Weight on tracking error
)

# Add box constraints
constraints = CVXConstraints.box_bounds(u_min=-2.0, u_max=2.0)

# Define target reaction forces
x_target = np.array([0.5, -0.3])  # Target state

# Solve optimization problem
result = controller.compute_cvx_control(
    objective_fn=objective,
    constraints_fn=constraints,
    x_target=x_target
)

print(f"Optimization status: {result['status']}")
print(f"Optimal control: {result['u_optimal']}")
print(f"Objective value: {result['objective_value']:.4f}")
```

### Step 3: Multi-Objective Control

Balance multiple competing objectives:

```python
# Multi-objective with custom weights
multi_obj = CVXObjectives.multi_objective({
    'tracking': 10.0,    # High priority on reaching target
    'control': 1.0,      # Control effort penalty (L2)
    'sparsity': 0.5      # Sparsity bonus (L1)
})

# Multiple constraints
constraints = CVXConstraints.combine(
    CVXConstraints.box_bounds(u_min=-1.5, u_max=1.5),
    CVXConstraints.control_budget(total_budget=3.0, norm_type=1)  # L1 budget
)

result_multi = controller.compute_cvx_control(
    objective_fn=multi_obj,
    constraints_fn=constraints,
    x_target=x_target
)

print(f"Multi-objective control: {result_multi['u_optimal']}")
```

### Step 4: Simulate with Optimal Control

Test the designed control through simulation:

```python
# Create control functions
def sparse_control(t):
    return result['u_optimal']

def multi_control(t):
    return result_multi['u_optimal']

# Simulate both approaches
t_sim = np.linspace(0, 10, 200)
sol_sparse = solver.solve(t_sim, u=sparse_control)
sol_multi = solver.solve(t_sim, u=multi_control)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Reaction forces
axes[0,0].plot(t_sim, sol_sparse.x[:, 0], label='x₁ (sparse)', linewidth=2)
axes[0,0].plot(t_sim, sol_sparse.x[:, 1], label='x₂ (sparse)', linewidth=2)
axes[0,0].axhline(x_target[0], color='red', linestyle='--', alpha=0.7, label='target x₁')
axes[0,0].axhline(x_target[1], color='blue', linestyle='--', alpha=0.7, label='target x₂')
axes[0,0].set_title('Sparse Control - State Trajectory')
axes[0,0].set_ylabel('Reaction Forces')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(t_sim, sol_multi.x[:, 0], label='x₁ (multi)', linewidth=2, linestyle='--')
axes[0,1].plot(t_sim, sol_multi.x[:, 1], label='x₂ (multi)', linewidth=2, linestyle='--')
axes[0,1].axhline(x_target[0], color='red', linestyle='--', alpha=0.7, label='target x₁')
axes[0,1].axhline(x_target[1], color='blue', linestyle='--', alpha=0.7, label='target x₂')
axes[0,1].set_title('Multi-Objective Control - State Trajectory')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Control signals (constant in this case)
axes[1,0].axhline(result['u_optimal'][0], color='green', linewidth=3, label='u₁ (sparse)')
axes[1,0].axhline(result['u_optimal'][1], color='orange', linewidth=3, label='u₂ (sparse)')
axes[1,0].set_title('Sparse Control Inputs')
axes[1,0].set_ylabel('Control')
axes[1,0].set_xlabel('Time')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].axhline(result_multi['u_optimal'][0], color='green', linewidth=3, linestyle='--', label='u₁ (multi)')
axes[1,1].axhline(result_multi['u_optimal'][1], color='orange', linewidth=3, linestyle='--', label='u₂ (multi)')
axes[1,1].set_title('Multi-Objective Control Inputs')
axes[1,1].set_xlabel('Time')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare final tracking errors
sparse_error = np.linalg.norm(sol_sparse.x[-1] - x_target)
multi_error = np.linalg.norm(sol_multi.x[-1] - x_target)

print(f"Final tracking errors:")
print(f"  Sparse control: {sparse_error:.4f}")
print(f"  Multi-objective: {multi_error:.4f}")
```

---

## Tutorial 6: Frequency-Domain Control Design

Design sinusoidal control inputs for periodic steady states.

### Setup

```python
import llrq
import numpy as np
import matplotlib.pyplot as plt
from llrq import FrequencySpaceController
```

### Step 1: Create Frequency Controller

```python
# Use the same two-reaction system from Tutorial 5
network = llrq.ReactionNetwork()
network.add_reaction_from_string("R1", "A <-> B", k_forward=2.0, k_backward=1.0)
network.add_reaction_from_string("R2", "B <-> C", k_forward=1.5, k_backward=0.8)

dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create frequency controller (control first reaction only)
freq_controller = FrequencySpaceController.from_llrq_solver(
    solver, controlled_reactions=["R1"]
)
```

### Step 2: Design Sinusoidal Control

```python
# Design 1 Hz oscillation with specific amplitude and phase
omega = 2 * np.pi * 1.0  # 1 Hz in rad/s

# Target complex amplitude (magnitude=1.0, phase=45°)
X_target = np.array([1.0, -0.5]) * np.exp(1j * np.pi/4)

# Design optimal sinusoidal control
U_optimal = freq_controller.design_sinusoidal_control(
    X_target=X_target,
    omega=omega,
    lam=0.01  # Regularization parameter
)

print(f"Optimal control amplitude: {U_optimal}")
print(f"Control magnitude: {np.abs(U_optimal)}")
print(f"Control phase: {np.angle(U_optimal) * 180 / np.pi} degrees")
```

### Step 3: Frequency Response Analysis

```python
# Compute frequency response for range of frequencies
omega_range = np.logspace(-2, 2, 100)  # 0.01 to 100 rad/s
magnitude_responses = []
phase_responses = []

for w in omega_range:
    H = freq_controller.compute_frequency_response(w)
    # Look at first input to first output
    magnitude_responses.append(np.abs(H[0, 0]))
    phase_responses.append(np.angle(H[0, 0]) * 180 / np.pi)

# Plot Bode diagram
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.semilogx(omega_range, 20 * np.log10(magnitude_responses))
ax1.axvline(omega, color='red', linestyle='--', alpha=0.7, label=f'Design ω = {omega:.2f}')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Frequency Response - Bode Plot')
ax1.grid(True)
ax1.legend()

ax2.semilogx(omega_range, phase_responses)
ax2.axvline(omega, color='red', linestyle='--', alpha=0.7, label=f'Design ω = {omega:.2f}')
ax2.set_xlabel('Frequency (rad/s)')
ax2.set_ylabel('Phase (degrees)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
```

### Step 4: Time-Domain Simulation

```python
# Convert complex control to time-domain function
def sinusoidal_control(t):
    return np.real(U_optimal * np.exp(1j * omega * t))

# Simulate the system
t_sim = np.linspace(0, 6, 300)  # 6 seconds (6 periods at 1 Hz)
sol = solver.solve(t_sim, u=sinusoidal_control)

# Analyze steady-state (last 2 periods)
steady_start = int(2*len(t_sim)/3)  # Start of steady state
t_steady = t_sim[steady_start:]
x_steady = sol.x[steady_start:]
u_steady = np.array([sinusoidal_control(t) for t in t_steady])

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Full trajectory
axes[0].plot(t_sim, sol.x[:, 0], label='x₁', linewidth=2)
axes[0].plot(t_sim, sol.x[:, 1], label='x₂', linewidth=2)
axes[0].axvline(t_sim[steady_start], color='black', linestyle='--', alpha=0.5, label='Steady state starts')
axes[0].set_title('Complete System Response')
axes[0].set_ylabel('Reaction Forces')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Steady-state oscillations
axes[1].plot(t_steady, x_steady[:, 0], label='x₁ (steady)', linewidth=2)
axes[1].plot(t_steady, x_steady[:, 1], label='x₂ (steady)', linewidth=2)
axes[1].set_title('Steady-State Periodic Oscillations')
axes[1].set_ylabel('Reaction Forces')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Control signal
axes[2].plot(t_steady, u_steady[:, 0], color='red', linewidth=2, label='u₁')
axes[2].set_title('Sinusoidal Control Input')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Control')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Verify we achieved the target oscillation
X_predicted = freq_controller.evaluate_steady_state(U_optimal, omega)
print(f"Target amplitude: {np.abs(X_target)}")
print(f"Predicted amplitude: {np.abs(X_predicted)}")
print(f"Target phase: {np.angle(X_target) * 180 / np.pi} degrees")
print(f"Predicted phase: {np.angle(X_predicted) * 180 / np.pi} degrees")
```

---

## Tutorial 7: Thermodynamic Accounting and Entropy Production

Analyze energy balance and entropy production in controlled systems.

### Setup

```python
import llrq
import numpy as np
import matplotlib.pyplot as plt
from llrq import ThermodynamicAccountant
```

### Step 1: System with External Drive

```python
# Create simple system with external driving
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0,
                              concentrations=[1.0, 0.5])
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)

# Define Onsager conductance matrix (governs entropy production)
L = np.array([[1.2]])  # Single reaction system

# Create thermodynamic accountant
accountant = ThermodynamicAccountant(network, onsager_conductance=L)
```

### Step 2: Compare Control Strategies

```python
# Define different control strategies
def no_control(t):
    return np.array([0.0])

def constant_drive(t):
    return np.array([0.5])

def sinusoidal_drive(t):
    return np.array([0.5 * np.sin(0.8 * t)])

def pulse_drive(t):
    return np.array([1.0 if 2 < t < 4 else 0.0])

strategies = {
    'No Control': no_control,
    'Constant Drive': constant_drive,
    'Sinusoidal Drive': sinusoidal_drive,
    'Pulse Drive': pulse_drive
}

# Simulate each strategy
t = np.linspace(0, 10, 200)
results = {}
entropy_results = {}

for name, u_func in strategies.items():
    print(f"Analyzing {name}...")

    # Simulate
    sol = solver.solve(t, u=u_func)

    # Compute entropy production
    entropy_result = accountant.entropy_from_x(t, sol.x)

    # Store results
    results[name] = sol
    entropy_results[name] = entropy_result

    print(f"  Total entropy production: {entropy_result.sigma_total:.4f}")
```

### Step 3: Energy Balance Analysis

```python
# Perform dual entropy accounting for one strategy
print("Detailed analysis of sinusoidal drive:")

sol = results['Sinusoidal Drive']
u_trajectory = np.array([sinusoidal_drive(t_i) for t_i in t])

# Dual accounting (reaction forces vs external drives)
dual_result = accountant.dual_entropy_accounting(t, sol.x, u_trajectory)

print(f"Entropy from reaction forces: {dual_result.from_x.sigma_total:.4f}")
print(f"Entropy from external drives: {dual_result.from_u.sigma_total:.4f}")
print(f"Relative difference: {abs(dual_result.from_x.sigma_total - dual_result.from_u.sigma_total) / dual_result.from_x.sigma_total * 100:.1f}%")

# Energy balance components
balance = dual_result.balance
print(f"Energy balance validation:")
print(f"  System potential change: {balance['V_dot_integral']:.4f}")
print(f"  Relaxation power: {balance['P_relax_integral']:.4f}")
print(f"  Control power: {balance['P_ctrl_integral']:.4f}")
print(f"  Balance residual: {balance['residual_integral']:.4f}")
```

### Step 4: Visualization and Analysis

```python
# Plot comprehensive comparison
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: State trajectories
for i, (name, sol) in enumerate(results.items()):
    color = plt.cm.Set1(i)
    axes[0, 0].plot(t, sol.x[:, 0], label=name, linewidth=2, color=color)
axes[0, 0].set_title('Reaction Forces')
axes[0, 0].set_ylabel('x (reaction force)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Entropy production rates
for i, (name, entropy_result) in enumerate(entropy_results.items()):
    color = plt.cm.Set1(i)
    axes[0, 1].plot(t, entropy_result.sigma_time, label=name, linewidth=2, color=color)
axes[0, 1].set_title('Entropy Production Rate')
axes[0, 1].set_ylabel('σ(t) (entropy rate)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Cumulative entropy
for i, (name, entropy_result) in enumerate(entropy_results.items()):
    color = plt.cm.Set1(i)
    cumulative = np.cumsum(entropy_result.sigma_time) * (t[1] - t[0])
    axes[1, 0].plot(t, cumulative, label=name, linewidth=2, color=color)
axes[1, 0].set_title('Cumulative Entropy Production')
axes[1, 0].set_ylabel('∫σ dt (total entropy)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Total entropy comparison (bar chart)
names = list(entropy_results.keys())
totals = [entropy_results[name].sigma_total for name in names]
bars = axes[1, 1].bar(range(len(names)), totals, color=plt.cm.Set1(np.arange(len(names))))
axes[1, 1].set_title('Total Entropy Production Comparison')
axes[1, 1].set_ylabel('Total Entropy')
axes[1, 1].set_xticks(range(len(names)))
axes[1, 1].set_xticklabels(names, rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Energy balance for sinusoidal case
balance = dual_result.balance
axes[2, 0].plot(t, balance['V_dot'], label='dV/dt', linewidth=2)
axes[2, 0].plot(t, balance['P_relax'], label='P_relax', linewidth=2)
axes[2, 0].plot(t, balance['P_ctrl'], label='P_ctrl', linewidth=2)
axes[2, 0].set_title('Energy Balance Components (Sinusoidal)')
axes[2, 0].set_xlabel('Time')
axes[2, 0].set_ylabel('Power')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Plot 6: Energy balance residual
axes[2, 1].plot(t, balance['residual'], color='red', linewidth=2)
axes[2, 1].set_title('Energy Balance Residual')
axes[2, 1].set_xlabel('Time')
axes[2, 1].set_ylabel('Residual')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 5: Entropy-Aware Control Design

```python
# Design control that minimizes entropy production
from llrq import CVXController, CVXObjectives

# Create CVX controller
controller = CVXController(solver)

# Custom entropy-penalized objective
def entropy_aware_objective(variables, params):
    import cvxpy as cp
    u = variables["u"]
    x = variables["x"]

    # Standard tracking cost
    if "x_target" in params and params["x_target"] is not None:
        tracking_cost = cp.sum_squares(x - params["x_target"])
    else:
        tracking_cost = cp.sum_squares(x)  # Minimize deviation from zero

    # Entropy penalty using Onsager matrix
    if "L" in params:
        entropy_cost = cp.quad_form(x, params["L"])
    else:
        entropy_cost = 0

    return tracking_cost + 0.1 * entropy_cost

# Solve entropy-aware control problem
from llrq import CVXConstraints

result = controller.compute_cvx_control(
    objective_fn=entropy_aware_objective,
    constraints_fn=CVXConstraints.box_bounds(u_min=-1, u_max=1),
    L=L  # Pass Onsager matrix
)

print(f"Entropy-aware optimal control: {result['u_optimal']}")

# Compare with standard control
standard_control = controller.compute_cvx_control(
    objective_fn=CVXObjectives.sparse_control(),
    constraints_fn=CVXConstraints.box_bounds(u_min=-1, u_max=1)
)

print(f"Standard optimal control: {standard_control['u_optimal']}")

# Simulate both and compare entropy
def entropy_aware_ctrl(t): return result['u_optimal']
def standard_ctrl(t): return standard_control['u_optimal']

sol_entropy_aware = solver.solve(t, u=entropy_aware_ctrl)
sol_standard = solver.solve(t, u=standard_ctrl)

entropy_aware_result = accountant.entropy_from_x(t, sol_entropy_aware.x)
standard_result = accountant.entropy_from_x(t, sol_standard.x)

print(f"\nEntropy production comparison:")
print(f"  Entropy-aware control: {entropy_aware_result.sigma_total:.4f}")
print(f"  Standard control: {standard_result.sigma_total:.4f}")
print(f"  Reduction: {(standard_result.sigma_total - entropy_aware_result.sigma_total) / standard_result.sigma_total * 100:.1f}%")
```

---

## Common Troubleshooting

### Issue 1: Negative Concentrations

**Problem**: Concentrations become negative during simulation.

**Solution**:
```python
# Use conservation law enforcement
solution = solver.solve(
    initial_conditions=concentrations,
    t_span=(0, 10),
    enforce_conservation=True,  # This helps prevent negative values
    method='numerical'
)
```

### Issue 2: Stiff Systems

**Problem**: System evolves very slowly or solver fails.

**Solution**:
```python
# Use stiff solver
solution = solver.solve(
    initial_conditions=concentrations,
    t_span=(0, 10),
    method='numerical',
    rtol=1e-8,  # Tighter tolerance
    atol=1e-10,
    solver='Radau'  # Stiff solver
)
```

### Issue 3: Large Equilibrium Constants

**Problem**: Reaction quotients become very large/small.

**Solution**:
```python
# Work in log-deviation space directly
x0 = dynamics.compute_log_deviation(initial_Q)
log_solution = solver.solve_analytical(x0, t_span)
```

---

## Next Steps

After completing these tutorials:

1. **[Explore the API](api-reference.html)** for advanced features
2. **[Study the Theory](theory.html)** to understand the mathematics
3. **[Try the Examples](examples.html)** for complete working code
4. **Experiment** with your own reaction networks!

## Tips for Success

- **Start simple**: Begin with single reactions before complex networks
- **Check conservation**: Always verify mass balance is preserved
- **Use analytical solutions**: When possible, they're faster and more accurate
- **Visualize everything**: Plots reveal insights that numbers don't
- **Read the literature**: The [original paper](https://arxiv.org/pdf/2508.18523) has more details
