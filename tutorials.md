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