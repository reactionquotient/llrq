---
layout: page
title: Getting Started
nav_order: 2
---

# Getting Started with LLRQ

This guide will walk you through installing LLRQ and running your first reaction network analysis.

## Installation

### Option 1: From PyPI (Recommended when available)
```bash
pip install llrq
```

### Option 2: From Source
```bash
git clone https://github.com/yourusername/llrq.git
cd llrq
pip install -e .
```

### Requirements

LLRQ requires Python 3.8 or higher and the following packages:
- numpy ≥1.20.0
- scipy ≥1.7.0
- matplotlib ≥3.3.0
- python-libsbml ≥5.19.0 (for SBML support)
- cvxpy ≥1.3.0 (for optimization-based control)

#### Optional Dependencies for Advanced Features

For full control system capabilities:
```bash
pip install cvxpy  # For optimization-based control
pip install control  # For classical control theory tools (optional)
```

For high-performance solvers (recommended):
```bash
# Commercial solver (free academic license)
conda install -c mosek mosek

# Or open-source alternatives
pip install clarabel  # Fast conic solver
pip install scs       # Splitting conic solver
```

### Using Conda Environment (Recommended)

If you're working with biological models, we recommend using the tellurium conda environment:

```bash
conda install -c sys-bio tellurium
conda activate tellurium
pip install -e .
```

## Your First Reaction Network

Let's start with the simplest possible example: a single reversible reaction A ⇌ B.

### Step 1: Import LLRQ

```python
import llrq
import numpy as np
import matplotlib.pyplot as plt
```

### Step 2: Create a Simple Reaction

```python
# Create a simple A ⇌ B reaction
network, dynamics, solver, visualizer = llrq.simple_reaction(
    reactant_species="A",
    product_species="B",
    equilibrium_constant=2.0,      # Keq = [B]/[A] at equilibrium
    relaxation_rate=1.0,           # How fast the system approaches equilibrium
    initial_concentrations={"A": 1.0, "B": 0.1}
)
```

### Step 3: Examine the Network

```python
# Print network information
print("Network Summary:")
print(network.summary())
print(f"Number of species: {network.n_species}")
print(f"Number of reactions: {network.n_reactions}")
print(f"Species names: {network.species_names}")
```

### Step 4: Solve the Dynamics

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

### Step 5: Visualize Results

```python
# Create plots
fig = visualizer.plot_dynamics(solution)
plt.show()

# You can also plot individual components
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot concentrations vs time
ax1.plot(solution['t'], solution['concentrations']['A'], label='A', linewidth=2)
ax1.plot(solution['t'], solution['concentrations']['B'], label='B', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Concentration')
ax1.set_title('Species Concentrations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot reaction quotient vs time
ax2.plot(solution['t'], solution['reaction_quotients'], linewidth=2)
ax2.axhline(y=2.0, color='r', linestyle='--', label='Equilibrium (Keq=2.0)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Reaction Quotient Q')
ax2.set_title('Reaction Quotient Evolution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Understanding the Results

### What You Should See

1. **Species A** starts at concentration 1.0 and decreases toward equilibrium
2. **Species B** starts at concentration 0.1 and increases toward equilibrium
3. **Reaction quotient Q = [B]/[A]** starts at 0.1 and approaches Keq = 2.0
4. The dynamics follow an exponential approach to equilibrium

### Key Insights

The LLRQ framework shows that:
- The reaction quotient evolves as: Q(t) = Keq × exp[(ln(Q₀/Keq)) × exp(-kt)]
- This is **linear in log-space**: d/dt ln Q = -k ln(Q/Keq)
- The solution is **analytically exact**, not an approximation

## Control System Workflow

Beyond basic simulation, LLRQ provides powerful control capabilities. Here's a quick example:

### Basic Control Example

```python
from llrq import CVXController, CVXObjectives, CVXConstraints

# Create a controlled system (using the same network from above)
controller = CVXController(solver, controlled_reactions=["R1"])

# Define control objective and constraints
objective = CVXObjectives.sparse_control(sparsity_weight=0.1)
constraints = CVXConstraints.box_bounds(u_min=-2.0, u_max=2.0)

# Design optimal control to reach target state
target_forces = np.array([0.5, -0.3])  # Target reaction forces
result = controller.compute_cvx_control(
    objective_fn=objective,
    constraints_fn=constraints,
    x_target=target_forces
)

print(f"Optimal control: {result['u_optimal']}")
print(f"Optimization status: {result['status']}")

# Simulate with optimal control
def control_function(t):
    return result['u_optimal']

controlled_solution = solver.solve(t_span=(0, 10), u=control_function)
```

### Frequency-Domain Control

```python
from llrq import FrequencySpaceController

# Create frequency controller
freq_controller = FrequencySpaceController.from_llrq_solver(solver)

# Design 1 Hz sinusoidal control
omega = 2 * np.pi * 1.0  # 1 Hz
X_target = np.array([1.0]) * np.exp(1j * np.pi/4)  # Amplitude and phase

U_optimal = freq_controller.design_sinusoidal_control(
    X_target=X_target, omega=omega, lam=0.01
)

# Convert to time-domain control
def sinusoidal_control(t):
    return np.real(U_optimal * np.exp(1j * omega * t))

oscillatory_solution = solver.solve(t_span=(0, 5), u=sinusoidal_control)
```

### Thermodynamic Analysis

```python
from llrq import ThermodynamicAccountant

# Set up entropy accounting
L = np.eye(len(network.reaction_ids))  # Onsager conductance matrix
accountant = ThermodynamicAccountant(network, onsager_conductance=L)

# Analyze entropy production
entropy_result = accountant.entropy_from_x(
    controlled_solution['t'],
    controlled_solution['x']
)

print(f"Total entropy production: {entropy_result.sigma_total:.4f}")
```

## Next Steps

Now that you've seen basic simulation and control, you can:

1. **[Explore the API](api-reference.html)** - Learn about all available classes and methods
2. **[Follow Tutorials](tutorials.html)** - Work through more complex examples including:
   - **Tutorial 5**: Control optimization with CVXPY
   - **Tutorial 6**: Frequency-domain control design
   - **Tutorial 7**: Thermodynamic accounting and entropy production
3. **[Study the Theory](theory.html)** - Understand the mathematical foundations
4. **[Try Examples](examples.html)** - See complete working examples
5. **[Control Documentation](cvx-control.html)** - Deep dive into advanced control features

## Common Issues

### Import Errors
If you get import errors, make sure you've installed all dependencies:
```bash
pip install numpy scipy matplotlib python-libsbml cvxpy
```

### CVXPY Installation Issues
For control features, CVXPY is required. If you encounter solver issues:
```bash
# Install CVXPY with additional solvers
pip install cvxpy[CLARABEL,SCS]

# For academic users, MOSEK provides fast performance
conda install -c mosek mosek
```

### SBML Support
For SBML model support, ensure python-libsbml is installed:
```bash
conda install -c conda-forge python-libsbml
# or
pip install python-libsbml
```

### Control System Errors
If optimization fails with "solver not available":
```python
import cvxpy as cp
print("Available solvers:", cp.installed_solvers())
```

Common solutions:
- Install additional solvers: `pip install clarabel scs`
- Use different solver: `solver_options={'solver': 'SCS'}`
- Reduce problem size or add regularization

### Plotting Issues
If plots don't display, you may need to configure matplotlib backend:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

## Getting Help

- Check the [API Reference](api-reference.html) for detailed documentation
- Look at [Examples](examples.html) for working code
- Review [Tutorials](tutorials.html) for step-by-step guides
- Read the [Theory](theory.html) section for mathematical details
