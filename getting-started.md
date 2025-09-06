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

## Next Steps

Now that you've run your first LLRQ simulation, you can:

1. **[Explore the API](api-reference.html)** - Learn about all available classes and methods
2. **[Follow Tutorials](tutorials.html)** - Work through more complex examples
3. **[Study the Theory](theory.html)** - Understand the mathematical foundations
4. **[Try Examples](examples.html)** - See complete working examples

## Common Issues

### Import Errors
If you get import errors, make sure you've installed all dependencies:
```bash
pip install numpy scipy matplotlib python-libsbml
```

### SBML Support
For SBML model support, ensure python-libsbml is installed:
```bash
conda install -c conda-forge python-libsbml
# or
pip install python-libsbml
```

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