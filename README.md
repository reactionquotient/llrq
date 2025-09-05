# LLRQ - Log-Linear Reaction Quotient Dynamics

A Python package for analyzing chemical reaction networks using the log-linear reaction quotient dynamics framework described in Diamond (2025) "Log-Linear Reaction Quotient Dynamics" ([arXiv:2508.18523](https://arxiv.org/pdf/2508.18523)).

## Overview

This framework introduces a novel approach to modeling chemical reaction networks where reaction quotients (Q) evolve exponentially toward equilibrium when viewed on a logarithmic scale. Unlike traditional mass action kinetics, this yields analytically tractable linear dynamics in log-space.

### Key Equation

For reaction networks, the dynamics follow:
```
d/dt ln Q = -K ln(Q/Keq) + u(t)
```

Where:
- **Q**: Vector of reaction quotients measuring distance from equilibrium
- **Keq**: Vector of equilibrium constants  
- **K**: Relaxation rate matrix
- **u(t)**: External drive vector (e.g., ATP/ADP ratios)

## Framework Advantages

1. **Analytical Solutions**: Exact solutions exist for arbitrary network topologies
2. **Thermodynamic Integration**: Automatic incorporation of constraints via ΔG = RT ln(Q/Keq)
3. **Decoupled Dynamics**: Conservation laws separate from reaction quotient evolution  
4. **Linear Control**: External energy sources couple linearly to dynamics
5. **Tractable Analysis**: Decades of linear systems theory become applicable

## Installation

### From PyPI (when available)
```bash
pip install llrq
```

### From source
```bash
git clone <repository-url>
cd llrq
pip install -e .
```

### Requirements
- Python ≥3.8
- numpy ≥1.20.0
- scipy ≥1.7.0
- matplotlib ≥3.3.0
- python-libsbml ≥5.19.0

## Quick Start

### Simple Reaction Example

```python
import llrq
import numpy as np

# Create a simple A ⇌ B reaction
network, dynamics, solver, visualizer = llrq.simple_reaction(
    reactant_species="A",
    product_species="B", 
    equilibrium_constant=2.0,
    relaxation_rate=1.0,
    initial_concentrations={"A": 1.0, "B": 0.1}
)

# Solve the dynamics
solution = solver.solve(
    initial_conditions={"A": 1.0, "B": 0.1},
    t_span=(0, 10),
    method='analytical'
)

# Visualize results
fig = visualizer.plot_dynamics(solution)
```

### SBML Model Example

```python
import llrq

# Load SBML model
network, dynamics, solver, visualizer = llrq.from_sbml(
    'model.xml',
    equilibrium_constants=np.array([2.0, 1.5, 0.8]),  # Keq for each reaction
    relaxation_matrix=np.diag([1.0, 0.5, 1.2])        # K matrix (diagonal)
)

# Solve with external drive
def atp_drive(t):
    return np.array([0.5, 0.0, -0.2])  # Drive for each reaction

dynamics.external_drive = atp_drive
solution = solver.solve(
    initial_conditions=network.get_initial_concentrations(),
    t_span=(0, 20)
)

# Plot results  
fig = visualizer.plot_dynamics(solution)
```

## API Reference

### Core Classes

- **`SBMLParser`**: Parse SBML models and extract network information
- **`ReactionNetwork`**: Represent reaction networks with stoichiometry and species
- **`LLRQDynamics`**: Implement log-linear dynamics system 
- **`LLRQSolver`**: Solve dynamics with analytical and numerical methods
- **`LLRQVisualizer`**: Create publication-quality plots

### Convenience Functions

- **`llrq.from_sbml()`**: Load SBML model and create complete LLRQ system
- **`llrq.simple_reaction()`**: Create simple A ⇌ B reaction system

## Examples

See the `examples/` directory for complete working examples:

- **`simple_example.py`**: Basic A ⇌ B reaction with visualization
- **`sbml_example.py`**: Loading and analyzing SBML models
- **`external_drive_example.py`**: Systems with time-varying external drives
- **`glycolysis_example.py`**: Oscillatory glycolytic pathway

## Mathematical Framework

### Single Reaction Dynamics
For a single reaction with external drive:
```
d/dt ln Q = -k ln(Q/Keq) + u(t)
```

**Analytical solution** (constant u):
```
Q(t) = Keq * exp[(ln(Q₀/Keq) - u/k) * exp(-kt) + u/k]
```

### Multiple Reactions
Vector form for reaction networks:
```
d/dt ln Q = -K ln(Q/Keq) + u(t)
```

**Matrix exponential solution** (constant u):
```
ln Q(t) = exp(-Kt) * [ln(Q₀/Keq) - K⁻¹u] + K⁻¹u + ln Keq
```

### Connection to Mass Action
For single reaction A ⇌ B with mass action rates kf, kr:
- Equilibrium constant: `Keq = kf/kr`  
- Relaxation rate: `k = kr(1 + Keq)` (ensures agreement near equilibrium)

## Applications

This framework enables:
- **Metabolic Engineering**: Optimize pathway design using K as design variable
- **Drug Discovery**: Predict drug effects throughout metabolic networks  
- **Systems Medicine**: Classify metabolic disorders via eigenvalue analysis
- **Control Theory**: Apply optimal control to cellular metabolism

## Citation

If you use this package in research, please cite:

```bibtex
@article{diamond2025loglinear,
  title={Log-Linear Reaction Quotient Dynamics},
  author={Diamond, Steven},
  journal={arXiv preprint arXiv:2508.18523},
  year={2025}
}
```

## License

Licensed under the Apache License 2.0. See `LICENSE` file for details.

## Contact

- **Author**: Steven Diamond
- **Email**: steven@gridmatic.com  
- **Paper**: [arXiv:2508.18523](https://arxiv.org/pdf/2508.18523)