---
layout: home
title: Home
nav_order: 1
---

# LLRQ - Log-Linear Reaction Quotient Dynamics

A Python package for analyzing chemical reaction networks using the log-linear reaction quotient dynamics framework described in Diamond (2025) "[Log-Linear Reaction Quotient Dynamics](https://arxiv.org/pdf/2508.18523)".

## Overview

This framework introduces a novel approach to modeling chemical reaction networks where reaction quotients (Q) evolve exponentially toward equilibrium when viewed on a logarithmic scale. Unlike traditional mass action kinetics, this yields analytically tractable linear dynamics in log-space.

### Key Equation

For reaction networks, the dynamics follow:

$$\frac{d}{dt} \ln Q = -K \ln(Q/K_{eq}) + u(t)$$

Where:
- **Q**: Vector of reaction quotients measuring distance from equilibrium
- **K<sub>eq</sub>**: Vector of equilibrium constants
- **K**: Relaxation rate matrix
- **u(t)**: External drive vector (e.g., ATP/ADP ratios)

## Framework Advantages

1. **Analytical Solutions**: Exact solutions exist for arbitrary network topologies
2. **Thermodynamic Integration**: Automatic incorporation of constraints via ΔG = RT ln(Q/K<sub>eq</sub>)
3. **Decoupled Dynamics**: Conservation laws separate from reaction quotient evolution
4. **Linear Control**: External energy sources couple linearly to dynamics
5. **Tractable Analysis**: Decades of linear systems theory become applicable
6. **Advanced Control Systems**: CVXPY optimization, frequency-domain design, LQR control
7. **Entropy Accounting**: Rigorous thermodynamic analysis with energy balance validation

## Interactive Demo

Try the LLRQ framework right here in your browser! Adjust parameters and watch how the reaction A ⇌ B evolves according to log-linear dynamics.

{% include interactive_demo.html %}

## Quick Start

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

## New Features

### 🎛️ Advanced Control Systems

**CVXPY-Based Optimization**: Design custom control strategies using convex optimization with flexible objectives and constraints.

```python
from llrq import CVXController, CVXObjectives, CVXConstraints

# Sparse control with bounds
controller = CVXController(solver)
objective = CVXObjectives.sparse_control(sparsity_weight=0.1)
constraints = CVXConstraints.box_bounds(u_min=-2, u_max=2)

result = controller.compute_cvx_control(
    objective_fn=objective,
    constraints_fn=constraints,
    x_target=np.array([0.5, -0.5])
)
```

**Frequency-Domain Control**: Design sinusoidal inputs for periodic steady states with entropy awareness.

```python
from llrq import FrequencySpaceController

freq_controller = FrequencySpaceController.from_llrq_solver(solver)
U_optimal = freq_controller.design_sinusoidal_control(
    X_target=np.array([1.0]) * np.exp(1j * np.pi/4),  # 45° phase
    omega=2.0,  # 2 rad/s
    lam=0.01    # regularization
)
```

**LQR and Adaptive Control**: Classical optimal control with real-time parameter adaptation.

```python
from llrq.control.lqr import lqr_control
from llrq import AdaptiveController

# LQR design
K_lqr, P, eigenvals = lqr_control(controller.A, controller.B, Q, R)

# Adaptive parameter estimation
adaptive = AdaptiveController(solver)
estimated_params = adaptive.estimate_parameters(t_data, x_data, u_data)
```

### 🔥 Thermodynamic Accounting

**Entropy Production Analysis**: Rigorous computation of entropy production from reaction forces and external drives.

```python
from llrq import ThermodynamicAccountant

accountant = ThermodynamicAccountant(network, onsager_conductance=L)

# Entropy from reaction forces
entropy_result = accountant.entropy_from_x(t, sol.x)

# Cross-validation with energy balance
dual_result = accountant.dual_entropy_accounting(t, sol.x, u_trajectory)
print(f"Energy balance residual: {dual_result.balance['residual_integral']}")
```

**Entropy-Aware Control**: Integrate thermodynamic costs directly into control optimization.

```python
from llrq import create_entropy_aware_cvx_controller

controller = create_entropy_aware_cvx_controller(solver, reactions, L)
entropy_obj = CVXObjectives.multi_objective({
    'tracking': 1.0,
    'entropy': 0.1  # Penalize entropy production
})
```

### 🔄 Enhanced Integration

**Mass Action Integration**: Seamless comparison between LLRQ and traditional kinetics with unified interfaces.

```python
# Convert mass action to LLRQ
llrq_dynamics = LLRQDynamics.from_mass_action(network, kinetic_constants)

# Compare control strategies
comparison = compare_control_methods(solver, methods, target)
```

**Workflow Integration**: High-level functions for common tasks with comprehensive analysis.

```python
# Automatic target reaching
result = simulate_to_target(solver, target, method='lqr', time_horizon=10)

# Control strategy comparison
comparison = compare_control_methods(solver, methods, target)
comparison.plot_comparison()
```

## Navigation

### Core Documentation
- **[Getting Started](getting-started.html)**: Installation and first steps
- **[API Reference](api-reference.html)**: Complete API documentation
- **[Tutorials](tutorials.html)**: Step-by-step guides and examples
- **[Theory](theory.html)**: Mathematical foundations
- **[Examples](examples.html)**: Complete working examples

### Control Systems
- **[CVXPY Control](cvx-control.html)**: Optimization-based control design
- **[Frequency Control](frequency-control.html)**: Sinusoidal control in frequency domain
- **[Advanced Control](advanced-control.html)**: LQR, adaptive control, simulation workflows

### Thermodynamic Analysis
- **[Thermodynamic Accounting](thermodynamics.html)**: Entropy production and energy balance

## Applications

This framework enables:
- **Metabolic Engineering**: Optimize pathway design using K as design variable with entropy-aware control
- **Drug Discovery**: Predict drug effects throughout metabolic networks using frequency-domain analysis
- **Systems Medicine**: Classify metabolic disorders via eigenvalue analysis and thermodynamic accounting
- **Optimal Control**: Apply LQR, CVXPY optimization, and adaptive control to cellular metabolism
- **Energy Harvesting**: Design sinusoidal control strategies for oscillatory biological systems
- **Process Control**: Thermodynamically-consistent control of chemical reactors and bioprocesses
- **Circadian Biology**: Frequency-domain control design for biological rhythm engineering

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

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Steven Diamond
- **Email**: steven@gridmatic.com
- **Paper**: [arXiv:2508.18523](https://arxiv.org/pdf/2508.18523)
