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

## Navigation

- **[Getting Started](getting-started.html)**: Installation and first steps
- **[API Reference](api-reference.html)**: Complete API documentation
- **[Tutorials](tutorials.html)**: Step-by-step guides and examples
- **[Theory](theory.html)**: Mathematical foundations
- **[Examples](examples.html)**: Complete working examples

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

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Steven Diamond
- **Email**: steven@gridmatic.com  
- **Paper**: [arXiv:2508.18523](https://arxiv.org/pdf/2508.18523)