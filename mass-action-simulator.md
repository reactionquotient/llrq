---
layout: page
title: Mass Action Simulator
nav_order: 7
---

# Mass Action Simulator & Comparison

This page provides interactive tools for comparing LLRQ dynamics with traditional mass action kinetics.

## LLRQ vs Mass Action Comparison

The LLRQ package includes built-in comparison capabilities with mass action kinetics. This allows you to:

1. **Compare dynamics side-by-side**: Plot both LLRQ and mass action solutions
2. **Analyze differences**: Understand when the approaches diverge
3. **Validate approximations**: See how well LLRQ captures mass action behavior

## Interactive Comparison Report

<iframe src="mass-action-comparison.html" width="100%" height="800" frameborder="0"></iframe>

## Using the Comparison Tools

### Python API

```python
import llrq
import numpy as np

# Create a reaction network
network, dynamics, solver, visualizer = llrq.simple_reaction(
    reactant_species="A",
    product_species="B",
    equilibrium_constant=2.0,
    relaxation_rate=1.0,
    initial_concentrations={"A": 1.0, "B": 0.1}
)

# Solve with LLRQ
llrq_solution = solver.solve(
    initial_conditions={"A": 1.0, "B": 0.1},
    t_span=(0, 10),
    method='analytical'
)

# Compare with mass action
fig = visualizer.plot_mass_action_comparison(
    llrq_solution,
    show_quotient_evolution=True
)
```

### Key Features

- **Analytical LLRQ Solutions**: Exact solutions for linear log-space dynamics
- **Mass Action Integration**: Numerical solution of traditional rate equations
- **Quotient Evolution**: Track how reaction quotients evolve toward equilibrium
- **Thermodynamic Consistency**: Both approaches respect detailed balance

## When to Use Each Approach

| Scenario | LLRQ | Mass Action |
|----------|------|-------------|
| **Near equilibrium** | ✓ Excellent | ✓ Good |
| **Far from equilibrium** | ✓ Good approximation | ✓ Exact |
| **Large networks** | ✓ Analytical solutions | ⚠ Numerical stiffness |
| **Control theory** | ✓ Linear dynamics | ⚠ Nonlinear complexity |
| **Thermodynamic analysis** | ✓ Direct ΔG coupling | ⚠ Requires careful handling |

## Mathematical Foundation

The LLRQ framework provides a linearized approximation to mass action kinetics in log-space:

**Mass Action:**
$$\frac{dc_i}{dt} = \sum_j S_{ij} \left( k_j^+ \prod_k c_k^{S_{kj}^+} - k_j^- \prod_k c_k^{S_{kj}^-} \right)$$

**LLRQ Approximation:**
$$\frac{d}{dt} \ln Q_j = -K_{jj} \ln(Q_j/K_{eq,j}) + \sum_{k \neq j} K_{jk} \ln(Q_k/K_{eq,k}) + u_j(t)$$

The LLRQ approach captures the essential thermodynamic driving forces while providing analytical tractability.

## Applications

This comparison capability is useful for:

- **Model validation**: Verify LLRQ predictions against established mass action models
- **Method selection**: Choose the appropriate framework for your application
- **Educational purposes**: Understand the relationship between different modeling approaches
- **Hybrid approaches**: Use LLRQ for analysis and mass action for detailed simulations

## See Also

- [Theory](theory.html) - Mathematical foundations
- [API Reference](api-reference.html) - Complete API documentation
- [Examples](examples.html) - Working code examples