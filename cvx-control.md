---
layout: page
title: CVXPY-Based Control Optimization
nav_order: 8
---

# CVXPY-Based Control Optimization

The LLRQ package provides powerful optimization-based control capabilities through the `CVXController` class, which leverages the CVXPY library for flexible control design. This enables solving complex control optimization problems with custom objectives and constraints that would be difficult or impossible with traditional analytical methods.

## Overview

The `CVXController` extends the base `LLRQController` with the ability to solve arbitrary convex optimization problems for control input design. The key innovation is a callback-based interface where you define objectives and constraints as functions that receive CVXPY variables and return CVXPY expressions.

### Key Features

- **Custom Objectives**: L1/L2 norms, tracking objectives, entropy-aware costs, robust control
- **Flexible Constraints**: Box bounds, budget constraints, state limits, steady-state constraints
- **Pre-built Templates**: Common objectives and constraints ready to use
- **CVXPY Integration**: Full access to CVXPY's optimization capabilities
- **Entropy-Aware Control**: Integration with thermodynamic accounting

## Quick Start

```python
import llrq
import numpy as np
from llrq import CVXController, CVXObjectives, CVXConstraints

# Create LLRQ system
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0)
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create CVX controller
controller = CVXController(solver, controlled_reactions=["R1"])

# Sparse control optimization
objective = CVXObjectives.sparse_control(sparsity_weight=0.1)
constraints = CVXConstraints.box_bounds(u_min=-2, u_max=2)

result = controller.compute_cvx_control(
    objective_fn=objective,
    constraints_fn=constraints,
    x_target=np.array([0.5, -0.5])  # Target reaction forces
)

print(f"Optimal control: {result['u_optimal']}")
print(f"Status: {result['status']}")
```

## CVXController Class

### Constructor

```python
CVXController(solver, controlled_reactions=None)
```

- **solver**: LLRQSolver with computed basis matrices
- **controlled_reactions**: List of reaction IDs/indices to control (None for all)

### Main Method: compute_cvx_control()

The core method that sets up and solves the optimization problem:

```python
result = controller.compute_cvx_control(
    objective_fn=None,           # Custom objective function
    constraints_fn=None,         # Custom constraints function
    x_target=None,              # Target reaction forces (r,)
    y_target=None,              # Target reduced state (rankS,)
    solver_options=None,        # CVXPY solver options
    **problem_params            # Additional parameters
)
```

**Returns** dictionary with:
- `u_optimal`: Optimal full control input
- `u_controlled`: Optimal controlled reactions only
- `problem`: CVXPY Problem object
- `variables`: Dictionary of CVXPY variables
- `objective_value`: Optimal objective value
- `status`: Solver status string

## Pre-built Objectives

### Sparse Control

L1-regularized objective for minimal control effort:

```python
from llrq import CVXObjectives

# Basic sparse control
sparse_obj = CVXObjectives.sparse_control(
    sparsity_weight=1.0,    # L1 penalty weight
    tracking_weight=1.0     # Tracking error weight
)

result = controller.compute_cvx_control(
    objective_fn=sparse_obj,
    x_target=target_forces
)
```

### Multi-Objective Optimization

Combine multiple objectives with custom weights:

```python
multi_obj = CVXObjectives.multi_objective({
    'tracking': 1.0,     # Tracking error
    'control': 0.1,      # Control effort (L2)
    'sparsity': 0.05,    # Control sparsity (L1)
    'entropy': 0.01      # Entropy production (requires M matrix)
})

result = controller.compute_cvx_control(
    objective_fn=multi_obj,
    x_target=target_forces,
    M=entropy_metric  # Entropy metric matrix
)
```

### Robust Control

Uncertainty-aware control design:

```python
robust_obj = CVXObjectives.robust_tracking(uncertainty_weight=0.1)

result = controller.compute_cvx_control(
    objective_fn=robust_obj,
    x_target=target_forces
)
```

## Pre-built Constraints

### Box Bounds

Simple upper and lower bounds on control inputs:

```python
from llrq import CVXConstraints

# Scalar bounds (all controls)
box_constraints = CVXConstraints.box_bounds(u_min=-1.0, u_max=2.0)

# Vector bounds (per control)
box_constraints = CVXConstraints.box_bounds(
    u_min=np.array([-1, -0.5, -2]),
    u_max=np.array([2, 1.5, 1])
)
```

### Control Budget

Total control effort constraints:

```python
# L1 budget constraint
budget_L1 = CVXConstraints.control_budget(total_budget=5.0, norm_type=1)

# L2 budget constraint
budget_L2 = CVXConstraints.control_budget(total_budget=2.0, norm_type=2)
```

### Steady-State Constraints

Enforce steady-state relationship x = K^(-1)u:

```python
steady_state = CVXConstraints.steady_state()

result = controller.compute_cvx_control(
    objective_fn=sparse_obj,
    constraints_fn=steady_state
)
```

### State Bounds

Bounds on reaction force states:

```python
state_bounds = CVXConstraints.state_bounds(
    x_min=-2.0,  # Lower bounds
    x_max=3.0    # Upper bounds
)
```

### Combining Constraints

Use `CVXConstraints.combine()` to use multiple constraint types:

```python
combined = CVXConstraints.combine(
    CVXConstraints.box_bounds(u_min=-1, u_max=1),
    CVXConstraints.control_budget(total_budget=3.0, norm_type=1),
    CVXConstraints.steady_state()
)

result = controller.compute_cvx_control(
    objective_fn=multi_obj,
    constraints_fn=combined
)
```

## Custom Objectives and Constraints

### Custom Objective Function

Define your own objective as a function that takes `(variables, params)` and returns a CVXPY expression:

```python
def custom_objective(variables, params):
    u = variables["u"]
    x = variables["x"]

    # Custom cost function
    control_cost = cp.sum_squares(u)
    state_penalty = cp.norm(x, 1)  # L1 penalty on states

    return control_cost + 0.1 * state_penalty

result = controller.compute_cvx_control(
    objective_fn=custom_objective,
    constraints_fn=steady_state
)
```

### Custom Constraints

Similarly, define constraints as functions returning lists of CVXPY constraints:

```python
def custom_constraints(variables, params):
    u = variables["u"]
    x = variables["x"]

    constraints = []

    # Custom constraint: sum of positive controls ≤ 2
    constraints.append(cp.sum(cp.maximum(u, 0)) <= 2)

    # Quadratic constraint on states
    constraints.append(cp.sum_squares(x) <= 5)

    return constraints

result = controller.compute_cvx_control(
    objective_fn=custom_objective,
    constraints_fn=custom_constraints
)
```

## Advanced Features

### Entropy-Aware Control

Integrate thermodynamic accounting into control design:

```python
from llrq import create_entropy_aware_cvx_controller

# Create controller with pre-computed entropy metric
L = np.eye(len(network.reaction_ids))  # Onsager conductance
controller = create_entropy_aware_cvx_controller(solver, controlled_reactions, L)

# Use entropy in multi-objective optimization
entropy_obj = CVXObjectives.multi_objective({
    'tracking': 1.0,
    'entropy': 0.1  # Penalize entropy production
})

result = controller.compute_cvx_control(
    objective_fn=entropy_obj,
    x_target=target_forces
)
```

### Variable Access

The `variables` dictionary provides access to all CVXPY variables:

```python
result = controller.compute_cvx_control(objective_fn=my_objective)

# Access variables after solving
print("Control variables:", result['variables']['u'].value)
print("State variables:", result['variables']['x'].value)
print("Reduced state:", result['variables']['y'].value)
```

### Solver Options

Pass options directly to CVXPY solvers:

```python
solver_options = {
    'solver': 'MOSEK',    # Specify solver
    'verbose': True,      # Enable verbose output
    'max_iters': 1000     # Solver-specific options
}

result = controller.compute_cvx_control(
    objective_fn=sparse_obj,
    solver_options=solver_options
)
```

## Complete Examples

### Example 1: Sparse Control with Bounds

```python
import llrq
import numpy as np
from llrq import CVXController, CVXObjectives, CVXConstraints

# Create system
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0)
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create controller
controller = CVXController(solver)

# Define sparse control objective
objective = CVXObjectives.sparse_control(sparsity_weight=0.1)

# Define box constraints
constraints = CVXConstraints.combine(
    CVXConstraints.box_bounds(u_min=-2, u_max=2),
    CVXConstraints.control_budget(total_budget=5.0, norm_type=1)
)

# Solve optimization problem
target = np.array([1.0, -1.0])  # Target reaction forces
result = controller.compute_cvx_control(
    objective_fn=objective,
    constraints_fn=constraints,
    x_target=target
)

print(f"Status: {result['status']}")
print(f"Optimal control: {result['u_optimal']}")
print(f"Objective value: {result['objective_value']}")
```

### Example 2: Multi-Objective with Custom Weights

```python
# Multi-objective optimization balancing tracking and efficiency
multi_obj = CVXObjectives.multi_objective({
    'tracking': 10.0,    # High priority on tracking
    'control': 1.0,      # Moderate control effort penalty
    'sparsity': 0.5      # Encourage sparse solutions
})

steady_state = CVXConstraints.steady_state()

result = controller.compute_cvx_control(
    objective_fn=multi_obj,
    constraints_fn=steady_state,
    x_target=np.array([0.5, -0.3])
)

# Simulate with optimal control
u_opt = result['u_optimal']
t_sim = np.linspace(0, 10, 100)
sol = solver.solve(t_sim, u=lambda t: u_opt)

print(f"Final reaction forces: {sol.x[-1]}")
```

## Integration with Other Methods

CVX control can be combined with other LLRQ capabilities:

```python
# Use with frequency control for periodic optimization
from llrq import FrequencySpaceController

freq_controller = FrequencySpaceController.from_llrq_solver(solver)
H = freq_controller.compute_frequency_response(omega=1.0)

# Use frequency response in custom CVX objective
def freq_aware_objective(variables, params):
    u = variables["u"]
    # Custom objective using frequency response
    return cp.sum_squares(H @ u)

result = controller.compute_cvx_control(objective_fn=freq_aware_objective)
```

## Error Handling

The `CVXController` provides informative error messages and warnings:

```python
result = controller.compute_cvx_control(
    objective_fn=problematic_objective,
    constraints_fn=infeasible_constraints
)

if result['status'] != 'optimal':
    print(f"Warning: Solver status is {result['status']}")

    if result['u_optimal'] is None:
        print("No feasible solution found")
    else:
        print("Using best available solution")
```

## Performance Tips

1. **Use appropriate solvers**: MOSEK for large problems, ECOS for smaller ones
2. **Warm starting**: Reuse `Problem` objects for similar optimization instances
3. **Problem scaling**: Use `solver_options` to adjust tolerances and scaling
4. **Sparsity patterns**: Design objectives to promote sparse solutions when appropriate

The CVXPY integration makes the LLRQ package incredibly flexible for control design, enabling everything from simple sparse control to complex multi-objective optimization with entropy awareness.
