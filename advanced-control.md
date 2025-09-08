---
layout: page
title: Advanced Control Features
nav_order: 11
---

# Advanced Control Features

The LLRQ package provides sophisticated control capabilities beyond basic optimization, including analytical control methods, adaptive parameter estimation, integrated simulation workflows, and control comparison utilities. These advanced features enable comprehensive control system design and analysis.

## Overview

Advanced control features include:

- **LLRQController**: Base analytical control methods with LQR design
- **AdaptiveController**: Real-time parameter adaptation and learning
- **ControlledSimulation**: Integrated workflow for control simulation and analysis
- **Control Comparison**: Utilities for comparing different control strategies
- **LQR Design**: Linear Quadratic Regulator for optimal control
- **Simulation Integration**: Seamless integration with LLRQ dynamics

## LLRQController Base Class

The `LLRQController` provides foundational control methods and serves as the base for specialized controllers.

### Basic Usage

```python
import llrq
import numpy as np
from llrq import LLRQController

# Create LLRQ system
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0)
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create controller
controller = LLRQController(solver, controlled_reactions=["R1"])

print(f"Controlled reactions: {controller.controlled_reactions}")
print(f"Control matrix G shape: {controller.G.shape}")
print(f"System rank: {controller.rankS}")
```

### Key Properties

The `LLRQController` automatically computes essential control matrices:

```python
# Control selection matrix (maps controlled to all reactions)
G = controller.G  # Shape: (n_reactions, n_controlled)

# Reduced system matrices in controllable subspace
B_reduced = controller.B  # Shape: (rankS, n_controlled)
A_reduced = controller.A  # Shape: (rankS, rankS)

# Basis matrices for reduced space
V = solver.basis.V  # Left nullspace basis of S^T
U = solver.basis.U  # Right nullspace basis of S

print(f"Reduced system dimensions: {controller.rankS}")
print(f"Full system dimensions: {len(network.reaction_ids)}")
```

### Control Matrix Computation

The controller provides methods for computing control-related matrices:

```python
# Control entropy metric for thermodynamic-aware control
L = np.eye(len(network.reaction_ids))  # Onsager conductance matrix
M = controller.compute_control_entropy_metric(L)

print(f"Entropy metric M shape: {M.shape}")
print(f"Is M positive semidefinite: {np.all(np.linalg.eigvals(M) >= 0)}")

# Use in quadratic cost: cost = u^T M u
control_input = np.array([0.5])
entropy_cost = control_input.T @ M @ control_input
print(f"Entropy cost for u={control_input}: {entropy_cost}")
```

## LQR Control Design

Linear Quadratic Regulator (LQR) design for optimal control of LLRQ systems.

### Basic LQR

```python
from llrq.control.lqr import lqr_control

# Design LQR controller for tracking
Q = np.eye(controller.rankS)  # State cost matrix
R = np.eye(controller.G.shape[1])  # Control cost matrix

K_lqr, P, eigenvals = lqr_control(controller.A, controller.B, Q, R)

print(f"LQR gain matrix K shape: {K_lqr.shape}")
print(f"Closed-loop eigenvalues: {eigenvals}")
print(f"Is system stable: {np.all(np.real(eigenvals) < 0)}")

# Use LQR for tracking control
def lqr_tracking_control(t, x_current, x_target):
    """LQR tracking controller."""
    # Convert to reduced coordinates if needed
    y_current = controller.solver.basis.V.T @ x_current
    y_target = controller.solver.basis.V.T @ x_target if x_target is not None else np.zeros(controller.rankS)

    # LQR control law: u = -K(y - y_target)
    error = y_current - y_target
    u_controlled = -K_lqr @ error

    return controller.G @ u_controlled

# Simulate with LQR control
target_forces = np.array([1.0, -0.5])
t = np.linspace(0, 10, 200)

def control_function(t):
    # For time-varying control, you'd solve ODE and use current state
    # Here we use steady-state target for simplicity
    return lqr_tracking_control(t, np.zeros(len(network.reaction_ids)), target_forces)

sol = solver.solve(t, u=control_function)

print(f"Final reaction forces: {sol.x[-1]}")
print(f"Tracking error: {np.linalg.norm(sol.x[-1] - target_forces)}")
```

### Weighted LQR

Design LQR with different state and control weights:

```python
# Emphasize tracking of first reaction more than second
Q_weighted = np.diag([10.0, 1.0])  # Higher weight on first state

# Penalize control effort
R_weighted = 2.0 * np.eye(controller.G.shape[1])

K_weighted, P_weighted, eigs_weighted = lqr_control(
    controller.A, controller.B, Q_weighted, R_weighted
)

print(f"Standard LQR gains: {K_lqr}")
print(f"Weighted LQR gains: {K_weighted}")
print(f"Weighted closed-loop eigenvalues: {eigs_weighted}")

# Compare control effort
u_standard = K_lqr @ np.array([1.0, 0.5])  # Example error
u_weighted = K_weighted @ np.array([1.0, 0.5])

print(f"Standard control effort: {np.linalg.norm(u_standard)}")
print(f"Weighted control effort: {np.linalg.norm(u_weighted)}")
```

## ControlledSimulation Workflow

The `ControlledSimulation` class provides an integrated workflow for control simulation, analysis, and comparison.

### Basic Controlled Simulation

```python
from llrq import ControlledSimulation

# Create controlled simulation
controlled_sim = ControlledSimulation(solver)

# Define control strategies
strategies = {
    'open_loop': lambda t, x: np.array([0.3 * np.sin(0.5 * t)]),
    'proportional': lambda t, x: -0.5 * x,  # Proportional feedback
    'lqr': lambda t, x: lqr_tracking_control(t, x, target_forces)
}

# Run simulation comparison
t_sim = np.linspace(0, 15, 300)
results = controlled_sim.compare_strategies(strategies, t_sim)

# Analyze results
for name, result in results.items():
    final_error = np.linalg.norm(result.x[-1] - target_forces)
    control_effort = np.trapz(np.linalg.norm(result.control_history, axis=1)**2, t_sim)

    print(f"{name}:")
    print(f"  Final tracking error: {final_error:.4f}")
    print(f"  Control effort: {control_effort:.4f}")
    print(f"  Final reaction forces: {result.x[-1]}")
    print()
```

### Advanced Simulation Features

```python
# Simulation with disturbances and noise
def simulate_with_disturbance(solver, control_func, t, disturbance_level=0.1):
    """Simulate with process noise and measurement noise."""

    # Add process noise to the dynamics
    def noisy_dynamics(t, x, u_func):
        u_nominal = u_func(t, x)
        process_noise = disturbance_level * np.random.randn(len(x))
        return -solver.dynamics.K @ x + u_nominal + process_noise

    # Custom integration with noise (simplified)
    dt = t[1] - t[0]
    x_trajectory = np.zeros((len(t), len(solver.network.reaction_ids)))
    x_current = np.zeros(len(solver.network.reaction_ids))  # Initial condition

    for i, t_i in enumerate(t):
        x_trajectory[i] = x_current
        if i < len(t) - 1:
            # Euler integration with noise
            u_current = strategies['lqr'](t_i, x_current)
            dx_dt = noisy_dynamics(t_i, x_current, lambda t, x: u_current)
            x_current = x_current + dt * dx_dt

    return x_trajectory

# Compare robustness to disturbances
print("Robustness Analysis:")
x_nominal = results['lqr'].x
x_disturbed = simulate_with_disturbance(solver, strategies['lqr'], t_sim)

nominal_error = np.linalg.norm(x_nominal[-1] - target_forces)
disturbed_error = np.linalg.norm(x_disturbed[-1] - target_forces)

print(f"Nominal final error: {nominal_error:.4f}")
print(f"Disturbed final error: {disturbed_error:.4f}")
print(f"Robustness ratio: {disturbed_error / nominal_error:.2f}")
```

## AdaptiveController

The `AdaptiveController` provides real-time parameter estimation and adaptation capabilities.

### Parameter Estimation

```python
from llrq import AdaptiveController

# Create adaptive controller
adaptive_controller = AdaptiveController(solver)

# Parameter estimation from trajectory data
t_data = np.linspace(0, 10, 100)
u_data = np.array([0.2 * np.sin(0.3 * t) for t in t_data]).reshape(-1, 1)
sol_data = solver.solve(t_data, u=lambda t: u_data[np.argmin(np.abs(t_data - t))])

# Estimate system parameters
estimated_params = adaptive_controller.estimate_parameters(
    t_data, sol_data.x, u_data
)

print("Parameter Estimation Results:")
print(f"Estimated K matrix: \n{estimated_params['K']}")
print(f"True K matrix: \n{dynamics.K}")
print(f"Estimation error: {np.linalg.norm(estimated_params['K'] - dynamics.K)}")

# Adaptation metrics
print(f"Parameter confidence: {estimated_params.get('confidence', 'N/A')}")
print(f"Adaptation rate: {estimated_params.get('adaptation_rate', 'N/A')}")
```

### Real-Time Adaptation

```python
class RealTimeAdaptiveController:
    """Real-time adaptive controller with online parameter updates."""

    def __init__(self, solver, initial_guess=None, adaptation_rate=0.01):
        self.solver = solver
        self.K_estimate = initial_guess or np.eye(len(solver.network.reaction_ids))
        self.adaptation_rate = adaptation_rate
        self.history = {'K': [], 'error': [], 'time': []}

    def update_parameters(self, t, x, u, x_predicted):
        """Update parameter estimates based on prediction error."""
        prediction_error = x - x_predicted

        # Simple gradient descent update (simplified)
        gradient = np.outer(prediction_error, x)
        self.K_estimate -= self.adaptation_rate * gradient

        # Store history
        self.history['K'].append(self.K_estimate.copy())
        self.history['error'].append(np.linalg.norm(prediction_error))
        self.history['time'].append(t)

    def adaptive_control(self, t, x, x_target):
        """Compute control using current parameter estimates."""
        if x_target is not None:
            error = x - x_target
            # Simple proportional control with adapted parameters
            u = -np.linalg.pinv(self.K_estimate) @ error
        else:
            u = np.zeros(len(self.solver.network.reaction_ids))

        return u

    def predict_next_state(self, x, u, dt):
        """Predict next state using current parameter estimates."""
        dx_dt = -self.K_estimate @ x + u
        return x + dt * dx_dt

# Example of real-time adaptation
adaptive = RealTimeAdaptiveController(solver, adaptation_rate=0.005)

# Simulate with parameter adaptation
t_adaptive = np.linspace(0, 20, 400)
dt = t_adaptive[1] - t_adaptive[0]
x_adaptive = np.zeros((len(t_adaptive), len(solver.network.reaction_ids)))
u_adaptive = np.zeros((len(t_adaptive), len(solver.network.reaction_ids)))

x_current = np.array([0.1, -0.1])  # Initial condition
target = np.array([1.0, -0.5])

for i, t in enumerate(t_adaptive):
    x_adaptive[i] = x_current

    # Compute control
    u_current = adaptive.adaptive_control(t, x_current, target)
    u_adaptive[i] = u_current

    # Predict next state
    if i < len(t_adaptive) - 1:
        x_predicted = adaptive.predict_next_state(x_current, u_current, dt)

        # True dynamics (what actually happens)
        x_next = x_current + dt * (-dynamics.K @ x_current + u_current)

        # Update parameters based on prediction error
        adaptive.update_parameters(t, x_next, u_current, x_predicted)

        x_current = x_next

print(f"Final parameter estimate:\n{adaptive.K_estimate}")
print(f"True parameters:\n{dynamics.K}")
print(f"Final adaptation error: {np.linalg.norm(adaptive.K_estimate - dynamics.K)}")
```

## Convenience Functions

The LLRQ package provides high-level convenience functions for common control tasks.

### simulate_to_target

Automatically design and simulate control to reach a target state:

```python
from llrq import simulate_to_target

# Simple target reaching
target_state = np.array([0.8, -0.3])
result = simulate_to_target(
    solver=solver,
    target=target_state,
    method='lqr',           # Control method
    time_horizon=10,        # Simulation time
    tolerance=0.01          # Convergence tolerance
)

print(f"Target reached: {result['converged']}")
print(f"Final error: {result['final_error']:.4f}")
print(f"Convergence time: {result['convergence_time']:.2f}s")
print(f"Control effort: {result['control_effort']:.4f}")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(result['time'], result['trajectory'][:, 0], label='x₁')
plt.plot(result['time'], result['trajectory'][:, 1], label='x₂')
plt.axhline(target_state[0], color='red', linestyle='--', alpha=0.7, label='target x₁')
plt.axhline(target_state[1], color='blue', linestyle='--', alpha=0.7, label='target x₂')
plt.ylabel('Reaction Forces')
plt.legend()
plt.grid(True)
plt.title('State Trajectory')

plt.subplot(2, 2, 2)
plt.plot(result['time'], result['control'])
plt.ylabel('Control Input')
plt.title('Control Signal')
plt.grid(True)

plt.subplot(2, 2, 3)
error_norm = np.linalg.norm(result['trajectory'] - target_state, axis=1)
plt.semilogy(result['time'], error_norm)
plt.ylabel('Tracking Error (log scale)')
plt.xlabel('Time')
plt.title('Convergence')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(result['trajectory'][:, 0], result['trajectory'][:, 1], alpha=0.7)
plt.plot(target_state[0], target_state[1], 'ro', markersize=10, label='Target')
plt.plot(result['trajectory'][0, 0], result['trajectory'][0, 1], 'go', markersize=8, label='Start')
plt.xlabel('Reaction Force x₁')
plt.ylabel('Reaction Force x₂')
plt.legend()
plt.title('Phase Portrait')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### compare_control_methods

Compare different control strategies systematically:

```python
from llrq import compare_control_methods

# Define methods to compare
methods = {
    'proportional': {
        'type': 'feedback',
        'gain': 0.5
    },
    'lqr': {
        'type': 'lqr',
        'Q': np.eye(controller.rankS),
        'R': np.eye(controller.G.shape[1])
    },
    'cvx_sparse': {
        'type': 'cvx',
        'objective': 'sparse',
        'sparsity_weight': 0.1
    }
}

comparison_result = compare_control_methods(
    solver=solver,
    methods=methods,
    target=target_state,
    time_horizon=12,
    metrics=['tracking_error', 'control_effort', 'convergence_time', 'robustness']
)

# Display comparison results
print("Control Method Comparison:")
print("=" * 50)

for method, metrics in comparison_result.items():
    print(f"\n{method.upper()}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

# Generate comparison plots
comparison_result.plot_comparison()
```

## Integration with Thermodynamic Accounting

Combine advanced control with entropy analysis:

```python
from llrq import ThermodynamicAccountant

# Set up entropy accounting
L = np.eye(len(network.reaction_ids))
accountant = ThermodynamicAccountant(network, L)

def entropy_aware_lqr(Q_tracking, R_control, entropy_weight=0.1):
    """Design LQR with entropy penalty."""

    # Entropy metric in control space
    M_entropy = controller.compute_control_entropy_metric(L)

    # Modified control cost includes entropy
    R_modified = R_control + entropy_weight * M_entropy

    K_entropy_lqr, P_entropy, eigs_entropy = lqr_control(
        controller.A, controller.B, Q_tracking, R_modified
    )

    return K_entropy_lqr, P_entropy, eigs_entropy

# Compare standard vs entropy-aware LQR
K_standard, _, _ = lqr_control(controller.A, controller.B, np.eye(controller.rankS), np.eye(controller.G.shape[1]))
K_entropy, _, _ = entropy_aware_lqr(np.eye(controller.rankS), np.eye(controller.G.shape[1]), entropy_weight=0.05)

print(f"Standard LQR gains: {K_standard}")
print(f"Entropy-aware LQR gains: {K_entropy}")

# Simulate both and compare entropy production
def simulate_and_analyze_entropy(K_lqr, label):
    def lqr_control_func(t):
        # Simplified for constant target
        y_error = -controller.solver.basis.V.T @ target_state  # Assuming zero initial state
        u_controlled = -K_lqr @ y_error
        return controller.G @ u_controlled

    sol = solver.solve(t_sim, u=lqr_control_func)
    entropy_result = accountant.entropy_from_x(t_sim, sol.x)

    return {
        'solution': sol,
        'entropy_total': entropy_result.sigma_total,
        'entropy_rate': entropy_result.sigma_time,
        'final_error': np.linalg.norm(sol.x[-1] - target_state),
        'control_effort': np.trapz(np.linalg.norm([lqr_control_func(t) for t in t_sim], axis=1)**2, t_sim)
    }

results_standard = simulate_and_analyze_entropy(K_standard, "Standard LQR")
results_entropy = simulate_and_analyze_entropy(K_entropy, "Entropy-Aware LQR")

print(f"\nComparison Results:")
print(f"Standard LQR - Entropy: {results_standard['entropy_total']:.4f}, Error: {results_standard['final_error']:.4f}")
print(f"Entropy-Aware LQR - Entropy: {results_entropy['entropy_total']:.4f}, Error: {results_entropy['final_error']:.4f}")
print(f"Entropy Reduction: {(results_standard['entropy_total'] - results_entropy['entropy_total'])/results_standard['entropy_total']*100:.1f}%")
```

## Performance Optimization

### Efficient Control Computation

```python
class OptimizedController:
    """Optimized controller with pre-computed matrices and caching."""

    def __init__(self, solver, controlled_reactions=None):
        self.solver = solver
        self.base_controller = LLRQController(solver, controlled_reactions)

        # Pre-compute commonly used matrices
        self._precompute_matrices()

        # Caching for expensive operations
        self._lqr_cache = {}

    def _precompute_matrices(self):
        """Pre-compute matrices for efficient control computation."""
        self.K_inv = np.linalg.pinv(self.solver.dynamics.K)
        self.V = self.solver.basis.V
        self.VT = self.V.T

        # Pre-factorize for fast solving
        try:
            from scipy.linalg import lu_factor
            self.K_lu = lu_factor(self.solver.dynamics.K)
            self._use_lu = True
        except ImportError:
            self._use_lu = False

    def fast_steady_state_control(self, x_target):
        """Fast computation of steady-state control."""
        if self._use_lu:
            from scipy.linalg import lu_solve
            u_ss = lu_solve(self.K_lu, x_target)
        else:
            u_ss = self.K_inv @ x_target

        return u_ss

    def cached_lqr_design(self, Q, R):
        """LQR design with caching."""
        # Create cache key (simplified - in practice, need proper hashing)
        cache_key = (Q.shape, R.shape, np.sum(Q), np.sum(R))

        if cache_key not in self._lqr_cache:
            K, P, eigs = lqr_control(self.base_controller.A, self.base_controller.B, Q, R)
            self._lqr_cache[cache_key] = (K, P, eigs)

        return self._lqr_cache[cache_key]

# Example of optimized controller usage
optimized = OptimizedController(solver)

# Benchmark control computation speed
import time

# Standard approach
start = time.time()
for _ in range(1000):
    u_std = np.linalg.solve(dynamics.K, target_state)
std_time = time.time() - start

# Optimized approach
start = time.time()
for _ in range(1000):
    u_opt = optimized.fast_steady_state_control(target_state)
opt_time = time.time() - start

print(f"Standard computation time: {std_time:.4f}s")
print(f"Optimized computation time: {opt_time:.4f}s")
print(f"Speedup factor: {std_time/opt_time:.2f}x")
print(f"Results match: {np.allclose(u_std, u_opt)}")
```

The advanced control features provide a comprehensive toolkit for sophisticated control system design in LLRQ systems, enabling everything from basic feedback control to adaptive systems with thermodynamic awareness.
