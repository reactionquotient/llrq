---
layout: page
title: Thermodynamic Accounting
nav_order: 10
---

# Thermodynamic Accounting

The LLRQ package provides comprehensive thermodynamic accounting capabilities through the `ThermodynamicAccountant` class. This enables rigorous analysis of entropy production, energy balance diagnostics, and thermodynamic consistency in both LLRQ and mass action systems.

## Overview

Thermodynamic accounting in LLRQ systems is based on the fundamental relationship between reaction forces, external drives, and entropy production. The framework supports:

- **Entropy from Reaction Forces**: `σ(t) = x(t)ᵀ L x(t)` where `x = ln(Q/Keq)`
- **Entropy from External Drives**: Quasi-steady approximation `σ ≈ u(t)ᵀ L u(t)`
- **Energy Balance**: Power balance between relaxation, external work, and system storage
- **Dual Accounting**: Cross-validation using both reaction forces and drives

### Key Concepts

- **Reaction Forces**: `x(t) = ln(Q(t)/Keq)` measure deviation from equilibrium
- **Onsager Conductance**: Matrix `L` relating forces to fluxes
- **Entropy Production Rate**: `σ(t) = x(t)ᵀ L x(t)` (non-negative for PSD Onsager matrix)
- **Energy Balance**: `dV/dt = P_relax + P_control` where `V` is thermodynamic potential

## Quick Start

```python
import llrq
import numpy as np
from llrq import ThermodynamicAccountant

# Create LLRQ system
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0)
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)

# Define Onsager conductance matrix
L = np.array([[1.5]])  # Single reaction system

# Create thermodynamic accountant
accountant = ThermodynamicAccountant(network, onsager_conductance=L)

# Simulate system with external drive
t = np.linspace(0, 10, 200)
u_control = lambda t: 0.5 * np.sin(2*t)  # Sinusoidal drive

sol = solver.solve(t, u=u_control)

# Compute entropy production from reaction forces
entropy_result = accountant.entropy_from_x(t, sol.x)

print(f"Total entropy production: {entropy_result.sigma_total:.4f}")
print(f"Average entropy rate: {np.mean(entropy_result.sigma_time):.4f}")

# Dual accounting (reaction forces vs external drives)
u_trajectory = np.array([u_control(t_i) for t_i in t])
dual_result = accountant.dual_entropy_accounting(t, sol.x, u_trajectory)

print(f"Entropy from x: {dual_result.from_x.sigma_total:.4f}")
print(f"Entropy from u: {dual_result.from_u.sigma_total:.4f}")
print(f"Energy balance residual: {dual_result.balance['residual_integral']:.4f}")
```

## ThermodynamicAccountant Class

### Constructor

```python
ThermodynamicAccountant(network, onsager_conductance=None)
```

- **network**: ReactionNetwork instance
- **onsager_conductance**: Pre-computed Onsager matrix L (optional)

If no Onsager conductance is provided, it will be estimated from trajectory data when needed.

### Core Methods

#### Entropy from Reaction Forces

Compute entropy production from `x(t) = ln(Q/Keq)` trajectories:

```python
result = accountant.entropy_from_x(
    t,                    # Time points (T,)
    x,                    # Reaction forces (T, n_reactions)
    L=None,              # Onsager conductance (uses self.L if None)
    scale=1.0,           # Scale factor (e.g., kB for physical units)
    psd_clip=0.0         # Clip negative eigenvalues to ensure PSD
)
```

Returns `AccountingResult` with:
- `sigma_time`: Instantaneous entropy production rate `σ(t)`
- `sigma_total`: Total entropy production `∫σ(t)dt`
- `notes`: Optional warnings or comments

#### Entropy from External Drives

Compute entropy using quasi-steady approximation `x ≈ K⁻¹u`:

```python
result = accountant.entropy_from_u_quasi_steady(
    t,                    # Time points (T,)
    u,                    # External drives (T, n_reactions)
    K,                    # Dynamics matrix (n_reactions, n_reactions)
    L=None,              # Onsager conductance matrix
    scale=1.0            # Scale factor
)
```

This method assumes the system is in quasi-steady state where reaction forces instantaneously balance external drives.

#### Dual Entropy Accounting

Cross-validate entropy calculations and perform energy balance diagnostics:

```python
dual_result = accountant.dual_entropy_accounting(
    t,                    # Time points (T,)
    x,                    # Reaction forces (T, n_reactions)
    u,                    # External drives (T, n_reactions)
    K=None,              # System matrix (estimated if None)
    L=None,              # Onsager conductance (estimated if None)
    scale=1.0            # Scale factor
)
```

Returns `DualAccountingResult` with:
- `from_x`: Entropy accounting from reaction forces
- `from_u`: Entropy accounting from external drives (quasi-steady)
- `balance`: Energy balance diagnostics dictionary

## Energy Balance Diagnostics

The energy balance analysis decomposes power flows in the system:

```python
# Access energy balance components
balance = dual_result.balance

print(f"System potential rate: {balance['V_dot_integral']:.4f}")
print(f"Relaxation power: {balance['P_relax_integral']:.4f}")
print(f"Control power: {balance['P_ctrl_integral']:.4f}")
print(f"Balance residual: {balance['residual_integral']:.4f}")

# Plot energy balance over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, balance['V_dot'], label='dV/dt')
plt.ylabel('Potential Rate')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, balance['P_relax'], label='P_relax', color='red')
plt.ylabel('Relaxation Power')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, balance['P_ctrl'], label='P_ctrl', color='green')
plt.ylabel('Control Power')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, balance['residual'], label='Residual', color='orange')
plt.ylabel('Balance Residual')
plt.xlabel('Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Energy Balance Interpretation

- **V_dot**: Rate of change of thermodynamic potential `V = ½xᵀKx`
- **P_relax**: Power dissipated by relaxation to equilibrium (`-xᵀKx`)
- **P_ctrl**: Power supplied by external control (`xᵀu`)
- **Residual**: `dV/dt - P_relax - P_ctrl` (should be ≈0 for consistent dynamics)

## Advanced Features

### Frequency-Domain Entropy Analysis

Analyze entropy production in frequency domain using FFT:

```python
def frequency_entropy_analysis(t, x, L, max_freq=None):
    """Analyze entropy production spectrum."""
    from scipy.fft import fft, fftfreq

    # FFT of reaction forces
    dt = t[1] - t[0]
    freqs = fftfreq(len(t), dt)
    x_fft = fft(x, axis=0)

    if max_freq is not None:
        mask = np.abs(freqs) <= max_freq
        freqs = freqs[mask]
        x_fft = x_fft[mask]

    # Entropy spectrum (approximate)
    entropy_spectrum = np.real(np.einsum('fi,ij,fj->f',
                                        x_fft.conj(), L, x_fft))

    return freqs, entropy_spectrum

# Example usage
freqs, entropy_spectrum = frequency_entropy_analysis(t, sol.x, L, max_freq=2.0)

plt.figure(figsize=(10, 6))
plt.semilogy(freqs[freqs >= 0], entropy_spectrum[freqs >= 0])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Entropy Spectrum')
plt.title('Frequency Domain Entropy Analysis')
plt.grid(True)
plt.show()
```

### Onsager Matrix Estimation

Estimate Onsager conductance from trajectory data:

```python
def estimate_onsager_matrix(t, x, method='correlation'):
    """Estimate Onsager conductance matrix from trajectories."""

    if method == 'correlation':
        # Simple correlation-based estimate
        dx_dt = np.gradient(x, t, axis=0)

        # Regression: dx/dt ≈ -Kx, so K ≈ -(dx/dt)ᵀx / (xᵀx)
        X = x[:-1]  # Remove last point due to gradient
        dX_dt = dx_dt[:-1]

        # Least squares: K = argmin ||dX_dt + KX||²
        K_est = -np.linalg.lstsq(X, dX_dt, rcond=None)[0].T

        # Symmetrize and use as Onsager matrix estimate
        L_est = 0.5 * (K_est + K_est.T)

    elif method == 'fluctuation':
        # Fluctuation-dissipation based estimate
        # More sophisticated method using equilibrium fluctuations
        cov_x = np.cov(x.T)
        L_est = np.linalg.pinv(cov_x)

    else:
        raise ValueError(f"Unknown estimation method: {method}")

    # Ensure positive semidefinite
    eigenvals, eigenvecs = np.linalg.eigh(L_est)
    eigenvals = np.maximum(eigenvals, 1e-12)  # Clip small negative values
    L_est = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    return L_est

# Estimate Onsager matrix from simulation data
L_estimated = estimate_onsager_matrix(t, sol.x)
print(f"True L: {L}")
print(f"Estimated L: {L_estimated}")

# Compare entropy calculations
entropy_true = accountant.entropy_from_x(t, sol.x, L=L)
entropy_estimated = accountant.entropy_from_x(t, sol.x, L=L_estimated)

print(f"Entropy (true L): {entropy_true.sigma_total:.4f}")
print(f"Entropy (estimated L): {entropy_estimated.sigma_total:.4f}")
```

### Integration with Control Systems

Use thermodynamic accounting to design entropy-aware control:

```python
from llrq import CVXController, CVXObjectives

# Create CVX controller with entropy awareness
controller = CVXController(solver)

def entropy_penalized_objective(entropy_weight=0.1):
    """Create objective that penalizes entropy production."""

    def objective(variables, params):
        import cvxpy as cp
        u = variables["u"]
        x = variables["x"] if "x" in variables else params["K"]**(-1) @ u

        # Standard tracking objective
        if "x_target" in params and params["x_target"] is not None:
            tracking_cost = cp.sum_squares(x - params["x_target"])
        else:
            tracking_cost = 0

        # Entropy penalty: x^T L x
        if "L" in params:
            entropy_cost = cp.quad_form(x, params["L"])
        else:
            entropy_cost = 0

        return tracking_cost + entropy_weight * entropy_cost

    return objective

# Use entropy-penalized control
entropy_obj = entropy_penalized_objective(entropy_weight=0.05)

result = controller.compute_cvx_control(
    objective_fn=entropy_obj,
    x_target=np.array([1.0, -0.5]),
    L=L,  # Pass Onsager matrix
    K=dynamics.K
)

print(f"Entropy-aware control: {result['u_optimal']}")
```

## Complete Examples

### Example 1: Comparing Control Strategies

```python
import llrq
import numpy as np
import matplotlib.pyplot as plt
from llrq import ThermodynamicAccountant

# Create two-reaction network
network = llrq.ReactionNetwork()
network.add_reaction_from_string("R1", "A <-> B", k_forward=2.0, k_backward=1.0)
network.add_reaction_from_string("R2", "B <-> C", k_forward=1.5, k_backward=0.8)

dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)

# Define Onsager conductance
L = np.array([[1.2, 0.1], [0.1, 0.9]])

# Create accountant
accountant = ThermodynamicAccountant(network, L)

# Compare different control strategies
t = np.linspace(0, 15, 300)
strategies = {
    'No Control': lambda t: np.zeros(2),
    'Constant Drive': lambda t: np.array([0.5, -0.3]),
    'Sinusoidal Drive': lambda t: np.array([0.5*np.sin(0.5*t), -0.3*np.cos(0.7*t)]),
    'Square Wave': lambda t: np.array([0.5*np.sign(np.sin(0.3*t)), -0.3*np.sign(np.cos(0.4*t))])
}

results = {}
entropy_totals = {}

for name, u_func in strategies.items():
    # Simulate
    sol = solver.solve(t, u=u_func)

    # Compute entropy
    entropy_result = accountant.entropy_from_x(t, sol.x)

    results[name] = sol
    entropy_totals[name] = entropy_result.sigma_total

    print(f"{name}: Total Entropy = {entropy_result.sigma_total:.4f}")

# Plot comparison
plt.figure(figsize=(15, 10))

for i, (name, sol) in enumerate(results.items()):
    plt.subplot(2, 3, i+1)
    plt.plot(t, sol.x[:, 0], label='x₁', linewidth=2)
    plt.plot(t, sol.x[:, 1], label='x₂', linewidth=2)
    plt.title(f'{name}\nEntropy = {entropy_totals[name]:.3f}')
    plt.xlabel('Time')
    plt.ylabel('Reaction Forces')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Bar chart of entropy production
plt.subplot(2, 3, 6)
names = list(entropy_totals.keys())
entropies = list(entropy_totals.values())
bars = plt.bar(range(len(names)), entropies, alpha=0.7)
plt.xticks(range(len(names)), names, rotation=45)
plt.ylabel('Total Entropy Production')
plt.title('Entropy Comparison')
plt.grid(True, alpha=0.3)

# Color bars by entropy level
colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.tight_layout()
plt.show()
```

### Example 2: Mass Action vs LLRQ Thermodynamics

```python
from llrq import MassActionSimulator

# Create network for comparison
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0,
                              concentrations=[2.0, 0.5])

# LLRQ setup
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)

# Mass action setup
mass_action = MassActionSimulator(network)

# Onsager matrix
L = np.array([[1.0]])

# Accountants for both methods
llrq_accountant = ThermodynamicAccountant(network, L)

# External drive
def sinusoidal_drive(t):
    return np.array([0.3 * np.sin(0.8 * t)])

t = np.linspace(0, 20, 400)

# Simulate both methods
print("Simulating LLRQ...")
sol_llrq = solver.solve(t, u=sinusoidal_drive)

print("Simulating Mass Action...")
sol_mass = mass_action.solve(t, u=sinusoidal_drive)

# Convert mass action to reaction forces for thermodynamic analysis
concentrations_mass = sol_mass.y
Q_mass = concentrations_mass[:, 0] / concentrations_mass[:, 1]  # [A]/[B]
Keq = dynamics.equilibrium_constants[0]
x_mass = np.log(Q_mass / Keq).reshape(-1, 1)

# Thermodynamic analysis
entropy_llrq = llrq_accountant.entropy_from_x(t, sol_llrq.x)
entropy_mass = llrq_accountant.entropy_from_x(t, x_mass)

print(f"\nThermodynamic Comparison:")
print(f"LLRQ Total Entropy: {entropy_llrq.sigma_total:.4f}")
print(f"Mass Action Total Entropy: {entropy_mass.sigma_total:.4f}")
print(f"Relative Difference: {abs(entropy_llrq.sigma_total - entropy_mass.sigma_total) / entropy_mass.sigma_total * 100:.2f}%")

# Plot detailed comparison
plt.figure(figsize=(16, 12))

plt.subplot(3, 2, 1)
plt.plot(t, sol_llrq.x[:, 0], label='LLRQ', linewidth=2)
plt.plot(t, x_mass[:, 0], label='Mass Action', linestyle='--', linewidth=2)
plt.ylabel('Reaction Force x')
plt.title('Reaction Forces Comparison')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t, entropy_llrq.sigma_time, label='LLRQ', linewidth=2)
plt.plot(t, entropy_mass.sigma_time, label='Mass Action', linestyle='--', linewidth=2)
plt.ylabel('Entropy Rate σ(t)')
plt.title('Entropy Production Rate')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(t, sol_llrq.concentrations[:, 0], label='LLRQ [A]', linewidth=2)
plt.plot(t, concentrations_mass[:, 0], label='Mass Action [A]', linestyle='--', linewidth=2)
plt.ylabel('Concentration [A]')
plt.title('Species A Concentration')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t, sol_llrq.concentrations[:, 1], label='LLRQ [B]', linewidth=2)
plt.plot(t, concentrations_mass[:, 1], label='Mass Action [B]', linestyle='--', linewidth=2)
plt.ylabel('Concentration [B]')
plt.title('Species B Concentration')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
drive_values = np.array([sinusoidal_drive(t_i)[0] for t_i in t])
plt.plot(t, drive_values, color='red', linewidth=2)
plt.ylabel('External Drive u(t)')
plt.xlabel('Time')
plt.title('Control Input')
plt.grid(True)

plt.subplot(3, 2, 6)
# Cumulative entropy comparison
entropy_cumulative_llrq = np.cumsum(entropy_llrq.sigma_time) * (t[1] - t[0])
entropy_cumulative_mass = np.cumsum(entropy_mass.sigma_time) * (t[1] - t[0])

plt.plot(t, entropy_cumulative_llrq, label='LLRQ', linewidth=2)
plt.plot(t, entropy_cumulative_mass, label='Mass Action', linestyle='--', linewidth=2)
plt.ylabel('Cumulative Entropy')
plt.xlabel('Time')
plt.title('Cumulative Entropy Production')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Performance and Best Practices

### Computational Efficiency

1. **Pre-compute Onsager Matrix**: Store L matrix to avoid repeated estimation
2. **Use Vectorized Operations**: Leverage numpy's einsum for entropy calculations
3. **Frequency Analysis**: Use FFT for spectral entropy analysis on large datasets
4. **Memory Management**: Process long trajectories in chunks for large-scale analysis

### Physical Interpretation

1. **Units**: Use appropriate scale factors (kB, R) for physical entropy units
2. **Sign Conventions**: Ensure Onsager matrix is positive semidefinite for physical consistency
3. **Equilibrium Reference**: Reaction forces x = ln(Q/Keq) require proper equilibrium constants
4. **Quasi-Steady Validity**: Check |dx/dt| << |Kx| for quasi-steady approximation

### Numerical Considerations

1. **Matrix Conditioning**: Use psd_clip parameter to handle near-singular Onsager matrices
2. **Time Resolution**: Ensure sufficient sampling for accurate trapezoidal integration
3. **Steady State**: Allow sufficient simulation time to reach quasi-steady behavior
4. **Balance Residuals**: Monitor energy balance residuals as model consistency check

The thermodynamic accounting framework provides deep insights into the energetic costs and efficiency of control strategies in chemical reaction networks, making it invaluable for both theoretical analysis and practical control design.
