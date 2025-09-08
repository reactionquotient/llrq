---
layout: page
title: Frequency-Domain Control Design
nav_order: 9
---

# Frequency-Domain Control Design

The LLRQ package provides sophisticated frequency-domain control capabilities through the `FrequencySpaceController` class. This enables design of sinusoidal control inputs to achieve target periodic steady states, with applications in oscillatory control, energy harvesting, and biological rhythms.

## Overview

Frequency-domain control leverages the linear nature of LLRQ dynamics to design optimal sinusoidal inputs. For the system `ẋ = -Kx + Bu(t)` with sinusoidal input `u(t) = Re{U e^(iωt)}`, the steady-state response is `x_ss(t) = Re{H(iω)U e^(iωt)}` where `H(iω) = (K + iωI)^(-1)B` is the frequency response matrix.

### Key Features

- **Frequency Response**: Compute `H(iω)` for any frequency
- **Optimal Sinusoidal Control**: Design `U` via weighted least squares with regularization
- **Entropy-Aware Design**: Incorporate thermodynamic costs via entropy kernels
- **Bode/Nyquist Analysis**: Classical frequency response analysis
- **Integration with LLRQ**: Direct construction from LLRQ solvers

## Quick Start

```python
import llrq
import numpy as np
from llrq import FrequencySpaceController

# Create LLRQ system
network = llrq.simple_reaction("A <-> B", k_forward=2.0, k_backward=1.0)
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create frequency controller from LLRQ solver
freq_controller = FrequencySpaceController.from_llrq_solver(
    solver, controlled_reactions=["R1"]
)

# Design sinusoidal control for target oscillation
omega = 1.0  # 1 rad/s
X_target = np.array([1.0, -1.0]) * np.exp(1j * 0.5)  # Phase-shifted target

U_optimal = freq_controller.design_sinusoidal_control(
    X_target=X_target,
    omega=omega,
    lam=0.01  # Regularization
)

print(f"Optimal control amplitude: {U_optimal}")
print(f"Control magnitude: {np.abs(U_optimal)}")
print(f"Control phase: {np.angle(U_optimal)}")
```

## FrequencySpaceController Class

### Constructor

```python
FrequencySpaceController(K, B)
```

- **K**: System matrix (n × n) from `ẋ = -Kx + Bu`
- **B**: Input matrix (n × m) from `ẋ = -Kx + Bu`

### Factory Method

The recommended way to create a frequency controller is from an LLRQ solver:

```python
freq_controller = FrequencySpaceController.from_llrq_solver(
    solver,
    controlled_reactions=None  # List of reaction IDs/indices, None for all
)
```

This automatically extracts the reduced system matrices and handles reaction selection.

## Core Methods

### Frequency Response

Compute the frequency response matrix `H(iω) = (K + iωI)^(-1)B`:

```python
omega = 2.0  # Frequency in rad/s
H = freq_controller.compute_frequency_response(omega)

print(f"Frequency response shape: {H.shape}")
print(f"DC gain (ω=0): {freq_controller.compute_frequency_response(0.0)}")
```

### Sinusoidal Control Design

Design optimal sinusoidal control inputs using weighted least squares:

```python
U_optimal = freq_controller.design_sinusoidal_control(
    X_target,           # Target complex amplitude (n,)
    omega,              # Frequency in rad/s
    W=None,             # Weight matrix (n × n), default: identity
    lam=0.01           # Regularization parameter
)
```

**Mathematical Formulation:**
The method solves: `U* = argmin ||WH(iω)U - WX*||² + λ||U||²`

Where:
- `H(iω)` is the frequency response matrix
- `W` is a weighting matrix (emphasizes certain states)
- `λ` is regularization (prevents ill-conditioning)

### Steady-State Evaluation

Evaluate the steady-state response for given control:

```python
# Complex amplitude
X_ss = freq_controller.evaluate_steady_state(U_optimal, omega)

# Time-domain response at specific time
t = 2.5
x_t = freq_controller.evaluate_response_at_time(U_optimal, omega, t)

print(f"Steady-state amplitude: {np.abs(X_ss)}")
print(f"Response at t={t}: {x_t}")
```

## Advanced Design Methods

### Weighted Control Design

Use weight matrices to emphasize tracking of specific states:

```python
# Emphasize first state more than second
W = np.diag([10.0, 1.0])

U_weighted = freq_controller.design_sinusoidal_control(
    X_target=X_target,
    omega=omega,
    W=W,
    lam=0.01
)

# Compare unweighted vs weighted tracking
X_unweighted = freq_controller.evaluate_steady_state(U_optimal, omega)
X_weighted = freq_controller.evaluate_steady_state(U_weighted, omega)

print("Unweighted tracking error:", np.abs(X_unweighted - X_target))
print("Weighted tracking error:", np.abs(X_weighted - X_target))
```

### Multi-Frequency Design

Design control for multiple frequencies simultaneously:

```python
frequencies = [0.5, 1.0, 2.0]
targets = [
    np.array([1.0, 0.0]),      # ω = 0.5 rad/s
    np.array([0.5, -0.5]),     # ω = 1.0 rad/s
    np.array([0.2, 0.8])       # ω = 2.0 rad/s
]

controls = []
for omega, X_target in zip(frequencies, targets):
    U = freq_controller.design_sinusoidal_control(X_target, omega)
    controls.append(U)

print("Multi-frequency controls:", controls)
```

### Entropy-Aware Frequency Control

Incorporate entropy costs into frequency-domain control design:

```python
from llrq import ThermodynamicAccountant

# Set up entropy accounting
L = np.eye(len(network.reaction_ids))  # Onsager conductance
accountant = ThermodynamicAccountant(network, L)

def entropy_aware_design(X_target, omega, entropy_weight=0.1):
    """Custom design incorporating entropy penalty."""

    # Standard frequency response
    H = freq_controller.compute_frequency_response(omega)

    # Compute entropy kernel (frequency-dependent)
    # For sinusoidal u(t) = Re{U e^(iωt)}, entropy rate involves |U|²
    entropy_penalty = entropy_weight * np.eye(H.shape[1])

    # Modified regularization includes entropy cost
    W = np.eye(H.shape[0])  # State weights
    H_weighted = W @ H

    # Solve: minimize ||H_weighted U - W X_target||² + λ||U||² + entropy_penalty
    A = H_weighted.conj().T @ H_weighted + 0.01 * np.eye(H.shape[1]) + entropy_penalty
    b = H_weighted.conj().T @ W @ X_target

    U = np.linalg.solve(A, b)
    return U

# Compare standard vs entropy-aware design
U_standard = freq_controller.design_sinusoidal_control(X_target, omega)
U_entropy = entropy_aware_design(X_target, omega, entropy_weight=0.05)

print(f"Standard control effort: {np.linalg.norm(U_standard)}")
print(f"Entropy-aware control effort: {np.linalg.norm(U_entropy)}")
```

## Frequency Response Analysis

### Bode Plot Data

Generate frequency response data for Bode plots:

```python
# Frequency sweep
omega_range = np.logspace(-2, 2, 100)  # 0.01 to 100 rad/s
frequency_responses = []

for omega in omega_range:
    H = freq_controller.compute_frequency_response(omega)
    frequency_responses.append(H)

H_array = np.array(frequency_responses)  # Shape: (freq, states, controls)

# Compute magnitude and phase for each input-output pair
magnitude_db = 20 * np.log10(np.abs(H_array))
phase_deg = np.angle(H_array) * 180 / np.pi

# Plot with matplotlib
import matplotlib.pyplot as plt

# Example: Bode plot for first input to first output
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.semilogx(omega_range, magnitude_db[:, 0, 0])
plt.ylabel('Magnitude (dB)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogx(omega_range, phase_deg[:, 0, 0])
plt.ylabel('Phase (degrees)')
plt.xlabel('Frequency (rad/s)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Nyquist Plot Data

Generate data for Nyquist plots:

```python
# Nyquist frequency range (positive frequencies only)
omega_nyquist = np.logspace(-2, 2, 200)
nyquist_data = []

for omega in omega_nyquist:
    H = freq_controller.compute_frequency_response(omega)
    nyquist_data.append(H[0, 0])  # First input-output pair

nyquist_array = np.array(nyquist_data)

# Plot Nyquist diagram
plt.figure(figsize=(8, 8))
plt.plot(nyquist_array.real, nyquist_array.imag, 'b-', label='H(iω)')
plt.plot(nyquist_array.real, -nyquist_array.imag, 'r--', label='H(-iω)')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Nyquist Plot')
plt.show()
```

## Integration with Time-Domain Simulation

### Simulating Sinusoidal Control

Use designed controls in time-domain LLRQ simulations:

```python
# Design control
omega = 1.0
X_target = np.array([0.5, -0.5])
U_complex = freq_controller.design_sinusoidal_control(X_target, omega)

# Create time-domain control function
def sinusoidal_control(t):
    """Convert complex amplitude to real sinusoidal control."""
    return np.real(U_complex * np.exp(1j * omega * t))

# Simulate with LLRQ solver
t_sim = np.linspace(0, 20, 1000)
sol = solver.solve(t_sim, u=sinusoidal_control)

# Analyze periodic steady state (last few periods)
t_steady = t_sim[-200:]  # Last 200 points
x_steady = sol.x[-200:]

# Check if we achieved target oscillation
from scipy.fft import fft, fftfreq

# FFT analysis to extract amplitude at target frequency
dt = t_steady[1] - t_steady[0]
freqs = fftfreq(len(t_steady), dt)
fft_x = fft(x_steady, axis=0)

# Find frequency closest to target
target_idx = np.argmin(np.abs(freqs - omega/(2*np.pi)))
achieved_amplitude = 2 * np.abs(fft_x[target_idx]) / len(t_steady)

print(f"Target amplitude: {np.abs(X_target)}")
print(f"Achieved amplitude: {achieved_amplitude}")
print(f"Tracking error: {np.abs(achieved_amplitude - np.abs(X_target))}")
```

### Phase Response Analysis

Analyze phase relationships in the response:

```python
def analyze_phase_response(sol, omega, t_analyze=None):
    """Analyze phase response of sinusoidal steady state."""

    if t_analyze is None:
        # Use last 20% of simulation for steady state
        n_steady = len(sol.t) // 5
        t_analyze = sol.t[-n_steady:]
        x_analyze = sol.x[-n_steady:]

    # Fit sinusoidal model: x(t) ≈ A cos(ωt + φ)
    phases = []
    amplitudes = []

    for i in range(x_analyze.shape[1]):  # For each state
        x_i = x_analyze[:, i]

        # Fit cos(ωt + φ) using least squares
        cos_terms = np.cos(omega * t_analyze)
        sin_terms = np.sin(omega * t_analyze)

        # x = A cos(ωt + φ) = A cos(φ)cos(ωt) - A sin(φ)sin(ωt)
        # x = a cos(ωt) + b sin(ωt) where a = A cos(φ), b = -A sin(φ)
        design_matrix = np.column_stack([cos_terms, sin_terms])
        coeffs, _, _, _ = np.linalg.lstsq(design_matrix, x_i, rcond=None)

        a, b = coeffs
        A = np.sqrt(a**2 + b**2)
        phi = np.arctan2(-b, a)

        amplitudes.append(A)
        phases.append(phi)

    return np.array(amplitudes), np.array(phases)

# Analyze the simulated response
amplitudes, phases = analyze_phase_response(sol, omega)

print(f"State amplitudes: {amplitudes}")
print(f"State phases: {phases * 180 / np.pi} degrees")

# Compare with frequency-domain prediction
H = freq_controller.compute_frequency_response(omega)
X_predicted = H @ U_complex

print(f"Predicted amplitudes: {np.abs(X_predicted)}")
print(f"Predicted phases: {np.angle(X_predicted) * 180 / np.pi} degrees")
```

## Complete Examples

### Example 1: Simple Oscillator Design

```python
import llrq
import numpy as np
import matplotlib.pyplot as plt

# Create simple reaction system
network = llrq.simple_reaction("A <-> B", k_forward=1.5, k_backward=0.5)
dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Create frequency controller
freq_controller = FrequencySpaceController.from_llrq_solver(solver)

# Design 1 Hz oscillation with 90° phase shift
omega = 2 * np.pi * 1.0  # 1 Hz in rad/s
X_target = np.array([1.0]) * np.exp(1j * np.pi/2)  # 90° phase

U_optimal = freq_controller.design_sinusoidal_control(
    X_target=X_target,
    omega=omega,
    lam=0.001
)

# Create control function and simulate
def control_func(t):
    return np.real(U_optimal * np.exp(1j * omega * t))

t = np.linspace(0, 5, 1000)  # 5 seconds
sol = solver.solve(t, u=control_func)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, sol.x[:, 0], label='State x₁')
plt.xlabel('Time (s)')
plt.ylabel('Reaction Force')
plt.title('State Response')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, [control_func(t_i) for t_i in t], label='Control u')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.title('Control Signal')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(sol.x[:, 0], np.gradient(sol.x[:, 0], t), alpha=0.7)
plt.xlabel('State x₁')
plt.ylabel('dx₁/dt')
plt.title('Phase Portrait')
plt.grid(True)

plt.subplot(2, 2, 4)
# Frequency response magnitude
omega_range = np.logspace(-1, 2, 100)
H_mag = []
for w in omega_range:
    H = freq_controller.compute_frequency_response(w)
    H_mag.append(np.abs(H[0, 0]))

plt.loglog(omega_range, H_mag)
plt.axvline(omega, color='r', linestyle='--', label=f'Design ω = {omega:.2f}')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('|H(iω)|')
plt.title('Frequency Response')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

### Example 2: Multi-State System with Complex Targets

```python
# Create more complex reaction network
reactions = [
    "A <-> B",  # R1
    "B <-> C",  # R2
    "C <-> A"   # R3 (creates cycle)
]

network = llrq.ReactionNetwork()
for i, reaction_str in enumerate(reactions):
    network.add_reaction_from_string(f"R{i+1}", reaction_str,
                                   k_forward=1.0 + 0.1*i,
                                   k_backward=0.5 + 0.05*i)

dynamics = llrq.LLRQDynamics(network)
solver = llrq.LLRQSolver(dynamics)
solver.compute_basis_matrices()

# Control first two reactions only
freq_controller = FrequencySpaceController.from_llrq_solver(
    solver, controlled_reactions=["R1", "R2"]
)

# Design complex target with different phases for each state
omega = 1.5  # rad/s
phases = [0, np.pi/3, 2*np.pi/3]  # 0°, 60°, 120°
magnitudes = [1.0, 0.8, 0.6]

X_target = np.array([mag * np.exp(1j * phase)
                    for mag, phase in zip(magnitudes, phases)])

# Include weighting - emphasize first two states
W = np.diag([2.0, 2.0, 1.0])

U_optimal = freq_controller.design_sinusoidal_control(
    X_target=X_target,
    omega=omega,
    W=W,
    lam=0.005
)

print(f"Optimal control shape: {U_optimal.shape}")
print(f"Control 1 - Mag: {np.abs(U_optimal[0]):.3f}, Phase: {np.angle(U_optimal[0])*180/np.pi:.1f}°")
print(f"Control 2 - Mag: {np.abs(U_optimal[1]):.3f}, Phase: {np.angle(U_optimal[1])*180/np.pi:.1f}°")

# Verify steady-state prediction
X_predicted = freq_controller.evaluate_steady_state(U_optimal, omega)
print("\nTarget vs Predicted:")
for i, (target, pred) in enumerate(zip(X_target, X_predicted)):
    print(f"State {i+1} - Target: {np.abs(target):.3f}∠{np.angle(target)*180/np.pi:.1f}°, "
          f"Predicted: {np.abs(pred):.3f}∠{np.angle(pred)*180/np.pi:.1f}°")
```

## Performance Considerations

1. **Matrix Conditioning**: Large regularization λ helps with ill-conditioned frequency response matrices
2. **Frequency Selection**: Avoid frequencies where `K + iωI` is nearly singular
3. **Computational Efficiency**: Cache frequency response matrices for repeated designs at same frequency
4. **Memory Usage**: For many frequencies, consider batch processing instead of storing all responses

The frequency-domain control capabilities make LLRQ particularly powerful for applications requiring periodic operation, such as circadian rhythms, metabolic oscillations, or energy harvesting systems.
