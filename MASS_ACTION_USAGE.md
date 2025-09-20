# Mass Action Dynamics - Usage Guide

This guide shows how to use the new mass action dynamics matrix computation functionality in the LLRQ package.

## Quick Start

```python
import numpy as np
from llrq import ReactionNetwork, LLRQDynamics

# 1. Define your reaction network stoichiometry
species_ids = ['A', 'B', 'C']
reaction_ids = ['R1', 'R2']  # A ⇌ B ⇌ C
S = np.array([
    [-1, 0],   # A: consumed in R1
    [1, -1],   # B: produced in R1, consumed in R2
    [0, 1]     # C: produced in R2
])

network = ReactionNetwork(species_ids, reaction_ids, S)

# 2. Define mass action parameters
c_star = [2.0, 1.5, 0.5]    # Steady-state concentrations
k_plus = [1.0, 2.0]         # Forward rate constants
k_minus = [0.5, 1.0]        # Backward rate constants

# 3. Create dynamics automatically from mass action
dynamics = LLRQDynamics.from_mass_action(
    network=network,
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium'
)

# 4. Analyze the system
print(f"Equilibrium constants: {dynamics.Keq}")
print(f"Dynamics matrix K:\n{dynamics.K}")

eigen_info = dynamics.compute_eigenanalysis()
print(f"System stable: {eigen_info['is_stable']}")
print(f"Relaxation timescales: {1/eigen_info['eigenvalues'].real}")
```

## Key Methods

### ReactionNetwork.compute_dynamics_matrix()

Low-level method that computes the dynamics matrix K directly:

```python
result = network.compute_dynamics_matrix(
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium',          # or 'nonequilibrium'
    reduce_to_image=True,        # Recommended
    enforce_symmetry=False       # For detailed balance systems
)

K = result['K']                  # Full dynamics matrix
K_red = result.get('K_reduced')  # Reduced matrix (if reduce_to_image=True)
phi = result.get('phi')          # Flux coefficients (equilibrium mode)
eigeninfo = result['eigenanalysis']  # Stability and timescales
```

### LLRQDynamics.from_mass_action()

High-level factory method that creates a complete dynamics object:

```python
dynamics = LLRQDynamics.from_mass_action(
    network=network,
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium'
)

# Access mass action metadata
info = dynamics.get_mass_action_info()
print(f"Mode used: {info['mode']}")
print(f"Equilibrium point: {info['equilibrium_point']}")
```

### Accelerated Far-from-Equilibrium Relaxation (Optional)

To capture mass-action-like acceleration far from equilibrium, enable the
accelerated relaxation law after building the dynamics object (requires
mass-action metadata):

```python
dynamics = LLRQDynamics.from_mass_action(
    network=network,
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium',
    relaxation_mode='accelerated',  # nonlinear modal relaxation
)

# The relaxation law can also be enabled post-hoc:
# dynamics.enable_accelerated_relaxation()
print(type(dynamics.relaxation_law).__name__)
```

## Algorithm Modes

### Equilibrium Mode (Recommended)

Use when the system operates near thermodynamic equilibrium:

```python
# Good for: metabolic networks, enzyme systems near equilibrium
dynamics = LLRQDynamics.from_mass_action(
    network, c_star, k_plus, k_minus,
    mode='equilibrium'
)
```

**Algorithm**: K = S^T (D*)^(-1) S Φ where Φ = Diag(φ_j), φ_j = k_j^+ (c*)^ν_reac

### Nonequilibrium Mode

Use for systems far from equilibrium or driven by external fluxes:

```python
# Good for: driven systems, systems with large concentration gradients
dynamics = LLRQDynamics.from_mass_action(
    network, c_star, k_plus, k_minus,
    mode='nonequilibrium'
)
```

**Algorithm**: K = -S^T (D*)^(-1) S J_u R where J_u = ∂v/∂u, R = D* S (S^T D* S)^(-1)

## Common Examples

### Simple Reversible Reaction: A ⇌ B

```python
network = ReactionNetwork(['A', 'B'], ['R1'], [[-1], [1]])

# Equilibrium: k_plus/k_minus should equal [B]/[A]
c_star = [1.0, 2.0]  # [B]/[A] = 2
k_plus = [2.0]       # k_plus/k_minus = 2 ✓
k_minus = [1.0]

dynamics = LLRQDynamics.from_mass_action(
    network, c_star, k_plus, k_minus
)

# For single reaction, K is 1x1 matrix
print(f"Relaxation rate: {dynamics.K[0,0]}")
print(f"Timescale: {1/dynamics.K[0,0]:.2f}")
```

### Enzymatic Reaction: E + S ⇌ ES → E + P

```python
# Species: [E, S, ES, P]
# Reactions: [binding, unbinding, catalysis]
species = ['E', 'S', 'ES', 'P']
reactions = ['bind', 'unbind', 'cat']
S = np.array([
    [-1, 1, 1],    # E: consumed in bind, produced in unbind and cat
    [-1, 1, 0],    # S: consumed in bind, produced in unbind
    [1, -1, -1],   # ES: produced in bind, consumed in unbind and cat
    [0, 0, 1]      # P: produced in cat
])

network = ReactionNetwork(species, reactions, S)

# Typical enzyme kinetics parameters
c_star = [1.0, 10.0, 0.5, 2.0]  # [E], [S], [ES], [P]
k_plus = [1e6, 1e3, 1e2]        # Binding, unbinding, catalysis
k_minus = [1e3, 1e6, 0.0]       # Reverse rates (catalysis irreversible)

dynamics = LLRQDynamics.from_mass_action(
    network, c_star, k_plus, k_minus,
    reduce_basis=True  # Recommended for networks with conservation
)

# Analyze coupling between binding and catalysis
print(f"Dynamics matrix shape: {dynamics.K.shape}")
print(f"Off-diagonal coupling: {np.max(np.abs(dynamics.K - np.diag(np.diag(dynamics.K))))}")
```

### Linear Pathway: A → B → C → D

```python
species = ['A', 'B', 'C', 'D']
reactions = ['R1', 'R2', 'R3']
S = np.array([
    [-1, 0, 0],   # A →
    [1, -1, 0],   # → B →
    [0, 1, -1],   # → C →
    [0, 0, 1]     # → D
])

network = ReactionNetwork(species, reactions, S)

# Pathway with varying driving forces
c_star = [5.0, 2.0, 1.0, 0.1]    # Decreasing gradient A→D
k_plus = [2.0, 3.0, 1.0]         # Forward rates
k_minus = [0.1, 0.2, 0.05]       # Small reverse rates

# Add external drive (substrate input, product removal)
def pathway_drive(t):
    return np.array([0.5, 0.0, -0.2])  # Input at R1, removal at R3

dynamics = LLRQDynamics.from_mass_action(
    network, c_star, k_plus, k_minus,
    external_drive=pathway_drive
)

# Check for propagation delays
eigeninfo = dynamics.compute_eigenanalysis()
timescales = 1/eigeninfo['eigenvalues'].real
print(f"Propagation timescales: {sorted(timescales)}")
```

## Tips and Best Practices

### 1. Always Use Basis Reduction
```python
# Recommended - eliminates conservation law null space
dynamics = LLRQDynamics.from_mass_action(..., reduce_basis=True)
```

### 2. Check Equilibrium Consistency
```python
# Verify equilibrium condition: k_plus/k_minus should match concentration ratios
Keq_rates = k_plus / k_minus
Keq_concs = compute_equilibrium_constants_from_concentrations(c_star, S)
print(f"Rate ratios: {Keq_rates}")
print(f"Concentration ratios: {Keq_concs}")
```

### 3. Validate Stability
```python
eigeninfo = dynamics.compute_eigenanalysis()
if not eigeninfo['is_stable']:
    print("Warning: System is unstable!")
    print(f"Unstable eigenvalues: {eigeninfo['eigenvalues'][eigeninfo['eigenvalues'].real < 0]}")
```

### 4. Handle Large Networks
```python
# For large networks, use nonequilibrium mode sparingly (more expensive)
if network.n_reactions > 50:
    mode = 'equilibrium'  # Faster and usually sufficient
else:
    mode = 'nonequilibrium'  # More accurate for driven systems
```

### 5. Symmetry for Detailed Balance
```python
# If your system satisfies detailed balance
dynamics = LLRQDynamics.from_mass_action(
    ...,
    enforce_symmetry=True  # Ensures K is symmetric positive definite
)
```

## Error Handling

Common issues and solutions:

```python
# Issue: Singular matrices
# Solution: Use basis reduction and check for conservation laws
try:
    dynamics = LLRQDynamics.from_mass_action(..., reduce_basis=True)
except LinAlgError:
    print("Check for conservation laws or linear dependencies")

# Issue: Non-equilibrium concentrations
# Solution: Verify equilibrium conditions
def check_equilibrium(c_star, k_plus, k_minus, S):
    for j in range(len(k_plus)):
        nu_reac = np.maximum(-S[:, j], 0)
        nu_prod = np.maximum(S[:, j], 0)
        forward_flux = k_plus[j] * np.prod(c_star ** nu_reac)
        backward_flux = k_minus[j] * np.prod(c_star ** nu_prod)
        if not np.isclose(forward_flux, backward_flux, rtol=0.1):
            print(f"Reaction {j}: forward={forward_flux:.3f}, backward={backward_flux:.3f}")

check_equilibrium(c_star, k_plus, k_minus, S)
```

## Integration with Existing LLRQ Workflow

The mass action functionality integrates seamlessly with existing LLRQ features:

```python
# 1. Create from mass action
dynamics = LLRQDynamics.from_mass_action(network, c_star, k_plus, k_minus)

# 2. Use all existing LLRQ functionality
from llrq import LLRQSolver, LLRQVisualizer

solver = LLRQSolver(dynamics)
visualizer = LLRQVisualizer(solver)

# 3. Solve dynamics
solution = solver.solve(
    initial_conditions={'A': 1.2, 'B': 0.8, 'C': 0.3},
    t_span=(0, 10)
)

# 4. Visualize results
fig = visualizer.plot_dynamics(solution)
```

This provides a complete pipeline from mass action kinetics to dynamic analysis and visualization.
