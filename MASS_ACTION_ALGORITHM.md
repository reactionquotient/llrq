# Mass Action Dynamics Matrix Computation

This document describes the implementation of the algorithm from Diamond (2025) "Log-Linear Reaction Quotient Dynamics" for computing the dynamics matrix **K** from arbitrary mass action reaction networks.

## Overview

The algorithm computes the dynamics matrix **K** that governs the log-linear reaction quotient dynamics:

```
d/dt ln Q = -K ln(Q/Keq) + u(t)
```

where:
- **Q** = vector of reaction quotients
- **Keq** = equilibrium constants
- **u(t)** = external drives
- **K** = relaxation/dynamics matrix (computed by this algorithm)

## Algorithm Description

### Inputs

- **νᵣₑₐc, νₚᵣₒd**: Reactant and product stoichiometries (extracted from stoichiometric matrix **S**)
- **S**: Stoichiometric matrix (species × reactions)
- **k⁺, k⁻**: Forward and backward rate constants
- **c\***: Equilibrium (or steady-state) concentrations

### Three Operating Modes

#### Mode A: Near Thermodynamic Equilibrium

**Use when**: The system is operating close to thermodynamic equilibrium.

**Algorithm**:
1. Compute flux coefficients: **φⱼ = kⱼ⁺ (c\*)^νⱼʳᵉᵃᶜ = kⱼ⁻ (c\*)^νⱼᵖʳᵒᵈ**
2. Form diagonal matrix: **Φ = Diag(φ)**
3. Form concentration matrix: **D\* = Diag(c\*)**
4. Compute dynamics matrix: **K = Sᵀ (D\*)⁻¹ S Φ**

**Physical interpretation**: The flux coefficients φⱼ represent the reaction rates at the equilibrium point. The matrix **K** captures how deviations from equilibrium relax back.

#### Mode B: Around Nonequilibrium Steady State

**Use when**: The system operates at a nonequilibrium steady state (e.g., driven by external fluxes).

**Algorithm**:
1. Evaluate Jacobian: **Jᵤ = ∂v/∂u** at **c\*** (where **u = ln c**, **v** = reaction rates)
2. Build projection matrix: **R = D\* S (Sᵀ D\* S)⁻¹**
3. Compute dynamics matrix: **K = -Sᵀ (D\*)⁻¹ S Jᵤ R**

**Physical interpretation**: **Jᵤ** captures how reaction rates change with concentration perturbations. **R** projects onto the reaction space.

#### Mode C: Enforcing "Niceness" (Optional)

**Positive stability**: Project **K** to its symmetric part: **½(K + Kᵀ)** and shift negative eigenvalues to small positive values.

**Units/time-scale**: Everything scales with time units via **Φ** (or **Jᵤ**).

**Conservation**: Only enters through **c\*** (via **D\***); doesn't restrict dynamics directly.

### Optional Enhancements

#### Basis Reduction
Projects the dynamics to **Im(Sᵀ)** to eliminate numerical issues from conservation laws:
- Find orthonormal basis **B** for **Im(Sᵀ)**
- Compute reduced matrix: **Kᵣₑd = Bᵀ K B**

#### Symmetry Enforcement
For systems satisfying detailed balance:
- Symmetrize: **K → ½(K + Kᵀ)**
- Ensure positive definiteness by eigenvalue shifting

## Implementation Details

### Key Methods

#### `ReactionNetwork.compute_dynamics_matrix()`
```python
result = network.compute_dynamics_matrix(
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium',  # or 'nonequilibrium'
    reduce_to_image=True,
    enforce_symmetry=False
)
```

**Returns**:
- `K`: Full dynamics matrix
- `K_reduced`: Reduced matrix (if `reduce_to_image=True`)
- `basis`: Basis matrix for **Im(Sᵀ)** (if `reduce_to_image=True`)
- `phi`: Flux coefficients (equilibrium mode only)
- `eigenanalysis`: Eigenvalues, eigenvectors, stability analysis

#### `LLRQDynamics.from_mass_action()`
```python
dynamics = LLRQDynamics.from_mass_action(
    network=network,
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium'
)
```

Factory method that creates a complete `LLRQDynamics` object with the computed **K** matrix.

## Mathematical Foundation

### Flux Coefficients (Equilibrium Mode)

For each reaction **j**, the flux coefficient is computed as:

**φⱼ = kⱼ⁺ ∏ᵢ [cᵢ\*]^νᵢⱼʳᵉᵃᶜ**

At thermodynamic equilibrium, this equals:

**φⱼ = kⱼ⁻ ∏ᵢ [cᵢ\*]^νᵢⱼᵖʳᵒᵈ**

### Jacobian Computation (Nonequilibrium Mode)

The Jacobian **Jᵤ** has elements:

**(Jᵤ)ⱼᵢ = ∂vⱼ/∂uᵢ = ∂vⱼ/∂ln(cᵢ) = cᵢ ∂vⱼ/∂cᵢ**

For mass action kinetics:
- Reactant contribution: **-νᵢⱼʳᵉᵃᶜ × kⱼ⁺ × cᵢ × ∏ₖ cₖ^νₖⱼʳᵉᵃᶜ**
- Product contribution: **+νᵢⱼᵖʳᵒᵈ × kⱼ⁻ × cᵢ × ∏ₖ cₖ^νₖⱼᵖʳᵒᵈ**

### Conservation Laws

Conservation laws arise from the left null space of **S**:
- Conservation matrix **C** satisfies **C S = 0**
- Conserved quantities: **C c = constant**
- Basis reduction eliminates these null directions

## Examples

### Simple Reversible Reaction: A ⇌ B

```python
# Setup
species_ids = ['A', 'B']
reaction_ids = ['R1']
S = np.array([[-1], [1]])  # A → B

network = ReactionNetwork(species_ids, reaction_ids, S)

# Parameters
c_star = [1.0, 2.0]  # Equilibrium: [B]/[A] = 2
k_plus = [2.0]       # Forward rate
k_minus = [1.0]      # Backward rate (Keq = k_plus/k_minus = 2)

# Compute dynamics matrix
result = network.compute_dynamics_matrix(
    equilibrium_point=c_star,
    forward_rates=k_plus,
    backward_rates=k_minus,
    mode='equilibrium'
)

print(f"Dynamics matrix: {result['K']}")
print(f"Stable: {result['eigenanalysis']['is_stable']}")
```

### Multi-Step Pathway: A ⇌ B ⇌ C

```python
# Linear pathway with coupling between reactions
species_ids = ['A', 'B', 'C']
reaction_ids = ['R1', 'R2']
S = np.array([
    [-1, 0],   # A
    [1, -1],   # B
    [0, 1]     # C
])

# The resulting K matrix will be 2×2 and non-diagonal,
# showing coupling between the two reactions
```

## Validation and Testing

The implementation includes comprehensive tests for:

1. **Equilibrium flux balance**: Forward and backward fluxes equal at equilibrium
2. **Stability analysis**: All eigenvalues have non-negative real parts
3. **Conservation laws**: Proper handling of conserved quantities
4. **Basis reduction**: Dimensionality reduction works correctly
5. **Error handling**: Invalid inputs are caught appropriately

## References

- Diamond, S. (2025). "Log-Linear Reaction Quotient Dynamics"
- Network thermodynamics and mass action kinetics principles
- Linear algebra for conservation law analysis

## Notes

- The algorithm handles both reversible and irreversible reactions
- Numerical stability is ensured through basis reduction and pseudoinverse methods
- The implementation is optimized for typical biochemical network sizes
- External drives **u(t)** can be added after computing **K**
