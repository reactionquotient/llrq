---
layout: page
title: API Reference
nav_order: 3
---

# API Reference

This page provides comprehensive documentation for all LLRQ classes and functions.

## Core Classes

### ReactionNetwork

The `ReactionNetwork` class represents a chemical reaction network with species and reactions.

```python
class ReactionNetwork:
    def __init__(self, 
                 species_ids: List[str],
                 reaction_ids: List[str], 
                 stoichiometric_matrix: np.ndarray,
                 species_info: Optional[Dict] = None,
                 reaction_info: Optional[List[Dict]] = None,
                 parameters: Optional[Dict] = None)
```

**Parameters:**
- `species_ids`: List of species identifiers (e.g., ['A', 'B', 'C'])
- `reaction_ids`: List of reaction identifiers (e.g., ['R1', 'R2'])  
- `stoichiometric_matrix`: Stoichiometric matrix S (species × reactions)
- `species_info`: Additional species information from SBML
- `reaction_info`: Additional reaction information from SBML
- `parameters`: Global parameters from SBML

**Properties:**
- `n_species`: Number of species in the network
- `n_reactions`: Number of reactions in the network
- `species_names`: List of species names
- `reaction_names`: List of reaction names
- `S`: Stoichiometric matrix

**Key Methods:**

#### `compute_reaction_quotients(concentrations)`
Compute reaction quotients for all reactions.

```python
Q = network.compute_reaction_quotients(concentrations)
```

**Parameters:**
- `concentrations`: Array of species concentrations

**Returns:**
- Array of reaction quotients Q_j = ∏_i [X_i]^S_ij

#### `get_initial_concentrations()`
Get initial concentrations from species info.

**Returns:**
- Array of initial concentrations in species order

#### `summary()`
Generate a human-readable summary of the network.

**Returns:**
- String description of network properties

---

### LLRQDynamics

The `LLRQDynamics` class implements the core log-linear dynamics system.

```python
class LLRQDynamics:
    def __init__(self, 
                 network: ReactionNetwork,
                 equilibrium_constants: Optional[np.ndarray] = None,
                 relaxation_matrix: Optional[np.ndarray] = None,
                 external_drive: Optional[Callable] = None)
```

**Parameters:**
- `network`: ReactionNetwork instance
- `equilibrium_constants`: Equilibrium constants K_eq for each reaction
- `relaxation_matrix`: Relaxation rate matrix K  
- `external_drive`: Function u(t) returning external drives

**Core Equation:**
The dynamics implement: d/dt ln Q = -K ln(Q/K_eq) + u(t)

**Properties:**
- `network`: Associated reaction network
- `Keq`: Equilibrium constants
- `K`: Relaxation rate matrix
- `external_drive`: External drive function

**Key Methods:**

#### `dynamics(t, x)`
Compute the log-linear dynamics dx/dt = -K*x + u(t).

**Parameters:**
- `t`: Current time
- `x`: Log deviations ln(Q/K_eq)

**Returns:**
- Time derivative dx/dt

#### `compute_log_deviation(Q)`
Convert reaction quotients to log deviations.

**Parameters:**
- `Q`: Reaction quotients

**Returns:**
- Log deviations x = ln(Q/K_eq)

#### `compute_reaction_quotients(x)`
Convert log deviations to reaction quotients.

**Parameters:**  
- `x`: Log deviations

**Returns:**
- Reaction quotients Q = K_eq * exp(x)

---

### LLRQSolver

The `LLRQSolver` class provides analytical and numerical solution methods.

```python
class LLRQSolver:
    def __init__(self, dynamics: LLRQDynamics)
```

**Parameters:**
- `dynamics`: LLRQDynamics instance

**Key Methods:**

#### `solve(initial_conditions, t_span, method='auto', **kwargs)`
Solve the dynamics with the specified method.

```python
solution = solver.solve(
    initial_conditions={'A': 1.0, 'B': 0.1},
    t_span=(0, 10),
    method='analytical'
)
```

**Parameters:**
- `initial_conditions`: Dict of species concentrations or concentration array
- `t_span`: Time span tuple (t_start, t_end) or time array
- `method`: Solution method ('analytical', 'numerical', or 'auto')
- `enforce_conservation`: Whether to enforce conservation laws (default: True)

**Returns:**
Dictionary with solution data:
```python
{
    'success': bool,
    'method': str,
    't': np.ndarray,  # Time points
    'concentrations': dict,  # Species concentrations over time
    'reaction_quotients': np.ndarray,  # Q(t) for each reaction
    'log_deviations': np.ndarray  # x(t) = ln(Q/Keq)
}
```

#### `solve_analytical(x0, t_span, constant_drive=None)`
Solve using exact analytical solution (when applicable).

**Parameters:**
- `x0`: Initial log deviations
- `t_span`: Time span or time array
- `constant_drive`: Constant external drive vector

**Returns:**
- Solution dictionary (same format as `solve`)

#### `solve_numerical(x0, t_span, **kwargs)`
Solve using numerical integration.

**Parameters:**
- `x0`: Initial log deviations  
- `t_span`: Time span or time array
- Additional kwargs passed to scipy.integrate.solve_ivp

**Returns:**
- Solution dictionary (same format as `solve`)

---

### LLRQVisualizer  

The `LLRQVisualizer` class creates publication-quality plots.

```python
class LLRQVisualizer:
    def __init__(self, network: ReactionNetwork)
```

**Key Methods:**

#### `plot_dynamics(solution, **kwargs)`
Create comprehensive dynamics visualization.

```python
fig = visualizer.plot_dynamics(solution, 
                              show_concentrations=True,
                              show_quotients=True,
                              figsize=(12, 8))
```

**Parameters:**
- `solution`: Solution dictionary from solver
- `show_concentrations`: Whether to plot species concentrations
- `show_quotients`: Whether to plot reaction quotients  
- `figsize`: Figure size tuple

**Returns:**
- matplotlib Figure object

#### `plot_phase_portrait(solution, species_pair, **kwargs)`
Create phase portrait for two species.

**Parameters:**
- `solution`: Solution dictionary
- `species_pair`: Tuple of species names ('A', 'B')

**Returns:**
- matplotlib Figure object

---

### SBMLParser

The `SBMLParser` class handles SBML model import.

```python
class SBMLParser:
    def __init__(self, sbml_file: str)
```

**Parameters:**
- `sbml_file`: Path to SBML file or SBML string content

**Key Methods:**

#### `extract_network_data()`
Extract network data from SBML model.

**Returns:**
Dictionary containing:
- `species_ids`: List of species IDs
- `reaction_ids`: List of reaction IDs  
- `stoichiometric_matrix`: Stoichiometric matrix
- `species_info`: Species metadata
- `reaction_info`: Reaction metadata

---

## Convenience Functions

### `llrq.from_sbml()`

Load SBML model and create complete LLRQ system.

```python
network, dynamics, solver, visualizer = llrq.from_sbml(
    sbml_file='model.xml',
    equilibrium_constants=np.array([2.0, 1.5]),
    relaxation_matrix=np.diag([1.0, 0.5]),
    external_drive=lambda t: np.array([0.1, -0.1])
)
```

**Parameters:**
- `sbml_file`: Path to SBML file
- `equilibrium_constants`: K_eq for each reaction
- `relaxation_matrix`: K matrix
- `external_drive`: u(t) function

**Returns:**
- Tuple of (network, dynamics, solver, visualizer)

### `llrq.simple_reaction()`

Create simple A ⇌ B reaction system.

```python
network, dynamics, solver, visualizer = llrq.simple_reaction(
    reactant_species="A",
    product_species="B",
    equilibrium_constant=2.0,
    relaxation_rate=1.0,
    initial_concentrations={"A": 1.0, "B": 0.1}
)
```

**Parameters:**
- `reactant_species`: Name of reactant species
- `product_species`: Name of product species
- `equilibrium_constant`: K_eq for the reaction
- `relaxation_rate`: Relaxation rate k
- `initial_concentrations`: Initial species concentrations

**Returns:**
- Tuple of (network, dynamics, solver, visualizer)

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid input parameters or dimensions
- `SBMLParseError`: SBML parsing errors
- `ConvergenceError`: Numerical solver convergence issues

### Example Error Handling

```python
try:
    solution = solver.solve(
        initial_conditions={'A': 1.0, 'B': 0.1},
        t_span=(0, 10),
        method='analytical'
    )
    if not solution['success']:
        print(f"Solution failed: {solution.get('message', 'Unknown error')}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

---

## Advanced Usage

### Custom External Drives

```python
def time_varying_drive(t):
    """Example oscillatory drive."""
    return np.array([0.1 * np.sin(t), -0.05 * np.cos(2*t)])

dynamics.external_drive = time_varying_drive
```

### Conservation Law Enforcement

```python
# Solve with conservation laws enforced (default)
solution = solver.solve(
    initial_conditions=concentrations,
    t_span=(0, 10),
    enforce_conservation=True
)
```

### Custom Relaxation Matrices

```python
# Non-diagonal coupling between reactions
K = np.array([[1.0, 0.1],
              [0.1, 1.5]])
dynamics = LLRQDynamics(network, Keq, K)
```