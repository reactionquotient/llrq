# LLRQ Package Documentation

## Overview

LLRQ (Log-Linear Reaction Quotient) is a Python package implementing the log-linear reaction quotient dynamics framework described in Diamond (2025). It provides a novel approach to modeling chemical reaction networks where reaction quotients evolve exponentially toward equilibrium when viewed on a logarithmic scale, yielding analytically tractable linear dynamics in log-space.

**Key Equation**: `d/dt ln Q = -K ln(Q/Keq) + u(t)`

Where Q is reaction quotients, K is relaxation rate matrix, Keq is equilibrium constants, and u(t) is external drive.

## Environment Setup

- Use the tellurium conda environment:
  ```bash
  source /Users/sdiamond/opt/anaconda3/etc/profile.d/conda.sh && conda activate tellurium
  ```
- Run tests with: `python -m pytest tests/test_mass_action_simple.py -v`

## Package Architecture

### Core Modules (`src/llrq/`)

#### Foundation Classes
- **`reaction_network.py`**: `ReactionNetwork` class - represents chemical reaction networks with stoichiometry, species, and reaction quotient computation
- **`llrq_dynamics.py`**: `LLRQDynamics` class - implements core log-linear dynamics system `d/dt ln Q = -K ln(Q/Keq) + u(t)`
- **`solver.py`**: `LLRQSolver` class - provides analytical and numerical solution methods with conservation law enforcement
- **`sbml_parser.py`**: `SBMLParser` class - parses SBML models and extracts network information

#### Analysis & Visualization
- **`visualization.py`**: `LLRQVisualizer` class - creates publication-quality plots for dynamics, phase portraits, and frequency analysis
- **`mass_action_simulator.py`**: `MassActionSimulator` class - compares LLRQ approximation with true mass action kinetics

#### Control Systems
- **`control.py`**: `LLRQController` class - implements control strategies based on LLRQ theory with LQR control
- **`frequency_control.py`**: `FrequencySpaceController` class - frequency-domain control for designing sinusoidal inputs
- **`control/lqr.py`**: Linear Quadratic Regulator design functions

#### Advanced Features
- **`thermodynamic_accounting.py`**: `ThermodynamicAccountant` class - entropy production calculations from reaction forces and external drives with energy balance diagnostics
- **`estimation/kalman.py`**: Kalman filter for state estimation in noisy LLRQ systems
- **`integrations/mass_action_drive.py`**: Integration with mass action kinetics for external drives
- **`ops/ltisolve.py`**: Linear time-invariant system solution utilities
- **`ops/time_varying.py`**: Time-varying system analysis tools

#### Utilities
- **`utils/concentration_utils.py`**: Concentration-related computations and conversions
- **`utils/equilibrium_utils.py`**: Equilibrium constant calculations and thermodynamic utilities

### Entry Points (`src/llrq/__init__.py`)

- **Convenience Functions**:
  - `from_sbml()`: Load SBML model and create complete LLRQ system
  - `simple_reaction()`: Create simple A ⇌ B reaction system

- **Main Exports**: All core classes plus specialized controllers and utilities

## Core Component Relationships

```
ReactionNetwork ──→ LLRQDynamics ──→ LLRQSolver ──→ Results
     ↓                   ↓              ↓            ↓
SBMLParser         Controllers    Visualizer  ThermodynamicAccountant
                      ↓                            ↑
            FrequencySpaceController               │
            LLRQController ─────────────────────────┘
            AdaptiveController
```

## Testing & Examples

### Test Suite (`tests/`, 17 files)
- **Core Tests**: `test_reaction_network.py`, `test_llrq_dynamics.py`, `test_solver.py`
- **Integration Tests**: `test_mass_action.py`, `test_api_integration.py`, `test_integration.py`
- **Control Tests**: `test_mass_action_control.py`, `test_frequency_control.py`
- **Specialized Tests**: `test_sbml_parser.py`, `test_thermodynamic_accounting.py`, `test_thermodynamic_accounting_entropy.py`, `test_performance.py`
- **Edge Cases**: `test_edge_cases.py`, `test_time_varying_drives.py`

### Examples (`examples/`, 21+ files)
- **Basic Usage**: `linear_vs_mass_action_simple.py`, `mass_action_example.py`
- **Control Examples**: `lqr_complete_example.py`, `frequency_space_control.py`, `cycle_closed_loop.py`
- **Advanced Features**: `example_thermodynamic_accounting.py`, `entropy_production_demo.py`, `integrated_control_demo.py`

## Key Features by Module

### Solver Capabilities
- **Analytical Solutions**: Matrix exponential for constant drives
- **Numerical Integration**: Robust ODE solving with conservation enforcement
- **Mass Action Comparison**: Switch between LLRQ approximation and true kinetics
- **Conservation Laws**: Automatic detection and enforcement of stoichiometric constraints
- **Entropy Production**: Optional entropy accounting during simulation with `compute_entropy=True`

### Control Systems
- **LQR Control**: Optimal control for linear LLRQ systems
- **Frequency Control**: Design sinusoidal inputs for periodic steady states
- **Adaptive Control**: Real-time parameter adaptation
- **Multi-objective**: Balance performance, control effort, and robustness

### Visualization
- **Time Series**: Concentration and reaction quotient evolution
- **Phase Portraits**: State space trajectories
- **Frequency Response**: Bode plots and Nyquist diagrams
- **Control Analysis**: Control effort and tracking performance

### Thermodynamic Accounting
- **Entropy from Reaction Forces**: Compute entropy production from x(t) = ln(Q/Keq) trajectories
- **Quasi-Steady Approximation**: Entropy estimation from external drives u(t) when x ≈ K^{-1}u
- **Energy Balance Diagnostics**: Power balance checks for model validation and noise assessment
- **Dual Accounting**: Compare entropy estimates from reaction forces vs external drives
- **Onsager Integration**: Seamless integration with existing Onsager conductance calculations

## Development Guidelines

1. **Testing**: All new features must include comprehensive tests
2. **Documentation**: Follow NumPy docstring conventions
3. **Mass Action Integration**: Support both LLRQ approximation and true kinetics comparison
4. **Conservation**: Always respect stoichiometric constraints
5. **Performance**: Prefer analytical solutions when available, fallback to robust numerics
