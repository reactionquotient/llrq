"""Log-Linear Reaction Quotient Dynamics (LLRQ).

A Python package for analyzing chemical reaction networks using the log-linear
reaction quotient dynamics framework described in Diamond (2025).

This framework provides:
- Analytical solutions for reaction quotient evolution
- Tractable analysis of complex networks
- Natural incorporation of thermodynamic constraints
- Linear control theory for biological systems
"""

from typing import Optional

__version__ = "0.0.3"
__author__ = "Steven Diamond"
__email__ = "steven@gridmatic.com"

# Control and simulation
from .control import AdaptiveController, ControlledSimulation, LLRQController, design_lqr_controller
from .frequency_control import FrequencySpaceController
from .llrq_dynamics import LLRQDynamics
from .reaction_network import ReactionNetwork

# Core imports
from .sbml_parser import SBMLParseError, SBMLParser
from .yaml_parser import YAMLParseError, YAMLModelParser, load_yaml_model
from .solver import LLRQSolver
from .thermodynamic_accounting import ThermodynamicAccountant, AccountingResult, DualAccountingResult
from .visualization import LLRQVisualizer

try:
    from .mass_action_simulator import MassActionSimulator
except ImportError:
    # Mass action simulation not available without roadrunner/tellurium
    pass

# CVXpy-based control (required dependency)
from .cvx_control import CVXController, CVXObjectives, CVXConstraints, create_entropy_aware_cvx_controller

# Genome-scale model utilities
from .genome_scale import GenomeScaleAnalyzer, load_genome_scale_model, compare_model_sizes

# Estimation utilities
from .estimation import KMatrixEstimator
from .utils.physical_bounds import (
    compute_diffusion_limit,
    enzyme_to_conductance,
    validate_physical_consistency,
    compute_spectral_caps,
    gershgorin_bounds,
    estimate_reaction_timescales,
)


# Convenience functions
def from_model(
    model_file: str,
    equilibrium_constants=None,
    relaxation_matrix=None,
    external_drive=None,
    use_genome_scale_analyzer=None,
    temperature: float = 298.15,
    compute_keq_from_thermodynamics: bool = True,
    verbose: bool = False,
):
    """Load model (SBML or YAML) and create LLRQ system.

    Args:
        model_file: Path to model file (.xml for SBML, .yml/.yaml for YAML)
        equilibrium_constants: Equilibrium constants for each reaction. If None and
                              YAML contains thermodynamic data, Keq will be computed.
        relaxation_matrix: Relaxation rate matrix K
        external_drive: External drive function u(t)
        use_genome_scale_analyzer: If True, use GenomeScaleAnalyzer for large SBML models.
                                  If None, auto-detect based on file size.
        temperature: Temperature for Keq computation from thermodynamic data (K)
        compute_keq_from_thermodynamics: Whether to compute Keq from ΔG° data in YAML files
        verbose: Print information about model loading and thermodynamic data

    Returns:
        Tuple of (network, dynamics, solver, visualizer)
    """
    import os

    # Determine file format
    file_ext = os.path.splitext(model_file)[1].lower()

    if file_ext in [".yml", ".yaml"]:
        # YAML model with potential thermodynamic data
        if verbose:
            print(f"Loading YAML model: {model_file}")

        # Load YAML model data
        model_data = load_yaml_model(
            model_file, compute_keq=compute_keq_from_thermodynamics, temperature=temperature, verbose=verbose
        )

        # Create network from YAML data
        network = ReactionNetwork(
            species_ids=model_data["species_ids"],
            reaction_ids=model_data["reaction_ids"],
            stoichiometric_matrix=model_data["stoichiometric_matrix"],
            species_info=model_data["species"],
            reaction_info=model_data["reactions"],
        )

        # Use computed Keq if available and not overridden
        if equilibrium_constants is None and "equilibrium_constants" in model_data:
            equilibrium_constants = model_data["equilibrium_constants"]
            if verbose:
                keq_info = model_data.get("keq_info", {})
                coverage = keq_info.get("coverage_partial", 0)
                print(f"Using computed equilibrium constants ({coverage:.1%} reactions with thermodynamic data)")

    elif file_ext in [".xml", ".sbml"]:
        # SBML model
        if verbose:
            print(f"Loading SBML model: {model_file}")

        # Auto-detect if we should use genome-scale analyzer
        if use_genome_scale_analyzer is None:
            # Quick check of file size as a heuristic
            try:
                file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
                use_genome_scale_analyzer = file_size_mb > 1.0  # Use for files > 1MB
            except:
                use_genome_scale_analyzer = False

        if use_genome_scale_analyzer:
            # Use genome-scale analyzer for large models
            analyzer = GenomeScaleAnalyzer(model_file, lazy_load=False)
            network = analyzer.create_network()

            # Warn about potential performance issues with large models
            stats = analyzer.get_model_statistics()
            if stats["n_reactions"] > 1000 and (equilibrium_constants is None or relaxation_matrix is None):
                import warnings

                warnings.warn(
                    f"Large model detected ({stats['n_reactions']} reactions). "
                    "Consider providing equilibrium_constants and relaxation_matrix "
                    "for better performance and numerical stability.",
                    UserWarning,
                )
        else:
            # Standard parsing for smaller models
            parser = SBMLParser(model_file)
            network_data = parser.extract_network_data()
            network = ReactionNetwork.from_sbml_data(network_data)

    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .xml/.sbml for SBML or .yml/.yaml for YAML")

    # Create dynamics
    dynamics = LLRQDynamics(network, equilibrium_constants, relaxation_matrix, external_drive)

    # Create solver and visualizer
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(solver)

    return network, dynamics, solver, visualizer


def from_sbml(
    sbml_file: str, equilibrium_constants=None, relaxation_matrix=None, external_drive=None, use_genome_scale_analyzer=None
):
    """Load SBML model and create LLRQ system.

    This is a backward compatibility wrapper around from_model().

    Args:
        sbml_file: Path to SBML file or SBML string content
        equilibrium_constants: Equilibrium constants for each reaction
        relaxation_matrix: Relaxation rate matrix K
        external_drive: External drive function u(t)
        use_genome_scale_analyzer: If True, use GenomeScaleAnalyzer for large models.
                                  If None, auto-detect based on model size.

    Returns:
        Tuple of (network, dynamics, solver, visualizer)
    """
    return from_model(
        sbml_file,
        equilibrium_constants=equilibrium_constants,
        relaxation_matrix=relaxation_matrix,
        external_drive=external_drive,
        use_genome_scale_analyzer=use_genome_scale_analyzer,
        compute_keq_from_thermodynamics=False,  # Don't compute Keq for SBML files
    )


def simple_reaction(
    reactant_species: str = "A",
    product_species: str = "B",
    equilibrium_constant: float = 1.0,
    relaxation_rate: float = 1.0,
    initial_concentrations: Optional[dict] = None,
):
    """Create a simple A ⇌ B reaction system.

    Args:
        reactant_species: Reactant species name
        product_species: Product species name
        equilibrium_constant: Equilibrium constant Keq
        relaxation_rate: Relaxation rate k
        initial_concentrations: Initial concentrations {species: value}

    Returns:
        Tuple of (network, dynamics, solver, visualizer)
    """
    import numpy as np

    if initial_concentrations is None:
        initial_concentrations = {reactant_species: 1.0, product_species: 0.1}

    # Create simple network manually
    species_ids = [reactant_species, product_species]
    reaction_ids = ["R1"]

    # Stoichiometric matrix: A -> B (A: -1, B: +1)
    S = np.array([[-1], [1]])

    # Species info
    species_info = {}
    for species_id in species_ids:
        species_info[species_id] = {
            "name": species_id,
            "initial_concentration": initial_concentrations.get(species_id, 0.0),
            "compartment": "cell",
            "boundary_condition": False,
        }

    # Reaction info
    reaction_info = [
        {
            "id": "R1",
            "name": f"{reactant_species} ⇌ {product_species}",
            "reactants": [(reactant_species, 1.0)],
            "products": [(product_species, 1.0)],
            "reversible": True,
            "kinetic_law": None,
        }
    ]

    # Create network
    network = ReactionNetwork(species_ids, reaction_ids, S, species_info, reaction_info)

    # Create dynamics
    dynamics = LLRQDynamics(
        network, equilibrium_constants=np.array([equilibrium_constant]), relaxation_matrix=np.array([[relaxation_rate]])
    )

    # Create solver and visualizer
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(solver)

    return network, dynamics, solver, visualizer


def simulate_to_target(
    network_or_dynamics,
    initial_concentrations: dict,
    target_concentrations: dict,
    controlled_reactions=None,
    t_span=(0, 100),
    method="linear",
    feedback_gain=1.0,
    **kwargs,
):
    """One-line controlled simulation to target concentrations.

    This function implements the core workflow from linear_vs_mass_action.py:
    1. Setup reaction with initial concentrations
    2. Choose target concentrations (must satisfy conservation laws)
    3. Compute static control input to reach target
    4. Simulate controlled dynamics

    Args:
        network_or_dynamics: ReactionNetwork or LLRQDynamics object
        initial_concentrations: Initial species concentrations (dict)
        target_concentrations: Target species concentrations (dict).
                              Must satisfy same conservation laws as initial.
        controlled_reactions: List of reactions to control. If None, uses all reactions
        t_span: Time span for simulation
        method: Simulation method ('linear' or 'mass_action')
        feedback_gain: Proportional feedback gain
        **kwargs: Additional arguments passed to solve_with_control

    Returns:
        Simulation results dictionary
    """
    # Handle both network and dynamics inputs
    if hasattr(network_or_dynamics, "network"):
        # LLRQDynamics object
        dynamics = network_or_dynamics
        network = dynamics.network
    else:
        # ReactionNetwork object - need to create dynamics
        network = network_or_dynamics
        # Create dynamics using automatic equilibrium computation
        dynamics = LLRQDynamics.from_mass_action(
            network=network,
            # Use default rates that will be reasonable
            forward_rates=kwargs.get("forward_rates", None),
            backward_rates=kwargs.get("backward_rates", None),
            initial_concentrations=list(initial_concentrations.values()),
            **{k: v for k, v in kwargs.items() if k not in ["forward_rates", "backward_rates"]},
        )

    # Create solver
    solver = LLRQSolver(dynamics)

    # Run controlled simulation
    result = solver.solve_with_control(
        initial_conditions=initial_concentrations,
        target_state=target_concentrations,
        t_span=t_span,
        controlled_reactions=controlled_reactions,
        method=method,
        feedback_gain=feedback_gain,
        **kwargs,
    )

    return result


def compare_control_methods(
    network_or_dynamics,
    initial_concentrations: dict,
    target_concentrations: dict,
    controlled_reactions=None,
    t_span=(0, 100),
    feedback_gain=1.0,
    **kwargs,
):
    """Compare linear LLRQ vs mass action control performance.

    Args:
        network_or_dynamics: ReactionNetwork or LLRQDynamics object
        initial_concentrations: Initial species concentrations (dict)
        target_concentrations: Target species concentrations (dict).
                              Must satisfy same conservation laws as initial.
        controlled_reactions: List of reactions to control
        t_span: Time span for simulation
        feedback_gain: Proportional feedback gain
        **kwargs: Additional simulation arguments

    Returns:
        Dictionary with 'linear_result', 'mass_action_result', and 'control_info' keys
    """
    # Handle both network and dynamics inputs
    if hasattr(network_or_dynamics, "network"):
        dynamics = network_or_dynamics
        network = dynamics.network
    else:
        network = network_or_dynamics
        dynamics = LLRQDynamics.from_mass_action(
            network=network,
            forward_rates=kwargs.get("forward_rates", None),
            backward_rates=kwargs.get("backward_rates", None),
            initial_concentrations=list(initial_concentrations.values()),
            **{k: v for k, v in kwargs.items() if k not in ["forward_rates", "backward_rates"]},
        )

    # Create solver
    solver = LLRQSolver(dynamics)

    # Run comparison
    result = solver.solve_with_control(
        initial_conditions=initial_concentrations,
        target_state=target_concentrations,
        t_span=t_span,
        controlled_reactions=controlled_reactions,
        compare_methods=True,
        feedback_gain=feedback_gain,
        **kwargs,
    )

    return result


def create_controlled_simulation(
    network_or_dynamics, controlled_reactions=None, forward_rates=None, backward_rates=None, initial_concentrations=None
):
    """Create a ControlledSimulation object for advanced controlled simulation workflows.

    Args:
        network_or_dynamics: ReactionNetwork or LLRQDynamics object
        controlled_reactions: List of reactions to control
        forward_rates: Forward rate constants (if creating from network)
        backward_rates: Backward rate constants (if creating from network)
        initial_concentrations: Initial concentrations for equilibrium computation

    Returns:
        ControlledSimulation instance
    """
    from .control import ControlledSimulation

    # Handle both network and dynamics inputs
    if hasattr(network_or_dynamics, "network"):
        # LLRQDynamics object
        dynamics = network_or_dynamics
        solver = LLRQSolver(dynamics)
    else:
        # ReactionNetwork object
        network = network_or_dynamics
        if forward_rates is None or backward_rates is None or initial_concentrations is None:
            raise ValueError(
                "forward_rates, backward_rates, and initial_concentrations " "are required when creating from ReactionNetwork"
            )

        return ControlledSimulation.from_mass_action(
            network=network,
            forward_rates=forward_rates,
            backward_rates=backward_rates,
            initial_concentrations=initial_concentrations,
            controlled_reactions=controlled_reactions,
        )

    # Create from dynamics
    from .control import LLRQController

    controller = LLRQController(solver, controlled_reactions)
    return ControlledSimulation(solver, controller)


## CVXPy is a required dependency; helper availability functions removed.


# Package metadata
__all__ = [
    # Core classes
    "SBMLParser",
    "SBMLParseError",
    "ReactionNetwork",
    "LLRQDynamics",
    "LLRQSolver",
    "LLRQVisualizer",
    # Thermodynamic accounting
    "ThermodynamicAccountant",
    "AccountingResult",
    "DualAccountingResult",
    # Control and simulation
    "LLRQController",
    "AdaptiveController",
    "design_lqr_controller",
    "ControlledSimulation",
    "FrequencySpaceController",
    "MassActionSimulator",  # May not be available
    # CVXpy-based control
    "CVXController",
    "CVXObjectives",
    "CVXConstraints",
    "create_entropy_aware_cvx_controller",
    # Genome-scale model utilities
    "GenomeScaleAnalyzer",
    "load_genome_scale_model",
    "compare_model_sizes",
    # Estimation utilities
    "KMatrixEstimator",
    "compute_diffusion_limit",
    "enzyme_to_conductance",
    "validate_physical_consistency",
    "compute_spectral_caps",
    "gershgorin_bounds",
    "estimate_reaction_timescales",
    # Convenience functions
    "from_sbml",
    "simple_reaction",
    "simulate_to_target",
    "compare_control_methods",
    "create_controlled_simulation",
]
