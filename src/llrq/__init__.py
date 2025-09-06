"""Log-Linear Reaction Quotient Dynamics (LLRQ).

A Python package for analyzing chemical reaction networks using the log-linear
reaction quotient dynamics framework described in Diamond (2025).

This framework provides:
- Analytical solutions for reaction quotient evolution
- Tractable analysis of complex networks
- Natural incorporation of thermodynamic constraints
- Linear control theory for biological systems
"""

__version__ = "0.0.2"
__author__ = "Steven Diamond"
__email__ = "steven@gridmatic.com"

# Core imports
from .sbml_parser import SBMLParser, SBMLParseError
from .reaction_network import ReactionNetwork
from .llrq_dynamics import LLRQDynamics
from .solver import LLRQSolver
from .visualization import LLRQVisualizer

# Convenience functions
def from_sbml(sbml_file: str, 
              equilibrium_constants=None,
              relaxation_matrix=None,
              external_drive=None):
    """Load SBML model and create LLRQ system.
    
    Args:
        sbml_file: Path to SBML file or SBML string content
        equilibrium_constants: Equilibrium constants for each reaction
        relaxation_matrix: Relaxation rate matrix K
        external_drive: External drive function u(t)
        
    Returns:
        Tuple of (network, dynamics, solver, visualizer)
    """
    # Parse SBML
    parser = SBMLParser(sbml_file)
    network_data = parser.extract_network_data()
    
    # Create network
    network = ReactionNetwork.from_sbml_data(network_data)
    
    # Create dynamics
    dynamics = LLRQDynamics(network, equilibrium_constants, 
                           relaxation_matrix, external_drive)
    
    # Create solver and visualizer
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(solver)
    
    return network, dynamics, solver, visualizer

def simple_reaction(reactant_species: str = "A", 
                   product_species: str = "B",
                   equilibrium_constant: float = 1.0,
                   relaxation_rate: float = 1.0,
                   initial_concentrations: dict = None):
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
            'name': species_id,
            'initial_concentration': initial_concentrations.get(species_id, 0.0),
            'compartment': 'cell',
            'boundary_condition': False
        }
    
    # Reaction info
    reaction_info = [{
        'id': 'R1',
        'name': f'{reactant_species} ⇌ {product_species}',
        'reactants': [(reactant_species, 1.0)],
        'products': [(product_species, 1.0)],
        'reversible': True,
        'kinetic_law': None
    }]
    
    # Create network
    network = ReactionNetwork(species_ids, reaction_ids, S, 
                             species_info, reaction_info)
    
    # Create dynamics  
    dynamics = LLRQDynamics(network, 
                           equilibrium_constants=np.array([equilibrium_constant]),
                           relaxation_matrix=np.array([[relaxation_rate]]))
    
    # Create solver and visualizer
    solver = LLRQSolver(dynamics)
    visualizer = LLRQVisualizer(solver)
    
    return network, dynamics, solver, visualizer

# Package metadata
__all__ = [
    # Core classes
    'SBMLParser',
    'SBMLParseError', 
    'ReactionNetwork',
    'LLRQDynamics',
    'LLRQSolver',
    'LLRQVisualizer',
    
    # Convenience functions
    'from_sbml',
    'simple_reaction'
]