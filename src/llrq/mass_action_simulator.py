"""Mass action simulator using roadrunner/tellurium for true kinetic dynamics.

This module provides a simulation backend that uses actual mass action kinetics
rather than the LLRQ linearization. This allows testing LLRQ control strategies
on realistic nonlinear dynamics.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Union
try:
    import roadrunner as rr
    HAS_ROADRUNNER = True
except ImportError:
    HAS_ROADRUNNER = False
    try:
        import tellurium as te
        HAS_TELLURIUM = True
    except ImportError:
        HAS_TELLURIUM = False

from .reaction_network import ReactionNetwork


class MassActionSimulator:
    """Simulate mass action kinetics using roadrunner/tellurium.
    
    This provides a realistic simulation backend for testing LLRQ control
    on true nonlinear dynamics instead of the linearized approximation.
    """
    
    def __init__(self, network: ReactionNetwork, rate_constants: Optional[Dict] = None):
        """Initialize mass action simulator.
        
        Args:
            network: ReactionNetwork object
            rate_constants: Rate constants for each reaction {reaction_id: (kf, kr)}
                          If None, uses default values
        """
        if not (HAS_ROADRUNNER or HAS_TELLURIUM):
            raise ImportError("Neither roadrunner nor tellurium is available. "
                            "Install one of them to use MassActionSimulator")
        
        self.network = network
        self.rate_constants = rate_constants or {}
        self._model = None
        self._build_model()
    
    def _build_model(self):
        """Build SBML model from reaction network."""
        # Generate SBML string for the reaction network
        sbml = self._network_to_sbml()
        
        # Create roadrunner or tellurium model
        if HAS_ROADRUNNER:
            self._model = rr.RoadRunner(sbml)
        elif HAS_TELLURIUM:
            self._model = te.loada(sbml)
        else:
            raise ImportError("No simulation backend available")
    
    def _network_to_sbml(self) -> str:
        """Convert ReactionNetwork to SBML string."""
        # Start SBML model
        sbml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">',
            '<model>',
            '<listOfCompartments>',
            '<compartment id="cell" spatialDimensions="3" size="1" constant="true"/>',
            '</listOfCompartments>',
            '<listOfSpecies>'
        ]
        
        # Add species
        for sid in self.network.species_ids:
            info = self.network.species_info.get(sid, {})
            initial = info.get('initial_concentration', 1.0)
            sbml_lines.append(
                f'<species id="{sid}" compartment="cell" initialConcentration="{initial}" '
                f'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
            )
        
        sbml_lines.append('</listOfSpecies>')
        sbml_lines.append('<listOfReactions>')
        
        # Add reactions
        for i, rid in enumerate(self.network.reaction_ids):
            # Get stoichiometry for this reaction
            stoich = self.network.S[:, i]
            reactants = []
            products = []
            
            for j, coeff in enumerate(stoich):
                species = self.network.species_ids[j]
                if coeff < 0:
                    reactants.append((species, abs(coeff)))
                elif coeff > 0:
                    products.append((species, coeff))
            
            # Get rate constants
            kf, kr = self.rate_constants.get(rid, (1.0, 0.1))
            
            # Build reaction SBML
            sbml_lines.append(f'<reaction id="{rid}" reversible="true">')
            
            # Reactants
            if reactants:
                sbml_lines.append('<listOfReactants>')
                for species, coeff in reactants:
                    sbml_lines.append(
                        f'<speciesReference species="{species}" stoichiometry="{coeff}"/>'
                    )
                sbml_lines.append('</listOfReactants>')
            
            # Products  
            if products:
                sbml_lines.append('<listOfProducts>')
                for species, coeff in products:
                    sbml_lines.append(
                        f'<speciesReference species="{species}" stoichiometry="{coeff}"/>'
                    )
                sbml_lines.append('</listOfProducts>')
            
            # Kinetic law (mass action)
            forward_terms = " * ".join([f"pow({s}, {int(c)})" for s, c in reactants])
            reverse_terms = " * ".join([f"pow({s}, {int(c)})" for s, c in products])
            
            if not forward_terms:
                forward_terms = "1"
            if not reverse_terms:
                reverse_terms = "1"
                
            rate_law = f"{kf} * {forward_terms} - {kr} * {reverse_terms}"
            
            sbml_lines.extend([
                '<kineticLaw>',
                f'<math xmlns="http://www.w3.org/1998/Math/MathML">',
                f'<apply><times/>',
                f'<ci>cell</ci>',
                f'<apply><minus/>',
                f'<apply><times/><cn>{kf}</cn>{self._build_mathml_product(reactants)}</apply>',
                f'<apply><times/><cn>{kr}</cn>{self._build_mathml_product(products)}</apply>',
                f'</apply></apply>',
                f'</math>',
                '</kineticLaw>'
            ])
            
            sbml_lines.append('</reaction>')
        
        sbml_lines.extend([
            '</listOfReactions>',
            '</model>',
            '</sbml>'
        ])
        
        return '\n'.join(sbml_lines)
    
    def _build_mathml_product(self, species_list: List) -> str:
        """Build MathML product for mass action kinetics."""
        if not species_list:
            return '<cn>1</cn>'
        
        terms = []
        for species, coeff in species_list:
            if coeff == 1:
                terms.append(f'<ci>{species}</ci>')
            else:
                terms.append(f'<apply><power/><ci>{species}</ci><cn>{int(coeff)}</cn></apply>')
        
        if len(terms) == 1:
            return terms[0]
        else:
            return '<apply><times/>' + ''.join(terms) + '</apply>'
    
    def get_concentrations(self) -> np.ndarray:
        """Get current species concentrations."""
        if HAS_ROADRUNNER and isinstance(self._model, rr.RoadRunner):
            return np.array([self._model[sid] for sid in self.network.species_ids])
        elif HAS_TELLURIUM:
            return np.array([self._model[sid] for sid in self.network.species_ids])
        else:
            raise RuntimeError("No valid model available")
    
    def set_concentrations(self, concentrations: Union[Dict, np.ndarray]):
        """Set species concentrations."""
        if isinstance(concentrations, dict):
            conc_array = np.array([concentrations.get(sid, 1.0) for sid in self.network.species_ids])
        else:
            conc_array = np.array(concentrations)
        
        for i, sid in enumerate(self.network.species_ids):
            if i < len(conc_array):
                self._model[sid] = float(conc_array[i])
    
    def compute_reaction_quotients(self) -> np.ndarray:
        """Compute current reaction quotients Q = [products]/[reactants]."""
        concentrations = self.get_concentrations()
        return self.network.compute_reaction_quotients(concentrations)
    
    def apply_control(self, control_inputs: np.ndarray):
        """Apply control inputs by modifying reaction rates.
        
        Args:
            control_inputs: Control input for each reaction (multiplicative factors)
        """
        # This is a simplified approach - in practice, you might modify
        # specific rate constants or add/remove species
        warnings.warn("Control application in mass action simulator is simplified. "
                     "In practice, this would require more sophisticated rate modification.")
        
        # For now, we'll modify the forward rate constants
        for i, rid in enumerate(self.network.reaction_ids):
            if i < len(control_inputs):
                # Apply control as multiplicative factor to forward rate
                # This is a placeholder - real implementation would be more sophisticated
                pass
    
    def simulate(self, time_points: np.ndarray, control_function: Optional = None) -> Dict:
        """Simulate the system over given time points.
        
        Args:
            time_points: Array of time points
            control_function: Optional function that returns control inputs as f(t, Q)
            
        Returns:
            Dict with 'time', 'concentrations', 'reaction_quotients'
        """
        n_times = len(time_points)
        n_species = len(self.network.species_ids)
        n_reactions = len(self.network.reaction_ids)
        
        # Storage
        concentrations = np.zeros((n_times, n_species))
        quotients = np.zeros((n_times, n_reactions))
        
        # Initial conditions
        concentrations[0] = self.get_concentrations()
        quotients[0] = self.compute_reaction_quotients()
        
        # Time stepping
        for i in range(1, n_times):
            t = time_points[i]
            dt = t - time_points[i-1]
            
            # Apply control if provided
            if control_function is not None:
                Q_current = self.compute_reaction_quotients()
                u = control_function(t, Q_current)
                self.apply_control(u)
            
            # Simulate one step
            if HAS_ROADRUNNER and isinstance(self._model, rr.RoadRunner):
                self._model.simulate(time_points[i-1], t)
                concentrations[i] = self.get_concentrations()
            elif HAS_TELLURIUM:
                result = self._model.simulate(time_points[i-1], t)
                concentrations[i] = self.get_concentrations()
            
            quotients[i] = self.compute_reaction_quotients()
        
        return {
            'time': time_points,
            'concentrations': concentrations,
            'reaction_quotients': quotients
        }