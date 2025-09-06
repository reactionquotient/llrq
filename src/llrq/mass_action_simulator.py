"""Mass action simulator using tellurium for true kinetic dynamics.

This module provides a simulation backend that uses actual mass action kinetics
rather than the LLRQ linearization. This allows testing LLRQ control strategies
on realistic nonlinear dynamics.

The key innovation is the mathematically correct mapping from LLRQ control inputs
to mass action rate modifications via asymmetric rate shifts.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Union, Callable, Tuple
try:
    import tellurium as te
    HAS_TELLURIUM = True
except ImportError:
    HAS_TELLURIUM = False

from .reaction_network import ReactionNetwork
from .integrations.mass_action_drive import apply_llrq_drive_to_rates


class MassActionSimulator:
    """Simulate mass action kinetics using tellurium with LLRQ control.
    
    This provides a realistic simulation backend for testing LLRQ control
    on true nonlinear dynamics. Uses mathematically correct asymmetric
    rate modifications to implement LLRQ control in mass action systems.
    """
    
    def __init__(self, network: ReactionNetwork, rate_constants: Optional[Dict] = None,
                 B: Optional[np.ndarray] = None, K_red: Optional[np.ndarray] = None):
        """Initialize mass action simulator with LLRQ control capability.
        
        Args:
            network: ReactionNetwork object
            rate_constants: Rate constants {reaction_id: (kf, kr)}. If None, uses defaults
            B: LLRQ basis matrix (r x rankS) - required for control
            K_red: Reduced relaxation matrix (rankS x rankS) - required for control
        """
        if not HAS_TELLURIUM:
            raise ImportError("tellurium is required for MassActionSimulator. "
                            "Install with: pip install tellurium")
        
        self.network = network
        self.rate_constants = rate_constants or self._default_rate_constants()
        self.B = B
        self.K_red = K_red
        self._model = None
        self._kf_base = None
        self._kr_base = None
        self._build_model()
    
    def _default_rate_constants(self) -> Dict:
        """Generate default rate constants for all reactions."""
        constants = {}
        for rid in self.network.reaction_ids:
            # Default: kf=1.0, kr=0.5 (so Keq=2.0)
            constants[rid] = (1.0, 0.5)
        return constants
    
    def _build_model(self):
        """Build tellurium model from reaction network using Antimony."""
        antimony_string = self._network_to_antimony()
        
        try:
            self._model = te.loada(antimony_string)
        except Exception as e:
            raise RuntimeError(f"Failed to create tellurium model: {e}")
        
        # Store base rate constants for control
        self._extract_base_rates()
    
    def _network_to_antimony(self) -> str:
        """Convert ReactionNetwork to Antimony string."""
        lines = ["model mass_action_controlled"]
        lines.append("")
        
        # Species with initial concentrations
        lines.append("  // Species")
        for sid in self.network.species_ids:
            info = self.network.species_info.get(sid, {})
            initial = info.get('initial_concentration', 1.0)
            lines.append(f"  {sid} = {initial};")
        lines.append("")
        
        # Reactions with mass action kinetics
        lines.append("  // Reactions")
        for i, rid in enumerate(self.network.reaction_ids):
            # Get stoichiometry for this reaction
            stoich = self.network.S[:, i]
            reactants = []
            products = []
            
            for j, coeff in enumerate(stoich):
                species = self.network.species_ids[j]
                if coeff < 0:
                    reactants.append((species, abs(int(coeff))))
                elif coeff > 0:
                    products.append((species, int(coeff)))
            
            # Build reaction string
            reactant_str = " + ".join([f"{coeff}*{species}" if coeff > 1 else species 
                                     for species, coeff in reactants])
            product_str = " + ".join([f"{coeff}*{species}" if coeff > 1 else species 
                                    for species, coeff in products])
            
            if not reactant_str:
                reactant_str = "$null"
            if not product_str:
                product_str = "$null"
            
            # Kinetic law (mass action with controllable rates)
            forward_terms = " * ".join([f"{species}^{coeff}" if coeff > 1 else species 
                                      for species, coeff in reactants])
            reverse_terms = " * ".join([f"{species}^{coeff}" if coeff > 1 else species 
                                      for species, coeff in products])
            
            if not forward_terms:
                forward_terms = "1"
            if not reverse_terms:
                reverse_terms = "1"
            
            # Use controllable rate constants
            lines.append(f"  {rid}: {reactant_str} -> {product_str}; kf{i+1} * {forward_terms} - kr{i+1} * {reverse_terms};")
        
        lines.append("")
        
        # Rate constants (will be modified for control)
        lines.append("  // Rate constants")
        for i, rid in enumerate(self.network.reaction_ids):
            kf, kr = self.rate_constants.get(rid, (1.0, 0.5))
            lines.append(f"  kf{i+1} = {kf};")
            lines.append(f"  kr{i+1} = {kr};")
        
        lines.append("")
        lines.append("end")
        
        return "\n".join(lines)
    
    def _extract_base_rates(self):
        """Extract base rate constants from the model."""
        n_reactions = len(self.network.reaction_ids)
        self._kf_base = np.zeros(n_reactions)
        self._kr_base = np.zeros(n_reactions)
        
        for i in range(n_reactions):
            self._kf_base[i] = self._model[f'kf{i+1}']
            self._kr_base[i] = self._model[f'kr{i+1}']
    
    def get_concentrations(self) -> np.ndarray:
        """Get current species concentrations."""
        return np.array([self._model[sid] for sid in self.network.species_ids])
    
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
    
    def apply_llrq_control(self, u_red: np.ndarray):
        """Apply LLRQ control by modifying reaction rates.
        
        Uses the mathematically correct asymmetric rate modification:
        kf' = kf * exp(+δ/2), kr' = kr * exp(-δ/2)
        
        Args:
            u_red: Reduced control input (rankS,)
        """
        if self.B is None or self.K_red is None:
            raise ValueError("B and K_red matrices required for LLRQ control. "
                           "Pass them to constructor.")
        
        # Apply LLRQ control mapping
        kf_new, kr_new = apply_llrq_drive_to_rates(
            self._kf_base, self._kr_base, self.B, self.K_red, u_red
        )
        
        # Update model rate constants
        for i in range(len(self.network.reaction_ids)):
            self._model[f'kf{i+1}'] = float(kf_new[i])
            self._model[f'kr{i+1}'] = float(kr_new[i])
    
    def reset_rates(self):
        """Reset rate constants to their base values (remove control)."""
        for i in range(len(self.network.reaction_ids)):
            self._model[f'kf{i+1}'] = float(self._kf_base[i])
            self._model[f'kr{i+1}'] = float(self._kr_base[i])
    
    def simulate(self, time_points: np.ndarray, 
                control_function: Optional[Callable[[float, np.ndarray], np.ndarray]] = None) -> Dict:
        """Simulate the system over given time points with LLRQ control.
        
        Args:
            time_points: Array of time points
            control_function: Function f(t, Q) -> u_red that returns reduced control input
            
        Returns:
            Dict with 'time', 'concentrations', 'reaction_quotients', 'u_red'
        """
        n_times = len(time_points)
        n_species = len(self.network.species_ids)
        n_reactions = len(self.network.reaction_ids)
        rankS = self.B.shape[1] if self.B is not None else 0
        
        # Storage
        concentrations = np.zeros((n_times, n_species))
        quotients = np.zeros((n_times, n_reactions))
        controls = np.zeros((n_times, rankS)) if rankS > 0 else None
        
        # Initial conditions
        concentrations[0] = self.get_concentrations()
        quotients[0] = self.compute_reaction_quotients()
        
        # Simulate step by step
        for i in range(1, n_times):
            t_start = time_points[i-1]
            t_end = time_points[i]
            
            # Apply control if provided
            u_red = np.zeros(rankS) if rankS > 0 else np.array([])
            if control_function is not None:
                Q_current = self.compute_reaction_quotients()
                u_red = control_function(t_start, Q_current)
                self.apply_llrq_control(u_red)
            
            if controls is not None:
                controls[i] = u_red
            
            # Simulate one time step
            try:
                self._model.simulate(t_start, t_end)
            except Exception as e:
                warnings.warn(f"Simulation failed at t={t_start}: {e}")
                # Copy previous values if simulation fails
                concentrations[i] = concentrations[i-1]
                quotients[i] = quotients[i-1]
                continue
            
            # Store results
            concentrations[i] = self.get_concentrations()
            quotients[i] = self.compute_reaction_quotients()
        
        result = {
            'time': time_points,
            'concentrations': concentrations,
            'reaction_quotients': quotients,
            'method': 'Mass Action (Tellurium)'
        }
        
        if controls is not None:
            result['u_red'] = controls
            
        return result
    
    def get_current_rates(self) -> Dict[str, Tuple[float, float]]:
        """Get current forward and reverse rate constants.
        
        Returns:
            Dict mapping reaction_id -> (kf, kr)
        """
        rates = {}
        for i, rid in enumerate(self.network.reaction_ids):
            kf = self._model[f'kf{i+1}']
            kr = self._model[f'kr{i+1}']
            rates[rid] = (kf, kr)
        return rates