"""Open system reaction network with flow parameters.

This module extends ReactionNetwork to handle open systems with
inlet/outlet flows, enabling moiety dynamics and flow-based control.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
from .reaction_network import ReactionNetwork


class OpenSystemNetwork(ReactionNetwork):
    """Reaction network with flow dynamics for open systems.

    Extends ReactionNetwork to support:
    - Inlet/outlet flows (CSTR, flow reactor, etc.)
    - Dilution and selective removal
    - Flow-dependent moiety dynamics
    """

    def __init__(
        self,
        species_ids: List[str],
        reaction_ids: List[str],
        stoichiometric_matrix: Union[np.ndarray, Any],  # sparse matrix
        flow_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize open system reaction network.

        Args:
            species_ids: List of species identifiers
            reaction_ids: List of reaction identifiers
            stoichiometric_matrix: Stoichiometric matrix S (species x reactions)
            flow_config: Flow configuration dictionary with keys:
                - 'type': 'cstr', 'plug_flow', 'batch_with_feeds', 'membrane'
                - 'volume': Reactor volume (scalar or callable)
                - 'inlet_rate': Volumetric flow rate in (scalar or callable)
                - 'outlet_rate': Volumetric flow rate out (scalar or callable)
                - 'inlet_composition': Species concentrations in inlet
                - 'removal_matrix': Selective removal matrix R
                - 'dilution_rate': D = F_out/V for CSTR
            **kwargs: Additional arguments for ReactionNetwork
        """
        super().__init__(species_ids, reaction_ids, stoichiometric_matrix, **kwargs)

        self.flow_config = flow_config or {}
        self.is_open_system = bool(flow_config)

        # Parse flow configuration
        self._parse_flow_config()

        # Validate flow configuration
        if self.is_open_system:
            self._validate_flow_config()

    def _parse_flow_config(self):
        """Parse and store flow configuration parameters."""
        if not self.is_open_system:
            return

        config = self.flow_config

        # System type
        self.flow_type = config.get("type", "cstr")

        # Volume
        self.volume = config.get("volume", 1.0)  # Default unit volume

        # Flow rates
        self.inlet_rate = config.get("inlet_rate", 0.0)
        self.outlet_rate = config.get("outlet_rate", None)

        # Compositions
        if "inlet_composition" in config:
            inlet_comp = config["inlet_composition"]
            if isinstance(inlet_comp, dict):
                # Convert species name dict to array
                c_in = np.zeros(self.n_species)
                for species, conc in inlet_comp.items():
                    if species in self.species_to_idx:
                        c_in[self.species_to_idx[species]] = conc
                self.inlet_composition = c_in
            else:
                self.inlet_composition = np.asarray(inlet_comp)
        else:
            self.inlet_composition = np.zeros(self.n_species)

        # Dilution rate for CSTR
        if "dilution_rate" in config:
            self.dilution_rate = config["dilution_rate"]
        elif self.flow_type == "cstr" and self.outlet_rate is not None:
            # Compute D = F_out/V
            if callable(self.outlet_rate) or callable(self.volume):
                self.dilution_rate = lambda t: self._get_value(self.outlet_rate, t) / self._get_value(self.volume, t)
            else:
                self.dilution_rate = self.outlet_rate / self.volume
        else:
            self.dilution_rate = 0.0

        # Removal matrix for selective processes
        self.removal_matrix = config.get("removal_matrix", None)
        if self.removal_matrix is not None:
            self.removal_matrix = np.asarray(self.removal_matrix)
            if self.removal_matrix.shape != (self.n_species, self.n_species):
                raise ValueError(f"Removal matrix must be {self.n_species} x {self.n_species}")

    def _get_value(self, param: Union[float, Callable], t: float = 0.0) -> float:
        """Get parameter value (handle callables)."""
        return param(t) if callable(param) else param

    def _validate_flow_config(self):
        """Validate flow configuration for consistency."""
        if self.flow_type == "cstr":
            # CSTR must have outlet rate or dilution rate (unless closed system)
            if self.outlet_rate is None and (not hasattr(self, "dilution_rate") or self.dilution_rate == 0.0):
                # Allow closed CSTR (batch reactor)
                pass

        # Check inlet composition dimensions
        if len(self.inlet_composition) != self.n_species:
            raise ValueError(f"Inlet composition must have {self.n_species} species")

    def is_moiety_respecting_flow(self, tol: float = 1e-10) -> bool:
        """Check if the flow configuration respects moiety structure.

        For flows to be moiety-respecting, the removal operator R must
        satisfy L @ R = A_y @ L for some matrix A_y, where L are the
        conservation law rows.

        Args:
            tol: Numerical tolerance

        Returns:
            True if flow is moiety-respecting
        """
        if not self.is_open_system:
            return True  # Closed system is trivially moiety-respecting

        if self.flow_type == "cstr":
            # CSTR with uniform dilution is always moiety-respecting
            return self.removal_matrix is None

        if self.removal_matrix is None:
            return True

        # Check if L @ R = A_y @ L for some A_y
        L = self.find_conservation_laws()
        if L.shape[0] == 0:
            return True  # No conservation laws to respect

        R = self.removal_matrix
        LR = L @ R

        # Check if LR can be written as A_y @ L
        L_pinv = np.linalg.pinv(L)
        A_y = LR @ L_pinv
        reconstructed = A_y @ L

        return np.allclose(LR, reconstructed, atol=tol)

    def compute_moiety_flow_matrix(self) -> Optional[np.ndarray]:
        """Compute flow matrix for moiety dynamics.

        For moiety-respecting flows, returns A_y such that
        d/dt(L^T c) = -A_y (L^T c) + inlet_terms

        Returns:
            A_y matrix or None if not moiety-respecting
        """
        if not self.is_open_system:
            return np.zeros((0, 0))

        L = self.find_conservation_laws()
        if L.shape[0] == 0:
            return np.zeros((0, 0))

        if self.flow_type == "cstr":
            # CSTR: A_y = D * I
            D = self._get_value(self.dilution_rate)
            return D * np.eye(L.shape[0])

        elif self.removal_matrix is not None:
            if not self.is_moiety_respecting_flow():
                return None  # Not moiety-respecting

            # General case: A_y = L @ R @ L^+
            R = self.removal_matrix
            L_pinv = np.linalg.pinv(L)
            return (L @ R) @ L_pinv

        else:
            # No removal - purely accumulating system
            return np.zeros((L.shape[0], L.shape[0]))

    def compute_moiety_inlet_terms(self, t: float = 0.0) -> np.ndarray:
        """Compute inlet terms for moiety dynamics.

        Args:
            t: Time point (for time-varying flows)

        Returns:
            Inlet contribution to moiety dynamics
        """
        L = self.find_conservation_laws()
        if L.shape[0] == 0:
            return np.array([])

        if not self.is_open_system:
            return np.zeros(L.shape[0])

        # Inlet moiety totals
        y_in = L @ self.inlet_composition

        if self.flow_type == "cstr":
            # CSTR: inlet rate = D * y_in
            D = self._get_value(self.dilution_rate, t)
            return D * y_in
        else:
            # General case: may need more sophisticated calculation
            F_in = self._get_value(self.inlet_rate, t)
            V = self._get_value(self.volume, t)
            return (F_in / V) * y_in

    def get_species_balance_terms(self, t: float = 0.0) -> Dict[str, np.ndarray]:
        """Get species balance terms for open system.

        Species balance: dc/dt = S @ r + inlet_terms - removal_terms

        Args:
            t: Time point

        Returns:
            Dictionary with:
                - 'inlet': Inlet contribution to species balance
                - 'removal': Removal terms (negative for outflow)
        """
        if not self.is_open_system:
            return {"inlet": np.zeros(self.n_species), "removal": np.zeros(self.n_species)}

        # Inlet terms
        F_in = self._get_value(self.inlet_rate, t)
        V = self._get_value(self.volume, t)
        inlet_terms = (F_in / V) * self.inlet_composition

        # Removal terms
        if self.flow_type == "cstr":
            # Uniform dilution
            D = self._get_value(self.dilution_rate, t)
            # Removal = -D * c (will be applied during simulation)
            removal_coeff = D * np.eye(self.n_species)
        elif self.removal_matrix is not None:
            # Selective removal
            removal_coeff = self.removal_matrix
        else:
            removal_coeff = np.zeros((self.n_species, self.n_species))

        return {"inlet": inlet_terms, "removal_matrix": removal_coeff}

    def __repr__(self) -> str:
        """String representation of open system network."""
        base_repr = super().__repr__()

        if not self.is_open_system:
            return base_repr + " (closed system)"

        flow_info = f" (open system: {self.flow_type}"

        if hasattr(self, "dilution_rate"):
            D = self._get_value(self.dilution_rate) if not callable(self.dilution_rate) else "time-varying"
            flow_info += f", D={D}"

        if np.any(self.inlet_composition > 0):
            n_inlet_species = np.sum(self.inlet_composition > 0)
            flow_info += f", {n_inlet_species} inlet species"

        flow_info += ")"

        return base_repr + flow_info


def create_cstr_network(
    species_ids: List[str],
    reaction_ids: List[str],
    stoichiometric_matrix: Union[np.ndarray, Any],
    dilution_rate: Union[float, Callable],
    inlet_composition: Union[np.ndarray, Dict[str, float]],
    **kwargs,
) -> OpenSystemNetwork:
    """Convenience function to create CSTR network.

    Args:
        species_ids: Species identifiers
        reaction_ids: Reaction identifiers
        stoichiometric_matrix: Stoichiometry matrix
        dilution_rate: D = F_out/V
        inlet_composition: Inlet species concentrations
        **kwargs: Additional ReactionNetwork arguments

    Returns:
        OpenSystemNetwork configured as CSTR
    """
    flow_config = {"type": "cstr", "dilution_rate": dilution_rate, "inlet_composition": inlet_composition}

    return OpenSystemNetwork(species_ids, reaction_ids, stoichiometric_matrix, flow_config=flow_config, **kwargs)
