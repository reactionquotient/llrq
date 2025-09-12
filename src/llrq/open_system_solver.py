"""Solver for open system LLRQ dynamics with moiety decomposition.

This module provides analytical solving for open systems using block-triangular
moiety dynamics, enabling efficient simulation of flow reactors, CSTRs, and
systems with selective removal.
"""

from typing import Any, Dict, Optional, Union, Callable, Tuple
import numpy as np
from .solver import LLRQSolver
from .moiety_dynamics import MoietyDynamics
from .open_system_network import OpenSystemNetwork
from .llrq_dynamics import LLRQDynamics


class OpenSystemSolver(LLRQSolver):
    """Solver for open LLRQ systems with flow dynamics.

    Combines MoietyDynamics and OpenSystemNetwork to enable:
    - Analytical solutions for block-triangular systems
    - Flow-based control design
    - Conservation-aware reconstruction
    """

    def __init__(
        self, dynamics: LLRQDynamics, network: Optional[OpenSystemNetwork] = None, flow_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize open system solver.

        Args:
            dynamics: LLRQ dynamics (will be extended to MoietyDynamics)
            network: Open system network (if None, will be created from dynamics.network)
            flow_config: Flow configuration for creating open system network
        """
        # Handle network creation
        if network is None:
            if hasattr(dynamics.network, "flow_config"):
                # Already an open system network - cast to correct type
                self.open_network: OpenSystemNetwork = dynamics.network  # type: ignore
            else:
                # Convert to open system network
                self.open_network = OpenSystemNetwork(
                    dynamics.network.species_ids, dynamics.network.reaction_ids, dynamics.network.S, flow_config=flow_config
                )
        else:
            self.open_network = network

        # Create moiety dynamics
        self.moiety_dynamics = MoietyDynamics(self.open_network, dynamics.K, dynamics.Keq, dynamics.external_drive)

        # Configure flow dynamics based on network
        if self.open_network.is_open_system:
            self._configure_flow_dynamics()

        # Initialize parent solver
        super().__init__(self.moiety_dynamics)

    def _configure_flow_dynamics(self):
        """Configure moiety dynamics based on open system network."""
        network = self.open_network

        if network.flow_type == "cstr":
            # CSTR configuration
            self.moiety_dynamics.configure_cstr(
                dilution_rate=network.dilution_rate, inlet_composition=network.inlet_composition
            )

        elif network.removal_matrix is not None:
            # Selective removal configuration
            inlet_totals = None
            if np.any(network.inlet_composition > 0):
                L = network.find_conservation_laws()
                if L.shape[0] > 0:
                    inlet_totals = L @ network.inlet_composition

            self.moiety_dynamics.configure_moiety_respecting_removal(
                removal_matrix=network.removal_matrix, inlet_totals=inlet_totals
            )

    def solve_analytical(
        self,
        initial_conditions: Union[np.ndarray, Dict[str, float]],
        t_span: Union[Tuple[float, float], np.ndarray],
        u_x: Optional[Union[np.ndarray, Callable]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Solve open system analytically using block-triangular decomposition.

        Args:
            initial_conditions: Initial concentrations
            t_span: Time span or time points
            u_x: Control input for reaction quotients (constant)
            **kwargs: Additional solver options

        Returns:
            Solution dictionary with time, concentrations, quotients, moiety totals
        """
        # Process initial conditions
        if isinstance(initial_conditions, dict):
            c0 = self._dict_to_array(initial_conditions)
        else:
            c0 = np.asarray(initial_conditions)

        # Process time span
        if isinstance(t_span, tuple):
            t0, tf = t_span
            dt = kwargs.get("dt", 0.01)
            t = np.arange(t0, tf + dt, dt)
        else:
            t = np.asarray(t_span)

        # Convert callable u_x to array if needed
        if callable(u_x):
            import warnings

            warnings.warn("Callable u_x not supported in analytical solution. Using u_x=None.")
            u_x_array = None
        else:
            u_x_array = u_x

        # Use analytical simulation from moiety dynamics
        results = self.moiety_dynamics.simulate_analytical(
            t=t, initial_concentrations=c0, u_x=u_x_array, return_concentrations=True
        )

        # Add flow-specific diagnostics
        diagnostics = self._compute_flow_diagnostics(results, t)
        for key, value in diagnostics.items():
            results[key] = value  # type: ignore

        return results

    def _compute_flow_diagnostics(self, results: Dict[str, np.ndarray], t: np.ndarray) -> Dict[str, Union[np.ndarray, bool]]:
        """Compute flow-specific diagnostics."""
        diagnostics: Dict[str, Union[np.ndarray, bool]] = {}

        if not self.open_network.is_open_system:
            return diagnostics

        n_times = len(t)

        # Moiety balance check
        if "y" in results and results["y"].size > 0:
            y_traj = results["y"]

            # Compute expected moiety evolution
            A_y = self.open_network.compute_moiety_flow_matrix()
            if A_y is not None:
                moiety_flow_rate = np.zeros_like(y_traj)
                for i in range(n_times):
                    inlet_terms = self.open_network.compute_moiety_inlet_terms(t[i])
                    if callable(A_y):
                        flow_rate = -A_y(t[i]) @ y_traj[i] + inlet_terms
                    else:
                        flow_rate = -A_y @ y_traj[i] + inlet_terms
                    moiety_flow_rate[i] = flow_rate
                diagnostics["moiety_flow_rate"] = moiety_flow_rate

        # Species flow terms
        if "concentrations" in results:
            c_traj = results["concentrations"]

            inlet_rates = np.zeros((n_times, self.open_network.n_species))
            removal_rates = np.zeros((n_times, self.open_network.n_species))

            for i in range(n_times):
                flow_terms = self.open_network.get_species_balance_terms(t[i])
                inlet_rates[i] = flow_terms["inlet"]

                # Removal rates
                if "removal_matrix" in flow_terms:
                    removal_rates[i] = flow_terms["removal_matrix"] @ c_traj[i]

            diagnostics["inlet_rates"] = inlet_rates
            diagnostics["removal_rates"] = removal_rates

        # Flow decoupling quality
        if self.open_network.is_open_system:
            diagnostics["is_moiety_respecting"] = self.open_network.is_moiety_respecting_flow()

        return diagnostics

    def solve_with_control(
        self,
        initial_conditions: Union[np.ndarray, Dict[str, float]],
        target_state: Union[np.ndarray, Dict[str, float], str],
        t_span: Union[Tuple[float, float], np.ndarray],
        controlled_reactions: Optional[list] = None,
        method: str = "linear",
        compare_methods: bool = False,
        feedback_gain: float = 1.0,
        disturbance_function: Optional[Callable[[float], np.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Solve with feedback control.

        Args:
            initial_conditions: Initial concentrations
            target_state: Target state (array, dict, or string identifier)
            t_span: Time span or time points
            controlled_reactions: Reaction indices for control (unused for open systems)
            method: Control method (unused for open systems)
            compare_methods: Whether to compare methods (unused)
            feedback_gain: Feedback gain (unused for analytical solution)
            disturbance_function: Disturbance function (unused)
            **kwargs: Additional options

        Returns:
            Solution with control inputs and tracking error
        """
        # For now, implement constant control
        # TODO: Extend to time-varying/feedback control

        import warnings

        warnings.warn("Feedback control not yet fully implemented for open systems. Using constant control.")

        # Use constant control for analytical solution
        u_x = kwargs.get("u_x", None)
        results = self.solve_analytical(initial_conditions, t_span, u_x=u_x, **kwargs)

        # Add control diagnostics
        if target_state is not None:
            # Convert target_state to dict format for consistency
            if isinstance(target_state, str):
                # Handle string identifiers - for now just pass through
                target_dict: Dict[str, np.ndarray] = {}
            elif isinstance(target_state, np.ndarray):
                # Assume it's concentration targets
                target_dict = {"concentrations": target_state}
            else:
                target_dict = target_state  # type: ignore

            results.update(self._compute_tracking_error(results, target_dict))

        return results

    def _compute_tracking_error(
        self, results: Dict[str, np.ndarray], target_state: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute tracking error metrics."""
        errors = {}

        if "x" in target_state and "x" in results:
            x_target = target_state["x"]
            x_actual = results["x"]
            errors["x_error"] = x_actual - x_target

        if "y" in target_state and "y" in results:
            y_target = target_state["y"]
            y_actual = results["y"]
            errors["y_error"] = y_actual - y_target

        if "concentrations" in target_state and "concentrations" in results:
            c_target = target_state["concentrations"]
            c_actual = results["concentrations"]
            errors["concentration_error"] = c_actual - c_target

        return errors

    def simulate_step_response(
        self,
        initial_conditions: Union[np.ndarray, Dict[str, float]],
        step_magnitude: Union[float, np.ndarray],
        step_time: float = 0.0,
        final_time: float = 10.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate step response in reaction quotients.

        Args:
            initial_conditions: Initial concentrations
            step_magnitude: Step magnitude for u_x
            step_time: Time when step occurs (for analytical solution, step at t=0)
            final_time: Simulation end time
            **kwargs: Additional options

        Returns:
            Step response results
        """
        if step_time != 0.0:
            import warnings

            warnings.warn("Non-zero step time not supported in analytical solution. Using step at t=0.")

        dt = kwargs.get("dt", 0.01)
        t = np.arange(0, final_time + dt, dt)

        u_x = np.asarray(step_magnitude)

        results = self.solve_analytical(initial_conditions, t, u_x=u_x, **kwargs)

        # Add step response metrics
        results["step_magnitude"] = step_magnitude
        results["settling_time"] = self._estimate_settling_time(results, **kwargs)

        return results

    def _estimate_settling_time(self, results: Dict[str, np.ndarray], settling_tolerance: float = 0.02) -> Optional[float]:
        """Estimate settling time for step response."""
        if "x" not in results:
            return None

        x_traj = results["x"]
        t = results["time"]

        # Use final value as steady state
        x_final = x_traj[-1]

        # Find last time when error exceeds tolerance
        error = np.linalg.norm(x_traj - x_final, axis=1)
        settled_mask = error <= settling_tolerance * np.linalg.norm(x_final)

        if not np.any(settled_mask):
            return None  # Never settled

        # First time it stays settled
        first_settled = np.where(settled_mask)[0][0]

        # Check if it stays settled
        if np.all(settled_mask[first_settled:]):
            return t[first_settled]

        return None  # Oscillatory, doesn't settle

    def _dict_to_array(self, concentration_dict: Dict[str, float]) -> np.ndarray:
        """Convert species concentration dictionary to array."""
        c = np.zeros(self.open_network.n_species)
        for species, conc in concentration_dict.items():
            if species in self.open_network.species_to_idx:
                c[self.open_network.species_to_idx[species]] = conc
        return c
