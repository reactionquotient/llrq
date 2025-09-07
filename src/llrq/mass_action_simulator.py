"""Mass action simulator using tellurium for true kinetic dynamics.

This module provides a simulation backend that uses actual mass action kinetics
rather than the LLRQ linearization. This allows testing LLRQ control strategies
on realistic nonlinear dynamics.

The key innovation is the mathematically correct mapping from LLRQ control inputs
to mass action rate modifications via asymmetric rate shifts.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import tellurium as te

    HAS_TELLURIUM = True
except ImportError:
    HAS_TELLURIUM = False

from .integrations.mass_action_drive import apply_llrq_drive_to_rates
from .reaction_network import ReactionNetwork


class MassActionSimulator:
    """Simulate mass action kinetics using tellurium with LLRQ control.

    This provides a realistic simulation backend for testing LLRQ control
    on true nonlinear dynamics. Uses mathematically correct asymmetric
    rate modifications to implement LLRQ control in mass action systems.
    """

    def __init__(
        self,
        network: ReactionNetwork,
        rate_constants: Optional[Dict] = None,
        B: Optional[np.ndarray] = None,
        K_red: Optional[np.ndarray] = None,
        lnKeq_consistent: Optional[np.ndarray] = None,
    ):
        """Initialize mass action simulator with LLRQ control capability.

        Args:
            network: ReactionNetwork object
            rate_constants: Rate constants {reaction_id: (kf, kr)}. If None, uses defaults
            B: LLRQ basis matrix (r x rankS) - required for control
            K_red: Reduced relaxation matrix (rankS x rankS) - required for control
            lnKeq_consistent: Consistent log equilibrium constants (r,) - required for disturbances
        """
        if not HAS_TELLURIUM:
            raise ImportError("tellurium is required for MassActionSimulator. " "Install with: pip install tellurium")

        self.network = network
        self.rate_constants = rate_constants or self._default_rate_constants()
        self.B = B
        self.K_red = K_red
        self.lnKeq_consistent = lnKeq_consistent
        self._model = None
        self._kf_base = None
        self._kr_base = None
        self._build_model()

    @classmethod
    def from_llrq_dynamics(cls, dynamics, network: Optional[ReactionNetwork] = None):
        """Create MassActionSimulator from LLRQDynamics with automatic rate extraction.

        This method extracts rate constants from an LLRQDynamics object that was
        created using from_mass_action(), enabling seamless transition from
        LLRQ dynamics to mass action simulation.

        Args:
            dynamics: LLRQDynamics object created with from_mass_action()
            network: ReactionNetwork object. If None, uses dynamics.network

        Returns:
            MassActionSimulator instance configured with the same rate constants

        Raises:
            ValueError: If dynamics was not created from mass action data
        """
        if network is None:
            network = dynamics.network

        # Extract mass action information
        mass_action_info = dynamics.get_mass_action_info()
        if mass_action_info is None:
            raise ValueError(
                "LLRQDynamics object was not created from mass action data. "
                "Use LLRQDynamics.from_mass_action() to create dynamics with "
                "extractable rate constants."
            )

        # Extract rate constants
        forward_rates = mass_action_info["forward_rates"]
        backward_rates = mass_action_info["backward_rates"]

        rate_constants = {}
        for i, rid in enumerate(network.reaction_ids):
            rate_constants[rid] = (forward_rates[i], backward_rates[i])

        # Create solver to get LLRQ matrices
        from .solver import LLRQSolver

        solver = LLRQSolver(dynamics)

        # Extract LLRQ control matrices
        B = solver._B
        K_red = B.T @ dynamics.K @ B
        lnKeq_consistent = solver._lnKeq_consistent

        return cls(network, rate_constants, B, K_red, lnKeq_consistent)

    @classmethod
    def from_controlled_simulation(cls, controlled_sim):
        """Create MassActionSimulator from ControlledSimulation object.

        Args:
            controlled_sim: ControlledSimulation instance

        Returns:
            MassActionSimulator instance
        """
        return cls.from_llrq_dynamics(controlled_sim.solver.dynamics, controlled_sim.network)

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
            initial = info.get("initial_concentration", 1.0)
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
            reactant_str = " + ".join([f"{coeff}*{species}" if coeff > 1 else species for species, coeff in reactants])
            product_str = " + ".join([f"{coeff}*{species}" if coeff > 1 else species for species, coeff in products])

            if not reactant_str:
                reactant_str = "$null"
            if not product_str:
                product_str = "$null"

            # Kinetic law (mass action with controllable rates)
            forward_terms = " * ".join([f"{species}^{coeff}" if coeff > 1 else species for species, coeff in reactants])
            reverse_terms = " * ".join([f"{species}^{coeff}" if coeff > 1 else species for species, coeff in products])

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
            self._kf_base[i] = self._model[f"kf{i+1}"]
            self._kr_base[i] = self._model[f"kr{i+1}"]

    def get_concentrations(self) -> np.ndarray:
        """Get current species concentrations."""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        return np.array([self._model[sid] for sid in self.network.species_ids])

    def set_concentrations(self, concentrations: Union[Dict, np.ndarray]):
        """Set species concentrations."""
        if isinstance(concentrations, dict):
            conc_array = np.array([concentrations.get(sid, 1.0) for sid in self.network.species_ids])
        else:
            conc_array = np.array(concentrations)

        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
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
            raise ValueError("B and K_red matrices required for LLRQ control. " "Pass them to constructor.")

        # Apply LLRQ control mapping
        kf_new, kr_new = apply_llrq_drive_to_rates(self._kf_base, self._kr_base, self.B, self.K_red, u_red)

        # Update model rate constants
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        for i in range(len(self.network.reaction_ids)):
            self._model[f"kf{i+1}"] = float(kf_new[i])
            self._model[f"kr{i+1}"] = float(kr_new[i])

    def reset_rates(self):
        """Reset rate constants to their base values (remove control)."""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        for i in range(len(self.network.reaction_ids)):
            self._model[f"kf{i+1}"] = float(self._kf_base[i])
            self._model[f"kr{i+1}"] = float(self._kr_base[i])

    def simulate(
        self,
        time_points: np.ndarray,
        control_function: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        disturbance_function: Optional[Callable[[float], np.ndarray]] = None,
    ) -> Dict:
        """Simulate the system over given time points with LLRQ control and state disturbances.

        Args:
            time_points: Array of time points
            control_function: Function f(t, Q) -> (u_red, u_total) that returns both reduced and full control
            disturbance_function: Function f(t) -> d that returns state disturbance

        Returns:
            Dict with 'time', 'concentrations', 'reaction_quotients', 'u_red', 'u_total'
        """
        n_times = len(time_points)
        n_species = len(self.network.species_ids)
        n_reactions = len(self.network.reaction_ids)
        rankS = self.B.shape[1] if self.B is not None else 0

        # Storage
        concentrations = np.zeros((n_times, n_species))
        quotients = np.zeros((n_times, n_reactions))
        controls_red = np.zeros((n_times, rankS)) if rankS > 0 else None
        controls_total = None  # Will be sized based on control function return

        # Initial conditions
        concentrations[0] = self.get_concentrations()
        quotients[0] = self.compute_reaction_quotients()

        # Simulate step by step
        for i in range(1, n_times):
            t_start = time_points[i - 1]
            t_end = time_points[i]

            # Apply control if provided
            u_red = np.zeros(rankS) if rankS > 0 else np.array([])
            u_total = None
            if control_function is not None:
                Q_current = self.compute_reaction_quotients()
                control_result = control_function(t_start, Q_current)

                # Handle both single return (u_red) and tuple return (u_red, u_total)
                if isinstance(control_result, tuple):
                    u_red, u_total = control_result
                else:
                    u_red = control_result
                    u_total = None

                self.apply_llrq_control(u_red)

            # Store controls
            if controls_red is not None:
                controls_red[i] = u_red
            if u_total is not None:
                if controls_total is None:
                    controls_total = np.zeros((n_times, len(u_total)))
                controls_total[i] = u_total

            # Apply state disturbances before simulation step
            if disturbance_function is not None:
                d_state = disturbance_function(t_start)
                if len(d_state) > 0:
                    # Scale disturbance by timestep to represent rate effect
                    # This makes it consistent with linear simulation where d affects derivative
                    dt = t_end - t_start
                    scaled_disturbance = d_state * dt
                    # Convert reduced state disturbance to concentration changes
                    self._apply_state_disturbance(scaled_disturbance)

            # Simulate one time step
            try:
                if self._model is None:
                    raise RuntimeError("Model not initialized. Call setup_model() first.")
                self._model.simulate(t_start, t_end)
            except Exception as e:
                warnings.warn(f"Simulation failed at t={t_start}: {e}")
                # Copy previous values if simulation fails
                concentrations[i] = concentrations[i - 1]
                quotients[i] = quotients[i - 1]
                continue

            # Store results
            concentrations[i] = self.get_concentrations()
            quotients[i] = self.compute_reaction_quotients()

        result = {
            "time": time_points,
            "concentrations": concentrations,
            "reaction_quotients": quotients,
            "method": "Mass Action (Tellurium)",
        }

        if controls_red is not None:
            result["u_red"] = controls_red
        if controls_total is not None:
            result["u_total"] = controls_total

        return result

    def get_current_rates(self) -> Dict[str, Tuple[float, float]]:
        """Get current forward and reverse rate constants.

        Returns:
            Dict mapping reaction_id -> (kf, kr)
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        rates = {}
        for i, rid in enumerate(self.network.reaction_ids):
            kf = self._model[f"kf{i+1}"]
            kr = self._model[f"kr{i+1}"]
            rates[rid] = (kf, kr)
        return rates

    def _apply_state_disturbance(self, d_reduced: np.ndarray):
        """Apply state disturbance in reduced coordinates to species concentrations.

        Uses proper concentration reconstruction with conservation laws.

        Args:
            d_reduced: Disturbance in reduced state coordinates (rankS,)
        """
        if self.B is None:
            warnings.warn("Cannot apply state disturbance without LLRQ basis matrix B")
            return

        if self.lnKeq_consistent is None:
            warnings.warn("Cannot apply state disturbance without consistent equilibrium constants")
            return

        # Get current concentrations
        c_current = self.get_concentrations()

        # Apply disturbance using proper concentration reconstruction
        try:
            from .utils.concentration_utils import apply_state_disturbance_to_concentrations

            c_new = apply_state_disturbance_to_concentrations(
                c_current, d_reduced, self.network, self.B, self.lnKeq_consistent
            )

            # Update model concentrations
            self.set_concentrations(c_new)

        except Exception as e:
            warnings.warn(f"Failed to apply state disturbance: {e}. Disturbance ignored.")
            return

    def simulate_with_controller(
        self,
        controller,
        target_state: Union[np.ndarray, Dict[str, float]],
        t_span: Union[Tuple[float, float], np.ndarray],
        feedback_gain: float = 1.0,
        disturbance_function: Optional[Callable[[float], np.ndarray]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate using LLRQController directly.

        This method provides a convenient interface for mass action simulation
        with LLRQ control, automatically handling the control function setup.

        Args:
            controller: LLRQController instance
            target_state: Target as concentrations dict or reduced state array
            t_span: Time span as tuple or array of time points
            feedback_gain: Proportional feedback gain
            disturbance_function: Optional disturbance function f(t) -> disturbance
            **kwargs: Additional simulation arguments

        Returns:
            Simulation results dictionary
        """
        # Parse time span
        if isinstance(t_span, tuple):
            t_eval = np.linspace(t_span[0], t_span[1], kwargs.get("n_points", 1000))
        else:
            t_eval = np.array(t_span)

        # Parse target state
        if isinstance(target_state, dict):
            c_target = controller.solver._parse_initial_dict(target_state)
            Q_target = self.network.compute_reaction_quotients(c_target)
            y_target = controller.reaction_quotients_to_reduced_state(Q_target)
        elif isinstance(target_state, np.ndarray):
            if len(target_state) == controller.solver._rankS:
                y_target = target_state
            elif len(target_state) == len(self.network.species_ids):
                Q_target = self.network.compute_reaction_quotients(target_state)
                y_target = controller.reaction_quotients_to_reduced_state(Q_target)
            elif len(target_state) == len(self.network.reaction_ids):
                y_target = controller.reaction_quotients_to_reduced_state(target_state)
            else:
                raise ValueError(f"Invalid target_state array length: {len(target_state)}")
        else:
            raise ValueError(f"Invalid target_state type: {type(target_state)}")

        # Compute steady-state control
        u_ss = controller.compute_steady_state_control(y_target)

        # Define control function
        def control_function(t, Q_current):
            y_current = controller.reaction_quotients_to_reduced_state(Q_current)
            u = u_ss + feedback_gain * (y_target - y_current)

            # Convert to reduced control
            u_red = controller.B_red @ u
            return u_red, u

        # Simulate
        result = self.simulate(t_eval, control_function, disturbance_function=disturbance_function)

        # Add target and controller information
        result.update(
            {"target_state": y_target, "steady_state_control": u_ss, "controller": controller, "feedback_gain": feedback_gain}
        )

        return result
