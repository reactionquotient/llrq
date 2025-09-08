"""Solvers for log-linear reaction quotient dynamics.

This module provides both analytical and numerical solution methods
for the log-linear dynamics system.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import expm
from scipy.optimize import fsolve, root_scalar

from .llrq_dynamics import LLRQDynamics
from .reaction_network import ReactionNetwork


class LLRQSolver:
    """Solver for log-linear reaction quotient dynamics.

    Provides methods for analytical solutions (when possible) and
    robust numerical integration with conservation law enforcement.
    """

    def __init__(self, dynamics: LLRQDynamics, simulation_mode: str = "linear"):
        """Initialize solver with dynamics system.

        Args:
            dynamics: LLRQDynamics system
            simulation_mode: 'linear' for LLRQ approximation, 'mass_action' for true kinetics
        """
        self.dynamics = dynamics
        self.network = dynamics.network
        self.simulation_mode = simulation_mode

        # Initialize mass action simulator if requested
        self._mass_action_sim = None
        if simulation_mode == "mass_action":
            try:
                from .mass_action_simulator import MassActionSimulator

                self._mass_action_sim = MassActionSimulator(self.network)
            except ImportError:
                warnings.warn("Mass action simulation not available. Falling back to linear mode.")
                self.simulation_mode = "linear"

        # --- Build reduced subspace for Im(S^T) (handles cycles) ---
        S = self.network.S  # (n x r)
        U, s, _ = np.linalg.svd(S.T, full_matrices=False)
        tol = max(S.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        rankS = int(np.sum(s > tol))
        self._B = U[:, :rankS]  # (r x rankS), orthonormal columns
        self._P = self._B @ self._B.T  # projector onto Im(S^T)
        self._rankS = rankS
        # Consistent ln Keq (optional projection, warn if inconsistent)
        lnKeq = np.log(self.dynamics.Keq)
        lnKeq_proj = self._P @ lnKeq
        if not np.allclose(lnKeq, lnKeq_proj, atol=1e-10):
            warnings.warn("ln(Keq) not in Im(S^T); projecting to satisfy Wegscheider identities.")
        self._lnKeq_consistent = lnKeq_proj
        self._Keq_consistent = np.exp(lnKeq_proj)

    def solve(
        self,
        initial_conditions: Union[np.ndarray, Dict[str, float]],
        t_span: Union[Tuple[float, float], np.ndarray],
        method: str = "auto",
        enforce_conservation: bool = True,
        compute_entropy: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Solve the log-linear dynamics system.

        Args:
            initial_conditions: Initial concentrations or reaction quotients
            t_span: Time span as (t0, tf) or array of time points
            method: Solution method ('auto', 'analytical', 'numerical')
            enforce_conservation: Whether to enforce conservation laws
            compute_entropy: Whether to compute entropy production (requires mass action data)
            **kwargs: Additional arguments passed to numerical solver

        Returns:
            Dictionary containing solution results and optionally entropy accounting
        """
        # Parse initial conditions
        if isinstance(initial_conditions, dict):
            c0 = self._parse_initial_dict(initial_conditions)
        else:
            c0 = np.array(initial_conditions)

        # Parse time span
        if isinstance(t_span, tuple):
            t_eval = np.linspace(t_span[0], t_span[1], kwargs.get("n_points", 1000))
        else:
            t_eval = np.array(t_span)

        # Initial log-deviation x0 = ln(Q/Keq) from concentrations
        Q0 = self.network.compute_reaction_quotients(c0)
        x0 = np.log(Q0) - self._lnKeq_consistent
        # Keep only the physically meaningful part
        x0 = self._P @ x0
        y0 = self._B.T @ x0

        # Validate method
        valid_methods = ["analytical", "numerical", "auto"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods are: {valid_methods}")

        # Reduced operators
        K = self.dynamics.K
        K_red = self._B.T @ K @ self._B

        def u_full(t):
            return self.dynamics.external_drive(t)

        def u_red(t):
            return self._B.T @ u_full(t)

        if method == "auto":
            # Prefer analytical if u is ~constant and K_red is well-conditioned and small
            if self.dynamics.n_reactions > 10:
                method = "numerical"
            elif not np.allclose(u_red(0.0), u_red(1.0), rtol=1e-3, atol=1e-9):
                method = "numerical"
            elif np.linalg.cond(K_red) > 1e8:
                method = "numerical"
            else:
                method = "analytical"

        if method == "analytical":
            try:
                # Constant u: y(t) = e^{-K_red t} y0 + K_red^{-1}(I - e^{-K_red t}) u
                u0 = u_red(0.0)
                if not np.allclose(u0, u_red(1.0), rtol=1e-3, atol=1e-9):
                    raise RuntimeError("External drive not constant; use numerical.")
                y_t = np.zeros((len(t_eval), len(y0)))
                for i, t in enumerate(t_eval - t_eval[0]):
                    Et = expm(-K_red * t)
                    if y0.size:
                        try:
                            corr = np.linalg.solve(K_red, (np.eye(K_red.shape[0]) - Et) @ u0)
                        except np.linalg.LinAlgError:
                            corr = np.linalg.lstsq(K_red, (np.eye(K_red.shape[0]) - Et) @ u0, rcond=None)[0]
                        y_t[i] = Et @ y0 + corr
                    else:
                        y_t[i] = np.zeros(0)
                success, message = True, "Analytical solution (reduced) computed successfully"
            except Exception as e:
                warnings.warn(f"Analytical solution failed: {e}. Switching to numerical.")
                y_t, success, message = self._numerical_solve_reduced(y0, t_eval, K_red, u_red, **kwargs)
        else:
            y_t, success, message = self._numerical_solve_reduced(y0, t_eval, K_red, u_red, **kwargs)

        # Map back to full x and then to Q (FIX: Q = Keq * exp(x))
        x_t = (self._B @ y_t.T).T
        Q_t = self._Keq_consistent * np.exp(x_t)

        # Compute concentrations (square system: conservation + reduced quotient constraints)
        c_t = self._compute_concentrations_from_reduced(Q_t, c0, enforce_conservation)

        # Prepare results dictionary
        results = {
            "time": t_eval,
            "concentrations": c_t,
            "reaction_quotients": Q_t,
            "log_deviations": x_t,
            "initial_concentrations": c0,
            "success": success,
            "message": message,
            "method": method,
        }

        # Compute entropy production if requested
        if compute_entropy:
            entropy_result = self._compute_entropy_production(t_eval, x_t, u_full, K, c0, kwargs.get("entropy_scale", 1.0))
            if entropy_result is not None:
                results["entropy_accounting"] = entropy_result

        return results

    def _parse_initial_dict(self, init_dict: Dict[str, float]) -> np.ndarray:
        """Parse initial conditions from dictionary."""
        # Validate that all species in dict are known
        invalid_species = set(init_dict.keys()) - set(self.network.species_ids)
        if invalid_species:
            raise ValueError(
                f"Invalid species in initial conditions: {list(invalid_species)}. "
                f"Valid species are: {self.network.species_ids}"
            )

        c0 = np.zeros(self.network.n_species)

        for i, species_id in enumerate(self.network.species_ids):
            if species_id in init_dict:
                c0[i] = init_dict[species_id]
            elif species_id in self.network.species_info:
                c0[i] = self.network.species_info[species_id].get("initial_concentration", 0.0)

        return c0

    def _compute_entropy_production(
        self, t: np.ndarray, x_t: np.ndarray, u_full_func: Callable, K: np.ndarray, c0: np.ndarray, scale: float = 1.0
    ) -> Optional[Any]:
        """Compute entropy production from solution trajectory.

        Args:
            t: Time points
            x_t: Log deviation trajectory
            u_full_func: External drive function
            K: Relaxation matrix
            c0: Initial concentrations
            scale: Physical scale factor

        Returns:
            Entropy accounting result or None if not available
        """
        try:
            # Check if we have mass action data for Onsager conductance
            if not hasattr(self.dynamics, "_forward_rates") or self.dynamics._forward_rates is None:
                warnings.warn("Entropy computation requires mass action data (forward/backward rates)")
                return None

            from .thermodynamic_accounting import ThermodynamicAccountant

            # Create accountant and compute Onsager conductance
            accountant = ThermodynamicAccountant(self.network)
            forward_rates = self.dynamics._forward_rates
            backward_rates = self.dynamics._backward_rates

            # Use initial concentrations for Onsager computation (could be improved)
            L = accountant.compute_onsager_conductance(c0, forward_rates, backward_rates)

            # Compute external drive trajectory
            u_t = np.array([u_full_func(t_i) for t_i in t])

            # Compute entropy from both reaction forces and drives
            if u_t.shape[1] > 0 and not np.allclose(u_t, 0):
                # Have non-zero drives, do dual accounting
                return accountant.entropy_from_xu(t, x_t, u_t, K, L, scale=scale)
            else:
                # Only reaction forces available
                return accountant.entropy_from_x(t, x_t, L, scale=scale)

        except ImportError:
            warnings.warn("Thermodynamic accounting module not available")
            return None
        except Exception as e:
            warnings.warn(f"Entropy computation failed: {e}")
            return None

    def _choose_method(self) -> str:
        """Automatically choose solution method based on system properties."""
        # Use analytical if:
        # 1. External drive is constant or zero
        # 2. K matrix is well-conditioned
        # 3. System is not too large

        if self.dynamics.n_reactions > 10:
            return "numerical"

        # Check if external drive is approximately constant
        u0 = self.dynamics.external_drive(0.0)
        u1 = self.dynamics.external_drive(1.0)

        if not np.allclose(u0, u1, rtol=1e-3):
            return "numerical"

        # Check condition number of K
        if np.linalg.cond(self.dynamics.K) > 1e8:
            return "numerical"

        return "analytical"

    def _numerical_solve_reduced(
        self, y0: np.ndarray, t_eval: np.ndarray, K_red: np.ndarray, u_red: Callable[[float], np.ndarray], **kwargs
    ) -> Tuple[np.ndarray, bool, str]:
        """Solve reduced system using numerical integration."""
        try:
            # Default solver options
            options = {
                "method": kwargs.get("integrator", "RK45"),
                "rtol": kwargs.get("rtol", 1e-6),
                "atol": kwargs.get("atol", 1e-9),
                "max_step": kwargs.get("max_step", np.inf),
            }

            # Define RHS function
            def rhs(t, x):
                return -K_red @ x + u_red(t)

            # Solve ODE
            sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, **options)

            if sol.success:
                return sol.y.T, True, "Numerical integration successful (reduced)"
            else:
                return sol.y.T, False, f"Integration failed: {sol.message}"

        except Exception as e:
            # Fallback to simpler method
            try:

                def rhs_odeint(y, t):
                    return (-K_red @ y + u_red(t)).astype(float)

                y_t = odeint(rhs_odeint, y0, t_eval, rtol=kwargs.get("rtol", 1e-6))
                return y_t, True, "Numerical integration successful (odeint, reduced)"

            except Exception as e2:
                return np.zeros((len(t_eval), len(y0))), False, f"All integration methods failed (reduced): {e2}"

    def _compute_concentrations_from_reduced(
        self, Q_t: np.ndarray, c0: np.ndarray, enforce_conservation: bool
    ) -> Optional[np.ndarray]:
        """Reconstruct concentrations using conservation + reduced quotient constraints."""
        from .utils.concentration_utils import compute_concentrations_from_quotients

        return compute_concentrations_from_quotients(
            Q_t, c0, self.network, self._B, self._lnKeq_consistent, enforce_conservation
        )

    def solve_single_reaction(
        self,
        reaction_id: str,
        initial_concentrations: Dict[str, float],
        t_span: Union[Tuple[float, float], np.ndarray],
        external_drive: Optional[Callable[[float], float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Solve dynamics for a single reaction.

        Args:
            reaction_id: Identifier for the reaction
            initial_concentrations: Initial species concentrations
            t_span: Time span or evaluation points
            external_drive: External drive function u(t) for this reaction

        Returns:
            Dictionary with solution results
        """
        if reaction_id not in self.network.reaction_to_idx:
            raise ValueError(f"Reaction '{reaction_id}' not found")

        j = self.network.reaction_to_idx[reaction_id]

        # Parse time span
        if isinstance(t_span, tuple):
            t = np.linspace(t_span[0], t_span[1], kwargs.get("n_points", 1000))
        else:
            t = np.array(t_span)

        # Get initial concentrations
        c0 = self._parse_initial_dict(initial_concentrations)
        Q0 = self.network.compute_single_reaction_quotient(reaction_id, c0)

        # Only valid if rank(S) == 1 (one DOF). Otherwise reactions are coupled.
        if self._rankS != 1:
            raise ValueError("solve_single_reaction is only valid when rank(S)=1.")
        Keq_j = self.dynamics.Keq[j]
        K_red = self._B.T @ self.dynamics.K @ self._B
        k_j = float(K_red.squeeze())  # effective scalar relaxation rate

        # Solve single reaction
        t_out, Q_t = self.dynamics.single_reaction_solution(Q0, Keq_j, k_j, external_drive, t)

        # Compute concentrations using conservation (for single reaction case)
        c_t = self._compute_single_reaction_concentrations(reaction_id, Q_t, c0)

        return {
            "time": t_out,
            "reaction_quotient": Q_t,
            "concentrations": c_t,
            "initial_concentrations": c0,
            "equilibrium_constant": Keq_j,
            "relaxation_rate": k_j,
        }

    def _compute_single_reaction_concentrations(
        self, reaction_id: str, Q_t: np.ndarray, c0: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute concentrations for single reaction case."""
        reactants, products = self.network.get_reaction_stoichiometry(reaction_id)

        # For simplest case: A ⇌ B with Q = [B]/[A]
        if len(reactants) == 1 and len(products) == 1 and list(reactants.values())[0] == 1 and list(products.values())[0] == 1:
            reactant_id = list(reactants.keys())[0]
            product_id = list(products.keys())[0]

            reactant_idx = self.network.species_to_idx[reactant_id]
            product_idx = self.network.species_to_idx[product_id]

            # Total concentration conserved: [A] + [B] = [A]0 + [B]0
            C_total = c0[reactant_idx] + c0[product_idx]

            # Q = [B]/[A], [A] + [B] = C_total
            # So [A] = C_total/(1 + Q), [B] = C_total*Q/(1 + Q)
            A_t = C_total / (1 + Q_t)
            B_t = C_total * Q_t / (1 + Q_t)

            concentrations = {}
            concentrations[reactant_id] = A_t
            concentrations[product_id] = B_t

            return concentrations

        else:
            warnings.warn(
                f"Concentration reconstruction for reaction {reaction_id} " "not implemented for complex stoichiometry"
            )
            return {}

    def compute_steady_state(self, external_drive: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute steady state of the system."""
        u_full = np.zeros(self.dynamics.n_reactions) if external_drive is None else np.array(external_drive)
        u_red = self._B.T @ u_full
        K_red = self._B.T @ self.dynamics.K @ self._B
        try:
            y_ss = np.linalg.solve(K_red, u_red)
        except np.linalg.LinAlgError:
            y_ss = np.linalg.lstsq(K_red, u_red, rcond=None)[0]
        x_ss = self._B @ y_ss
        Q_ss = self._Keq_consistent * np.exp(x_ss)
        return {
            "log_deviations": x_ss,
            "reaction_quotients": Q_ss,
            "external_drive": u_full,
            "exists": True,
            "method": "reduced",
        }

    def simulate_closed_loop(self, initial_conditions, t_span, controller, y_ref, **kwargs):
        """
        Integrate reduced dynamics with state feedback u_full = controller.u_full(t, y, y_ref).
        Returns the same structure as .solve(...).
        """
        # Parse ICs and time
        if isinstance(initial_conditions, dict):
            c0 = self._parse_initial_dict(initial_conditions)
        else:
            c0 = np.array(initial_conditions, float)

        if isinstance(t_span, tuple):
            t_eval = np.linspace(t_span[0], t_span[1], kwargs.get("n_points", 1000))
        else:
            t_eval = np.array(t_span, float)

        # Build reduced IC y0 from c0
        Q0 = self.network.compute_reaction_quotients(c0)
        x0 = np.log(Q0) - self._lnKeq_consistent
        x0 = self._P @ x0
        y0 = self._B.T @ x0

        # Reduced matrices
        K_red = self._B.T @ self.dynamics.K @ self._B
        A = -K_red

        # Exogenous drive projected
        def d_red(t):
            return self._B.T @ self.dynamics.external_drive(t)

        # RHS
        def rhs(t, y):
            u_full_ctrl = controller.u_full(t, y, np.array(y_ref, float))
            return A @ y + self._B.T @ u_full_ctrl + d_red(t)

        # Integrate
        options = {
            "method": kwargs.get("integrator", "RK45"),
            "rtol": kwargs.get("rtol", 1e-6),
            "atol": kwargs.get("atol", 1e-9),
            "max_step": kwargs.get("max_step", np.inf),
        }
        from scipy.integrate import solve_ivp

        sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, **options)

        y_t = sol.y.T
        x_t = (self._B @ y_t.T).T
        Q_t = self._Keq_consistent * np.exp(x_t)
        c_t = self._compute_concentrations_from_reduced(Q_t, c0, enforce_conservation=True)

        return {
            "time": t_eval,
            "concentrations": c_t,
            "reaction_quotients": Q_t,
            "log_deviations": x_t,
            "initial_concentrations": c0,
            "success": bool(sol.success),
            "message": "Closed-loop simulation complete" if sol.success else f"Integration failed: {sol.message}",
        }

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
        """Solve dynamics with automatic control to reach target concentrations.

        This is the main entry point for controlled simulations, implementing
        the core workflow: setup → target → control → simulate.

        Args:
            initial_conditions: Initial species concentrations (dict or array)
            target_state: Target concentrations (dict or array) or 'equilibrium'.
                         Target concentrations must satisfy conservation laws.
            t_span: Time span as (t0, tf) or array of time points
            controlled_reactions: List of reaction IDs/indices to control.
                                If None, uses all reactions
            method: Simulation method ('linear' or 'mass_action')
            compare_methods: If True, return both linear and mass action results
            feedback_gain: Proportional feedback gain for control
            disturbance_function: Function f(t) -> disturbance for testing robustness
            **kwargs: Additional arguments passed to simulation methods

        Returns:
            Dictionary containing solution results. If compare_methods=True,
            contains both 'linear_result' and 'mass_action_result' keys.
        """
        from .control import LLRQController

        # Parse initial conditions
        if isinstance(initial_conditions, dict):
            c0 = self._parse_initial_dict(initial_conditions)
        else:
            c0 = np.array(initial_conditions)

        # Create controller
        controller = LLRQController(self, controlled_reactions)

        # Parse target state
        y_target = self._parse_target_state(target_state, c0)

        # Compute steady-state control
        u_ss = controller.compute_steady_state_control(y_target)

        # Parse time span
        if isinstance(t_span, tuple):
            t_eval = np.linspace(t_span[0], t_span[1], kwargs.get("n_points", 1000))
        else:
            t_eval = np.array(t_span)

        results = {}

        # Determine simulation methods to use
        if compare_methods:
            methods_to_run = ["linear", "mass_action"]
        elif method in ["linear", "mass_action"]:
            methods_to_run = [method]
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'linear' or 'mass_action'")

        # Run simulations
        for sim_method in methods_to_run:
            if sim_method == "linear":
                result = self._solve_linear_with_control(
                    controller, y_target, u_ss, c0, t_eval, feedback_gain, disturbance_function, **kwargs
                )
                result["method"] = "Linear LLRQ"

            elif sim_method == "mass_action":
                result = self._solve_mass_action_with_control(
                    controller, y_target, u_ss, c0, t_eval, feedback_gain, disturbance_function, **kwargs
                )
                result["method"] = "Mass Action"

            # Store result
            if compare_methods:
                results[f"{sim_method}_result"] = result
            else:
                results = result

        # Add control information
        if compare_methods:
            results["control_info"] = {
                "controller": controller,
                "target_state": y_target,
                "steady_state_control": u_ss,
                "controlled_reactions": controller.controlled_reactions,
            }
        else:
            results.update(
                {
                    "controller": controller,
                    "target_state": y_target,
                    "steady_state_control": u_ss,
                    "controlled_reactions": controller.controlled_reactions,
                }
            )

        return results

    def validate_conservation_laws(self, c_initial: np.ndarray, c_target: np.ndarray, rtol: float = 1e-6) -> Tuple[bool, str]:
        """Validate that target concentrations satisfy conservation laws.

        This ensures that the target concentrations are thermodynamically consistent
        with the initial concentrations - they must conserve the same quantities.

        Args:
            c_initial: Initial species concentrations
            c_target: Target species concentrations
            rtol: Relative tolerance for conservation check

        Returns:
            (is_valid, error_message): Tuple with validation result and error description
        """
        C = self.network.find_conservation_laws()
        if C.shape[0] == 0:
            return True, ""  # No conservation laws to check

        cons_initial = C @ c_initial
        cons_target = C @ c_target

        if not np.allclose(cons_initial, cons_target, rtol=rtol):
            violations = cons_target - cons_initial
            max_violation = np.max(np.abs(violations))

            # Create helpful error message
            error_msg = f"Target concentrations violate conservation laws.\n"
            error_msg += f"Maximum violation: {max_violation:.6e} (tolerance: {rtol:.6e})\n"
            error_msg += f"Initial conserved quantities: {cons_initial}\n"
            error_msg += f"Target conserved quantities:  {cons_target}\n"
            error_msg += f"Violations: {violations}"

            return False, error_msg

        return True, ""

    def _parse_target_state(self, target_state, c0: np.ndarray) -> np.ndarray:
        """Parse target concentrations and convert to reduced state coordinates.

        Target must be species concentrations that satisfy conservation laws.
        The workflow is: concentrations → validate conservation → quotients → reduced state

        Args:
            target_state: Target concentrations (dict or array) or 'equilibrium'
            c0: Initial concentrations for conservation validation

        Returns:
            Target in reduced state coordinates (y)
        """
        # Special case: equilibrium target
        if isinstance(target_state, str):
            if target_state == "equilibrium":
                # Equilibrium corresponds to y = 0 (no deviation from equilibrium)
                return np.zeros(self._rankS)
            else:
                raise ValueError(f"Unknown target state '{target_state}'. Only 'equilibrium' is supported.")

        # Parse concentrations from dict or array
        if isinstance(target_state, dict):
            c_target = self._parse_initial_dict(target_state)
        elif isinstance(target_state, np.ndarray):
            if len(target_state) != len(self.network.species_ids):
                raise ValueError(
                    f"Target concentrations must have {len(self.network.species_ids)} "
                    f"elements (one per species), got {len(target_state)}"
                )
            c_target = target_state
        else:
            raise ValueError(
                "Target must be concentration dict, concentration array, or 'equilibrium'. "
                "Quotients and reduced states are not accepted - use concentrations."
            )

        # Validate conservation laws
        is_valid, error_msg = self.validate_conservation_laws(c0, c_target)
        if not is_valid:
            raise ValueError(f"Target violates conservation laws.\n{error_msg}")

        # Convert concentrations to reduced state: c → Q → x → y
        Q_target = self.network.compute_reaction_quotients(c_target)
        x_target = np.log(Q_target) - self._lnKeq_consistent
        y_target = self._B.T @ self._P @ x_target

        return y_target

    def _solve_linear_with_control(
        self, controller, y_target, u_ss, c0, t_eval, feedback_gain, disturbance_function, **kwargs
    ):
        """Solve using linear LLRQ dynamics with control."""
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.01
        n = len(t_eval)

        # Get system matrices
        A = controller.A  # -K_red
        B_red = controller.B_red

        # Initial state
        Q0 = self.network.compute_reaction_quotients(c0)
        y0 = controller.reaction_quotients_to_reduced_state(Q0)

        # Storage
        Y = np.zeros((n, len(y0)))
        U = np.zeros((n, len(controller.controlled_reactions)))
        Q_traj = np.zeros((n, len(self.network.reaction_ids)))

        y = y0.copy()

        # Simulate
        for i, t in enumerate(t_eval):
            # Control (feedforward + feedback)
            u = u_ss + feedback_gain * (y_target - y)

            # Disturbance
            d = disturbance_function(t) if disturbance_function else np.zeros_like(y)

            # Dynamics
            u_red = B_red @ u
            ydot = A @ y + u_red + d

            # Store
            Y[i] = y
            U[i] = u
            Q_traj[i] = controller.reduced_state_to_reaction_quotients(y)

            # Integrate
            if i < n - 1:
                y = y + dt * ydot

        # Compute concentrations
        C_traj = np.zeros((n, len(self.network.species_ids)))
        for i in range(n):
            C_traj[i] = self._compute_concentrations_from_reduced(Q_traj[i : i + 1], c0, enforce_conservation=True)

        return {
            "time": t_eval,
            "concentrations": C_traj,
            "reaction_quotients": Q_traj,
            "reduced_state": Y,
            "control_inputs": U,
            "success": True,
            "message": "Linear LLRQ controlled simulation completed",
        }

    def _solve_mass_action_with_control(
        self, controller, y_target, u_ss, c0, t_eval, feedback_gain, disturbance_function, **kwargs
    ):
        """Solve using mass action dynamics with control."""
        try:
            from .mass_action_simulator import MassActionSimulator
        except ImportError:
            raise ImportError("Mass action simulation requires tellurium. " "Install with: pip install tellurium")

        # Get rate constants from dynamics if available
        mass_action_info = self.dynamics.get_mass_action_info()
        if mass_action_info:
            rate_constants = {}
            for i, rid in enumerate(self.network.reaction_ids):
                kf = mass_action_info["forward_rates"][i]
                kr = mass_action_info["backward_rates"][i]
                rate_constants[rid] = (kf, kr)
        else:
            # Use default rate constants
            rate_constants = None

        # Create simulator
        K_red = self._B.T @ self.dynamics.K @ self._B
        sim = MassActionSimulator(
            self.network, rate_constants, B=self._B, K_red=K_red, lnKeq_consistent=self._lnKeq_consistent
        )

        # Define control function
        def control_function(t, Q_current):
            y_current = controller.reaction_quotients_to_reduced_state(Q_current)
            u = u_ss + feedback_gain * (y_target - y_current)

            # Convert to reduced control
            u_red = controller.B_red @ u
            return u_red, u

        # Define disturbance function for mass action
        def mass_action_disturbance(t):
            if disturbance_function:
                return disturbance_function(t)
            else:
                return np.zeros(self._rankS)

        # Simulate
        result = sim.simulate(t_eval, control_function, disturbance_function=mass_action_disturbance)

        # Add reduced state trajectory
        n = len(t_eval)
        Y = np.zeros((n, self._rankS))
        for i in range(n):
            Y[i] = controller.reaction_quotients_to_reduced_state(result["reaction_quotients"][i])
        result["reduced_state"] = Y

        return result

    def get_reduced_system_matrices(self, controlled_reactions=None):
        """Get reduced system matrices for control design.

        This method provides the reduced system matrices K_red and B_red needed
        for frequency-domain control design. These matrices define the reduced
        dynamics: dy/dt = -K_red * y + B_red * u, where y is the reduced state.

        Args:
            controlled_reactions: List of reaction IDs or indices to control.
                                If None, returns matrices for controlling all reactions.
                                Can be strings (reaction IDs) or integers (indices).

        Returns:
            tuple: (K_red, B_red) where:
                - K_red: Reduced system matrix (rankS x rankS)
                - B_red: Reduced input matrix (rankS x m) where m = len(controlled_reactions)

        Example:
            # Control all reactions
            K_red, B_red = solver.get_reduced_system_matrices()

            # Control specific reactions
            K_red, B_red = solver.get_reduced_system_matrices(["R1", "R3"])
        """
        # Reduced system matrix (same regardless of which reactions controlled)
        K_red = self._B.T @ self.dynamics.K @ self._B

        if controlled_reactions is None:
            # Control all reactions - B_red is just B^T
            B_red = self._B.T
        else:
            # Build selection matrix G for specified reactions
            r = len(self.network.reaction_ids)
            m = len(controlled_reactions)
            G = np.zeros((r, m))

            for j, rid in enumerate(controlled_reactions):
                if isinstance(rid, str):
                    if rid not in self.network.reaction_to_idx:
                        raise ValueError(f"Unknown reaction ID: {rid}")
                    idx = self.network.reaction_to_idx[rid]
                else:
                    idx = int(rid)
                    if idx < 0 or idx >= r:
                        raise ValueError(f"Reaction index {idx} out of range [0, {r-1}]")
                G[idx, j] = 1.0

            # Reduced input matrix: B_red = B^T @ G
            B_red = self._B.T @ G

        return K_red, B_red
