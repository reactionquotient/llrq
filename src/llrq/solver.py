"""Solvers for log-linear reaction quotient dynamics.

This module provides both analytical and numerical solution methods
for the log-linear dynamics system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import warnings
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve, root_scalar
from .llrq_dynamics import LLRQDynamics
from .reaction_network import ReactionNetwork


class LLRQSolver:
    """Solver for log-linear reaction quotient dynamics.
    
    Provides methods for analytical solutions (when possible) and
    robust numerical integration with conservation law enforcement.
    """
    
    def __init__(self, dynamics: LLRQDynamics):
        """Initialize solver with dynamics system.
        
        Args:
            dynamics: LLRQDynamics instance
        """
        self.dynamics = dynamics
        self.network = dynamics.network
    
    def solve(self, 
              initial_conditions: Union[np.ndarray, Dict[str, float]],
              t_span: Union[Tuple[float, float], np.ndarray],
              method: str = 'auto',
              enforce_conservation: bool = True,
              **kwargs) -> Dict[str, Any]:
        """Solve the log-linear dynamics system.
        
        Args:
            initial_conditions: Initial concentrations or reaction quotients
            t_span: Time span as (t0, tf) or array of time points
            method: Solution method ('auto', 'analytical', 'numerical')
            enforce_conservation: Whether to enforce conservation laws
            **kwargs: Additional arguments passed to numerical solver
            
        Returns:
            Dictionary containing solution results
        """
        # Parse initial conditions
        if isinstance(initial_conditions, dict):
            c0 = self._parse_initial_dict(initial_conditions)
        else:
            c0 = np.array(initial_conditions)
        
        # Parse time span
        if isinstance(t_span, tuple):
            t_eval = np.linspace(t_span[0], t_span[1], kwargs.get('n_points', 1000))
        else:
            t_eval = np.array(t_span)
        
        # Compute initial reaction quotients and log deviations
        Q0 = self.network.compute_reaction_quotients(c0)
        x0 = self.dynamics.compute_log_deviation(Q0)
        
        # Validate method
        valid_methods = ['analytical', 'numerical', 'auto']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods are: {valid_methods}")
        
        # Choose solution method
        if method == 'auto':
            method = self._choose_method()
        
        if method == 'analytical':
            try:
                x_t = self.dynamics.analytical_solution(x0, t_eval)
                success = True
                message = "Analytical solution computed successfully"
            except Exception as e:
                warnings.warn(f"Analytical solution failed: {e}. Switching to numerical.")
                x_t, success, message = self._numerical_solve(x0, t_eval, **kwargs)
        else:
            x_t, success, message = self._numerical_solve(x0, t_eval, **kwargs)
        
        # Convert back to reaction quotients
        Q_t = np.zeros_like(x_t)
        for i in range(len(t_eval)):
            Q_t[i] = self.dynamics.compute_reaction_quotients(x_t[i])
        
        # Compute concentrations (if conservation laws allow)
        c_t = self._compute_concentrations(Q_t, c0, enforce_conservation)
        
        return {
            'time': t_eval,
            'concentrations': c_t,
            'reaction_quotients': Q_t,
            'log_deviations': x_t,
            'initial_concentrations': c0,
            'success': success,
            'message': message,
            'method': method
        }
    
    def _parse_initial_dict(self, init_dict: Dict[str, float]) -> np.ndarray:
        """Parse initial conditions from dictionary."""
        # Validate that all species in dict are known
        invalid_species = set(init_dict.keys()) - set(self.network.species_ids)
        if invalid_species:
            raise ValueError(f"Invalid species in initial conditions: {list(invalid_species)}. "
                           f"Valid species are: {self.network.species_ids}")
        
        c0 = np.zeros(self.network.n_species)
        
        for i, species_id in enumerate(self.network.species_ids):
            if species_id in init_dict:
                c0[i] = init_dict[species_id]
            elif species_id in self.network.species_info:
                c0[i] = self.network.species_info[species_id].get('initial_concentration', 0.0)
        
        return c0
    
    def _choose_method(self) -> str:
        """Automatically choose solution method based on system properties."""
        # Use analytical if:
        # 1. External drive is constant or zero
        # 2. K matrix is well-conditioned
        # 3. System is not too large
        
        if self.dynamics.n_reactions > 10:
            return 'numerical'
        
        # Check if external drive is approximately constant
        u0 = self.dynamics.external_drive(0.0)
        u1 = self.dynamics.external_drive(1.0)
        
        if not np.allclose(u0, u1, rtol=1e-3):
            return 'numerical'
        
        # Check condition number of K
        if np.linalg.cond(self.dynamics.K) > 1e8:
            return 'numerical'
        
        return 'analytical'
    
    def _numerical_solve(self, x0: np.ndarray, t_eval: np.ndarray, 
                        **kwargs) -> Tuple[np.ndarray, bool, str]:
        """Solve using numerical integration."""
        try:
            # Default solver options
            options = {
                'method': kwargs.get('integrator', 'RK45'),
                'rtol': kwargs.get('rtol', 1e-6),
                'atol': kwargs.get('atol', 1e-9),
                'max_step': kwargs.get('max_step', np.inf)
            }
            
            # Define RHS function
            def rhs(t, x):
                return self.dynamics.dynamics(t, x)
            
            # Solve ODE
            sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], x0, 
                          t_eval=t_eval, **options)
            
            if sol.success:
                return sol.y.T, True, "Numerical integration successful"
            else:
                return sol.y.T, False, f"Integration failed: {sol.message}"
                
        except Exception as e:
            # Fallback to simpler method
            try:
                def rhs_odeint(x, t):
                    return self.dynamics.dynamics(t, x)
                
                x_t = odeint(rhs_odeint, x0, t_eval, 
                           rtol=kwargs.get('rtol', 1e-6))
                return x_t, True, "Numerical integration successful (odeint)"
                
            except Exception as e2:
                return np.zeros((len(t_eval), len(x0))), False, f"All integration methods failed: {e2}"
    
    def _compute_concentrations(self, Q_t: np.ndarray, c0: np.ndarray,
                               enforce_conservation: bool) -> Optional[np.ndarray]:
        """Compute species concentrations from reaction quotients.
        
        This is the inverse problem: given Q(t), find concentrations c(t)
        that satisfy the conservation laws and Q = f(c).
        """
        if not enforce_conservation:
            # Cannot uniquely determine concentrations without conservation
            return None
        
        # Find conservation laws
        C = self.network.find_conservation_laws()
        conserved_quantities = self.network.compute_conserved_quantities(c0)
        
        if C.shape[0] == 0:
            # No conservation laws - cannot determine concentrations
            warnings.warn("No conservation laws found, cannot compute concentrations")
            return None
        
        n_conserved = C.shape[0]
        n_species = self.network.n_species
        n_reactions = self.network.n_reactions
        
        if n_species - n_conserved != n_reactions:
            warnings.warn(f"System has {n_species} species, {n_conserved} conservation laws, "
                         f"and {n_reactions} reactions. Concentration reconstruction may be underdetermined.")
        
        # Solve for concentrations at each time point
        c_t = np.zeros((len(Q_t), n_species))
        
        for i, Q in enumerate(Q_t):
            try:
                c_t[i] = self._solve_concentrations_at_time(Q, C, conserved_quantities)
            except Exception as e:
                warnings.warn(f"Failed to compute concentrations at time point {i}: {e}")
                c_t[i] = c0  # Fallback to initial concentrations
        
        return c_t
    
    def _solve_concentrations_at_time(self, Q: np.ndarray, C: np.ndarray, 
                                    conserved_quantities: np.ndarray) -> np.ndarray:
        """Solve for concentrations at a single time point."""
        n_species = self.network.n_species
        
        def equations(c):
            # Constraint 1: Conservation laws
            conservation_residual = C @ c - conserved_quantities
            
            # Constraint 2: Reaction quotients
            Q_computed = self.network.compute_reaction_quotients(c)
            quotient_residual = np.log(Q_computed) - np.log(Q)
            
            return np.concatenate([conservation_residual, quotient_residual])
        
        # Initial guess (use reasonable values based on conserved quantities)
        if len(conserved_quantities) == 1:
            # Single conservation law - distribute equally among species
            c_guess = np.ones(n_species) * conserved_quantities[0] / n_species
        else:
            # Multiple conservation laws - use average of conserved quantities
            avg_conserved = np.mean(conserved_quantities)
            c_guess = np.ones(n_species) * avg_conserved / n_species
        
        # Ensure positive values
        c_guess = np.maximum(c_guess, 1e-6)
        
        # Solve nonlinear system
        try:
            solution = fsolve(equations, c_guess, xtol=1e-8)
            
            # Check if solution is physically reasonable
            if np.any(solution < 0):
                # Try with different initial guess - use sum of conserved quantities
                total_conserved = np.sum(conserved_quantities)
                c_guess = np.ones(n_species) * total_conserved / n_species
                c_guess = np.maximum(c_guess, 1e-6)
                solution = fsolve(equations, c_guess, xtol=1e-8)
            
            return np.maximum(solution, 1e-12)  # Ensure positive concentrations
            
        except Exception as e:
            raise ValueError(f"Could not solve for concentrations: {e}")
    
    def solve_single_reaction(self, 
                            reaction_id: str,
                            initial_concentrations: Dict[str, float],
                            t_span: Union[Tuple[float, float], np.ndarray],
                            external_drive: Optional[Callable[[float], float]] = None,
                            **kwargs) -> Dict[str, Any]:
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
            t = np.linspace(t_span[0], t_span[1], kwargs.get('n_points', 1000))
        else:
            t = np.array(t_span)
        
        # Get initial concentrations
        c0 = self._parse_initial_dict(initial_concentrations)
        Q0 = self.network.compute_single_reaction_quotient(reaction_id, c0)
        
        # Get parameters for this reaction
        Keq_j = self.dynamics.Keq[j]
        k_j = self.dynamics.K[j, j]  # Assume diagonal relaxation
        
        # Solve single reaction
        t_out, Q_t = self.dynamics.single_reaction_solution(
            Q0, Keq_j, k_j, external_drive, t
        )
        
        # Compute concentrations using conservation (for single reaction case)
        c_t = self._compute_single_reaction_concentrations(
            reaction_id, Q_t, c0
        )
        
        return {
            'time': t_out,
            'reaction_quotient': Q_t,
            'concentrations': c_t,
            'initial_concentrations': c0,
            'equilibrium_constant': Keq_j,
            'relaxation_rate': k_j
        }
    
    def _compute_single_reaction_concentrations(self, reaction_id: str, 
                                              Q_t: np.ndarray, 
                                              c0: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute concentrations for single reaction case."""
        reactants, products = self.network.get_reaction_stoichiometry(reaction_id)
        
        # For simplest case: A ⇌ B with Q = [B]/[A]
        if (len(reactants) == 1 and len(products) == 1 and 
            list(reactants.values())[0] == 1 and list(products.values())[0] == 1):
            
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
            warnings.warn(f"Concentration reconstruction for reaction {reaction_id} "
                         "not implemented for complex stoichiometry")
            return {}
    
    def compute_steady_state(self, 
                           external_drive: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute steady state of the system.
        
        Args:
            external_drive: Constant external drive u
            
        Returns:
            Dictionary with steady state results
        """
        if external_drive is None:
            u = np.zeros(self.dynamics.n_reactions)
        else:
            u = np.array(external_drive)
        
        try:
            # Steady state: K * x_ss = u
            if np.abs(np.linalg.det(self.dynamics.K)) > 1e-12:
                x_ss = np.linalg.solve(self.dynamics.K, u)
                Q_ss = self.dynamics.compute_reaction_quotients(x_ss)
                
                return {
                    'log_deviations': x_ss,
                    'reaction_quotients': Q_ss,
                    'external_drive': u,
                    'exists': True
                }
            else:
                return {
                    'exists': False,
                    'reason': 'Relaxation matrix K is singular'
                }
                
        except np.linalg.LinAlgError:
            return {
                'exists': False,
                'reason': 'Failed to solve for steady state'
            }