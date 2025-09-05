"""Solvers for log-linear reaction quotient dynamics.

This module provides both analytical and numerical solution methods
for the log-linear dynamics system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import warnings
from scipy.integrate import solve_ivp, odeint
from scipy.linalg import expm
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
        """
        self.dynamics = dynamics
        self.network = dynamics.network

        # --- Build reduced subspace for Im(S^T) (handles cycles) ---
        S = self.network.S  # (n x r)
        U, s, _ = np.linalg.svd(S.T, full_matrices=False)
        tol = max(S.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        rankS = int(np.sum(s > tol))
        self._B = U[:, :rankS]                      # (r x rankS), orthonormal columns
        self._P = self._B @ self._B.T               # projector onto Im(S^T)
        self._rankS = rankS
        # Consistent ln Keq (optional projection, warn if inconsistent)
        lnKeq = np.log(self.dynamics.Keq)
        lnKeq_proj = self._P @ lnKeq
        if not np.allclose(lnKeq, lnKeq_proj, atol=1e-10):
            warnings.warn("ln(Keq) not in Im(S^T); projecting to satisfy Wegscheider identities.")
        self._lnKeq_consistent = lnKeq_proj
        self._Keq_consistent = np.exp(lnKeq_proj)
    
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
        
        # Initial log-deviation x0 = ln(Q/Keq) from concentrations
        Q0 = self.network.compute_reaction_quotients(c0)
        x0 = np.log(Q0) - self._lnKeq_consistent
        # Keep only the physically meaningful part
        x0 = self._P @ x0
        y0 = self._B.T @ x0
        
        # Validate method
        valid_methods = ['analytical', 'numerical', 'auto']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods are: {valid_methods}")
        
        # Reduced operators
        K = self.dynamics.K
        K_red = self._B.T @ K @ self._B
        def u_full(t): return self.dynamics.external_drive(t)
        def u_red(t):  return self._B.T @ u_full(t)

        if method == 'auto':
            # Prefer analytical if u is ~constant and K_red is well-conditioned and small
            if self.dynamics.n_reactions > 10:
                method = 'numerical'
            elif not np.allclose(u_red(0.0), u_red(1.0), rtol=1e-3, atol=1e-9):
                method = 'numerical'
            elif np.linalg.cond(K_red) > 1e8:
                method = 'numerical'
            else:
                method = 'analytical'
        
        if method == 'analytical':
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
    
    def _numerical_solve_reduced(self, y0: np.ndarray, t_eval: np.ndarray, K_red: np.ndarray, u_red: Callable[[float], np.ndarray],
                        **kwargs) -> Tuple[np.ndarray, bool, str]:
        """Solve reduced system using numerical integration."""
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
                return -K_red @ x + u_red(t)
            
            # Solve ODE
            sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, 
                          t_eval=t_eval, **options)
            
            if sol.success:
                return sol.y.T, True, "Numerical integration successful (reduced)"
            else:
                return sol.y.T, False, f"Integration failed: {sol.message}"
                
        except Exception as e:
            # Fallback to simpler method
            try:
                def rhs_odeint(y, t):
                    return (-K_red @ y + u_red(t)).astype(float)
                y_t = odeint(rhs_odeint, y0, t_eval, 
                           rtol=kwargs.get('rtol', 1e-6))
                return y_t, True, "Numerical integration successful (odeint, reduced)"
                
            except Exception as e2:
                return np.zeros((len(t_eval), len(y0))), False, f"All integration methods failed (reduced): {e2}"
    
    def _compute_concentrations_from_reduced(self, Q_t: np.ndarray, c0: np.ndarray,
                               enforce_conservation: bool) -> Optional[np.ndarray]:
        """Reconstruct concentrations using conservation + reduced quotient constraints."""
        if not enforce_conservation:
            return None
        C = self.network.find_conservation_laws()               # (n_c x n)
        cons0 = self.network.compute_conserved_quantities(c0)   # (n_c,)
        if C.shape[0] == 0:
            warnings.warn("No conservation laws found, cannot compute concentrations")
            return None
        S = self.network.S
        B = self._B
        lnKeq = self._lnKeq_consistent
        n = self.network.n_species
        c_t = np.zeros((len(Q_t), n))
        c_guess = np.maximum(c0, 1e-9)
        for i in range(len(Q_t)):
            # y_target = B^T x, with x = ln Q - ln Keq
            y_target = B.T @ (np.log(Q_t[i]) - lnKeq)
            def residual(u):
                c = np.exp(u)
                r1 = C @ c - cons0
                r2 = y_target - (B.T @ (S.T @ u - lnKeq))
                return np.concatenate([r1, r2])
            try:
                u0 = np.log(np.maximum(c_guess, 1e-12))
                u_sol = fsolve(residual, u0, xtol=1e-9, maxfev=2000)
                c_sol = np.maximum(np.exp(u_sol), 1e-12)
                c_t[i] = c_sol
                c_guess = c_sol
            except Exception as e:
                warnings.warn(f"Concentration solve failed at i={i}: {e}; using previous guess.")
                c_t[i] = c_guess
        return c_t
    
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
        
        # Only valid if rank(S) == 1 (one DOF). Otherwise reactions are coupled.
        if self._rankS != 1:
            raise ValueError("solve_single_reaction is only valid when rank(S)=1.")
        Keq_j = self.dynamics.Keq[j]
        K_red = self._B.T @ self.dynamics.K @ self._B
        k_j = float(K_red.squeeze())  # effective scalar relaxation rate
        
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
        """
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
            'log_deviations': x_ss,
            'reaction_quotients': Q_ss,
            'external_drive': u_full,
            'exists': True,
            'method': 'reduced'
        }

    def simulate_closed_loop(self,
                             initial_conditions,
                             t_span,
                             controller,
                             y_ref,
                             **kwargs):
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
            t_eval = np.linspace(t_span[0], t_span[1], kwargs.get('n_points', 1000))
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
            'method': kwargs.get('integrator', 'RK45'),
            'rtol': kwargs.get('rtol', 1e-6),
            'atol': kwargs.get('atol', 1e-9),
            'max_step': kwargs.get('max_step', np.inf)
        }
        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, **options)
    
        y_t = sol.y.T
        x_t = (self._B @ y_t.T).T
        Q_t = self._Keq_consistent * np.exp(x_t)
        c_t = self._compute_concentrations_from_reduced(Q_t, c0, enforce_conservation=True)
    
        return {
            'time': t_eval,
            'concentrations': c_t,
            'reaction_quotients': Q_t,
            'log_deviations': x_t,
            'initial_concentrations': c0,
            'success': bool(sol.success),
            'message': "Closed-loop simulation complete" if sol.success else f"Integration failed: {sol.message}",
        }

