"""Core log-linear reaction quotient dynamics implementation.

This module implements the log-linear framework for reaction quotient dynamics
described in Diamond (2025) "Log-Linear Reaction Quotient Dynamics".
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .reaction_network import ReactionNetwork


class LLRQDynamics:
    """Log-linear reaction quotient dynamics system.

    Implements the core dynamics:
    d/dt ln Q = -K ln(Q/Keq) + u(t)

    where Q is the vector of reaction quotients, K is the relaxation rate matrix,
    Keq are equilibrium constants, and u(t) are external drives.
    """

    def __init__(
        self,
        network: ReactionNetwork,
        equilibrium_constants: Optional[np.ndarray] = None,
        relaxation_matrix: Optional[np.ndarray] = None,
        external_drive: Optional[Callable[[float], np.ndarray]] = None,
    ):
        """Initialize log-linear dynamics system.

        Args:
            network: Reaction network
            equilibrium_constants: Equilibrium constants Keq for each reaction
            relaxation_matrix: Relaxation rate matrix K
            external_drive: Function u(t) returning external drives
        """
        self.network = network
        self.n_reactions = network.n_reactions

        # Set equilibrium constants
        if equilibrium_constants is None:
            self.Keq = np.ones(self.n_reactions)
            warnings.warn("No equilibrium constants provided, using Keq = 1 for all reactions")
        else:
            self.Keq = np.array(equilibrium_constants)
            if len(self.Keq) != self.n_reactions:
                raise ValueError(f"Expected {self.n_reactions} equilibrium constants, " f"got {len(self.Keq)}")

        # Set relaxation matrix
        if relaxation_matrix is None:
            self.K = np.eye(self.n_reactions)
            warnings.warn("No relaxation matrix provided, using K = I")
        else:
            self.K = np.array(relaxation_matrix)
            if self.K.shape != (self.n_reactions, self.n_reactions):
                raise ValueError(
                    f"Expected relaxation matrix shape " f"{(self.n_reactions, self.n_reactions)}, " f"got {self.K.shape}"
                )

        self.external_drive = external_drive or self._zero_drive

        # Initialize mass action attributes (set by from_mass_action classmethod)
        self._mass_action_data: Optional[Dict[str, Any]] = None
        self._mass_action_mode: Optional[str] = None
        self._equilibrium_point: Optional[np.ndarray] = None
        self._forward_rates: Optional[np.ndarray] = None
        self._backward_rates: Optional[np.ndarray] = None

    def _zero_drive(self, t: float) -> np.ndarray:
        """Default zero external drive."""
        return np.zeros(self.n_reactions)

    def compute_log_deviation(self, Q: np.ndarray) -> np.ndarray:
        """Compute log deviation x = ln(Q/Keq).

        Args:
            Q: Reaction quotients

        Returns:
            Log deviations from equilibrium
        """
        if len(Q) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} reaction quotients, " f"got {len(Q)}")

        # Avoid log(0)
        eps = 1e-12
        safe_Q = np.maximum(Q, eps)
        safe_Keq = np.maximum(self.Keq, eps)

        return np.log(safe_Q) - np.log(safe_Keq)

    def compute_reaction_quotients(self, x: np.ndarray) -> np.ndarray:
        """Compute reaction quotients from log deviations.

        Args:
            x: Log deviations ln(Q/Keq)

        Returns:
            Reaction quotients Q = Keq * exp(x)
        """
        return self.Keq * np.exp(x)

    def dynamics(self, t: float, x: np.ndarray) -> np.ndarray:
        """Log-linear dynamics dx/dt = -K*x + u(t).

        Args:
            t: Time
            x: Log deviations ln(Q/Keq)

        Returns:
            Time derivatives dx/dt
        """
        if len(x) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} state variables, " f"got {len(x)}")

        u = self.external_drive(t)
        return -self.K @ x + u

    def analytical_solution(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Analytical solution for constant external drive.

        This works when u(t) = u0 (constant) and K is diagonalizable.
        Solution: x(t) = exp(-K*t) * (x0 - K^-1*u0) + K^-1*u0

        Args:
            x0: Initial log deviations
            t: Time points

        Returns:
            Log deviations x(t) at each time point
        """
        if len(x0) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} initial conditions, " f"got {len(x0)}")

        # Check if external drive is constant
        u0 = self.external_drive(0.0)
        u_end = self.external_drive(t[-1] if len(t) > 1 else 1.0)

        if not np.allclose(u0, u_end, rtol=1e-6):
            warnings.warn("External drive appears time-varying, analytical solution " "may not be accurate")

        try:
            # Compute matrix exponential and steady state
            eigenvals, eigenvecs = np.linalg.eig(self.K)

            # Check if K is diagonalizable
            if np.linalg.cond(eigenvecs) > 1e12:
                warnings.warn("Relaxation matrix K is poorly conditioned, " "analytical solution may be inaccurate")

            # Steady state (if K is invertible)
            if np.abs(np.linalg.det(self.K)) > 1e-12:
                x_ss = np.linalg.solve(self.K, u0)
            else:
                x_ss = np.zeros_like(x0)
                warnings.warn("Relaxation matrix K is singular, assuming zero steady state")

            # Time evolution
            x_t = np.zeros((len(t), self.n_reactions))

            for i, time in enumerate(t):
                # Matrix exponential using eigendecomposition
                exp_Kt = eigenvecs @ np.diag(np.exp(-eigenvals * time)) @ np.linalg.inv(eigenvecs)
                x_t[i] = exp_Kt @ (x0 - x_ss) + x_ss

            return x_t

        except np.linalg.LinAlgError as e:
            warnings.warn(f"Analytical solution failed: {e}. " "Use numerical integration instead.")
            return self._fallback_solution(x0, t)

    def _fallback_solution(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Fallback numerical solution using simple Euler integration."""
        x_t = np.zeros((len(t), self.n_reactions))
        x_t[0] = x0

        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            dx = self.dynamics(t[i - 1], x_t[i - 1])
            x_t[i] = x_t[i - 1] + dt * dx

        return x_t

    def single_reaction_solution(
        self,
        Q0: float,
        Keq: float,
        k: float,
        u_func: Optional[Callable[[float], float]] = None,
        t: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical solution for single reaction with external drive.

        For single reaction: d/dt ln Q = -k ln(Q/Keq) + u(t)

        Args:
            Q0: Initial reaction quotient
            Keq: Equilibrium constant
            k: Relaxation rate
            u_func: External drive function u(t)
            t: Time points (if None, uses default)

        Returns:
            Tuple of (time_points, reaction_quotients)
        """
        if t is None:
            t = np.linspace(0, 10, 1000)

        if u_func is None:
            # No external drive: Q(t) = Keq * (Q0/Keq)^exp(-kt)
            Q_t = Keq * (Q0 / Keq) ** np.exp(-k * t)
        else:
            # With external drive - use numerical integration
            from scipy.integrate import solve_ivp

            def rhs(time, ln_Q):
                return -k * (ln_Q - np.log(Keq)) + u_func(time)

            sol = solve_ivp(rhs, [t[0], t[-1]], [np.log(Q0)], t_eval=t, method="RK45", rtol=1e-8)

            if not sol.success:
                warnings.warn(f"Numerical integration failed: {sol.message}")

            Q_t = np.exp(sol.y[0])

        return t, Q_t

    def compute_eigenanalysis(self) -> Dict[str, np.ndarray]:
        """Compute eigenanalysis of relaxation matrix K.

        Returns:
            Dictionary containing eigenvalues, eigenvectors, and timescales
        """
        eigenvals, eigenvecs = np.linalg.eig(self.K)

        # Sort by real part (fastest to slowest decay)
        idx = np.argsort(-eigenvals.real)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Compute timescales (1/Re(λ) for stable modes)
        timescales = np.zeros_like(eigenvals, dtype=complex)
        stable_mask = eigenvals.real > 1e-12
        timescales[stable_mask] = 1.0 / eigenvals[stable_mask]

        return {
            "eigenvalues": eigenvals,
            "eigenvectors": eigenvecs,
            "timescales": timescales,
            "is_stable": np.all(eigenvals.real >= -1e-12),  # Allow numerical errors
            "has_oscillations": np.any(np.abs(eigenvals.imag) > 1e-12),
        }

    def set_mass_action_parameters(self, forward_rates: np.ndarray, backward_rates: Optional[np.ndarray] = None):
        """Set parameters from mass action kinetics (legacy method).

        Note: This is a simplified approach for independent reactions.
        For full mass action networks, use LLRQDynamics.from_mass_action() instead.

        For single reaction A ⇌ B with rates kf, kr:
        - Keq = kf/kr
        - k = kr(1 + Keq) (from paper)

        Args:
            forward_rates: Forward rate constants
            backward_rates: Backward rate constants (computed from Keq if None)
        """
        if len(forward_rates) != self.n_reactions:
            raise ValueError(f"Expected {self.n_reactions} forward rates, " f"got {len(forward_rates)}")

        warnings.warn(
            "set_mass_action_parameters() is a legacy method for independent reactions. "
            "For coupled mass action networks, use LLRQDynamics.from_mass_action() instead.",
            DeprecationWarning,
        )

        kf = np.array(forward_rates)

        if backward_rates is None:
            kr = kf / self.Keq
        else:
            kr = np.array(backward_rates)
            if len(kr) != self.n_reactions:
                raise ValueError(f"Expected {self.n_reactions} backward rates, " f"got {len(kr)}")
            # Update Keq to be consistent
            self.Keq = kf / kr

        # Set relaxation rates: k = kr(1 + Keq)
        k_diag = kr * (1 + self.Keq)
        self.K = np.diag(k_diag)

    def validate_parameters(self) -> Dict[str, bool]:
        """Validate parameter consistency and stability.

        Returns:
            Dictionary of validation results
        """
        results = {}

        # Check positive equilibrium constants
        results["positive_Keq"] = np.all(self.Keq > 0)

        # Check positive relaxation rates (diagonal of K)
        k_diag = np.diag(self.K)
        results["positive_relaxation"] = np.all(k_diag > 0)

        # Check stability (all eigenvalues have non-negative real parts)
        eigenvals = np.linalg.eigvals(self.K)
        results["stable"] = np.all(eigenvals.real >= -1e-12)

        # Check symmetry of K (for detailed balance)
        results["symmetric_K"] = np.allclose(self.K, self.K.T, rtol=1e-10)

        return results

    @classmethod
    def from_mass_action(
        cls,
        network: ReactionNetwork,
        forward_rates: np.ndarray,
        backward_rates: np.ndarray,
        initial_concentrations: Optional[np.ndarray] = None,
        mode: str = "equilibrium",
        external_drive: Optional[Callable[[float], np.ndarray]] = None,
        reduce_basis: bool = True,
        enforce_symmetry: bool = False,
    ) -> "LLRQDynamics":
        """Create LLRQDynamics from mass action kinetics.

        This factory method implements the complete pipeline from mass action
        kinetics to log-linear reaction quotient dynamics using the algorithm
        from Diamond (2025).

        **Pipeline**:
        1. Computes dynamics matrix K from mass action parameters
        2. Derives equilibrium constants Keq = k⁺/k⁻
        3. Creates LLRQDynamics instance with computed parameters
        4. Stores mass action metadata for later retrieval

        The resulting dynamics object satisfies:
            d/dt ln Q = -K ln(Q/Keq) + u(t)

        where the matrix K captures the network's relaxation behavior and
        coupling between reactions.

        **Typical workflow**:
        ```python
        # 1. Create network from stoichiometry
        network = ReactionNetwork(species, reactions, S_matrix)

        # 2. Define mass action parameters
        c_star = [...]  # Steady-state concentrations
        k_plus = [...]  # Forward rate constants
        k_minus = [...] # Backward rate constants

        # 3. Create dynamics automatically
        dynamics = LLRQDynamics.from_mass_action(
            network, c_star, k_plus, k_minus, mode='equilibrium'
        )

        # 4. Use for simulation, analysis, control design
        eigeninfo = dynamics.compute_eigenanalysis()
        solution = dynamics.analytical_solution(x0, t)
        ```

        Args:
            network: ReactionNetwork defining stoichiometry and species
            forward_rates: Forward rate constants k⁺ [reactions]
            backward_rates: Backward rate constants k⁻ [reactions]
            initial_concentrations: Species concentrations [species]
                                  If at equilibrium, used directly; otherwise equilibrium is computed
                                  If None, uses get_initial_concentrations() from network
            mode: Algorithm mode - 'equilibrium' (near thermodynamic equilibrium)
                  or 'nonequilibrium' (general steady state)
            external_drive: External drive function u(t) → array[reactions]
            reduce_basis: Whether to reduce to Im(S^T) basis (recommended)
            enforce_symmetry: Whether to enforce detailed balance symmetry

        Returns:
            LLRQDynamics instance with:
            - K: Computed dynamics matrix (possibly reduced)
            - Keq: Equilibrium constants from k⁺/k⁻
            - Mass action metadata accessible via get_mass_action_info()

        Examples:
            Enzymatic reaction E + S ⇌ ES → E + P:
            >>> network = ReactionNetwork(
            ...     ['E', 'S', 'ES', 'P'],
            ...     ['binding', 'unbinding', 'catalysis'],
            ...     [[-1, 1, 1], [-1, 1, 0], [1, -1, -1], [0, 0, 1]]
            ... )
            >>> dynamics = LLRQDynamics.from_mass_action(
            ...     network,
            ...     forward_rates=[2.0, 1.0, 5.0],
            ...     backward_rates=[1.0, 2.0, 0.0],
            ...     initial_concentrations=[1.0, 2.0, 0.1, 0.5],
            ...     mode='equilibrium'
            ... )
            >>> print(f"Relaxation timescales: {1/dynamics.compute_eigenanalysis()['eigenvalues'].real}")

        See Also:
            ReactionNetwork.compute_dynamics_matrix: Lower-level matrix computation
            get_mass_action_info: Retrieve stored mass action parameters

        References:
            Diamond, S. (2025). "Log-Linear Reaction Quotient Dynamics"
        """
        # Compute dynamics matrix from mass action
        dynamics_data = network.compute_dynamics_matrix(
            forward_rates=forward_rates,
            backward_rates=backward_rates,
            initial_concentrations=initial_concentrations,
            mode=mode,
            reduce_to_image=reduce_basis,
            enforce_symmetry=enforce_symmetry,
        )

        # Use the full matrix to match number of reactions
        K_matrix = dynamics_data["K"]

        # Compute equilibrium constants from mass action
        k_plus = np.array(forward_rates)
        k_minus = np.array(backward_rates)
        Keq = k_plus / k_minus

        # Create dynamics instance
        dynamics = cls(network=network, equilibrium_constants=Keq, relaxation_matrix=K_matrix, external_drive=external_drive)

        # Store additional mass action data
        dynamics._mass_action_data = dynamics_data
        dynamics._mass_action_mode = mode
        dynamics._equilibrium_point = dynamics_data.get("equilibrium_point", None)
        dynamics._forward_rates = k_plus
        dynamics._backward_rates = k_minus

        return dynamics

    def get_mass_action_info(self) -> Optional[Dict[str, Any]]:
        """Get mass action computation details if available.

        Returns:
            Dictionary with mass action parameters and matrices,
            or None if not created from mass action
        """
        if not hasattr(self, "_mass_action_data") or self._mass_action_data is None:
            return None

        return {
            "mode": self._mass_action_mode,
            "equilibrium_point": self._equilibrium_point,
            "forward_rates": self._forward_rates,
            "backward_rates": self._backward_rates,
            "dynamics_data": self._mass_action_data,
        }

    def compute_steady_state_concentrations(self, conserved_quantities: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute steady state concentrations from dynamics.

        This uses the equilibrium point from mass action if available,
        otherwise attempts to solve from the dynamics.

        Args:
            conserved_quantities: Values of conserved quantities

        Returns:
            Steady state concentrations
        """
        if hasattr(self, "_equilibrium_point") and self._equilibrium_point is not None:
            return self._equilibrium_point.copy()

        # Fallback: solve from conservation laws and equilibrium conditions
        if conserved_quantities is not None:
            # This is a placeholder - full implementation would solve
            # the nonlinear system of conservation laws and equilibrium conditions
            warnings.warn("Steady state computation from conserved quantities " "not fully implemented. Returning zeros.")

        return np.zeros(self.network.n_species)
