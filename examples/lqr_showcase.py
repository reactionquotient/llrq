"""
A showy LQR demo on the reduced LLRQ model:
- integral setpoint tracking (nonzero y*)
- disturbance rejection (sinusoid + step)
- input saturation with basic anti-windup
- steady-state Kalman filter on noisy y-measurements
- maps results back to Q and c, checks conservation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from llrq.solver import LLRQSolver          # your reduced, cycle-safe solver
from llrq.llrq_dynamics import LLRQDynamics # whatever you use to build dynamics
from llrq.reaction_network import ReactionNetwork
from llrq.control.lqr import LQRController  # internal LQR controller

# ---------- Anti-windup wrapper for internal LQR controller ----------
class AntiWindupLQRController:
    """Wrapper for internal LQRController with anti-windup logic."""
    def __init__(self, internal_controller, uhat_bounds=None):
        self.controller = internal_controller
        self.uhat_bounds = uhat_bounds
        
    def reset(self):
        self.controller.reset()
        
    def step(self, t, y, y_ref):
        """Step method compatible with showcase simulation loop."""
        u_full_desired = self.controller.u_full(t, y, y_ref)
        
        if self.uhat_bounds is not None:
            # Extract uhat from u_full by inverting G @ uhat = u_full
            # Since G has 1's in selected positions, we can extract directly
            uhat_desired = np.zeros(self.controller.m)
            for j in range(self.controller.m):
                # Find which reaction this control input affects
                reaction_idx = np.where(self.controller.G[:, j] == 1.0)[0][0]
                uhat_desired[j] = u_full_desired[reaction_idx]
            
            # Apply bounds
            umin, umax = self.uhat_bounds
            uhat_clipped = np.minimum(np.maximum(uhat_desired, umin), umax)
            
            # Simple anti-windup: if saturated, reduce integrator
            if self.controller.integral and np.any(uhat_clipped != uhat_desired):
                # Reset integrator when saturated (simple approach)
                correction_factor = 0.9
                self.controller._eta *= correction_factor
            
            # Reconstruct u_full from clipped uhat
            u_full = self.controller.G @ uhat_clipped
        else:
            u_full = u_full_desired
            
        return u_full

# ---------- Tiny steady-state continuous-time Kalman filter on y ----------
class KalmanFilterReduced:
    """Steady-state CT Kalman filter with z = C y + noise, here C=I."""
    def __init__(self, A, W=1e-6, V=1e-3):
        self.A = A
        self.C = np.eye(A.shape[0])
        self.W = W if np.ndim(W) == 2 else (W * np.eye(A.shape[0]))
        self.V = V if np.ndim(V) == 2 else (V * np.eye(A.shape[0]))
        # Solve dual ARE: A P + P A^T - P C^T V^{-1} C P + W = 0
        P = solve_continuous_are(A.T, self.C.T, self.W, self.V)
        self.L = P @ self.C.T @ np.linalg.inv(self.V)  # Kalman gain
        self.yhat = np.zeros(A.shape[0])

    def reset(self, y0=None):
        self.yhat[:] = 0.0 if y0 is None else y0

    def step(self, y_meas, u_red, dt):
        # predictor-corrector for dot yhat = A yhat + B u + L (z - C yhat)
        # B u is handled by caller (easier for composition); we only do correction here
        self.yhat = self.yhat + dt * (self.A @ self.yhat + self.L @ (y_meas - self.yhat))
        return self.yhat

# ------------------------ Build a spicy scenario --------------------------
def build_demo_network():
    """
    Build or load any network you like. A 3-reaction cycle is great for demos:
    A <-> B, B <-> C, C <-> A  (rank(S)=2). Use your normal constructors.
    Replace this stub with your real builder.
    """
    # Define species and reactions
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2", "R3"]
    
    # Define stoichiometric matrix for cycle: A <-> B, B <-> C, C <-> A
    # Each column is a reaction: [A, B, C] rows
    S = np.array([
        [-1,  0,  1],  # A: consumed in R1, produced in R3
        [ 1, -1,  0],  # B: produced in R1, consumed in R2
        [ 0,  1, -1]   # C: produced in R2, consumed in R3
    ])
    
    # Species information with initial concentrations
    species_info = {
        "A": {"initial_concentration": 2.0},
        "B": {"initial_concentration": 0.2},
        "C": {"initial_concentration": 0.1}
    }
    
    # Create reaction network
    rn = ReactionNetwork(species_ids, reaction_ids, S, species_info=species_info)

    # Pick consistent Keq for each reaction (logKeq in Im(S^T) automatically in solver)
    equilibrium_constants = np.array([2.0, 0.5, 0.9])  # arbitrary but consistent up to projection
    # Give some K (we can let LLRQDynamics compute from mass-action or set directly)
    dyn = LLRQDynamics(network=rn, equilibrium_constants=equilibrium_constants)

    return rn, dyn

def main():
    # --- network + solver ---
    network, dynamics = build_demo_network()
    solver = LLRQSolver(dynamics)

    # Use internal LQR controller
    controlled_reactions = ["R1", "R3"]  # reactions to actuate
    
    # LQR weights (reduced to avoid numerical issues)
    Qw = np.eye(solver._rankS) * 0.5       # penalize y error
    Rw = np.eye(len(controlled_reactions)) * 0.1  # penalize input effort
    
    # Create internal LQR controller
    internal_ctrl = LQRController(solver, controlled_reactions, Q=Qw, R=Rw, 
                                  integral=True, Ki_weight=0.1)
    
    # Wrap with anti-windup
    ctrl = AntiWindupLQRController(internal_ctrl, 
                                   uhat_bounds=(np.array([-1.0]*len(controlled_reactions)), 
                                               np.array([1.0]*len(controlled_reactions))))

    # Desired nonzero reduced setpoint (drive a cycle affinity)
    y_ref = np.array([0.5, -0.2])[:solver._rankS]

    # Disturbances in reduced coordinates (project external drive & Keq drift)
    def d_red(t):
        # periodic exogenous forcing
        sinus = 0.15 * np.array([np.sin(0.8*t), 0.3*np.sin(1.1*t)])[:solver._rankS]
        # step at t=20 (e.g., Keq jump / temp change)
        step = (t >= 20.0) * np.array([0.2, -0.1])[:solver._rankS]
        return sinus + step

    # Measurement model: y_meas = y + noise
    rng = np.random.default_rng(7)
    def measure_y(y_true):
        noise = 0.03 * rng.standard_normal(size=y_true.shape)  # 3% noise on y
        return y_true + noise

    # Kalman filter needs the reduced dynamics matrix A = -K_red
    B = solver._B
    K_red = B.T @ dynamics.K @ B
    A = -K_red
    kf = KalmanFilterReduced(A, W=1e-6, V=9e-4)  # std ~ 0.03^2
    kf.reset()

    # Initial conditions from concentrations
    c0_dict = {sid: info.get("initial_concentration", 0.0)
               for sid, info in network.species_info.items()}
    c0 = solver._parse_initial_dict(c0_dict)
    Q0 = network.compute_reaction_quotients(c0)
    x0 = np.log(Q0) - solver._lnKeq_consistent
    x0 = solver._P @ x0
    y0 = solver._B.T @ x0

    # Integrate reduced closed loop ourselves to allow measurement & KF
    t0, tf, n = 0.0, 60.0, 2500
    t_eval = np.linspace(t0, tf, n)
    dt = (tf - t0) / (n - 1)

    y = y0.copy()
    y_traj = np.zeros((n, solver._rankS))
    u_traj = np.zeros((n, dynamics.n_reactions))
    yref_traj = np.zeros((n, solver._rankS))
    yhat_traj = np.zeros_like(y_traj)

    ctrl.reset()

    for i, t in enumerate(t_eval):
        # Measure & estimate
        y_meas = measure_y(y)
        yhat = kf.step(y_meas, internal_ctrl.B @ np.zeros(len(controlled_reactions)), dt)   # we let controller compute u; KF uses only correction here
        # Controller uses estimate (swap in y for "full-state" case)
        u_full = ctrl.step(t, yhat, y_ref)
        # Reduced RHS: dot y = A y + B_red u_red + d_red(t)
        # Since ctrl.step returns full u_full, reduce it:
        u_red = B.T @ u_full
        y += dt * (A @ y + u_red + d_red(t))
        # Log
        y_traj[i] = y
        yhat_traj[i] = yhat
        yref_traj[i] = y_ref
        u_traj[i] = u_full

    # Map back to full x, Q, and reconstruct concentrations
    x_traj = (B @ y_traj.T).T
    Q_traj = solver._Keq_consistent * np.exp(x_traj)
    c_traj = solver._compute_concentrations_from_reduced(Q_traj, c0, enforce_conservation=True)

    # Check a conserved total (first row of C)
    C = network.find_conservation_laws()
    cons = (C @ c_traj.T).T if C.size else None

    # ----------------------------- Plots -----------------------------
    fig, axs = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    # Reduced coordinates
    axs[0].plot(t_eval, y_traj)
    axs[0].plot(t_eval, yref_traj, '--', alpha=0.7)
    axs[0].plot(t_eval, yhat_traj, ':', alpha=0.6)
    axs[0].set_ylabel("y (reduced)")
    axs[0].legend([*(f"y{i+1}" for i in range(solver._rankS)),
                   *(f"y*_i" for i in range(solver._rankS)),
                   *(f"yhat{i+1}" for i in range(solver._rankS))], ncol=3)
    axs[0].set_title("Reduced state y: tracking nonzero setpoint with disturbances + KF")

    # Control inputs per reaction (full space)
    axs[1].plot(t_eval, u_traj)
    axs[1].set_ylabel("u_full (per reaction)")
    axs[1].set_title("Control signals (saturated where flat)")

    # Reaction quotients (log-scale helpful if wide)
    axs[2].plot(t_eval, Q_traj)
    axs[2].set_ylabel("Q_j")
    axs[2].set_title("Reaction quotients Q(t)")

    # Conservation check
    if cons is not None and cons.shape[1] > 0:
        axs[3].plot(t_eval, cons)
        axs[3].set_ylabel("Conserved totals")
        axs[3].set_title("Conservation stays constant (num. tolerance)")

    axs[3].set_xlabel("time")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

