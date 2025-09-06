# src/llrq/examples/cycle_closed_loop.py
import os, io, base64, json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from llrq.reaction_network import ReactionNetwork
from llrq.llrq_dynamics import LLRQDynamics
from llrq.solver import LLRQSolver

# Estimation & control (from earlier additions)
from llrq.estimation.kalman import ReducedKalmanFilterCT, reduce_y_from_Q
from llrq.control.lqr import LQRController


@dataclass
class DemoConfig:
    T: float = 60.0
    npoints: int = 2500
    y_ref: np.ndarray = None   # set after we know rankS
    uhat_bounds: tuple = None  # (umin, umax) in R^m
    seed: int = 7
    # disturbances (reduced for better tracking demo)
    sinus_amp: float = 0.05
    step_time: float = 30.0
    step_vec: np.ndarray = None  # set after we know rankS
    # noise
    meas_std: float = 0.01
    # LQR weights (better tuned for tracking)
    Qw_scale: float = 10.0  # Higher penalty on tracking error
    Rw_scale: float = 0.1   # Lower penalty on control effort
    Ki_weight: float = 1.0  # Higher integral weight


def _build_3cycle_network() -> ReactionNetwork:
    """
    Minimal 3-reaction ring:
        R1: A <-> B
        R2: B <-> C
        R3: C <-> A
    """
    # Species and reactions
    species_ids = ["A", "B", "C"]
    reaction_ids = ["R1", "R2", "R3"]
    
    # Stoichiometric matrix (3 species x 3 reactions)
    # R1: A(-1) -> B(+1), R2: B(-1) -> C(+1), R3: C(-1) -> A(+1)
    S = np.array([[-1, 0, 1],   # A: -1 in R1, 0 in R2, +1 in R3
                  [1, -1, 0],   # B: +1 in R1, -1 in R2, 0 in R3
                  [0, 1, -1]])  # C: 0 in R1, +1 in R2, -1 in R3
    
    # Species info with initial concentrations
    species_info = {
        "A": {"name": "A", "initial_concentration": 2.0, "compartment": "cell", "boundary_condition": False},
        "B": {"name": "B", "initial_concentration": 0.2, "compartment": "cell", "boundary_condition": False},
        "C": {"name": "C", "initial_concentration": 0.1, "compartment": "cell", "boundary_condition": False}
    }
    
    # Reaction info
    reaction_info = [
        {"id": "R1", "name": "A ⇌ B", "reactants": [("A", 1.0)], "products": [("B", 1.0)], "reversible": True},
        {"id": "R2", "name": "B ⇌ C", "reactants": [("B", 1.0)], "products": [("C", 1.0)], "reversible": True},
        {"id": "R3", "name": "C ⇌ A", "reactants": [("C", 1.0)], "products": [("A", 1.0)], "reversible": True}
    ]
    
    return ReactionNetwork(species_ids, reaction_ids, S, species_info, reaction_info)


def _fit_K_red(solver: LLRQSolver) -> np.ndarray:
    """
    LLRQ-Fit at the chosen operating point.
    Default: analytic projection K_red = B^T K B (small, SPD near equilibrium).
    You can swap this with a data-driven linearization off your mass-action simulator later.
    """
    B = solver._B                 # (r x rS), basis for Im(S^T)
    K = solver.dynamics.K         # (r x r)
    return B.T @ K @ B            # (rS x rS)


def _setup_analytical_control(solver: LLRQSolver, cfg: DemoConfig, actuated=("R1","R3")):
    """Set up analytical steady-state control instead of LQR tracking."""
    B = solver._B
    rankS = solver._rankS

    # Reduced matrices
    K_red = _fit_K_red(solver)
    A = -K_red

    # Actuation map: choose a couple reactions to actuate
    r = solver.dynamics.n_reactions
    idx = []
    for rid in actuated:
        idx.append(solver.network.reaction_to_idx[rid])
    m = len(idx)
    G = np.zeros((r, m))
    for j, k in enumerate(idx):
        G[k, j] = 1.0
    Bred = B.T @ G

    # Set target reduced state
    if cfg.y_ref is None:
        cfg.y_ref = np.array([0.1, -0.05])[:rankS]
    
    # Analytical steady-state control: u_ss = -A @ y_target
    # At steady state: 0 = A @ y + B @ u, so u = -A @ y / B
    u_ss = np.linalg.pinv(Bred) @ (-A @ cfg.y_ref)
    
    # Control bounds for saturation
    if cfg.uhat_bounds is None:
        cfg.uhat_bounds = (np.full(m, -5.0), np.full(m, 5.0))

    return A, Bred, G, u_ss


def _bode_data(A: np.ndarray, Bred: np.ndarray) -> dict:
    """Simple MIMO Bode-like curves: singular values of (iωI - A)^{-1} B over ω."""
    ws = np.logspace(-2, 2, 200)
    sigmax = []
    for w in ws:
        M = 1j*w*np.eye(A.shape[0]) - A
        H = np.linalg.solve(M, Bred)
        svals = np.linalg.svd(H, compute_uv=False)
        sigmax.append(svals[0])
    return {"w": ws, "sigmax": np.array(sigmax)}


def _png(fig) -> str:
    """Return a data: URI PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_and_run(out_dir: str = "llrq_report_cycle"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Build network & LLRQ objects
    network = _build_3cycle_network()
    Keq = np.array([2.0, 0.5, 0.9])
    
    # Let LLRQDynamics set up the relaxation matrix automatically
    # The solver will analyze the network and _fit_K_red will extract the effective dynamics
    dynamics = LLRQDynamics(network=network, equilibrium_constants=Keq)
    solver = LLRQSolver(dynamics)

    # --- Analytical control setup  
    cfg = DemoConfig()
    A, Bred, G, u_steady = _setup_analytical_control(solver, cfg)

    # --- Time grid and ICs
    T, n = cfg.T, cfg.npoints
    t_eval = np.linspace(0.0, T, n)
    dt = t_eval[1] - t_eval[0]
    B = solver._B
    rankS = solver._rankS
    rng = np.random.default_rng(cfg.seed)

    # Initial concentrations -> reduced y0
    c0 = solver._parse_initial_dict({sid: info.get("initial_concentration", 0.0)
                                     for sid, info in network.species_info.items()})
    Q0 = network.compute_reaction_quotients(c0)
    x0 = np.log(Q0) - solver._lnKeq_consistent
    x0 = solver._P @ x0
    y = (B.T @ x0)

    # Disturbances in reduced space (Keq step + sinusoid) - increased magnitude
    step = np.array([0.3, -0.2])[:rankS]  # Larger step disturbance to make it visible
    if cfg.step_vec is None:
        cfg.step_vec = step

    def d_red(t):
        sinus = cfg.sinus_amp * np.array([np.sin(0.8*t), 0.3*np.sin(1.1*t)])[:rankS]
        # Impulse disturbance: only applied for one time step at cfg.step_time
        impulse = np.zeros(rankS)
        return sinus + impulse

    # Storage
    Y = np.zeros((n, rankS))
    Yhat = np.zeros_like(Y)
    Ufull = np.zeros((n, dynamics.n_reactions))
    Qtraj = np.zeros((n, dynamics.n_reactions))
    C = network.find_conservation_laws()
    Cons = None
    if C.size:
        Cons = np.zeros((n, C.shape[0]))

    # --- Simulate with analytical steady-state control + simple feedback
    feedback_gain = 1.0  # Simple proportional feedback for disturbances
    
    for i, t in enumerate(t_eval):
        # Current state from previous iteration
        if i == 0:
            Q_true = Q0.copy()
            x_true = np.log(Q_true) - solver._lnKeq_consistent
            x_true = solver._P @ x_true
            y = B.T @ x_true
        else:
            x_true = (B @ y).reshape(-1)
            Q_true = solver._Keq_consistent * np.exp(x_true)

        # Control: steady-state value + simple feedback + disturbance compensation
        disturbance = d_red(t)
        error = y - cfg.y_ref
        u_red = Bred @ u_steady - feedback_gain * error - disturbance
        
        # Apply control bounds
        u_full = G @ np.clip(u_steady - feedback_gain * np.linalg.pinv(Bred) @ error, 
                            cfg.uhat_bounds[0], cfg.uhat_bounds[1])
        u_red = B.T @ u_full
        
        # System dynamics: dy/dt = A*y + B*u + d
        ydot = A @ y + u_red + disturbance

        # Integrate (explicit Euler)
        y = y + dt * ydot
        
        # Apply impulse disturbance at specific time (kick to the state)
        if abs(t - cfg.step_time) < dt/2:  # Apply once when t ≈ step_time
            y = y + cfg.step_vec  # Direct kick to the state

        # Log results
        Y[i] = y
        Yhat[i] = y  # No estimation needed, we have the true state
        Ufull[i] = u_full  
        Qtraj[i] = Q_true
        if Cons is not None:
            Cons[i] = (C @ solver._compute_concentrations_from_reduced(Q_true[None, :], c0, True)[0])

    # Map back x/Q/c for full plots
    X = (B @ Y.T).T
    Q = solver._Keq_consistent * np.exp(X)
    Ctraj = solver._compute_concentrations_from_reduced(Q, c0, enforce_conservation=True)

    # --- Bode-style frequency response
    bode = _bode_data(A, Bred)

    # --- Figures
    figs = {}

    # y, y_ref, yhat with better annotations
    fig = plt.figure(figsize=(8.2, 3.8))
    for k in range(rankS):
        plt.plot(t_eval, Y[:, k], label=f"y{k+1} (actual)", linewidth=2)
    for k in range(rankS):
        plt.plot(t_eval, np.full_like(t_eval, cfg.y_ref[k]), "--", alpha=0.8, label=f"y*{k+1} (reference)", linewidth=2)
    for k in range(rankS):
        plt.plot(t_eval, Yhat[:, k], ":", alpha=0.8, label=f"ŷ{k+1} (estimate)", linewidth=1.5)
    
    # Mark step disturbance
    plt.axvline(cfg.step_time, color='red', linestyle=':', alpha=0.7, label='Impulse disturbance')
    
    plt.xlabel("Time (s)"); plt.ylabel("Reduced state y"); plt.legend(ncol=2); 
    plt.title("Reduced State: Analytical Convergence to Target\n(y = projected log-deviation from reaction equilibria)")
    plt.grid(True, alpha=0.3)
    figs["y.png"] = _png(fig)

    # Tracking error
    fig = plt.figure(figsize=(8.2, 3.2))
    tracking_error = np.linalg.norm(Y - cfg.y_ref, axis=1)
    plt.plot(t_eval, tracking_error, 'b-', linewidth=2, label='||y - y*||₂')
    plt.axvline(cfg.step_time, color='red', linestyle=':', alpha=0.7, label='Impulse disturbance')
    plt.xlabel("Time (s)"); plt.ylabel("Tracking error"); plt.legend()
    plt.title("Control Performance: Distance from Reference")
    plt.grid(True, alpha=0.3)
    figs["error.png"] = _png(fig)

    # u (per reaction) - improved
    fig = plt.figure(figsize=(8.2, 3.2))
    reaction_names = ["R1: A⇌B", "R2: B⇌C", "R3: C⇌A"]
    for j in range(dynamics.n_reactions):
        plt.plot(t_eval, Ufull[:, j], label=reaction_names[j], linewidth=2)
    plt.axvline(cfg.step_time, color='red', linestyle=':', alpha=0.7, label='Impulse disturbance')
    plt.axhline(0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel("Time (s)"); plt.ylabel("Control effort u"); 
    plt.title("Control Signals: Steady-State Drives + Feedback")
    plt.legend(); plt.grid(True, alpha=0.3)
    figs["u.png"] = _png(fig)

    # Q/Keq ratios (more intuitive than raw Q)
    fig = plt.figure(figsize=(8.2, 3.2))
    Q_ratios = Q / solver._Keq_consistent
    for j in range(dynamics.n_reactions):
        plt.plot(t_eval, Q_ratios[:, j], label=f"{reaction_names[j]}", linewidth=2)
    plt.axhline(1, color='black', linestyle='--', alpha=0.7, label='Equilibrium (Q/Keq = 1)')
    plt.axvline(cfg.step_time, color='red', linestyle=':', alpha=0.7, label='Impulse disturbance')
    plt.xlabel("Time (s)"); plt.ylabel("Q/Keq ratio"); 
    plt.title("Reaction Quotient Ratios: Deviation from Equilibrium\n(>1 = forward favored, <1 = reverse favored)")
    plt.legend(); plt.grid(True, alpha=0.3)
    figs["Q.png"] = _png(fig)

    # conservation totals
    if Cons is not None and Cons.shape[1] > 0:
        fig = plt.figure(figsize=(8.2, 3.2))
        for j in range(Cons.shape[1]):
            plt.plot(t_eval, Cons[:, j], label=f"C{j+1}")
        plt.xlabel("time"); plt.ylabel("conserved totals"); plt.title("Conservation (constant up to tolerance)")
        figs["cons.png"] = _png(fig)

    # Species concentration trajectories (if available)
    if Ctraj is not None:
        fig = plt.figure(figsize=(8.2, 3.2))
        colors = ['blue', 'green', 'orange']
        for k, sid in enumerate(network.species_ids):
            plt.plot(t_eval, Ctraj[:, k], label=f'[{sid}]', color=colors[k], linewidth=2)
        
        # Compute and show reference concentrations corresponding to y_ref
        # This requires solving what concentrations would give y_ref
        try:
            # Approximate reference concentrations by computing steady state at y_ref
            y_ss = cfg.y_ref
            x_ss = B @ y_ss
            Q_ss = solver._Keq_consistent * np.exp(x_ss)
            c_ref = solver._compute_concentrations_from_reduced(Q_ss[None, :], c0, True)[0]
            
            for k, sid in enumerate(network.species_ids):
                plt.axhline(c_ref[k], color=colors[k], linestyle='--', alpha=0.7, 
                           label=f'[{sid}]* (target)')
        except:
            pass  # Skip reference concentrations if computation fails
            
        plt.axvline(cfg.step_time, color='red', linestyle=':', alpha=0.7, label='Impulse disturbance')
        plt.xlabel("Time (s)"); plt.ylabel("Concentration"); 
        plt.title("Species Concentrations: The Physical Variables We Care About")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        figs["c.png"] = _png(fig)

    # Bode-like (largest singular value of (iωI - A)^{-1} B)
    fig = plt.figure(figsize=(6.6, 3.2))
    plt.semilogx(bode["w"], 20*np.log10(np.maximum(1e-12, bode["sigmax"])))
    plt.xlabel("ω (rad/s)"); plt.ylabel("||H(iω)||₂ (dB)"); plt.title("Frequency response of A=-K_red")
    figs["bode.png"] = _png(fig)

    # --- Simple summary stats
    stats = {
        "tracking_rms": float(np.sqrt(np.mean((Y - cfg.y_ref)**2))),
        "tracking_final_err": float(np.linalg.norm(Y[-1] - cfg.y_ref)),
        "saturation_fraction": float(np.mean(
            (np.isclose(np.abs(Ufull[:, solver.network.reaction_to_idx['R1']]), cfg.uhat_bounds[1][0], atol=1e-6)) |
            (np.isclose(np.abs(Ufull[:, solver.network.reaction_to_idx['R3']]), cfg.uhat_bounds[1][1], atol=1e-6))
        )),
        "meas_std": cfg.meas_std,
        "step_time": cfg.step_time,
        "step_vec": cfg.step_vec.tolist(),
    }

    # --- HTML report
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LLRQ Flagship Demo – Cycle Closed Loop</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0.3em 0; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 18px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    img {{ max-width: 100%; height: auto; }}
    code {{ background:#f6f8fa; padding:2px 4px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>LLRQ Analytical Control Demo</h1>
  <p><b>Model:</b> 3-reaction ring (A⇌B⇌C⇌A). <b>Controller:</b> Analytical steady-state control with proportional feedback.<br/>
     <b>Disturbances:</b> sinusoid + impulse disturbance at t={cfg.step_time:g}s. <b>Runtime horizon:</b> {cfg.T:g}s.</p>

  <h2>What This Demo Shows</h2>
  <div class="card">
    <p><b>Control Objective:</b> Maintain a desired non-equilibrium steady state using the analytical LLRQ solution.</p>
    <ul>
      <li><b>Analytical insight:</b> To maintain any non-equilibrium state, you just need a constant drive u = -A⁻¹y*</li>
      <li><b>Target state y*:</b> {cfg.y_ref} (deviations from reaction equilibria)</li>
      <li><b>Steady-state control:</b> Constant drives to reactions R1 (A⇌B) and R3 (C⇌A)</li>
      <li><b>Disturbance feedback:</b> Simple proportional control to reject perturbations</li>
    </ul>
    <p><b>Key insight:</b> LLRQ dynamics have analytical solutions - no complex control theory needed!</p>
  </div>

  <h2>Summary</h2>
  <pre>{json.dumps(stats, indent=2)}</pre>

  <h2>1. Control Performance: Is it tracking?</h2>
  <div class="card"><img src="{figs['error.png']}" /></div>

  <h2>2. Physical Results: Concentrations</h2>
  {"<div class='card'><img src='"+figs['c.png']+"'/></div>" if 'c.png' in figs else "<p>Concentration plot not available</p>"}

  <h2>3. Control Signals: What drives the system?</h2>
  <div class="card"><img src="{figs['u.png']}" /></div>

  <h2>4. Reaction Quotients: How far from equilibrium?</h2>
  <div class="card"><img src="{figs['Q.png']}" /></div>

  <h2>5. Reduced State: The control coordinate system</h2>
  <div class="card"><img src="{figs['y.png']}" /></div>

  {"<h2>6. Conservation Laws</h2><div class='card'><img src='"+figs['cons.png']+"'/></div>" if 'cons.png' in figs else ""}

  <h2>7. System Analysis: Frequency Response</h2>
  <div class="card"><img src="{figs['bode.png']}" /></div>

  <p style="color:#888;margin-top:24px">Generated by llrq.examples.run('cycle_closed_loop').</p>
</body>
</html>
"""
    out_html = os.path.join(out_dir, "report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    return out_html


if __name__ == "__main__":
    import sys
    import os
    
    # Parse command line arguments
    output_dir = "llrq_report_cycle"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"Running LLRQ Cycle Closed Loop Example")
    print(f"Output directory: {output_dir}")
    
    # Run the demo
    report_path = build_and_run(output_dir)
    
    print(f"\nDemo completed successfully!")
    print(f"Report generated at: {report_path}")
    print(f"\nTo view the report, open: {os.path.abspath(report_path)}")

