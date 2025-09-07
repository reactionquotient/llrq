#!/usr/bin/env python3
"""
Quick LQR control example snippet for LLRQ dynamics.

This shows the essential code for setting up LQR control.
For a complete working example, see lqr_complete_example.py
"""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llrq.control.lqr import LQRController
from src.llrq.llrq_dynamics import LLRQDynamics
from src.llrq.reaction_network import ReactionNetwork
from src.llrq.solver import LLRQSolver

# Assume you have already created:
# - network: ReactionNetwork object
# - dynamics: LLRQDynamics object with network, Keq, and K matrix
# - init_conc_dict: initial concentrations dictionary

# Example setup (you would replace with your actual network):
species_ids = ["A", "B", "C"]
reaction_ids = ["R1", "R2", "R3"]
S = np.array([[-1, 1, 0], [1, -1, -1], [0, 0, 1]])  # Example stoichiometry

network = ReactionNetwork(species_ids, reaction_ids, S)
dynamics = LLRQDynamics(network, equilibrium_constants=np.array([2.0, 1.5, 3.0]), relaxation_matrix=np.eye(3))

# Create solver
solver = LLRQSolver(dynamics)

# Initial conditions
init_conc_dict = {"A": 1.0, "B": 0.1, "C": 0.1}

# Choose 2 actuated reactions by name or index
ctrl = LQRController(
    solver, controlled_reactions=["R1", "R3"], Q=1.0 * np.eye(solver._rankS), R=0.1 * np.eye(2), integral=True, Ki_weight=0.1
)

# Track equilibrium in reduced coordinates
y_ref = np.zeros(solver._rankS)

# Run closed-loop simulation
res = solver.simulate_closed_loop(
    initial_conditions=init_conc_dict, t_span=(0.0, 50.0), controller=ctrl, y_ref=y_ref, n_points=1500
)

print(f"Simulation complete: {res['success']}")
print(f"Final time: {res['time'][-1]:.2f}")
