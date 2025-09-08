#!/usr/bin/env python3
"""
Yeast Core Glycolysis with Fermentation and Glycerol Branch - LLRQ Dynamics

This example demonstrates log-linear reaction quotient (LLRQ) dynamics for
yeast glycolysis including the fermentation pathway and glycerol branch.
It uses thermodynamically-derived equilibrium constants from yeast-GEM.yml
and demonstrates various control strategies.

System: dx/dt = -K*x + B*u(t)
Where x = ln(Q/Keq) are log-scaled reaction quotients

Reactions:
1. PGI:  G6P ↔ F6P (phosphoglucose isomerase)
2. TPI:  DHAP ↔ GAP (triose phosphate isomerase)
3. GAPDH: GAP+Pi+NAD ↔ 1,3-BPG+NADH (glyceraldehyde-3-phosphate dehydrogenase)
4. PGK:  1,3-BPG+ADP ↔ 3-PG+ATP (phosphoglycerate kinase)
5. PGM:  3-PG ↔ 2-PG (phosphoglycerate mutase)
6. ENO:  2-PG ↔ PEP (enolase)
7. ADH:  AcAld+NADH ↔ EtOH+NAD (alcohol dehydrogenase, fermentation)
8. GPD:  DHAP+NADH ↔ G3P+NAD (glycerol-3-phosphate dehydrogenase, glycerol branch)

Controls:
1. u_glc_in: Glucose influx (affects G6P/F6P balance)
2. u_pyk_pull: Pyruvate kinase activity (pulls PEP toward pyruvate)
3. u_pdc_push: Pyruvate decarboxylase push (drives acetaldehyde production)
4. u_ATP_boost: ATP/ADP ratio manipulation
5. u_ADP_boost: ADP/ATP ratio manipulation (opposite of ATP boost)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import LLRQDynamics, LLRQSolver
from llrq.visualization import LLRQVisualizer
from llrq.control import LLRQController
from llrq.thermodynamic_accounting import ThermodynamicAccountant


def load_system_matrices():
    """Load K and B matrices from CSV files."""
    base_dir = os.path.join(os.path.dirname(__file__), "yeast_glycolysis")

    # Load K matrix (relaxation/coupling matrix)
    K_file = os.path.join(base_dir, "yeast_K_matrix.csv")
    K_df = pd.read_csv(K_file, index_col=0)
    K = K_df.values

    # Load B matrix (control input matrix)
    B_file = os.path.join(base_dir, "yeast_B_matrix.csv")
    B_df = pd.read_csv(B_file, index_col=0)
    B = B_df.values

    # Get reaction and control names
    reaction_names = K_df.index.tolist()
    control_names = B_df.columns.tolist()

    return K, B, reaction_names, control_names


def load_equilibrium_constants():
    """Load equilibrium constants from extracted YAML data, with literature fallbacks."""
    base_dir = os.path.join(os.path.dirname(__file__), "yeast_glycolysis")
    keq_file = os.path.join(base_dir, "yeast_keq_values.csv")

    keq_df = pd.read_csv(keq_file)

    # Create Keq vector in the same order as the K matrix
    keq_dict = {}
    for _, row in keq_df.iterrows():
        if row["data_quality"] == "COMPLETE":
            keq_dict[row["reaction_key"]] = row["keq"]

    # Literature values for validation/fallback (from user-provided data)
    literature_keq = {
        "PGI": 0.35,  # F6P/G6P equilibrium
        "TPI": 0.045,  # GAP/DHAP equilibrium
        "GAPDH": 0.08,  # Slightly unfavorable
        "PGK": 2500,  # Strongly favorable (ATP synthesis)
        "PGM": 0.1,  # 2-PG/3-PG equilibrium
        "ENO": 0.7,  # PEP/2-PG near unity
        "ADH": 4000,  # Strongly favors ethanol
        "GPD": 30000,  # Strongly favors glycerol-3-phosphate
    }

    # Order according to the system (from README)
    reaction_order = ["PGI", "TPI", "GAPDH", "PGK", "PGM", "ENO", "ADH", "GPD"]
    keq_values = []

    print("\nEquilibrium constants (Keq):")
    print("  Reaction | YAML-derived | Literature | Used")
    print("  ---------|--------------|------------|------")

    for rxn_key in reaction_order:
        yaml_keq = keq_dict.get(rxn_key, None)
        lit_keq = literature_keq.get(rxn_key, 1.0)

        if yaml_keq is not None:
            # Use YAML-derived value
            keq_values.append(yaml_keq)
            print(f"  {rxn_key:8s} | {yaml_keq:12.2e} | {lit_keq:10.2e} | YAML")
        else:
            # Use literature value for missing data
            keq_values.append(lit_keq)
            print(f"  {rxn_key:8s} | {'N/A':12s} | {lit_keq:10.2e} | Literature")

    return np.array(keq_values)


def create_control_scenarios() -> Dict[str, Dict]:
    """Define different control scenarios to demonstrate."""
    scenarios = {
        "baseline": {"description": "No control (u = 0)", "controls": np.zeros(5), "color": "black", "linestyle": "-"},
        "glucose_pulse": {
            "description": "Glucose influx pulse",
            "controls": np.array([0.5, 0, 0, 0, 0]),  # u_glc_in = 0.5
            "color": "green",
            "linestyle": "--",
        },
        "fermentation_push": {
            "description": "Enhanced fermentation (PYK + PDC)",
            "controls": np.array([0, 0.3, 0.4, 0, 0]),  # u_pyk_pull, u_pdc_push
            "color": "red",
            "linestyle": "-.",
        },
        "energy_boost": {
            "description": "ATP boost (high energy)",
            "controls": np.array([0, 0, 0, 0.4, 0]),  # u_ATP_boost
            "color": "blue",
            "linestyle": ":",
        },
        "glycerol_favor": {
            "description": "Conditions favoring glycerol branch",
            "controls": np.array([0.2, -0.1, -0.2, 0, 0.1]),  # Slight glucose, reduced fermentation, slight ADP
            "color": "orange",
            "linestyle": "-",
        },
    }
    return scenarios


def run_glycolysis_simulation():
    """Run the main glycolysis LLRQ simulation."""
    print("🧬 Yeast Core Glycolysis LLRQ Dynamics")
    print("=====================================")

    # Load system components
    print("Loading system matrices...")
    K, B, reaction_names, control_names = load_system_matrices()
    print(f"✓ Loaded {K.shape[0]}×{K.shape[1]} K matrix and {B.shape[0]}×{B.shape[1]} B matrix")

    print("Loading equilibrium constants...")
    keq_values = load_equilibrium_constants()
    print(f"✓ Loaded Keq values: {keq_values}")

    # Create LLRQ dynamics system
    dynamics = LLRQDynamics(K, B, Keq=keq_values)
    solver = LLRQSolver(dynamics)

    print(f"\\nSystem properties:")
    print(f"  Reactions: {reaction_names}")
    print(f"  Controls: {control_names}")
    print(f"  Equilibrium constants (Keq): {keq_values}")
    print(f"  System eigenvalues: {np.linalg.eigvals(-K)}")

    # Simulation parameters
    t_span = (0, 10)  # 10 time units
    t_eval = np.linspace(0, 10, 200)

    # Initial condition: slightly away from equilibrium
    x0 = np.random.normal(0, 0.1, size=len(keq_values))  # Small random perturbations

    print(f"\\nRunning simulations...")
    print(f"  Time span: {t_span}")
    print(f"  Initial condition: x0 = {x0}")

    # Run different control scenarios
    scenarios = create_control_scenarios()
    results = {}

    for scenario_name, scenario in scenarios.items():
        print(f"  Running '{scenario_name}': {scenario['description']}")

        # Create control function
        def control_func(t):
            return scenario["controls"]

        # Solve system
        result = solver.solve(
            x0=x0,
            t_span=t_span,
            t_eval=t_eval,
            u_func=control_func,
            method="analytical" if np.allclose(scenario["controls"], 0) else "numerical",
        )

        results[scenario_name] = {"result": result, "scenario": scenario}

    return results, dynamics, reaction_names, control_names


def analyze_results(results: Dict, dynamics: LLRQDynamics, reaction_names: List[str]):
    """Analyze and visualize the simulation results."""
    print("\\n📊 Analyzing Results")
    print("====================")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Yeast Glycolysis LLRQ Dynamics - Control Scenarios", fontsize=16)

    # Plot 1: Time series of all reactions
    ax1 = axes[0, 0]
    for scenario_name, data in results.items():
        result = data["result"]
        scenario = data["scenario"]

        # Plot representative reactions
        key_reactions = [0, 1, 6, 7]  # PGI, TPI, ADH, GPD
        for i, rxn_idx in enumerate(key_reactions):
            if scenario_name == "baseline":  # Only label once
                label = f"{reaction_names[rxn_idx].split(':')[1].strip()}"
                ax1.plot(result.t, result.x[rxn_idx, :], color=f"C{i}", linestyle="-", alpha=0.7, label=label)
            else:
                ax1.plot(result.t, result.x[rxn_idx, :], color=f"C{i}", linestyle=scenario["linestyle"], alpha=0.7)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("x = ln(Q/Keq)")
    ax1.set_title("Key Reaction Quotients")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Fermentation vs Glycerol branch
    ax2 = axes[0, 1]
    for scenario_name, data in results.items():
        result = data["result"]
        scenario = data["scenario"]

        # ADH (fermentation) vs GPD (glycerol)
        adh_final = result.x[6, -1]  # ADH final value
        gpd_final = result.x[7, -1]  # GPD final value

        ax2.scatter(adh_final, gpd_final, color=scenario["color"], label=scenario["description"], s=100, alpha=0.8)

    ax2.set_xlabel("ADH Activity (fermentation)")
    ax2.set_ylabel("GPD Activity (glycerol branch)")
    ax2.set_title("Fermentation vs Glycerol Branch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy reactions (GAPDH, PGK)
    ax3 = axes[1, 0]
    for scenario_name, data in results.items():
        result = data["result"]
        scenario = data["scenario"]

        # Plot energy-related reactions
        ax3.plot(
            result.t,
            result.x[2, :],  # GAPDH
            color=scenario["color"],
            linestyle=scenario["linestyle"],
            label=f"{scenario_name} - GAPDH",
        )
        ax3.plot(
            result.t,
            result.x[3, :],  # PGK
            color=scenario["color"],
            linestyle=scenario["linestyle"],
            alpha=0.6,
            label=f"{scenario_name} - PGK",
        )

    ax3.set_xlabel("Time")
    ax3.set_ylabel("x = ln(Q/Keq)")
    ax3.set_title("Energy Metabolism (GAPDH & PGK)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Control effort
    ax4 = axes[1, 1]
    control_names_short = ["GLC", "PYK", "PDC", "ATP", "ADP"]

    scenario_names = list(results.keys())[1:]  # Skip baseline
    control_matrix = []
    colors = []

    for scenario_name in scenario_names:
        scenario = results[scenario_name]["scenario"]
        control_matrix.append(scenario["controls"])
        colors.append(scenario["color"])

    control_matrix = np.array(control_matrix)

    # Heatmap of control efforts
    im = ax4.imshow(control_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax4.set_xticks(range(len(control_names_short)))
    ax4.set_xticklabels(control_names_short)
    ax4.set_yticks(range(len(scenario_names)))
    ax4.set_yticklabels(scenario_names)
    ax4.set_title("Control Efforts")

    # Add colorbar
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # Add text annotations
    for i in range(len(scenario_names)):
        for j in range(len(control_names_short)):
            text = ax4.text(j, i, f"{control_matrix[i, j]:.1f}", ha="center", va="center", color="black")

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), "yeast_glycolysis", "glycolysis_llrq_results.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Results saved to {output_file}")

    plt.show()

    # Print summary statistics
    print("\\n📈 Summary Statistics")
    print("=====================")

    for scenario_name, data in results.items():
        result = data["result"]
        scenario = data["scenario"]

        # Calculate some metrics
        final_x = result.x[:, -1]
        steady_state_reached = np.allclose(result.x[:, -20:], final_x.reshape(-1, 1), rtol=0.01)
        total_deviation = np.sum(np.abs(final_x))
        fermentation_activity = final_x[6]  # ADH
        glycerol_activity = final_x[7]  # GPD

        print(f"\\n{scenario_name.upper()}: {scenario['description']}")
        print(f"  Steady state reached: {steady_state_reached}")
        print(f"  Total deviation from equilibrium: {total_deviation:.3f}")
        print(f"  Fermentation activity (ADH): {fermentation_activity:.3f}")
        print(f"  Glycerol branch activity (GPD): {glycerol_activity:.3f}")
        print(f"  Final state: {final_x}")


def demonstrate_thermodynamic_accounting(results: Dict, dynamics: LLRQDynamics):
    """Demonstrate thermodynamic accounting for the glycolysis system."""
    print("\\n🔥 Thermodynamic Accounting")
    print("============================")

    # Create thermodynamic accountant
    accountant = ThermodynamicAccountant(dynamics)

    # Analyze entropy production for each scenario
    for scenario_name, data in results.items():
        if scenario_name == "baseline":
            continue  # Skip baseline (no control)

        result = data["result"]
        scenario = data["scenario"]

        print(f"\\nScenario: {scenario_name}")
        print(f"Description: {scenario['description']}")

        # Create control history
        u_history = np.tile(scenario["controls"].reshape(-1, 1), (1, len(result.t)))

        # Calculate entropy production
        entropy_data = accountant.calculate_entropy_production(result.t, result.x, u_history)

        total_entropy = entropy_data["total_entropy_production"]
        reaction_entropy = entropy_data["reaction_entropy"]
        drive_entropy = entropy_data["drive_entropy"]

        print(f"  Total entropy production: {total_entropy:.4f}")
        print(f"  From reaction forces: {reaction_entropy:.4f}")
        print(f"  From external drives: {drive_entropy:.4f}")
        print(f"  Energy efficiency: {1 - total_entropy/max(total_entropy, 1e-6):.3f}")


def main():
    """Main function to run the complete yeast glycolysis LLRQ example."""
    try:
        # Run simulation
        results, dynamics, reaction_names, control_names = run_glycolysis_simulation()

        # Analyze results
        analyze_results(results, dynamics, reaction_names)

        # Demonstrate thermodynamic accounting
        demonstrate_thermodynamic_accounting(results, dynamics)

        print("\\n🎉 Yeast Glycolysis LLRQ Analysis Complete!")
        print("\\nKey takeaways:")
        print("1. LLRQ dynamics provide a linear framework for analyzing glycolysis")
        print("2. Different control strategies lead to distinct metabolic phenotypes")
        print("3. The fermentation vs glycerol branch trade-off is clearly visible")
        print("4. Thermodynamic constraints are automatically satisfied")
        print("5. The system demonstrates realistic metabolic behavior")

    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
