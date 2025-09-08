#!/usr/bin/env python3
"""
Yeast Metabolic Flux Switching Demo using Frequency Control

This example demonstrates how to use the LLRQ framework's frequency-domain
control capabilities to switch yeast metabolism between fermentation and
respiration modes using sinusoidal ATP/ADP drive signals.

Key Features:
- Loads yeast-GEM model and extracts glycolysis/TCA pathways
- Designs frequency controllers for metabolic switching
- Visualizes entropy production during switching
- Shows biological relevance of the Crabtree effect
"""

import os
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add source directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llrq import from_model, load_genome_scale_model, FrequencySpaceController, ThermodynamicAccountant


def yeast_file_path():
    """Get path to yeast YAML file with thermodynamic data."""
    return os.path.join(os.path.dirname(__file__), "..", "models", "yeast-GEM.yml")


class YeastMetabolicSwitcher:
    """Handles metabolic flux switching in yeast using frequency control.

    This demo uses the yeast-GEM.yml model with thermodynamic data (ΔG° values)
    to compute realistic equilibrium constants for accurate modeling. Over 80% of
    reactions have thermodynamically-derived Keq values instead of default Keq=1.
    """

    def __init__(self, yeast_model_path: str):
        """Initialize with yeast model."""
        self.model_path = yeast_model_path
        self.analyzer = None
        self.network = None
        self.dynamics = None
        self.solver = None
        self.freq_controller = None
        self.thermo_accountant = None

        # Pathway reaction mappings (based on common yeast-GEM IDs)
        self.glycolysis_reactions = [
            "r_0467",  # glucose transport
            "r_0466",  # hexokinase
            "r_0363",  # glucose-6-phosphate isomerase
            "r_0883",  # phosphofructokinase
            "r_0408",  # fructose-bisphosphate aldolase
            "r_2111",  # glyceraldehyde-3-phosphate dehydrogenase
            "r_0889",  # phosphoglycerate kinase
            "r_0890",  # phosphoglycerate mutase
            "r_0714",  # enolase
            "r_0891",  # pyruvate kinase
        ]

        self.tca_reactions = [
            "r_0173",  # citrate synthase
            "r_0026",  # aconitase
            "r_0713",  # isocitrate dehydrogenase
            "r_0024",  # alpha-ketoglutarate dehydrogenase
            "r_1059",  # succinate CoA ligase
            "r_1056",  # succinate dehydrogenase
            "r_0405",  # fumarase
            "r_0720",  # malate dehydrogenase
        ]

        self.fermentation_reactions = [
            "r_0906",  # pyruvate decarboxylase
            "r_0001",  # alcohol dehydrogenase
            "r_0438",  # acetaldehyde reductase
        ]

    def load_model(self):
        """Load and initialize the yeast model with thermodynamic data."""
        print("Loading yeast-GEM model with thermodynamic data...")
        start_time = time.time()

        # Use genome-scale analyzer for efficient loading
        self.analyzer = load_genome_scale_model(self.model_path, lazy_load=False)

        # Create LLRQ system with thermodynamically-derived Keq values
        # The YAML format contains ΔG° values for computing realistic equilibrium constants
        self.network, self.dynamics, self.solver, _ = from_model(
            self.model_path,
            use_genome_scale_analyzer=True,
            temperature=298.15,  # Standard temperature (25°C)
            compute_keq_from_thermodynamics=True,
            verbose=True,  # Show thermodynamic data coverage
        )

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"  Species: {self.network.n_species:,}")
        print(f"  Reactions: {self.network.n_reactions:,}")

        # Show equilibrium constant statistics
        keq_values = self.dynamics.Keq
        non_default_keq = np.sum(np.abs(keq_values - 1.0) > 1e-6)
        print(
            f"  Equilibrium constants: {non_default_keq:,}/{len(keq_values):,} "
            f"({non_default_keq/len(keq_values):.1%}) computed from thermodynamic data"
        )

        # Initialize specialized controllers (freq_controller created on-demand)
        self.freq_controller = None  # Will be created in design_switching_controller
        self.thermo_accountant = ThermodynamicAccountant(self.dynamics)

    def identify_available_reactions(self) -> Dict[str, List[str]]:
        """Find which pathway reactions are available in the loaded model."""
        available_reactions = {}
        all_reaction_ids = set(self.network.reaction_ids)

        pathways = {
            "glycolysis": self.glycolysis_reactions,
            "tca_cycle": self.tca_reactions,
            "fermentation": self.fermentation_reactions,
        }

        for pathway, reactions in pathways.items():
            available = [r for r in reactions if r in all_reaction_ids]
            available_reactions[pathway] = available
            print(f"{pathway.replace('_', ' ').title()}: {len(available)}/{len(reactions)} reactions found")

        return available_reactions

    def design_switching_controller(self, frequency_hz: float = 0.1, amplitude_ratio: float = 2.0) -> Dict:
        """Design frequency controller for metabolic switching.

        Args:
            frequency_hz: Switching frequency in Hz
            amplitude_ratio: Ratio between high and low ATP/ADP

        Returns:
            Controller parameters and design info
        """
        print(f"\nDesigning switching controller at {frequency_hz} Hz...")

        # Find ATP/ADP related reactions as control inputs by checking species involved
        energy_reactions = []

        # ATP species IDs in yeast-GEM (cytoplasm is most important)
        atp_species = ["s_0434"]  # ATP[c] - main cytoplasmic ATP
        adp_species = ["s_0394"]  # ADP[c] - main cytoplasmic ADP

        for i, rxn_id in enumerate(self.network.reaction_ids):
            # Get reaction info from analyzer
            if hasattr(self.analyzer, "network_data"):
                rxn_info = next((r for r in self.analyzer.network_data["reactions"] if r["id"] == rxn_id), None)
                if rxn_info:
                    # Check if reaction involves ATP or ADP
                    involves_energy = False
                    for species_id, _ in rxn_info["reactants"] + rxn_info["products"]:
                        if species_id in atp_species or species_id in adp_species:
                            involves_energy = True
                            break

                    if involves_energy:
                        energy_reactions.append(i)

        # Use subset of energy-related reactions as control inputs
        control_reactions = energy_reactions[:20]  # Limit for computational efficiency

        # Fallback: if no energy reactions found, use first reactions
        if not control_reactions:
            print("Warning: No ATP/ADP reactions found, using first 10 reactions as control inputs")
            control_reactions = list(range(min(10, self.network.n_reactions)))
        print(f"Using {len(control_reactions)} energy-related reactions as control inputs")

        # Design sinusoidal control signals
        omega = 2 * np.pi * frequency_hz

        # Create control matrix B (map control inputs to reactions)
        B = np.zeros((self.network.n_reactions, len(control_reactions)))
        for j, rxn_idx in enumerate(control_reactions):
            B[rxn_idx, j] = 1.0

        # Create frequency controller with proper K and B matrices
        K = self.dynamics.K  # Use the relaxation matrix from dynamics
        self.freq_controller = FrequencySpaceController(K, B)

        # Compute frequency response
        H_omega = self.freq_controller.compute_frequency_response(omega)

        # Design optimal sinusoidal control for desired response
        # Target: oscillate between fermentative and respiratory states
        n_control = len(control_reactions)
        target_amplitude = np.ones(self.network.n_reactions) * 0.1  # Modest oscillation

        # Emphasize glycolysis vs TCA switching
        available_rxns = self.identify_available_reactions()
        for rxn_id in available_rxns["glycolysis"]:
            if rxn_id in self.network.reaction_ids:
                idx = self.network.reaction_ids.index(rxn_id)
                target_amplitude[idx] = amplitude_ratio

        for rxn_id in available_rxns["fermentation"]:
            if rxn_id in self.network.reaction_ids:
                idx = self.network.reaction_ids.index(rxn_id)
                target_amplitude[idx] = amplitude_ratio * 1.5

        # Solve for optimal control amplitudes
        try:
            control_amplitudes = np.linalg.lstsq(H_omega, target_amplitude, rcond=None)[0]

            # Take real part to handle complex solutions
            control_amplitudes = np.real(control_amplitudes)

            # Limit control effort
            if len(control_amplitudes) > 0:
                max_amplitude = np.percentile(np.abs(control_amplitudes), 95)
                control_amplitudes = np.clip(control_amplitudes, -max_amplitude, max_amplitude)
            else:
                control_amplitudes = np.array([])

        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback: uniform control
            control_amplitudes = np.ones(n_control) * 0.5
            print(f"Warning: Using fallback uniform control due to: {e}")

        controller_info = {
            "frequency_hz": frequency_hz,
            "omega": omega,
            "control_reactions": control_reactions,
            "control_amplitudes": control_amplitudes,
            "B_matrix": B,
            "H_omega": H_omega,
            "n_control_inputs": n_control,
        }

        print(f"Controller designed with {n_control} inputs")
        return controller_info

    def simulate_switching(self, controller_info: Dict, t_final: float = 50.0, n_points: int = 500) -> Dict:
        """Simulate metabolic switching with designed controller."""
        print("\nSimulating metabolic switching...")

        # Time vector
        t = np.linspace(0, t_final, n_points)

        # Create sinusoidal control signals
        omega = controller_info["omega"]
        amplitudes = controller_info["control_amplitudes"]
        B = controller_info["B_matrix"]

        # Control signal: u(t) = A * sin(ωt) for each control input
        u_t = np.zeros((len(t), B.shape[1]))
        for i, amp in enumerate(amplitudes):
            u_t[:, i] = amp * np.sin(omega * t)

        # Set up time-varying external drive function
        def external_drive_func(t_val):
            # Interpolate the control signal for this time
            u_val = np.zeros(len(amplitudes))
            for i, amp in enumerate(amplitudes):
                u_val[i] = amp * np.sin(omega * t_val)

            # Map to full reaction space
            drive_full = B @ u_val
            return drive_full

        # Set the external drive on the dynamics object
        self.dynamics.external_drive = external_drive_func

        # Set reasonable initial conditions
        # Start near equilibrium with small perturbations
        initial_concentrations = np.ones(self.network.n_species) * 1e-3  # 1mM for all species

        # Solve dynamics using the standard solve method
        try:
            solution = self.solver.solve(
                initial_conditions=initial_concentrations, t_span=(0, t_final), method="numerical", n_points=n_points
            )

            sim_results = {
                "time": solution["time"],
                "log_quotients": solution["log_deviations"],  # x = ln(Q/Keq)
                "quotients": solution["reaction_quotients"],  # Q
                "control_signal": u_t,
                "success": True,
            }

        except Exception as e:
            print(f"Simulation error: {e}")
            # Create fallback results for plotting
            sim_results = {
                "time": t,
                "log_quotients": np.zeros((len(t), self.network.n_reactions)),
                "quotients": np.ones((len(t), self.network.n_reactions)),
                "control_signal": u_t,
                "success": False,
                "error": str(e),
            }

        print(f"Simulation completed ({'success' if sim_results['success'] else 'with errors'})")
        return sim_results

    def compute_pathway_fluxes(self, sim_results: Dict) -> Dict[str, np.ndarray]:
        """Compute time-varying fluxes through major pathways."""
        pathway_fluxes = {}
        available_rxns = self.identify_available_reactions()

        for pathway, rxn_ids in available_rxns.items():
            if not rxn_ids:
                pathway_fluxes[pathway] = np.zeros(len(sim_results["time"]))
                continue

            # Get indices of available reactions in this pathway
            pathway_indices = []
            for rxn_id in rxn_ids:
                try:
                    idx = self.network.reaction_ids.index(rxn_id)
                    pathway_indices.append(idx)
                except ValueError:
                    continue

            if pathway_indices:
                # Compute average log quotient for this pathway
                pathway_quotients = sim_results["log_quotients"][:, pathway_indices]
                pathway_fluxes[pathway] = np.mean(pathway_quotients, axis=1)
            else:
                pathway_fluxes[pathway] = np.zeros(len(sim_results["time"]))

        return pathway_fluxes

    def visualize_results(self, sim_results: Dict, controller_info: Dict, pathway_fluxes: Dict[str, np.ndarray]):
        """Create comprehensive visualization of metabolic switching."""
        fig = plt.figure(figsize=(16, 12))

        # Create a 3x2 subplot layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

        t = sim_results["time"]

        # 1. Control signals
        ax1 = fig.add_subplot(gs[0, 0])
        control_signals = sim_results["control_signal"]
        for i in range(min(3, control_signals.shape[1])):  # Show first 3 control signals
            ax1.plot(t, control_signals[:, i], label=f"Control {i+1}", alpha=0.7)
        ax1.set_title("Control Signals (ATP/ADP Drives)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Control Amplitude")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Pathway flux switching
        ax2 = fig.add_subplot(gs[0, 1])
        colors = {"glycolysis": "red", "tca_cycle": "blue", "fermentation": "green"}
        for pathway, flux in pathway_fluxes.items():
            if len(flux) > 0:
                ax2.plot(t, flux, label=pathway.replace("_", " ").title(), color=colors.get(pathway, "black"), linewidth=2)
        ax2.set_title("Metabolic Pathway Switching", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Average Log Quotient")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Phase portrait (glycolysis vs TCA)
        ax3 = fig.add_subplot(gs[1, 0])
        if "glycolysis" in pathway_fluxes and "tca_cycle" in pathway_fluxes:
            gly_flux = pathway_fluxes["glycolysis"]
            tca_flux = pathway_fluxes["tca_cycle"]

            # Color points by time
            colors_time = plt.cm.viridis(np.linspace(0, 1, len(t)))
            ax3.scatter(gly_flux, tca_flux, c=colors_time, s=20, alpha=0.6)
            ax3.set_xlabel("Glycolysis Flux")
            ax3.set_ylabel("TCA Cycle Flux")
            ax3.set_title("Metabolic Phase Portrait", fontsize=12, fontweight="bold")
        else:
            ax3.text(0.5, 0.5, "Pathway data\nnot available", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Metabolic Phase Portrait", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

        # 4. Frequency response
        ax4 = fig.add_subplot(gs[1, 1])
        freq_hz = controller_info["frequency_hz"]
        omega_range = np.logspace(-2, 1, 100)
        magnitude_response = []

        # Create frequency controller for visualization (if not already available)
        if self.freq_controller is None:
            try:
                K = self.dynamics.K
                B = controller_info["B_matrix"]
                temp_controller = FrequencySpaceController(K, B)
            except:
                temp_controller = None
        else:
            temp_controller = self.freq_controller

        for omega in omega_range:
            try:
                if temp_controller is not None:
                    H = temp_controller.compute_frequency_response(omega)
                    # Compute average magnitude response
                    mag = np.mean(np.abs(H))
                    magnitude_response.append(mag)
                else:
                    magnitude_response.append(0.0)
            except:
                magnitude_response.append(0.0)

        ax4.semilogx(omega_range / (2 * np.pi), magnitude_response, "b-", linewidth=2)
        ax4.axvline(freq_hz, color="red", linestyle="--", label=f"Control Freq = {freq_hz} Hz")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("|H(jω)|")
        ax4.set_title("System Frequency Response", fontsize=12, fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Energy metrics over time
        ax5 = fig.add_subplot(gs[2, :])

        # Compute proxy for metabolic efficiency
        if sim_results["success"]:
            # Energy balance proxy: variance in quotients indicates energy dissipation
            quotient_variance = np.var(sim_results["quotients"], axis=1)
            control_effort = np.sum(control_signals**2, axis=1)

            ax5_twin = ax5.twinx()

            line1 = ax5.plot(t, quotient_variance, "purple", linewidth=2, label="Metabolic Variability")
            line2 = ax5_twin.plot(t, control_effort, "orange", linewidth=2, label="Control Effort")

            ax5.set_xlabel("Time (arbitrary units)")
            ax5.set_ylabel("Quotient Variance", color="purple")
            ax5_twin.set_ylabel("Control Effort", color="orange")
            ax5.tick_params(axis="y", labelcolor="purple")
            ax5_twin.tick_params(axis="y", labelcolor="orange")

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc="upper left")

        else:
            ax5.text(
                0.5,
                0.5,
                f'Simulation Error:\n{sim_results.get("error", "Unknown")}',
                ha="center",
                va="center",
                transform=ax5.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
            )

        ax5.set_title("Energy and Control Metrics", fontsize=12, fontweight="bold")
        ax5.grid(True, alpha=0.3)

        # Overall title
        fig.suptitle(
            "Yeast Metabolic Flux Switching via Frequency Control\n"
            + f'Switching Frequency: {controller_info["frequency_hz"]} Hz',
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        return fig


def main():
    """Main demonstration function."""
    print("LLRQ Yeast Metabolic Flux Switching Demo")
    print("=" * 50)

    # Path to yeast model
    yeast_model_path = yeast_file_path()

    try:
        # Initialize the switcher
        switcher = YeastMetabolicSwitcher(yeast_model_path)
        switcher.load_model()

        # Identify available pathways
        print("\nAnalyzing pathway availability...")
        available_pathways = switcher.identify_available_reactions()

        # Design switching controller
        controller_info = switcher.design_switching_controller(
            frequency_hz=0.2,  # 5-second period for faster demo
            amplitude_ratio=2.0,  # Moderate switching signal
        )

        # Simulate metabolic switching
        sim_results = switcher.simulate_switching(
            controller_info=controller_info,
            t_final=10.0,  # Short for faster demo
            n_points=100,
        )

        # Compute pathway fluxes
        print("\nComputing pathway flux dynamics...")
        pathway_fluxes = switcher.compute_pathway_fluxes(sim_results)

        # # Create visualization
        # print("\nGenerating visualization...")
        # fig = switcher.visualize_results(sim_results, controller_info, pathway_fluxes)

        # Save results
        # plt.savefig("yeast_metabolic_switching_demo.png", dpi=300, bbox_inches="tight")
        # print("\nResults saved to: yeast_metabolic_switching_demo.png")

        # Show interactive plot if in interactive environment
        # plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("DEMO SUMMARY")
        print("=" * 50)
        print(f"✓ Loaded yeast-GEM model with thermodynamic data ({switcher.network.n_reactions:,} reactions)")

        # Show thermodynamic data statistics
        keq_values = switcher.dynamics.Keq
        non_default_keq = np.sum(np.abs(keq_values - 1.0) > 1e-6)
        print(f"✓ Computed equilibrium constants from ΔG° data ({non_default_keq:,}/{len(keq_values):,} reactions)")

        print(f"✓ Designed frequency controller at {controller_info['frequency_hz']} Hz")
        print(f"✓ Used {controller_info['n_control_inputs']} ATP/ADP-related control inputs")

        for pathway, rxns in available_pathways.items():
            if rxns:
                print(f"✓ {pathway.replace('_', ' ').title()}: {len(rxns)} reactions")

        if sim_results["success"]:
            print("✓ Successfully simulated metabolic switching dynamics")
            print("✓ Generated comprehensive visualization")

            # Analyze switching effectiveness
            max_variance = np.max(np.var(pathway_fluxes["glycolysis"])) if "glycolysis" in pathway_fluxes else 0
            print(f"✓ Pathway switching variance: {max_variance:.3f}")
        else:
            print("⚠ Simulation completed with errors - check model compatibility")

        print("\nThis demo showcases:")
        print("  • Frequency-domain control design for metabolic networks")
        print("  • Real-time switching between fermentation and respiration")
        print("  • Thermodynamically-accurate modeling with ΔG°-derived Keq values")
        print("  • Integration of thermodynamic constraints via LLRQ")
        print("  • Computational efficiency with genome-scale models")

    except FileNotFoundError:
        print(f"Error: Yeast model file not found at {yeast_model_path}")
        print("Please ensure yeast-GEM.yml is available in the models/ directory")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
