#!/usr/bin/env python3
"""
Extract equilibrium constants for yeast glycolysis reactions from yeast-GEM.yml

This script loads the yeast-GEM model, identifies the glycolysis reactions,
extracts thermodynamic data (ΔG° values), and computes equilibrium constants
for use in LLRQ dynamics.

Reactions analyzed:
1. PGI: G6P ↔ F6P (phosphoglucose isomerase)
2. TPI: DHAP ↔ GAP (triose phosphate isomerase)
3. GAPDH: GAP+Pi+NAD ↔ 1,3-BPG+NADH (glyceraldehyde-3-phosphate dehydrogenase)
4. PGK: 1,3-BPG+ADP ↔ 3-PG+ATP (phosphoglycerate kinase)
5. PGM: 3-PG ↔ 2-PG (phosphoglycerate mutase)
6. ENO: 2-PG ↔ PEP (enolase)
7. ADH: AcAld+NADH ↔ EtOH+NAD (alcohol dehydrogenase, fermentation)
8. GPD: DHAP+NADH ↔ G3P+NAD (glycerol-3-phosphate dehydrogenase, glycerol branch)
"""

import os
import sys
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llrq.yaml_parser import YAMLModelParser
from llrq.utils.thermodynamics import delta_g_to_keq


def main():
    # Load yeast-GEM.yml model
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yeast-GEM.yml")

    if not os.path.exists(model_path):
        print(f"Error: Could not find yeast-GEM.yml at {model_path}")
        return

    print("Loading yeast-GEM.yml model...")
    try:
        parser = YAMLModelParser(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get species information (contains ΔG° values)
    species_info = parser.get_species_info()
    print(f"Found {len(species_info)} metabolites")

    # Get reaction information
    reaction_info = parser.get_reaction_info()
    print(f"Found {len(reaction_info)} reactions")

    # Define the glycolysis reactions we're interested in
    target_reactions = {
        "PGI": {"name": "PGI", "description": "G6P ↔ F6P"},
        "TPI": {"name": "TPI", "description": "DHAP ↔ GAP"},
        "GAPDH": {"name": "GAPDH", "description": "GAP+Pi+NAD ↔ 1,3-BPG+NADH"},
        "PGK": {"name": "PGK", "description": "1,3-BPG+ADP ↔ 3-PG+ATP"},
        "PGM": {"name": "PGM", "description": "3-PG ↔ 2-PG"},
        "ENO": {"name": "ENO", "description": "2-PG ↔ PEP"},
        "ADH": {"name": "ADH", "description": "AcAld+NADH ↔ EtOH+NAD"},
        "GPD": {"name": "GPD", "description": "DHAP+NADH ↔ G3P+NAD"},
    }

    print("\nSearching for target reactions...")
    found_reactions = find_target_reactions(reaction_info, target_reactions)

    print(f"\nFound {len(found_reactions)} target reactions:")
    for rxn_key, rxn_data in found_reactions.items():
        print(f"  {rxn_key}: {rxn_data['id']} - {rxn_data.get('name', 'No name')}")

    # Calculate Keq values for found reactions
    print("\nCalculating equilibrium constants...")
    keq_results = calculate_reaction_keq_values(found_reactions, species_info)

    # Save results to CSV
    output_file = os.path.join(os.path.dirname(__file__), "yeast_keq_values.csv")
    save_keq_results(keq_results, target_reactions, output_file)
    print(f"\n✓ Results saved to {output_file}")


def find_target_reactions(reaction_info: List[Dict], target_reactions: Dict) -> Dict:
    """Find the target glycolysis reactions in the model."""
    found_reactions = {}

    # Direct mapping of reaction IDs found in the model
    direct_mapping = {
        "PGI": "r_0467",  # glucose-6-phosphate isomerase
        "TPI": "r_1054",  # triose-phosphate isomerase
        "GAPDH": "r_0486",  # glyceraldehyde-3-phosphate dehydrogenase
        "PGK": "r_0892",  # phosphoglycerate kinase
        "PGM": "r_0893",  # phosphoglycerate mutase
        "ENO": "r_0366",  # enolase
        "ADH": "r_0163",  # alcohol dehydrogenase
        "GPD": "r_0490",  # glycerol-3-phosphate dehydrogenase
    }

    # Create lookup by ID
    rxn_by_id = {rxn.get("id"): rxn for rxn in reaction_info}

    # Find reactions using direct mapping
    for target_key, rxn_id in direct_mapping.items():
        if rxn_id in rxn_by_id:
            found_reactions[target_key] = rxn_by_id[rxn_id]
            print(f"  Found {target_key}: {rxn_id} - {rxn_by_id[rxn_id].get('name', 'No name')}")

    return found_reactions


def calculate_reaction_keq_values(reactions: Dict, species_info: Dict) -> Dict:
    """Calculate Keq values for reactions using thermodynamic data."""
    results = {}

    for rxn_key, rxn_data in reactions.items():
        print(f"\nProcessing {rxn_key} ({rxn_data['id']})...")

        # Get metabolites and stoichiometry
        metabolites = rxn_data.get("metabolites", {})

        if not metabolites:
            print(f"  Warning: No metabolites found for {rxn_key}")
            continue

        # Calculate reaction ΔG° = Σ(products·ΔG°) - Σ(reactants·ΔG°)
        reaction_delta_g = 0.0
        missing_data = []

        print(f"  Metabolites and stoichiometry:")
        for species_id, coeff in metabolites.items():
            if species_id in species_info:
                species_dg = species_info[species_id].get("delta_g")
                if species_dg is not None and species_dg != 10000000:  # 10000000 is placeholder
                    contribution = float(coeff) * float(species_dg)
                    reaction_delta_g += contribution
                    print(f"    {species_id}: coeff={coeff:.3f}, ΔG°={species_dg:.2f} kJ/mol, contribution={contribution:.2f}")
                else:
                    missing_data.append(species_id)
                    print(f"    {species_id}: coeff={coeff:.3f}, ΔG°=MISSING")
            else:
                missing_data.append(species_id)
                print(f"    {species_id}: NOT FOUND in species database")

        if missing_data:
            print(f"  Warning: Missing ΔG° data for {len(missing_data)} species: {missing_data}")
            keq_value = None
        else:
            # Convert to Keq
            try:
                keq_value = delta_g_to_keq(reaction_delta_g, T=298.15, units="kJ/mol")
                print(f"  Reaction ΔG° = {reaction_delta_g:.2f} kJ/mol")
                print(f"  Keq = {keq_value:.6e}")
            except Exception as e:
                print(f"  Error calculating Keq: {e}")
                keq_value = None

        results[rxn_key] = {
            "reaction_id": rxn_data["id"],
            "name": rxn_data.get("name", ""),
            "delta_g": reaction_delta_g if not missing_data else None,
            "keq": keq_value,
            "missing_species": missing_data,
            "metabolites": metabolites,
        }

    return results


def save_keq_results(results: Dict, target_reactions: Dict, output_file: str):
    """Save Keq calculation results to CSV file."""
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["reaction_key", "reaction_id", "description", "name", "delta_g_kj_mol", "keq", "data_quality", "notes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for rxn_key in target_reactions:
            description = target_reactions[rxn_key]["description"]

            if rxn_key in results:
                result = results[rxn_key]

                # Determine data quality
                if result["keq"] is not None:
                    quality = "COMPLETE"
                    notes = ""
                elif result["missing_species"]:
                    quality = "INCOMPLETE"
                    notes = f"Missing ΔG° for: {', '.join(result['missing_species'])}"
                else:
                    quality = "ERROR"
                    notes = "Calculation failed"

                writer.writerow(
                    {
                        "reaction_key": rxn_key,
                        "reaction_id": result["reaction_id"],
                        "description": description,
                        "name": result["name"],
                        "delta_g_kj_mol": result["delta_g"] if result["delta_g"] is not None else "",
                        "keq": result["keq"] if result["keq"] is not None else "",
                        "data_quality": quality,
                        "notes": notes,
                    }
                )
            else:
                writer.writerow(
                    {
                        "reaction_key": rxn_key,
                        "reaction_id": "NOT_FOUND",
                        "description": description,
                        "name": "",
                        "delta_g_kj_mol": "",
                        "keq": "",
                        "data_quality": "NOT_FOUND",
                        "notes": "Reaction not found in model",
                    }
                )

    print(f"\nSummary of results:")
    complete_count = sum(1 for r in results.values() if r["keq"] is not None)
    print(f"  Complete Keq calculations: {complete_count}/{len(target_reactions)}")
    print(f"  Missing thermodynamic data: {len(results) - complete_count}")
    print(f"  Reactions not found: {len(target_reactions) - len(results)}")


if __name__ == "__main__":
    main()
