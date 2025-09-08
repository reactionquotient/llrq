#!/usr/bin/env python3
"""
Quick test to check yeast model availability and basic LLRQ functionality.
"""

import os
import sys

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    # Check if yeast model exists
    yeast_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yeast-GEM.xml")

    print("Checking yeast model availability...")
    print(f"Looking for: {yeast_model_path}")

    if os.path.exists(yeast_model_path):
        size_mb = os.path.getsize(yeast_model_path) / (1024 * 1024)
        print(f"✓ Yeast model found ({size_mb:.1f} MB)")

        # Try basic import
        try:
            from llrq import load_genome_scale_model

            print("✓ LLRQ imports successful")

            # Try loading (with timeout protection)
            print("Testing model loading...")
            analyzer = load_genome_scale_model(yeast_model_path, lazy_load=True)
            print("✓ Yeast model loads successfully")

            stats = analyzer.get_model_statistics()
            print(f"✓ Model has {stats['n_species']:,} species and {stats['n_reactions']:,} reactions")

            print("\n🎉 Ready to run yeast_metabolic_flux_switching.py!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")

    else:
        print("❌ Yeast model not found")
        print("\nTo get the yeast model:")
        print("1. Download yeast-GEM.xml from https://github.com/SysBioChalmers/yeast-GEM")
        print("2. Place it in ../models/yeast-GEM.xml")
        print("3. Or run the demo with a different SBML model")


if __name__ == "__main__":
    main()
