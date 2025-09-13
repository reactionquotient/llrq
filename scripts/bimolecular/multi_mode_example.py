#!/usr/bin/env python3
"""
Simple example of using the multi-mode LLRQ fitting with CM dynamics.

This demonstrates how to use the integrated functionality from cm_multi_mode_fitting.py
"""

import numpy as np
from cm_multi_mode_fitting import run_cm_multi_mode_comparison, fit_llrq_multi_exp, fit_llrq_with_offset
from cm_rate_law_integrated import sample_params, simulate


def simple_example():
    """Run a simple multi-mode fitting example."""

    print("=" * 60)
    print("Simple Multi-Mode LLRQ Fitting Example")
    print("=" * 60)

    # Generate CM parameters
    params = sample_params(seed=42)
    print(
        f"CM Parameters: u={params.u:.3f}, Keq={(params.k_plus/params.k_minus)*(params.kM_C**2)/(params.kM_A*params.kM_B):.2e}"
    )

    # Run the complete comparison
    results = run_cm_multi_mode_comparison(params=params, t_span=(0.0, 2.0), n_points=150, save_plots=True)

    # Extract results
    single_mode = results["single_mode"]
    multi_mode = results["multi_mode"]
    offset = results["offset"]

    print("\n" + "=" * 60)
    print("FITTING RESULTS SUMMARY")
    print("=" * 60)

    if single_mode.get("K_fit") is not None:
        print(f"Single-mode LLRQ:")
        print(f"  K = {single_mode['K_fit']:.3f}")
        print(f"  R² (ln Q) = {single_mode.get('r_squared_lnQ', 'N/A'):.4f}")

    if multi_mode.get("success"):
        print(f"\nTwo-mode LLRQ:")
        print(f"  K rates = {multi_mode['k']}")
        print(f"  Weights = {multi_mode['w']}")
        print(f"  Fitted Keq = {multi_mode['Keq']:.2e}")
        print(f"  Cost = {multi_mode['cost']:.2e}")

        # Check if it's really using two modes or collapsing to one
        if multi_mode["w"][0] < 1e-10 or multi_mode["w"][1] < 1e-10:
            print("  Note: Effectively single-mode (one weight ≈ 0)")
        else:
            print("  Note: True two-mode dynamics detected")

    if offset.get("success"):
        print(f"\nOffset LLRQ:")
        print(f"  k = {offset['k']:.3f}")
        print(f"  u (external drive) = {offset['u']:.3f}")
        print(f"  Fitted Keq = {offset['Keq']:.2e}")
        print(f"  Cost = {offset['cost']:.2e}")

    print(f"\nTrue Keq = {results['true_Keq']:.2e}")

    return results


def advanced_example():
    """Show how to use individual fitting functions directly."""

    print("\n" + "=" * 60)
    print("Advanced Example: Direct Function Usage")
    print("=" * 60)

    # Generate some data
    params = sample_params(seed=123)
    t_eval = np.linspace(0, 1.5, 100)

    # Forward experiment
    t_fwd, A_fwd, B_fwd, C_fwd, _ = simulate((0, 1.5), [0.5, 0.8, 0.01], params, t_eval=t_eval)

    # Reverse experiment
    t_rev, A_rev, B_rev, C_rev, _ = simulate((0, 1.5), [0.1, 0.1, 0.5], params, t_eval=t_eval)

    # Prepare data lists
    ts_list = [t_fwd, t_rev]
    A_list = [A_fwd, A_rev]
    B_list = [B_fwd, B_rev]
    C_list = [C_fwd, C_rev]

    print("Running multi-mode fitting directly...")

    # Try different numbers of modes
    for M in [2, 3]:
        print(f"\nTrying {M}-mode fit:")
        result = fit_llrq_multi_exp(ts_list, A_list, B_list, C_list, M=M)

        if result.get("success"):
            print(f"  Success! K = {result['k']}")
            print(f"  Weights = {result['w']}")
            print(f"  Cost = {result['cost']:.2e}")
        else:
            print(f"  Failed: {result.get('message', 'Unknown error')}")

    # Try offset fitting
    print(f"\nTrying offset fitting:")
    offset_result = fit_llrq_with_offset(ts_list, A_list, B_list, C_list)

    if offset_result.get("success"):
        print(f"  Success! k = {offset_result['k']:.3f}, u = {offset_result['u']:.3f}")
        print(f"  Cost = {offset_result['cost']:.2e}")
    else:
        print(f"  Failed: {offset_result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    # Run simple example
    results = simple_example()

    # Run advanced example
    advanced_example()

    print("\n" + "=" * 60)
    print("Example completed! Check the output/ directory for plots.")
    print("=" * 60)
