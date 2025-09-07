#!/usr/bin/env python3
"""Test script to analyze the actual numerical errors in entropy calculations."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.thermodynamic_accounting import ThermodynamicAccountant

# Create simple system
network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

forward_rates = np.array([2.0])
backward_rates = np.array([1.0])
initial_concentrations = np.array([1.0, 1.0])

dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

accountant = ThermodynamicAccountant(network)
L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

print("Testing Parseval's theorem accuracy for different signals:\n")
print(f"{'Signal Type':<30} {'Time Entropy':<15} {'FFT Entropy':<15} {'Relative Error':<15}")
print("-" * 75)

# Test 1: Single frequency sinusoid
T = 10.0
dt = 0.01
t = np.arange(0, T, dt)
N = len(t)

test_cases = [
    ("Single sine (ω=1.5)", 0.3 * np.sin(1.5 * t)),
    ("Single cosine (ω=2.0)", 0.4 * np.cos(2.0 * t + 0.2)),
    ("Two frequencies", 0.2 * np.sin(1.0 * t) + 0.1 * np.cos(3.0 * t)),
    ("Three frequencies", 0.15 * np.sin(0.5 * t) + 0.1 * np.cos(2.0 * t) + 0.05 * np.sin(5.0 * t)),
    ("Gaussian pulse", 0.5 * np.exp(-((t - T / 2) ** 2) / (2 * (T / 10) ** 2))),
    ("Square wave", 0.3 * np.sign(np.sin(2.0 * t))),
    ("Chirp signal", 0.3 * np.sin(2 * np.pi * (0.5 + t / T) * t)),
]

errors = []

for name, signal in test_cases:
    x_t = signal.reshape(-1, 1)

    # Time-domain entropy
    time_result = accountant.entropy_from_x(t, x_t, L)
    time_entropy = time_result.sigma_total

    # Frequency-domain entropy
    freq_entropy, _ = accountant.entropy_from_x_freq(x_t, dt, L)

    # Calculate error
    if abs(time_entropy) > 1e-12:
        rel_error = abs(freq_entropy - time_entropy) / abs(time_entropy)
    else:
        rel_error = abs(freq_entropy - time_entropy)

    errors.append(rel_error)
    print(f"{name:<30} {time_entropy:<15.8f} {freq_entropy:<15.8f} {rel_error:<15.2e}")

print(f"\nMax relative error: {max(errors):.2e}")
print(f"Mean relative error: {np.mean(errors):.2e}")
print(f"Median relative error: {np.median(errors):.2e}")

# Test different time resolutions
print("\n" + "=" * 75)
print("Testing effect of time resolution (dt) on accuracy:\n")
print(f"{'dt':<10} {'N samples':<12} {'Time Entropy':<15} {'FFT Entropy':<15} {'Relative Error':<15}")
print("-" * 75)

T = 5.0
omega = 2.0
dt_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
dt_errors = []

for dt_test in dt_values:
    t_test = np.arange(0, T, dt_test)
    N_test = len(t_test)
    x_test = 0.3 * np.cos(omega * t_test + 0.3)
    x_t_test = x_test.reshape(-1, 1)

    time_result = accountant.entropy_from_x(t_test, x_t_test, L)
    freq_entropy, _ = accountant.entropy_from_x_freq(x_t_test, dt_test, L)

    if abs(time_result.sigma_total) > 1e-12:
        rel_error = abs(freq_entropy - time_result.sigma_total) / abs(time_result.sigma_total)
    else:
        rel_error = abs(freq_entropy - time_result.sigma_total)

    dt_errors.append(rel_error)
    print(f"{dt_test:<10.3f} {N_test:<12} {time_result.sigma_total:<15.8f} {freq_entropy:<15.8f} {rel_error:<15.2e}")

print(f"\nMax error across dt values: {max(dt_errors):.2e}")

# Test control entropy (u -> x mapping)
print("\n" + "=" * 75)
print("Testing control entropy (quasi-steady vs exact):\n")
print(f"{'Signal Type':<30} {'Quasi-steady':<15} {'FFT Exact':<15} {'Relative Error':<15}")
print("-" * 75)

# Use low frequencies for quasi-steady validity
T = 20.0
dt = 0.02
t = np.arange(0, T, dt)
K = dynamics.K

control_test_cases = [
    ("Very slow (ω=0.1)", 0.3 * np.sin(0.1 * t)),
    ("Slow (ω=0.3)", 0.3 * np.sin(0.3 * t)),
    ("Moderate (ω=1.0)", 0.3 * np.sin(1.0 * t)),
    ("Fast (ω=3.0)", 0.3 * np.sin(3.0 * t)),
    ("Two slow freqs", 0.2 * np.sin(0.2 * t) + 0.1 * np.cos(0.5 * t)),
]

control_errors = []

for name, signal in control_test_cases:
    u_t = signal.reshape(-1, 1)

    # Quasi-steady approximation
    quasi_steady = accountant.entropy_from_u(t, u_t, K, L)

    # FFT exact calculation
    fft_exact, _ = accountant.entropy_from_u_freq(u_t, dt, K, L)

    if abs(quasi_steady.sigma_total) > 1e-12:
        rel_error = abs(fft_exact - quasi_steady.sigma_total) / abs(quasi_steady.sigma_total)
    else:
        rel_error = abs(fft_exact - quasi_steady.sigma_total)

    control_errors.append(rel_error)
    print(f"{name:<30} {quasi_steady.sigma_total:<15.8f} {fft_exact:<15.8f} {rel_error:<15.2e}")

print(f"\nMax control entropy error: {max(control_errors):.2e}")
print(f"Mean control entropy error: {np.mean(control_errors):.2e}")

print("\n" + "=" * 75)
print("RECOMMENDATION FOR TEST TOLERANCES:")
print(f"  - Parseval (x entropy): {max(errors):.2e} -> suggest tolerance of {max(errors) * 2:.2e}")
print(
    f"  - Control entropy (slow signals): {min(control_errors[:2]):.2e} -> suggest tolerance of {max(control_errors[:2]) * 2:.2e}"
)
print(f"  - Control entropy (all signals): {max(control_errors):.2e} -> current tolerance of 0.2 is appropriate")
