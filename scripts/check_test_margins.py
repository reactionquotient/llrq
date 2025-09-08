#!/usr/bin/env python3
"""Check actual errors in the tests to see how much we can tighten tolerances."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llrq import ReactionNetwork, LLRQDynamics, LLRQSolver
from llrq.thermodynamic_accounting import ThermodynamicAccountant

# Recreate the test fixtures
network = ReactionNetwork(species_ids=["A", "B"], reaction_ids=["R1"], stoichiometric_matrix=[[-1], [1]])

forward_rates = np.array([2.0])
backward_rates = np.array([1.0])
initial_concentrations = np.array([1.0, 1.0])

dynamics = LLRQDynamics.from_mass_action(network, forward_rates, backward_rates, initial_concentrations)

accountant = ThermodynamicAccountant(network)
L = accountant.compute_onsager_conductance(initial_concentrations, forward_rates, backward_rates, mode="local")

print("Checking actual errors in test cases:\n")

# Test 1: test_entropy_from_x_freq_parseval
print("1. test_entropy_from_x_freq_parseval")
T = 5.0
dt = 0.02
t = np.arange(0, T, dt)
omega_test = 1.5
x_t = 0.3 * np.cos(omega_test * t + 0.2)
x_t = x_t.reshape(-1, 1)

time_result = accountant.entropy_from_x(t, x_t, L)
freq_entropy, _ = accountant.entropy_from_x_freq(x_t, dt, L)
relative_error = abs(freq_entropy - time_result.sigma_total) / abs(time_result.sigma_total)

print(f"   Time entropy: {time_result.sigma_total:.8f}")
print(f"   FFT entropy:  {freq_entropy:.8f}")
print(f"   Relative error: {relative_error:.2e}")
print(f"   Current tolerance: 1e-2 (0.01)")
print(f"   Can tighten to: {relative_error * 2:.2e}\n")

# Test 2: test_entropy_from_u_freq_consistency
print("2. test_entropy_from_u_freq_consistency")
T = 10.0
dt = 0.02
t = np.arange(0, T, dt)
u_t = 0.3 * np.sin(0.2 * t) + 0.1 * np.cos(0.5 * t)  # Low frequencies
u_t = u_t.reshape(-1, 1)
K = dynamics.K

entropy_u_fft, _ = accountant.entropy_from_u_freq(u_t, dt, K, L)
entropy_u_time = accountant.entropy_from_u(t, u_t, K, L)
relative_error = abs(entropy_u_fft - entropy_u_time.sigma_total) / abs(entropy_u_time.sigma_total)

print(f"   Quasi-steady entropy: {entropy_u_time.sigma_total:.8f}")
print(f"   FFT exact entropy:    {entropy_u_fft:.8f}")
print(f"   Relative error: {relative_error:.2e}")
print(f"   Current tolerance: 0.2")
print(f"   Can tighten to: {relative_error * 2:.2e}\n")

# Test 3: test_validate_parseval_entropy
print("3. test_validate_parseval_entropy")
T = 6.0
dt = 0.02
t = np.arange(0, T, dt)
x_t = 0.4 * np.cos(1.0 * t) + 0.2 * np.sin(3.0 * t)
x_t = x_t.reshape(-1, 1)

validation = accountant.validate_parseval_entropy(x_t, dt, L)
print(f"   Time domain: {validation['time_domain']:.8f}")
print(f"   Freq domain: {validation['frequency_domain']:.8f}")
print(f"   Relative error: {validation['relative_error']:.2e}")
print(f"   Current tolerance: 1e-2 (0.01)")
print(f"   Can tighten to: {validation['relative_error'] * 2:.2e}\n")

print("=" * 60)
print("SUMMARY OF RECOMMENDED TOLERANCE CHANGES:")
print("  1. test_entropy_from_x_freq_parseval: 0.01 -> 0.002")
print("  2. test_entropy_from_u_freq_consistency: 0.2 -> 0.05")
print("  3. test_validate_parseval_entropy: 0.01 -> 0.002")
