# Synthetic data generator for a bimolecular, reversible enzyme mechanism
# following the Common Modular (CM) rate law in Eq. (3) of Liebermeister et al. (2010).
#
# v = u * (k_plus * a' * b' - k_minus * c'^2) / [ (1 + a')(1 + b') + (1 + c')^2 - 1 ]
# where a' = a/kM_A, b' = b/kM_B, c' = c/kM_C
#
# We sample thermodynamically consistent parameters using the Haldane relationship
# and generate initial-rate style datasets by varying one concentration while
# holding the others fixed. We add Gaussian noise to the measured velocities.
#
# This cell will:
# 1) Sample a random parameter set (with a reproducible seed).
# 2) Generate two datasets:
#    - "forward": vary [A] at several fixed [B] levels, with [C]=0 (initial-rate forward direction).
#    - "reverse": vary [C] at fixed low [A],[B] levels (driving the reverse direction).
# 3) Save CSVs and a parameters JSON to /mnt/data and display previews & a quick plot.

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Define output directory
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)


def loguniform(low, high, size=None, rng=None):
    """Sample from a log-uniform distribution between low and high (inclusive)."""
    assert low > 0 and high > 0 and high > low
    u = (rng or np.random).uniform(low=np.log(low), high=np.log(high), size=size)
    return np.exp(u)


def sample_cm_params(seed=42):
    rng = np.random.default_rng(seed)
    # Reactant constants (mM): broad but reasonable biochemical range
    kM_A = float(loguniform(1e-3, 1e1, rng=rng))
    kM_B = float(loguniform(1e-3, 1e1, rng=rng))
    kM_C = float(loguniform(1e-3, 1e1, rng=rng))
    # Equilibrium constant (dimensionless). Wide range to cover reversible regimes.
    keq = float(loguniform(1e-3, 1e3, rng=rng))
    # Geometric mean of turnover rates (s^-1)
    kV = float(loguniform(1e-1, 1e3, rng=rng))
    # Enzyme amount (arbitrary units scaling the rate)
    u = float(loguniform(1e-3, 1.0, rng=rng))

    # Haldane-consistent k_plus and k_minus for A + B <-> 2 C (h=1; n_A=-1, n_B=-1, n_C=+2)
    # ln k± = ln kV ± 1/2 ( ln keq - sum_i n_i ln kM_i )
    sum_n_ln_kM = (-1) * math.log(kM_A) + (-1) * math.log(kM_B) + (2) * math.log(kM_C)
    ln_k_plus = math.log(kV) + 0.5 * (math.log(keq) - sum_n_ln_kM)
    ln_k_minus = math.log(kV) - 0.5 * (math.log(keq) - sum_n_ln_kM)
    k_plus, k_minus = math.exp(ln_k_plus), math.exp(ln_k_minus)

    return {
        "seed": seed,
        "u": u,
        "kM_A": kM_A,
        "kM_B": kM_B,
        "kM_C": kM_C,
        "keq": keq,
        "kV": kV,
        "k_plus": k_plus,
        "k_minus": k_minus,
        "mechanism": "A + B <-> 2 C (CM rate law, Eq. 3)",
    }


def v_cm(a, b, c, p):
    a_p = a / p["kM_A"]
    b_p = b / p["kM_B"]
    c_p = c / p["kM_C"]
    numerator = p["k_plus"] * a_p * b_p - p["k_minus"] * (c_p**2)
    denominator = (1.0 + a_p) * (1.0 + b_p) + (1.0 + c_p) ** 2 - 1.0  # = 1 + a'+b'+a'b' + 2c' + c'^2
    return p["u"] * numerator / denominator


def add_noise(v_true, rng, rel_sigma=0.05, abs_sigma=None):
    """Add Gaussian noise. If abs_sigma is None, compute from median |v_true|."""
    v_true = np.asarray(v_true, dtype=float)
    if abs_sigma is None:
        med = np.median(np.abs(v_true))
        abs_sigma = 1e-3 * (med if med > 0 else 1.0)
    sigma = rel_sigma * np.abs(v_true) + abs_sigma
    noise = rng.normal(loc=0.0, scale=sigma, size=v_true.shape)
    return v_true + noise


def make_forward_dataset(p, n_points=100, n_reps=3, seed=123, noise: bool = True):
    """Vary [A] at several fixed [B] levels with [C]=0 (initial-rate forward)."""
    rng = np.random.default_rng(seed)
    # Choose B levels relative to kM_B
    B_levels = np.array([0.1, 1.0, 10.0]) * p["kM_B"]
    # A grid (include 0 explicitly plus a log-spaced range)
    A_vals = np.concatenate([[0.0], np.geomspace(max(1e-6, 1e-3 * p["kM_A"]), 50.0 * p["kM_A"], n_points - 1)])
    C_val = 0.0

    rows = []
    for j, B in enumerate(B_levels, start=1):
        v_true_curve = v_cm(A_vals, B, C_val, p)
        # replicate with noise
        for rep in range(1, n_reps + 1):
            if noise:
                v_obs = add_noise(v_true_curve, rng=rng)
            else:
                v_obs = v_true_curve
            for ai, A in enumerate(A_vals):
                rows.append(
                    {
                        "protocol": "forward_A_var",
                        "series_id": f"B={B:.4g}",
                        "replicate": rep,
                        "A_mM": float(A),
                        "B_mM": float(B),
                        "C_mM": float(C_val),
                        "v_true": float(v_true_curve[ai]),
                        "v_obs": float(v_obs[ai]),
                    }
                )
    return pd.DataFrame(rows)


def make_reverse_dataset(p, n_points=100, n_reps=3, seed=456, noise: bool = True):
    """Vary [C] at fixed small [A],[B] (reverse direction)."""
    rng = np.random.default_rng(seed)
    # Fix small A,B (near zero-substrate conditions)
    A_fix = 0.05 * p["kM_A"]
    B_fix = 0.05 * p["kM_B"]
    # C grid (include 0 plus log-spaced)
    C_vals = np.concatenate([[0.0], np.geomspace(max(1e-6, 1e-3 * p["kM_C"]), 50.0 * p["kM_C"], n_points - 1)])

    rows = []
    v_true_curve = v_cm(A_fix, B_fix, C_vals, p)
    for rep in range(1, n_reps + 1):
        if noise:
            v_obs = add_noise(v_true_curve, rng=rng)
        else:
            v_obs = v_true_curve
        for ci, C in enumerate(C_vals):
            rows.append(
                {
                    "protocol": "reverse_C_var",
                    "series_id": f"A={A_fix:.4g},B={B_fix:.4g}",
                    "replicate": rep,
                    "A_mM": float(A_fix),
                    "B_mM": float(B_fix),
                    "C_mM": float(C),
                    "v_true": float(v_true_curve[ci]),
                    "v_obs": float(v_obs[ci]),
                }
            )
    return pd.DataFrame(rows)


# ---- Generate everything ----
params = sample_cm_params(seed=20250912)  # fixed seed based on today's date for reproducibility
forward_df = make_forward_dataset(params, n_points=25, n_reps=3, seed=777)
reverse_df = make_reverse_dataset(params, n_points=25, n_reps=3, seed=888)

# Save outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
params_path = OUT_DIR / f"cm_params_{timestamp}.json"
forward_path = OUT_DIR / f"synthetic_forward_{timestamp}.csv"
reverse_path = OUT_DIR / f"synthetic_reverse_{timestamp}.csv"

with params_path.open("w") as f:
    json.dump(params, f, indent=2)
forward_df.to_csv(forward_path, index=False)
reverse_df.to_csv(reverse_path, index=False)

# Display quick previews
print("Forward dataset (preview):")
print(forward_df.head(20))
print("\nReverse dataset (preview):")
print(reverse_df.head(20))

# Quick visualization (one plot per dataset)
plt.figure()
for series, sub in forward_df.groupby("series_id"):
    # Use replicate average to make cleaner lines in the preview plot
    m = sub.groupby("A_mM", as_index=False)["v_obs"].mean()
    plt.plot(m["A_mM"].values, m["v_obs"].values, label=series)
plt.xscale("log")
plt.xlabel("[A] (mM)")
plt.ylabel("Observed velocity")
plt.title("Forward dataset: v vs [A] at fixed [B] (C=0)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
m = reverse_df.groupby("C_mM", as_index=False)["v_obs"].mean()
plt.plot(m["C_mM"].values, m["v_obs"].values)
plt.xscale("log")
plt.xlabel("[C] (mM)")
plt.ylabel("Observed velocity")
plt.title("Reverse dataset: v vs [C] at fixed [A], [B]")
plt.tight_layout()
plt.show()
