# Generate synthetic data (if missing) and fit the LLRQ (Wiener) model.
# This cell is self-contained.

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

OUT_DIR = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Synthetic data generator (same CM rate-law idea) ----------
def loguniform(low, high, size=None, rng=None):
    assert low > 0 and high > 0 and high > low
    u = (rng or np.random).uniform(low=np.log(low), high=np.log(high), size=size)
    return np.exp(u)


def sample_cm_params(seed=42):
    rng = np.random.default_rng(seed)
    kM_A = float(loguniform(1e-3, 1e1, rng=rng))
    kM_B = float(loguniform(1e-3, 1e1, rng=rng))
    kM_C = float(loguniform(1e-3, 1e1, rng=rng))
    keq = float(loguniform(1e-3, 1e3, rng=rng))
    kV = float(loguniform(1e-1, 1e3, rng=rng))
    u = float(loguniform(1e-3, 1.0, rng=rng))
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
    denominator = (1.0 + a_p) * (1.0 + b_p) + (1.0 + c_p) ** 2 - 1.0
    return p["u"] * numerator / denominator


def add_noise(v_true, rng, rel_sigma=0.05, abs_sigma=None):
    v_true = np.asarray(v_true, dtype=float)
    if abs_sigma is None:
        med = np.median(np.abs(v_true))
        abs_sigma = 1e-3 * (med if med > 0 else 1.0)
    sigma = rel_sigma * np.abs(v_true) + abs_sigma
    noise = rng.normal(loc=0.0, scale=sigma, size=v_true.shape)
    return v_true + noise


def make_forward_dataset(p, n_points=25, n_reps=3, seed=777):
    rng = np.random.default_rng(seed)
    B_levels = np.array([0.1, 1.0, 10.0]) * p["kM_B"]
    A_vals = np.concatenate([[0.0], np.geomspace(max(1e-6, 1e-3 * p["kM_A"]), 50.0 * p["kM_A"], n_points - 1)])
    C_val = 0.0
    rows = []
    for B in B_levels:
        v_true_curve = v_cm(A_vals, B, C_val, p)
        for rep in range(1, n_reps + 1):
            v_obs = add_noise(v_true_curve, rng=rng)
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


def make_reverse_dataset(p, n_points=25, n_reps=3, seed=888):
    rng = np.random.default_rng(seed)
    A_fix = 0.05 * p["kM_A"]
    B_fix = 0.05 * p["kM_B"]
    C_vals = np.concatenate([[0.0], np.geomspace(max(1e-9, 1e-3 * p["kM_C"]), 50.0 * p["kM_C"], n_points - 1)])
    rows = []
    v_true_curve = v_cm(A_fix, B_fix, C_vals, p)
    for rep in range(1, n_reps + 1):
        v_obs = add_noise(v_true_curve, rng=rng)
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


# If synthetic files are missing, create them now
def ensure_synthetic():
    fwd_latest = sorted(OUT_DIR.glob("synthetic_forward_*.csv"))
    rev_latest = sorted(OUT_DIR.glob("synthetic_reverse_*.csv"))
    if fwd_latest and rev_latest:
        return fwd_latest[-1], rev_latest[-1], None
    # regenerate
    params = sample_cm_params(seed=20250912)
    forward_df = make_forward_dataset(params)
    reverse_df = make_reverse_dataset(params)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_path = OUT_DIR / f"cm_params_{timestamp}.json"
    forward_path = OUT_DIR / f"synthetic_forward_{timestamp}.csv"
    reverse_path = OUT_DIR / f"synthetic_reverse_{timestamp}.csv"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    forward_df.to_csv(forward_path, index=False)
    reverse_df.to_csv(reverse_path, index=False)
    return forward_path, reverse_path, params_path


fwd_csv, rev_csv, maybe_params = ensure_synthetic()

# ---------- LLRQ (Wiener) fit ----------
df_fwd = pd.read_csv(fwd_csv)
df_rev = pd.read_csv(rev_csv)
df = pd.concat([df_fwd, df_rev], ignore_index=True)

eps = 1e-12
A = df["A_mM"].values + eps
B = df["B_mM"].values + eps
C = df["C_mM"].values + eps
y = df["v_obs"].values.astype(float)

logA = np.log(A)
logB = np.log(B)
logC = np.log(C)


def predict_unconstrained(p):
    log_alpha, t0, tA, tB, tC = p
    alpha = np.exp(log_alpha)
    x = t0 + tA * logA + tB * logB + tC * logC
    return alpha * (1.0 - np.exp(x))


def loss_unconstrained(p):
    yhat = predict_unconstrained(p)
    resid = yhat - y
    return float(np.mean(resid**2))


alpha0 = max(np.std(y), 1e-6)
p0 = np.array([np.log(alpha0), 0.0, -0.5, -0.5, 1.0], dtype=float)

if SCIPY_OK:
    res_u = minimize(loss_unconstrained, p0, method="L-BFGS-B")
    p_u = res_u.x
else:
    rng = np.random.default_rng(0)
    p_u = p0.copy()
    best = loss_unconstrained(p_u)
    scale = np.array([0.2, 0.5, 0.5, 0.5, 0.5])
    for _ in range(3000):
        cand = p_u + rng.normal(scale=scale)
        val = loss_unconstrained(cand)
        if val < best:
            best, p_u = val, cand

yhat_u = predict_unconstrained(p_u)
resid_u = yhat_u - y
rmse_u = float(np.sqrt(np.mean(resid_u**2)))
r2_u = float(1.0 - np.sum(resid_u**2) / np.sum((y - y.mean()) ** 2))


def predict_stoich(q):
    log_alpha, w0 = q
    alpha = np.exp(log_alpha)
    x = w0 - logA - logB + 2.0 * logC
    return alpha * (1.0 - np.exp(x))


def loss_stoich(q):
    yhat = predict_stoich(q)
    resid = yhat - y
    return float(np.mean(resid**2))


alpha0_s = max(np.std(y), 1e-6)
q0 = np.array([np.log(alpha0_s), 0.0])

if SCIPY_OK:
    res_s = minimize(loss_stoich, q0, method="L-BFGS-B")
    q_s = res_s.x
else:
    rng = np.random.default_rng(1)
    q_s = q0.copy()
    best = loss_stoich(q_s)
    scale = np.array([0.2, 0.5])
    for _ in range(3000):
        cand = q_s + rng.normal(scale=scale)
        val = loss_stoich(cand)
        if val < best:
            best, q_s = val, cand

yhat_s = predict_stoich(q_s)
resid_s = yhat_s - y
rmse_s = float(np.sqrt(np.mean(resid_s**2)))
r2_s = float(1.0 - np.sum(resid_s**2) / np.sum((y - y.mean()) ** 2))

alpha_u = float(np.exp(p_u[0]))
thetas = dict(theta0=float(p_u[1]), thetaA=float(p_u[2]), thetaB=float(p_u[3]), thetaC=float(p_u[4]))

alpha_s = float(np.exp(q_s[0]))
w0 = float(q_s[1])
Keq_hat = float(np.exp(-w0))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pred_path = OUT_DIR / f"llrq_predictions_{timestamp}.csv"
params_path = OUT_DIR / f"llrq_fit_params_{timestamp}.json"

out = df.copy()
out["v_hat_unconstrained"] = yhat_u
out["v_hat_stoich"] = yhat_s
out.to_csv(pred_path, index=False)

params = {
    "unconstrained": {"alpha": alpha_u, **thetas, "rmse": rmse_u, "r2": r2_u},
    "stoichiometry_constrained": {"alpha": alpha_s, "w0": w0, "Keq_hat": Keq_hat, "rmse": rmse_s, "r2": r2_s},
    "data_files": {
        "forward_csv": fwd_csv.as_posix(),
        "reverse_csv": rev_csv.as_posix(),
        "cm_params_json": None if maybe_params is None else maybe_params.as_posix(),
    },
    "notes": "LLRQ-Wiener fit with v = alpha * (1 - exp(x)); x is log-linear in concentrations. Stoichiometric model enforces x = ln(Q/Keq).",
}
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)

# --- Plots ---
plt.figure()
plt.scatter(df["v_obs"].values, yhat_u, s=10)
minv, maxv = float(np.min(df["v_obs"].values)), float(np.max(df["v_obs"].values))
plt.plot([minv, maxv], [minv, maxv])
plt.xlabel("Observed v")
plt.ylabel("Predicted v (unconstrained)")
plt.title("LLRQ-Wiener fit: observed vs predicted (unconstrained)")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(df["v_obs"].values, yhat_s, s=10)
minv, maxv = float(np.min(df["v_obs"].values)), float(np.max(df["v_obs"].values))
plt.plot([minv, maxv], [minv, maxv])
plt.xlabel("Observed v")
plt.ylabel("Predicted v (stoichiometry-constrained)")
plt.title("LLRQ-Wiener fit: observed vs predicted (stoichiometry-constrained)")
plt.tight_layout()
plt.show()

# Forward overlay plot (unconstrained fit means by [A])
plt.figure()
mask_fwd = (df["protocol"] == "forward_A_var").values
df_f = df.loc[mask_fwd, ["A_mM", "series_id", "v_obs"]].copy()
df_f["yhat"] = out.loc[mask_fwd, "v_hat_unconstrained"].values
m_obs = df_f.groupby(["series_id", "A_mM"], as_index=False)["v_obs"].mean()
m_hat = df_f.groupby(["series_id", "A_mM"], as_index=False)["yhat"].mean()

for sid in sorted(m_obs["series_id"].unique()):
    sub_o = m_obs[m_obs["series_id"] == sid]
    sub_h = m_hat[m_hat["series_id"] == sid]
    sub = sub_o.merge(sub_h, on=["series_id", "A_mM"], suffixes=("_obs", "_hat"))
    plt.plot(sub["A_mM"].values, sub["v_obs"].values, label=f"{sid} obs")
    plt.plot(sub["A_mM"].values, sub["v_hat"].values, linestyle="--", label=f"{sid} fit")
plt.xscale("log")
plt.xlabel("[A] (mM)")
plt.ylabel("v")
plt.title("Forward protocol: mean curves per B level (unconstrained fit)")
plt.legend()
plt.tight_layout()
plt.show()

(
    params_path.as_posix(),
    pred_path.as_posix(),
    {
        "unconstrained": {"alpha": alpha_u, **thetas, "rmse": rmse_u, "r2": r2_u},
        "stoich": {"alpha": alpha_s, "Keq_hat": Keq_hat, "rmse": rmse_s, "r2": r2_s},
    },
)
