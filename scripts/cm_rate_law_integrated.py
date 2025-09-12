# Re-run generator (state was reset)

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime
from caas_jupyter_tools import display_dataframe_to_user
import json

OUT_DIR = Path("/mnt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CMParams:
    u: float
    kM_A: float
    kM_B: float
    kM_C: float
    k_plus: float
    k_minus: float


def rate_cm(a, b, c, p: CMParams):
    a_p = a / p.kM_A
    b_p = b / p.kM_B
    c_p = c / p.kM_C
    num = p.k_plus * a_p * b_p - p.k_minus * (c_p**2)
    den = (1.0 + a_p) * (1.0 + b_p) + (1.0 + c_p) ** 2 - 1.0
    return p.u * num / den


def rhs(t, y, p: CMParams):
    A, B, C = y
    v = rate_cm(A, B, C, p)
    return [-v, -v, 2.0 * v]


def simulate(t_span, y0, params: CMParams, t_eval=None, rtol=1e-8, atol=1e-10):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)
    sol = solve_ivp(lambda t, y: rhs(t, y, params), t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol)
    A, B, C = sol.y
    v = rate_cm(A, B, C, params)
    return sol.t, A, B, C, v


def loguniform(rng, low, high):
    return np.exp(rng.uniform(np.log(low), np.log(high)))


def sample_params(seed=20250912):
    rng = np.random.default_rng(seed)
    kM_A = loguniform(rng, 1e-3, 1e1)
    kM_B = loguniform(rng, 1e-3, 1e1)
    kM_C = loguniform(rng, 1e-3, 1e1)
    Keq = loguniform(rng, 1e-3, 1e3)
    kV = loguniform(rng, 1e-1, 1e3)
    u = loguniform(rng, 1e-3, 1.0)
    sum_n_ln_kM = (-np.log(kM_A)) + (-np.log(kM_B)) + (2 * np.log(kM_C))
    ln_kplus = np.log(kV) + 0.5 * (np.log(Keq) - sum_n_ln_kM)
    ln_kminus = np.log(kV) - 0.5 * (np.log(Keq) - sum_n_ln_kM)
    k_plus, k_minus = np.exp(ln_kplus), np.exp(ln_kminus)
    return CMParams(float(u), float(kM_A), float(kM_B), float(kM_C), float(k_plus), float(k_minus))


# Parameters & experiments
params = sample_params()

t_eval = np.linspace(0.0, 200.0, 400)
expts = {
    "forward": dict(y0=[1.0 * params.kM_A, 1.0 * params.kM_B, 0.0]),
    "reverse": dict(y0=[0.05 * params.kM_A, 0.05 * params.kM_B, 1.0 * params.kM_C]),
}

csv_paths = {}
for name, spec in expts.items():
    t, A, B, C, v = simulate((t_eval[0], t_eval[-1]), spec["y0"], params, t_eval=t_eval)
    df = pd.DataFrame({"t": t, "A": A, "B": B, "C": C, "v": v, "exp_name": name})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"cm_timeseries_{name}_{ts}.csv"
    df.to_csv(path, index=False)
    csv_paths[name] = path.as_posix()
    display_dataframe_to_user(f"Preview: {name} timeseries", df.head(15))

# Quick plot for forward experiment
t, A, B, C, v = simulate((t_eval[0], t_eval[-1]), expts["forward"]["y0"], params, t_eval=t_eval)
plt.figure()
plt.plot(t, A, label="A")
plt.plot(t, B, label="B")
plt.plot(t, C, label="C")
plt.xlabel("time")
plt.ylabel("concentration")
plt.title("A + B <-> 2C (forward experiment)")
plt.legend()
plt.tight_layout()
plt.show()

param_path = OUT_DIR / "cm_params_timeseries.json"
with open(param_path, "w") as f:
    json.dump(params.__dict__, f, indent=2)

csv_paths, param_path.as_posix()
