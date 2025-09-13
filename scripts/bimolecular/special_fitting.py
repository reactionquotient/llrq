# Install required packages (scipy, numpy, pandas, matplotlib are already available in the environment)
import sys, os, json, math, textwrap, numpy as np, pandas as pd
from pathlib import Path

# --- Prep: import the user's module ---
# Create a tiny dummy `llrq` module to satisfy the import in cm_rate_law_integrated.py
open("/mnt/data/llrq.py", "w").write("# dummy llrq module\n")
sys.path.append("/mnt/data")

import importlib.util

spec = importlib.util.spec_from_file_location("cm", "/mnt/data/cm_rate_law_integrated.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from scipy.optimize import least_squares


# --------------------------
# Utility: compute Q and lnQ
# --------------------------
def _compute_lnQ(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    Q = cm.compute_reaction_quotient(A, B, C)
    return np.log(Q)


# ----------------------------------------------
# Multi-mode (Prony-style) LLRQ fit: shared Keq
# ----------------------------------------------
def fit_llrq_multi_exp(
    ts_list: List[np.ndarray], A_list: List[np.ndarray], B_list: List[np.ndarray], C_list: List[np.ndarray], M: int = 2
) -> Dict[str, Any]:
    """
    Fit ln Q(t) with a mixture of exponentials that share a common Keq and spectrum across runs:
       y_r(t) = ln Q_r(t) - ln Keq = y0_r * sum_m w_m * exp(-k_m t)
    Returns dict with fitted parameters and predictions.
    """
    R = len(ts_list)
    assert R == len(A_list) == len(B_list) == len(C_list)

    lnQ_list = [_compute_lnQ(A_list[r], B_list[r], C_list[r]) for r in range(R)]

    # --- parameterization helpers ---
    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def softmax(b):
        e = np.exp(b - np.max(b))
        return e / e.sum()

    def unpack_params(theta):
        off = 0
        th_k = theta[off : off + M]
        off += M
        b_w = theta[off : off + M]
        off += M
        lnKeq = theta[off]
        off += 1
        y0 = theta[off : off + R]
        off += R

        k = np.empty(M)
        k[0] = softplus(th_k[0])
        for m in range(1, M):
            k[m] = k[m - 1] + softplus(th_k[m])  # ordered rates
        w = softmax(b_w)
        Keq = np.exp(lnKeq)
        return k, w, Keq, y0

    def y_model(t, k, w, y0):
        # y(t) = y0 * sum_m w_m exp(-k_m t)
        E = np.exp(-np.outer(t, k))  # shape (T, M)
        return y0 * (E @ w)

    def residuals(theta):
        k, w, Keq, y0s = unpack_params(theta)
        res = []
        for r in range(R):
            t = ts_list[r]
            lnQ = lnQ_list[r]
            y_obs = lnQ - np.log(Keq)
            y_fit = y_model(t, k, w, y0s[r])
            res.append(y_obs - y_fit)
        return np.concatenate(res)

    # --- initialization ---
    t_all = np.concatenate(ts_list)
    tmin, tmax = float(t_all.min()), float(t_all.max())
    # choose rate guesses spanning the window
    # avoid zero times causing issues
    span_lo = 1.0 / max(1e-6 + tmax * 5.0, 1e-3)
    span_hi = max(5.0 / max(tmin, tmax * 1e-3), 0.1)
    ks0 = np.logspace(np.log10(max(span_lo, 1e-3)), np.log10(max(span_hi, 1e-2)), M)
    th_k0 = np.log(np.exp(ks0) - 1.0)
    b_w0 = np.zeros(M)
    Keq0 = np.median([np.median(np.exp(lnQ[-max(5, len(lnQ) // 5) :])) for lnQ in lnQ_list])
    Keq0 = max(Keq0, 1e-12)
    lnKeq0 = np.log(Keq0)
    y00 = np.array([lnQ_list[r][0] - lnKeq0 for r in range(R)])
    theta0 = np.concatenate([th_k0, b_w0, [lnKeq0], y00])

    out = least_squares(residuals, theta0, max_nfev=20000)
    k, w, Keq, y0s = unpack_params(out.x)

    # predictions
    preds = []
    for r in range(R):
        t = ts_list[r]
        lnQ_pred = np.log(Keq) + y_model(t, k, w, y0s[r])
        preds.append(lnQ_pred)

    return {
        "success": out.success,
        "cost": float(out.cost),
        "k": k,
        "w": w,
        "Keq": Keq,
        "y0": y0s,
        "lnQ_pred_list": preds,
        "lnQ_obs_list": lnQ_list,
        "times": ts_list,
        "message": out.message,
    }


# ----------------------------------------------------
# Piecewise-constant K(t) via segmentation of ln|y(t)|
# ----------------------------------------------------
def _piecewise_linear_fit(t: np.ndarray, z: np.ndarray, J: int) -> Dict[str, Any]:
    """
    Least-squares piecewise linear regression with exactly J segments.
    Returns changepoints (indices), and per-segment slope/intercept.
    """
    n = len(t)
    assert n == len(z)
    # Precompute necessary sums for O(1) SSE on intervals
    # We'll fit z ~ a + b * t on any [i, j)
    T1 = np.ones(n).cumsum()
    Tt = t.cumsum()
    Tz = z.cumsum()
    Ttt = (t * t).cumsum()
    Ttz = (t * z).cumsum()

    def sums(i, j):  # interval [i, j)  (i < j)
        cnt = T1[j - 1] - (T1[i - 1] if i > 0 else 0.0)
        st = Tt[j - 1] - (Tt[i - 1] if i > 0 else 0.0)
        sz = Tz[j - 1] - (Tz[i - 1] if i > 0 else 0.0)
        stt = Ttt[j - 1] - (Ttt[i - 1] if i > 0 else 0.0)
        stz = Ttz[j - 1] - (Ttz[i - 1] if i > 0 else 0.0)
        return cnt, st, sz, stt, stz

    def fit_interval(i, j):
        cnt, st, sz, stt, stz = sums(i, j)
        # Solve for [a, b] in least squares closed-form
        den = cnt * stt - st * st
        if abs(den) < 1e-12:
            b = 0.0
            a = sz / cnt
        else:
            b = (cnt * stz - st * sz) / den
            a = (sz - b * st) / cnt
        # SSE
        # SSE = sum(z^2) - a*sum(z) - b*sum(t z)  ... + computed terms
        # but to keep it simple: compute residuals directly (n is small)
        tt = t[i:j]
        zz = z[i:j]
        res = zz - (a + b * tt)
        sse = float((res**2).sum())
        return a, b, sse

    # DP
    INF = 1e300
    dp = np.full((J + 1, n + 1), INF)
    prev = [[-1] * (n + 1) for _ in range(J + 1)]
    dp[0, 0] = 0.0

    # Precompute interval fits to avoid repetition
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    SSE = np.zeros((n, n)) + INF
    for i in range(n):
        for j in range(i + 1, n + 1):
            a, b, sse = fit_interval(i, j)
            A[i, j - 1] = a
            B[i, j - 1] = b
            SSE[i, j - 1] = sse

    for jseg in range(1, J + 1):
        for j in range(1, n + 1):
            best = INF
            arg = -1
            for i in range(jseg - 1, j):  # at least one point per segment
                cand = dp[jseg - 1, i] + SSE[i, j - 1]
                if cand < best:
                    best = cand
                    arg = i
            dp[jseg, j] = best
            prev[jseg][j] = arg

    # backtrack to get changepoints
    cps = [n]
    jseg = J
    j = n
    while jseg > 0:
        i = prev[jseg][j]
        cps.append(i)
        j, jseg = i, jseg - 1
    cps = list(reversed(cps))  # [0 = start, cp1, cp2, ..., n]
    # build segments
    segs = []
    for s in range(J):
        i = cps[s]
        j = cps[s + 1]
        a = A[i, j - 1]
        b = B[i, j - 1]
        segs.append({"i": i, "j": j, "a": a, "b": b})
    return {"changepoints": cps, "segments": segs, "sse": float(dp[J, n])}


def fit_llrq_piecewiseK(
    t: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, J: int = 3, Keq: Optional[float] = None
) -> Dict[str, Any]:
    """
    Piecewise-constant K(t) by segmenting z(t) = ln|y(t)| where y = ln Q - ln Keq.
    If Keq is None, estimate it from tail median of Q.
    Returns K per segment, changepoints, and predicted lnQ(t).
    """
    lnQ = _compute_lnQ(A, B, C)
    if Keq is None:
        Keq = float(np.median(np.exp(lnQ[-max(5, len(lnQ) // 5) :])))
    lnKeq = float(np.log(Keq))

    y = lnQ - lnKeq
    # avoid log(0)
    eps = 1e-12
    z = np.log(np.maximum(np.abs(y), eps))

    # segment z(t) with J segments
    res_seg = _piecewise_linear_fit(t, z, J)
    segs = res_seg["segments"]

    # map slopes to K_j = -slope (since z' = d/dt ln|y| = -K(t) for monotone y)
    Ks = []
    for s in segs:
        Ks.append(max(1e-12, -float(s["b"])))

    # reconstruct piecewise exponential y_hat and lnQ_hat
    lnQ_hat = np.zeros_like(lnQ)
    y_hat = np.zeros_like(y)
    # initial y0
    y0 = float(y[0])
    start_idx = segs[0]["i"]
    assert start_idx == 0
    y_at_start = y0
    t0 = float(t[0])

    for s_idx, s in enumerate(segs):
        i, j = s["i"], s["j"]
        Kseg = Ks[s_idx]
        t_start = float(t[i])
        # ensure continuity
        if i > 0:
            y_at_start = float(y_hat[i - 1])
        for idx in range(i, j):
            dt = float(t[idx] - t_start)
            y_hat[idx] = y_at_start * np.exp(-Kseg * dt)
            lnQ_hat[idx] = lnKeq + y_hat[idx]

    return {
        "Keq": Keq,
        "Ks": np.array(Ks),
        "segments": segs,
        "changepoints": res_seg["changepoints"],
        "lnQ_obs": lnQ,
        "lnQ_hat": lnQ_hat,
        "y_obs": y,
        "y_hat": y_hat,
        "t": t,
        "sse": res_seg["sse"],
    }


# -------------------------------------------------
# Optional: single-mode with constant drive (u > 0)
# -------------------------------------------------
def fit_llrq_with_offset(
    ts_list: List[np.ndarray], A_list: List[np.ndarray], B_list: List[np.ndarray], C_list: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Fit affine LLRQ: d/dt ln Q = -k (ln Q - ln Keq) + u  ->
    y(t) = (y0 - u/k) e^{-k t} + u/k, with shared (k, u, Keq) across runs, run-specific y0.
    """
    R = len(ts_list)
    lnQ_list = [_compute_lnQ(A_list[r], B_list[r], C_list[r]) for r in range(R)]

    def unpack(theta):
        k = np.exp(theta[0])  # >0
        u = theta[1]  # can be any real
        lnKeq = theta[2]
        y0 = theta[3 : 3 + R]
        return k, u, lnKeq, y0

    def model_y(t, k, u, y0):
        if k < 1e-12:  # avoid division by zero
            return y0 + u * t
        return (y0 - u / k) * np.exp(-k * t) + u / k

    def residuals(theta):
        k, u, lnKeq, y0s = unpack(theta)
        res = []
        for r in range(R):
            t = ts_list[r]
            y_obs = lnQ_list[r] - lnKeq
            y_fit = model_y(t, k, u, y0s[r])
            res.append(y_obs - y_fit)
        return np.concatenate(res)

    # init
    Keq0 = np.median([np.median(np.exp(lnQ[-max(5, len(lnQ) // 5) :])) for lnQ in lnQ_list])
    lnKeq0 = float(np.log(max(Keq0, 1e-12)))
    k0 = 1.0 / max(1e-6 + max([ts.max() for ts in ts_list]), 1e-3)
    theta0 = np.concatenate([[np.log(k0)], [0.0], [lnKeq0], [lnQ[0] - lnKeq0 for lnQ in lnQ_list]])
    out = least_squares(residuals, theta0, max_nfev=20000)

    k, u, lnKeq, y0s = unpack(out.x)
    Keq = float(np.exp(lnKeq))
    # predictions
    lnQ_pred_list = []
    for r in range(R):
        y_fit = model_y(ts_list[r], k, u, y0s[r])
        lnQ_pred_list.append(lnKeq + y_fit)

    return {
        "success": out.success,
        "message": out.message,
        "cost": float(out.cost),
        "k": float(k),
        "u": float(u),
        "Keq": Keq,
        "y0": y0s,
        "lnQ_pred_list": lnQ_pred_list,
        "lnQ_obs_list": lnQ_list,
        "times": ts_list,
    }


# -----------------------
# Demo on synthetic data
# -----------------------
# 1) simulate two runs (forward and reverse) using CM dynamics
params = cm.sample_params(seed=20250912)

# forward (start low C)
t_eval = np.linspace(0.0, 3.0, 220)
y0_forward = [0.20, 1.25, 2e-4]
t_fwd, A_fwd, B_fwd, C_fwd, v_fwd = cm.simulate((t_eval[0], t_eval[-1]), y0_forward, params, t_eval=t_eval)

# reverse (start near high C)
y0_rev = [0.05, 0.32, 0.50]
t_rev, A_rev, B_rev, C_rev, v_rev = cm.simulate((t_eval[0], t_eval[-1]), y0_rev, params, t_eval=t_eval)

# 2) Fit two exponentials (shared Keq across experiments)
res_multi = fit_llrq_multi_exp(
    ts_list=[t_fwd, t_rev], A_list=[A_fwd, A_rev], B_list=[B_fwd, B_rev], C_list=[C_fwd, C_rev], M=2
)

# 3) If residual structure remains, fit piecewise K(t) on forward run with 3 segments
res_piece = fit_llrq_piecewiseK(t_fwd, A_fwd, B_fwd, C_fwd, J=3, Keq=res_multi["Keq"])

# 4) (Optional) fit with constant drive u across both runs
res_u = fit_llrq_with_offset(
    ts_list=[t_fwd, t_rev],
    A_list=[A_fwd, A_rev],
    B_list=[B_fwd, B_rev],
    C_list=[C_fwd, C_rev],
)

# Save results as JSON for you to inspect
out = {
    "multi_exp": {
        "success": res_multi["success"],
        "k": res_multi["k"].tolist(),
        "w": res_multi["w"].tolist(),
        "Keq": float(res_multi["Keq"]),
        "cost": float(res_multi["cost"]),
        "message": res_multi["message"],
    },
    "piecewise": {
        "Keq": float(res_piece["Keq"]),
        "Ks": res_piece["Ks"].tolist(),
        "changepoints": res_piece["changepoints"],
        "sse": float(res_piece["sse"]),
    },
    "with_offset": {
        "success": res_u["success"],
        "k": float(res_u["k"]),
        "u": float(res_u["u"]),
        "Keq": float(res_u["Keq"]),
        "cost": float(res_u["cost"]),
        "message": res_u["message"],
    },
}

with open("/mnt/data/llrq_fit_results.json", "w") as f:
    json.dump(out, f, indent=2)

# -------------
# Visualization
# -------------
import matplotlib.pyplot as plt

# (1) ln Q fits for both runs with 2-mode multi-exp
plt.figure()
lnQ_fwd = _compute_lnQ(A_fwd, B_fwd, C_fwd)
lnQ_rev = _compute_lnQ(A_rev, B_rev, C_rev)
plt.plot(t_fwd, lnQ_fwd, label="forward lnQ (obs)")
plt.plot(t_fwd, res_multi["lnQ_pred_list"][0], label="forward lnQ (2-mode fit)")
plt.legend()
plt.xlabel("time")
plt.ylabel("ln Q")
plt.title("Forward run: ln Q — 2-mode fit")
plt.tight_layout()
plt.savefig("/mnt/data/forward_lnQ_multimode.png")
plt.close()

plt.figure()
plt.plot(t_rev, lnQ_rev, label="reverse lnQ (obs)")
plt.plot(t_rev, res_multi["lnQ_pred_list"][1], label="reverse lnQ (2-mode fit)")
plt.legend()
plt.xlabel("time")
plt.ylabel("ln Q")
plt.title("Reverse run: ln Q — 2-mode fit")
plt.tight_layout()
plt.savefig("/mnt/data/reverse_lnQ_multimode.png")
plt.close()

# (2) piecewise-K reconstruction on forward run
plt.figure()
plt.plot(t_fwd, res_piece["lnQ_obs"], label="forward lnQ (obs)")
plt.plot(t_fwd, res_piece["lnQ_hat"], label="forward lnQ (piecewise K)")
plt.legend()
plt.xlabel("time")
plt.ylabel("ln Q")
plt.title("Forward run: ln Q — piecewise-constant K(t)")
plt.tight_layout()
plt.savefig("/mnt/data/forward_lnQ_piecewiseK.png")
plt.close()


# (3) Effective rate curve K_eff(t) from 2-mode fit (for interpretation)
def K_eff_curve(t, k, w):
    num = (k * np.exp(-np.outer(t, k))).dot(w)
    den = (np.exp(-np.outer(t, k))).dot(w)
    return num / den


Keff = K_eff_curve(t_fwd, res_multi["k"], res_multi["w"])
plt.figure()
plt.plot(t_fwd, Keff, label="K_eff(t) from 2-mode")
plt.legend()
plt.xlabel("time")
plt.ylabel("K_eff(t)")
plt.title("Implied effective rate from 2-mode fit")
plt.tight_layout()
plt.savefig("/mnt/data/forward_Keff_multimode.png")
plt.close()

# (4) Save the extension functions in a drop-in module for you
ext_code_path = "/mnt/data/llrq_extensions.py"
with open(ext_code_path, "w") as f:
    f.write(
        textwrap.dedent("""
    import numpy as np
    from typing import List, Dict, Any, Optional
    from scipy.optimize import least_squares

    # These rely on your existing cm_rate_law_integrated.py functions:
    # - compute_reaction_quotient
    # - simulate_llrq_concentrations (for back-mapping if desired)

    def _compute_lnQ(cm, A, B, C):
        return np.log(cm.compute_reaction_quotient(A, B, C))

    def fit_llrq_multi_exp(cm,
                           ts_list: List[np.ndarray],
                           A_list: List[np.ndarray],
                           B_list: List[np.ndarray],
                           C_list: List[np.ndarray],
                           M: int = 2) -> Dict[str, Any]:
        def softplus(x):
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        def softmax(b):
            e = np.exp(b - np.max(b))
            return e / e.sum()
        R = len(ts_list)
        lnQ_list = [ _compute_lnQ(cm, A_list[r], B_list[r], C_list[r]) for r in range(R) ]

        def unpack(theta):
            off = 0
            th_k = theta[off:off+M]; off+=M
            b_w  = theta[off:off+M]; off+=M
            lnKeq = theta[off]; off+=1
            y0 = theta[off:off+R]; off+=R
            k = np.empty(M)
            k[0] = softplus(th_k[0])
            for m in range(1,M):
                k[m] = k[m-1] + softplus(th_k[m])
            e = np.exp(b_w - np.max(b_w)); w = e / e.sum()
            Keq = np.exp(lnKeq)
            return k, w, Keq, y0

        def y_model(t, k, w, y0):
            E = np.exp(-np.outer(t, k))
            return y0 * (E @ w)

        def residuals(theta):
            k, w, Keq, y0s = unpack(theta)
            res = []
            for r in range(R):
                y_obs = lnQ_list[r] - np.log(Keq)
                y_fit = y_model(ts_list[r], k, w, y0s[r])
                res.append(y_obs - y_fit)
            return np.concatenate(res)

        t_all = np.concatenate(ts_list)
        tmin, tmax = float(t_all.min()), float(t_all.max())
        span_lo = 1.0 / max(1e-6 + tmax*5.0, 1e-3)
        span_hi = max(5.0 / max(tmin, tmax*1e-3), 0.1)
        ks0 = np.logspace(np.log10(max(span_lo, 1e-3)), np.log10(max(span_hi, 1e-2)), M)
        th_k0 = np.log(np.exp(ks0) - 1.0)
        b_w0 = np.zeros(M)
        Keq0 = np.median([ np.median(np.exp(lnQ[-max(5, len(lnQ)//5):])) for lnQ in lnQ_list ])
        Keq0 = max(Keq0, 1e-12)
        lnKeq0 = np.log(Keq0)
        y00 = np.array([ lnQ_list[r][0] - lnKeq0 for r in range(R) ])
        theta0 = np.concatenate([th_k0, b_w0, [lnKeq0], y00])
        out = least_squares(residuals, theta0, max_nfev=20000)
        k, w, Keq, y0s = unpack(out.x)
        preds = []
        for r in range(R):
            lnQ_pred = np.log(Keq) + y_model(ts_list[r], k, w, y0s[r])
            preds.append(lnQ_pred)
        return {"k": k, "w": w, "Keq": Keq, "y0": y0s,
                "lnQ_pred_list": preds, "lnQ_obs_list": lnQ_list, "times": ts_list,
                "success": out.success, "cost": float(out.cost), "message": out.message}

    def _piecewise_linear_fit(t, z, J):
        n = len(t)
        T1 = np.ones(n).cumsum()
        Tt = t.cumsum(); Tz = z.cumsum()
        Ttt = (t*t).cumsum(); Ttz = (t*z).cumsum()
        def sums(i,j):
            cnt = T1[j-1] - (T1[i-1] if i>0 else 0.0)
            st  = Tt[j-1] - (Tt[i-1] if i>0 else 0.0)
            sz  = Tz[j-1] - (Tz[i-1] if i>0 else 0.0)
            stt = Ttt[j-1] - (Ttt[i-1] if i>0 else 0.0)
            stz = Ttz[j-1] - (Ttz[i-1] if i>0 else 0.0)
            return cnt, st, sz, stt, stz
        def fit_interval(i,j):
            cnt, st, sz, stt, stz = sums(i,j)
            den = cnt*stt - st*st
            if abs(den) < 1e-12:
                b=0.0; a=sz/cnt
            else:
                b = (cnt*stz - st*sz)/den
                a = (sz - b*st)/cnt
            tt=t[i:j]; zz=z[i:j]
            sse = float(((zz - (a + b*tt))**2).sum())
            return a,b,sse
        INF=1e300
        A=np.zeros((n,n)); B=np.zeros((n,n)); SSE=np.zeros((n,n))+INF
        for i in range(n):
            for j in range(i+1, n+1):
                a,b,sse = fit_interval(i,j)
                A[i,j-1]=a; B[i,j-1]=b; SSE[i,j-1]=sse
        dp = np.full((J+1, n+1), INF); prev=[[ -1]*(n+1) for _ in range(J+1)]
        dp[0,0]=0.0
        for jseg in range(1,J+1):
            for j in range(1,n+1):
                best=INF; arg=-1
                for i in range(jseg-1, j):
                    cand = dp[jseg-1,i] + SSE[i,j-1]
                    if cand < best: best=cand; arg=i
                dp[jseg,j]=best; prev[jseg][j]=arg
        cps=[n]; jseg=J; j=n
        while jseg>0:
            i = prev[jseg][j]; cps.append(i); j=i; jseg-=1
        cps=list(reversed(cps))
        segs=[]
        for s in range(J):
            i=cps[s]; j=cps[s+1]
            segs.append({"i":i,"j":j,"a":A[i,j-1],"b":B[i,j-1]})
        return {"changepoints": cps, "segments": segs, "sse": float(dp[J,n])}

    def fit_llrq_piecewiseK(cm, t, A, B, C, J: int = 3, Keq: Optional[float] = None) -> Dict[str, Any]:
        lnQ = _compute_lnQ(cm, A, B, C)
        if Keq is None:
            Keq = float(np.median(np.exp(lnQ[-max(5, len(lnQ)//5):])))
        lnKeq = float(np.log(Keq))
        y = lnQ - lnKeq
        eps = 1e-12; z = np.log(np.maximum(np.abs(y), eps))
        res_seg = _piecewise_linear_fit(t, z, J)
        Ks = [max(1e-12, -float(s["b"])) for s in res_seg["segments"]]
        lnQ_hat = np.zeros_like(lnQ); y_hat=np.zeros_like(y)
        y_at_start = float(y[0])
        for s_idx, s in enumerate(res_seg["segments"]):
            i, j = s["i"], s["j"]
            Kseg = Ks[s_idx]
            t_start = float(t[i])
            if i>0: y_at_start = float(y_hat[i-1])
            for idx in range(i, j):
                dt = float(t[idx] - t_start)
                y_hat[idx] = y_at_start * np.exp(-Kseg*dt)
                lnQ_hat[idx] = lnKeq + y_hat[idx]
        return {"Keq": Keq, "Ks": np.array(Ks), "segments": res_seg["segments"],
                "changepoints": res_seg["changepoints"], "lnQ_obs": lnQ, "lnQ_hat": lnQ_hat,
                "y_obs": y, "y_hat": y_hat, "t": t, "sse": res_seg["sse"]}

    def fit_llrq_with_offset(cm, ts_list, A_list, B_list, C_list):
        R = len(ts_list)
        lnQ_list = [ _compute_lnQ(cm, A_list[r], B_list[r], C_list[r]) for r in range(R) ]
        def unpack(theta):
            k = np.exp(theta[0]); u = theta[1]; lnKeq = theta[2]; y0 = theta[3:3+R]
            return k, u, lnKeq, y0
        def model_y(t, k, u, y0):
            return (y0 - u/k) * np.exp(-k*t) + u/k if k>1e-12 else y0 + u*t
        def residuals(theta):
            k, u, lnKeq, y0s = unpack(theta)
            res = []
            for r in range(R):
                y_obs = lnQ_list[r] - lnKeq
                y_fit = model_y(ts_list[r], k, u, y0s[r])
                res.append(y_obs - y_fit)
            return np.concatenate(res)
        Keq0 = np.median([ np.median(np.exp(lnQ[-max(5, len(lnQ)//5):])) for lnQ in lnQ_list ])
        lnKeq0 = float(np.log(max(Keq0, 1e-12)))
        k0 = 1.0 / max(1e-6 + max([ts.max() for ts in ts_list]), 1e-3)
        theta0 = np.concatenate([[np.log(k0)], [0.0], [lnKeq0], [lnQ[0]-lnKeq0 for lnQ in lnQ_list]])
        out = least_squares(residuals, theta0, max_nfev=20000)
        k,u,lnKeq,y0s = unpack(out.x); Keq = float(np.exp(lnKeq))
        lnQ_pred_list = []
        for r in range(R):
            y_fit = model_y(ts_list[r], k, u, y0s[r])
            lnQ_pred_list.append(lnKeq + y_fit)
        return {"success": out.success, "message": out.message, "cost": float(out.cost),
                "k": float(k), "u": float(u), "Keq": Keq, "y0": y0s,
                "lnQ_pred_list": lnQ_pred_list, "lnQ_obs_list": lnQ_list, "times": ts_list}
    """)
    )

# Final: show a quick textual summary inline
out
