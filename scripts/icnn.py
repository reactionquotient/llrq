"""
ICNN vector-field learning for reaction-quotient log-space dynamics
===================================================================

Goal
----
We want a learned dynamics law in x-space (x = log(Q / Keq)):
    ẋ = f(x) = - M(x) ∇Φ(x)

Thermo safety:
- Φ(x) convex (ICNN)  => ∇Φ is monotone
- M(x) ⪰ 0 (PSD)      => xᵀ f(x) = - xᵀ M(x) ∇Φ(x) ≤ 0    (dissipation)

This script fits Φ (and optionally M) from (x, xdot) samples by minimizing:
    L = mean || fθ,ψ(x_i) - xdot_i ||^2   +   λ_pass * mean ReLU(x_iᵀ fθ,ψ(x_i))
with mild weight decay. It also reports passivity violations and test MSE.

Data
----
CSV with columns:
    x1..xr, xdot1..xdotr   (r = state dim)
You can use the demo file generated earlier: /mnt/data/icnn_vectorfield_dataset.csv

Usage
-----
$ pip install torch pandas numpy
$ python train_icnn_vectorfield.py --csv icnn_vectorfield_dataset.csv --epochs 8000 --learn_mobility 0

Notes
-----
- Start with identity mobility (learn_mobility=0). Only turn on learned mobility
  if field-matching stalls (it increases capacity).
- For real data, estimate xdot by smoothing x(t) then finite differences.
"""

import argparse
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# Utilities
# -------------------------------
def softplus_pos(x):  # strictly positive
    return torch.nn.functional.softplus(x) + 1e-6


def standardize(train, test):
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True).clamp_min(1e-6)
    return (train - mu) / sd, (test - mu) / sd, (mu, sd)


def batched_outer(B):  # (N,d) -> (N,d,d), outer products per row
    return B.unsqueeze(-1) @ B.unsqueeze(-2)


# -------------------------------
# ICNN potential Φ(x)
# -------------------------------
class ICNN(nn.Module):
    """
    Input-Convex Neural Network representing Φ(x), convex in x.
    Convexity trick:
      - activations: convex & nondecreasing (softplus)
      - 'z' branch weights constrained to be elementwise nonnegative
      - last layer forms a convex combination + quadratic term
    """

    def __init__(self, dim, widths=(64, 64)):
        super().__init__()
        self.dim = dim
        self.widths = widths

        layers = []
        in_dim = dim
        for w in widths:
            layers.append(nn.Linear(in_dim, w, bias=True))  # Wx x + b
            # Wz z term added in forward with nonnegativity constraint
            layers.append(nn.Linear(w, w, bias=False))  # Wz
            in_dim = w
        self.layers = nn.ModuleList(layers)

        # Output: Φ(x) = 0.5||Px||^2 + cᵀ z_L + dᵀ x + b0  with c ≥ 0
        self.P = nn.Parameter(torch.randn(dim, dim) * 0.05)
        self.c_raw = nn.Parameter(torch.randn(widths[-1]) * 0.05)
        self.d = nn.Parameter(torch.zeros(dim))
        self.b0 = nn.Parameter(torch.tensor(0.0))

        # simple initialization for monotone z-path
        self._zero_Wz_bias()

    def _zero_Wz_bias(self):
        # init Wz to small nonnegatives, biases to 0
        for i in range(0, len(self.layers), 2):
            Wz = self.layers[i + 1]
            nn.init.uniform_(Wz.weight, a=0.0, b=0.01)

    def forward(self, x):
        # compute z via alternating (Wx x + b) and (Wz z), enforcing Wz >= 0
        act = torch.nn.functional.softplus
        z = torch.zeros(x.shape[0], self.widths[0], device=x.device)
        for i in range(0, len(self.layers), 2):
            Wx = self.layers[i]
            Wz = self.layers[i + 1]
            if i == 0:
                z = act(Wx(x))  # first layer: just x-branch
            else:
                z = act(Wx(x) + softplus_pos(Wz.weight) @ z.T).T  # z-branch nonneg
        # convex readout
        c = softplus_pos(self.c_raw)  # c >= 0
        quad = 0.5 * (x @ self.P.T).pow(2).sum(dim=1)  # 0.5||Px||^2
        phi = quad + (z * c).sum(dim=1) + (self.d * x).sum(dim=1) + self.b0
        return phi

    def grad_phi(self, x):
        x = x.requires_grad_(True)
        phi = self.forward(x).sum()
        (g,) = torch.autograd.grad(phi, x, create_graph=True)
        return g


# -------------------------------
# Positive semidefinite mobility M(x) = B(x) B(x)ᵀ
# -------------------------------
class Mobility(nn.Module):
    """
    Simple mobility with small MLP outputting B(x) ∈ R^{dim×r},
    then M(x) = B Bᵀ  (PSD). Set rank r small (e.g., r=dim for full).
    If learn_mobility=False, this module returns Identity.
    """

    def __init__(self, dim, rank=None, hidden=32, learn_mobility=False):
        super().__init__()
        self.dim = dim
        self.rank = rank or dim
        self.learn = learn_mobility
        if self.learn:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, dim * self.rank)
            )

    def forward(self, x):
        if not self.learn:
            # Identity mobility
            N, d = x.shape
            return torch.eye(d, device=x.device).expand(N, d, d)
        N = x.shape[0]
        B = self.net(x).view(N, self.dim, self.rank)
        return batched_outer(B)  # (N, d, d)


# -------------------------------
# Learned vector field f(x) = - M(x) ∇Φ(x)
# -------------------------------
@dataclass
class Model:
    phi: ICNN
    M: Mobility

    def f(self, x):
        g = self.phi.grad_phi(x)  # ∇Φ
        Mx = self.M(x)  # (N,d,d)
        v = -(Mx @ g.unsqueeze(-1)).squeeze(-1)  # -M∇Φ
        return v


# -------------------------------
# Training / evaluation
# -------------------------------
def train(model, x_train, v_train, x_val, v_val, cfg):
    phi_params = list(model.phi.parameters())
    mob_params = list(model.M.parameters())
    params = phi_params + mob_params
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.wd)

    def batches(X, V, bs):
        idx = torch.randperm(X.shape[0], device=X.device)
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            yield X[j], V[j]

    best = {"val_mse": float("inf"), "state": None}
    for step in range(cfg.epochs):
        model.phi.train()
        model.M.train()
        for xb, vb in batches(x_train, v_train, cfg.batch):
            fb = model.f(xb)
            loss_field = (fb - vb).pow(2).mean()
            # passive penalty: ReLU(xᵀ f(x))
            loss_pass = torch.relu((xb * fb).sum(dim=1)).mean()
            loss = loss_field + cfg.lambda_pass * loss_pass
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.epochs - 1:
            model.phi.eval()
            model.M.eval()
            with torch.no_grad():
                f_tr = model.f(x_train)
                f_v = model.f(x_val)
                mse_tr = (f_tr - v_train).pow(2).mean().sqrt().item()
                mse_v = (f_v - v_val).pow(2).mean().sqrt().item()
                # passivity violation rate on val
                viol = torch.gt((x_val * f_v).sum(dim=1), 1e-7).float().mean().item()
            print(f"[{step+1:05d}] RMSE train={mse_tr:.4f} val={mse_v:.4f} | pass-viol={100*viol:.2f}%")
            if mse_v < best["val_mse"]:
                best["val_mse"] = mse_v
                best["state"] = {k: v.cpu().clone() for k, v in model.phi.state_dict().items()}
    # restore best Φ (mobility is small, leave as is)
    if best["state"] is not None:
        model.phi.load_state_dict(best["state"])
    return model


def rollout(model, x0, dt=0.05, steps=200):
    """Simple fixed-step RK4 rollout of ẋ=f(x)."""
    xs = [x0]
    x = x0
    for _ in range(steps):
        k1 = model.f(x)
        k2 = model.f(x + 0.5 * dt * k1)
        k3 = model.f(x + 0.5 * dt * k2)
        k4 = model.f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return torch.stack(xs, dim=0)


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV with columns x1..xr, xdot1..xdotr")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--lambda_pass", type=float, default=0.05)
    ap.add_argument("--widths", type=str, default="64,64")
    ap.add_argument("--learn_mobility", type=int, default=0, help="0: identity; 1: learn M(x)")
    ap.add_argument("--mob_rank", type=int, default=0, help="rank for B(x); 0->dim")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--rollout", action="store_true", help="run a short rollout sanity check")
    cfg = ap.parse_args()

    df = pd.read_csv(cfg.csv)
    x_cols = [c for c in df.columns if c.startswith("x") and not c.startswith("xdot")]
    v_cols = [c for c in df.columns if c.startswith("xdot")]
    X = torch.tensor(df[x_cols].values, device=DEVICE).float()
    V = torch.tensor(df[v_cols].values, device=DEVICE).float()
    N, dim = X.shape
    print(f"Loaded {N} samples with dim={dim}")

    # train/val split (random; for time series use time-blocked)
    n_val = int(math.ceil(cfg.val_frac * N))
    perm = torch.randperm(N, device=DEVICE)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Xval = X[tr_idx], X[val_idx]
    Vtr, Vval = V[tr_idx], V[val_idx]

    # standardize (helps conditioning)
    Xtr_s, Xval_s, (mu, sd) = standardize(Xtr, Xval)
    Vtr_s, Vval_s, _ = standardize(Vtr, Vval)

    widths = tuple(int(w) for w in cfg.widths.split(",") if w)
    phi = ICNN(dim=dim, widths=widths).to(DEVICE)
    M = Mobility(dim=dim, rank=(cfg.mob_rank if cfg.mob_rank > 0 else dim), learn_mobility=bool(cfg.learn_mobility)).to(DEVICE)
    model = Model(phi=phi, M=M)

    # pack a tiny config object for logging
    class C:
        pass

    C.epochs = cfg.epochs
    C.batch = cfg.batch
    C.lr = cfg.lr
    C.wd = cfg.wd
    C.lambda_pass = cfg.lambda_pass
    C.eval_every = cfg.eval_every
    model = train(model, Xtr_s, Vtr_s, Xval_s, Vval_s, C)

    # final metrics & passivity on val
    with torch.no_grad():
        f_val = model.f(Xval_s)
        rmse = (f_val - Vval_s).pow(2).mean().sqrt().item()
        pass_viol = torch.gt((Xval_s * f_val).sum(dim=1), 1e-7).float().mean().item()
    print(f"\nFinal VAL RMSE={rmse:.4f} | passivity violations={100*pass_viol:.2f}%")

    # optional rollout sanity check (uses standardized coords)
    if cfg.rollout:
        x0 = Xval_s[0:1].clone()
        traj = rollout(model, x0, dt=0.05, steps=200)
        # Report monotone energy decay Φ(traj[k]) (should be nonincreasing)
        with torch.no_grad():
            phi_vals = model.phi(traj).cpu().numpy()
        print("Rollout Φ values (first 10):", np.round(phi_vals[:10], 3))
        # If you want to visualize, save traj to CSV and plot externally
        out = np.concatenate([traj.cpu().numpy(), phi_vals[:, None]], axis=1)
        np.savetxt(
            "rollout_traj_standardized.csv",
            out,
            delimiter=",",
            header=",".join([f"x{i+1}" for i in range(dim)]) + ",Phi",
            comments="",
        )
        print("Saved rollout to rollout_traj_standardized.csv")


if __name__ == "__main__":
    main()
