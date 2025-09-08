# Rebuild K and S (so this cell is self-contained), then derive Onsager L

import numpy as np
import pandas as pd
from dataclasses import dataclass
from caas_jupyter_tools import display_dataframe_to_user

# ----- Define state order and reaction names (x1..x8) -----
state_names = [
    "x1: PGI  (G6P ↔ F6P)",
    "x2: TPI  (DHAP ↔ GAP)",
    "x3: GAPDH (GAP+Pi+NAD ↔ 1,3-BPG+NADH)",
    "x4: PGK  (1,3-BPG+ADP ↔ 3-PG+ATP)",
    "x5: PGM  (3-PG ↔ 2-PG)",
    "x6: ENO  (2-PG ↔ PEP)",
    "x7: ADH  (AcAld+NADH ↔ EtOH+NAD)",
    "x8: GPD  (DHAP+NADH ↔ G3P+NAD)",
]
rxn_names = [
    "r1: PGI (G6P↔F6P)",
    "r2: TPI (DHAP↔GAP)",
    "r3: GAPDH (GAP+Pi+NAD↔1,3-BPG+NADH)",
    "r4: PGK (1,3-BPG+ADP↔3-PG+ATP)",
    "r5: PGM (3-PG↔2-PG)",
    "r6: ENO (2-PG↔PEP)",
    "r7: ADH (AcAld+NADH↔EtOH+NAD)",
    "r8: GPD (DHAP+NADH↔G3P+NAD)",
]

# ----- Build S (metabolites × reactions) consistent with x definitions -----
met_names = [
    "G6P",
    "F6P",
    "DHAP",
    "GAP",
    "1,3-BPG",
    "3-PG",
    "2-PG",
    "PEP",
    "Pi",
    "NAD",
    "NADH",
    "ADP",
    "ATP",
    "AcAld",
    "EtOH",
    "G3P",
]
m, n = len(met_names), len(rxn_names)
S = np.zeros((m, n), dtype=float)

# r1: G6P -> F6P
S[met_names.index("G6P"), 0] = -1
S[met_names.index("F6P"), 0] = +1
# r2: DHAP -> GAP
S[met_names.index("DHAP"), 1] = -1
S[met_names.index("GAP"), 1] = +1
# r3: GAP + Pi + NAD -> 1,3-BPG + NADH
S[met_names.index("GAP"), 2] = -1
S[met_names.index("Pi"), 2] = -1
S[met_names.index("NAD"), 2] = -1
S[met_names.index("1,3-BPG"), 2] = +1
S[met_names.index("NADH"), 2] = +1
# r4: 1,3-BPG + ADP -> 3-PG + ATP
S[met_names.index("1,3-BPG"), 3] = -1
S[met_names.index("ADP"), 3] = -1
S[met_names.index("3-PG"), 3] = +1
S[met_names.index("ATP"), 3] = +1
# r5: 3-PG -> 2-PG
S[met_names.index("3-PG"), 4] = -1
S[met_names.index("2-PG"), 4] = +1
# r6: 2-PG -> PEP (+H2O omitted in x)
S[met_names.index("2-PG"), 5] = -1
S[met_names.index("PEP"), 5] = +1
# r7: AcAld + NADH -> EtOH + NAD
S[met_names.index("AcAld"), 6] = -1
S[met_names.index("NADH"), 6] = -1
S[met_names.index("EtOH"), 6] = +1
S[met_names.index("NAD"), 6] = +1
# r8: DHAP + NADH -> G3P + NAD
S[met_names.index("DHAP"), 7] = -1
S[met_names.index("NADH"), 7] = -1
S[met_names.index("G3P"), 7] = +1
S[met_names.index("NAD"), 7] = +1

S_df = pd.DataFrame(S, index=met_names, columns=rxn_names)
S_df.to_csv("/mnt/data/yeast_S_logQ.csv")

# ----- Rebuild K (same construction as earlier example) -----
alpha = 0.05


def laplacian_from_edges(n, edges):
    L = np.zeros((n, n))
    for i, j, g in edges:
        L[i, i] += g
        L[j, j] += g
        L[i, j] -= g
        L[j, i] -= g
    return L


edges = [
    (0, 1, 0.8),
    (1, 2, 0.6),
    (2, 3, 0.6),
    (3, 4, 0.4),
    (4, 5, 0.4),
    (2, 6, 0.30),
    (2, 7, 0.30),
    (6, 7, 0.20),
]
L_sym = laplacian_from_edges(8, edges)

J = np.zeros((8, 8))


def add_rot(J, i, j, omega):
    J[i, j] += omega
    J[j, i] -= omega


add_rot(J, 2, 3, 0.50)
add_rot(J, 6, 7, 0.40)
add_rot(J, 1, 2, 0.20)
add_rot(J, 5, 6, 0.10)

u = np.zeros((8, 1))
u[[2, 3, 6, 7], 0] = 1.0
u = u / np.linalg.norm(u)
gamma = 0.10
UUT = gamma * (u @ u.T)

K = alpha * np.eye(8) + L_sym + J + UUT

K_df = pd.DataFrame(K, index=state_names, columns=state_names)
K_df.to_csv("/mnt/data/yeast_K_matrix.csv")

# ----- Derive Onsager L from K and S with reference concentrations c* -----
# Default c* = 1 for all metabolites -> C = I
c_ref = np.ones(len(met_names))
Cinv = np.diag(1.0 / c_ref)
G = S.T @ Cinv @ S

# Pseudoinverse in case G is singular (it shouldn't be here)
G_pinv = np.linalg.pinv(G)

L_min = G_pinv @ K
L_sym = 0.5 * (L_min + L_min.T)
eigvals, eigvecs = np.linalg.eigh(L_sym)
eigvals_clip = np.clip(eigvals, 0.0, None)
L_psd = (eigvecs * eigvals_clip) @ eigvecs.T

# Recon error
from numpy.linalg import norm

rel_err_min = norm(K - G @ L_min, "fro") / max(1.0, norm(K, "fro"))
rel_err_psd = norm(K - G @ L_psd, "fro") / max(1.0, norm(K, "fro"))

# Display & save
diag = pd.Series(
    {
        "rank(G)": int(np.linalg.matrix_rank(G)),
        "relative_fit_err_min": rel_err_min,
        "relative_fit_err_psd": rel_err_psd,
        "min_eig_Lsym_before_clip": float(eigvals.min()),
    }
)
display_dataframe_to_user("L-derivation diagnostics", diag.to_frame("value"))

L_min_df = pd.DataFrame(L_min, index=state_names, columns=state_names)
L_psd_df = pd.DataFrame(L_psd, index=state_names, columns=state_names)
display_dataframe_to_user("L_min (G^+ K)", L_min_df)
display_dataframe_to_user("L_psd (Onsager-symmetric PSD)", L_psd_df)

L_min_df.to_csv("/mnt/data/yeast_L_min.csv")
L_psd_df.to_csv("/mnt/data/yeast_L_psd.csv")
pd.DataFrame(G @ L_min, index=state_names, columns=state_names).to_csv("/mnt/data/yeast_K_recon_from_Lmin.csv")
pd.DataFrame(G @ L_psd, index=state_names, columns=state_names).to_csv("/mnt/data/yeast_K_recon_from_Lpsd.csv")

"/mnt/data/yeast_L_min.csv and /mnt/data/yeast_L_psd.csv written (K and S rebuilt here for reproducibility)."
