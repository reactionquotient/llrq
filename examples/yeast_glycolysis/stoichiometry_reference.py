import numpy as np
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

# Reactions (columns) in the same order as x1..x8
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

# Metabolites (rows) used in the Q definitions (water omitted for log-quotients)
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

# r6: 2-PG -> PEP (+H2O)  [Water omitted in log-quotient mapping]
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
display_dataframe_to_user("Stoichiometry matrix S (metabolites × reactions) for log-quotients", S_df)
S_df.to_csv("/mnt/data/yeast_S_logQ.csv")

# Optional mass-balance version that includes water produced by ENO
met_names_mass = met_names + ["H2O"]
S_mass = np.zeros((len(met_names_mass), n), dtype=float)
S_mass[: S.shape[0], :] = S
S_mass[met_names_mass.index("H2O"), 5] = +1  # ENO produces water
S_mass_df = pd.DataFrame(S_mass, index=met_names_mass, columns=rxn_names)
display_dataframe_to_user("Stoichiometry matrix S_mass (includes H2O for ENO)", S_mass_df)
S_mass_df.to_csv("/mnt/data/yeast_S_mass.csv")


# Provide a helper to compute x from concentrations and Keq:  x = S^T ln c - ln Keq
def x_from_conc(conc: dict, Keq: dict):
    ln_c = np.array([np.log(conc.get(m, 1.0)) for m in met_names], dtype=float)
    ln_Keq = np.array([np.log(Keq.get(r, 1.0)) for r in rxn_names], dtype=float)
    x = S.T @ ln_c - ln_Keq
    return pd.Series(x, index=[f"x{i+1}" for i in range(n)])


# Quick numeric smoke test (placeholders)
conc_demo = {m: 1.0 for m in met_names}
conc_demo.update({"ATP": 2.0, "ADP": 0.5, "NAD": 1.2, "NADH": 0.8})
Keq_demo = {r: 1.0 for r in rxn_names}
x_from_conc(conc_demo, Keq_demo)
