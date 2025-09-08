
Yeast log-linear model (states x = ln(Q/Keq))
States (8):
  1. x1: PGI  (G6P ↔ F6P)
  2. x2: TPI  (DHAP ↔ GAP)
  3. x3: GAPDH (GAP+Pi+NAD ↔ 1,3-BPG+NADH)
  4. x4: PGK  (1,3-BPG+ADP ↔ 3-PG+ATP)
  5. x5: PGM  (3-PG ↔ 2-PG)
  6. x6: ENO  (2-PG ↔ PEP)
  7. x7: ADH  (AcAld+NADH ↔ EtOH+NAD)
  8. x8: GPD  (DHAP+NADH ↔ G3P+NAD)

Q definitions:
  PGI: Q = [F6P] / [G6P]
  TPI: Q = [GAP] / [DHAP]
  GAPDH: Q = ([1,3-BPG][NADH]) / ([GAP][Pi][NAD])
  PGK: Q = ([3-PG][ATP]) / ([1,3-BPG][ADP])
  PGM: Q = [2-PG] / [3-PG]
  ENO: Q = [PEP] / [2-PG]   (H2O constant)
  ADH: Q = ([EtOH][NAD]) / ([AcAld][NADH])
  GPD: Q = ([G3P][NAD]) / ([DHAP][NADH])

Controls (5):
  1. u_glc_in
  2. u_pyk_pull
  3. u_pdc_push
  4. u_ATP_boost
  5. u_ADP_boost

Dynamics:  dx/dt = -K x + B u(t)

Files:
- K matrix CSV: /mnt/data/yeast_K_matrix.csv
- B matrix CSV: /mnt/data/yeast_B_matrix.csv
