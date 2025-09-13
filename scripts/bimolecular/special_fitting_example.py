# import numpy as np
# import importlib.util

# # load your module that computes Q
# spec = importlib.util.spec_from_file_location("cm", "/path/to/cm_rate_law_integrated.py")
# cm = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(cm)

# # load the extensions
# spec2 = importlib.util.spec_from_file_location("llrqx", "/path/to/llrq_extensions.py")
# llrqx = importlib.util.module_from_spec(spec2)
# spec2.loader.exec_module(llrqx)

# # Suppose you have R experiments with arrays: t[r], A[r], B[r], C[r]
# ts_list = [t1, t2]  # list of 1D time arrays
# A_list = [A1, A2]
# B_list = [B1, B2]
# C_list = [C1, C2]

# # 1) Two-mode fit on lnQ with shared Keq
# res2 = llrqx.fit_llrq_multi_exp(cm, ts_list, A_list, B_list, C_list, M=2)
# print("Keq:", res2["Keq"], "k:", res2["k"], "w:", res2["w"])

# # 2) If residuals have structure, try piecewise K(t) on a run (e.g., run 0)
# res_pw = llrqx.fit_llrq_piecewiseK(cm, ts_list[0], A_list[0], B_list[0], C_list[0], J=3, Keq=res2["Keq"])
# print("Segments K:", res_pw["Ks"], "changepoints (indices):", res_pw["changepoints"])

# # 3) If you suspect a steady offset, try single-mode + u across runs
# res_u = llrqx.fit_llrq_with_offset(cm, ts_list, A_list, B_list, C_list)
# print("k:", res_u["k"], "u:", res_u["u"], "Keq:", res_u["Keq"])
