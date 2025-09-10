# README — Mathematical Rationale for CVXPY Bounds on the Relaxation Matrix **K**

This note explains the math behind the Python/CVXPY scaffold that enforces **physically meaningful bounds** on the multi-reaction relaxation matrix
\(
K
\)
used in the log-linear reaction-quotient (affinity) model.

---

## 1) Model recap (log-linear, multi-reaction)

- Let \(c\in\mathbb{R}^m_{>0}\) be species concentrations and \(N\in\mathbb{R}^{m\times r}\) the stoichiometric matrix (products \(+\), reactants \(-\)).
- For reaction \(j\), define the **affinity** (dimensionless free-energy drop) as
  \[
  x_j \;=\; \sum_{s=1}^m \nu_{sj}\,\ln c_s \;-\; \ln K_{\mathrm{eq},j}
  \quad \Longleftrightarrow \quad
  x \;=\; N^\top \ln c \;-\; \ln K_{\mathrm{eq}}.
  \]
- Species dynamics: \(\dot c = N\,v\), with flux vector \(v\in\mathbb{R}^r\).
  Differentiating \(x\) gives
  \[
  \dot x \;=\; N^\top \operatorname{diag}\!\big(1/c\big)\,\dot c
           \;=\; \underbrace{N^\top \operatorname{diag}\!\big(1/c\big) N}_{A(c)}\,v.
  \]
- Near equilibrium, linear irreversible thermodynamics gives a **force–flux** relation
  \[
  v \;\approx\; -\,G\,x \;+\; B\,u,
  \]
  where \(G\in\mathbb{R}^{r\times r}\) is the **reaction conductance** (Onsager) matrix, symmetric positive semidefinite (PSD), and \(u\) collects exogenous thermodynamic drives.

Combining the two, at the chosen equilibrium concentrations \(c_{\rm eq}\),
\[
\dot x \;=\; -\underbrace{A(c_{\rm eq})}_{\displaystyle A}\,G\,x \;+\; A\,B\,u,
\qquad\text{so the relaxation matrix is}\qquad
\boxed{\,K \;=\; A\,G\,}.
\]

Here
\(
A \equiv N^\top \operatorname{diag}(1/c_{\rm eq}) N \succeq 0
\)
is set by **stoichiometry + equilibrium pools**, and \(G\succeq 0\) captures **catalysis/transport physics**.

---

## 2) Spectral structure of \(K\)

Because
\(
A^{-1/2} K A^{1/2} = A^{1/2} G A^{1/2}
\)
is **symmetric PSD**, the non-zero eigenvalues of \(K\) are **real and non-negative** on the subspace of independent affinities. This is the key convex handle we exploit.

- **Similarity:** \(K\) is similar to \(A^{1/2} G A^{1/2}\).
- **Implication:** There exists \(G\succeq 0\) with \(K=AG\) **iff** \(A^{-1/2} K A^{1/2}\succeq 0\) in the range of \(A\).

When \(A\) is strictly PD (or we add a tiny ridge), we can work with \(A^{\pm 1/2}\) safely.

---

## 3) Physical bounds on \(G\) (how fast can relaxation be?)

For a single enzyme step, near equilibrium, the “self-conductance” scales like
\[
G_{ii}\ \sim\ \Big(\frac{k_{\rm cat}}{K_m}\Big)_i\,[E_i]\times(\text{thermodynamic factor}),
\]
and is **capped by diffusion**:
\[
0 \le G_{ii} \le G^{\max}_{ii} \approx k_{\rm on,i}\,[E_i]\times(\text{factor}),
\]
with \(k_{\rm on}\in[10^6,10^9]\ \mathrm{M^{-1}s^{-1}}\) typically. Since \(G\) is PSD, off-diagonals obey \(|G_{ij}|\le \sqrt{G_{ii}G_{jj}}\).

These translate into **per-reaction diagonal caps** and **global spectral caps** on \(G\).

---

## 4) Convex encoding in CVXPY

We do **not** introduce \(G\) explicitly. Instead, define the **similarity variable**
\[
W \;\equiv\; A^{-1/2}\,K\,A^{1/2} \;=\; A^{1/2}GA^{1/2}.
\]

All constraints we need become **linear matrix inequalities (LMIs)** in the variables \(K\) and \(W\):

1. **Existence of \(G\succeq 0\) and compatibility with \(A\):**
   \[
   W = A^{-1/2} K A^{1/2},\qquad W \succeq 0.
   \]
   This is equivalent to “there exists some \(G\succeq 0\) with \(K=AG\)”.

2. **Spectral caps/floors on \(G\):**
   - If \(G \preceq \gamma_{\max} I\), then \(A^{1/2}GA^{1/2} \preceq \gamma_{\max} A\)
     \(\Rightarrow\) **LMI:** \(W \preceq \gamma_{\max} A\).
   - If \(\gamma_{\min} I \preceq G\) (optional), then \(\gamma_{\min} A \preceq W\).

3. **Per-reaction diagonal bounds on \(G\):**
   Since \(G = A^{-1/2} W A^{-1/2}\), the diagonal is **linear in \(W\)**:
   \[
   \operatorname{diag}(A^{-1/2} W A^{-1/2}) \le g^{\max}, \qquad
   \operatorname{diag}(A^{-1/2} W A^{-1/2}) \ge g^{\min}\ (\text{optional}).
   \]
   These encode diffusion-limited **upper caps** \(g^{\max}\) and any known **lower bounds** \(g^{\min}\).

All of the above are convex constraints.

---

## 5) What the helper functions do

- **`make_A(N, c_eq, eps)`**
  Builds \(A = N^\top \operatorname{diag}(1/c_{\rm eq}) N\); symmetrizes and adds a tiny ridge \( \varepsilon I \) for numerical positive-definiteness (enables \(A^{-1/2}\)).

- **`sqrt_and_invsqrt(A)`**
  Eigen-decomposition to compute \(A^{1/2}\) and \(A^{-1/2}\).

- **`K_bounds_constraints(K, N, c_eq, ...)`**
  Creates a symmetric variable \(W\), ties it to \(K\) through
  \(W = A^{-1/2} K A^{1/2}\), enforces \(W\succeq 0\), and adds any of:
  - spectral caps \(W \preceq \gamma_{\max} A\),
  - spectral floors \(\gamma_{\min} A \preceq W\),
  - per-reaction diagonal bounds on \(G\) via
    \(\operatorname{diag}(A^{-1/2}WA^{-1/2}) \le g^{\max}\) and/or \(\ge g^{\min}\).

It returns the list of CVXPY constraints plus the handy \(W\) and \(A\) objects in case you want to regularize them in your objective.

---

## 6) Relationship to bounds on \(K\) itself

From \(K=AG\) and the similarity above, the **eigenvalues** satisfy
\[
\lambda_{\min}(K) \ge \lambda_{\min}(A)\,\lambda_{\min}(G),\qquad
\lambda_{\max}(K) \le \lambda_{\max}(A)\,\lambda_{\max}(G).
\]
Since \(A\) is fixed by \(N\) and \(c_{\rm eq}\), and \(G\) is bounded by your physical priors, these give **global lower/upper bounds** on the spectrum of \(K\).

Entrywise bounds (e.g., Gershgorin-style) can also be derived using
\(|G_{ij}|\le\sqrt{G_{ii}G_{jj}}\) and the entries of \(A\), but the convex program enforces the **stronger PSD/spectral structure** directly.

---

## 7) Practical notes

- **Singular \(A\):** If reactions are not independent, \(A\) is only PSD; adding a tiny \(\varepsilon I\) makes the transform well-posed. Alternatively, work in a reduced independent basis for \(x\).
- **Choosing caps:** Set \(g^{\max}_i \approx k_{\rm on,i}\,[E_i]\times(\text{stoichiometric factor})\). If you only know a global cap, use \(\gamma_{\max}=\max_i g^{\max}_i\).
- **Lower bounds:** If you believe every step has nonzero conductance, supply \(g^{\min}_i>0\) or \(\gamma_{\min}>0\) to ensure strictly positive decay along all independent modes.
- **Sparsity/structure:** You can add masks or \(\ell_1\) penalties on \(K\) (or on \(W\)) in the objective to encourage near-Laplacian structure or weak cross-coupling.

---

## 8) Minimal usage pattern (pseudo-code)

```python
import cvxpy as cp
import numpy as np
from your_module import K_bounds_constraints   # or inline the function

# Given: N (m×r), c_eq (m,), physical caps for G
K = cp.Variable((r, r))              # decision variable
constraints, W, A = K_bounds_constraints(
    K, N, c_eq,
    g_diag_max=g_diag_max,           # per-reaction caps (vector)
    g_diag_min=g_diag_min,           # optional (vector)
    gamma_max=gamma_max,             # optional (scalar)
    gamma_min=gamma_min              # optional (scalar)
)

# Add your objective (e.g., fit data, regularize structure)
obj = cp.Minimize( data_fit_loss(K) + 1e-3*cp.norm1(K - cp.diag(cp.diag(K))) )

cp.Problem(obj, constraints).solve()
