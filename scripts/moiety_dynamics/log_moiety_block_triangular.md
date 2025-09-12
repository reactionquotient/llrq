
# Decoupling Log-Quotient and Moiety Totals for Linear Control (Block‑Triangular Model)

**Short answer.** Yes—under mild, implementable conditions you can make the **reaction‑quotient (log) dynamics** and the **moiety totals** evolve **separately and linearly**, so the overall control model is linear (block‑triangular).

---

## Structure you want

Let
- \(x = \ln Q = S^\top \ln c\)  (reaction‑quotient logs; “ratios”),
- \(y = L^\top c\)  (moiety totals; “levels”), with \(L^\top S = 0\).

Target an LTI form
\[
\dot x = -K\,x + B_x u + w_x(t), \qquad
\dot y = A_y\,y + B_y u + w_y(t),
\]
with **no \(y \to x\)** or **\(x \to y\)** coupling. Then you can design estimation/control for \(x\) (ratios) and \(y\) (totals) **separately**, and run them together.

---

## When this is *exactly* true

**1) Closed system (no in/out flows).**
\(\dot y = 0\) (constants), and \(\dot x = -Kx + B_x u\) (your log‑linear model). Trivially block‑triangular and linear.

**2) CSTR‑type open system with uniform dilution.**
Species balance \(\dot c = S\,r(c) + F_{\text{in}}(t)/V - D(t)\,c\) with one well‑mixed outlet \(D=F_{\text{out}}/V\). Then
\[
\dot y = L^\top \dot c = L^\top \frac{F_{\text{in}}(t)}{V} - D(t)\,y,
\]
which is **linear in \(y\)** and **independent of \(x\)** (reactions drop out since \(L^\top S = 0\)). In parallel, the log‑dynamics for \(x\) stay linear/LTI near equilibrium. Result: **exact decoupling** and linearity for both blocks.

**3) Any outlet/side‑process that is “moiety‑respecting.”**
If the net removal operator \(R\) satisfies \(L^\top R = A_y L^\top\) for some constant matrix \(A_y\) (intuitively: it removes entire moieties proportionally), then
\[
\dot y = L^\top F_{\text{in}}(t) - A_y\,y
\]
is again linear and decoupled.

---

## When it’s only approximately true (still very usable)

- **Selective removal (membranes, precipitation)** that isn’t moiety‑respecting: \(L^\top R c\) may depend on how mass is split **within** a moiety (hence on \(x\)). Near an operating point, you can still linearize to get \(\dot y \approx A_y\,y + B_y u\) with small, bounded \(x \to y\) couplings—often negligible if selectivity differences inside a moiety are mild or if you regulate \(x\) tightly.

- **Activity‑coefficient/ionic‑strength effects**: if they vary with totals, roll them into slow scheduling of \(K\). You still have a **linear, block‑triangular, gain‑scheduled** model.

---

## Control takeaway

For the combined state \(z = [x^\top\; y^\top]^\top\) you obtain a **linear, block‑triangular** system in the exact cases above. You can:
- run an **LQR/MPC on \(x\)** to regulate affinities/ratios, independent of totals;
- run a **PI/LQR on \(y\)** (each moiety follows a first‑order linear ODE driven by feeds/outflow);
- superpose inputs if actuators affect both (the optimizer just sees one linear plant).

Even with mild departures (non‑uniform removal), the small residual couplings keep the problem in well‑behaved **LTV/LTI‑scheduled** territory.

---

## Mini example (A \( \leftrightharpoons \) B in a CSTR)

Let \(x = \ln([B]/[A])\), \(y = [A]+[B]\). Near equilibrium
\[
\dot x = -k\,x + b\,u \quad \text{(chemostats/energy drives)},
\]
and with one outlet at dilution \(D\) and inlet total \(y_{\text{in}}(t)\)
\[
\dot y = -D\,y + D\,y_{\text{in}}(t).
\]
This is **fully linear and decoupled**; design controllers for \(x\) and \(y\) separately.

---

## Tiny Python demo

Below we simulate the decoupled system with a step in \(u(t)\) (for \(x\)) and a step in \(y_{\text{in}}(t)\) (for \(y\)).
