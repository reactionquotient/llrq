# Frequency‑Domain Control → Entropy Production (LLRQ quick note)

This note translates the linear log–linear reaction quotient model (LLRQ)
\[
\dot x(t) = -K\,x(t) + u(t),\qquad K \succ 0
\]
into the **frequency domain** and shows how designing \(u\) in frequency space maps
into **entropy production**. We keep the near‑equilibrium convention that the
thermodynamic **force** is \(x=\ln(Q/K_{\rm eq})\) and the **entropy production rate**
is the quadratic
\[
\sigma(t)=x(t)^\top L\,x(t),\qquad L\succeq 0.
\]

---

## 1) Frequency response and the “entropy kernel”

Let \( \mathcal{F}\{\,\cdot\,\} \) denote the Fourier transform. The transfer from
control \(u\) to state \(x\) is
\[
G(i\omega) \;=\; (K + i\omega I)^{-1}, \qquad
X(\omega) \;=\; G(i\omega)\,U(\omega).
\]

By Parseval/Plancherel,
\[
\int_{-\infty}^{\infty} \sigma(t)\,dt
= \frac{1}{2\pi}\int_{-\infty}^{\infty} X(\omega)^{\!H}\,L\,X(\omega)\,d\omega
= \frac{1}{2\pi}\int_{-\infty}^{\infty} U(\omega)^{\!H}\,
\underbrace{G(i\omega)^{\!H} L\,G(i\omega)}_{H_u(\omega)\ \succeq 0}\,U(\omega)\,d\omega.
\]

- \(H_u(\omega)=G(i\omega)^{\!H} L\,G(i\omega)\) is a **frequency‑dependent, PSD weight** on the
  control spectrum.
- Designing \(u\) in frequency space means the **total entropy** is a quadratic spectral
  form with kernel \(H_u(\omega)\).

### Single sinusoid
If \(u(t)=\Re\{\hat u\,e^{i\omega_0 t}\}\), then \(x(t)=\Re\{\hat x\,e^{i\omega_0 t}\}\) with
\(\hat x=G(i\omega_0)\hat u\). The **time‑averaged** entropy rate is
\[
\overline{\sigma}=\tfrac12 \hat u^{H} H_u(\omega_0) \hat u
=\tfrac12 \hat x^{H} L \hat x.
\]

---

## 2) Targeting \(x\) (state spectrum) vs targeting \(u\) (control spectrum)

If you **prescribe the state spectrum** \(X_{\rm ref}(\omega)\), then the required control is
\[
U(\omega) = (K + i\omega I)\,X_{\rm ref}(\omega),
\]
and the entropy is simply
\[
\frac{1}{2\pi}\int X_{\rm ref}(\omega)^{\!H} L\,X_{\rm ref}(\omega)\,d\omega,
\]
i.e., **independent of \(K\)** once \(X_{\rm ref}\) is fixed (though \(U\) depends on \(K\)).

If instead you **shape \(u\)** directly, use the kernel \(H_u(\omega)=G^{H}LG\) to evaluate the
resulting entropy.

---

## 3) Frequency‑by‑frequency optimal control (closed form)

For weights \(W\succeq0\) (on \(x\)) and \(R\succ0\) (on \(u\)), consider
\[
J=\frac{1}{2\pi}\!\int \Big[(G U - X_{\rm ref})^{H}W(G U - X_{\rm ref})
 + \lambda\,U^{H}R\,U\Big]\,d\omega.
\]
This **separates pointwise** in \(\omega\). The optimizer is
\[
\boxed{~U^\star(\omega)=\big(G^{H}W G+\lambda R\big)^{-1}G^{H}W\,X_{\rm ref}(\omega)~,}
\]
and the realized \(X^\star=G U^\star\). The associated entropy is
\[
\Sigma = \frac{1}{2\pi}\!\int X^\star(\omega)^{\!H} L\,X^\star(\omega)\,d\omega
= \frac{1}{2\pi}\!\int U^\star(\omega)^{\!H} H_u(\omega)\,U^\star(\omega)\,d\omega.
\]

**Intuition.** Since \(G(i\omega)\sim (i\omega)^{-1}I\) at high \(|\omega|\),
\(H_u(\omega)\sim \omega^{-2} L\). For a *fixed* control spectrum \(U\), high‑frequency content
generates less entropy than low‑frequency content; but to realize a *fixed* state amplitude
at high \(\omega\), the required control grows like \(|\omega|\), so there is no free lunch.

---

## 4) Stochastic inputs

If \(u\) is zero‑mean wide‑sense stationary with spectral density \(S_u(\omega)\),
\[
\mathbb{E}\,\sigma
= \frac{1}{2\pi}\int \mathrm{tr}\!\big(S_u(\omega)\,H_u(\omega)\big)\,d\omega.
\]

---

## 5) Discrete‑time implementation (FFT recipe)

Suppose you sample \(u(t)\) and/or \(x(t)\) at step \(\Delta t\) with \(N\) samples. Let
\(\omega_k = 2\pi\,k/(N\Delta t)\) for \(k=0,\dots,N-1\). Using NumPy’s FFT convention
(\(X_k=\sum_n x_n e^{-2\pi i kn/N}\), inverse \(x_n=\frac1N\sum_k X_k e^{2\pi i kn/N}\)),
the continuous integrals are approximated by
\[
\boxed{~\int x^\top L x\,dt \;\approx\; \frac{1}{N\,\Delta t}\sum_{k=0}^{N-1} X_k^{H} L\,X_k~,}
\]
\[
\boxed{~\int u^{H} H_u(\omega) u\,\frac{d\omega}{2\pi}
\;\approx\; \frac{1}{N\,\Delta t}\sum_{k=0}^{N-1} U_k^{H}\,H_u(\omega_k)\,U_k~,}
\]
with \(G_k=(K+i\omega_k I)^{-1}\) and \(H_u(\omega_k)=G_k^{H} L G_k\). This “full‑spectrum”
sum already accounts for both positive/negative frequencies.

> **Tip.** If you prefer real‑signal `rfft`, adjust the DC/Nyquist terms by 1× and
> interior bins by 2× to recover the two‑sided sum.

---

## 6) Minimal NumPy helpers

```python
import numpy as np

def _freqs(N, dt):
    """Angular frequencies matching np.fft.fft bins (two-sided)."""
    return 2*np.pi*np.fft.fftfreq(N, d=dt)  # shape (N,)

def entropy_from_x_freq(x_t, dt, L):
    """
    Approximate Σ_x = ∫ x(t)^T L x(t) dt using FFT (two-sided sum).
    x_t: (N, m), dt: float, L: (m, m)
    Returns (Sigma, per_bin_contrib) where per_bin_contrib has shape (N,).
    """
    N, m = x_t.shape
    X = np.fft.fft(x_t, axis=0)                 # (N, m)
    Ls = 0.5*(L + L.T)
    # per-bin quadratic: X_k^H L X_k
    quad = np.einsum('ki,ij,kj->k', np.conj(X), Ls, X, optimize=True)
    Sigma = (quad.real).sum() / (N*dt)
    return float(Sigma), quad.real / (N*dt)

def entropy_from_u_freq(u_t, dt, K, L):
    """
    Approximate Σ_u = (1/2π) ∫ U(ω)^H G(ω)^H L G(ω) U(ω) dω via FFT sum.
    u_t: (N, m); K,L: (m,m). Returns (Sigma, per_bin_contrib).
    """
    N, m = u_t.shape
    U = np.fft.fft(u_t, axis=0)                 # (N, m)
    w = _freqs(N, dt)                           # (N,)
    Sigma_bins = np.empty(N, dtype=float)
    for k, wk in enumerate(w):
        G = np.linalg.inv(K + 1j*wk*np.eye(m))      # (m,m)
        Hu = np.conj(G.T) @ L @ G                   # G^H L G
        Sigma_bins[k] = np.real(np.conj(U[k]).T @ Hu @ U[k])
    Sigma = Sigma_bins.sum() / (N*dt)
    return float(Sigma), Sigma_bins / (N*dt)

def map_xref_to_u(Xref, dt, K):
    """
    Given desired state spectrum Xref[k] (FFT grid), return the required control spectrum:
      U[k] = (K + i ω_k I) Xref[k].
    Xref: (N, m) complex array on FFT bins; returns U of same shape.
    """
    N, m = Xref.shape
    w = _freqs(N, dt)
    U = np.empty_like(Xref)
    for k, wk in enumerate(w):
        U[k] = (K + 1j*wk*np.eye(m)) @ Xref[k]
    return U
```

### Usage sketch
```python
# Given samples x_t, u_t with shape (N, m), sample time dt, and matrices K, L:
Sigma_x, _ = entropy_from_x_freq(x_t, dt, L)
Sigma_u, _ = entropy_from_u_freq(u_t, dt, K, L)

print("Entropy (time-integrated) from x:", Sigma_x)
print("Entropy (via control + kernel)  :", Sigma_u, "(should match x if model holds)")
```

---

## 7) Relation to the time‑domain identities

- **Thermodynamic accounting.** Time‑domain entropy: \(\Sigma_x=\int x^\top L x\,dt\).
  Frequency‑domain: same scalar via the quadratic spectral form.
- **Model‑space power identity.** For \(\dot x=-Kx+u\),
  \[
  \frac{d}{dt}\Big(\tfrac12\|x\|_2^2\Big) \;=\; -\,x^\top K_{\rm sym} x \;+\; x^\top u,
  \]
  which averages (over a period) to \(\langle x^\top u\rangle = \langle x^\top K_{\rm sym} x\rangle\)
  for sinusoids. This is a **storage/dissipation balance** in the LLRQ model coordinates
  (not the thermodynamic entropy unless \(K\) aligns with \(L\)).

---

### One‑line takeaway
Design in \(u\)-frequency space and compute entropy with the **kernel**
\[
\boxed{H_u(\omega)=G(i\omega)^{\!H} L\,G(i\omega)},\qquad G(i\omega)=(K+i\omega I)^{-1}.
\]
Design in \(x\)-frequency space and entropy is just \(\int X(\omega)^{\!H} L\,X(\omega)\,d\omega/(2\pi)\).
