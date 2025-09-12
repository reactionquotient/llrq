import numpy as np
import matplotlib.pyplot as plt

k = 0.6
b = 1.0
D = 0.25

t0, tf, dt = 0.0, 60.0, 0.05
t = np.arange(t0, tf + dt, dt)

u = np.zeros_like(t)
u[t >= 10.0] = 1.0

y_in = np.ones_like(t) * 1.0
y_in[t >= 20.0] = 2.0

x = np.zeros_like(t)
y = np.zeros_like(t)
x[0] = 0.3
y[0] = 1.2

for i in range(len(t) - 1):
    dx = -k * x[i] + b * u[i]
    dy = -D * y[i] + D * y_in[i]
    x[i + 1] = x[i] + dt * dx
    y[i + 1] = y[i] + dt * dy

plt.figure(figsize=(7, 4))
plt.plot(t, x, label="x (log-quotient)")
plt.plot(t, u, linestyle="--", label="u (input to x)")
plt.xlabel("time")
plt.ylabel("value")
plt.title("Decoupled x-dynamics: ẋ = -k x + b u")
plt.legend()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(t, y, label="y (moiety total)")
plt.plot(t, y_in, linestyle="--", label="y_in (inlet total)")
plt.xlabel("time")
plt.ylabel("value")
plt.title("Decoupled y-dynamics: ẏ = -D y + D y_in")
plt.legend()
plt.show()
