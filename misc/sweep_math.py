import numpy as np
import matplotlib.pyplot as plt


v_offset = 20
v_slope = 2

tau = 0.4
T = 0.2

t_sim = tau + T + 0.05
dt = 0.001
time = np.arange(0, t_sim, dt)

x = [0]
r1 = [0]
r2 = [0, 0]
for t in time:
    v = v_offset + v_slope * x[-1]
    x.append(x[-1] + v * dt)
    v = v_offset + v_slope * r1[-1]
    r1.append(r1[-1] + v * tau / T * dt)
    v = v_offset + v_slope * r2[-1]
    r2.append(r2[-1] + (v * tau / T + v_slope * (r2[-1] - r2[-2])/dt * (t * tau / T)) * dt)


fig, ax = plt.subplots()
ax.plot(time, x[:-1])
ax.plot(time, r1[:-1])
ax.plot(time, r2[1:-1])
ax.axvline(T, linestyle='dashed', color='C4')
ax.axvline(T + tau, linestyle='dotted', color='k')
i_T_tau = np.argmax(time >= T + tau)
ax.axhline(x[i_T_tau], linestyle='dotted', color='k')
i_tau = np.argmax(time >= tau)
ax.axvline(tau, linestyle='dotted', color='C7')
ax.axhline(x[i_tau], linestyle='dotted', color='C7')

ax.set_ylim(0, max(x))

plt.show()
