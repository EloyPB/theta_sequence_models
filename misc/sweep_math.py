import numpy as np
import matplotlib.pyplot as plt


v_offset = 20
v_slope = 2
v_factor = 1.5

tau = 0.4
T = 0.1

t_sim = tau + T + 0.05
dt = 0.001
time = np.arange(0, t_sim, dt)

x = [0]
r0 = [0]
r1 = [0]
r2 = [0, 0]

for t in time:
    if t < T:
        v = (v_offset + v_slope * x[-1]) * v_factor
        x.append(x[-1] + v * dt)

        v0 = v_offset + v_slope * r0[-1]
        r0.append(r0[-1] + v0 * tau / T * dt)

        v1 = v_offset + v_slope * r1[-1]
        r1.append(r1[-1] + (v + v1 * tau / T) * dt)

        v2 = v_offset + v_slope * r2[-1]
        r2.append(r2[-1] + (v2 * tau / T + v_slope * (r2[-1] - r2[-2])/dt * (t * tau / T)) * dt)

    else:
        v = v_offset + v_slope * x[-1]
        x.append(x[-1] + v * dt)


fig, ax = plt.subplots()
ax.axvline(T, color='C4')
i_T = np.argmax(time >= T)
ax.plot(time, x[:-1], label='x')
ax.plot(time[:i_T], r0[1:], label='r0')
ax.plot(time[:i_T], r1[1:], label='r1')
ax.plot(time[:i_T], r2[2:], label='r2')
ax.axvline(T + tau, linestyle='dotted', color='k')
i_T_tau = np.argmax(time >= T + tau)
ax.axhline(x[i_T_tau], linestyle='dotted', color='k')
i_tau = np.argmax(time >= tau)
ax.axvline(tau, linestyle='dotted', color='C7')
ax.axhline(x[i_tau], linestyle='dotted', color='C7')
ax.legend(loc='lower center')
ax.set_ylim(0, max(x))
ax.set_title(f"v_slope = {v_slope}; v_factor = {v_factor}")

plt.show()
