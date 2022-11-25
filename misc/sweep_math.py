import numpy as np
import matplotlib.pyplot as plt


v_offset = 10
v_slope_offset = 1
v_slope_slope = 0.1

v_factor = 0.5

tau = 0.5
T = 0.1

t_sim = tau + T + 0.05
dt = 0.0001
time = np.arange(0, t_sim, dt)

x = [0]
v_bars = []
r0 = [0]
r1 = [0]
r2 = [0]

for t in time:
    v_bar = v_offset + v_slope_offset * x[-1] + v_slope_slope * (x[-1]**2)
    v_bars.append(v_bar)

    if t < T:
        v = v_bar * v_factor
        x.append(x[-1] + v * dt)

        v_r1 = v_offset + v_slope_offset * r1[-1] + v_slope_slope * (r1[-1]**2)
        k = 1 + (tau - T) * (v_slope_offset + v_slope_slope * x[-1])
        r1.append(r1[-1] + ((v - v_bar) * k + v_r1 * (1 + tau / T)) * dt)

        r2_s = x[-1]
        for s in np.arange(0, t * tau / T, dt):
            v_r2 = v_offset + v_slope_offset * r2_s + v_slope_slope * (r2_s**2)
            r2_s += v_r2 * dt
        r2.append(r2_s)

    else:
        x.append(x[-1] + v_bar * dt)


fig, ax = plt.subplots()
ax.axvline(T, color='C4')
i_T = np.argmax(time >= T)
ax.plot(time, x[:-1], label='x')
ax.plot(time[:i_T], r1[1:], label='r1')
ax.plot(time[:i_T], r2[1:], label='true')

ax.axvline(T + tau, linestyle='dotted', color='k')
i_T_tau = np.argmax(time >= T + tau)
ax.axhline(x[i_T_tau], linestyle='dotted', color='k')
# i_tau = np.argmax(time >= tau)
# ax.axvline(tau, linestyle='dotted', color='C7')
# ax.axhline(x[i_tau], linestyle='dotted', color='C7')
ax.legend(loc='lower center')
ax.set_ylim(0, max(x))

fig, ax = plt.subplots()
ax.plot(x[1:], v_bars)
ax.set_ylabel(r"$\bar{v}$ (cm/s)")
ax.set_xlabel("position (cm)")

plt.show()
