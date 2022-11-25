import numpy as np
import matplotlib.pyplot as plt


# parameters defining v_bar(x)
# v_bar(x) = v_offset + v_slope_offset * x + v_slope_slope * x^2
v_offset = 10
v_slope_offset = 1
v_slope_slope = 0.1

v_factor = 0.5  # v(x) = v_factor * v_bar(x)

tau = 0.5
T = 0.1

dt = 0.0001
t_sim = tau + T + 0.05
time = np.arange(0, t_sim, dt)

x = [0]
v_bar_x = []
r1 = [0]
r2 = [0]

for t in time:
    v_bar_x.append(v_offset + v_slope_offset * x[-1] + v_slope_slope * (x[-1] ** 2))

    if t < T:
        v_x = v_bar_x[-1] * v_factor
        x.append(x[-1] + v_x * dt)

        # the approximation:
        v_bar_r1 = v_offset + v_slope_offset * r1[-1] + v_slope_slope * (r1[-1] ** 2)  # v_bar(r(t))
        k = 1 + (tau - T) * (v_slope_offset + v_slope_slope * x[-1])
        r1.append(r1[-1] + ((v_x - v_bar_x[-1]) * k + v_bar_r1 * (1 + tau / T)) * dt)

        # the real deal:
        r2_s = x[-1]  # starts at the current position and then we iterate forward
        for s in np.arange(0, t * tau / T, dt):
            v_bar_r2 = v_offset + v_slope_offset * r2_s + v_slope_slope * (r2_s ** 2)  # v_bar(r(t))
            r2_s += v_bar_r2 * dt
        r2.append(r2_s)

    else:
        x.append(x[-1] + v_bar_x[-1] * dt)


# plot x(t) and r(t)
fig, ax = plt.subplots()
ax.axvline(T, color='C4')  # marks the theta period
ax.plot(time, x[:-1], label='x')
i_T = np.argmax(time >= T)
ax.plot(time[:i_T], r1[1:], label='approx.')
ax.plot(time[:i_T], r2[1:], label='true')
ax.axvline(T + tau, linestyle='dotted', color='k')
i_T_tau = np.argmax(time >= T + tau)
ax.axhline(x[i_T_tau], linestyle='dotted', color='k')
ax.legend(loc='lower center')
ax.set_ylim(0, max(x))


# plot v_bar
fig, ax = plt.subplots()
ax.plot(x[1:], v_bar_x)
ax.set_ylabel(r"$\bar{v}$ (cm/s)")
ax.set_xlabel("position (cm)")

plt.show()
