import numpy as np
from theta_color_map import cm
from small_plots import *


length = 100
theta_distance = 20
dt = 0.001
x_start = 0
x_end = 100
v = 41.1
sigma = 3
num_laps = 50

x = np.linspace(0, length, 1000)
true_field = np.exp(-(x - length/2)**2/(2*sigma**2)) * 0.85 + 0.1
measured_field = np.max(np.exp(-(np.linspace(x - theta_distance/2, x + theta_distance/2, 100) - length/2)**2/(2*sigma**2)), axis=0)
measured_field = measured_field * 0.85 + 0.1

x_log = [x_start]
r_log = [x_start - theta_distance/2]
t_log = np.arange(0, num_laps * (x_end - x_start)/v, dt)
spike_x_log = []
spike_phase_log = []

for t in t_log[1:]:
    theta_phase = (t % 0.125) / 0.125 - 0.5  # goes between -0.5 and 0.5
    x_log.append(x_log[-1] + v * dt)
    if x_log[-1] >= x_end:
        x_log[-1] = x_start
    r_log.append(x_log[-1] + theta_distance * theta_phase)
    if np.random.random() < 30 * dt * np.exp(-(r_log[-1] - length/2)**2/(2*sigma**2)):
        spike_phase_log.append((theta_phase + 0.5) * 360)
        spike_x_log.append(x_log[-1])


fig, ax = plt.subplots(2, sharex='col', figsize=(8*CM, 4*CM), constrained_layout=True)
ax[0].set_xlim((0, length))
ax[0].set_ylim((0, 1))
ax[0].axis('off')
rect = plt.Rectangle((0, 0), length, 0.1, facecolor='lightgray', edgecolor=None, zorder=0)
ax[0].add_patch(rect)
ax[0].plot(x, true_field, color='C2')
ax[0].plot(x, measured_field, color='C0')

sweep = ax[0].imshow(np.vstack((np.linspace(0, 1, 100), np.linspace(0, 1, 100))), cmap=cm,
                     extent=(length/2 - theta_distance/2, length/2 + theta_distance/2, 0, 0.1), aspect='auto')


sc = ax[1].scatter(spike_x_log, spike_phase_log, c=spike_phase_log, vmin=0, vmax=360, cmap=cm, s=1.2)
ax[1].set_ylim((0, 360))
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
# ax[1].set_xticks([])
ax[1].set_yticks([0, 180, 360])
ax[1].set_xlabel("Position")
ax[1].set_ylabel(r"$\theta$ phase")

fig.savefig(f"figures/sweep_{theta_distance}.pdf")

plt.show()
