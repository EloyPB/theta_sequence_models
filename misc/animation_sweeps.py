import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from theta_color_map import cm


length = 120
aspect = 0.2
theta_distance = 30
dt = 0.005
x_start = 30
x_end = 90
v = 40
sigma = 3

x = np.linspace(0, length, 1000)
true_field = np.exp(-(x - length/2)**2/(2*sigma**2)) * length * aspect * 0.85 + length * aspect * 0.1

x_log = [x_start]
r_log = [x_start - theta_distance/2]
t_log = np.arange(0, (x_end - x_start)/v, dt)
field = [np.exp(-(r_log[-1] - length/2)**2/(2*sigma**2))]
spike_x_log = []
spike_phase_log = []
spike_times = []

for t in t_log[1:]:
    theta_phase = (t % 0.125) / 0.125 - 0.5  # goes between -0.5 and 0.5
    x_log.append(x_log[-1] + v * dt)
    r_log.append(x_log[-1] + theta_distance * theta_phase)
    field.append(np.max(np.exp(-(np.linspace(x_log[-1] - theta_distance/2, x_log[-1] + theta_distance/2, 100) - length/2)**2/(2*sigma**2))))
    if np.random.random() < np.exp(-(r_log[-1] - length/2)**2/(2*sigma**2)):
        spike_phase_log.append((theta_phase + 0.5) * 360)
        spike_x_log.append(x_log[-1])
        spike_times.append(t)

x_log = np.array(x_log)
t_log = np.array(t_log)
field = np.array(field) * length * aspect * 0.85 + length * aspect * 0.1
spike_times = np.array(spike_times)
spike_phase_log = np.array(spike_phase_log)
spike_x_log = np.array(spike_x_log)

# fig, ax = plt.subplots()
# ax.scatter(spike_x_log, spike_phase_log)

# fig, ax = plt.subplots()
# ax.plot(t_log, x_log)
# ax.plot(t_log, r_log)
# plt.show()


fig, axes = plt.subplots(2, sharex='col', figsize=(7, 3.5), constrained_layout=True)
ax = axes[0]
ax.set_xlim((0, length))
ax.set_ylim((0, length*aspect))
ax.set_aspect('equal')
ax.axis('off')
# ax.axvline(length/2, linestyle='dashed', color='C7', zorder=0)
rect = plt.Rectangle((0, 0), length, length*aspect*0.1, facecolor='lightgray', edgecolor=None, zorder=0)
ax.add_patch(rect)
ax.plot(x, true_field, color='C2')
ln, = ax.plot([], [], color='C0')

sc = axes[1].scatter(spike_x_log, spike_phase_log, c=spike_phase_log, vmin=0, vmax=360, cmap=cm)
sc.set_array(np.full(spike_times.size, np.nan))
axes[1].set_ylim((0, 360))
axes[1].spines.right.set_visible(False)
axes[1].spines.top.set_visible(False)
axes[1].set_xticks([])
axes[1].set_yticks([0, 180, 360])
axes[1].set_xlabel("Position")
axes[1].set_ylabel(r"$\theta$ phase")

rat = plt.imread('rat.png')
rat_size = 20
rat_shift = 2
rat_y_bottom = length * aspect * 0.1
rat_y_top = length * aspect * 0.1 + rat_size / rat.shape[1] * rat.shape[0]
rat1 = ax.imshow(rat, extent=(x_log[0] - rat_size + rat_shift, x_log[0] + rat_shift, rat_y_bottom, rat_y_top), zorder=3)
rat2 = ax.imshow(rat, extent=(r_log[0] - rat_size + rat_shift, r_log[0] + rat_shift, rat_y_bottom, rat_y_top), alpha=0.5, zorder=2)
sweep = ax.imshow(np.vstack((np.linspace(0, 1, 100), np.linspace(0, 1, 100))), cmap=cm,
                  extent=(x_log[0] - theta_distance/2, x_log[0] + theta_distance/2, 0, length*aspect*0.1))


def update(t_step):
    rat1.set(extent=(x_log[t_step] - rat_size + rat_shift, x_log[t_step] + rat_shift, rat_y_bottom, rat_y_top))
    rat2.set(extent=(r_log[t_step] - rat_size + rat_shift, r_log[t_step] + rat_shift, rat_y_bottom, rat_y_top))
    sweep.set(extent=(x_log[t_step] - theta_distance/2, x_log[t_step] + theta_distance/2, 0, length*aspect*0.1))
    t = t_step * dt
    ln.set_data(x_log[t_log < t], field[t_log < t])
    sc.set_array(np.where(spike_times < t, spike_phase_log, np.nan))
    if t_step == 250:
        fig.savefig("figures/sweep.pdf")
    return [rat1, rat2, ln, sc]


ani = FuncAnimation(fig, update, frames=len(x_log), interval=40, blit=False, repeat=1)
ani.save("figures/sweep_animation.mp4", dpi=400, writer='ffmpeg')

plt.show()