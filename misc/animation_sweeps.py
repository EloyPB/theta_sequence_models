import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation


# purple - blue - cyan colormap
c0 = np.array([209, 107, 165])/255
c1 = np.array([134, 168, 231])/255
c2 = np.array([95, 251, 241])/255
color_dict = {'red':   [[0,  c0[0], c0[0]],
                        [0.50,  c1[0], c1[0]],
                        [1.0,  c2[0], c2[0]]],

              'green': [[0,  c0[1], c0[1]],
                        [0.50,  c1[1], c1[1]],
                        [1.0,  c2[1], c2[1]]],

              'blue':  [[0,  c0[2], c0[2]],
                        [0.50,  c1[2], c1[2]],
                        [1.0,  c2[2], c2[2]]]}
cm = LinearSegmentedColormap('PurpleCyan', color_dict)


length = 120
aspect = 0.2
theta_distance = 40
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
for t in t_log[1:]:
    theta_phase = (t % 0.125) / 0.125 - 0.5  # goes between -0.5 and 0.5
    x_log.append(x_log[-1] + v * dt)
    r_log.append(x_log[-1] + theta_distance * theta_phase)


# fig, ax = plt.subplots()
# ax.plot(t_log, x_log)
# ax.plot(t_log, r_log)
# plt.show()


fig, ax = plt.subplots()
ax.set_xlim((0, length))
ax.set_ylim((0, length*aspect))
ax.set_aspect('equal')
ax.axis('off')
ax.axvline(length/2, linestyle='dashed', color='C7', zorder=0)
rect = plt.Rectangle((0, 0), length, length*aspect*0.1, facecolor='lightgray', edgecolor=None, zorder=0)
ax.add_patch(rect)

ax.plot(x, true_field)

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
    return [rat1, rat2]


ani = FuncAnimation(fig, update, frames=len(x_log), interval=40, blit=False, repeat=1)

plt.show()