import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter
from matplotlib import cm, colors, image


length = 200
aspect = 0.2
fig, ax = plt.subplots(figsize=(7, 1.5))
ax.set_xlim((-1, length+1))
ax.set_ylim((-1, length*aspect+1))
ax.set_aspect('equal')
ax.axis('off')
rect = plt.Rectangle((0, length*aspect*0.7), length, length*aspect*0.1, facecolor='whitesmoke', edgecolor='gray')
ax.add_patch(rect)
rat = plt.imread('rat.png')
rat_size = 20
rat_y_bottom = length * aspect * 0.8
rat_y_top = length * aspect * 0.8 + rat_size / rat.shape[1] * rat.shape[0]
im = ax.imshow(rat, extent=(-rat_size, 0, rat_y_bottom, rat_y_top), zorder=1)
fig.tight_layout()

arrows = []
arrow_positions = []

circles = []
circle_centers = []
radius = 4
num_circles = 20
for x in np.linspace(radius, length-radius, num_circles):
    circle_centers.append(x)
    circles.append(plt.Circle((x, radius), radius, edgecolor='k', facecolor='w'))
    ax.add_patch(circles[-1])

edge_speed = 10
center_speed = 80
mean_speed = (edge_speed + center_speed) / 2
dt = 0.01

# calculate how long it takes the rat to reach the end
time = 0
rat_pos = radius
while rat_pos <= length - radius:
    d = abs(rat_pos - length/2) / (length/2)
    speed = d * edge_speed + (1 - d) * center_speed
    rat_pos += speed * dt
    time += dt

neural_speed = (num_circles - 1) / time
circles_sigma = 0.2


def init():
    global rat_pos
    rat_pos = radius
    return []


def update(t):
    global rat_pos
    neural_pos = neural_speed * t
    d = abs(rat_pos - length/2) / (length/2)
    speed = d * edge_speed + (1 - d) * center_speed
    rat_pos += speed * dt

    for circle_num, circle in enumerate(circles):
        act = np.exp(-(neural_pos - circle_num)**2 / (2 * circles_sigma**2))
        color = cm.Blues(act)
        circle.set_facecolor(color)

    im.set(extent=(rat_pos - rat_size, rat_pos, rat_y_bottom, rat_y_top))

    if len(arrows) <= int(neural_pos):
        print(int(neural_pos))
        x_bottom = circle_centers[len(arrows)]
        arrows.append(ax.annotate("", xy=(x_bottom, radius + 4), xytext=(rat_pos, rat_y_bottom - 4),
                                  arrowprops=dict(arrowstyle="<->")))
        arrow_positions.append(rat_pos)

    return circles + [im] + arrows


ani = FuncAnimation(fig, update, frames=np.arange(0, time + dt, dt), init_func=init, interval=30, blit=False,
                    repeat=False)
ani.save("figures/animation.gif", dpi=400)
fig.savefig("figures/result1.pdf")
# plt.show()


# plot each neuron at the position it gets associated with

fig, ax = plt.subplots(figsize=(7, 1.5))
ax.set_xlim((-1, length+1))
ax.set_ylim((-1, length*aspect+1))
ax.set_aspect('equal')
ax.axis('off')
rect = plt.Rectangle((0, length*aspect*0.7), length, length*aspect*0.1, facecolor='whitesmoke', edgecolor='gray')
ax.add_patch(rect)
im = ax.imshow(rat, extent=(0, rat_size, rat_y_bottom, rat_y_top), zorder=1)
for arrow_position in arrow_positions:
    circle = plt.Circle((arrow_position, radius), radius, edgecolor='k', facecolor='w')
    ax.add_patch(circle)
    ax.annotate("", xy=(arrow_position, radius + 4), xytext=(arrow_position, rat_y_bottom - 4),
                arrowprops=dict(arrowstyle="<->"))
fig.tight_layout()
fig.savefig("figures/result2.pdf")

plt.show()