import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# each line segment has ((x_left, y_left), (x_right, y_right))
segments = [[[0, 0], [1, 2]], [[1, 2], [2, 0]]]

fig, ax = plt.subplots()

norm = plt.Normalize(0, 1)  # a continuous norm to map from data points to colors
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array([0, 1])  # the colors of the lines
lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)
ax.set_xlim((0, 2))
ax.set_ylim((0, 2))

plt.show()