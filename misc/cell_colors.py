import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


c0 = np.array([38, 84, 124])/255
c1 = np.array([239, 71, 111])/255
c2 = np.array([255, 209, 102])/255
c3 = np.array([6, 214, 160])/255
color_dict = {'red':   [[0,  c0[0], c0[0]],
                        [1/3,  c1[0], c1[0]],
                        [2/3,  c2[0], c2[0]],
                        [1.0,  c3[0], c3[0]]],

              'green': [[0,  c0[1], c0[1]],
                        [1/3,  c1[1], c1[1]],
                        [2/3,  c2[1], c2[1]],
                        [1.0,  c3[1], c3[1]]],

              'blue':  [[0,  c0[2], c0[2]],
                        [1 / 3, c1[2], c1[2]],
                        [2 / 3, c2[2], c2[2]],
                        [1.0,  c2[2], c2[2]]]}
cm = LinearSegmentedColormap('BlueYellow', color_dict)


cm = 'terrain'
num_cells = 5

fig, ax = plt.subplots()
ax.scatter(np.arange(num_cells), np.ones(num_cells), c=np.arange(num_cells), cmap=cm, vmax=1.2*num_cells)
plt.show()