import numpy as np
from matplotlib.colors import LinearSegmentedColormap


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