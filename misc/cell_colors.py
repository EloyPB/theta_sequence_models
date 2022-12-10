import numpy as np
import matplotlib.pyplot as plt


cm = 'terrain'
num_cells = 5

fig, ax = plt.subplots()
ax.scatter(np.arange(num_cells), np.ones(num_cells), c=np.arange(num_cells), cmap=cm, vmax=1.2*num_cells)
plt.show()
