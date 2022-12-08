import numpy as np
import matplotlib.pyplot as plt


cm = 'terrain'
num_cells = 6

fig, ax = plt.subplots()
ax.scatter(np.arange(num_cells), np.ones(num_cells), c=np.arange(num_cells), cmap=cm, vmax=num_cells*1.2)
plt.show()