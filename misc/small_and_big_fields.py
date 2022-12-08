import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from small_plots import *


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
cm = LinearSegmentedColormap('BlueYellow', color_dict)


# length = 100
# x = np.linspace(0, length, 1000)
# sigmas = [4, 8]
#
# for sigma in sigmas:
#     half_width = np.sqrt(-2 * sigma**2 * np.log(0.5))
#     tenth_width = np.sqrt(-2 * sigma**2 * np.log(0.02))
#     num_fields = int(round((length - 2*tenth_width) / (2 * half_width))) + 1
#     field_centers = np.linspace(tenth_width, length - tenth_width, num_fields)[np.newaxis].T
#     y = np.exp(-(x - field_centers)**2 / (2 * sigma**2))
#
#     fig, ax = plt.subplots(figsize=(5, 0.8))
#     for field_num in range(num_fields):
#         ax.plot(x, y[field_num], color='darkgrey')
#     ax.axis('off')
#     fig.tight_layout()
#     fig.savefig(f"figures/fields_sigma_{sigma}.pdf")


# small and big fields with a smooth transition

length = 120
transition_length = 30
sigma = 4
width = 2 * np.sqrt(-2 * sigma**2 * np.log(0.02))
print(f"standard width = {width}")
separation = np.sqrt(-2 * sigma**2 * np.log(0.6))
num_fields = int(round((length - width) / (2 * separation))) + 1
field_centers = np.linspace(width/2, length - width/2, num_fields)[np.newaxis].T

x = [0]
ds = []
left = 0.6*length - transition_length/2
right = 0.6*length + transition_length/2
while x[-1] < length:
    if x[-1] < left:
        ds.append(0.01)
    elif x[-1] > right:
        ds.append(0.005)
    else:
        ds.append(0.005 * (x[-1] - left) / transition_length + 0.01 * (right - x[-1]) / transition_length)
    x.append(x[-1] + ds[-1])
x = np.array(x)

y = np.exp(-(x - field_centers) ** 2 / (2 * sigma ** 2))

# phase precession clouds
precession_range = 2 * np.sqrt(-2 * sigma**2 * np.log(0.06))
pref_phases = np.minimum(np.maximum(180 - (x - field_centers) * 360 / precession_range, 0), 360)
spike_phases = []

for x_bin, y_bin, bin_pref_phases in zip(x, y.T, pref_phases.T):
    bin_spike_phases = []
    for cell_num, (pref_phase, cell_y) in enumerate(zip(bin_pref_phases, y_bin)):
        if np.random.random() < 0.1 * cell_y:
            bin_spike_phases.append(np.random.normal(pref_phase, 30, 1) % 360)
        else:
            bin_spike_phases.append(np.nan)
    spike_phases.append(bin_spike_phases)


fig, ax = plt.subplots(3, sharex='col', figsize=(11*CM, 4*CM))


speeds = length / len(x) / np.array(ds)
ax[0].plot((speeds - speeds.min()) / (speeds.max() - speeds.min()) * 30 + 30)
ax[0].set_ylim(bottom=25)
ax[0].set_yticks([30, 60])
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)

for field_num in range(num_fields):
    if field_num == 2:
        color = 'k'
    elif field_num == 11:
        color = 'k'
    else:
        color = 'darkgray'
    ax[1].plot(y[field_num], color=color)
# ax[1].axis('off')
ax[1].set_yticks([])
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)

for bin_num, bin_spike_phases in enumerate(spike_phases):
    cell_spike_phase = bin_spike_phases[2]
    if ~np.isnan(cell_spike_phase):
        ax[2].scatter(bin_num, cell_spike_phase, color=cm(cell_spike_phase/360), s=0.3)

    cell_spike_phase = bin_spike_phases[11]
    if ~np.isnan(cell_spike_phase):
        ax[2].scatter(bin_num, cell_spike_phase, color=cm(cell_spike_phase/360), s=0.3)

ax[2].spines.right.set_visible(False)
ax[2].spines.top.set_visible(False)
ax[2].set_yticks([0, 360])
ax[2].set_xticks([])
ax[2].set_xlabel(r"Position")

fig.tight_layout()
fig.savefig(f"figures/fields_chirp.pdf")


plt.show()