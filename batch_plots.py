import os
from typing import Type
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generic.smart_sim import SmartSim
from batch_config import *
from PlaceFields import PlaceFields
from PhasePrecession import PhasePrecession
from ThetaSweeps import ThetaSweeps


# PLOT RESULTS

plt.rcParams.update({'font.size': 11})


def plot(name, class_def: Type[SmartSim], x_label, rel_path_x, y_label, rel_path_y, c_label="", rel_path_c=None, s=18,
         alpha=1., fig_size=(4, 3.5), format='pdf'):

    def load(rel_path):
        with open(f"{path}{rel_path}", 'rb') as f:
            return pickle.load(f)

    all_x = []
    all_y = []
    all_c = []

    for identifier in range(NUM_RUNS):
        path = class_def.complete_path(pickles_path, str(identifier), variants)
        all_x += load(rel_path_x)
        all_y += load(rel_path_y)
        if rel_path_c:
            all_c += load(rel_path_c)

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    sc = ax.scatter(all_x, all_y, c=all_c if len(all_c) else None, s=s, alpha=alpha, edgecolors='none',
                    vmin=0, vmax=200)
    if len(all_c):
        c_bar = fig.colorbar(sc)
        c_bar.set_label(c_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    fig.savefig(f"{figures_path}/ALL/{name}.{format}", dpi=400)


def plot_speed_increments(name, class_def: Type[SmartSim], y_label, rel_path_y, format='pdf'):
    all_increments = []
    for identifier in range(NUM_RUNS):
        path = class_def.complete_path(pickles_path, str(identifier), variants)

        with open(f"{path}{rel_path_y}", 'rb') as f:
            y = pickle.load(f)

        increments = y[1] - y[0]
        all_increments += increments[~np.isnan(increments)].tolist()

    fig, ax = plt.subplots()
    ax.plot(np.random.random(len(all_increments)), all_increments, 'o')
    ax.axhline(np.mean(all_increments), color='k')
    ax.set_ylabel(y_label)
    max_v = max(max(all_increments), -min(all_increments)) * 1.05
    ax.set_ylim((-max_v, max_v))

    fig, ax = plt.subplots(figsize=(4, 2.5), constrained_layout=True)
    last_edge = max(-(min(all_increments) - 1), max(all_increments) + 1)
    ax.hist(all_increments, bins=np.arange(-last_edge, last_edge + 1, 2), density=True)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Place field size at fast inst. speed minus\n"
                  "place field size at slow inst. speed (cm)")

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    fig.savefig(f"{figures_path}/ALL/{name}.{format}", dpi=400)


path = f"{figures_path}/ALL"
if not os.path.exists(path):
    os.makedirs(path)

x_label = "Mean speed (cm/s)"

plot("sizes", PlaceFields, x_label, "speeds", "Place field size (cm)", "sizes", "Position (cm)", "positions")
# plot("densities", PlaceFields, x_label, "density/speeds", "Place field density (peaks/cm)", "density/densities",
#      alpha=0.5)
# plot("separations", PlaceFields, x_label, "separation/speeds", "Place field separation (cm)",
#      "separation/separations")
# plot("shifts", PlaceFields, x_label, "shifts/speeds", "Place field peak shift (cm)", "shifts/shifts", "Position (cm)",
#      "shifts/positions")
# plot_speed_increments("size_increments", PlaceFields, r"$\Delta$ Place field size (cm)", "slow_and_fast_sizes")

# plot("slopes", PhasePrecession, x_label, "speeds", "Inverse phase precession slope (cm/deg)", "slopes",
#      "Position (cm)", "positions")
#
# plot("sweeps", ThetaSweeps, x_label, "speeds", "Theta sweep length (cm)", "lengths", "Position (cm)", "positions",
#      s=10, format='png')

plt.show()
