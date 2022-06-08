from typing import Type
import pickle
import matplotlib.pyplot as plt
from generic.smart_sim import SmartSim
from batch_config import *
from PlaceFields import PlaceFields
from PhasePrecession import PhasePrecession


# PLOT RESULTS

def plot(class_def: Type[SmartSim], x_label, rel_path_x, y_label, rel_path_y, rel_path_c=None, c_label=""):

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

    fig, ax = plt.subplots()
    sc = ax.scatter(all_x, all_y, c=all_c if len(all_c) else None)
    if len(all_c):
        c_bar = fig.colorbar(sc)
        c_bar.set_label(c_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


x_label = "Mean speed (cm/s)"
# plot(PlaceFields, x_label, "speeds", "Place field size (cm)", "sizes", "positions", "Position (cm)")
plot(PlaceFields, x_label, "density/speeds", "Place field density (peaks/cm)", "density/densities")
plot(PlaceFields, x_label, "density/speeds", "Place field separation (cm/peak)", "density/separations")
# plot(PhasePrecession, x_label, "speeds", "Inverse phase precession slope (cm/deg)", "slopes", "positions", "Position (cm)")
plt.show()
