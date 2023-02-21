import os
import math
from typing import Type
import pickle
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import ticker
import seaborn as sns
from generic.smart_sim import SmartSim
from batch_config import *
from PlaceFields import PlaceFields
from PhasePrecession import PhasePrecession
from ThetaSweeps import ThetaSweeps
from small_plots import *


# PLOT RESULTS

# plt.rcParams.update({'font.size': 11})


def plot(name, class_def: Type[SmartSim], x_label, rel_path_x, y_label, rel_path_y, z_label="", rel_path_z=None,
         z_binned_average=False, z_bin_size=1, z_bin_min_count=8, x_binned_average=False, x_bin_size=2,
         x_bin_min_count=8, s=8, alpha=1., fig_size=(5*CM, 5*CM), format='pdf', z_min=0, z_max=200, extra_plotting=None,
         multiple_locator=None, linear_fit=False):

    def load(rel_path):
        with open(f"{path}{rel_path}", 'rb') as f:
            return pickle.load(f)

    all_x = []
    all_y = []
    all_z = []

    for identifier in range(NUM_RUNS):
        path = class_def.complete_path(pickles_path, str(identifier), variants)
        all_x += list(load(rel_path_x))
        all_y += list(load(rel_path_y))
        if rel_path_z:
            all_z += list(load(rel_path_z))

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    sc = ax.scatter(all_x, all_y, c=all_z if len(all_z) else 'C7', s=s, alpha=alpha, edgecolors='none',
                    vmin=z_min, vmax=z_max)
    if len(all_z):
        c_bar = fig.colorbar(sc)
        c_bar.set_label(z_label)

    if x_binned_average:
        num_x_bins = int(math.ceil(max(all_x) / x_bin_size))
        x_centers = np.arange(x_bin_size / 2, num_x_bins * x_bin_size, x_bin_size)
        y_averages = np.zeros(num_x_bins)
        counts = np.zeros(num_x_bins)

        for x, y in zip(all_x, all_y):
            index = int(x / x_bin_size)
            y_averages[index] += y
            counts[index] += 1

        valid = counts >= x_bin_min_count
        y_averages[valid] = y_averages[valid] / counts[valid]
        ax.plot(x_centers[valid], y_averages[valid], color='k', linewidth=2)

    if z_binned_average:
        num_z_bins = int(math.ceil(max(all_z) / z_bin_size))
        z_centers = np.arange(z_bin_size / 2, num_z_bins*z_bin_size, z_bin_size)
        x_averages = np.zeros(num_z_bins)
        y_averages = np.zeros_like(x_averages)
        counts = np.zeros_like(x_averages)

        for x, y, z in zip(all_x, all_y, all_z):
            index = int(z / z_bin_size)
            x_averages[index] += x
            y_averages[index] += y
            counts[index] += 1

        valid = counts >= z_bin_min_count
        x_averages[valid] = x_averages[valid] / counts[valid]
        y_averages[valid] = y_averages[valid] / counts[valid]

        # ax.plot(x_averages[valid], y_averages[valid], color='white', linewidth=5)
        ax.plot(x_averages[valid], y_averages[valid], color='white', linewidth=3)

        # create line segments
        segments = []
        colors = []
        for x_left, y_left, z_left, x_right, y_right, z_right in \
                zip(x_averages[valid][:-1], y_averages[valid][:-1], z_centers[valid][:-1],
                    x_averages[valid][1:], y_averages[valid][1:], z_centers[valid][1:]):
            x_mid = (x_left + x_right) / 2
            y_mid = (y_left + y_right) / 2
            segments.append([(x_left, y_left), (x_mid, y_mid)])
            segments.append([(x_mid, y_mid), (x_right, y_right)])
            colors += [z_left, z_right]

        norm = plt.Normalize(z_min, z_max)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(colors)
        # lc.set_linewidth(2)
        lc.set_linewidth(1)
        line = ax.add_collection(lc)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if multiple_locator is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator))

    if extra_plotting is not None:
        extra_plotting(ax)

    if linear_fit:
        fit = linregress(all_x, all_y)
        x_range = np.array((min(all_x), max(all_x)))
        ax.plot(x_range, x_range*fit.slope + fit.intercept, color='k')
        print(f"slope = {fit.slope}, intercept = {fit.intercept}, r = {fit.rvalue}")

    fig.savefig(f"{figures_path}/ALL/{name}.{format}", dpi=500)


def plot_speed_ratios(name, class_def: Type[SmartSim], x_label, rel_path_y, bin_size=0.02, format='pdf',
                      fig_size=(4*CM, 3*CM)):
    all_ratios = []
    for identifier in range(NUM_RUNS):
        path = class_def.complete_path(pickles_path, str(identifier), variants)

        with open(f"{path}{rel_path_y}", 'rb') as f:
            y = pickle.load(f)

        ratio = y[1] / y[0]
        all_ratios += ratio[~np.isnan(ratio)].tolist()

    # fig, ax = plt.subplots()
    # ax.plot(np.random.random(len(all_ratios)), all_ratios, 'o')
    # ax.axhline(np.mean(all_ratios), color='k')
    # ax.set_ylabel(y_label)
    # max_v = max(max(all_ratios), -min(all_ratios)) * 1.05
    # ax.set_ylim((-max_v, max_v))

    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    max_deviation = np.abs(np.array(all_ratios) - 1).max()
    num_bins = int((2 * max_deviation / bin_size) // 2 * 2 + 1)
    left_edge = 1 - num_bins * bin_size / 2
    right_edge = 1 + num_bins * bin_size / 2

    ax.hist(all_ratios, bins=np.linspace(left_edge, right_edge, num_bins + 1), color='C7')
    ax.axvline(1, color='k', linewidth=0.75)
    ax.axvline(np.mean(all_ratios), linestyle='dotted', color='lightgray')
    ax.set_ylabel("Count")
    ax.set_xlabel(x_label)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    fig.savefig(f"{figures_path}/ALL/{name}.{format}", dpi=600)


path = f"{figures_path}/ALL"
if not os.path.exists(path):
    os.makedirs(path)

x_label = "Average speed (cm/s)"
small_fig = (5*CM, 4*CM)

plot("sizes", PlaceFields, x_label, "speeds", "Place field size (cm)", "sizes", "Position (cm)", "positions",
     z_binned_average=True, z_bin_size=2, fig_size=(4.94*CM, 5*CM))
plot_speed_ratios("size_increments", PlaceFields, "Fast / slow", "slow_and_fast_sizes",
                  bin_size=0.05, fig_size=(3.75*CM, 2.3*CM))
plot("densities", PlaceFields, x_label, "density/speeds", "Place field density (peaks/cm)", "density/densities",
     alpha=0.5, x_binned_average=True, x_bin_size=2, fig_size=(3.73*CM, 4.9*CM))
plot("shifts", PlaceFields, x_label, "shifts/speeds", "Place field shift (cm)", "shifts/shifts", "Position (cm)",
     "shifts/positions", z_binned_average=True, z_bin_size=2, fig_size=small_fig)

# # only for NetworkIndep:
# plot("sizes_vs_induction_speed", PlaceFields, "Induction speed (cm/s)", "induction_speeds/induction_speeds",
#      "Place field size (cm)", "induction_speeds/sizes", fig_size=(3.7*CM, 4.9*CM), linear_fit=True)

plot("slopes", PhasePrecession, x_label, "speeds", "1 / phase precession slope (cm/deg)", "slopes",
     "Position (cm)", "positions", z_binned_average=True, z_bin_size=2, fig_size=(5.25*CM, 5*CM), multiple_locator=0.05)
plot_speed_ratios("slope_increments", PhasePrecession, "Fast / slow", "slow_and_fast_slopes", fig_size=(3.75*CM, 2.3*CM))

plot("sweep lengths", ThetaSweeps, x_label, "speeds", "Theta sweep length (cm)", "lengths", "Position (cm)", "positions",
     z_binned_average=True, z_bin_size=2, s=5, format='png', fig_size=small_fig)
plot("ahead lengths", ThetaSweeps, x_label, "ahead_and_behind/ahead_speeds", "Look-ahead distance (cm)",
     "ahead_and_behind/ahead_lengths", "Position (cm)", "ahead_and_behind/ahead_real_pos",
     z_binned_average=True, z_bin_size=2, s=5, format='png', fig_size=small_fig)
plot("behind lengths", ThetaSweeps, x_label, "ahead_and_behind/behind_speeds", "Look-behind distance (cm)",
     "ahead_and_behind/behind_lengths", "Position (cm)", "ahead_and_behind/behind_real_pos",
     z_binned_average=True, z_bin_size=2, s=5, format='png', fig_size=small_fig)
plot("matched shifts", ThetaSweeps, "Place field shift (cm)", "ahead_and_behind/shifts",
     "Look-behind distance (cm)", "ahead_and_behind/behind_lengths", "Position (cm)",
     "ahead_and_behind/behind_real_pos", z_binned_average=True, z_bin_size=2, s=5, format='png',
     extra_plotting=lambda ax: ax.plot((0, 20), (0, 20), linestyle='dashed', color='black'), fig_size=small_fig)

plt.show()
