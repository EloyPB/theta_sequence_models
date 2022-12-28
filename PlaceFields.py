import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from LinearTrack import LinearTrack
from AbstractNetwork import AbstractNetwork
from NetworkClass import NetworkClass
from generic.smart_sim import Config, SmartSim
from batch_config import *
from small_plots import *


class PlaceFields(SmartSim):
    dependencies = [NetworkClass]

    def __init__(self, bin_size, sigma, min_peak, threshold, prominence_threshold=0.33, last_unit=None,
                 dens_window_size=10, dens_window_stride=2, config=Config(), d={}):
        SmartSim.__init__(self, config, d)

        self.network: AbstractNetwork = d[NetworkClass.__name__]
        self.track: LinearTrack = self.network.track
        self.track.compute_summary_speeds(bin_size)

        self.bin_size = bin_size
        self.num_bins = int(self.track.length / bin_size)
        self.bins_x = np.linspace(self.bin_size / 2, self.bin_size * (self.num_bins - 0.5), self.num_bins)
        self.sigma = sigma / bin_size
        self.min_peak = min_peak
        self.threshold = threshold
        self.prominence_threshold = prominence_threshold
        self.last_unit = min(self.network.num_units, self.network.num_units if last_unit is None else last_unit)
        self.dens_window_size = dens_window_size
        self.dens_window_stride = dens_window_stride

        self.occupancies = np.zeros(self.num_bins, dtype=int)
        self.activations = np.full((self.last_unit, self.num_bins), np.nan)
        self.pos_activations = np.full((self.last_unit, self.num_bins), np.nan)

        self.compute_activations()
        self.field_peak_indices, self.field_bound_indices, self.field_bounds_ok, self.field_prominence_ok = \
            self.compute_fields(self.activations)

        if self.network.log_pos_input:
            self.compute_true_fields()
            self.true_peaks = None
            self.shifts = None
            self.field_peak_shifts()

    def compute_activations(self):
        activations = np.zeros(self.activations.shape)
        for t_step in range(self.network.first_logged_step, len(self.track.x_log)):
            bin_num = int(self.track.x_log[t_step] / self.bin_size)
            self.occupancies[bin_num] += 1
            activations[:, bin_num] += self.network.act_out_log[t_step - self.network.first_logged_step][:self.last_unit]
        self.activations = activations / self.occupancies

        if self.sigma > 0:
            self.activations = gaussian_filter1d(self.activations, sigma=self.sigma, mode='nearest')

    def plot_activations(self, fig_size=(6.4, 4.8)):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        # c_map = copy.copy(plt.cm.get_cmap('viridis'))
        # c_map.set_bad(color='white')
        c_map = 'Blues'
        last_active_unit = np.argmax(np.nanmax(self.activations, axis=1) < 0.05)
        if last_active_unit == 0:
            last_active_unit = self.network.num_units
        mat = ax.matshow(self.activations[:last_active_unit], aspect='auto', cmap=c_map,
                         extent=(0, self.num_bins*self.bin_size, last_active_unit-0.5, -0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(0, self.track.length, 50))
        ax.set_ylabel("Place cell #")
        ax.set_xlabel("Position (cm)")
        bar = fig.colorbar(mat)
        bar.set_label("Activation")
        self.maybe_save_fig(fig, "place fields")

    def compute_fields(self, activations):
        peak_indices = np.full(self.last_unit, np.nan, dtype=int)
        bound_indices = np.full((self.last_unit, 2), np.nan, dtype=int)
        bounds_ok = np.full((self.last_unit, 2), False)
        prominence_ok = np.full(self.last_unit, False)

        for field_num, activations in enumerate(activations):
            peak_index = np.argmax(activations)
            if activations[peak_index] >= self.min_peak:
                peak_indices[field_num] = peak_index
                threshold = activations[peak_index] * self.threshold
                below_threshold = activations < threshold

                if np.sum(below_threshold[peak_index:]):
                    right_index = peak_index + np.argmax(below_threshold[peak_index:])
                else:
                    right_index = self.num_bins - 1

                if np.sum(below_threshold[:peak_index]):
                    left_index = peak_index - np.argmax(below_threshold[:peak_index + 1][::-1])
                else:
                    left_index = 0

                bound_indices[field_num] = (left_index, right_index)
                bounds_ok[field_num] = below_threshold[bound_indices[field_num]]

                prominence_threshold = activations[peak_index] * (1 - self.prominence_threshold)
                prominence_ok[field_num] \
                    = ((activations[left_index:peak_index] <= prominence_threshold).any() and
                       (activations[peak_index:right_index + 1] < prominence_threshold).any())

        return peak_indices, bound_indices, bounds_ok, prominence_ok

    def size(self, peak_index, bound_indices, bounds_ok):
        if all(bounds_ok):
            bins = bound_indices[1] - bound_indices[0]
        else:
            bound_index = bound_indices[0] if bounds_ok[0] else bound_indices[1]
            bins = 2 * abs(peak_index - bound_index)
        return bins * self.bin_size

    def half_size(self, peak_index, bound_indices, bounds_ok):
        if all(bounds_ok):
            bins = max(peak_index - bound_indices[0], bound_indices[1] - peak_index)
        else:
            bound_index = bound_indices[0] if bounds_ok[0] else bound_indices[1]
            bins = abs(peak_index - bound_index)
        return bins * self.bin_size

    def sizes_vs_mean_speed(self, half_size=False, plot=False, colour_by_position=True):
        speeds = []
        sizes = []
        positions = []

        for field_num, prominence_ok in enumerate(self.field_prominence_ok):
            if not prominence_ok:
                continue

            peak_index = self.field_peak_indices[field_num]
            bound_indices = self.field_bound_indices[field_num]
            speeds.append(np.nanmean(self.track.mean_speeds[bound_indices[0]:bound_indices[1] + 1]))
            positions.append((self.field_peak_indices[field_num] + 0.5) * self.bin_size)

            if half_size:
                sizes.append(self.half_size(peak_index, bound_indices, self.field_bounds_ok[field_num]))
            else:
                sizes.append(self.size(peak_index, bound_indices, self.field_bounds_ok[field_num]))

        self.maybe_pickle_results(speeds, "speeds")
        self.maybe_pickle_results(sizes, "sizes")
        self.maybe_pickle_results(positions, "positions")

        if plot:
            fig, ax = plt.subplots()
            if colour_by_position:
                sc = ax.scatter(speeds, sizes, c=positions)
                bar = fig.colorbar(sc)
                bar.set_label("Peak position (cm)")
            else:
                ax.plot(speeds, sizes, 'o')
            y_label = "Place field half-size (cm)" if half_size else "Place field size (cm)"
            ax.set_ylabel(y_label)
            ax.set_xlabel("Mean running speed (cm/s)")
            self.maybe_save_fig(fig, "size_vs_speed")

    def density_vs_mean_speed(self, plot=False, first_to_last=True):
        peak_positions = (self.field_peak_indices[self.field_prominence_ok] + 0.5) * self.bin_size
        if first_to_last:
            start = peak_positions.min()
            end = peak_positions.max()
        else:
            start = 0
            end = self.track.length
        starts = np.arange(start, end, self.dens_window_stride)
        ends = np.arange(self.dens_window_size, end, self.dens_window_stride)

        speeds = []
        densities = []

        for start, end in zip(starts, ends):
            count = np.sum((start <= peak_positions) & (peak_positions < end))
            densities.append(count / self.dens_window_size)
            start_index = int(start / self.bin_size)
            end_index = int(end / self.bin_size)
            speeds.append(np.nanmean(self.track.mean_speeds[start_index:end_index+1]))

        self.maybe_pickle_results(speeds, "speeds", sub_folder="density")
        self.maybe_pickle_results(densities, "densities", sub_folder="density")

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(speeds, densities)
            ax.set_xlabel("Mean speed (cm/s)")
            ax.set_ylabel("Place field density (peaks/cm)")
            self.maybe_save_fig(fig, "density_vs_speed")

    def separation_vs_mean_speed(self, plot=False):
        """Distance between neighbouring peaks.
        """
        peak_indices = np.sort(self.field_peak_indices[self.field_prominence_ok])
        speeds = []
        separations = []

        for left, right in zip(peak_indices, peak_indices[1:]):
            separations.append((right - left) * self.bin_size)
            speeds.append(np.nanmean(self.track.mean_speeds[left:right+1]))

        self.maybe_pickle_results(speeds, "speeds", sub_folder="separation")
        self.maybe_pickle_results(separations, "separations", sub_folder="separation")

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(speeds, separations)
            ax.set_xlabel("Mean speed (cm/s)")
            ax.set_ylabel("Place field separation (cm)")
            self.maybe_save_fig(fig, "separation_vs_speed")

    def compute_true_fields(self):
        """Compute 'true' place fields based on each cell's positional input.
        """
        pos_activations = np.zeros(self.pos_activations.shape)
        for t_step in range(self.network.first_logged_step, len(self.track.x_log)):
            bin_num = int(self.track.x_log[t_step] / self.bin_size)
            pos_activations[:, bin_num] += self.network.pos_input_log[t_step - self.network.first_logged_step][:self.last_unit]
        self.pos_activations = pos_activations / self.occupancies

        if self.sigma > 0:
            self.pos_activations = gaussian_filter1d(self.pos_activations, sigma=self.sigma, mode='nearest')

    def plot_true_field(self, unit, start=0, fig_size=(6.4, 4.8)):
        activations = self.pos_activations[unit]
        first_bin = int(start/self.bin_size)

        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        ax.axvline(self.bins_x[np.argmax(activations)], color='C2', linestyle='dashed')
        ax.axvline(self.bins_x[np.argmax(self.activations[unit])], color='C3', linestyle='dashed')
        ax.plot(self.bins_x[first_bin:], self.activations[unit, first_bin:], color='C3', label='measured')
        ax.plot(self.bins_x[first_bin:], activations[first_bin:], color='C2', label='spatial input')
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Activation")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.legend(loc='upper right', fontsize='small')
        self.maybe_save_fig(fig, "true_field")

    def field_peak_shifts(self, plot=False):
        field_nums = []
        speeds = []
        measured_peaks = []
        self.true_peaks = []

        # x = np.arange(self.num_bins)
        # true_cms = []
        # measured_cms = []

        for field_num, prominence_ok in enumerate(self.field_prominence_ok):
            if not prominence_ok:
                continue

            field_nums.append(field_num)
            peak_index = self.field_peak_indices[field_num]
            bound_indices = self.field_bound_indices[field_num]
            speeds.append(np.nanmean(self.track.mean_speeds[bound_indices[0]:bound_indices[1] + 1]))
            measured_peaks.append((peak_index + 0.5) * self.bin_size)
            self.true_peaks.append((np.argmax(self.pos_activations[field_num]) + 0.5) * self.bin_size)

            # true_cms.append(np.sum(self.pos_activations[field_num] * x)/np.sum(self.pos_activations[field_num]))
            # measured_cms.append(np.sum(self.activations[field_num] * x)/np.sum(self.activations[field_num]))

        self.shifts = [m - t for m, t in zip(measured_peaks, self.true_peaks)]
        self.maybe_pickle_results(self.shifts, "shifts", sub_folder="shifts")
        self.maybe_pickle_results(speeds, "speeds", sub_folder="shifts")
        self.maybe_pickle_results(self.true_peaks, "positions", sub_folder="shifts")

        # self.cm_shifts = [(m - t)*self.bin_size for m, t in zip(measured_cms, true_cms)]

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            ax[0].scatter(measured_peaks, field_nums, label='measured')
            ax[0].scatter(self.true_peaks, field_nums, label='spatial input')
            ax[0].set_ylabel("Place field #")
            ax[0].set_xlabel("Peak position (cm)")
            ax[0].legend()

            sc = ax[1].scatter(speeds, self.shifts, c=self.true_peaks)
            ax[1].set_ylabel("Peak shift (cm)")
            ax[1].set_xlabel("Mean speed (cm/s)")
            bar = fig.colorbar(sc)
            bar.set_label("Peak position (cm)")

    def slow_and_fast_sizes(self, plot=False):
        occupancies = np.zeros((2, self.num_bins), dtype=int) + 0.1
        activations = np.zeros((self.last_unit, 2, self.num_bins))
        for t_step in range(self.network.first_logged_step, len(self.track.x_log)):
            bin_num = int(self.track.x_log[t_step] / self.bin_size)
            i = int(self.track.speed_log[t_step] > self.track.median_speeds[bin_num])
            occupancies[i, bin_num] += 1
            activations[:, i, bin_num] += self.network.act_out_log[t_step - self.network.first_logged_step][:self.last_unit]
        activations /= occupancies

        sizes = np.full((2, self.last_unit), np.nan)
        for i in range(2):
            if self.sigma > 0:
                activations[:, i] = gaussian_filter1d(activations[:, i], sigma=self.sigma, mode='nearest')
            peak_indices, bound_indices, bounds_ok, prominence_ok = self.compute_fields(activations[:, i])
            for unit_num in range(self.last_unit):
                if prominence_ok[unit_num]:
                    sizes[i, unit_num] = self.size(peak_indices[unit_num], bound_indices[unit_num], bounds_ok[unit_num])

        self.maybe_pickle_results(sizes, "slow_and_fast_sizes")

        if plot:
            fig, ax = plt.subplots()
            ax.plot(sizes, color='C7')
            ax.plot(sizes, 'o', color='k')
            ax.set_xticks((0, 1))
            ax.set_xticklabels(('Slow', 'Fast'))
            ax.set_ylabel("Place field size (cm)")
            self.maybe_save_fig(fig, "slow_and_fast")


if __name__ == "__main__":
    # plt.rcParams.update({'font.size': 11})

    variants = {
        # 'LinearTrack': 'Many',
        'NetworkIntDriven': 'IntDrivenLog80',
        'NetworkExtDriven': 'ExtDrivenLog100'
    }
    pf = PlaceFields.current_instance(Config(identifier=1, variants=variants, pickle_instances=True,
                                             figures_root_path=figures_path, pickles_root_path=pickles_path,
                                             save_figures=True, figure_format='pdf'))
    pf.plot_activations(fig_size=(5*CM, 5*CM))
    pf.sizes_vs_mean_speed(colour_by_position=True, plot=True)
    # pf.density_vs_mean_speed()

    # pf.compute_true_fields()
    # pf.plot_true_field(unit=67, start=25, fig_size=(4.25*CM, 3*CM))
    # pf.field_peak_shifts(plot=True)

    # pf.slow_and_fast_sizes(plot=True)

    plt.show()
