import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Network import Network, LinearTrack
from generic.smart_class import Config, SmartClass


class PlaceFields(SmartClass):
    dependencies = [Network]

    def __init__(self, bin_size, sigma, min_peak, threshold, prominence_threshold=0.33, discard=0, last_unit=None,
                 config=Config(), d={}):
        SmartClass.__init__(self, config, d)

        self.network = d['Network']
        self.track: LinearTrack = self.network.d['LinearTrack']
        self.track.compute_mean_speeds(bin_size)

        self.bin_size = bin_size
        self.num_bins = int(self.track.length / bin_size)
        self.bins_x = np.linspace(self.bin_size / 2, self.bin_size * (self.num_bins - 0.5), self.num_bins)
        self.sigma = sigma / bin_size
        self.min_peak = min_peak
        self.threshold = threshold
        self.prominence_threshold = prominence_threshold
        self.first_t_step = int(discard / self.track.dt)
        self.last_unit = self.network.num_units if last_unit is None else last_unit

        self.activations = np.full((self.last_unit, self.num_bins), np.nan)
        self.field_peak_ok = np.full(self.last_unit, False)
        self.field_peak_indices = np.full(self.last_unit, np.nan, dtype=int)
        self.field_bound_indices = np.full((self.last_unit, 2), np.nan, dtype=int)
        self.field_bounds_ok = np.full((self.last_unit, 2), False)
        self.field_prominence_ok = np.full(self.last_unit, False)
        self.field_sizes = np.full(self.last_unit, np.nan)

        self.compute_activations()
        self.compute_fields()

    def compute_activations(self):
        occupancies = np.zeros(self.num_bins, dtype=int)
        activations = np.zeros(self.activations.shape)
        for t_step in range(self.first_t_step, len(self.track.x_log)):
            bin_num = int(self.track.x_log[t_step] / self.bin_size)
            occupancies[bin_num] += 1
            activations[:, bin_num] += self.network.act_out_log[t_step][:self.last_unit]
        self.activations = activations / occupancies

        if self.sigma > 0:
            self.activations = gaussian_filter1d(self.activations, sigma=self.sigma, mode='nearest')

    def plot_activations(self):
        fig, ax = plt.subplots()
        c_map = copy.copy(plt.cm.get_cmap('viridis'))
        c_map.set_bad(color='white')
        mat = ax.matshow(self.activations, aspect='auto', origin='lower', cmap=c_map,
                         extent=(0, self.num_bins*self.bin_size, -0.5, self.last_unit-0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel("Unit #")
        ax.set_xlabel("Position (cm)")
        bar = fig.colorbar(mat)
        bar.set_label("Activation")
        fig.tight_layout()
        self.maybe_save_fig(fig, "place fields")

    def compute_fields(self):
        for field_num, activations in enumerate(self.activations):
            peak_index = np.argmax(activations)
            if activations[peak_index] < self.min_peak:
                continue

            self.field_peak_ok[field_num] = True
            self.field_peak_indices[field_num] = peak_index
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

            self.field_bound_indices[field_num] = (left_index, right_index)
            self.field_bounds_ok[field_num] = below_threshold[self.field_bound_indices[field_num]]

            prominence_threshold = activations[peak_index] * (1 - self.prominence_threshold)
            self.field_prominence_ok[field_num] \
                = ((activations[left_index:peak_index] <= prominence_threshold).any() and
                   (activations[peak_index:right_index + 1] < prominence_threshold).any())

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

    def sizes_vs_mean_speed(self, half_size=False, colour_by_position=True):
        speeds = []
        sizes = []

        for field_num, prominence_ok in enumerate(self.field_prominence_ok):
            if not prominence_ok:
                continue

            peak_index = self.field_peak_indices[field_num]
            bound_indices = self.field_bound_indices[field_num]
            speeds.append(np.nanmean(self.track.mean_speeds[bound_indices[0]:bound_indices[1] + 1]))

            if half_size:
                sizes.append(self.size(peak_index, bound_indices, self.field_bounds_ok[field_num]))
            else:
                sizes.append(self.half_size(peak_index, bound_indices, self.field_bounds_ok[field_num]))

        fig, ax = plt.subplots()
        if colour_by_position:
            positions = [index * self.bin_size for ok, index in
                         zip(self.field_prominence_ok, self.field_peak_indices) if ok]
            sc = ax.scatter(speeds, sizes, c=positions)
            bar = fig.colorbar(sc)
            bar.set_label("Peak position (cm)")

        else:
            ax.plot(speeds, sizes, 'o')
        y_label = "Place field half-size (cm)" if half_size else "Place field size (cm)"
        ax.set_ylabel(y_label)
        ax.set_xlabel("Mean running speed (cm/s)")
        self.maybe_save_fig(fig, "size_vs_speed")


if __name__ == "__main__":
    pf = PlaceFields.current_instance(Config(pickle_instances=True))
    pf.plot_activations()
    pf.sizes_vs_mean_speed(colour_by_position=True)

    plt.show()