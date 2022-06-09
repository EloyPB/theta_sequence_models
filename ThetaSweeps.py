import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from generic.smart_sim import Config, SmartSim
from Decoder import Decoder


class ThetaSweeps(SmartSim):
    dependencies = [Decoder]

    def __init__(self, min_fraction_ok, config=Config(), d={}):
        SmartSim.__init__(self, config, d)
        self.decoder: Decoder = d['Decoder']
        self.fields = self.decoder.fields
        self.network = self.decoder.network
        self.track = self.network.track

        self.min_steps_ok = self.network.theta_cycle_steps * min_fraction_ok

        self.slopes = []
        self.intercepts = []
        self.lengths = []
        self.fit_starts = []
        self.start_indices = []
        self.end_indices = []

        self.fit()

    def fit(self):
        indices_max = np.full(self.decoder.correlations.shape[0], np.nan)
        not_nan = ~np.isnan(self.decoder.correlations).any(axis=1)
        indices_max[not_nan] = np.argmax(self.decoder.correlations[not_nan], axis=1)

        x = np.arange(self.network.theta_cycle_steps)

        for start, end in zip(self.network.theta_cycle_starts[self.decoder.first_cycle:-1],
                              self.network.theta_cycle_starts[self.decoder.first_cycle+1:]):
            indices_max_cycle = indices_max[start-self.decoder.first_index: end-self.decoder.first_index]
            if (indices_max_cycle == 0).any() or (indices_max_cycle == self.decoder.fields.num_bins - 1).any():
                continue

            ok = ~np.isnan(indices_max_cycle)
            if np.sum(ok) < self.min_steps_ok:
                continue

            fit = linregress(x[ok], indices_max_cycle[ok])
            self.slopes.append(fit.slope)
            self.intercepts.append(fit.intercept)
            self.fit_starts.append(start)
            self.start_indices.append(start + np.argmax(ok))
            self.end_indices.append(end - np.argmax(ok[::-1]))
            self.lengths.append(fit.slope * (self.end_indices[-1] - self.start_indices[-1] - 1) * self.fields.bin_size)

    def plot(self):
        fig, ax = self.decoder.plot()
        fit_x = np.arange(self.network.theta_cycle_steps)
        for fit_start, start_index, end_index, slope, intercept \
                in zip(self.fit_starts, self.start_indices, self.end_indices, self.slopes, self.intercepts):
            x = np.arange(start_index, end_index) * self.track.dt

            rel_start = start_index - fit_start
            rel_end = end_index - fit_start
            ax.plot(x, (fit_x * slope + intercept)[rel_start:rel_end]*self.fields.bin_size, color='k')

    def length_vs_mean_speed(self, plot=True):
        speeds = []
        positions = []
        for start_index, end_index in zip(self.start_indices, self.end_indices):
            start_pos = self.track.x_log[start_index]
            positions.append(start_pos)
            speeds.append(self.track.mean_speeds[int(start_pos/self.fields.bin_size)])

        fig, ax = plt.subplots()
        sc = ax.scatter(speeds, self.lengths, c=positions)
        c_bar = fig.colorbar(sc)
        c_bar.set_label("Position (cm)")
        ax.set_xlabel("Mean speed (cm/s)")
        ax.set_ylabel("Theta sweep length (cm)")

        self.maybe_save_fig(fig, "length_vs_mean_speed")
        self.maybe_pickle_results(speeds, "speeds")
        self.maybe_pickle_results(self.lengths, "lengths")
        self.maybe_pickle_results(positions, "positions")


if __name__ == "__main__":
    sweeps = ThetaSweeps.current_instance(Config(pickle_instances=True))
    sweeps.plot()
    sweeps.length_vs_mean_speed()

    plt.show()
