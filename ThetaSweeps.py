import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from generic.smart_sim import Config, SmartSim
from LinearTrack import LinearTrack
from Network import Network
from PlaceFields import PlaceFields
from Decoder import Decoder


class ThetaSweeps(SmartSim):
    dependencies = [Decoder]

    def __init__(self, min_fraction_ok, side_portion, config=Config(), d={}):
        SmartSim.__init__(self, config, d)
        self.decoder: Decoder = d['Decoder']
        self.fields: PlaceFields = self.decoder.fields
        self.network: Network = self.decoder.network
        self.track: LinearTrack = self.network.track

        self.min_steps_ok = self.network.theta_cycle_steps * min_fraction_ok
        self.side_steps = int(round(self.network.theta_cycle_steps * side_portion))

        self.trajectory_starts = []
        self.trajectory_ends = []
        self.real_positions = []

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
            left_offset = np.argmax(ok)
            self.start_indices.append(start + left_offset)
            right_offset = np.argmax(ok[::-1])
            self.end_indices.append(end - right_offset)
            self.lengths.append(fit.slope * (self.end_indices[-1] - self.start_indices[-1] - 1) * self.fields.bin_size)

            self.trajectory_starts.append(np.nanmean(indices_max_cycle[left_offset:left_offset+self.side_steps]))
            self.trajectory_ends.append(np.nanmean(indices_max_cycle[-right_offset - self.side_steps:-right_offset]))
            self.real_positions.append(self.decoder.track.x_log[int(round(start + self.network.theta_cycle_steps/2))])

    def plot(self, t_start=0, t_end=None):
        fig, ax = self.decoder.plot(t_start, t_end)
        fit_x = np.arange(self.network.theta_cycle_steps)
        first_index = int(t_start / self.track.dt)
        if t_end is None:
            last_index = self.network.theta_cycle_starts[-1]
        else:
            last_index = int(t_end / self.track.dt)
        for cycle_num, (fit_start, start_index, end_index, slope, intercept) \
                in enumerate(zip(self.fit_starts, self.start_indices, self.end_indices, self.slopes, self.intercepts)):
            if start_index > first_index and end_index < last_index:
                x = np.arange(start_index, end_index) * self.track.dt

                rel_start = start_index - fit_start
                rel_end = end_index - fit_start
                ax.plot(x, (fit_x * slope + intercept)[rel_start:rel_end]*self.fields.bin_size, color='k')

                ax.plot((start_index + self.side_steps/2) * self.track.dt, self.trajectory_starts[cycle_num] * self.fields.bin_size, '*', color='C1')
                ax.plot((fit_start + self.network.theta_cycle_steps/2) * self.track.dt, self.real_positions[cycle_num], '*', color='white')
                ax.plot((end_index - self.side_steps/2) * self.track.dt, self.trajectory_ends[cycle_num] * self.fields.bin_size, '*', color='C3')

        custom_lines = [Line2D([0], [0], color='white'),
                        Line2D([0], [0], color='black')]
        ax.legend(custom_lines, ['actual pos.', 'decoded pos.'], loc="upper left")
        self.maybe_save_fig(fig, "sweeps", dpi=500)

    def length_vs_mean_speed(self, plot=True):
        speeds = []
        positions = []
        for start_index, end_index in zip(self.start_indices, self.end_indices):
            start_pos = self.track.x_log[start_index]
            positions.append(start_pos)
            speeds.append(self.track.mean_speeds[int(start_pos/self.fields.bin_size)])

        if plot:
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
    plt.rcParams.update({'font.size': 11})

    sweeps = ThetaSweeps.current_instance(Config(identifier=1, pickle_instances=True, save_figures=False,
                                                 figure_format='png'))
    sweeps.plot(t_start=150.62)
    # sweeps.plot(t_start=151.256, t_end=151.632)
    # sweeps.length_vs_mean_speed()

    plt.show()
