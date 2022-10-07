import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
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

        self.trajectory_starts = []  # spatial index where each theta trajectory starts
        self.trajectory_ends = []  # spatial index where each theta trajectory ends
        self.real_pos_starts = []  # real position corresponding to the start of the theta trajectory
        self.real_pos_ends = []  # real position corresponding to the end of the theta trajectory

        self.slopes = []  # slope of the linear fit to the theta trajectory
        self.intercepts = []  # intercept of the linear fit to the theta trajectory
        self.lengths = []  # length of the theta trajectory obtained from the fit
        self.fit_starts = []  # index where each of the linear fits starts
        self.start_indices = []  # first indices without nan values for each theta trajectory
        self.end_indices = []  # last indices without nan values for each theta trajectory

        self.behind_lengths = None

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
            self.real_pos_starts.append(np.mean(self.decoder.track.x_log[start+left_offset:start+left_offset+self.side_steps]))
            self.real_pos_ends.append(np.mean(self.decoder.track.x_log[end-right_offset-self.side_steps:end-right_offset]))

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

                ax.plot((start_index + self.side_steps/2) * self.track.dt,
                        self.trajectory_starts[cycle_num] * self.fields.bin_size, '*', color='C1')
                ax.plot((end_index - self.side_steps/2) * self.track.dt,
                        self.trajectory_ends[cycle_num] * self.fields.bin_size, '*', color='C3')

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

    def ahead_and_behind_vs_mean_speed(self, plot=False):
        ahead_lengths = np.array(self.trajectory_ends) * self.fields.bin_size - np.array(self.real_pos_ends)
        ahead_speeds = [self.track.mean_speeds[int(pos/self.fields.bin_size)] for pos in self.real_pos_ends]
        self.behind_lengths = np.array(self.real_pos_starts) - np.array(self.trajectory_starts) * self.fields.bin_size
        behind_speeds = [self.track.mean_speeds[int(pos/self.fields.bin_size)] for pos in self.real_pos_starts]

        self.maybe_pickle_results(ahead_lengths, "ahead_lengths", sub_folder="ahead_and_behind")
        self.maybe_pickle_results(ahead_speeds, "ahead_speeds", sub_folder="ahead_and_behind")
        self.maybe_pickle_results(self.real_pos_ends, "ahead_real_pos", sub_folder="ahead_and_behind")

        self.maybe_pickle_results(self.behind_lengths, "behind_lengths", sub_folder="ahead_and_behind")
        self.maybe_pickle_results(behind_speeds, "behind_speeds", sub_folder="ahead_and_behind")
        self.maybe_pickle_results(self.real_pos_starts, "behind_real_pos", sub_folder="ahead_and_behind")

        if plot:
            fig, ax = plt.subplots(1, 2, sharey='row', figsize=(8, 4))
            ax[0].scatter(ahead_speeds, ahead_lengths, c=self.real_pos_ends, vmin=0, vmax=self.track.length)
            ax[0].set_title("Ahead length")
            ax[0].set_xlabel("Mean speed (cm/s)")
            ax[0].set_ylabel("Length (cm)")
            sc = ax[1].scatter(behind_speeds, self.behind_lengths, c=self.real_pos_starts, vmin=0, vmax=self.track.length)
            ax[1].set_title("Behind length")
            ax[1].set_xlabel("Mean speed (cm/s)")
            c_bar = fig.colorbar(sc)
            c_bar.set_label("Position (cm)")
            self.maybe_save_fig(fig, "ahead_and_behind_lengths")

    def behind_length_vs_peak_shift(self, plot=False):
        # calculate average place field shift for each spatial bin
        shifts = np.full(self.fields.num_bins, np.nan)
        for bin_num in range(self.fields.num_bins):
            pos = (bin_num + 0.5) * self.fields.bin_size
            matches = np.nonzero(np.array(self.fields.true_peaks) == pos)
            if matches[0].size:
                shifts[bin_num] = np.mean(np.array(self.fields.shifts)[matches])
        shifts *= -1

        # interpolate missing values
        ok = ~np.isnan(shifts)
        interp_range = range(np.argmax(ok), self.fields.num_bins - np.argmax(ok[::-1]))
        first_pos_ok = (interp_range.start + 0.5) * self.fields.bin_size
        last_pos_ok = (interp_range.stop - 0.5) * self.fields.bin_size
        nan = np.isnan(shifts[interp_range])

        # find matching shifts for each behind length
        matched_shifts = []
        for real_pos in self.real_pos_starts:
            if first_pos_ok <= real_pos <= last_pos_ok:
                matched_shifts.append(np.interp((real_pos - first_pos_ok) / self.fields.bin_size, np.nonzero(~nan)[0],
                                                shifts[interp_range][~nan]))
            else:
                matched_shifts.append(np.nan)

        if plot:
            data = {'x': matched_shifts, 'y': self.behind_lengths}
            frame = pd.DataFrame(data, columns=['x', 'y'])
            g = sns.lmplot(data=frame, x='x', y='y', ci=95)
            max_value = max(np.nanmax(matched_shifts), np.nanmax(self.behind_lengths))
            g.ax.plot((0, max_value), (0, max_value), linestyle='dashed', color='k', zorder=0)
            g.ax.set_xlabel("Place field shift (cm)")
            g.ax.set_ylabel("Behind length (cm)")


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 11})
    variants = {'Network': 'LogPosInput'}

    sweeps = ThetaSweeps.current_instance(Config(identifier=2, variants=variants, pickle_instances=True, save_figures=False,
                                                 figure_format='png'))
    # sweeps.plot(t_start=150.62)
    # sweeps.plot(t_start=151.256, t_end=151.632)
    # sweeps.length_vs_mean_speed()

    sweeps.ahead_and_behind_vs_mean_speed(plot=True)
    sweeps.behind_length_vs_peak_shift(plot=True)

    plt.show()
