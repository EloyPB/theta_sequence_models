import copy
import numpy as np
from scipy import odr
import matplotlib.pyplot as plt
from matplotlib import ticker
from PlaceFields import LinearTrack, Network, PlaceFields, SmartSim, Config
from small_plots import *
from batch_config import *


class PhasePrecession(SmartSim):
    dependencies = [PlaceFields]

    def __init__(self, phase_bin_size, config=Config(), d={}):
        SmartSim.__init__(self, config, d)

        self.fields: PlaceFields = d['PlaceFields']
        self.network: Network = self.fields.d['Network']
        self.track: LinearTrack = self.network.d['LinearTrack']

        self.num_units = self.fields.last_unit
        self.spatial_bin_size = self.fields.bin_size
        self.num_spatial_bins = self.fields.num_bins
        self.phase_bin_size = phase_bin_size / 180 * np.pi
        self.num_phase_bins = int(2 * np.pi / self.phase_bin_size)

        self.clouds = np.full((self.num_units, self.num_phase_bins, self.num_spatial_bins), np.nan)
        self.slopes = np.full(self.num_units, np.nan)
        self.intercepts = np.full(self.num_units, np.nan)

        self.compute_clouds()

    def compute_clouds(self):
        occupancies = np.zeros((self.num_phase_bins, self.num_spatial_bins))
        clouds = np.zeros_like(self.clouds)

        for t_step in range(len(self.network.act_out_log)):
            spatial_bin_num = int(self.track.x_log[t_step + self.network.first_logged_step] / self.spatial_bin_size)
            phase_bin_num = int(self.network.theta_phase_log[t_step] / self.phase_bin_size)
            occupancies[phase_bin_num, spatial_bin_num] += 1
            clouds[:, phase_bin_num, spatial_bin_num] += self.network.act_out_log[t_step][:self.num_units]

        positive = occupancies > 0
        self.clouds[:, positive] = clouds[:, positive] / occupancies[positive]

        phase_span = self.phase_bin_size * (self.num_phase_bins - 1) * 180 / np.pi
        for unit_num, (cloud, bounds, bounds_ok) in \
                enumerate(zip(self.clouds, self.fields.field_bound_indices,
                              self.fields.field_bounds_ok)):
            if all(bounds_ok):
                slope, intercept = self.fit_cloud(cloud[:, bounds[0]:bounds[1]+1])
                self.slopes[unit_num] = slope * phase_span / (self.spatial_bin_size * (bounds[1] - bounds[0]))
                self.intercepts[unit_num] = intercept * phase_span

    def fit_cloud(self, cloud):
        pos, phase = np.meshgrid(np.linspace(0, 1, cloud.shape[1]), np.linspace(0, 1, cloud.shape[0]))
        valid = ~np.isnan(cloud) & (cloud != 0)
        return self.odr_fit(pos[valid], phase[valid], cloud[valid])

    @staticmethod
    def odr_fit(normalized_positions, normalized_phases, weights):
        data = odr.Data(normalized_positions, normalized_phases, we=weights, wd=weights)
        odr_instance = odr.ODR(data, odr.unilinear, beta0=[-1, 1])
        odr_output = odr_instance.run()
        slope = odr_output.beta[0]
        intercept = odr_output.beta[1]
        return slope, intercept

    def plot_cloud(self, unit):
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': (1, 0.5)}, sharey='row', figsize=(8, 4))

        bounds = self.fields.field_bound_indices[unit]
        fit_x = self.fields.bins_x[bounds[0]:bounds[1]+1]
        fit_x_rel = fit_x - self.fields.bins_x[bounds[0]]
        fit_y = self.intercepts[unit] + self.slopes[unit]*fit_x_rel

        mat = ax[0].matshow(self.clouds[unit], aspect='auto', origin='lower',
                            extent=(0, self.num_spatial_bins*self.spatial_bin_size, 0, 360))
        ax[0].plot(fit_x, fit_y, color='orange')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].set_ylabel("Phase (deg)")
        ax[0].set_xlabel("Position (cm)")

        bar = fig.colorbar(mat, ax=ax[1])
        bar.set_label("Activation")

        ax[1].matshow(self.clouds[unit, :, bounds[0]:bounds[1]+1], aspect='auto', origin='lower',
                      extent=(bounds[0]*self.spatial_bin_size, (bounds[1]+1)*self.spatial_bin_size, 0, 360))
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].set_xlabel("Position (cm)")
        ax[1].plot(fit_x, fit_y, color='orange')

        ax[0].set_yticks([0, 180, 360])
        ax[0].set_ylim((0, 360))

        fig.tight_layout()

        self.maybe_save_fig(fig, f"phase precession {unit}")

    def plot_clouds(self, units, fig_size=(4.8, 6)):
        fig, ax = plt.subplots(len(units), 1, sharex='col', figsize=fig_size, constrained_layout=True)

        c_map = copy.copy(plt.cm.get_cmap('viridis'))
        c_map.set_bad(color='gray')

        for row, unit in enumerate(units):
            bounds = self.fields.field_bound_indices[unit]
            fit_x = self.fields.bins_x[bounds[0]:bounds[1] + 1]
            fit_x_rel = fit_x - self.fields.bins_x[bounds[0]]
            fit_y = self.intercepts[unit] + self.slopes[unit] * fit_x_rel

            mat = ax[row].matshow(self.clouds[unit], aspect='auto', origin='lower', cmap=c_map,
                                  extent=(0, self.num_spatial_bins * self.spatial_bin_size, 0, 360), vmin=0, vmax=1)
            ax[row].plot(fit_x, fit_y, color='orange')
            ax[row].xaxis.set_ticks_position('bottom')
            if row == len(units) // 2:
                ax[row].set_ylabel("Phase (deg)")

            if row < len(units) - 1:
                ax[row].tick_params(labelbottom=False)
            else:
                bar = fig.colorbar(mat, ax=ax[row])
                bar.set_label("Act.")
                # bar.locator = ticker.MultipleLocator(0.5)
                # bar.update_ticks()

            ax[row].set_yticks([0, 180, 360])
            ax[row].set_ylim((0, 360))

        ax[-1].set_xlabel("Position (cm)")
        self.maybe_save_fig(fig, "clouds")

    def slopes_vs_mean_speed(self, plot=False, colour_by_position=True):
        speeds = []
        slopes = []
        positions = []
        not_nan = ~np.isnan(self.slopes)
        for slope, bounds, peak_index in zip(self.slopes[not_nan], self.fields.field_bound_indices[not_nan],
                                             self.fields.field_peak_indices[not_nan]):
            speeds.append(np.nanmean(self.track.mean_speeds[bounds[0]:bounds[1]+1]))
            slopes.append(1/slope)
            positions.append((peak_index + 0.5) * self.spatial_bin_size)

        self.maybe_pickle_results(speeds, "speeds")
        self.maybe_pickle_results(slopes, "slopes")
        self.maybe_pickle_results(positions, "positions")

        if plot:
            fig, ax = plt.subplots()
            if colour_by_position:
                sc = ax.scatter(speeds, slopes, c=positions)
                bar = fig.colorbar(sc)
                bar.set_label("Peak position (cm)")
            else:
                ax.plot(speeds, slopes, 'o')

            ax.set_ylabel("Inverse phase precession slope (cm/deg)")
            ax.set_xlabel("Mean running speed (cm/s)")
            self.maybe_save_fig(fig, "slope_vs_speed")

    def fast_and_slow_slopes(self, plot=False):
        occupancies = np.zeros((2, self.num_phase_bins, self.num_spatial_bins))
        clouds = np.zeros((self.num_units, 2, self.num_phase_bins, self.num_spatial_bins))

        for t_step in range(len(self.network.act_out_log)):
            spatial_bin_num = int(self.track.x_log[t_step + self.network.first_logged_step] / self.spatial_bin_size)
            phase_bin_num = int(self.network.theta_phase_log[t_step] / self.phase_bin_size)
            speed_factor = self.track.speed_factor_log[t_step + self.network.first_logged_step]
            i = int(speed_factor > 1)
            occupancies[i, phase_bin_num, spatial_bin_num] += 1
            clouds[:, i, phase_bin_num, spatial_bin_num] += self.network.act_out_log[t_step][:self.num_units]

        positive = occupancies > 0
        clouds[:, positive] = clouds[:, positive] / occupancies[positive]

        slopes = np.full((2, self.num_units), np.nan)
        # intercepts = np.full((2, self.num_units), np.nan)

        phase_span = self.phase_bin_size * (self.num_phase_bins - 1) * 180 / np.pi
        for i in range(2):
            for unit_num, (cloud, bounds, bounds_ok) in enumerate(zip(clouds[:, i], self.fields.field_bound_indices,
                                                                      self.fields.field_bounds_ok)):
                if all(bounds_ok):
                    slope, intercept = self.fit_cloud(cloud[:, bounds[0]:bounds[1]+1])
                    slopes[i, unit_num] = slope * phase_span / (self.spatial_bin_size * (bounds[1] - bounds[0]))
                    # intercepts[i, unit_num] = intercept * phase_span

        slopes = 1/slopes
        self.maybe_pickle_results(slopes, "slow_and_fast_slopes")

        # This is just to check
        # unit = 80
        # fig, ax = plt.subplots(3, sharey='all', sharex='all')
        # bounds = self.fields.field_bound_indices[unit]
        # ax[0].set_ylim(0, 360)
        # ax[0].set_xlim(0, (bounds[1] - bounds[0])*self.spatial_bin_size)
        # ax[0].matshow(clouds[unit, 0, :, bounds[0]:bounds[1]+1], origin='lower', aspect='auto',
        #               extent=(0, (bounds[1]-bounds[0]) * self.spatial_bin_size, 0, 360))
        # fit_x = self.fields.bins_x[bounds[0]:bounds[1] + 1]
        # fit_x_rel = fit_x - self.fields.bins_x[bounds[0]]
        # fit_y = intercepts[0, unit] + slopes[0, unit] * fit_x_rel
        # ax[0].plot(fit_x_rel, fit_y, color='orange')
        # ax[1].matshow(clouds[unit, 1, :, bounds[0]:bounds[1]+1], origin='lower', aspect='auto',
        #               extent=(0, (bounds[1]-bounds[0]) * self.spatial_bin_size, 0, 360))
        # fit_y = intercepts[1, unit] + slopes[1, unit] * fit_x_rel
        # ax[1].plot(fit_x_rel, fit_y, color='orange')
        # ax[2].matshow(self.clouds[unit, :, bounds[0]:bounds[1]+1], origin='lower', aspect='auto',
        #               extent=(0, (bounds[1]-bounds[0]) * self.spatial_bin_size, 0, 360))
        # fit_y = self.intercepts[unit] + self.slopes[unit] * fit_x_rel
        # ax[2].plot(fit_x_rel, fit_y, color='orange')
        # print(slopes[0, unit], slopes[1, unit])

        if plot:
            fig, ax = plt.subplots()
            ax.plot(slopes, color='C7')
            ax.plot(slopes, 'o', color='k')
            ax.set_xticks((0, 1))
            ax.set_xticklabels(('Slow', 'Fast'))
            ax.set_ylabel("Inverse phase precession slope (cm/º)")
            self.maybe_save_fig(fig, "slow_and_fast")


if __name__ == "__main__":
    # plt.rcParams.update({'font.size': 11})

    pp = PhasePrecession.current_instance(Config(variants={'LinearTrack': 'ManyLaps', 'Network': 'Log80'}, identifier=1,
                                                 pickle_instances=True, save_figures=True,
                                                 figures_root_path=figures_path, pickles_root_path=pickles_path,
                                                 figure_format='pdf'))
    # for unit in [40, 60, 80, 100, 120]:
    #     pp.plot_cloud(unit)
    # pp.slopes_vs_mean_speed()

    pp.plot_clouds((40, 60, 80, 100, 120), fig_size=(7*CM, 9*CM))

    # pp.fast_and_slow_slopes(plot=True)

    plt.show()
