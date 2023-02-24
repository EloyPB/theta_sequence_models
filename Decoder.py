import copy
import numpy as np
from generic.smart_sim import Config, SmartSim
from PlaceFields import PlaceFields
from small_plots import *
from batch_config import *


class Decoder(SmartSim):
    dependencies = [PlaceFields]

    def __init__(self, min_peak, config=Config(), d={}):
        SmartSim.__init__(self, config, d)
        self.fields: PlaceFields = d['PlaceFields']
        self.network = self.fields.network
        self.track = self.network.track
        self.min_peak = min_peak

        num_time_bins = self.network.theta_cycle_starts[-1]
        self.correlations = np.empty((num_time_bins, self.fields.num_bins))

        self.decode()

    def decode(self):
        """Calculates the correlation between the population vector at some time window and the pop vector of
        firing rates at each spatial bin.
        """
        n = self.fields.last_unit
        mean_y = np.mean(self.fields.activations, axis=0)
        denom_y = np.sqrt(np.sum(self.fields.activations**2, axis=0) - n * mean_y**2)

        for t_step in range(self.network.theta_cycle_starts[-1]):
            x = self.network.act_out_log[t_step, :self.fields.last_unit]

            if np.max(x) < self.min_peak:
                self.correlations[t_step] = np.nan
            else:
                mean_x = np.mean(x)
                denom_x = np.sqrt(np.sum(x ** 2) - n * mean_x ** 2)
                self.correlations[t_step] = (x @ self.fields.activations - n * mean_x * mean_y) / (denom_x * denom_y)

    def plot(self, t_start=0, t_end=None, fig_size=(6, 6)):
        fig, ax = plt.subplots(figsize=fig_size)

        first_index = max(self.network.first_logged_step, int(t_start / self.track.dt))
        if t_end is None:
            last_index = self.network.theta_cycle_starts[-1] + self.network.first_logged_step
        else:
            last_index = min(self.network.theta_cycle_starts[-1] + self.network.first_logged_step,
                             int(t_end / self.track.dt))

        t_start = first_index * self.track.dt
        t_end = last_index * self.track.dt
        extent = (t_start - self.track.dt/2, t_end - self.track.dt/2, 0, self.fields.num_bins * self.fields.bin_size)

        c_map = copy.copy(plt.cm.get_cmap('rainbow'))
        c_map.set_bad(color='C7')
        mat = ax.matshow(self.correlations[first_index-self.network.first_logged_step:
                                           last_index-self.network.first_logged_step].T,
                         aspect='auto', origin='lower', extent=extent, cmap=c_map)
        ax.xaxis.set_ticks_position('bottom')
        c_bar = fig.colorbar(mat)
        c_bar.set_label("P.V. Correlation")

        t = np.arange(first_index, last_index) * self.track.dt
        ax.plot(t, self.track.x_log[first_index:last_index], color='k', label='real position')
        ax.legend(loc="upper left")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (cm)")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        self.maybe_save_fig(fig, "decoding", dpi=600)
        return fig, ax


if __name__ == "__main__":
    decoder = Decoder.current_instance(Config(identifier=1, variants=variants, pickle_instances=True,
                                              figures_root_path=figures_path, pickles_root_path=pickles_path,
                                              save_figures=True, figure_format='pdf'))
    # decoder.network.plot_activities(apply_f=True)
    # decoder.plot()
    decoder.plot(t_start=150.62, fig_size=(15.9*CM, 6*CM))
    # decoder.plot(t_start=151.256, t_end=151.632, fig_size=(3.5*CM, 3.5*CM))  # zoom in
    plt.show()
