import numpy as np
import matplotlib.pyplot as plt
from generic.smart_sim import Config, SmartSim
from PlaceFields import PlaceFields


class Decoder(SmartSim):
    dependencies = [PlaceFields]

    def __init__(self, min_peak, config=Config(), d={}):
        SmartSim.__init__(self, config, d)
        self.fields: PlaceFields = d['PlaceFields']
        self.network = self.fields.network
        self.track = self.network.track

        self.min_peak = min_peak

        self.first_cycle = np.searchsorted(self.network.theta_cycle_starts, self.fields.first_t_step)
        num_time_bins = self.network.theta_cycle_starts[-1] - self.network.theta_cycle_starts[self.first_cycle]

        self.correlations = np.empty((num_time_bins, self.fields.num_bins))

        self.decode()

    def decode(self):
        """Calculates the correlation between the population vector at some time window and the pop vector of
        firing rates at each spatial bin.
        """
        n = self.fields.last_unit
        mean_y = np.mean(self.fields.activations, axis=0)
        denom_y = np.sqrt(np.sum(self.fields.activations**2, axis=0) - n * mean_y**2)

        i = 0
        for start, end in zip(self.network.theta_cycle_starts[self.first_cycle:-1],
                              self.network.theta_cycle_starts[self.first_cycle+1:]):
            for t in range(start, end):
                x = self.network.act_out_log[t, :self.fields.last_unit]

                if np.max(x) < self.min_peak:
                    self.correlations[i] = 0
                else:
                    mean_x = np.mean(x)
                    denom_x = np.sqrt(np.sum(x ** 2) - n * mean_x ** 2)
                    self.correlations[i] = (x @ self.fields.activations - n * mean_x * mean_y) / (denom_x * denom_y)

                i += 1
                print(i)

    def plot(self):
        fig, ax = plt.subplots()

        first_index = self.network.theta_cycle_starts[self.first_cycle]
        t_start = first_index * self.track.dt
        t_end = self.network.theta_cycle_starts[-1] * self.track.dt
        extent = (t_start, t_end, 0, self.fields.num_bins * self.fields.bin_size)
        mat = ax.matshow(self.correlations.T, aspect='auto', origin='lower', extent=extent)
        ax.xaxis.set_ticks_position('bottom')
        c_bar = fig.colorbar(mat)
        c_bar.set_label("P.V. Correlation")

        t = np.arange(t_start, t_end, self.track.dt)
        ax.plot(t, self.track.x_log[first_index:self.network.theta_cycle_starts[-1]], color='white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (cm)")


if __name__ == "__main__":
    decoder = Decoder.current_instance(Config(pickle_instances=True))
    # decoder.network.plot_activities(apply_f=True)
    decoder.decode()
    decoder.plot()
    plt.show()
