import sys
import numpy as np
import matplotlib.pyplot as plt
from generic.noise import smoothed_noise
from generic.smart_class import Config, SmartClass


class LinearTrack(SmartClass):
    def __init__(self, length, ds, dt, num_features, speed_profile_points, speed_factor_sigma=3,
                 speed_factor_amplitude=1, config=Config(), d={}):

        SmartClass.__init__(self, config, d)

        if length % ds != 0:
            sys.exit("the perimeter should be divisible by ds")

        self.length = length  # in cm
        self.ds = ds  # spatial bin size
        self.num_bins = int(length / ds)  # number of spatial bins
        self.dt = dt  # temporal bin size

        self.num_features = num_features
        self.features = []

        self.speed_factor_sigma = speed_factor_sigma
        self.speed_factor_amplitude = speed_factor_amplitude

        self.speed_profile = np.zeros(self.num_bins)  # the typical running speed at each spatial bin

        # lists to keep track of positions, speeds, speed factor and indices separating laps
        self.x_log = []
        self.speed_log = []
        self.speed_factor_log = []
        self.lap_indices = []

        self.pl_speed_profile(speed_profile_points)

        # calculate mean lap duration
        x = 0
        steps = 0
        while x < self.length:
            x += self.speed_profile[int(x/self.ds)] * self.dt
            steps += 1
        self.mean_lap_duration = steps * self.dt

    def pl_speed_profile(self, points):
        """Define a speed profile as a piecewise linear function.

        Args:
            points (tuple(tuple(float))): A tuple of (position, speed) pairs.
        """
        if len(points) < 2:
            sys.exit("at least two points are needed")
        for (x_l, s_l), (x_r, s_r) in zip(points[:-1], points[1:]):
            x_l_index = max(0, int(x_l / self.ds))
            x_r_index = min(self.num_bins - 1, int(x_r / self.ds))
            slope = (s_r - s_l) / (x_r - x_l)
            for x_index in range(x_l_index, x_r_index + 1):
                self.speed_profile[x_index] = s_l + slope * ((x_index + 0.5) * self.ds - x_l)

    def plot_speed_profile(self):
        fig, ax = plt.subplots()
        x = np.arange(self.ds/2, self.length, self.ds)
        ax.plot(x, self.speed_profile)
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Speed (cm/s)")

    def random_features(self, sigma_range, tanh_gain=3, amplitude=2, offset=0):
        for sigma in np.random.uniform(sigma_range[0], sigma_range[1], self.num_features):
            features = np.tanh(tanh_gain * smoothed_noise(self.length, self.ds, sigma)) * amplitude/2 + offset
            self.features.append(features)
        self.features = np.array(self.features)

        # sort by peak position
        self.features = self.features[np.argsort(np.argmax(self.features, axis=1))]

    def plot_features(self, features_per_col=12, num_cols=3):
        features_per_plot = features_per_col * num_cols
        x = np.arange(self.ds/2, self.length, self.ds)
        for feature_num, feature in enumerate(self.features):
            figure_plot_num = feature_num % features_per_plot
            if figure_plot_num == 0:
                fig, ax = plt.subplots(features_per_col, num_cols, sharex="all", sharey="all", figsize=(5, 9))
            row_num = int(figure_plot_num / num_cols)
            col_num = figure_plot_num % num_cols
            ax[row_num, col_num].plot(x, feature)

    def plot_features_heatmap(self):
        fig, ax = plt.subplots()
        ax.matshow(self.features, aspect='auto', origin='lower',
                   extent=(0, self.num_bins*self.ds, -0.5, self.num_features - 0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Feature #")

    def run_laps(self, num_laps, interlap_t=0):
        # generate speed factor
        duration = num_laps * (interlap_t + self.mean_lap_duration + self.dt)
        speed_factor = smoothed_noise(length=duration, ds=self.dt, sigma=self.speed_factor_sigma,
                                      amplitude=self.speed_factor_amplitude, mean=1)
        interlap_steps = int(interlap_t / self.dt)

        i = 0
        for lap_num in range(num_laps):
            x = 0
            self.lap_indices.append(len(self.x_log))
            if interlap_steps:
                self.x_log += [x for _ in range(interlap_steps)]
                self.speed_log += [0 for _ in range(interlap_steps)]
                i += interlap_steps

            while x < self.length:
                speed = self.speed_profile[int(x / self.ds)] * speed_factor[i]
                x += speed * self.dt
                self.speed_log.append(speed)
                self.x_log.append(x)
                i += 1

        self.speed_factor_log += speed_factor[:i].tolist()

    def plot_trajectory(self):
        fig, ax = plt.subplots(3, sharex="col")
        time = np.arange(len(self.x_log)) * self.dt
        ax[0].plot(time, self.x_log)
        ax[0].set_ylabel("Linearized\nposition (cm)")
        ax[1].plot(time, self.speed_factor_log)
        ax[1].set_ylabel("Speed factor")
        ax[2].plot(time, self.speed_log)
        ax[2].set_ylabel("Speed (cm/s)")
        ax[2].set_xlabel("Time (s)")
        fig.align_ylabels()


if __name__ == "__main__":
    track = LinearTrack.current_instance()
    track.run_laps(10, interlap_t=0)
    # track.run_laps(3, interlap_t=2)
    track.plot_trajectory()
    plt.show()
