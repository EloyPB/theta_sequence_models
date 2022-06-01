import sys
import numpy as np
import matplotlib.pyplot as plt
from generic.noise import smoothed_noise
from generic.smart_class import Config, SmartClass


class LinearTrack(SmartClass):
    def __init__(self, length, ds, dt, num_features, features_sigma_range, speed_profile_points, speed_factor_sigma=3,
                 speed_factor_amplitude=1, num_laps=0, interlap_t=0, config=Config(), d={}):

        SmartClass.__init__(self, config, d)

        if length % ds != 0:
            sys.exit("the perimeter should be divisible by ds")

        self.length = length  # in cm
        self.ds = ds  # spatial bin size
        self.num_bins = int(length / ds)  # number of spatial bins
        self.dt = dt  # temporal bin size

        self.num_features = num_features
        self.features = self.random_features(features_sigma_range)

        self.speed_factor_sigma = speed_factor_sigma
        self.speed_factor_amplitude = speed_factor_amplitude

        # the typical running speed at each spatial bi
        self.speed_profile = self.pl_speed_profile(speed_profile_points)

        # lists to keep track of positions, speeds, speed factor and indices separating laps
        self.x_log = []
        self.speed_log = []
        self.speed_factor_log = []
        self.lap_start_indices = []

        # calculate mean lap duration
        x = 0
        steps = 0
        while x < self.length:
            x += self.speed_profile[int(x/self.ds)] * self.dt
            steps += 1
        self.mean_lap_duration = steps * self.dt

        self.interlap_steps = int(interlap_t / self.dt)
        self.run_laps(num_laps)

    def pl_speed_profile(self, points):
        """Define a speed profile as a piecewise linear function.

        Args:
            points (tuple(tuple(float))): A tuple of (position, speed) pairs.
        """
        speed_profile = np.zeros(self.num_bins)
        if len(points) < 2:
            sys.exit("at least two points are needed")
        for (x_l, s_l), (x_r, s_r) in zip(points[:-1], points[1:]):
            x_l_index = max(0, int(x_l / self.ds))
            x_r_index = min(self.num_bins - 1, int(x_r / self.ds))
            slope = (s_r - s_l) / (x_r - x_l)
            for x_index in range(x_l_index, x_r_index + 1):
                speed_profile[x_index] = s_l + slope * ((x_index + 0.5) * self.ds - x_l)
        return speed_profile

    def plot_speed_profile(self):
        fig, ax = plt.subplots()
        x = np.arange(self.ds/2, self.length, self.ds)
        ax.plot(x, self.speed_profile)
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Speed (cm/s)")

    def random_features(self, sigma_range, amplitude=2, offset=0):
        features = np.empty((self.num_bins, self.num_features))
        for feature_num, sigma in enumerate(np.random.uniform(sigma_range[0], sigma_range[1], self.num_features)):
            feature = smoothed_noise(self.length, self.ds, sigma, amplitude, offset)
            features[:, feature_num] = feature

        # return features sorted by peak position
        return features[:, np.argsort(np.argmax(features, axis=0))]

    def plot_features(self, features_per_col=12, num_cols=3):
        features_per_plot = features_per_col * num_cols
        x = np.arange(self.ds/2, self.length, self.ds)
        for feature_num, feature in enumerate(self.features.T):
            figure_plot_num = feature_num % features_per_plot
            if figure_plot_num == 0:
                fig, ax = plt.subplots(features_per_col, num_cols, sharex="all", sharey="all", figsize=(5, 9))
            row_num = int(figure_plot_num / num_cols)
            col_num = figure_plot_num % num_cols
            ax[row_num, col_num].plot(x, feature)

    def plot_features_heatmap(self):
        fig, ax = plt.subplots()
        ax.matshow(self.features.T, aspect='auto', origin='lower',
                   extent=(0, self.num_bins*self.ds, -0.5, self.num_features - 0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Feature #")

    def run_laps(self, num_laps):
        # generate speed factor
        duration = num_laps * (self.mean_lap_duration + self.dt)  # dt seems to be necessary because of some stochastic numerical error
        speed_factor = smoothed_noise(length=duration, ds=self.dt, sigma=self.speed_factor_sigma,
                                      amplitude=self.speed_factor_amplitude, mean=1)

        i = 0
        for lap_num in range(num_laps):
            x = 0
            self.lap_start_indices.append(len(self.x_log))
            if self.interlap_steps:
                self.x_log += [x for _ in range(self.interlap_steps)]
                self.speed_log += [0 for _ in range(self.interlap_steps)]
                self.speed_factor_log += [np.nan for _ in range(self.interlap_steps)]

            while True:
                speed = self.speed_profile[int(x / self.ds)] * speed_factor[i]
                x += speed * self.dt
                if x >= self.length:
                    break
                self.speed_log.append(speed)
                self.x_log.append(x)
                self.speed_factor_log.append(speed_factor[i])
                i += 1

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

    print("plotting...")
    track.plot_trajectory()
    plt.show()