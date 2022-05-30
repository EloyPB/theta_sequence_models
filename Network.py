import sys
import numpy as np
import matplotlib.pyplot as plt
from LinearTrack import LinearTrack
from generic.smart_class import Config, SmartClass


class Network(SmartClass):
    dependencies = [LinearTrack]

    def __init__(self, num_units, tau, w_rec_sigma, w_rec_exc, w_rec_inh, w_rec_shift, sigmoid_gain, sigmoid_midpoint,
                 theta_min, theta_max, theta_concentration, config=Config(), d={}):

        SmartClass.__init__(self, config, d)

        if 'LinearTrack' in d:
            self.track: LinearTrack = d['LinearTrack']
            dt = self.track.dt
        else:
            sys.exit("A LinearTrack instance should be provided in d")

        self.num_units = num_units
        self.dt_over_tau = dt / tau

        # initialize weights
        self.w_rec = np.empty((num_units, num_units))
        two_sigma_squared = 2 * w_rec_sigma**2
        n = np.arange(self.num_units)
        self.w_rec = np.exp(-(n.reshape(-1, 1) - n - w_rec_shift) ** 2 / two_sigma_squared)
        self.w_rec = self.w_rec * (w_rec_exc + w_rec_inh) - w_rec_inh

        self.sigmoid_gain = sigmoid_gain
        self.sigmoid_midpoint = sigmoid_midpoint
        self.theta_max = theta_max
        self.theta_amplitude = theta_max - theta_min
        self.theta_concentration = theta_concentration
        self.theta_concentration_exp = np.exp(theta_concentration)
        self.theta_cycle_steps = 1 / (8 * self.track.dt)

        self.act_log = np.empty((len(self.track.x_log), num_units))
        self.theta_log = np.empty(len(self.track.x_log))

    def plot_rec_weights(self):
        fig, ax = plt.subplots(1, constrained_layout=True)
        mat = ax.matshow(self.w_rec, aspect='auto', origin='lower')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Recurrent weights' matrix")
        ax.set_xlabel("Input unit number")
        ax.set_ylabel("Output unit number")
        plt.colorbar(mat, ax=ax)

    def run(self, reset_indices=(), reset_value=1):
        for lap, lap_start_index in enumerate(self.track.lap_start_indices):
            # reset
            act = np.zeros(self.num_units)
            if reset_indices:
                act[slice(reset_indices[0], reset_indices[1])] = reset_value

            if lap + 1 < len(self.track.lap_start_indices):
                last_lap_index = self.track.lap_start_indices[lap + 1]
            else:
                last_lap_index = len(self.track.x_log)

            for index in range(lap_start_index, last_lap_index):
                act_out = self.f_act(act)
                rec_input = self.w_rec @ act_out
                self.theta_log[index] = self.theta(index)
                act += (-act + rec_input + self.theta_log[index]) * self.dt_over_tau
                self.act_log[index] = act.copy()

    def theta(self, index):
        theta_phase = 2 * np.pi * (index % self.theta_cycle_steps) / self.theta_cycle_steps
        return (-np.exp(self.theta_concentration * np.cos(theta_phase)) / self.theta_concentration_exp
                * self.theta_amplitude + self.theta_max)

    def f_act(self, x):
        return 1 / (1 + np.exp(-self.sigmoid_gain * (x - self.sigmoid_midpoint)))

    def plot_activities(self, t_start=0, t_end=None, apply_f=False):
        index_start = int(t_start / self.track.dt)
        index_end = int(t_end / self.track.dt) if t_end is not None else len(self.act_log)

        act_log = np.array(self.act_log[index_start:index_end])
        if apply_f:
            act_log = self.f_act(act_log)
        extent = (index_start * self.track.dt - self.track.dt / 2, index_end * self.track.dt + self.track.dt / 2,
                  -0.5, act_log.shape[1] - 0.5)

        fig, axes = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': (1, 0.2), 'width_ratios': (1, 0.03)})
        ax = axes[0, 0]
        mat = ax.matshow(act_log.T, aspect="auto", origin="lower", extent=extent)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlim([extent[0], extent[1]])
        ax.set_title("Network activities")
        ax.set_ylabel("Unit #")
        color_bar = plt.colorbar(mat, cax=axes[0, 1])
        color_bar.set_label("Activation")

        ax = axes[1, 0]
        time = np.arange(len(self.act_log)) * self.track.dt
        ax.plot(time, self.theta_log)
        ax.set_ylabel("Theta")
        ax.set_xlabel("Time (s)")

        axes[1, 1].set_visible(False)

        fig.tight_layout()
