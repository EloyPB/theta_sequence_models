import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from LinearTrack import LinearTrack
from generic.smart_class import Config, SmartClass
from generic.timer import timer


class Network(SmartClass):
    dependencies = [LinearTrack]

    def __init__(self, num_units, tau, w_rec_sigma, w_rec_exc, w_rec_inh, w_rec_shift, act_sigmoid_gain,
                 act_sigmoid_midpoint, theta_min, theta_max, theta_concentration, base_f, tau_f, tau_d, pos_factor_0,
                 pos_factor_concentration, pos_factor_phase, pos_sigmoid_gain, pos_sigmoid_midpoint, log_pos_input=False,
                 log_dynamics=False, config=Config(), d={}):

        SmartClass.__init__(self, config, d)

        if 'LinearTrack' in d:
            self.track: LinearTrack = d['LinearTrack']
        else:
            sys.exit("A LinearTrack instance should be provided in d")

        self.num_units = num_units
        self.tau = tau
        self.dt_over_tau = self.track.dt / tau

        # initialize recurrent weights
        self.w_rec = np.empty((num_units, num_units))
        two_sigma_squared = 2 * w_rec_sigma**2
        n = np.arange(self.num_units)
        self.w_rec = np.exp(-(n.reshape(-1, 1) - n - w_rec_shift) ** 2 / two_sigma_squared)
        self.w_rec = self.w_rec * (w_rec_exc + w_rec_inh) - w_rec_inh

        self.act_sigmoid_gain = act_sigmoid_gain
        self.act_sigmoid_midpoint = act_sigmoid_midpoint
        self.theta_max = theta_max
        self.theta_amplitude = theta_max - theta_min
        self.theta_concentration = theta_concentration
        self.theta_concentration_exp = np.exp(theta_concentration)
        self.theta_cycle_steps = 1 / (8 * self.track.dt)

        self.base_f = base_f
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.depression = np.zeros(self.num_units)
        self.facilitation = np.full(self.num_units, self.base_f)

        self.pos_factor_0 = pos_factor_0
        self.pos_factor_concentration = pos_factor_concentration
        self.pos_factor_phase = pos_factor_phase / 180 * np.pi
        self.pos_sigmoid_gain = pos_sigmoid_gain
        self.pos_sigmoid_midpoint = pos_sigmoid_midpoint
        self.w_pos = np.zeros((self.num_units, self.track.num_features))
        self.log_pos_input = log_pos_input
        if log_pos_input:
            self.pos_input_log = np.empty((len(self.track.x_log), num_units))

        self.log_dynamics = log_dynamics
        if log_dynamics:
            self.depression_log = np.empty((len(self.track.x_log), num_units))
            self.facilitation_log = np.empty((len(self.track.x_log), num_units))

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

    def run(self, reset_indices, reset_value=1, l_rate=0):
        self.act_log[-1] = 0
        for lap, lap_start_index in enumerate(self.track.lap_start_indices):
            # reset activity and internal dynamics
            act = np.zeros(self.num_units)
            self.depression = np.zeros(self.num_units)
            self.facilitation = np.full(self.num_units, self.base_f)

            if lap + 1 < len(self.track.lap_start_indices):
                last_lap_index = self.track.lap_start_indices[lap + 1]
            else:
                last_lap_index = len(self.track.x_log)

            for index in range(lap_start_index, last_lap_index):
                theta_phase, self.theta_log[index] = self.theta(index)

                pos_factor = (np.exp(self.pos_factor_concentration * np.cos(theta_phase - self.pos_factor_phase))
                              / np.exp(self.pos_factor_concentration))
                features = self.track.features[int(self.track.x_log[index] / self.track.ds)]
                pos_input = self.f_pos(self.w_pos @ features) * pos_factor
                if self.log_pos_input:
                    self.pos_input_log[index] = pos_input.copy()

                if index - lap_start_index < self.theta_cycle_steps:
                    clamp = np.zeros(self.num_units)
                    clamp[slice(*reset_indices)] = reset_value * pos_factor
                else:
                    clamp = 0

                act_out = self.f_act(act)
                ready = (1 - self.depression) * self.facilitation
                rec_input = self.w_rec @ (act_out * ready)
                act += (-act + clamp + self.theta_log[index] + rec_input + self.pos_factor_0 * pos_input) * self.dt_over_tau
                self.act_log[index] = act.copy()

                self.depression += (-self.depression + act_out) * self.track.dt / self.tau_d
                self.facilitation += (-self.facilitation + self.base_f + (1 - self.facilitation)*act_out) * self.track.dt / self.tau_f
                # self.facilitation += (-self.facilitation + self.base_f + act_out) * self.track.dt / self.tau_f  # simpler

                if self.log_dynamics:
                    self.depression_log[index] = self.depression.copy()
                    self.facilitation_log[index] = self.facilitation.copy()

                if l_rate:
                    self.w_pos += l_rate * pos_factor * (act_out * (act_out - pos_input))[np.newaxis].T * features

    def theta(self, index):
        theta_phase = 2 * np.pi * (index % self.theta_cycle_steps) / self.theta_cycle_steps
        theta = (-np.exp(self.theta_concentration * np.cos(theta_phase)) / self.theta_concentration_exp
                 * self.theta_amplitude + self.theta_max)
        return theta_phase, theta

    def f_act(self, x):
        return 1 / (1 + np.exp(-self.act_sigmoid_gain * (x - self.act_sigmoid_midpoint)))

    def f_pos(self, x):
        return 1 / (1 + np.exp(-self.pos_sigmoid_gain * (x - self.pos_sigmoid_midpoint)))

    def plot_activities(self, t_start=0, t_end=None, apply_f=False, pos_input=False, theta=False):
        index_start = int(t_start / self.track.dt)
        index_end = int(t_end / self.track.dt) if t_end is not None else len(self.act_log)

        act_log = np.array(self.act_log[index_start:index_end])
        if apply_f:
            act_log = self.f_act(act_log)

        extent = (index_start * self.track.dt - self.track.dt / 2, index_end * self.track.dt - self.track.dt / 2,
                  -0.5, act_log.shape[1] - 0.5)

        rows = 2 + theta
        fig = plt.figure(constrained_layout=True)
        height_ratios = [1, 1, 0.5] if theta else [1, 1]
        spec = fig.add_gridspec(rows, 2, height_ratios=height_ratios, width_ratios=[1, 0.03])

        ax0 = fig.add_subplot(spec[0:2, 0])
        mat = ax0.matshow(act_log.T, aspect="auto", origin="lower", extent=extent, cmap='viridis')

        ax0.set_title("Network activities")
        ax0.set_ylabel("Unit #")
        color_bar = plt.colorbar(mat, cax=fig.add_subplot(spec[1, 1]))
        color_bar.set_label("Activation")

        if pos_input:
            foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 1, 1, 1)], N=100)
            matb = ax0.matshow(self.pos_input_log[index_start:index_end].T, aspect="auto", origin="lower",
                               extent=extent, cmap=foreground)
            c_map = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 1), (1, 1, 1, 1)], N=100)
            color_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=matb.norm, cmap=c_map), cax=fig.add_subplot(spec[0, 1]))
            color_bar.set_label("Pos Input")

        if theta:
            ax1 = fig.add_subplot(spec[2, 0], sharex=ax0)
            time = np.arange(len(self.act_log)) * self.track.dt
            ax1.plot(time, self.theta_log)
            ax1.set_ylabel("Theta")
            ax1.set_xlabel("Time (s)")

        ax0.xaxis.set_ticks_position('bottom')
        ax0.set_xlim(*extent)

    def plot_dynamics(self, t_start=0, t_end=None):
        index_start = int(t_start / self.track.dt)
        index_end = int(t_end / self.track.dt) if t_end is not None else len(self.act_log)

        act_log = np.array(self.act_log[index_start:index_end]).T

        foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 1, 1, 1)], N=100)
        extent = (index_start * self.track.dt - self.track.dt / 2, index_end * self.track.dt - self.track.dt / 2,
                  -0.5, act_log.shape[0] - 0.5)

        fig = plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 0.03])
        c_map = 'viridis'
        ax0 = fig.add_subplot(spec[0:2, 0])
        mat0 = ax0.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map)
        mat0b = ax0.matshow(self.depression_log[index_start:index_end].T, aspect="auto", origin="lower",
                            extent=extent, cmap=foreground)
        bar0 = plt.colorbar(mat0, cax=fig.add_subplot(spec[0, 1]))
        c_map_bar = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 1), (1, 1, 1, 1)], N=100)
        bar0b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat0b.norm, cmap=c_map_bar), cax=fig.add_subplot(spec[1, 1]))
        bar0b.set_label("D")

        ax1 = fig.add_subplot(spec[2:4, 0], sharex=ax0, sharey=ax0)
        mat1 = ax1.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map)
        mat1b = ax1.matshow(self.facilitation_log[index_start:index_end].T, aspect="auto", origin="lower",
                            extent=extent, cmap=foreground)
        bar1 = plt.colorbar(mat1, cax=fig.add_subplot(spec[2, 1]))
        bar1b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat1b.norm, cmap=c_map_bar), cax=fig.add_subplot(spec[3, 1]))
        bar1b.set_label("F")

        ax2 = fig.add_subplot(spec[4:6, 0], sharex=ax0, sharey=ax0)
        mat2 = ax2.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map)
        mat2b = ax2.matshow(((1-self.depression_log[index_start:index_end])
                             * self.facilitation_log[index_start:index_end]).T,
                            aspect="auto", origin="lower", extent=extent, cmap=foreground)
        bar2 = plt.colorbar(mat2, cax=fig.add_subplot(spec[4, 1]))
        bar2b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat2b.norm, cmap=c_map_bar), cax=fig.add_subplot(spec[5, 1]))
        bar2b.set_label("(1-D) F")

        ax0.set_ylabel("Unit #")
        ax0.xaxis.set_ticks_position('bottom')
        ax1.set_ylabel("Unit #")
        ax1.xaxis.set_ticks_position('bottom')
        ax2.set_ylabel("Unit #")
        ax2.set_xlabel("Time (s)")
        ax2.xaxis.set_ticks_position('bottom')

