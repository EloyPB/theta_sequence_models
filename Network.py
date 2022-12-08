import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from LinearTrack import LinearTrack
from generic.smart_sim import Config, SmartSim
from small_plots import *
from batch_config import *


TWO_PI = 2 * np.pi

class Network(SmartSim):
    dependencies = [LinearTrack]

    def __init__(self, num_units, tau, w_rec_sigma, w_rec_exc, w_rec_inh, w_rec_shift, act_sigmoid_gain,
                 act_sigmoid_midpoint, theta_min, theta_max, theta_concentration, base_f, tau_f, tau_d, pos_factor_0,
                 pos_factor_concentration, pos_factor_phase, pos_sigmoid_gain, pos_sigmoid_midpoint, reset_indices,
                 reset_value, learning_rate, log_act=False, log_theta=False, log_pos_input=False, log_dynamics=False,
                 log_after=0, config=Config(), d={}):

        SmartSim.__init__(self, config, d)

        if 'LinearTrack' in d:
            self.track: LinearTrack = d['LinearTrack']
        else:
            sys.exit("A LinearTrack instance should be provided in d")

        self.num_units = num_units
        self.tau = tau
        self.dt_over_tau = self.track.dt / tau
        self.first_logged_step = int(log_after / self.track.dt)
        logged_steps = len(self.track.x_log) - self.first_logged_step

        # initialize recurrent weights
        self.w_rec = np.empty((num_units, num_units))
        two_sigma_squared = 2 * w_rec_sigma**2
        n = np.arange(self.num_units)
        self.w_rec = np.exp(-(n.reshape(-1, 1) - n - w_rec_shift) ** 2 / two_sigma_squared)
        self.w_rec = self.w_rec * (w_rec_exc + w_rec_inh) - w_rec_inh

        self.act_sigmoid_gain = act_sigmoid_gain
        self.act_sigmoid_midpoint = act_sigmoid_midpoint
        self.theta_max = theta_max
        self.theta_concentration = theta_concentration
        self.theta_multiplier = (theta_max - theta_min) / np.exp(theta_concentration)
        self.theta_cycle_steps = 1 / (8 * self.track.dt)
        self.theta_phase_inc = TWO_PI / self.theta_cycle_steps

        self.base_f = base_f
        self.tau_f = tau_f
        self.tau_d = tau_d

        self.pos_factor_0 = pos_factor_0
        self.pos_factor_concentration = pos_factor_concentration
        self.pos_factor_phase = pos_factor_phase / 180 * np.pi
        self.pos_sigmoid_gain = pos_sigmoid_gain
        self.pos_sigmoid_midpoint = pos_sigmoid_midpoint
        self.w_pos = np.zeros((self.num_units, self.track.num_features))
        self.log_pos_input = log_pos_input
        if log_pos_input:
            self.pos_input_log = np.empty((logged_steps, num_units))

        self.log_dynamics = log_dynamics
        if log_dynamics:
            self.depression_log = np.empty((logged_steps, num_units))
            self.facilitation_log = np.empty((logged_steps, num_units))

        self.log_act = log_act
        if log_act:
            self.act_log = np.empty((logged_steps, num_units))
        self.act_out_log = np.empty((logged_steps, num_units))

        self.log_theta = log_theta
        if log_theta:
            self.theta_log = np.empty(logged_steps)
        self.theta_phase_log = np.empty(logged_steps)
        self.theta_phase_log[-1] = 0
        self.theta_cycle_starts = []

        self.run(reset_indices, reset_value, learning_rate)

    def plot_rec_weights(self, fig_size=(5, 5), inset_up_to=None, c_map='binary'):
        fig, ax = plt.subplots(1, constrained_layout=True, figsize=fig_size)
        mat = ax.matshow(self.w_rec, aspect='auto', origin='lower', cmap=c_map)
        ax.plot((0, self.num_units), (0, self.num_units), linestyle='dashed', color='C3')
        ax.set_xlim((-0.5, self.num_units - 0.5))
        ax.set_ylim((-0.5, self.num_units - 0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title(r"$W_{rec}$")
        ax.set_xlabel("Input place cell #")
        ax.set_ylabel("Output place cell #")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        plt.colorbar(mat, ax=ax, ticks=(self.w_rec.min(), 0.0, self.w_rec.max()), format=FormatStrFormatter('%.1f'))

        if inset_up_to is not None:
            axins = inset_axes(ax, width="35%", height="35%", loc='lower left', bbox_to_anchor=(0.615, 0.12, 1.1, 1.1),
                               borderpad=0, bbox_transform=ax.transAxes)
            axins.matshow(self.w_rec[:inset_up_to, :inset_up_to], aspect='auto', origin='lower', cmap=c_map)
            axins.plot((0, inset_up_to), (0, inset_up_to), linestyle='dashed', color='C3')
            axins.set_xlim((-0.5, inset_up_to - 0.5))
            axins.set_ylim((-0.5, inset_up_to - 0.5))
            axins.set_xticks((0, inset_up_to/2))
            axins.set_yticks((0, inset_up_to/2))
            axins.xaxis.set_ticks_position('bottom')
            axins.tick_params(axis='x', labelsize=5)
            axins.tick_params(axis='y', labelsize=5)
            axins.spines.right.set_visible(False)
            axins.spines.top.set_visible(False)

        self.maybe_save_fig(fig, "rec_weights")

    def run(self, reset_indices, reset_value=1, learning_rate=0, verbose=0):
        exp_concentration = np.exp(self.pos_factor_concentration)
        theta_phase = 0

        for lap, lap_start_step in enumerate(self.track.lap_start_steps):
            if verbose:
                print(f"running lap {lap}...")

            # reset activity and internal dynamics
            act = np.zeros(self.num_units)
            depression = np.zeros(self.num_units)
            facilitation = np.full(self.num_units, self.base_f)

            if lap + 1 < len(self.track.lap_start_steps):
                last_lap_step = self.track.lap_start_steps[lap + 1]
            else:
                last_lap_step = len(self.track.x_log)

            for t_step in range(lap_start_step, last_lap_step):
                i = t_step - self.first_logged_step

                # compute theta phase and theta inhibition
                theta_phase += self.theta_phase_inc
                if theta_phase > TWO_PI:
                    theta_phase -= TWO_PI
                    if i >= 0:
                        self.theta_cycle_starts.append(i)
                theta = (-np.exp(self.theta_concentration * np.cos(theta_phase))
                         * self.theta_multiplier + self.theta_max)

                # compute spatial input
                features = self.track.features[int(self.track.x_log[t_step] / self.track.ds)]
                pos_factor = (np.exp(self.pos_factor_concentration
                                     * np.cos(theta_phase - self.pos_factor_phase))
                              / exp_concentration)
                pos_input = self.f_pos(self.w_pos @ features) * pos_factor

                if t_step - lap_start_step < self.theta_cycle_steps:
                    clamp = np.zeros(self.num_units)
                    clamp[slice(*reset_indices)] = reset_value * pos_factor
                else:
                    clamp = 0

                act_out = self.f_act(act)
                ready = (1 - depression) * facilitation
                rec_input = self.w_rec @ (act_out * ready)
                act += (-act + clamp + theta + rec_input + self.pos_factor_0 * pos_input) * self.dt_over_tau

                depression += (-depression + act_out) * self.track.dt / self.tau_d
                facilitation += (-facilitation + self.base_f + (1 - facilitation)*act_out) * self.track.dt / self.tau_f
                # self.facilitation += (-self.facilitation + self.base_f + act_out) * self.track.dt / self.tau_f  # simpler

                # self.w_pos += learning_rate * pos_factor * (act_out * (act_out - pos_input))[np.newaxis].T * features
                self.w_pos += learning_rate * pos_factor * (act_out - pos_input)[np.newaxis].T * features

                if i >= 0:
                    self.theta_phase_log[i] = theta_phase
                    if self.log_theta:
                        self.theta_log[i] = theta
                    self.act_out_log[i] = act_out
                    if self.log_act:
                        self.act_log[i] = act.copy()
                    if self.log_pos_input:
                        self.pos_input_log[i] = pos_input.copy()
                    if self.log_dynamics:
                        self.depression_log[i] = depression.copy()
                        self.facilitation_log[i] = facilitation.copy()

    def f_act(self, x):
        return 1 / (1 + np.exp(-self.act_sigmoid_gain * (x - self.act_sigmoid_midpoint)))

    def f_pos(self, x):
        return 1 / (1 + np.exp(-self.pos_sigmoid_gain * (x - self.pos_sigmoid_midpoint)))

    def plot_activities(self, t_start=0, t_end=None, first_unit=0, last_unit=None, apply_f=False, pos_input=False,
                        theta=False, speed=False, fig_size=(6.4, 4.8)):
        index_start = max(int(t_start / self.track.dt) - self.first_logged_step, 0)
        index_end = int(t_end / self.track.dt) - self.first_logged_step if t_end is not None else len(self.act_out_log)

        if last_unit is None:
            last_unit = self.num_units

        if apply_f:
            act_log = self.act_out_log[index_start:index_end, first_unit:last_unit]
            v_min = 0
            v_max = 1
        else:
            act_log = self.act_log[index_start:index_end, first_unit:last_unit]
            v_min = act_log.min()
            v_max = act_log.max()

        extent = ((index_start + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  (index_end + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  first_unit - 0.5, last_unit - 0.5)

        rows = 2 + theta + speed
        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        height_ratios = [1, 1, 0.5] if theta else [1, 1]
        if speed:
            height_ratios.append(0.5)
        spec = fig.add_gridspec(rows, 2, height_ratios=height_ratios, width_ratios=[1, 0.03])

        ax0 = fig.add_subplot(spec[0:2, 0])
        mat = ax0.matshow(act_log.T, aspect="auto", origin="lower", extent=extent,
                          cmap='Blues',
                          # cmap='viridis',
                          vmin=v_min, vmax=v_max)

        ax0.set_title("Network activities")
        ax0.set_ylabel("Place cell #")
        if rows == 2:
            ax0.set_xlabel("Time (s)")
        color_bar = plt.colorbar(mat, cax=fig.add_subplot(spec[1, 1]))
        color_bar.set_label("Activation")
        color_bar.locator = ticker.MultipleLocator(0.5)
        color_bar.update_ticks()
        ax0.spines.right.set_visible(False)
        ax0.spines.top.set_visible(False)

        if pos_input:
            # foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 1, 1, 1)], N=100)  # white
            # foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 0, 1, 1)], N=100)  # magenta
            foreground = colors.LinearSegmentedColormap.from_list('f', [(44/255, 160/255, 44/255, 0), (44/255, 160/255, 44/255, 1)], N=100)  # tab:green
            # foreground = colors.LinearSegmentedColormap.from_list('f', [(148/255, 103/255, 189/255, 0), (148/255, 103/255, 189/255, 1)], N=100)  # tab:purple
            # foreground = colors.LinearSegmentedColormap.from_list('f', [(31/255, 119/255, 188/255, 0), (31/255, 119/255, 188/255, 1)], N=100)  # tab:blue


            matb = ax0.matshow(self.pos_input_log[index_start:index_end, first_unit:last_unit].T, aspect="auto", origin="lower",
                               extent=extent, cmap=foreground, vmin=0, vmax=1)
            # c_map = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 1), (1, 1, 1, 1)], N=100)
            # color_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=matb.norm, cmap=c_map), cax=fig.add_subplot(spec[0, 1]))
            color_bar = fig.colorbar(matb, cax=fig.add_subplot(spec[0, 1]))
            color_bar.set_label("Spatial Input")
            color_bar.locator = ticker.MultipleLocator(0.5)
            color_bar.update_ticks()

        if theta:
            ax1 = fig.add_subplot(spec[2, 0], sharex=ax0)
            time = np.arange(self.first_logged_step, self.first_logged_step + len(self.act_out_log)) * self.track.dt
            ax1.plot(time, self.theta_log)
            ax1.set_ylabel("Theta")
            ax1.set_xlabel("Time (s)")
            ax1.spines.right.set_visible(False)
            ax1.spines.top.set_visible(False)

        if speed:
            ax2 = fig.add_subplot(spec[2 + theta, 0], sharex=ax0)
            time = np.arange(self.first_logged_step, self.first_logged_step + len(self.act_out_log)) * self.track.dt
            ax2.plot(time, self.track.speed_log[self.first_logged_step:], color='C7')
            # max_v = max(max(self.track.speed_log) - 1, 1 - min(self.track.speed_log)) * 1.05
            # ax2.set_ylim(1 - max_v, 1 + max_v)
            ax2.set_ylabel("Speed (cm/s)")
            ax2.set_xlabel("Time (s)")
            ax2.spines.right.set_visible(False)
            ax2.spines.top.set_visible(False)

        ax0.xaxis.set_ticks_position('bottom')
        ax0.set_xlim(*extent)

        self.maybe_save_fig(fig, "activities", dpi=500)

    def plot_dynamics(self, t_start=0, t_end=None, first_unit=0, last_unit=None, apply_f=False, fig_size=(6, 5)):
        """Plots the short-term synaptic facilitation/depression values on top of the activation values.
        """
        index_start = max(int(t_start / self.track.dt) - self.first_logged_step, 0)
        index_end = int(t_end / self.track.dt) - self.first_logged_step if t_end is not None else len(self.act_out_log)

        if last_unit is None:
            last_unit = self.num_units

        if apply_f:
            act_log = np.array(self.act_out_log[index_start:index_end, first_unit:last_unit]).T
            v_min = 0
            v_max = 1
        else:
            act_log = np.array(self.act_log[index_start:index_end, first_unit:last_unit]).T
            v_min = act_log.min()
            v_max = act_log.max()

        # foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 1, 1, 1)], N=100)
        blue = colors.LinearSegmentedColormap.from_list('f', [(31/255, 119/255, 188/255, 0), (31/255, 119/255, 188/255, 1)], N=100)  # tab:blue
        green = colors.LinearSegmentedColormap.from_list('f', [(44/255, 160/255, 44/255, 0), (44/255, 160/255, 44/255, 1)], N=100)  # tab:green
        orange = colors.LinearSegmentedColormap.from_list('f', [(255/255, 127/255, 14/255, 0), (255/255, 127/255, 14/255, 1)], N=100)  # tab:orange

        extent = ((index_start + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  (index_end + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  first_unit - 0.5, last_unit - 0.5)

        fig = plt.figure(figsize=fig_size)
        spec = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 0.03])
        c_map = 'binary'
        ax0 = fig.add_subplot(spec[0:2, 0])
        mat0 = ax0.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map, vmin=v_min, vmax=v_max)
        mat0b = ax0.matshow(self.depression_log[index_start:index_end, first_unit:last_unit].T, aspect="auto",
                            origin="lower", extent=extent, cmap=orange, vmin=0)
        bar0 = plt.colorbar(mat0, cax=fig.add_subplot(spec[0, 1]))
        bar0.locator = ticker.MultipleLocator(0.5)
        bar0.update_ticks()
        bar0.set_label("Act.")
        # c_map_bar = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 1), (1, 1, 1, 1)], N=100)
        bar0b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat0b.norm, cmap=orange), cax=fig.add_subplot(spec[1, 1]))
        bar0b.set_label("D")
        bar0b.locator = ticker.MultipleLocator(0.3)
        bar0b.update_ticks()

        ax1 = fig.add_subplot(spec[2:4, 0], sharex=ax0, sharey=ax0)
        mat1 = ax1.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map, vmin=v_min, vmax=v_max)

        mat1b = ax1.matshow(self.facilitation_log[index_start:index_end, first_unit:last_unit].T, aspect="auto", origin="lower",
                            extent=extent, cmap=green)
        bar1 = plt.colorbar(mat1, cax=fig.add_subplot(spec[2, 1]))
        bar1.locator = ticker.MultipleLocator(0.5)
        bar1.update_ticks()
        bar1.set_label("Act.")
        bar1b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat1b.norm, cmap=green), cax=fig.add_subplot(spec[3, 1]))
        bar1b.set_label("F")
        bar1b.locator = ticker.MultipleLocator(0.2)
        bar1b.update_ticks()

        ax2 = fig.add_subplot(spec[4:6, 0], sharex=ax0, sharey=ax0)
        # green = colors.LinearSegmentedColormap.from_list('f', [(44/255, 160/255, 44/255, 0), (44/255, 160/255, 44/255, 1)], N=100)  # tab:green
        mat2 = ax2.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map, vmin=v_min, vmax=v_max)
        mat2b = ax2.matshow(((1-self.depression_log[index_start:index_end, first_unit:last_unit])
                             * self.facilitation_log[index_start:index_end, first_unit:last_unit]).T,
                            aspect="auto", origin="lower", extent=extent, cmap=blue)
        bar2 = plt.colorbar(mat2, cax=fig.add_subplot(spec[4, 1]))
        bar2.locator = ticker.MultipleLocator(0.5)
        bar2.update_ticks()
        bar2.set_label("Act.")
        bar2b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat2b.norm, cmap=blue), cax=fig.add_subplot(spec[5, 1]))
        bar2b.set_label("(1-D) F")
        bar2b.locator = ticker.MultipleLocator(0.05)
        bar2b.update_ticks()

        ax0.set_ylabel("Place cell #")
        ax0.xaxis.set_ticks_position('bottom')
        ax0.tick_params(labelbottom=False)
        ax1.set_ylabel("Place cell #")
        ax1.xaxis.set_ticks_position('bottom')
        ax1.tick_params(labelbottom=False)
        ax2.set_ylabel("Place cell #")
        ax2.set_xlabel("Time (s)")
        ax2.xaxis.set_ticks_position('bottom')

        fig.tight_layout()

        self.maybe_save_fig(fig, "dynamics", dpi=500)

    def plot_dynamics_and_act_profile(self, t_start=0, t_end=None, first_unit=0, last_unit=None, fig_size=(6, 5)):
        """Plots the short-term synaptic facilitation/depression values and a profile of the theta sweeps.
        """
        index_start = max(int(t_start / self.track.dt) - self.first_logged_step, 0)
        index_end = int(t_end / self.track.dt) - self.first_logged_step if t_end is not None else len(self.act_out_log)

        if last_unit is None:
            last_unit = self.num_units

        # calculate the profiles of the theta sweeps
        act_log = np.array(self.act_out_log[index_start:index_end, first_unit:last_unit])
        time = np.arange(index_start + self.first_logged_step, index_end + self.first_logged_step) * self.track.dt
        top_profiles = []
        bottom_profiles = []
        for i, activities in enumerate(act_log):
            if (activities > 0.5).any():
                bottom_profiles.append(np.argmax(activities > 0.5))
                top_profiles.append(bottom_profiles[-1] + np.argmax(activities[bottom_profiles[-1]:] < 0.5))
            else:
                top_profiles.append(np.nan)
                bottom_profiles.append(np.nan)

        in_sweep_before = False
        for i in range(len(time)):
            in_sweep_now = ~np.isnan(top_profiles[i]) | ~np.isnan(bottom_profiles[i])
            if (in_sweep_now and not in_sweep_before) or (not in_sweep_now and in_sweep_before):
                time = np.insert(time, i, time[i - in_sweep_before])
                bottom_profiles.insert(i, top_profiles[i - in_sweep_before])
                top_profiles.insert(i, bottom_profiles[i - in_sweep_before])
                in_sweep_before = not in_sweep_before
        bottom_profiles = np.array(bottom_profiles) + first_unit
        top_profiles = np.array(top_profiles) + first_unit

        extent = ((index_start + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  (index_end + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  first_unit - 0.5, last_unit - 0.5)

        fig, ax = plt.subplots(3, sharex='col', figsize=fig_size)
        mat0 = ax[0].matshow(self.depression_log[index_start:index_end, first_unit:last_unit].T, aspect="auto",
                             origin="lower", extent=extent, cmap='cividis')
        ax[0].plot(time, np.array(top_profiles), 'k', linewidth=0.5)
        ax[0].plot(time, np.array(bottom_profiles), 'k', linewidth=0.5)
        bar0 = plt.colorbar(mat0, ax=ax[0])
        bar0.set_label("D")

        mat1 = ax[1].matshow(self.facilitation_log[index_start:index_end, first_unit:last_unit].T, aspect="auto",
                             origin="lower", extent=extent, cmap='cividis')
        ax[1].plot(time, np.array(top_profiles), 'k', linewidth=0.5)
        ax[1].plot(time, np.array(bottom_profiles), 'k', linewidth=0.5)
        bar1 = plt.colorbar(mat1, ax=ax[1])
        bar1.set_label("F")

        mat2 = ax[2].matshow(((1-self.depression_log[index_start:index_end, first_unit:last_unit])
                             * self.facilitation_log[index_start:index_end, first_unit:last_unit]).T,
                             aspect="auto", origin="lower", extent=extent, cmap='cividis')
        ax[2].plot(time, np.array(top_profiles), 'k', linewidth=0.5)
        ax[2].plot(time, np.array(bottom_profiles), 'k', linewidth=0.5)
        bar2 = plt.colorbar(mat2, ax=ax[2])
        bar2.set_label("(1-D) F")

        ax[0].set_ylabel("Place cell #")
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].tick_params(labelbottom=False)
        ax[1].set_ylabel("Place cell #")
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].tick_params(labelbottom=False)
        ax[2].set_ylabel("Place cell #")
        ax[2].set_xlabel("Time (s)")
        ax[2].xaxis.set_ticks_position('bottom')

        fig.tight_layout()

        self.maybe_save_fig(fig, "dynamics", dpi=500)

    def plot_theta_and_pos_factor(self, cycles=2, num_points=1000, fig_size=(4.8, 2.5)):
        time = np.linspace(0, cycles/8, num_points)
        theta = 2 * np.pi * 8 * time
        theta_inh = -np.exp(self.theta_concentration * np.cos(theta)) * self.theta_multiplier + self.theta_max
        pos_factor = (np.exp(self.pos_factor_concentration * np.cos(theta - self.pos_factor_phase))
                      / np.exp(self.pos_factor_concentration))
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        for cycle in range(cycles + 1):
            ax.axvline(cycle / 8, color='gray', linestyle='dashed')
        ax.plot(time, theta_inh, color='k', label=r"$i_{\theta}$")
        ax.plot(time, pos_factor, color='C2', label=r"$\beta_{\theta}$")
        ax.set_xlabel("Time (s)")
        ax.legend(ncol=2)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        self.maybe_save_fig(fig, "components")


if __name__ == "__main__":

    config = Config(identifier=2, variants={
        'LinearTrack': 'OneLap',
        # 'LinearTrack': 'FixSpeed',
        'Network': 'LogAll'
        # 'Network': 'LogPosInput80'
    }, pickle_instances=True, save_figures=True, figures_root_path=figures_path, pickles_root_path=pickles_path,
                    figure_format='pdf')
    network = Network.current_instance(config)

    # network.track.plot_trajectory()
    # network.track.plot_features()
    # network.track.plot_features_heatmap()

    network.plot_rec_weights(fig_size=(5.5*CM, 4.42*CM), inset_up_to=15, c_map='binary')
    # network.plot_activities(apply_f=1)

    # show facilitation and depression in a few cycles at the beginning:
    # network.plot_dynamics(t_start=1.255, t_end=2.265, first_unit=28, last_unit=78, apply_f=1, fig_size=(12*CM, 10*CM))
    # network.plot_dynamics_and_act_profile(t_start=1.255, t_end=2.265, first_unit=28, last_unit=78,
    #                                       fig_size=(8.5 * CM, 11.35 * CM))

    # # zoom in on one run at the beginning:
    # network.plot_activities(apply_f=1, pos_input=0, theta=0, speed=1, t_start=1.255, t_end=2.265,
    #                         first_unit=28, last_unit=78)

    # all runs, id=2
    # network.plot_activities(apply_f=1, pos_input=1, theta=0, speed=1, last_unit=200, t_end=99.25, fig_size=(10*CM, 6.6*CM))
    # network.plot_activities(apply_f=1, pos_input=1, theta=0, speed=0, first_unit=57, last_unit=92,
    #                         t_start=2.51, t_end=3.005, fig_size=(5.2*CM, 5*CM))
    # network.plot_activities(apply_f=1, pos_input=1, theta=0, speed=0, first_unit=57, last_unit=92,
    #                         t_start=96.128, t_end=96.629, fig_size=(5.2*CM, 5*CM))



    # zoom in on one run at the end, id=1:
    # network.plot_activities(apply_f=1, pos_input=1, theta=0, speed=1, first_unit=34, last_unit=84, t_start=140, t_end=141.015)

    # network.plot_theta_and_pos_factor(fig_size=(5*CM, 4*CM))
    plt.show()
