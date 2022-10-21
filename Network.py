import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
from LinearTrack import LinearTrack
from generic.smart_sim import Config, SmartSim
from generic.timer import timer


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

    def plot_rec_weights(self):
        fig, ax = plt.subplots(1, constrained_layout=True)
        mat = ax.matshow(self.w_rec, aspect='auto', origin='lower')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title("Recurrent weights' matrix")
        ax.set_xlabel("Input unit number")
        ax.set_ylabel("Output unit number")
        plt.colorbar(mat, ax=ax)

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
        mat = ax0.matshow(act_log.T, aspect="auto", origin="lower", extent=extent, cmap='viridis',
                          vmin=v_min, vmax=v_max)

        ax0.set_title("Network activities")
        ax0.set_ylabel("Unit #")
        if rows == 2:
            ax0.set_xlabel("Time (s)")
        color_bar = plt.colorbar(mat, cax=fig.add_subplot(spec[1, 1]))
        color_bar.set_label("Activation")
        color_bar.locator = ticker.MultipleLocator(0.5)
        color_bar.update_ticks()

        if pos_input:
            # foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 1, 1, 1)], N=100)
            foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 0, 1, 1)], N=100)  # magenta
            matb = ax0.matshow(self.pos_input_log[index_start:index_end, first_unit:last_unit].T, aspect="auto", origin="lower",
                               extent=extent, cmap=foreground, vmin=0, vmax=1)
            # c_map = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 1), (1, 1, 1, 1)], N=100)
            # color_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=matb.norm, cmap=c_map), cax=fig.add_subplot(spec[0, 1]))
            color_bar = fig.colorbar(matb, cax=fig.add_subplot(spec[0, 1]))
            color_bar.set_label("Pos Input")
            color_bar.locator = ticker.MultipleLocator(0.5)
            color_bar.update_ticks()

        if theta:
            ax1 = fig.add_subplot(spec[2, 0], sharex=ax0)
            time = np.arange(self.first_logged_step, self.first_logged_step + len(self.act_out_log)) * self.track.dt
            ax1.plot(time, self.theta_log)
            ax1.set_ylabel("Theta")
            ax1.set_xlabel("Time (s)")

        if speed:
            ax2 = fig.add_subplot(spec[2 + theta, 0], sharex=ax0)
            time = np.arange(self.first_logged_step, self.first_logged_step + len(self.act_out_log)) * self.track.dt
            ax2.plot(time, self.track.speed_log[self.first_logged_step:])
            # max_v = max(max(self.track.speed_log) - 1, 1 - min(self.track.speed_log)) * 1.05
            # ax2.set_ylim(1 - max_v, 1 + max_v)
            ax2.set_ylabel("Speed (cm/s)")
            ax2.set_xlabel("Time (s)")
            ax2.spines.right.set_visible(False)
            ax2.spines.top.set_visible(False)

        ax0.xaxis.set_ticks_position('bottom')
        ax0.set_xlim(*extent)

        self.maybe_save_fig(fig, "activities", dpi=500)

    def plot_dynamics(self, t_start=0, t_end=None, first_unit=0, last_unit=None, apply_f=False):
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

        foreground = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 0), (1, 1, 1, 1)], N=100)
        extent = ((index_start + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  (index_end + self.first_logged_step) * self.track.dt - self.track.dt / 2,
                  first_unit - 0.5, last_unit - 0.5)

        fig = plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 0.03])
        c_map = 'viridis'
        ax0 = fig.add_subplot(spec[0:2, 0])
        mat0 = ax0.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map, vmin=v_min, vmax=v_max)
        mat0b = ax0.matshow(self.depression_log[index_start:index_end, first_unit:last_unit].T, aspect="auto",
                            origin="lower", extent=extent, cmap=foreground, vmin=0)
        bar0 = plt.colorbar(mat0, cax=fig.add_subplot(spec[0, 1]))
        bar0.locator = ticker.MultipleLocator(0.5)
        bar0.update_ticks()
        c_map_bar = colors.LinearSegmentedColormap.from_list('f', [(0, 0, 0, 1), (1, 1, 1, 1)], N=100)
        bar0b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat0b.norm, cmap=c_map_bar), cax=fig.add_subplot(spec[1, 1]))
        bar0b.set_label("D")
        bar0b.locator = ticker.MultipleLocator(0.3)
        bar0b.update_ticks()

        ax1 = fig.add_subplot(spec[2:4, 0], sharex=ax0, sharey=ax0)
        mat1 = ax1.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map, vmin=v_min, vmax=v_max)
        mat1b = ax1.matshow(self.facilitation_log[index_start:index_end, first_unit:last_unit].T, aspect="auto", origin="lower",
                            extent=extent, cmap=foreground)
        bar1 = plt.colorbar(mat1, cax=fig.add_subplot(spec[2, 1]))
        bar1.locator = ticker.MultipleLocator(0.5)
        bar1.update_ticks()
        bar1b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat1b.norm, cmap=c_map_bar), cax=fig.add_subplot(spec[3, 1]))
        bar1b.set_label("F")
        bar1b.locator = ticker.MultipleLocator(0.2)
        bar1b.update_ticks()

        ax2 = fig.add_subplot(spec[4:6, 0], sharex=ax0, sharey=ax0)
        mat2 = ax2.matshow(act_log, aspect="auto", origin="lower", extent=extent, cmap=c_map, vmin=v_min, vmax=v_max)
        mat2b = ax2.matshow(((1-self.depression_log[index_start:index_end, first_unit:last_unit])
                             * self.facilitation_log[index_start:index_end, first_unit:last_unit]).T,
                            aspect="auto", origin="lower", extent=extent, cmap=foreground)
        bar2 = plt.colorbar(mat2, cax=fig.add_subplot(spec[4, 1]))
        bar2.locator = ticker.MultipleLocator(0.5)
        bar2.update_ticks()
        bar2b = fig.colorbar(mpl.cm.ScalarMappable(norm=mat2b.norm, cmap=c_map_bar), cax=fig.add_subplot(spec[5, 1]))
        bar2b.set_label("(1-D) F")
        bar2b.locator = ticker.MultipleLocator(0.05)
        bar2b.update_ticks()

        ax0.set_ylabel("Unit #")
        ax0.xaxis.set_ticks_position('bottom')
        ax0.tick_params(labelbottom=False)
        ax1.set_ylabel("Unit #")
        ax1.xaxis.set_ticks_position('bottom')
        ax1.tick_params(labelbottom=False)
        ax2.set_ylabel("Unit #")
        ax2.set_xlabel("Time (s)")
        ax2.xaxis.set_ticks_position('bottom')

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
        ax.plot(time, theta_inh, label=r"$i_{\theta}$")
        ax.plot(time, pos_factor, color='k', label=r"$\beta_{\theta}$")
        ax.set_xlabel("Time (s)")
        ax.legend(ncol=2)
        self.maybe_save_fig(fig, "components")


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 11})

    config = Config(identifier=2, variants={
        # 'LinearTrack': 'OneLap',
        # 'LinearTrack': 'FixSpeed',
        'Network': 'LogAll'
        # 'Network': 'LogPosInput80'
    }, pickle_instances=True, save_figures=True, figure_format='png')
    network = Network.current_instance(config)

    # network.track.plot_trajectory()
    # network.track.plot_features()
    # network.track.plot_features_heatmap()

    # network.plot_rec_weights()
    # network.plot_activities(apply_f=1)

    # # show facilitation and depression on a few runs at the beginning:
    # network.plot_dynamics(t_start=1.255, t_end=2.265, first_unit=28, last_unit=78, apply_f=1)

    # # zoom in on one run at the beginning:
    # network.plot_activities(apply_f=1, pos_input=0, theta=0, speed=1, t_start=1.255, t_end=2.265,
    #                         first_unit=28, last_unit=78)

    # all runs:
    network.plot_activities(apply_f=1, pos_input=0, theta=0, speed=1, last_unit=200, t_end=99.25, fig_size=(8, 4.8))
    # zoom in on one run at the end:
    # network.plot_activities(apply_f=1, pos_input=1, theta=0, speed=1, first_unit=34, last_unit=84, t_start=140, t_end=141.015)

    # network.plot_theta_and_pos_factor(fig_size=(4, 2))
    plt.show()
