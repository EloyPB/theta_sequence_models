import sys
import numpy as np
from matplotlib import colors
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from AbstractNetwork import AbstractNetwork
from LinearTrack import LinearTrack
from generic.smart_sim import Config
from batch_config import pickles_path, figures_path
from small_plots import *


TWO_PI = 2 * np.pi


class NetworkExtDriven(AbstractNetwork):
    dependencies = [LinearTrack]

    def __init__(self, num_units, tau, et_tau, ins_signal_tau, sensory_field_sigma, act_sigmoid_gain, act_sigmoid_midpoint,
                 theta_min, theta_max, theta_concentration,
                 pos_factor_0, pos_factor_concentration, pos_factor_phase,
                 learning_rate, w_inh, w_exc_max, k_plus, alpha_plus, beta_plus, k_minus, alpha_minus, beta_minus,
                 log_act=False, log_et=False, log_theta=False, log_pos_input=False,
                 log_after=0, config=Config(), d={}):

        AbstractNetwork.__init__(self, num_units, tau, log_act, log_theta, log_pos_input, log_after, config, d)

        self.et_tau = et_tau
        self.ins_signal_tau = ins_signal_tau
        self.log_et = log_et
        if self.log_et:
            self.et_log = np.empty((self.logged_steps, num_units))

        self.act_sigmoid_gain = act_sigmoid_gain
        self.act_sigmoid_midpoint = act_sigmoid_midpoint

        self.pos_factor_0 = pos_factor_0
        self.pos_factor_concentration = pos_factor_concentration
        self.pos_factor_phase = pos_factor_phase / 180 * np.pi

        self.theta_max = theta_max
        self.theta_concentration = theta_concentration
        self.theta_multiplier = (theta_max - theta_min) / np.exp(theta_concentration)

        sigma = sensory_field_sigma / self.track.ds
        centers = np.linspace(0, self.track.num_bins, num_units)
        self.sensory_fields = np.exp(-(np.arange(self.track.num_bins).reshape(-1, 1) - centers) ** 2 / (2 * sigma ** 2))

        self.w_inh = w_inh
        self.w_exc_max = w_exc_max
        self.k_plus = k_plus
        self.alpha_plus = alpha_plus
        self.beta_plus = beta_plus
        self.k_minus = k_minus
        self.alpha_minus = alpha_minus
        self.beta_minus = beta_minus

        # self.w_exc = np.zeros((self.num_units, self.num_units))
        self.w_exc = np.random.uniform(0, 0.01, (self.num_units, self.num_units))

        self.run(learning_rate, verbose=1)

    def plot_sensory_fields(self, fig_size=(5.5*CM, 4.42*CM)):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        mat = ax.matshow(self.sensory_fields.T, aspect='auto', origin='lower',
                         extent=(0, self.track.length, -0.5, self.num_units+0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Unit #")
        ax.set_title("Sensory fields")
        fig.colorbar(mat)

    def plot_rec_weights(self, fig_size=(5.5*CM, 4.42*CM), c_map='binary'):
        fig, ax = plt.subplots(1, constrained_layout=True, figsize=fig_size)
        mat = ax.matshow(self.w_exc + self.w_inh, aspect='auto', origin='lower', cmap=c_map)
        ax.plot((0, self.num_units), (0, self.num_units), linestyle='dashed', color='C3')
        ax.set_xlim((-0.5, self.num_units - 0.5))
        ax.set_ylim((-0.5, self.num_units - 0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title(r"$W_{rec}$")
        ax.set_xlabel("Input place cell #")
        ax.set_ylabel("Output place cell #")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        plt.colorbar(mat, ax=ax)
        self.maybe_save_fig(fig, "rec_weights")

    def plot_q_functions(self):
        x = np.linspace(0, 1, 1000)
        q_plus = self.k_plus * self.sigmoid(x, self.alpha_plus, self.beta_plus)
        q_minus = self.k_minus * self.sigmoid(x, self.alpha_minus, self.beta_minus)
        fig, ax = plt.subplots()
        ax.plot(x, q_plus, label=r'$k^+ q^+ (ET \cdot IS)$')
        ax.plot(x, q_minus, label=r'$k^- q^- (ET \cdot IS)$')
        ax.set_xlabel(r"$ET \cdot IS$")
        ax.legend()

    @staticmethod
    def sigmoid(x, alpha, beta):
        return 1 / (1 + np.exp(-beta * (x - alpha)))

    def f_act(self, x):
        return 1 / (1 + np.exp(-self.act_sigmoid_gain * (x - self.act_sigmoid_midpoint)))

    def run(self, learning_rate, verbose=0):

        exp_concentration = np.exp(self.pos_factor_concentration)
        theta_phase = 0

        for lap, lap_start_step in enumerate(self.track.lap_start_steps):
            # reset at the beginning of each lap
            act = np.zeros(self.num_units)
            act_out = np.zeros(self.num_units)
            et = np.zeros(self.num_units)
            ins_signal = np.zeros(self.num_units)

            if verbose:
                print(f"running lap {lap}...")
                print(f"w_exc_max = {self.w_exc.max()}")

            if lap + 1 < len(self.track.lap_start_steps):
                last_lap_step = self.track.lap_start_steps[lap + 1]
            else:
                last_lap_step = len(self.track.x_log)

            max_overlaps = np.zeros((self.num_units, self.num_units))

            for t_step in range(lap_start_step, last_lap_step):
                i = t_step - self.first_logged_step


                # compute theta phase and theta inhibition
                theta_phase += self.theta_phase_inc
                if theta_phase > TWO_PI:
                    theta_phase -= TWO_PI
                    if i >= 0:
                        self.theta_cycle_starts.append(i)

                # if 0 < theta_phase < np.pi/4:
                #     theta = -1
                # else:
                #     theta = 0
                #
                # if np.pi/4 < theta_phase < np.pi/2:
                #     spatial_bin = int(self.track.x_log[t_step] / self.track.ds)
                #     pos_input = self.pos_factor_0 * self.sensory_fields[spatial_bin]
                #
                #     delta_et = (act_out - et) * self.track.dt / self.et_tau
                #     et = np.maximum(act_out, et + delta_et)
                #
                #     delta_ins_signal = (act_out - ins_signal) * self.track.dt / self.ins_signal_tau
                #     ins_signal = np.maximum(act_out, ins_signal + delta_ins_signal)
                # else:
                #     pos_input = 0
                #     et += -et * self.track.dt / self.et_tau
                #     ins_signal += -ins_signal * self.track.dt / self.ins_signal_tau
                #
                # if theta_phase > np.pi/2:
                #     rec_input = (self.w_exc + self.w_inh) @ act_out
                # else:
                #     rec_input = 0
                # # rec_input = 0
                #
                # act += (-act + pos_input + theta + rec_input) * self.dt_over_tau
                # act_out = self.f_act(act)
                #
                # et_overlap = et * ins_signal.reshape(-1, 1)

                theta = (-np.exp(self.theta_concentration * np.cos(theta_phase))
                         * self.theta_multiplier + self.theta_max)

                # compute sensory input
                pos_factor = (np.exp(self.pos_factor_concentration
                                     * np.cos(theta_phase - self.pos_factor_phase))
                              / exp_concentration)
                spatial_bin = int(self.track.x_log[t_step] / self.track.ds)
                pos_input = self.pos_factor_0 * pos_factor * self.sensory_fields[spatial_bin]

                # compute recurrent input
                rec_factor = 1 - pos_factor
                # rec_input = self.sigmoid(self.w_exc @ act_out, 0.5, 4) + self.w_inh * np.sum(act_out)
                rec_input = (self.w_exc + self.w_inh) @ act_out

                # update activity
                act += (-act + pos_input + theta + rec_factor * rec_input) * self.dt_over_tau
                # act += (-act + pos_input + theta) * self.dt_over_tau
                act_out = self.f_act(act)

                # update eligibility trace
                delta_et = (act_out * pos_factor - et) * self.track.dt / self.et_tau
                et = np.maximum(act_out * pos_factor, et + delta_et)

                # update instructive signals
                delta_ins_signal = (act_out * pos_factor - ins_signal) * self.track.dt / self.ins_signal_tau
                ins_signal = np.maximum(act_out * pos_factor, ins_signal + delta_ins_signal)

                # update weights
                et_overlap = et * ins_signal.reshape(-1, 1) * pos_factor
                # self.w_exc += learning_rate * ((self.w_exc_max - self.w_exc) * self.k_plus
                #                                * self.sigmoid(et_overlap, self.alpha_plus, self.beta_plus)
                #                                - self.w_exc * self.k_minus
                #                                * self.sigmoid(et_overlap, self.alpha_minus, self.beta_minus))

                max_overlaps = np.maximum(max_overlaps, et_overlap)

                # self.w_exc += learning_rate * et_overlap * pos_factor
                # # self.w_exc -= 2*learning_rate*learning_rate*self.w_exc
                # self.w_exc /= np.max(self.w_exc, axis=0)

                if i >= 0:
                    self.theta_phase_log[i] = theta_phase
                    if self.log_theta:
                        self.theta_log[i] = theta
                    self.act_out_log[i] = act_out
                    if self.log_act:
                        self.act_log[i] = act.copy()
                    if self.log_pos_input:
                        self.pos_input_log[i] = pos_input
                    if self.log_et:
                        self.et_log[i] = et

            self.w_exc += 0.1 * max_overlaps
            # self.w_exc /= np.max(self.w_exc, axis=0)
            self.w_exc /= np.max(np.maximum(self.w_exc, 0.8), axis=0)

    def plot_activities(self, t_start=0, t_end=None, first_unit=0, last_unit=None, apply_f=False, et=False,
                        pos_input=False, theta=False, speed=False, fig_size=(6.4, 4.8)):

        index_start = max(int(t_start / self.track.dt) - self.first_logged_step, 0)
        t_start = (index_start + self.first_logged_step) * self.track.dt
        index_end = min(int(t_end / self.track.dt) - self.first_logged_step, len(self.track.x_log)) \
            if t_end is not None else len(self.act_out_log)
        t_end = (index_end + self.first_logged_step) * self.track.dt

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

        extent = (t_start - self.track.dt / 2, t_end - self.track.dt / 2, first_unit - 0.5, last_unit - 0.5)

        rows = 2 + theta + speed
        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        height_ratios = [1, 1, 0.5] if theta else [1, 1]
        if speed:
            height_ratios.append(0.5)
        spec = fig.add_gridspec(rows, 2, height_ratios=height_ratios, width_ratios=[1, 0.03])

        ax0 = fig.add_subplot(spec[0:2, 0])
        mat = ax0.matshow(act_log.T, aspect="auto", origin="lower", extent=extent, cmap='Blues', vmin=v_min, vmax=v_max)

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

        green = colors.LinearSegmentedColormap.from_list('f', [(44 / 255, 160 / 255, 44 / 255, 0),
                                                               (44 / 255, 160 / 255, 44 / 255, 1)], N=100)

        if et:
            matb = ax0.matshow(self.et_log[index_start:index_end, first_unit:last_unit].T, aspect="auto",
                               origin="lower", extent=extent, cmap=green, vmin=0, vmax=1)
            color_bar = fig.colorbar(matb, cax=fig.add_subplot(spec[0, 1]))
            color_bar.set_label("Eligibility trace")
            color_bar.locator = ticker.MultipleLocator(0.5)
            color_bar.update_ticks()

        if pos_input:
            matb = ax0.matshow(self.pos_input_log[index_start:index_end, first_unit:last_unit].T, aspect="auto",
                               origin="lower", extent=extent, cmap=green, vmin=0, vmax=1)

            color_bar = fig.colorbar(matb, cax=fig.add_subplot(spec[0, 1]))
            color_bar.set_label("Spatial Input")
            color_bar.locator = ticker.MultipleLocator(0.5)
            color_bar.update_ticks()

        if theta:
            ax1 = fig.add_subplot(spec[2, 0], sharex=ax0)
            time = np.arange(t_start, t_end, self.track.dt)
            ax1.plot(time, self.theta_log[index_start:index_end])
            ax1.set_ylabel("Theta")
            ax1.set_xlabel("Time (s)")
            ax1.spines.right.set_visible(False)
            ax1.spines.top.set_visible(False)

        if speed:
            ax2 = fig.add_subplot(spec[2 + theta, 0], sharex=ax0)
            time = np.arange(t_start, t_end, self.track.dt)
            ax2.plot(time, self.track.speed_log[index_start:index_end], color='C7')
            ax2.set_ylabel("Speed (cm/s)")
            ax2.set_xlabel("Time (s)")
            ax2.spines.right.set_visible(False)
            ax2.spines.top.set_visible(False)

        ax0.xaxis.set_ticks_position('bottom')
        ax0.set_xlim(*extent)

        self.maybe_save_fig(fig, "activities", dpi=500)


if __name__ == "__main__":
    config = Config(identifier=1, variants={
        # 'LinearTrack': 'OneLap',
        'LinearTrack': 'TenLaps',
        'NetworkExtDriven': 'ExtDrivenLogAll'
    }, pickle_instances=True, save_figures=False, figures_root_path=figures_path, pickles_root_path=pickles_path,
                    figure_format='pdf')
    network = NetworkExtDriven.current_instance(config)
    # network.plot_sensory_fields()
    network.plot_rec_weights()
    # network.plot_q_functions()
    network.plot_activities(apply_f=1, speed=1, et=0)
    plt.show()
