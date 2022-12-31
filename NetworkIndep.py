import numpy as np
from AbstractNetwork import AbstractNetwork
from generic.smart_sim import Config
from batch_config import pickles_path, figures_path
from small_plots import *


TWO_PI = 2 * np.pi


class NetworkIndep(AbstractNetwork):
    """Network based on behavioral timescale synaptic plasticity and independent theta phase precession.
    """
    def __init__(self, num_units, num_inputs, et_tau, ins_signal_tau, c,
                 input_sigma, plateau_prob, w_max, k_plus, alpha_plus, beta_plus, k_minus, alpha_minus, beta_minus,
                 size_offset, size_slope, pp_trigger, pp_sigma, theta_concentration,
                 log_et_is=False, log_after=0, config=Config(), d={}):
        tau = 1
        log_act = False
        log_theta = False
        log_pos_input = False
        AbstractNetwork.__init__(self, num_units, tau, log_act, log_theta, log_pos_input, log_after, config, d)

        self.num_inputs = num_inputs
        self.et_tau = et_tau
        self.ins_signal_tau = ins_signal_tau
        self.log_et_is = log_et_is
        if self.log_et_is:
            self.et_log = np.empty((self.logged_steps, num_inputs))
            self.is_log = np.empty((self.logged_steps, num_units))

        self.c = c
        self.size_offset = size_offset
        self.size_slope = size_slope
        self.pp_trigger = pp_trigger
        self.pp_sigma = pp_sigma
        self.theta_concentration = theta_concentration
        self.theta_multiplier = 1 / np.exp(theta_concentration)

        sigma = input_sigma / self.track.ds
        centers = np.linspace(0, self.track.num_bins, num_inputs)
        self.inputs = np.exp(-(np.arange(self.track.num_bins).reshape(-1, 1) - centers) ** 2 / (2 * sigma ** 2))

        self.plateau_prob = plateau_prob * self.track.dt
        self.plateau_missing = np.ones(num_units, dtype=bool)

        self.w_max = w_max
        self.w = np.zeros((num_units, num_inputs))
        self.sorted_peak_indices = None

        self.k_plus = k_plus
        self.alpha_plus = alpha_plus
        self.beta_plus = beta_plus
        self.k_minus = k_minus
        self.alpha_minus = alpha_minus
        self.beta_minus = beta_minus

        self.induction_speeds = np.zeros(self.num_units)

        self.run(verbose=0)

    def q_plus(self, x):
        return self.k_plus * 1 / (1 + np.exp(-self.beta_plus * (x - self.alpha_plus)))

    def q_minus(self, x):
        return self.k_minus * 1 / (1 + np.exp(-self.beta_minus * (x - self.alpha_minus)))

    def plot_inputs(self, fig_size=(5.5*CM, 4.42*CM)):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        mat = ax.matshow(self.inputs, aspect='auto', origin='lower', cmap='Oranges',
                         extent=(0, self.track.length, -0.5, self.num_inputs - 0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel("Position (cm)")
        ax.set_xlabel("Input #")
        c_bar = plt.colorbar(mat, ax=ax)
        c_bar.set_label("Activation")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.maybe_save_fig(fig, "sensory inputs", dpi=500)

    def plot_q_functions(self, fig_size=(5.5*CM, 4.42*CM)):
        x = np.linspace(0, 1, 1000)
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        ax.plot(x, self.q_plus(x), label=r'$k^+ q^+ (ET \cdot IS)$')
        ax.plot(x, self.q_minus(x), label=r'$k^- q^- (ET \cdot IS)$')
        ax.set_xlabel(r"$ET \cdot IS$")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.legend()

    def run(self, verbose=0):
        theta_phase = 0
        pp_slopes = np.zeros(self.num_units)
        gauss_denom = 2 * self.pp_sigma**2

        for lap, lap_start_step in enumerate(self.track.lap_start_steps):
            # reset at the beginning of each lap
            et = np.zeros(self.num_inputs)
            ins_signal = np.zeros(self.num_units)
            pref_phases = np.ones(self.num_units) * TWO_PI

            if verbose:
                print(f"running lap {lap}, {np.sum(self.plateau_missing)} plateau potentials missing")

            if lap + 1 < len(self.track.lap_start_steps):
                last_lap_step = self.track.lap_start_steps[lap + 1]
            else:
                last_lap_step = len(self.track.x_log)

            for t_step in range(lap_start_step, last_lap_step):
                i = t_step - self.first_logged_step

                # compute theta phase
                theta_phase += self.theta_phase_inc
                if theta_phase > TWO_PI:
                    theta_phase -= TWO_PI
                    if i >= 0:
                        self.theta_cycle_starts.append(i)
                theta = -np.exp(self.theta_concentration * np.cos(theta_phase)) * self.theta_multiplier + 1

                # get spatial inputs and update et
                inputs = self.inputs[int(self.track.x_log[t_step] / self.track.ds)]
                delta_et = (inputs - et) * self.track.dt / self.et_tau
                et = np.maximum(inputs, et + delta_et)

                # update activities
                act = self.c * (self.w @ inputs)
                in_field = act > self.pp_trigger
                pref_phases[in_field] -= pp_slopes[in_field] * self.track.speed_log[t_step] * self.track.dt
                pref_phases = np.maximum(0, pref_phases)
                act_out = act * np.exp(-(theta_phase - pref_phases)**2 / gauss_denom) * theta

                # generate plateau potentials
                if np.sum(self.plateau_missing):
                    plateau_prob = self.plateau_prob * self.plateau_missing
                    plateaus = np.where(np.random.random(self.num_units) < plateau_prob, 1, 0)
                    if np.sum(plateaus):
                        speed = self.track.speed_log[t_step]
                        self.induction_speeds[plateaus.astype(bool)] = speed
                        pp_slopes[plateaus.astype(bool)] = TWO_PI / (self.size_offset + self.size_slope * speed)
                        self.plateau_missing = np.maximum(self.plateau_missing - plateaus, 0)
                else:
                    plateaus = 0

                # generate instructive signals
                ins_signal_decay = ins_signal - ins_signal * self.track.dt / self.ins_signal_tau
                ins_signal = np.maximum(plateaus, ins_signal_decay)

                # weight update
                if np.max(ins_signal) > 0.01:
                    overlap = et * ins_signal.reshape(-1, 1)
                    self.w += self.track.dt * ((self.w_max - self.w) * self.q_plus(overlap)
                                               - self.w * self.q_minus(overlap))

                # keep track of things
                if i >= 0:
                    self.theta_phase_log[i] = theta_phase
                    self.act_out_log[i] = act_out
                    if self.log_et_is:
                        self.et_log[i] = et
                        self.is_log[i] = ins_signal

        self.sorted_peak_indices = np.argsort(np.argmax(self.w, axis=1))
        self.act_out_log = self.act_out_log[:, self.sorted_peak_indices]
        self.induction_speeds = self.induction_speeds[self.sorted_peak_indices]

    def plot_weights(self, fig_size=(5.5*CM, 4.42*CM)):
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        mat = ax.matshow(self.w[self.sorted_peak_indices], aspect='auto', origin='lower', cmap='Greens')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Input #')
        ax.set_ylabel('Place cell # (sorted by peak)')
        c_bar = plt.colorbar(mat, ax=ax)
        c_bar.set_label(r"$W_s$")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.maybe_save_fig(fig, "Weights", dpi=500)

    def plot_activities(self, t_start=0, t_end=None, first_unit=0, last_unit=None, speed=False, fig_size=(6.4, 4.8)):

        index_start = max(int(t_start / self.track.dt) - self.first_logged_step, 0)
        t_start = (index_start + self.first_logged_step) * self.track.dt
        index_end = min(int(t_end / self.track.dt) - self.first_logged_step, len(self.track.x_log)) \
            if t_end is not None else len(self.act_out_log)
        t_end = (index_end + self.first_logged_step) * self.track.dt

        if last_unit is None:
            last_unit = self.num_units

        act_log = self.act_out_log[index_start:index_end, first_unit:last_unit]

        gridspec_kw = {'width_ratios': (1, 0.03)}
        if speed:
            gridspec_kw |= {'height_ratios': (1, 0.2)}

        fig, ax = plt.subplots(1+speed, 2, figsize=fig_size, gridspec_kw=gridspec_kw, sharex='col', squeeze=False)
        ax0 = ax[0, 0]
        
        extent = (t_start - self.track.dt / 2, t_end - self.track.dt / 2, first_unit - 0.5, last_unit - 0.5)
        mat = ax0.matshow(act_log.T, aspect="auto", origin="lower", extent=extent, cmap='Blues')

        ax0.xaxis.set_ticks_position('bottom')
        ax0.set_title("Network activities")
        ax0.set_ylabel("Place cell #")
        ax0.spines.right.set_visible(False)
        ax0.spines.top.set_visible(False)
        if not speed:
            ax0.set_xlabel("Time (s)")

        color_bar = plt.colorbar(mat, cax=ax[0, 1])
        color_bar.set_label("Activation")

        if speed:
            time = np.arange(t_start, t_end, self.track.dt)
            ax[1, 0].plot(time, self.track.speed_log[index_start:index_end], color='C7')
            ax[1, 0].set_ylabel("Speed (cm/s)")
            ax[1, 0].set_xlabel("Time (s)")
            ax[1, 0].spines.right.set_visible(False)
            ax[1, 0].spines.top.set_visible(False)

        ax[1, 1].axis('off')
        self.maybe_save_fig(fig, "activities", dpi=500)

    def plot_learning_traces(self, input_unit, output_unit, before, after, fig_size=(7*CM, 4*CM)):
        i = np.argmax(self.is_log[:, output_unit])
        i_start = max(0, int(i - before/self.track.dt))
        i_end = min(self.logged_steps, int(i + after / self.track.dt))
        time = np.arange(i_start, i_end)*self.track.dt
        fig, ax = plt.subplots(figsize=fig_size)
        et = self.et_log[i_start:i_end, input_unit]
        ins_signal = self.is_log[i_start:i_end, output_unit]
        ax.plot(time, et, label=rf"$e_{{{input_unit}}}$")
        ax.plot(time, ins_signal, label=rf"$s_{{{output_unit}}}$")
        ax.plot(time, et * ins_signal, label=rf"$o_{{{output_unit, input_unit}}}$")
        ax.set_xlabel("Time (s)")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.legend()
        self.maybe_save_fig(fig, "activities", dpi=500)


if __name__ == "__main__":

    config = Config(identifier=1, variants={
        # 'LinearTrack': 'OneLap',
        # 'LinearTrack': 'TenLaps',
        # 'NetworkIndep': 'IndepLogAll',
        'NetworkIndep': 'IndepLog80'
    },
                    pickle_instances=True, save_figures=True, figures_root_path=figures_path,
                    pickles_root_path=pickles_path, figure_format='pdf')

    network = NetworkIndep.current_instance(config)
    # network.plot_inputs()
    # network.plot_q_functions()
    network.plot_weights(fig_size=(5.5*CM, 4.33*CM))

    # network.plot_activities(speed=1)
    # network.plot_learning_traces(input_unit=70, output_unit=20, before=1, after=1, fig_size=(5.5*CM, 4.42*CM))
    plt.show()
