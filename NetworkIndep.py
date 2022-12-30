import numpy as np
import matplotlib.ticker as ticker
from AbstractNetwork import AbstractNetwork
from generic.smart_sim import Config
from batch_config import pickles_path, figures_path
from small_plots import *


TWO_PI = 2 * np.pi


class NetworkIndep(AbstractNetwork):
    """Network based on behavioral timescale synaptic plasticity and independent theta phase precession.
    """
    def __init__(self, num_units, num_inputs, et_tau, ins_signal_tau, act_sigmoid_gain, act_sigmoid_midpoint,
                 input_sigma, plateau_prob, log_act=False, log_et_is=False, log_after=0, config=Config(), d={}):
        tau = 1
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

        self.act_sigmoid_gain = act_sigmoid_gain
        self.act_sigmoid_midpoint = act_sigmoid_midpoint
        self.inputs = self.gaussian_inputs(sigma=input_sigma / self.track.ds)
        self.act_baseline = np.mean(np.sum(self.inputs, axis=1))
        self.plateau_prob = plateau_prob * self.track.dt
        self.plateau_missing = np.ones(num_units, dtype=bool)

        self.w = np.ones((num_units, num_inputs))

        self.run()

    def gaussian_inputs(self, sigma):
        separation = self.track.length / self.num_inputs
        centers = np.arange(separation / 2, self.track.length, separation).reshape(-1, 1)
        x = np.arange(self.track.ds / 2, self.track.length, self.track.ds)
        distances = np.min(np.stack((np.abs(centers - x),
                                     np.abs(centers - x + self.track.length),
                                     np.abs(centers - x - self.track.length))), axis=0)
        return np.exp(-distances ** 2 / (2 * sigma ** 2)).T

    def plot_inputs(self):
        fig, ax = plt.subplots()
        mat = ax.matshow(self.inputs.T, aspect='auto', origin='lower',
                         extent=(0, self.track.length, -0.5, self.num_inputs - 0.5))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Input #")
        c_bar = plt.colorbar(mat, ax=ax)
        c_bar.set_label("Activation")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.maybe_save_fig(fig, "sensory inputs", dpi=500)

    def run(self, verbose=1):
        theta_phase = 0

        for lap, lap_start_step in enumerate(self.track.lap_start_steps):
            # reset at the beginning of each lap
            et = np.zeros(self.num_inputs)
            ins_signal = np.zeros(self.num_units)

            if verbose:
                print(f"running lap {lap}, {np.sum(self.plateau_missing)} plateau potentials missing")

            if lap + 1 < len(self.track.lap_start_steps):
                last_lap_step = self.track.lap_start_steps[lap + 1]
            else:
                last_lap_step = len(self.track.x_log)

            for t_step in range(lap_start_step, last_lap_step):

                # compute theta phase
                theta_phase += self.theta_phase_inc
                if theta_phase > TWO_PI:
                    theta_phase -= TWO_PI

                # get spatial inputs and update et
                inputs = self.inputs[int(self.track.x_log[t_step] / self.track.ds)]
                delta_et = (inputs - et) * self.track.dt / self.et_tau
                et = np.maximum(inputs, et + delta_et)

                # update activities
                act = self.w @ inputs - self.act_baseline
                act_out = self.f_act(act)

                # generate plateau potentials
                if np.sum(self.plateau_missing):
                    plateau_prob = self.plateau_prob * self.plateau_missing
                    plateaus = np.where(np.random.random(self.num_units) < plateau_prob, 1, 0)
                    self.plateau_missing = np.maximum(self.plateau_missing - plateaus, 0)
                else:
                    plateaus = 0

                # generate instructive signals
                ins_signal_decay = - ins_signal * self.track.dt / self.ins_signal_tau
                ins_signal = np.maximum(plateaus, ins_signal_decay)

                # weight update

                # keep track of things
                i = t_step - self.first_logged_step
                if i >= 0:
                    self.theta_phase_log[i] = theta_phase
                    self.act_out_log[i] = act_out
                    if self.log_act:
                        self.act_log[i] = act
                    if self.log_et_is:
                        self.et_log[i] = et

    def f_act(self, x):
        return 1 / (1 + np.exp(-self.act_sigmoid_gain * (x - self.act_sigmoid_midpoint)))

    def plot_activities(self, t_start=0, t_end=None, first_unit=0, last_unit=None, apply_f=False,
                        speed=False, fig_size=(6.4, 4.8)):

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

        gridspec_kw = {'width_ratios': (1, 0.03)}
        if speed:
            gridspec_kw |= {'height_ratios': (1, 0.2)}

        fig, ax = plt.subplots(1+speed, 2, figsize=fig_size, gridspec_kw=gridspec_kw, sharex='col', squeeze=False)
        ax0 = ax[0, 0]
        
        extent = (t_start - self.track.dt / 2, t_end - self.track.dt / 2, first_unit - 0.5, last_unit - 0.5)
        mat = ax0.matshow(act_log.T, aspect="auto", origin="lower", extent=extent, cmap='Blues', vmin=v_min, vmax=v_max)

        ax0.xaxis.set_ticks_position('bottom')
        ax0.set_title("Network activities")
        ax0.set_ylabel("Place cell #")
        ax0.spines.right.set_visible(False)
        ax0.spines.top.set_visible(False)
        if not speed:
            ax0.set_xlabel("Time (s)")

        color_bar = plt.colorbar(mat, cax=ax[0, 1])
        color_bar.set_label("Activation")
        # color_bar.locator = ticker.MultipleLocator(0.5)
        # color_bar.update_ticks()

        if speed:
            time = np.arange(t_start, t_end, self.track.dt)
            ax[1, 0].plot(time, self.track.speed_log[index_start:index_end], color='C7')
            ax[1, 0].set_ylabel("Speed (cm/s)")
            ax[1, 0].set_xlabel("Time (s)")
            ax[1, 0].spines.right.set_visible(False)
            ax[1, 0].spines.top.set_visible(False)

        ax[1, 1].axis('off')

        self.maybe_save_fig(fig, "activities", dpi=500)

    def plot_learning_traces(self, input_unit, output_unit, fig_size=(7*CM, 4*CM)):
        ...


if __name__ == "__main__":

    config = Config(identifier=1, variants={
        # 'LinearTrack': 'OneLap',
        'LinearTrack': 'TenLaps',
        'NetworkIndep': 'LogAll'},
                    pickle_instances=True, save_figures=False, figures_root_path=figures_path,
                    pickles_root_path=pickles_path, figure_format='pdf')

    network = NetworkIndep.current_instance(config)
    network.plot_inputs()
    network.plot_activities(speed=1)
    plt.show()
