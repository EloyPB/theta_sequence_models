import numpy as np
from LinearTrack import LinearTrack
from generic.smart_sim import Config, SmartSim
from batch_config import *


class AbstractNetwork(SmartSim):
    dependencies = [LinearTrack]

    def __init__(self, num_units, tau, log_act=False, log_theta=False, log_pos_input=False, log_after=0,
                 config=Config(), d={}):

        SmartSim.__init__(self, config, d)

        if 'LinearTrack' in d:
            self.track: LinearTrack = d['LinearTrack']
        else:
            sys.exit("A LinearTrack instance should be provided in d")

        self.num_units = num_units
        self.tau = tau
        self.dt_over_tau = self.track.dt / tau
        self.first_logged_step = int(log_after / self.track.dt)
        self.logged_steps = len(self.track.x_log) - self.first_logged_step

        self.log_act = log_act
        if log_act:
            self.act_log = np.empty((self.logged_steps, num_units))
        self.act_out_log = np.empty((self.logged_steps, num_units))

        self.theta_cycle_steps = 1 / (8 * self.track.dt)
        self.theta_phase_inc = 2 * np.pi / self.theta_cycle_steps

        self.log_theta = log_theta
        if log_theta:
            self.theta_log = np.empty(self.logged_steps)
        self.theta_phase_log = np.empty(self.logged_steps)
        self.theta_phase_log[-1] = 0
        self.theta_cycle_starts = []

        self.log_pos_input = log_pos_input
        if log_pos_input:
            self.pos_input_log = np.empty((self.logged_steps, num_units))