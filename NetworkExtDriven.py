import sys
import numpy as np
import matplotlib.pyplot as plt
from AbstractNetwork import AbstractNetwork
from LinearTrack import LinearTrack
from generic.smart_sim import Config, SmartSim


class NetworkExtDriven(AbstractNetwork):
    dependencies = [LinearTrack]

    def __init__(self, num_units, tau, act_sigmoid_gain, act_sigmoid_midpoint, theta_min, theta_max,
                 theta_concentration, pos_factor_0, pos_factor_concentration, pos_factor_phase,
                 learning_rate, log_act=False, log_theta=False, log_pos_input=False,
                 log_after=0, config=Config(), d={}):

        AbstractNetwork.__init__(self, num_units, tau, log_act, log_theta, log_pos_input, log_after, config, d)

