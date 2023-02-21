from multiprocessing import Pool
from generic.smart_sim import Config
from batch_config import *
from PlaceFields import PlaceFields
from PhasePrecession import PhasePrecession
from ThetaSweeps import ThetaSweeps
from generic.timer import timer


# DEFINE ANALYSIS PIPELINE

def analyze(identifier):
    config = Config(identifier, variants, parameters_path='parameters', save_figures=True, figure_format='pdf',
                    figures_root_path=figures_path, pickle_instances=True, pickle_results=True,
                    pickles_root_path=pickles_path)

    pf = PlaceFields.current_instance(config)
    pf.plot_activations(fig_size=(4, 4))
    pf.sizes_vs_mean_speed(plot=True)
    pf.slow_and_fast_sizes()
    pf.density_vs_mean_speed()
    pf.field_peak_shifts(plot=True)
    # pf.size_vs_induction_speed()  # only for NetworkIndep
    del pf

    pp = PhasePrecession.current_instance(config)
    pp.slopes_vs_mean_speed(plot=True)
    pp.fast_and_slow_slopes()
    del pp

    sweeps = ThetaSweeps.current_instance(config)
    sweeps.length_vs_mean_speed(plot=True)
    sweeps.ahead_and_behind_vs_mean_speed(plot=True)
    sweeps.behind_length_vs_peak_shift(plot=True)


# RUN ANALYSES

with Pool(processes=2) as pool:
    pool.map(analyze, range(NUM_RUNS))

