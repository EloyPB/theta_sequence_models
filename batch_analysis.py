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
    # pf.plot_activations(fig_size=(4, 4))
    pf.sizes_vs_mean_speed()
    pf.slow_and_fast_sizes()
    # pf.density_vs_mean_speed()
    del pf

    # pp = PhasePrecession.current_instance(config)
    # pp.slopes_vs_mean_speed()
    # del pp

    # sweeps = ThetaSweeps.current_instance(config)
    # sweeps.length_vs_mean_speed()


# RUN ANALYSES

with Pool(processes=2) as pool:
    pool.map(analyze, range(NUM_RUNS))



