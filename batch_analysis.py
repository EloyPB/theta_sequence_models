from multiprocessing import Pool
from generic.smart_sim import Config
from batch_config import *
from PhasePrecession import PhasePrecession
from ThetaSweeps import ThetaSweeps
from generic.timer import timer


# DEFINE ANALYSIS PIPELINE

def analyze(identifier):
    config = Config(identifier, variants, parameters_path='parameters', save_figures=True, figure_format='pdf',
                    figures_root_path=figures_path, pickle_instances=True, pickle_results=True,
                    pickles_root_path=pickles_path)

    pp = PhasePrecession.current_instance(config)
    pp.fields.plot_activations(fig_size=(4, 4))
    # pp.fields.sizes_vs_mean_speed()
    # pp.fields.density_vs_mean_speed()
    # pp.fields.separation_vs_mean_speed()
    # pp.slopes_vs_mean_speed()
    del pp

    # sweeps = ThetaSweeps.current_instance(config)
    # sweeps.length_vs_mean_speed()


# RUN ANALYSES

with Pool(processes=2) as pool:
    pool.map(analyze, range(NUM_RUNS))



