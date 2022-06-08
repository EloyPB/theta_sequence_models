from multiprocessing import Pool
from generic.smart_sim import Config
from batch_config import *
from PhasePrecession import PhasePrecession
from generic.timer import timer


# DEFINE ANALYSIS PIPELINE

def analyze(identifier):
    config = Config(identifier, variants, parameters_path='parameters', save_figures=True, figure_format='png',
                    figures_root_path=figures_path, pickle_instances=True, pickle_results=True,
                    pickles_root_path=pickles_path)

    pp = PhasePrecession.current_instance(config)
    # pp.fields.sizes_vs_mean_speed(colour_by_position=True)
    pp.fields.density_vs_mean_speed()
    # pp.slopes_vs_mean_speed()


# RUN ANALYSES

with Pool(processes=2) as pool:
    pool.map(analyze, range(NUM_RUNS))



