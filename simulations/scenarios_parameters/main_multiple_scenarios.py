# -*- coding: latin-1 -*-

"""
    This script launches in parallel multiple calls to the function run_one_scenario from main_one_scenario.py.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import multiprocessing as mp
import os
import time

import pandas as pd
from simulations.scenarios_parameters.main_one_scenario import run_one_scenario


scenarios_df = pd.read_csv(os.path.join('inputs', 'scenarios_list.csv'), index_col='Scenario')
scenarios_df['Scenario'] = scenarios_df.index
scenarios = scenarios_df.Scenario

if __name__ == '__main__':
    tstart = time.time()
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)

    mp_solutions = p.map(run_one_scenario, list(scenarios))
    p.terminate()
    p.join()

    tend = time.time()
    tmp = (tend - tstart) / 60.
    print("multiprocessing: %8.3f minutes" % tmp)
