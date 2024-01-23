#  -*- coding: utf-8 -*-

"""
    rhizodep.simulation
    ~~~~~~~~~~~~~

    The module :mod:`rhizodep.simulation` is the front-end to run the RhizoDep model.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import pandas as pd
import os
import time

import pickle
from rhizodep import Model

# We define the main simulation program:
def main_simulation(time_step_in_seconds=3600, simulation_period_in_days=20., **scenario):
    # Initializes the model and provide a scenario to superimpose default parameters
    rhizodep = Model(time_step=time_step_in_seconds, **scenario)

    # Print documentation if necessary to build a scenario
    #print(rhizodep.documentation)

    g = rhizodep.g

    # Computational loop
    steps = int(simulation_period_in_days / (time_step_in_seconds / 3600 / 24))
    for step in range(steps):
        rhizodep.run()
        print(step+1)

        # Here : MTG logging method

    # Here : save method
    with open(f"root{steps}.pckl", "wb") as (f):
        pickle.dump(g, f)

    # Here : display / analysis method
    #launch_analysis(g)


if __name__ == "__main__":
    scenarios = [{}]
    main_simulation(time_step_in_seconds=3600, **scenarios[0])
