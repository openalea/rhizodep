"""
    This script allows to run RhizoDep over over default parameters using the function 'running_simulation'.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import os
from openalea.rhizodep import model, running_simulation

########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == '__main__':
    # (Note: this condition avoids launching automatically the program when imported in another file)

    g = model.initiate_mtg(random=True)
    running_simulation.main_simulation(g,
                                       simulation_period_in_days=10., time_step_in_days=1./24.,
                                       forcing_constant_inputs=True,
                                       constant_sucrose_input_rate=1.e-6,
                                       constant_soil_temperature_in_Celsius=20,
                                       outputs_directory='outputs',
                                       recording_g=True,
                                       recording_g_properties=True,
                                       recording_images=True
                                       )