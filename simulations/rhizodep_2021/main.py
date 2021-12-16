# Todo: Introduce explicit threshold concentration for limiting processes, rather than "0"
# Todo: Consider giving priority to root maintenance, and trigger senescence when root maintenance is not ensured
# Todo: Watch the calculation of surface and volume for the apices - if they correspond to cones, the mass balance for segmentation may not be correct!
# Todo: Include cell sloughing and mucilage production
# Todo: Make a distinction between seminal roots and adventitious roots
# Todo: Check actual_elongation_rate with thermal time...

# Importation of functions from the system:
###########################################

import numpy as np
import timeit

import rhizodep.parameters as param
from rhizodep.simulation import *

import pickle

# Setting the randomness in the whole code to reproduce the same root system over different runs:
# random_choice = int(round(np.random.normal(100,50)))
random_choice = 8
param.__dict__.update({'random_choice': random_choice})  # update random_choice in the parameters file
print("The random seed used for this run is", random_choice)
np.random.seed(random_choice)


# RUNNING THE SIMULATION:
#########################

if __name__ == "__main__":

    # Create an outputs directory if does not exist
    outputs_dirpath = 'outputs'
    if not os.path.exists(outputs_dirpath):
        os.mkdir(outputs_dirpath)

    # We record the time when the run starts:
    start_time = timeit.default_timer()

    # We initiate the properties of the MTG "g":
    g = model.initiate_mtg(random=True)

    # We launch the main simulation program:
    print("Simulation starts ...")
    main_simulation(g, simulation_period_in_days=20., time_step_in_days=1. / 24., radial_growth="Possible",
                    ArchiSimple=False,
                    # property="net_hexose_exudation_rate_per_day_per_cm", vmin=1e-9, vmax=1e-6, log_scale=True, cmap='jet',
                    property="C_hexose_root", vmin=1e-4, vmax=1e-1, log_scale=True, cmap='jet',
                    # property="C_sucrose_root", vmin=1e-4, vmax=1e2, log_scale=True, cmap='brg',
                    # property="C_hexose_reserve", vmin=1e-4, vmax=1e4, log_scale=True, cmap='brg',
                    input_file=os.path.join("inputs", "sucrose_input_0047.csv"),
                    constant_sucrose_input_rate=5e-9,
                    constant_soil_temperature_in_Celsius=20,
                    nodules=False,
                    simulation_results_file=os.path.join(outputs_dirpath, 'simulation_results.csv'),
                    x_center=0, y_center=0, z_center=-1, z_cam=-2,
                    camera_distance=4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                    z_classification=False, z_min=0.00, z_max=1., z_interval=0.05,
                    recording_images=True,
                    printing_sum=False,
                    recording_sum=True,
                    printing_warnings=False,
                    recording_g=True,
                    recording_g_properties=False,
                    random=True)

    print("")
    print("***************************************************************")
    end_time = timeit.default_timer()
    print("Run is done! The system took", round(end_time - start_time, 1), "seconds to complete the run.")

    # We save the final MTG:
    with open('g_file.pckl', 'wb') as output:
        pickle.dump(g, output, protocol=2)

    print("The whole root system has been saved in the file 'g_file.pckl'.")

    # To avoid closing PlantGL as soon as the run is done:
    input()
