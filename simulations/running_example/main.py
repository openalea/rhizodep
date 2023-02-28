# Importation of functions from the system:
###########################################

import numpy as np
import os
import timeit
from rhizodep.simulation import *

# # Setting the randomness in the whole code to reproduce the same root system over different runs:
# # random_choice = int(round(np.random.normal(100,50)))
# random_choice = 8
# param.__dict__.update({'random_choice': random_choice})  # update random_choice in the parameters file
# print("The random seed used for this run is", random_choice)
# np.random.seed(random_choice)

# RUNNING THE SIMULATION:
#########################

if __name__ == "__main__":

    # -- OUTPUTS DIRECTORY OF THE SCENARIO --
    # We define the path of the directory that will contain the outputs of the model:
    outputs_dirpath = 'outputs'
    if not os.path.exists(outputs_dirpath):
        # We create it:
        os.mkdir(outputs_dirpath)
    else:
        # Otherwise, we delete all the files that are already present inside:
        for root, dirs, files in os.walk(outputs_dirpath):
            for file in files:
                os.remove(os.path.join(root, file))

    # We define the specific directory in which the images of the root systems may be recorded:
    images_dirpath = os.path.join(outputs_dirpath, "root_images")
    if not os.path.exists(images_dirpath):
        os.mkdir(images_dirpath)
    else:
        # Otherwise, we delete all the files that are already present inside:
        for root, dirs, files in os.walk(images_dirpath):
            for file in files:
                os.remove(os.path.join(root, file))

    # We define the specific directory in which the MTG files may be recorded:
    MTG_files_dirpath = os.path.join(outputs_dirpath, "MTG_files")
    if not os.path.exists(MTG_files_dirpath):
        os.mkdir(MTG_files_dirpath)
    else:
        # Otherwise, we delete all the files that are already present inside:
        for root, dirs, files in os.walk(MTG_files_dirpath):
            for file in files:
                os.remove(os.path.join(root, file))

    # We define the specific directory in which the MTG properties may be recorded:
    MTG_properties_dirpath = os.path.join(outputs_dirpath, "MTG_properties")
    if not os.path.exists(MTG_properties_dirpath):
        os.mkdir(MTG_properties_dirpath)
    else:
        # Otherwise, we delete all the files that are already present inside:
        for root, dirs, files in os.walk(MTG_properties_dirpath):
            for file in files:
                os.remove(os.path.join(root, file))

    # We record the time when the run starts:
    start_time = timeit.default_timer()

    # We initiate the properties of the MTG "g":
    g = model.initiate_mtg(random=True)

    # We launch the main simulation program:
    print("Simulation starts ...")
    main_simulation(g, simulation_period_in_days=20., time_step_in_days=1.,
                    radial_growth="Impossible", ArchiSimple=False, ArchiSimple_C_fraction=0.10,
                    input_file="None",
                    outputs_directory=outputs_dirpath,
                    forcing_constant_inputs=True,
                    constant_sucrose_input_rate=1.e-6,
                    constant_soil_temperature_in_Celsius=20,
                    nodules=False,
                    specific_model_option=None,
                    simulation_results_file='simulation_results.csv',
                    recording_interval_in_days=5,
                    recording_images=False,
                    root_images_directory=images_dirpath,
                    z_classification=False, z_min=0., z_max=1., z_interval=0.5,
                    z_classification_file='z_classification.csv',
                    printing_sum=True,
                    recording_sum=True,
                    printing_warnings=False,
                    recording_g=False,
                    g_directory=MTG_files_dirpath,
                    recording_g_properties=False,
                    g_properties_directory=MTG_properties_dirpath,
                    random=True,
                    plotting=True,
                    scenario_id=1,
                    displayed_property="C_hexose_root", displayed_vmin=1e-6, displayed_vmax=1e-0,
                    log_scale=True, cmap='brg',
                    x_center=0, y_center=0, z_center=-1, z_cam=-1,
                    camera_distance=10., step_back_coefficient=0., camera_rotation=False, n_rotation_points=24 * 5)

    print("")
    print("***************************************************************")
    end_time = timeit.default_timer()
    print("Run is done! The system took", round(end_time - start_time, 1), "seconds to complete the run.")

    # # To avoid closing PlantGL as soon as the run is done:
    # input()
