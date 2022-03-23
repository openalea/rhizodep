# -*- coding: latin-1 -*-

"""
    This script allows to run a simulation of RhizoDep using a specific set of arguments and parameters, which are given in an input csv file.
    See the function run_one_scenario.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import sys
import getopt
import pandas as pd
import multiprocessing as mp
import shutil
import time

import rhizodep.tools as tools
import rhizodep.simulation as simulation
import rhizodep.parameters as param
import rhizodep.model as model


def run_one_scenario(scenario_id=1, inputs_dir_path=None, outputs_dir_path='outputs'):
    """
    Run main_simulation() of simulation.py using parameters of a specific scenario

    :param int scenario_id: the index of the scenario to be read in the CSV file containing the list of scenarios
    :param str inputs_dir_path: the path directory of inputs
    :param str outputs_dir_path: the path to save outputs
    """
    # -- OUTPUTS DIRECTORY OF THE SCENARIO --
    # We define the path of the directory that will contain the outputs of the model:
    if not outputs_dir_path:
        OUTPUTS_DIRPATH = outputs_dir_path
    else:
        OUTPUTS_DIRPATH = 'outputs'
    if not os.path.exists(OUTPUTS_DIRPATH):
        # We create it:
        os.mkdir(OUTPUTS_DIRPATH)

    # We define the path of the directory that contains the inputs of the model:
    if inputs_dir_path:
        INPUTS_DIRPATH = inputs_dir_path
    else:
        INPUTS_DIRPATH = 'inputs'

    # We read the scenario to be run:
    scenarios_df = pd.read_csv(os.path.join(INPUTS_DIRPATH, 'scenarios_list.csv'), index_col='Scenario')
    scenario = scenarios_df.loc[scenario_id].to_dict()
    scenario_name = 'Scenario_%.4d' % scenario_id

    # We define the specific directory in which the outputs of this scenario will be recorded:
    scenario_dirpath = os.path.join(OUTPUTS_DIRPATH, scenario_name)
    # If the output directory doesn't exist:
    if not os.path.exists(scenario_dirpath):
        # We create it:
        os.mkdir(scenario_dirpath)
    else:
        # Otherwise, we delete all the files that are already present inside:
        for root, dirs, files in os.walk(scenario_dirpath):
            for file in files:
                os.remove(os.path.join(root, file))
    # # We create the csv file that will contain the results corresponding to this scenario:
    # SCENARIO_OUTPUT_FILE = os.path.join(scenario_dirpath, 'simulation_results.csv')
    # Z_CLASSIFICATION_FILE = os.path.join(scenario_dirpath, 'z_classification.csv')
    # We define the specific directory in which the images of the root systems may be recorded:
    images_dirpath = os.path.join(scenario_dirpath, "root_images")
    if not os.path.exists(images_dirpath):
        os.mkdir(images_dirpath)
    # We define the specific directory in which the MTG files may be recorded:
    MTG_files_dirpath = os.path.join(scenario_dirpath, "MTG_files")
    if not os.path.exists(MTG_files_dirpath):
        os.mkdir(MTG_files_dirpath)
    # We define the specific directory in which the MTG properties may be recorded:
    MTG_properties_dirpath = os.path.join(scenario_dirpath, "MTG_properties")
    if not os.path.exists(MTG_properties_dirpath):
        os.mkdir(MTG_properties_dirpath)

    # -- SIMULATION PARAMETERS --
    # We create a dictionary containing the parameters specified in the scenario:
    scenario_parameters = tools.buildDic(scenario)
    # We update the parameters of the model with the parameters indicated in the scenario:
    param.__dict__.update(scenario_parameters)
    # OPTIONAL: We can print a CSV file containing the updated parameters:
    data_frame_parameters = pd.DataFrame.from_dict(param.__dict__)
    useless_columns = ["__name__","__doc__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"]
    data_frame_parameters.drop(useless_columns, axis=1, inplace=True)
    data_frame_parameters.head(1).to_csv(os.path.join(scenario_dirpath, 'updated_parameters.csv'),
                                 na_rep='NA', index=False, header=True)

    # We read the input file containing data on time, temperature and sucrose input
    # (Note: The following function "get" looks if the first argument is present in the scenario parameters.
    # If not, it returns the default value indicated in the second argument)
    SCENARIO_INPUT_FILENAME = scenario_parameters.get('input_file', None)
    if SCENARIO_INPUT_FILENAME:
        SCENARIO_INPUT_FILE = os.path.join(INPUTS_DIRPATH, SCENARIO_INPUT_FILENAME)
    else:
        SCENARIO_INPUT_FILE = None
    # We read the other specific instructions of the scenario that don't correspond to the parameters in the list:
    SIMULATION_PERIOD = scenario_parameters.get('simulation_period_in_days', 10.)
    TIME_STEP = scenario_parameters.get('time_step_in_days', 1. / 24.)
    RADIAL_GROWTH = scenario_parameters.get('radial_growth', True)
    ARCHISIMPLE_OPTION = scenario_parameters.get('ArchiSimple', False)
    ARCHISIMPLE_C_FRACTION = scenario_parameters.get('ArchiSimple_C_fraction', 0.20)
    NODULES_OPTION = scenario_parameters.get('nodules_option', False)
    ROOT_ORDER_LIMITATION_OPTION = scenario_parameters.get('root_order_limitation', False)
    ROOT_ORDER_TRESHOLD = scenario_parameters.get('root_order_treshold', 2)
    SPECIFIC_MODEL_OPTION = scenario_parameters.get('specific_model_option', None)
    FORCING_INPUTS = scenario_parameters.get('forcing_constant_inputs', False)
    SUCROSE_INPUT_RATE = scenario_parameters.get('constant_sucrose_input_rate', 5e-9)
    SOIL_TEMPERATURE = scenario_parameters.get('constant_soil_temperature_in_Celsius', 20)

    PLOTTING = scenario_parameters.get('plotting', True)
    DISPLAYED_PROPERTY = scenario_parameters.get('displayed_property', 'C_hexose_root')
    DISPLAYED_MIN_VALUE = scenario_parameters.get('displayed_min_value', 1e-15)
    DISPLAYED_MAX_VALUE = scenario_parameters.get('displayed_max_value', 1000)
    LOG_SCALE = scenario_parameters.get('log_scale', True)
    COLOR_MAP = scenario_parameters.get('color_map', 'jet')
    CAMERA_ROTATION_OPTION = scenario_parameters.get('camera_rotation', False)
    CAMERA_ROTATION_N_POINTS = scenario_parameters.get('camera_rotation_n_points', 12 * 10)
    X_CENTER = scenario_parameters.get('x_center', 0)
    Y_CENTER = scenario_parameters.get('y_center', 0)
    Z_CENTER = scenario_parameters.get('z_center', -1)
    Z_CAMERA = scenario_parameters.get('z_camera', -2)
    CAMERA_DISTANCE = scenario_parameters.get('camera_distance', 4)
    STEP_BACK_COEFFICIENT = scenario_parameters.get('step_back_coefficient', 0)

    CLASSIFICATION_BY_LAYERS = scenario_parameters.get('classification_by_layers', False)
    LAYERS_Z_MIN = scenario_parameters.get('layers_z_min', 0.0)
    LAYERS_Z_MAX = scenario_parameters.get('layers_z_max', 1.0)
    LAYERS_THICKNESS = scenario_parameters.get('layers_thickness', 0.1)

    RECORDING_IMAGES_OPTION = scenario_parameters.get('recording_images', False)
    PRINTING_SUM_OPTION = scenario_parameters.get('printing_all_properties', False)
    RECORDING_SUM_OPTION = scenario_parameters.get('recording_properties', True)
    RECORDING_INTERVAL_IN_DAYS = scenario_parameters.get('recording_interval_in_days', 10)
    PRINTING_WARNINGS_OPTION = scenario_parameters.get('printing_warnings', False)
    RECORDING_MTG_FILES_OPTION = scenario_parameters.get('recording_MTG_files', False)
    RECORDING_MTG_PROPERTIES_OPTION = scenario_parameters.get('recording_MTG_properties', False)
    RANDOM_OPTION = scenario_parameters.get('random', True)


    # # -- RUN main_simulation --
    # try:
    #     # We initiate the properties of the MTG "g":
    #     g = model.initiate_mtg(random=True)
    #
    #     # We launch the main simulation program:
    #     simulation.main_simulation(g, simulation_period_in_days=SIMULATION_PERIOD, time_step_in_days=TIME_STEP,
    #                                radial_growth=RADIAL_GROWTH,
    #                                ArchiSimple=ARCHISIMPLE_OPTION,
    #                                ArchiSimple_C_fraction= ARCHISIMPLE_C_FRACTION,
    #                                input_file=SCENARIO_INPUT_FILE,
    #                                outputs_directory=scenario_dirpath,
    #                                forcing_constant_inputs=FORCING_INPUTS,
    #                                constant_sucrose_input_rate=SUCROSE_INPUT_RATE,
    #                                constant_soil_temperature_in_Celsius=SOIL_TEMPERATURE,
    #                                nodules=NODULES_OPTION,
    #                                root_order_limitation=ROOT_ORDER_LIMITATION_OPTION,
    #                                root_order_treshold=ROOT_ORDER_TRESHOLD,
    #                                specific_model_option=SPECIFIC_MODEL_OPTION,
    #                                simulation_results_file='simulation_results.csv',
    #                                z_classification=CLASSIFICATION_BY_LAYERS,
    #                                z_classification_file='z_classification.csv',
    #                                recording_interval_in_days=RECORDING_INTERVAL_IN_DAYS,
    #                                z_min=LAYERS_Z_MIN, z_max=LAYERS_Z_MAX, z_interval=LAYERS_THICKNESS,
    #                                recording_images=RECORDING_IMAGES_OPTION,
    #                                root_images_directory=images_dirpath,
    #                                printing_sum=PRINTING_SUM_OPTION,
    #                                recording_sum=RECORDING_SUM_OPTION,
    #                                printing_warnings=PRINTING_WARNINGS_OPTION,
    #                                recording_g=RECORDING_MTG_FILES_OPTION,
    #                                g_directory=MTG_files_dirpath,
    #                                recording_g_properties=RECORDING_MTG_PROPERTIES_OPTION,
    #                                g_properties_directory=MTG_properties_dirpath,
    #                                random=RANDOM_OPTION,
    #                                plotting=PLOTTING,
    #                                scenario_id=scenario_id,
    #                                displayed_property=DISPLAYED_PROPERTY,
    #                                displayed_vmin=DISPLAYED_MIN_VALUE, displayed_vmax=DISPLAYED_MAX_VALUE,
    #                                log_scale=LOG_SCALE, cmap=COLOR_MAP,
    #                                x_center=X_CENTER, y_center=Y_CENTER, z_center=Z_CENTER, z_cam=Z_CAMERA,
    #                                camera_distance=CAMERA_DISTANCE, step_back_coefficient=STEP_BACK_COEFFICIENT,
    #                                camera_rotation=CAMERA_ROTATION_OPTION, n_rotation_points=CAMERA_ROTATION_N_POINTS)
    #
    # except Exception as ex:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     template = "PROBLEM: an exception of type {0} occurred. Message:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     print(message, fname, exc_tb.tb_lineno)

    # We initiate the properties of the MTG "g":
    g = model.initiate_mtg(random=True)

    # We launch the main simulation program:
    simulation.main_simulation(g, simulation_period_in_days=SIMULATION_PERIOD, time_step_in_days=TIME_STEP,
                               radial_growth=RADIAL_GROWTH,
                               ArchiSimple=ARCHISIMPLE_OPTION,
                               ArchiSimple_C_fraction=ARCHISIMPLE_C_FRACTION,
                               input_file=SCENARIO_INPUT_FILE,
                               outputs_directory=scenario_dirpath,
                               forcing_constant_inputs=FORCING_INPUTS,
                               constant_sucrose_input_rate=SUCROSE_INPUT_RATE,
                               constant_soil_temperature_in_Celsius=SOIL_TEMPERATURE,
                               nodules=NODULES_OPTION,
                               root_order_limitation=ROOT_ORDER_LIMITATION_OPTION,
                               root_order_treshold=ROOT_ORDER_TRESHOLD,
                               specific_model_option=SPECIFIC_MODEL_OPTION,
                               simulation_results_file='simulation_results.csv',
                               z_classification=CLASSIFICATION_BY_LAYERS,
                               z_classification_file='z_classification.csv',
                               recording_interval_in_days=RECORDING_INTERVAL_IN_DAYS,
                               z_min=LAYERS_Z_MIN, z_max=LAYERS_Z_MAX, z_interval=LAYERS_THICKNESS,
                               recording_images=RECORDING_IMAGES_OPTION,
                               root_images_directory=images_dirpath,
                               printing_sum=PRINTING_SUM_OPTION,
                               recording_sum=RECORDING_SUM_OPTION,
                               printing_warnings=PRINTING_WARNINGS_OPTION,
                               recording_g=RECORDING_MTG_FILES_OPTION,
                               g_directory=MTG_files_dirpath,
                               recording_g_properties=RECORDING_MTG_PROPERTIES_OPTION,
                               g_properties_directory=MTG_properties_dirpath,
                               random=RANDOM_OPTION,
                               plotting=PLOTTING,
                               scenario_id=scenario_id,
                               displayed_property=DISPLAYED_PROPERTY,
                               displayed_vmin=DISPLAYED_MIN_VALUE, displayed_vmax=DISPLAYED_MAX_VALUE,
                               log_scale=LOG_SCALE, cmap=COLOR_MAP,
                               x_center=X_CENTER, y_center=Y_CENTER, z_center=Z_CENTER, z_cam=Z_CAMERA,
                               camera_distance=CAMERA_DISTANCE, step_back_coefficient=STEP_BACK_COEFFICIENT,
                               camera_rotation=CAMERA_ROTATION_OPTION, n_rotation_points=CAMERA_ROTATION_N_POINTS)

    return

def previous_outputs_clearing(clearing = False):

    if clearing:
        # If the output directory already exists:
        if os.path.exists('outputs'):
            # print("Deleting the files in the 'outputs' folder...")
            # # We delete all the directories and files that are already present inside:
            # for root, dirs, files in os.walk('outputs'):
            #     # We first remove all the files:
            #     for file in files:
            #         os.remove(os.path.join(root, file))
            #     # # Then we remove empty folders:
            #     # for dir in dirs:
            #     #     os.rmdir(os.path.join(root, dir))
            try:
                # We remove all files and subfolders:
                print("Deleting the 'outputs' folder...")
                shutil.rmtree('outputs')
                print("Creating a new 'outputs' folder...")
                os.mkdir('outputs')
            except OSError as e:
                print("An error occured when trying to delete the output folder: %s - %s." % (e.filename, e.strerror))
        else:
            # We recreate an empty folder 'outputs':
            print("Creating a new 'outputs' folder...")
            os.mkdir('outputs')
    return

def run_multiple_scenarios():

    # We read the data frame containing the different scenarios to be simulated:
    print("Loading the instructions of scenarios...")
    scenarios_df = pd.read_csv(os.path.join('inputs', 'scenarios_list.csv'), index_col='Scenario')
    # We copy the list of scenarios' properties in the 'outputs' directory:
    scenarios_df.to_csv(os.path.join('outputs', 'scenarios_list.csv'), na_rep='NA', index=False, header=True)
    # We record the number of each scenario to be simulated:
    scenarios_df['Scenario'] = scenarios_df.index
    scenarios = scenarios_df.Scenario

    # We record the starting time of the simulation:
    t_start = time.time()
    # We look at the maximal number of parallel processes that can be run at the same time:
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)

    # We run all scenarios in parallel:
    p.map(run_one_scenario, list(scenarios))
    p.terminate()
    p.join()

    # We indicate the total time the simulations took:
    t_end = time.time()
    tmp = (t_end - t_start) / 60.
    print("")
    print("===================================")
    print("Multiprocessing took %4.3f minutes!" % tmp)

    return

if __name__ == '__main__':
# (Note: this condition avoids launching automatically the program when imported in another file)

    #CALLING ONE SCENARIO ONLY:
#     inputs = None
#     outputs = None
#     scenario = 1
#
#     try:
#         opts, args = getopt.getopt(sys.argv[1:], "i:o:s:d", ["inputs=", "outputs=", "scenario="])
#     except getopt.GetoptError as err:
#         print(str(err))
#         sys.exit(2)
#
#     for opt, arg in opts:
#         if opt in ("-i", "--inputs"):
#             inputs = arg
#         elif opt in ("-o", "--outputs"):
#             outputs = arg
#         elif opt in ("-s", "--scenario"):
#             scenario = int(arg)
#
#     run_one_scenario(inputs_dir_path=inputs, outputs_dir_path=outputs, scenario_id=scenario)

    # CALLING MULTIPLE SCENARIOS:
    # We can clear the folder containing previous outputs:
    previous_outputs_clearing(clearing=True)
    # We run the scenarios in parallel:
    run_multiple_scenarios()