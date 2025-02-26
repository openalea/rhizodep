# -*- coding: latin-1 -*-

"""
    This script allows to run RhizoDep over one scenario or a list of tutorial, using a specific set of parameters and
    options, which are specified in an input file.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import shutil
import time
import pickle


from openalea.rhizodep import model
from openalea.rhizodep import running_simulation 
from openalea.rhizodep import parameters as param
from openalea.rhizodep.tool import tools
from openalea.rhizodep import mycorrhizae 


########################################################################################################################

# Function for running the instruction of one scenario:
#------------------------------------------------------
def run_one_scenario(scenario_id=1,
                     inputs_dir_path="inputs",
                     outputs_dir_path="outputs",
                     scenarios_list="scenarios_list.xlsx"):
    """
    This function runs RhizoDep using instructions from the scenario [scenario_id] read in the file [scenarios_list].
    The input file can either be a .csv file or a .xlsx file.
    For missing instructions, default values (e.g. those defined in 'parameters.py') will be used.
    The outputs of the simulation for this scenario are recorded in a folder 'Scenario_XXXX' (the simulation will erase
    any existing folder with the same scenario number).

    :param int scenario_id: the number of the scenario to be read in the file containing the list of tutorial
    :param str inputs_dir_path: the path of the directory containing the file [scenarios_list]
    :param str outputs_dir_path: the path of the directory where outputs of RhizoDep will be saved
    :param str scenarios_list: the name of the .csv or .xlsx file where scenario's instructions are written
    """

    # HANDLING GENERAL INPUTS AND OUTPUTS FOLDERS:
    # We define the path of the directory that will contain the outputs of the model:
    if outputs_dir_path:
        OUTPUTS_DIRPATH = outputs_dir_path
    else:
        OUTPUTS_DIRPATH = 'outputs'
    # If the folder doesn't exist:
    if not os.path.exists(OUTPUTS_DIRPATH):
        # We create it:
        os.mkdir(OUTPUTS_DIRPATH)

    # We define the path of the directory that contains the inputs of the model:
    if inputs_dir_path:
        INPUTS_DIRPATH = inputs_dir_path
    else:
        INPUTS_DIRPATH = 'inputs'

    # READING SCENARIO'S INSTRUCTIONS:
    # If we handle a CSV file:
    if os.path.splitext(scenarios_list)[1] == ".csv":
        # FOR A CSV FILE, ONE SCENARIO CORRESPONDS TO ONE LINE:
        # We read the scenario to be run:
        scenarios_df = pd.read_csv(os.path.join(INPUTS_DIRPATH, scenarios_list), index_col='Scenario')
        scenario = scenarios_df.loc[scenario_id].to_dict()
        scenario_name = 'Scenario_%.4d' % scenario_id
    # If we handle an Excel file:
    elif os.path.splitext(scenarios_list)[1] == ".xlsx":
        # FOR AN EXCEL FILE, ONE SCENARIO CORRESPONDS TO ONE COLUMN:
        # We read the scenario to be run:
        scenarios_df = pd.read_excel(os.path.join(INPUTS_DIRPATH, scenarios_list), header=0,
                                     sheet_name="scenarios_as_columns")
        # We remove the columns containing unnecessary details about parameters:
        useless_columns = ["Explanation", "Type/ Unit", "Reference_value"]
        scenarios_df.drop(useless_columns, axis=1, inplace=True)
        # Before transposing the dataframe, we rename the first column as 'Scenario':
        scenarios_df.rename(columns={'Parameter': 'Scenario'}, inplace=True)
        # Because the way of dealing with data type is not well suited with the current Excel file, we transpose the
        # dataframe, record it as a CSV file in the outputs folder, and reload it with the proper type for each
        # parameter:
        transposed_df = scenarios_df.T
        transposed_df.to_csv(os.path.join(OUTPUTS_DIRPATH, 'scenarios_list.csv'), na_rep='NA', header=False)
        new_scenarios_df = pd.read_csv(os.path.join(OUTPUTS_DIRPATH, 'scenarios_list.csv'), index_col='Scenario', header=0)
        scenario = new_scenarios_df.loc[scenario_id].to_dict()
        scenario_name = 'Scenario_%.4d' % scenario_id
    else:
        print("The extension of the 'scenarios_list' file has not been recognized (either .csv or .xlsx)!")

    # CREATING THE GENERAL OUTPUT FOLDER FOR THIS SCENARIO:
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

    # CREATING/VERIFYING THE OTHER FOLDERS (in which data are read or recorded):
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

    # SIMULATION PARAMETERS:
    # We create a dictionary containing the parameters specified in the scenario:
    scenario_parameters = tools.buildDic(scenario)
    # We update the parameters of the model with the parameters indicated in the scenario:
    param.__dict__.update(scenario_parameters)
    # OPTIONAL: We can print a CSV file containing the updated parameters:
    data_frame_parameters = pd.DataFrame.from_dict(param.__dict__)
    useless_columns = ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"]
    data_frame_parameters.drop(useless_columns, axis=1, inplace=True)
    data_frame_parameters.head(1).to_csv(os.path.join(scenario_dirpath, 'updated_parameters.csv'),
                                         na_rep='NA', index=False, header=True)

    # We read the other specific instructions of the scenario that do not correspond to the parameters in the list
    # (Note: The following function "get" looks if the first argument is present in the scenario parameters.
    # If not, it returns the default value indicated in the second argument):
    START_FROM_A_KNOWN_ROOT_MTG = scenario_parameters.get('start_from_a_known_root_MTG', False)
    ROOT_MTG_FILE = scenario_parameters.get('root_MTG_file', 'initial_root_MTG.pckl')
    MYCORRHIZAL_FUNGUS = scenario_parameters.get('mycorrhizal_fungus', False)

    INPUT_FILE_TIME_STEP = scenario_parameters.get('input_file_time_step_in_days', 1.)
    STARTING_TIME_IN_DAYS = scenario_parameters.get('starting_time_in_days', 0.)
    SIMULATION_PERIOD = scenario_parameters.get('simulation_period_in_days', 10.)
    TIME_STEP = scenario_parameters.get('time_step_in_days', 1. / 24.)

    RADIAL_GROWTH = scenario_parameters.get('radial_growth', True)
    ARCHISIMPLE_OPTION = scenario_parameters.get('ArchiSimple', False)
    GROWTH_DURATION_BY_FREQUENCY = scenario_parameters.get('GD_by_frequency', False)
    ARCHISIMPLE_C_FRACTION = scenario_parameters.get('ArchiSimple_C_fraction', 0.20)
    NODULES_OPTION = scenario_parameters.get('nodules_option', False)
    ROOT_ORDER_LIMITATION_OPTION = scenario_parameters.get('root_order_limitation', False)
    ROOT_ORDER_TRESHOLD = scenario_parameters.get('root_order_treshold', 2)
    USING_SOLVER = scenario_parameters.get('using_solver', False)
    PRINTING_SOLVER_OUTPUTS = scenario_parameters.get('printing_solver_outputs', False)
    FORCING_INPUTS = scenario_parameters.get('forcing_constant_inputs', False)
    SUCROSE_INPUT_RATE = scenario_parameters.get('constant_sucrose_input_rate', 5e-9)
    SOIL_TEMPERATURE = scenario_parameters.get('constant_soil_temperature_in_Celsius', 20)
    FORCING_SEMINAL_ROOTS_EVENTS = scenario_parameters.get('forcing_seminal_roots_events', False)
    FORCING_ADVENTITIOUS_ROOTS_EVENTS = scenario_parameters.get('forcing_seminal_roots_events', False)
    HOMOGENIZING_ROOT_CONCENTRATIONS = scenario_parameters.get('homogenizing_root_sugar_concentrations', False)
    HOMOGENIZING_SOIL_CONCENTRATIONS = scenario_parameters.get('homogenizing_soil_concentrations', False)
    RENEWAL_OF_SOIL_SOLUTION = scenario_parameters.get('renewal_of_soil_solution', False)
    INTERVAL_BETWEEN_RENEWAL_EVENTS = scenario_parameters.get('interval_between_renewal_events', 1. * 60. * 60. * 24.)

    INITIAL_SEGMENT_LENGTH = scenario_parameters.get('initial_segment_length', 1e-3)
    INITIAL_APEX_LENGTH = scenario_parameters.get('initial_apex_length', 0)
    INITIAL_C_SUCROSE_ROOT = scenario_parameters.get('initial_C_sucrose_root', 1e-4)
    INITIAL_C_HEXOSE_ROOT = scenario_parameters.get('initial_C_hexose_root', 1e-4)

    PLOTTING = scenario_parameters.get('plotting', True)
    DISPLAYED_PROPERTY = scenario_parameters.get('displayed_property', 'C_hexose_root')
    DISPLAYED_MIN_VALUE = scenario_parameters.get('displayed_min_value', 1e-15)
    DISPLAYED_MAX_VALUE = scenario_parameters.get('displayed_max_value', 1000)
    LOG_SCALE = scenario_parameters.get('log_scale', True)
    COLOR_MAP = scenario_parameters.get('color_map', 'jet')
    ROOT_HAIRS_DISPLAY = scenario_parameters.get('root_hairs_display', True)
    CAMERA_ROTATION_OPTION = scenario_parameters.get('camera_rotation', False)
    CAMERA_ROTATION_N_POINTS = scenario_parameters.get('camera_rotation_n_points', 120)
    WIDTH = scenario_parameters.get('PlantGL_window_width', 1200)
    HEIGHT = scenario_parameters.get('PlantGL_window_height', 1200)
    # For importing a color RGB vector, we need to do this step by step:
    background_color_string = scenario_parameters.get('background_color', '[0,0,0]') # The information is recorded as a string
    background_color_array = [int(i.strip()) for i in background_color_string[1:-1].split(",")] # Then converted into a list
    BACKGROUND_COLOR = np.array(background_color_array) # And finally as the proper vector.
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

    if USING_SOLVER:
        print("NOTE: a solver will be used to compute the balance between C pools within each time step!")
        print("")
    else:
        print("No solver will be used to compute the balance between C pools.")
        print("")

    # LOOKING AT THE INPUT DATA OVER TIME:
    # We finally read the input file containing data on time, temperature and sucrose input:
    SCENARIO_INPUT_FILENAME = scenario_parameters.get('input_file', None)
    if SCENARIO_INPUT_FILENAME:
        SCENARIO_INPUT_FILE = os.path.join(INPUTS_DIRPATH, SCENARIO_INPUT_FILENAME)
    else:
        SCENARIO_INPUT_FILE = None
    # If the instructions are to start the tutorial(s) at a specific time step in the input file:
    if SCENARIO_INPUT_FILE != None and STARTING_TIME_IN_DAYS > 0:
        # Then we read the file and copy it in a dataframe "df":
        original_input_frame = pd.read_csv(SCENARIO_INPUT_FILE, sep=',')
        # And we only keep the lines for which time_in days is higher than starting_time_in_days:
        new_input_frame = original_input_frame.loc[original_input_frame["time_in_days"] >= STARTING_TIME_IN_DAYS ]
        # We finally record the new data frame in the output directory of the scenario:
        new_input_file_name = 'adjusted_inputs_S' + str(scenario_id) + '.csv'
        new_input_frame.to_csv(os.path.join(OUTPUTS_DIRPATH, new_input_file_name),
                               na_rep='NA', index=False, header=True)
        # And we assign the SCENARIO_INPUT_FILE path to this dataframe:
        SCENARIO_INPUT_FILE = os.path.join(OUTPUTS_DIRPATH, new_input_file_name)

    # LOOKING AT THE INITIAL ROOT MTG:
    # We initalize a Boolean for authorizing the tutorial:
    simulation_allowed = True
    # If we decide to start from a given root MTG to be loaded:
    if START_FROM_A_KNOWN_ROOT_MTG:
        filename = os.path.join(inputs_dir_path, ROOT_MTG_FILE)
        if not os.path.exists(filename):
            print("!!! ERROR: the file", ROOT_MTG_FILE,"could not be found in the folder", inputs_dir_path, "!!!")
            print("The tutorial stops here!")
            simulation_allowed=False
        else:
            # We load the MTG file and name it "g":
            f = open(filename, 'rb')
            g = pickle.load(f)
            f.close()
            print("The MTG", ROOT_MTG_FILE,"has been loaded!")
            # And by precaution we save the initial MTG in the outputs:
            g_file_name = os.path.join(OUTPUTS_DIRPATH, 'initial_root_MTG.pckl')
            with open(g_file_name, 'wb') as output:
                pickle.dump(g, output, protocol=2)
            print("The initial MTG file has been saved in the outputs.")
    # Otherwise we initiate the properties of the MTG "g":
    else:
        # seminal_file = os.path.join(INPUTS_DIRPATH,"seminal_roots_inputs.csv")
        # adventitious_file = os.path.join(INPUTS_DIRPATH, "adventitious_roots_inputs.csv")
        g = model.initiate_mtg(random=RANDOM_OPTION,
                               simple_growth_duration=not GROWTH_DURATION_BY_FREQUENCY,
                               initial_segment_length=INITIAL_SEGMENT_LENGTH,
                               initial_apex_length=INITIAL_APEX_LENGTH,
                               initial_C_sucrose_root=INITIAL_C_SUCROSE_ROOT,
                               initial_C_hexose_root=INITIAL_C_HEXOSE_ROOT,
                               input_file_path=INPUTS_DIRPATH,
                               forcing_seminal_roots_events=FORCING_SEMINAL_ROOTS_EVENTS,
                               forcing_adventitious_roots_events=FORCING_ADVENTITIOUS_ROOTS_EVENTS,
                               seminal_roots_events_file="seminal_roots_inputs.csv",
                               adventitious_roots_events_file="adventitious_roots_inputs.csv")
        print("The root MTG has been initialized!")

    if MYCORRHIZAL_FUNGUS:
        f = mycorrhizae.initiate_mycorrhizal_fungus()
    else:
        f = None

    # LAUNCHING THE SIMULATION:
    # If the tutorial has been allowed (i.e. the MTG "g" has been defined):
    if simulation_allowed:
        # We launch the main tutorial program:
        running_simulation.main_simulation(g, simulation_period_in_days=SIMULATION_PERIOD, time_step_in_days=TIME_STEP,
                                       radial_growth=RADIAL_GROWTH,
                                       ArchiSimple=ARCHISIMPLE_OPTION,
                                       ArchiSimple_C_fraction=ARCHISIMPLE_C_FRACTION,
                                       simple_growth_duration=not GROWTH_DURATION_BY_FREQUENCY,
                                       input_file=SCENARIO_INPUT_FILE,
                                       input_file_time_step_in_days=INPUT_FILE_TIME_STEP,
                                       outputs_directory=scenario_dirpath,
                                       forcing_constant_inputs=FORCING_INPUTS,
                                       constant_sucrose_input_rate=SUCROSE_INPUT_RATE,
                                       constant_soil_temperature_in_Celsius=SOIL_TEMPERATURE,
                                       homogenizing_root_sugar_concentrations=HOMOGENIZING_ROOT_CONCENTRATIONS,
                                       homogenizing_soil_concentrations=HOMOGENIZING_SOIL_CONCENTRATIONS,
                                       renewal_of_soil_solution=RENEWAL_OF_SOIL_SOLUTION,
                                       interval_between_renewal_events=INTERVAL_BETWEEN_RENEWAL_EVENTS,
                                       nodules=NODULES_OPTION,
                                       mycorrhizal_fungus=MYCORRHIZAL_FUNGUS,
                                       fungus_MTG=f,
                                       root_order_limitation=ROOT_ORDER_LIMITATION_OPTION,
                                       root_order_treshold=ROOT_ORDER_TRESHOLD,
                                       using_solver=USING_SOLVER,
                                       printing_solver_outputs=PRINTING_SOLVER_OUTPUTS,
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
                                       root_hairs_display=ROOT_HAIRS_DISPLAY,
                                       width=WIDTH, height=HEIGHT,
                                       background_color=BACKGROUND_COLOR,
                                       x_center=X_CENTER, y_center=Y_CENTER, z_center=Z_CENTER, z_cam=Z_CAMERA,
                                       camera_distance=CAMERA_DISTANCE, step_back_coefficient=STEP_BACK_COEFFICIENT,
                                       camera_rotation=CAMERA_ROTATION_OPTION, n_rotation_points=CAMERA_ROTATION_N_POINTS)

    return

# Function for running several tutorial in parallel:
#----------------------------------------------------
def run_multiple_scenarios(scenarios_list="scenarios_list.xlsx", input_path='inputs', output_path='outputs'):

    """
    This function runs RhizoDep simultaneously for different tutorial read in the file [scenarios_list],
    by parallelizing the function 'run_one_scenario' using multiprocessing.
    The input file can either be a .csv file or a .xlsx file.
    For missing instructions, default values (e.g. those defined in 'parameters.py') will be used.
    The outputs of the simulation for each scenario are recorded in different folders 'Scenario_XXXX'.
    :param str scenarios_list: the name of the .csv or .xlsx file where scenario's instructions are written
    :param str input_path: the path of the directory containing the file [scenarios_list]
    :param str output_path: the path of the directory where outputs of RhizoDep will be saved
    """

    # READING SCENARIO INSTRUCTIONS:
    # FROM A CSV FILE where tutorial are present in different lines:
    if os.path.splitext(scenarios_list)[1] == ".csv":
        # We read the data frame containing the different tutorial to be simulated:
        print("Loading the instructions of tutorial...")
        scenarios_df = pd.read_csv(os.path.join(input_path, scenarios_list), index_col='Scenario')
        # We copy the list of tutorial' properties in the 'outputs' directory:
        scenarios_df.to_csv(os.path.join(output_path, 'scenarios_list.csv'), na_rep='NA', index=False, header=True)
        # We record the number of each scenario to be simulated:
        scenarios_df['Scenario'] = scenarios_df.index
        scenarios = scenarios_df.Scenario
    # FROM AN EXCEL FILE where tutorial are present in different columns:
    elif os.path.splitext(scenarios_list)[1] == ".xlsx":
        # We read the data frame containing the different tutorial to be simulated:
        print("Loading the instructions of scenario(s)...")
        scenarios_df = pd.read_excel(os.path.join(input_path, scenarios_list), sheet_name="scenarios_as_columns")
        # We get a list of the names of the columns, which correspond to the tutorial' numbers:
        scenarios = list(scenarios_df.columns)
        # We remove the unnecessary names of the first 4 columns, so that tutorial only contains the scenario numbers:
        del scenarios[0:4]

    # We record the starting time of the tutorial:
    t_start = time.time()
    # We look at the maximal number of parallel processes that can be run at the same time:
    num_processes = mp.cpu_count()
    # We compare this to the number of tutorial:
    print("The list of tutorial to be run is", str(list(scenarios)),
          "and the maximal number of parallel processes is", num_processes, "...")
    if num_processes < len(list(scenarios)):
        print(" !!! WATCH OUT: it is currently not possible to run all tutorial at the same time, "
              "the maximal number of processes is", num_processes, "!!!")
    print("Please check which scenario folders have actually been created in the outputs, some may be missing!")

    # We run all tutorial in parallel:
    p = mp.Pool(num_processes)
    p.map(run_one_scenario, list(scenarios))
    # WATCH OUT: p.map does not allow to have multiple arguments in the function to be run in parallel !!!
    # Arguments of 'run_one_scenario' should be modifed directly in the default parameters of the function!
    p.terminate()
    p.join()

    # We indicate the total time the simulations took:
    t_end = time.time()
    tmp = (t_end - t_start) / 60.
    print("")
    print("===================================")
    print("Simulations are done! Multiprocessing took %4.3f minutes!" % tmp)

    return


# Function for clearing the previous results:
# --------------------------------------------
def previous_outputs_clearing(output_path="outputs"):
    """
    This function is used to delete previous output files and folders.
    :param output_path: the path of the outputs directory to be cleared
    """
    # If the output directory already exists:
    if os.path.exists(output_path):
        try:
            # We remove all files and subfolders:
            print("Deleting the 'outputs' folder...")
            shutil.rmtree(output_path)
            print("Creating a new 'outputs' folder...")
            os.mkdir(output_path)
        except OSError as e:
            print("An error occured when trying to delete the output folder: %s - %s." % (e.filename, e.strerror))
    else:
        # We recreate an empty folder 'outputs':
        print("Creating a new 'outputs' folder...")
        os.mkdir(output_path)

    return

########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == '__main__':
    # (Note: this condition avoids launching automatically the program when imported in another file)

    # # CASE 1 - CALLING ONE SCENARIO ONLY:
    # #####################################
    #
    # inputs = None
    # outputs = 'outputs'
    # scenario = 4
    #
    # try:
    #     opts, args = getopt.getopt(sys.argv[1:], "i:o:s:d", ["inputs=", "outputs=", "scenario="])
    # except getopt.GetoptError as err:
    #     print(str(err))
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt in ("-i", "--inputs"):
    #         inputs = arg
    #     elif opt in ("-o", "--outputs"):
    #         outputs = arg
    #     elif opt in ("-s", "--scenario"):
    #         scenario = int(arg)
    #
    # run_one_scenario(scenario_id=1,
    #                  inputs_dir_path="C:/Users/frees/rhizodep/simulations/running_scenarios/inputs",
    #                  outputs_dir_path='outputs',
    #                  scenarios_list="scenarios_list.xlsx")

    # CASE 2 - CALLING MULTIPLE SCENARIOS:
    ######################################
    # We can clear the folder containing previous outputs:
    previous_outputs_clearing(output_path="outputs")
    # We run the tutorial in parallel :
    # WATCH OUT: you will still need to manually modify the default arguments of 'run_one_scenarios',
    # e.g. the name of the file where to read scenario instructions, even if you have entered it below!
    run_multiple_scenarios(scenarios_list="scenarios_list.xlsx",
                           input_path="inputs",
                           output_path="outputs")