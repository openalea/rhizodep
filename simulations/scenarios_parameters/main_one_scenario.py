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

    # Path of the directory which contains the inputs of the model
    if inputs_dir_path:
        INPUTS_DIRPATH = inputs_dir_path
    else:
        INPUTS_DIRPATH = 'inputs'

    # Path of the directory which contains the outputs of the model
    if inputs_dir_path:
        OUTPUTS_DIRPATH = outputs_dir_path
    else:
        OUTPUTS_DIRPATH = 'outputs'

    # Scenario to be run
    scenarios_df = pd.read_csv(os.path.join(INPUTS_DIRPATH, 'scenarios_list.csv'), index_col='Scenario')
    scenario = scenarios_df.loc[scenario_id].to_dict()
    scenario_name = 'Scenario_%.4d' % scenario_id

    # -- OUTPUTS DIRECTORY OF THE SCENARIO --

    # Path of the directory which contains the outputs of the scenario
    scenario_dirpath = os.path.join(OUTPUTS_DIRPATH, scenario_name)
    if not os.path.exists(OUTPUTS_DIRPATH):
        os.mkdir(OUTPUTS_DIRPATH)

    # Create the directory of the Scenario where results will be stored
    if not os.path.exists(scenario_dirpath):
        os.mkdir(scenario_dirpath)

    # -- SIMULATION PARAMETERS --

    # Create dict of parameters for the scenario
    scenario_parameters = tools.buildDic(scenario)

    # Update parameters
    param.__dict__.update(scenario_parameters)

    # -- SIMULATION CONDITIONS

    SIMULATION_PERIOD = scenario_parameters.get('simulation_period_in_days', 20.)  # second argument of .get() is the default value
    TIME_STEP = scenario_parameters.get('time_step_in_days', 1. / 24.)

    # Scenario input file
    SCENARIO_INPUT_FILENAME = scenario_parameters.get('input_file', None)
    if SCENARIO_INPUT_FILENAME:
        SCENARIO_INPUT_FILE = os.path.join(INPUTS_DIRPATH, SCENARIO_INPUT_FILENAME)
    else:
        SCENARIO_INPUT_FILE = None

    # Scenario output file
    SCENARIO_OUTPUT_FILE = os.path.join(scenario_dirpath, 'simulation_results_{}.csv'.format(scenario_name))

    # -- RUN main_simulation --
    try:
        # We initiate the properties of the MTG "g":
        g = model.initiate_mtg(random=True)

        # We launch the main simulation program:
        simulation.main_simulation(g, simulation_period_in_days=SIMULATION_PERIOD, time_step_in_days=TIME_STEP,
                                   radial_growth="Possible",
                                   ArchiSimple=False,
                                   property="C_hexose_root", vmin=1e-4, vmax=1e-1, log_scale=True, cmap='jet',
                                   input_file=SCENARIO_INPUT_FILE,
                                   constant_sucrose_input_rate=5e-9,
                                   constant_soil_temperature_in_Celsius=20,
                                   nodules=False,
                                   simulation_results_file=SCENARIO_OUTPUT_FILE,
                                   x_center=0, y_center=0, z_center=-1, z_cam=-2,
                                   camera_distance=4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                                   z_classification=False, z_min=0.00, z_max=1., z_interval=0.05,
                                   recording_images=False,
                                   printing_sum=False,
                                   recording_sum=True,
                                   printing_warnings=False,
                                   recording_g=False,
                                   recording_g_properties=False,
                                   random=True)

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    inputs = None
    outputs = None
    scenario = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:s:d", ["inputs=", "outputs=", "scenario="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--inputs"):
            inputs = arg
        elif opt in ("-o", "--outputs"):
            outputs = arg
        elif opt in ("-s", "--scenario"):
            scenario = int(arg)

    run_one_scenario(inputs_dir_path=inputs, outputs_dir_path=outputs, scenario_id=scenario)
