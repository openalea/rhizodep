# -*- coding: latin-1 -*-

"""
    The script 'test_rhizodep' enables to run a complete simulation with the model RhizoDep or with its sub-component
    ArchiSimple alone,and to compare its actual outputs with expected outputs.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
import pandas as pd
import os
from rhizodep.model import initiate_mtg
from rhizodep.running_simulation import main_simulation
from tutorial.running_scenarios import run_one_scenario

########################################################################################################################
# DEFINING INPUT/OUTPUT FOLDERS AND SPECIFIC PARAMETERS FOR THE TEST:
########################################################################################################################

# Defining the tolerance when comparing the results of the tutorial with the expected results:
PRECISION = 6
RELATIVE_TOLERANCE = 10 ** -PRECISION
ABSOLUTE_TOLERANCE = RELATIVE_TOLERANCE

########################################################################################################################
# DEFINING THE FUNCTIONS
########################################################################################################################

# Function for comparing the results of the tutorial with the expected results:
#--------------------------------------------------------------------------------
def compare_actual_to_desired(desired_data_path, actual_data_path, overwrite_desired_data=False):
    """
    Function that compares the actual tutorial results to desired tutorial results.
    An exception is raised if the actual results do not matched the desired ones.

    :param str data_dirpath: The directory path were the tutorial results are stored.
    :param str desired_data_filename: The filename of the desired tutorial results.
    :param str actual_data_filename: The filename of the actual tutorial results.
    :param bool overwrite_desired_data: If True, the desired tutorial results are overwritten by the actual tutorial results.
    """
    # We read the desired results:
    desired_data_df = pd.read_csv(desired_data_path)

    # We read the actual results:
    actual_data_df = pd.read_csv(actual_data_path)

    # In case we want to update the desired data after the model has been upgraded:
    if overwrite_desired_data:
        # Then we transform the "desired data" file with the "actual data file" for the next time:
        actual_data_df.to_csv(desired_data_path, na_rep='NA', index=False)
    else:
        # Otherwise, we compare both files:
        for column in ([]):
            if column in desired_data_df.columns:
                assert desired_data_df[column].equals(actual_data_df[column])
                del desired_data_df[column]
                del actual_data_df[column]

    # We compare the actual data to the desired data, and raise an error message otherwise:
    error_message = "Sorry, the test failed, the new outputs are different from the previous ones!" +\
                    "\nIn particular, final root length is " + str(desired_data_df['Root length (m)'].iloc[-1]) +\
                    " cm in the original outputs, and " + str(actual_data_df['Root length (m)'].iloc[-1]) + \
                    " cm in the new outputs." + " \nSee details below:"
    np.testing.assert_allclose(actual_data_df.values, desired_data_df.values, RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE,
                               err_msg=error_message,
                               verbose=False)

    return

# Function for running the test tutorial:
#------------------------------------------
def run_reference_simulation(run_test_scenario=True, scenario_ID=1,
                             outputs_path='outputs',
                             ArchiSimple_results_file = "simulation_results_test_1.csv",
                             images_path='root_images', MTG_path='MTG_files', MTG_properties_path='MTG_properties'):
    """
    This function performs the actual tutorial for the test, either using the default value of parameters with an
    input file, or following the instructions of a test scenario read from an Excel file.
    :param bool run_test_scenario: if True, the test is based on the instructions from 'scenario_test.xlsx'
    """

    # We launch the main tutorial program:
    print("Simulation starts ...")

    if run_test_scenario:
        # OPTION 1: We run a default scenario for RhizoDep, starting from an already existing root MTG,
        # using the instructions read in the input file, using 'run_one_scenario' without multiprocessing:
        run_one_scenario(scenario_id=scenario_ID,
                         inputs_dir_path="C:/Users/frees/rhizodep/test/inputs",
                         outputs_dir_path=outputs_path,
                         scenarios_list="scenario_test.xlsx")
        print("Simulation done!")
    else:
        # OPTION 2: We run a simulation with ArchiSimple only after initializing a new root MTG, using the default case
        # with all the parameters indicated in parameters.py and using the function 'main_simulation':
        OUTPUTS_DIRPATH = 'outputs'
        # We initiate the properties of the MTG "g":
        g = initiate_mtg(random=True)
        # We run the tutorial by specifying here the input conditions and the duration:
        main_simulation(g, simulation_period_in_days=5., time_step_in_days=1./24.,
                        radial_growth="Possible", ArchiSimple=True, ArchiSimple_C_fraction=0.10,
                        outputs_directory=outputs_path,
                        forcing_constant_inputs=True, constant_sucrose_input_rate=1e-10,
                        constant_soil_temperature_in_Celsius=20,
                        root_order_limitation=False,
                        simulation_results_file=ArchiSimple_results_file,
                        recording_interval_in_days=5,
                        recording_images=True,
                        root_images_directory=images_path,
                        z_classification=False,
                        printing_sum=False,
                        recording_sum=True,
                        printing_warnings=False,
                        recording_g=False,
                        g_directory=MTG_path,
                        recording_g_properties=True,
                        g_properties_directory=MTG_properties_path,
                        random=True,
                        plotting=False,
                        displayed_property="length", displayed_vmin=0, displayed_vmax=0.01,
                        log_scale=False, cmap='copper',
                        width=800, height=800,
                        x_center=0, y_center=0, z_center=0.08, x_cam=0.2, y_cam=0.2, z_cam=-0.15)

    return

# Function for testing the run:
#------------------------------
def test_run(overwrite_desired_data=False, run_test_scenario=True, scenario_ID=1,
             reference_path='reference', reference_file='desired_simulation_results.csv',
             outputs_path='outputs', results_file='simulation_results.csv'):
    """
    This function performs a test that compares the results of a reference tutorial of RhizoDep to the desired,
    expected tutorial results. An exception is raised if the actual results do not matched the desired ones.
    :param bool overwrite_desired_data: If True, the desired tutorial results are overwritten by the actual tutorial results.
    """

    # 1. Organizing the files and the folders:
    #-----------------------------------------

    desired_data_path = os.path.join(reference_path, reference_file)
    if run_test_scenario:
        scenario_output_path = os.path.join(outputs_path, 'Scenario_%.4d' % scenario_ID)
        actual_data_path = os.path.join(scenario_output_path, results_file)
    else:
        actual_data_path = os.path.join(outputs_path, results_file)

    # Defining the root images' directory path:
    IMAGES_DIRPATH = os.path.join(outputs_path, 'root_images')
    if not os.path.exists(IMAGES_DIRPATH):
        # Then we create it:
        os.mkdir(IMAGES_DIRPATH)
    else:
        # Otherwise, we delete all the images that are already present inside:
        for root, dirs, files in os.walk(IMAGES_DIRPATH):
            for file in files:
                os.remove(os.path.join(root, file))

    # Defining the MTG files' directory path:
    MTG_DIRPATH = os.path.join(outputs_path, 'MTG_files')
    if not os.path.exists(MTG_DIRPATH):
        # Then we create it:
        os.mkdir(MTG_DIRPATH)
    else:
        # Otherwise, we delete all the images that are already present inside:
        for root, dirs, files in os.walk(MTG_DIRPATH):
            for file in files:
                os.remove(os.path.join(root, file))

    # Defining the MTG properties' directory path:
    MTG_PROP_DIRPATH = os.path.join(outputs_path, 'MTG_properties')
    if not os.path.exists(MTG_PROP_DIRPATH):
        # Then we create it:
        os.mkdir(MTG_PROP_DIRPATH)
    else:
        # Otherwise, we delete all the images that are already present inside:
        for root, dirs, files in os.walk(MTG_PROP_DIRPATH):
            for file in files:
                os.remove(os.path.join(root, file))

    # 2. Running the new tutorial:
    #-------------------------------

    # We run the reference tutorial:
    run_reference_simulation(run_test_scenario=run_test_scenario, scenario_ID=scenario_ID,
                             outputs_path=outputs_path, ArchiSimple_results_file = "simulation_results_test_1.csv",
                             images_path=IMAGES_DIRPATH, MTG_path=MTG_DIRPATH, MTG_properties_path=MTG_PROP_DIRPATH)

    # 3. Comparing the new results with the reference results:
    #---------------------------------------------------------

    # We compare actual to desired outputs (an exception is raised if the test failed):
    print('')
    print("Comparing '{}' to '{}'".format(results_file, reference_file),"...")
    compare_actual_to_desired(desired_data_path, actual_data_path,
                              overwrite_desired_data)
    print("CONGRATULATIONS! The test is passed! The new results are consistent with the reference results!")

    return

########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == '__main__':

    CREATING_NEW_REFERENCE_DATA=False
    # WATCH OUT: Set this to True only if you want to modify the reference outputs file with the new outputs!

    # TEST 1 WITH A SIMULATION CREATING A NEW ROOT MTG:
    print("\nStarting Test 1...")
    test_run(overwrite_desired_data=CREATING_NEW_REFERENCE_DATA,
             run_test_scenario=False,
             reference_path="C:/Users/frees/rhizodep/test/reference", reference_file='desired_simulation_results_1.csv',
             outputs_path="C:/Users/frees/rhizodep/test/outputs/", results_file='simulation_results_test_1.csv')

    # TEST 2 WITH ONE SCENARIO LOADING AN EXISTING MTG:
    print("\nStarting Test 2...")
    test_run(overwrite_desired_data=CREATING_NEW_REFERENCE_DATA,
             run_test_scenario=True, scenario_ID=1,
             reference_path="C:/Users/frees/rhizodep/test/reference", reference_file='desired_simulation_results_2.csv',
             outputs_path="C:/Users/frees/rhizodep/test/outputs/", results_file='simulation_results.csv')

    print("\nCONCLUSION: Both tests have been successfully passed!")


