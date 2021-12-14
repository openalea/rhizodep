# Todo: Introduce explicit threshold concentration for limiting processes, rather than "0"
# Todo: Consider giving priority to root maintenance, and trigger senescence when root maintenance is not ensured
# Todo: Watch the calculation of surface and volume for the apices - if they correspond to cones, the mass balance for segmentation may not be correct!
# Todo: Include cell sloughing and mucilage production
# Todo: Make a distinction between seminal roots and adventitious roots
# Todo: Check actual_elongation_rate with thermal time...

# Importation of functions from the system:
###########################################

from math import trunc, floor
from decimal import Decimal
import numpy as np
import pandas as pd
import os
import os.path
import timeit

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl
import rhizodep.model as model
import rhizodep.tools as tools
import rhizodep.parameters as param

import pickle

# Setting the randomness in the whole code to reproduce the same root system over different runs:
# random_choice = int(round(np.random.normal(100,50)))
random_choice = 8
print("The random seed used for this run is", random_choice)
np.random.seed(random_choice)


# SIMULATION OVER TIME:
#######################

def formatted_inputs(original_input_file="None", original_time_step_in_days=1 / 24., final_time_step_in_days=1.,
                     simulation_period_in_days=60., do_not_execute_if_file_with_suitable_size_exists=False):
    """
    This function creates a new input file containing data on soil temperature and sucrose input rate (in mol of sucrose per second),
    based on an original file. The new file is adapted to the required time step (higher, equal or lower than the original time step).
    If the option 'do_not_execute_if_file_with_suitable_size_exists' is set to True, a new file will be created only if the already-
    existing 'input_file.csv' does not contain the correct number of lines.
    """

    # If there is a file where the inputs of sucrose in the root system have to be read:
    if original_input_file != "None":

        # We first define the path and the file to read as a .csv:
        PATH1 = os.path.join('.', original_input_file)
        # Then we read the file and copy it in a dataframe "df":
        df = pd.read_csv(PATH1, sep=',', header=0)

        # simulation_period_in_days = df['time_in_days'].max()-df['time_in_days'].min()
        n_steps = int(floor(simulation_period_in_days / final_time_step_in_days)) + 1

        if do_not_execute_if_file_with_suitable_size_exists:
            PATH2 = os.path.join('.', 'input_file.csv')
            previous_file = pd.read_csv(PATH2, sep=',', header=0)

            if len(previous_file['step_number']) == n_steps:
                print("There is already an 'input_file.csv' of proper length for this simulation, "
                      "we therefore did not create a new input file here (if you wish to do so, please select 'do_not_execute_if_file_with_suitable_size_exists=False').")
                return previous_file

        # We create a new, final dataframe that will be used to read inputs of sucrose:
        input_frame = pd.DataFrame(
            columns=["step_number", "initial_time_in_days", "final_time_in_days", "soil_temperature_in_degree_Celsius",
                     "sucrose_input_rate"])
        input_frame["step_number"] = range(1, n_steps + 1)
        input_frame["initial_time_in_days"] = (input_frame["step_number"] - 1) * final_time_step_in_days * 1.
        input_frame["final_time_in_days"] = input_frame["initial_time_in_days"] + final_time_step_in_days

        print("Creating a new input file adapted to the required time step (time step =", final_time_step_in_days, "days):")

        # CASE 1: the final time step is higher than the original one, so we have to calculate an average value:
        # -------------------------------------------------------------------------------------------------------
        if final_time_step_in_days >= original_time_step_in_days:

            # We initialize the row ID "j" in which exact sucrose inputs will be recorded:
            j = 0
            # We initialize the time over which sucrose input will be integrated:
            cumulated_time_in_days = 0.
            time_in_days = 0.
            sucrose_input = 0.
            sum_of_temperature_in_degree_days = 0.

            # For each line of the data frame that contains information on sucrose input:
            for i in range(0, len(df['time_in_days'])):

                # If the current cumulated time is below the time step of the main simulation:
                if (cumulated_time_in_days + original_time_step_in_days) < 0.9999 * final_time_step_in_days:
                    # Then the amount of time elapsed here is the initial time step:
                    net_elapsed_time_in_days = original_time_step_in_days
                    remaining_time = 0.
                else:
                    # Otherwise, we limit the elapsed time to the one necessary for reaching a round number of final_time_step_in_days:
                    net_elapsed_time_in_days = original_time_step_in_days - (
                            cumulated_time_in_days + original_time_step_in_days - final_time_step_in_days)
                    remaining_time = (cumulated_time_in_days + original_time_step_in_days - final_time_step_in_days)

                # In any case, the sucrose input and the sum of temperature are increased according to the elapsed time considered here:
                sucrose_input += df.loc[i, 'sucrose_input_rate'] * net_elapsed_time_in_days * 60 * 60 * 24.
                sum_of_temperature_in_degree_days += df.loc[i, 'soil_temperature_in_Celsius'] * net_elapsed_time_in_days

                # If at the end of the original time step corresponding to the line i in df, one final time step has been reached
                # or if this is the last line of the original table df to be read:
                if cumulated_time_in_days + original_time_step_in_days >= 0.9999 * final_time_step_in_days \
                        or i == len(df['time_in_days']) - 1:

                    # We move to the next row in the table to create:
                    j += 1
                    print("   Creating line", j, "on", n_steps, "lines in the new input frame...")
                    # We record the final temperature and sucrose input rate:
                    input_frame.loc[j - 1, 'sucrose_input_rate'] = sucrose_input / (
                            (cumulated_time_in_days + net_elapsed_time_in_days) * 60. * 60. * 24.)
                    input_frame.loc[j - 1, 'soil_temperature_in_degree_Celsius'] = sum_of_temperature_in_degree_days \
                                                                                   / ((
                            cumulated_time_in_days + net_elapsed_time_in_days))

                    # We reinitialize the counter with the exact time that has not been included yet:
                    cumulated_time_in_days = remaining_time
                    # print "After reinitializing cumulated time, it is equal to", cumulated_time_in_days
                    # We reinitialize the sucrose input with the remaining sucrose that has not been taken into account:
                    sucrose_input = df.loc[i, 'sucrose_input_rate'] * (cumulated_time_in_days) * 60 * 60 * 24.
                    sum_of_temperature_in_degree_days = df.loc[i, 'soil_temperature_in_Celsius'] * (
                        cumulated_time_in_days)

                else:
                    cumulated_time_in_days += original_time_step_in_days

                # If the number of total lines of the final input frame has been reached:
                if j >= n_steps:
                    # Then we stop the loop here:
                    break

        # CASE 2: the final time step is higher than the original one, so we have to calculate an average value:
        # -------------------------------------------------------------------------------------------------------
        if final_time_step_in_days < original_time_step_in_days:

            # We initialize the row ID "i" in which the original values are read:
            i = 1
            # print "      Considering now line", i, "in the original table for time = ", df.loc[i, 'time_in_days']

            # We initialize the time over which sucrose input will be integrated:
            cumulated_time_in_days = 0.
            time_in_days = 0.
            sucrose_input = 0.
            sum_of_temperature_in_degree_days = 0.

            sucrose_input_rate_initial = df.loc[0, 'sucrose_input_rate']
            temperature_initial = df.loc[0, 'soil_temperature_in_Celsius']

            # For each line of the final table:
            for j in range(0, n_steps):

                # If the current cumulated time is below the time step of the main simulation:
                if cumulated_time_in_days + final_time_step_in_days < 0.9999 * original_time_step_in_days:
                    # Then the amount of time elapsed here is the initial time step:
                    net_elapsed_time_in_days = final_time_step_in_days
                    remaining_time = 0.
                else:
                    # Otherwise, we limit the elapsed time to the one necessary for reaching a round number of original_time_step_in_days:
                    net_elapsed_time_in_days = final_time_step_in_days - (
                            cumulated_time_in_days + final_time_step_in_days - original_time_step_in_days)
                    remaining_time = (cumulated_time_in_days + final_time_step_in_days - original_time_step_in_days)

                sucrose_input_rate = sucrose_input_rate_initial \
                                     + (df.loc[
                                            i, 'sucrose_input_rate'] - sucrose_input_rate_initial) / original_time_step_in_days * cumulated_time_in_days
                temperature = temperature_initial \
                              + (df.loc[
                                     i, 'soil_temperature_in_Celsius'] - temperature_initial) / original_time_step_in_days * cumulated_time_in_days

                print("   Creating line", j + 1, "on", n_steps, "lines in the new input frame...")
                # We record the final temperature and sucrose input rate:
                input_frame.loc[j, 'sucrose_input_rate'] = sucrose_input_rate
                input_frame.loc[j, 'soil_temperature_in_degree_Celsius'] = temperature

                # If at the end of the final time step corresponding to the line j in input_frame, one original time step has been reached:
                if cumulated_time_in_days + final_time_step_in_days >= 0.9999 * original_time_step_in_days:

                    # We record the current sucrose input rate and temperature in the original table:
                    sucrose_input_rate_initial = df.loc[i, 'sucrose_input_rate']
                    temperature = df.loc[i, 'soil_temperature_in_Celsius']

                    # We move to the next row in the original table, unless we are already at the last line:
                    if i < len(df['time_in_days']) - 1:
                        i += 1
                        # print "      Considering now line", i, "in the original table for time = ", df.loc[i, 'time_in_days']

                    # We reinitialize the counter with the exact time that has not been included yet:
                    cumulated_time_in_days = remaining_time

                else:

                    cumulated_time_in_days += final_time_step_in_days

                # If the number of total lines of the final input frame has been reached:
                if j >= n_steps:
                    # Then we stop the loop here:
                    break

    input_frame.to_csv('input_file.csv', na_rep='NA', index=False, header=True)
    print("The new input file adapted to the required time step has been created and saved as 'input_file.csv'.")

    return input_frame


# We define the main simulation program:
def main_simulation(g, simulation_period_in_days=20., time_step_in_days=1.,
                    radial_growth="Impossible", ArchiSimple=False,
                    property="C_hexose_root", vmin=1e-6, vmax=1e-0, log_scale=True, cmap='brg',
                    input_file="None",
                    constant_sucrose_input_rate=1.e-6,
                    constant_soil_temperature_in_Celsius=20,
                    nodules=False,
                    x_center=0, y_center=0, z_center=-1, z_cam=-1,
                    camera_distance=10., step_back_coefficient=0., camera_rotation=False, n_rotation_points=24 * 5,
                    recording_images=False,
                    z_classification=False, z_min=0., z_max=1., z_interval=0.5,
                    printing_sum=False,
                    recording_sum=False,
                    printing_warnings=False,
                    recording_g=False,
                    recording_g_properties=True,
                    random=False):
    # We convert the time step in seconds:
    time_step_in_seconds = time_step_in_days * 60. * 60. * 24.
    # We calculate the number of steps necessary to reach the end of the simulation period:
    if simulation_period_in_days == 0. or time_step_in_days == 0.:
        print("WATCH OUT: No simulation was done, as time input was 0.")
        n_steps = 0
    else:
        n_steps = trunc(simulation_period_in_days / time_step_in_days) + 1

    print("n_steps is ", n_steps)  # TODO: delete

    # We call global variables:
    global thermal_time_since_last_adventitious_root_emergence
    global adventitious_root_emergence

    # We initialize empty variables at t=0:
    step = 0
    time = 0.
    total_struct_mass = 0.
    cumulated_hexose_exudation = 0.
    cumulated_respired_CO2 = 0.
    cumulated_struct_mass_production = 0.
    sucrose_input_rate = 0.
    C_cumulated_in_the_degraded_pool = 0.
    C_cumulated_in_the_gaz_phase = 0.

    # We initialize empty lists for recording the macro-results:
    time_in_days_series = []
    sucrose_input_series = []
    total_living_root_length_series = []
    total_dead_root_length_series = []
    total_living_root_surface_series = []
    total_dead_root_surface_series = []
    total_living_root_struct_mass_series = []
    total_dead_root_struct_mass_series = []
    total_sucrose_root_series = []
    total_hexose_root_series = []
    total_hexose_reserve_series = []
    total_hexose_soil_series = []

    total_sucrose_root_deficit_series = []
    total_hexose_root_deficit_series = []
    total_hexose_soil_deficit_series = []

    total_respiration_series = []
    total_respiration_root_growth_series = []
    total_respiration_root_maintenance_series = []
    total_structural_mass_production_series = []
    total_hexose_production_from_phloem_series = []
    total_sucrose_loading_in_phloem_series = []
    total_hexose_mobilization_from_reserve_series = []
    total_hexose_immobilization_as_reserve_series = []
    total_hexose_exudation_series = []
    total_hexose_uptake_series = []
    total_hexose_degradation_series = []
    total_net_hexose_exudation_series = []
    C_in_the_root_soil_system_series = []
    C_cumulated_in_the_degraded_pool_series = []
    C_cumulated_in_the_gaz_phase_series = []
    global_sucrose_deficit_series = []
    tip_C_hexose_root_series = []

    # We create an empty dictionary that will contain the results of z classification:
    z_dictionary_series = {}

    if recording_images:
        # We define the directory "video"
        video_dir = 'video'
        # If this directory doesn't exist:
        if not os.path.exists(video_dir):
            # Then we create it:
            os.mkdir(video_dir)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g:
        # We define the directory "MTG_files"
        g_dir = 'MTG_files'
        # If this directory doesn't exist:
        if not os.path.exists(g_dir):
            # Then we create it:
            os.mkdir(g_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(g_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g_properties:
        # We define the directory "MTG_properties"
        prop_dir = 'MTG_properties'
        # If this directory doesn't exist:
        if not os.path.exists(prop_dir):
            # Then we create it:
            os.mkdir(prop_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(prop_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    # READING THE INPUT FILE:
    # -----------------------
    if input_file != "None" and (constant_sucrose_input_rate <= 0 or constant_soil_temperature_in_Celsius <= 0):
        # # We first define the path and the file to read as a .csv:
        # PATH = os.path.join('.', input_file)
        # # Then we read the file and copy it in a dataframe "df":
        # input_frame = pd.read_csv(PATH, sep=',')
        # We use the function 'formatted inputs' to create a table containing the input data (soil temperature and sucrose input)
        # for each required step, depending on the chosen time step:
        input_frame = formatted_inputs(original_input_file=input_file,
                                       original_time_step_in_days=1 / 24.,
                                       final_time_step_in_days=time_step_in_days,
                                       simulation_period_in_days=simulation_period_in_days,
                                       do_not_execute_if_file_with_suitable_size_exists=False)

    # RECORDING THE INITIAL STATE OF THE MTG:
    # ---------------------------------------
    step = 0

    # If the rotation of the camera around the root system is required:
    if camera_rotation:
        # We calculate the coordinates of the camera on the circle around the center:
        x_coordinates, y_coordinates, z_coordinates = tools.circle_coordinates(z_center=z_cam, radius=camera_distance,
                                                                         n_points=n_rotation_points)
        # We initialize the index for reading each coordinates:
        index_camera = 0
        x_cam = x_coordinates[index_camera]
        y_cam = y_coordinates[index_camera]
        z_cam = z_coordinates[index_camera]
        sc = tools.plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                      x_center=x_center,
                      y_center=y_center,
                      z_center=z_center,
                      x_cam=x_cam,
                      y_cam=y_cam,
                      z_cam=z_cam)
    else:
        x_camera = camera_distance
        x_cam = camera_distance
        z_camera = z_cam
        print("OK")
        sc = tools.plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                      x_center=x_center,
                      y_center=y_center,
                      z_center=z_center,
                      x_cam=x_camera,
                      y_cam=0,
                      z_cam=z_camera)
        # We move the camera further from the root system:
        x_camera = x_cam + x_cam * step_back_coefficient * step
        z_camera = z_cam + z_cam * step_back_coefficient * step
    # We finally display the MTG on PlantGL:
    print("OK")
    pgl.Viewer.display(sc)

    # For recording the graph at each time step to make a video later:
    # -----------------------------------------------------------------
    if recording_images:
        image_name = os.path.join(video_dir, 'root%.5d.png')
        pgl.Viewer.saveSnapshot(image_name % step)

    # For integrating root variables on the z axis:
    # ----------------------------------------------
    if z_classification:
        z_dictionary = model.classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
        z_dictionary["time_in_days"] = 0
        z_dictionary_series.update(z_dictionary)
        print(z_dictionary_series)

    # For recording the MTG at each time step to load it later on:
    # ------------------------------------------------------------
    if recording_g:
        g_file_name = os.path.join(g_dir, 'root%.5d.pckl')
        with open(g_file_name % step, 'wb') as output_file:
            pickle.dump(g, output_file, protocol=2)

    # For recording the properties of g in a csv file:
    # ------------------------------------------------
    if recording_g_properties:
        prop_file_name = os.path.join(prop_dir, 'root%.5d.csv')
        model.recording_MTG_properties(g, file_name=prop_file_name % step)

    # SUMMING AND PRINTING VARIABLES ON THE ROOT SYSTEM:
    # --------------------------------------------------

    # We reset to 0 all growth-associated C costs:
    model.reinitializing_growth_variables(g)

    if printing_sum:
        dictionary = model.summing(g,
                             printing_total_length=True,
                             printing_total_struct_mass=True,
                             printing_all=True)
    elif not printing_sum and recording_sum:
        dictionary = model.summing(g,
                             printing_total_length=True,
                             printing_total_struct_mass=True,
                             printing_all=False)
    if recording_sum:
        time_in_days_series.append(time_step_in_days * step)
        sucrose_input_series.append(sucrose_input_rate * time_step_in_seconds)
        total_living_root_length_series.append(dictionary["total_living_root_length"])
        total_dead_root_length_series.append(dictionary["total_dead_root_length"])
        total_living_root_struct_mass_series.append(dictionary["total_living_root_struct_mass"])
        total_dead_root_struct_mass_series.append(dictionary["total_dead_root_struct_mass"])
        total_living_root_surface_series.append(dictionary["total_living_root_surface"])
        total_dead_root_surface_series.append(dictionary["total_dead_root_surface"])
        total_sucrose_root_series.append(dictionary["total_sucrose_root"])
        total_hexose_root_series.append(dictionary["total_hexose_root"])
        total_hexose_reserve_series.append(dictionary["total_hexose_reserve"])
        total_hexose_soil_series.append(dictionary["total_hexose_soil"])

        total_sucrose_root_deficit_series.append(dictionary["total_sucrose_root_deficit"])
        total_hexose_root_deficit_series.append(dictionary["total_hexose_root_deficit"])
        total_hexose_soil_deficit_series.append(dictionary["total_hexose_soil_deficit"])

        total_respiration_series.append(dictionary["total_respiration"])
        total_respiration_root_growth_series.append(dictionary["total_respiration_root_growth"])
        total_respiration_root_maintenance_series.append(dictionary["total_respiration_root_maintenance"])
        total_structural_mass_production_series.append(dictionary["total_structural_mass_production"])
        total_hexose_production_from_phloem_series.append(dictionary["total_hexose_production_from_phloem"])
        total_sucrose_loading_in_phloem_series.append(dictionary["total_sucrose_loading_in_phloem"])
        total_hexose_mobilization_from_reserve_series.append(dictionary["total_hexose_mobilization_from_reserve"])
        total_hexose_immobilization_as_reserve_series.append(dictionary["total_hexose_immobilization_as_reserve"])
        total_hexose_exudation_series.append(dictionary["total_hexose_exudation"])
        total_hexose_uptake_series.append(dictionary["total_hexose_uptake"])
        total_hexose_degradation_series.append(dictionary["total_hexose_degradation"])
        total_net_hexose_exudation_series.append(dictionary["total_net_hexose_exudation"])

        C_in_the_root_soil_system_series.append(dictionary["C_in_the_root_soil_system"])
        C_cumulated_in_the_degraded_pool += dictionary["C_degraded_in_the_soil"]
        C_cumulated_in_the_degraded_pool_series.append(C_cumulated_in_the_degraded_pool)
        C_cumulated_in_the_gaz_phase += dictionary["C_respired_by_roots"]
        C_cumulated_in_the_gaz_phase_series.append(C_cumulated_in_the_gaz_phase)
        global_sucrose_deficit_series.append(global_sucrose_deficit)

        tip_C_hexose_root_series.append(g.node(0).C_hexose_root)

        # Initializing the amount of C in the root_soil_CO2 system:
        previous_C_in_the_system = dictionary["C_in_the_root_soil_system"] + C_cumulated_in_the_gaz_phase
        theoretical_cumulated_C_in_the_system = previous_C_in_the_system

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # The code will try to run the following code until it is finished or an error has been raised:
    try:
        # An iteration is done for each time step:
        for step in range(1, n_steps):

            # At the beginning of the time step, we reset the global variable allowing the emergence of adventitious roots:
            adventitious_root_emergence = "Possible"
            # We keep in memory the value of the global variable time_since_adventitious_root_emergence at the beginning of the time steo:
            initial_time_since_adventitious_root_emergence = thermal_time_since_last_adventitious_root_emergence

            # We calculate the current time in hours:
            current_time_in_hours = step * time_step_in_days * 24.

            # DEFINING THE INPUT OF CARBON TO THE ROOTS FOR THIS TIME STEP:
            # --------------------------------------------------------------
            if constant_sucrose_input_rate > 0 or input_file == "None":
                sucrose_input_rate = constant_sucrose_input_rate
            else:
                sucrose_input_rate = input_frame.loc[step, 'sucrose_input_rate']

            # DEFINING THE TEMPERATURE OF THE SOIL FOR THIS TIME STEP:
            # --------------------------------------------------------
            if constant_soil_temperature_in_Celsius > 0 or input_file == "None":
                soil_temperature = constant_soil_temperature_in_Celsius
            else:
                soil_temperature = input_frame.loc[step, 'soil_temperature_in_degree_Celsius']

            # CALCULATING AN EQUIVALENT OF THERMAL TIME:
            # -------------------------------------------

            # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
            temperature_time_adjustment = model.temperature_modification(temperature_in_Celsius=soil_temperature,
                                                                   process_at_T_ref=1,
                                                                   T_ref=param.T_ref_growth,
                                                                   A=param.growth_increase_with_temperature,
                                                                   B=1,
                                                                   C=0)

            # STARTING THE ACTUAL SIMULATION:
            # --------------------------------
            print("")
            print("From t =", "{:.2f}".format(Decimal((step - 1) * time_step_in_days)), "days to t =",
                  "{:.2f}".format(Decimal(step * time_step_in_days)), "days:")
            print("------------------------------------")
            print("   Soil temperature is", soil_temperature, "degree Celsius.")
            print("   The input rate of sucrose to the root for time=", current_time_in_hours, "h is",
                  "{:.2E}".format(Decimal(sucrose_input_rate)), "mol of sucrose per second, i.e.",
                  "{:.2E}".format(Decimal(sucrose_input_rate * 60. * 60. * 24.)), "mol of sucrose per day.")

            print("   The root system initially includes", len(g) - 1, "root elements.")

            # CASE 1: WE REPRODUCE THE GROWTH WITHOUT CONSIDERATIONS OF LOCAL CONCENTRATIONS
            # -------------------------------------------------------------------------------

            if ArchiSimple:

                # The input of C (gram of C) from shoots is calculated from the input of sucrose:
                C_input = sucrose_input_rate * 12 * 12.01 * time_step_in_seconds
                # We assume that only a fraction of this C_input will be used for producing struct_mass:
                fraction = 0.20
                struct_mass_input = C_input / param.struct_mass_C_content * fraction

                # We calculate the potential growth, already based on ArchiSimple rules:
                model.potential_growth(g, time_step_in_seconds=time_step_in_seconds,
                                 radial_growth=radial_growth,
                                 ArchiSimple=True,
                                 soil_temperature_in_Celsius=soil_temperature)

                # We use the function ArchiSimple_growth to adapt the potential growth to the available struct_mass:
                SC = model.satisfaction_coefficient(g, struct_mass_input=struct_mass_input)
                model.ArchiSimple_growth(g, SC, time_step_in_seconds,
                                   soil_temperature_in_Celsius=soil_temperature,
                                   printing_warnings=printing_warnings)

                # We proceed to the segmentation of the whole root system (NOTE: segmentation should always occur AFTER actual growth):
                model.segmentation_and_primordia_formation(g, time_step_in_seconds, printing_warnings=printing_warnings,
                                                     soil_temperature_in_Celsius=soil_temperature, random=random,
                                                     nodules=nodules)

            else:

                # CASE 2: WE PERFORM THE COMPLETE MODEL WITH C BALANCE IN EACH ROOT ELEMENT
                # --------------------------------------------------------------------------

                # We reset to 0 all growth-associated C costs:
                model.reinitializing_growth_variables(g)

                # Calculation of potential growth without consideration of available hexose:
                model.potential_growth(g, time_step_in_seconds=time_step_in_seconds,
                                 radial_growth=radial_growth,
                                 soil_temperature_in_Celsius=soil_temperature,
                                 ArchiSimple=False)

                # Calculation of actual growth based on the hexose remaining in the roots,
                # and corresponding consumption of hexose in the root:
                model.actual_growth_and_corresponding_respiration(g, time_step_in_seconds=time_step_in_seconds,
                                                            soil_temperature_in_Celsius=soil_temperature,
                                                            printing_warnings=printing_warnings)
                # # We proceed to the segmentation of the whole root system (NOTE: segmentation should always occur AFTER actual growth):
                model.segmentation_and_primordia_formation(g, time_step_in_seconds,
                                                     soil_temperature_in_Celsius=soil_temperature,
                                                     random=random,
                                                     nodules=nodules)
                model.dist_to_tip(g)

                # Consumption of hexose in the soil:
                model.soil_hexose_degradation(g, time_step_in_seconds=time_step_in_seconds,
                                        soil_temperature_in_Celsius=soil_temperature,
                                        printing_warnings=printing_warnings)

                # Transfer of hexose from the root to the soil, consumption of hexose inside the roots:
                model.root_hexose_exudation(g, time_step_in_seconds=time_step_in_seconds,
                                      soil_temperature_in_Celsius=soil_temperature,
                                      printing_warnings=printing_warnings)
                # Transfer of hexose from the soil to the root, consumption of hexose in the soil:
                model.root_hexose_uptake(g, time_step_in_seconds=time_step_in_seconds,
                                   soil_temperature_in_Celsius=soil_temperature,
                                   printing_warnings=printing_warnings)

                # Consumption of hexose in the root by maintenance respiration:
                model.maintenance_respiration(g, time_step_in_seconds=time_step_in_seconds,
                                        soil_temperature_in_Celsius=soil_temperature,
                                        printing_warnings=printing_warnings)

                # Unloading of sucrose from phloem and conversion of sucrose into hexose:
                model.exchange_with_phloem(g, time_step_in_seconds=time_step_in_seconds,
                                     soil_temperature_in_Celsius=soil_temperature,
                                     printing_warnings=printing_warnings)

                # Net immobilization of hexose within a reserve pool:
                model.exchange_with_reserve(g, time_step_in_seconds=time_step_in_seconds,
                                      soil_temperature_in_Celsius=soil_temperature,
                                      printing_warnings=printing_warnings)

                # Calculation of the new concentrations in hexose and sucrose once all the processes have been done:
                tip_C_hexose_root = model.balance(g, printing_warnings=printing_warnings)

                # Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
                model.shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=sucrose_input_rate,
                                                   time_step_in_seconds=time_step_in_seconds,
                                                   printing_warnings=printing_warnings)
                # WARNING: The function "shoot_sucrose_supply_and_spreading" must be called AFTER the function "balance",!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # otherwise the deficit in sucrose may be counted twice!!!

                # # OPTIONAL: checking of possible anomalies in the root system:
                model.control_of_anomalies(g)

            # A the end of the time step, if the global variable "time_since_adventitious_root_emergence" has been unchanged:
            if thermal_time_since_last_adventitious_root_emergence == initial_time_since_adventitious_root_emergence:
                # Then we increment it by the time step:
                thermal_time_since_last_adventitious_root_emergence += time_step_in_seconds * temperature_time_adjustment
            # Otherwise, the variable has already been reset when the emergence of one adventitious root has been allowed.

            # PLOTTING THE MTG:
            # ------------------

            # If the rotation of the camera around the root system is required:
            if camera_rotation:
                x_cam = x_coordinates[index_camera]
                y_cam = y_coordinates[index_camera]
                z_cam = z_coordinates[index_camera]
                sc = tools.plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                              x_center=x_center,
                              y_center=y_center,
                              z_center=z_center,
                              x_cam=x_cam,
                              y_cam=y_cam,
                              z_cam=z_cam)
                # We define the index of the coordinates to read at the next step:
                index_camera = index_camera + 1
                # If this index is higher than the number of coordinates in each vector:
                if index_camera >= n_rotation_points:
                    # Then we reset the index to 0:
                    index_camera = 0
            # Otherwise, the camera will stay on a fixed position:
            else:

                sc = tools.plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                              x_center=x_center,
                              y_center=y_center,
                              z_center=z_center,
                              x_cam=x_camera,
                              y_cam=0,
                              z_cam=z_camera)
                # We move the camera further from the root system:
                x_camera = x_cam + x_cam * step_back_coefficient * step
                z_camera = z_cam + z_cam * step_back_coefficient * step
            # We finally display the MTG on PlantGL:
            pgl.Viewer.display(sc)

            # For recording the graph at each time step to make a video later:
            # -----------------------------------------------------------------
            if recording_images:
                image_name = os.path.join(video_dir, 'root%.5d.png')
                pgl.Viewer.saveSnapshot(image_name % step)

            # For integrating root variables on the z axis:
            # ----------------------------------------------
            if z_classification:
                z_dictionary = model.classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
                z_dictionary["time_in_days"] = time_step_in_days * step
                z_dictionary_series.update(z_dictionary)

            # For recording the MTG at each time step to load it later on:
            # ------------------------------------------------------------
            if recording_g:
                g_file_name = os.path.join(g_dir, 'root%.5d.pckl')
                with open(g_file_name % step, 'wb') as output:
                    pickle.dump(g, output, protocol=2)

            # For recording the properties of g in a csv file:
            # --------------------------------------------------
            if recording_g_properties:
                prop_file_name = os.path.join(prop_dir, 'root%.5d.csv')
                model.recording_MTG_properties(g, file_name=prop_file_name % step)

            # SUMMING AND PRINTING VARIABLES ON THE ROOT SYSTEM:
            # --------------------------------------------------
            if printing_sum:
                dictionary = model.summing(g,
                                     printing_total_length=True,
                                     printing_total_struct_mass=True,
                                     printing_all=True)
            elif not printing_sum and recording_sum:
                dictionary = model.summing(g,
                                     printing_total_length=True,
                                     printing_total_struct_mass=True,
                                     printing_all=False)
            if recording_sum:
                time_in_days_series.append(time_step_in_days * step)
                sucrose_input_series.append(sucrose_input_rate * time_step_in_seconds)
                total_living_root_length_series.append(dictionary["total_living_root_length"])
                total_dead_root_length_series.append(dictionary["total_dead_root_length"])
                total_living_root_struct_mass_series.append(dictionary["total_living_root_struct_mass"])
                total_dead_root_struct_mass_series.append(dictionary["total_dead_root_struct_mass"])
                total_living_root_surface_series.append(dictionary["total_living_root_surface"])
                total_dead_root_surface_series.append(dictionary["total_dead_root_surface"])
                total_sucrose_root_series.append(dictionary["total_sucrose_root"])
                total_hexose_root_series.append(dictionary["total_hexose_root"])
                total_hexose_reserve_series.append(dictionary["total_hexose_reserve"])
                total_hexose_soil_series.append(dictionary["total_hexose_soil"])

                total_sucrose_root_deficit_series.append(dictionary["total_sucrose_root_deficit"])
                total_hexose_root_deficit_series.append(dictionary["total_hexose_root_deficit"])
                total_hexose_soil_deficit_series.append(dictionary["total_hexose_soil_deficit"])

                total_respiration_series.append(dictionary["total_respiration"])
                total_respiration_root_growth_series.append(dictionary["total_respiration_root_growth"])
                total_respiration_root_maintenance_series.append(dictionary["total_respiration_root_maintenance"])
                total_structural_mass_production_series.append(dictionary["total_structural_mass_production"])
                total_hexose_production_from_phloem_series.append(dictionary["total_hexose_production_from_phloem"])
                total_sucrose_loading_in_phloem_series.append(dictionary["total_sucrose_loading_in_phloem"])
                total_hexose_mobilization_from_reserve_series.append(
                    dictionary["total_hexose_mobilization_from_reserve"])
                total_hexose_immobilization_as_reserve_series.append(
                    dictionary["total_hexose_immobilization_as_reserve"])
                total_hexose_exudation_series.append(dictionary["total_hexose_exudation"])
                total_hexose_uptake_series.append(dictionary["total_hexose_uptake"])
                total_hexose_degradation_series.append(dictionary["total_hexose_degradation"])
                total_net_hexose_exudation_series.append(dictionary["total_net_hexose_exudation"])

                C_in_the_root_soil_system_series.append(dictionary["C_in_the_root_soil_system"])
                C_cumulated_in_the_degraded_pool += dictionary["C_degraded_in_the_soil"]
                C_cumulated_in_the_degraded_pool_series.append(C_cumulated_in_the_degraded_pool)
                C_cumulated_in_the_gaz_phase += dictionary["C_respired_by_roots"]
                C_cumulated_in_the_gaz_phase_series.append(C_cumulated_in_the_gaz_phase)
                global_sucrose_deficit_series.append(global_sucrose_deficit)

                tip_C_hexose_root_series.append(tip_C_hexose_root)

                # CHECKING CARBON BALANCE:
                current_C_in_the_system = dictionary[
                                              "C_in_the_root_soil_system"] + C_cumulated_in_the_gaz_phase + C_cumulated_in_the_degraded_pool
                theoretical_current_C_in_the_system = (
                        previous_C_in_the_system + sucrose_input_rate * time_step_in_seconds * 12.)
                theoretical_cumulated_C_in_the_system += sucrose_input_rate * time_step_in_seconds * 12.

                if abs(
                        current_C_in_the_system - theoretical_cumulated_C_in_the_system) / current_C_in_the_system > 1e-10:
                    print("!!! ERROR ON CARBON BALANCE: the current amount of C in the system is",
                          "{:.2E}".format(Decimal(current_C_in_the_system)), "but it should be",
                          "{:.2E}".format(Decimal(theoretical_current_C_in_the_system)), "mol of C")
                    print("This corresponds to a net disappearance of C of",
                          "{:.2E}".format(Decimal(theoretical_current_C_in_the_system - current_C_in_the_system)),
                          "mol of C, and the cumulated difference since the start of the simulation and the current one is",
                          "{:.2E}".format(
                              Decimal(theoretical_cumulated_C_in_the_system - current_C_in_the_system)), "mol of C.")

                    # We reinitialize the "previous" amount of C in the system with the current one for the next time step:
                previous_C_in_the_system = current_C_in_the_system

            print("      The root system finally includes", len(g) - 1, "root elements.")


    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # At the end of the simulation (or just before an error is about to interrupt the program!):
    # -------------------------------------------------------------------------------------------
    finally:
        print("")
        print("The program has stopped at time t = {:.2f}".format(Decimal(step * time_step_in_days)), "days.")
        # We can record all the results in a CSV file:
        if recording_sum:
            # We create a data_frame from the vectors generated in the main program up to this point:
            data_frame = pd.DataFrame({"Time (days)": time_in_days_series,
                                       "Sucrose input (mol of sucrose)": sucrose_input_series,
                                       "Root structural mass (g)": total_living_root_struct_mass_series,
                                       "Root necromass (g)": total_dead_root_struct_mass_series,
                                       "Root length (m)": total_living_root_length_series,
                                       "Root surface (m2)": total_living_root_surface_series,
                                       "Sucrose in the root (mol of sucrose)": total_sucrose_root_series,
                                       "Hexose in the mobile pool of the roots (mol of hexose)": total_hexose_root_series,
                                       "Hexose in the reserve pool of the roots (mol of hexose)": total_hexose_reserve_series,
                                       "Hexose in the soil (mol of hexose)": total_hexose_soil_series,

                                       "Deficit of sucrose in the root (mol of sucrose)": total_sucrose_root_deficit_series,
                                       "Deficit of hexose in the mobile pool of the roots (mol of hexose)": total_hexose_root_deficit_series,
                                       "Deficit of hexose in the soil (mol of hexose)": total_hexose_soil_deficit_series,

                                       "CO2 originating from root growth (mol of C)": total_respiration_root_growth_series,
                                       "CO2 originating from root maintenance (mol of C)": total_respiration_root_maintenance_series,
                                       "Structural mass produced (g)": total_structural_mass_production_series,
                                       "Hexose unloaded from phloem (mol of hexose)": total_hexose_production_from_phloem_series,
                                       "Sucrose reloaded in the phloem (mol of hexose)": total_sucrose_loading_in_phloem_series,
                                       "Hexose mobilized from reserve (mol of hexose)": total_hexose_mobilization_from_reserve_series,
                                       "Hexose stored as reserve (mol of hexose)": total_hexose_immobilization_as_reserve_series,
                                       "Hexose emitted in the soil (mol of hexose)": total_hexose_exudation_series,
                                       "Hexose taken up from the soil (mol of hexose)": total_hexose_uptake_series,
                                       "Hexose degraded in the soil (mol of hexose)": total_hexose_degradation_series,

                                       "Cumulated amount of C present in the root-soil system (mol of C)": C_in_the_root_soil_system_series,
                                       "Cumulated amount of C that has been degraded in the soil (mol of C)": C_cumulated_in_the_degraded_pool_series,
                                       "Cumulated amount of C that has been respired by roots (mol of C)": C_cumulated_in_the_gaz_phase_series,
                                       "Final deficit in sucrose of the whole root system (mol of sucrose)": global_sucrose_deficit_series,

                                       "Concentration of hexose in the main root tip (mol of hexose per g)": tip_C_hexose_root_series
                                       },
                                      # We re-order the columns:
                                      columns=["Time (days)",
                                               "Sucrose input (mol of sucrose)",
                                               "Final deficit in sucrose of the whole root system (mol of sucrose)",
                                               "Cumulated amount of C present in the root-soil system (mol of C)",
                                               "Cumulated amount of C that has been respired by roots (mol of C)",
                                               "Cumulated amount of C that has been degraded in the soil (mol of C)",
                                               "Root structural mass (g)",
                                               "Root necromass (g)",
                                               "Root length (m)",
                                               "Root surface (m2)",
                                               "Sucrose in the root (mol of sucrose)",
                                               "Hexose in the mobile pool of the roots (mol of hexose)",
                                               "Hexose in the reserve pool of the roots (mol of hexose)",
                                               "Hexose in the soil (mol of hexose)",
                                               "Deficit of sucrose in the root (mol of sucrose)",
                                               "Deficit of hexose in the mobile pool of the roots (mol of hexose)",
                                               "Deficit of hexose in the soil (mol of hexose)",
                                               "CO2 originating from root growth (mol of C)",
                                               "CO2 originating from root maintenance (mol of C)",
                                               "Structural mass produced (g)",
                                               "Hexose unloaded from phloem (mol of hexose)",
                                               "Sucrose reloaded in the phloem (mol of hexose)",
                                               "Hexose mobilized from reserve (mol of hexose)",
                                               "Hexose stored as reserve (mol of hexose)",
                                               "Hexose emitted in the soil (mol of hexose)",
                                               "Hexose taken up from the soil (mol of hexose)",
                                               "Hexose degraded in the soil (mol of hexose)",
                                               "Concentration of hexose in the main root tip (mol of hexose per g)"
                                               ])
            # We save the data_frame in a CSV file:
            try:
                # In case the results file is not opened, we simply re-write it:
                data_frame.to_csv('simulation_results.csv', na_rep='NA', index=False, header=True)
                print("The main results have been written in the file 'simulation_results.csv'.")
            except:
                # Otherwise we write the data in a new result file as back-up option:
                data_frame.to_csv('simulation_results_BACKUP.csv', na_rep='NA', index=False, header=True)
                print("")
                print("WATCH OUT: The main results have been written in the alternative file 'simulation_results_BACKUP.csv'.")

        # We create another data frame that contains the results classified by z intervals:
        if z_classification:
            # We create a data_frame from the vectors generated in the main program up to this point:
            data_frame_z = pd.DataFrame.from_dict(z_dictionary_series)
            # We save the data_frame in a CSV file:
            data_frame_z.to_csv('z_classification.csv', na_rep='NA', index=False, header=True)


# RUNNING THE SIMULATION:
#########################

if __name__ == "__main__":

    # We set the working directory:
    my_path = r'C:\\Users\\Marion\\Documents\\Marion\\rhizodep\\test'
    if not os.path.exists(my_path):
        my_path = os.path.abspath('.')
    os.chdir(my_path)
    print("The current directory is:", os.getcwd())

    # We record the time when the run starts:
    start_time = timeit.default_timer()

    # We initiate the properties of the MTG "g":
    g = model.initiate_mtg(random=True)
    # We initiate the time variable that will be used to determine the emergence of adventitious roots:
    thermal_time_since_last_adventitious_root_emergence = 0.
    # We initiate the global variable that corresponds to a possible general deficit in sucrose of the whole root system:
    global_sucrose_deficit = 0.

    # We launch the main simulation program:
    print("Simulation starts ...")
    main_simulation(g, simulation_period_in_days=2., time_step_in_days=1. / 24., radial_growth="Possible",
                    ArchiSimple=False,
                    # property="net_hexose_exudation_rate_per_day_per_cm", vmin=1e-9, vmax=1e-6, log_scale=True, cmap='jet',
                    property="C_hexose_root", vmin=1e-4, vmax=1e-1, log_scale=True, cmap='jet',
                    # property="C_sucrose_root", vmin=1e-4, vmax=1e2, log_scale=True, cmap='brg',
                    # property="C_hexose_reserve", vmin=1e-4, vmax=1e4, log_scale=True, cmap='brg',
                    input_file="sucrose_input_0047.csv",
                    constant_sucrose_input_rate=0,
                    constant_soil_temperature_in_Celsius=0,
                    nodules=False,
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
