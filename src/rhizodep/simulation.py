#  -*- coding: utf-8 -*-

"""
    rhizodep.simulation
    ~~~~~~~~~~~~~

    The module :mod:`rhizodep.simulation` is the front-end to run the RhizoDep model.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

from math import trunc
from decimal import Decimal
import pandas as pd
import os
import time

import openalea.plantgl.all as pgl
import rhizodep.model as model
import rhizodep.tools as tools
import rhizodep.parameters as param

import pickle

# TODO: explicitly add 'surfaces_and_volumes()' in the sequence of modelling!

# We define the main simulation program:
def main_simulation(g, simulation_period_in_days=20., time_step_in_days=1.,
                    radial_growth="Impossible", ArchiSimple=False, ArchiSimple_C_fraction=0.10,
                    input_file="None",
                    input_file_time_step_in_days=1/24.,
                    outputs_directory='outputs',
                    forcing_constant_inputs=False, constant_sucrose_input_rate=1.e-6,
                    constant_soil_temperature_in_Celsius=20,
                    nodules=False,
                    root_order_limitation=False,
                    root_order_treshold=2,
                    using_solver=False,
                    printing_solver_outputs=False,
                    simulation_results_file='simulation_results.csv',
                    recording_interval_in_days=5,
                    recording_images=False,
                    root_images_directory="root_images",
                    z_classification=False, z_min=0., z_max=1., z_interval=0.5,
                    z_classification_file='z_classification.csv',
                    printing_sum=True,
                    recording_sum=True,
                    printing_warnings=False,
                    recording_g=False,
                    g_directory="MTG_files",
                    recording_g_properties=True,
                    g_properties_directory="MTG_properties",
                    random=True,
                    plotting=True,
                    scenario_id=1,
                    displayed_property="C_hexose_root", displayed_vmin=1e-6, displayed_vmax=1e-0,
                    log_scale=True, cmap='brg',
                    root_hairs_display=True,
                    width=1200, height=1200,
                    x_center=0, y_center=0, z_center=-1, z_cam=-1,
                    camera_distance=10., step_back_coefficient=0., camera_rotation=False, n_rotation_points=24 * 5):
    """
    This general function controls the actual simulation of root growth and C fluxes over the whole simulation period.
    :param g: the root MTG to consider
    :param simulation_period_in_days: the length of the simulation period (days)
    :param time_step_in_days: the regular time step over the simulation (days)
    :param radial_growth: if True, radial growth will be enabled
    :param ArchiSimple: if True, only original ArchiSimple rules will be used, without C fluxes
    :param ArchiSimple_C_fraction: in case of ArchiSimple only, this fraction is used to determine the fraction of the incoming C that us actually used to produce “root biomass”
    :param input_file: the path/name of the CSV file where inputs (sucrose and temperature) are read
    :param input_file_time_step_in_days: the time step used in the input file (days)
    :param outputs_directory: the name of the folder where simulation outputs will be registered
    :param forcing_constant_inputs: if True, input file will be ignored and a constant sucrose input rate and a constant soil temperature will be applied as inputs
    :param constant_sucrose_input_rate: input of sucrose applied at every time step (mol of sucrose per second per plant)
    :param constant_soil_temperature_in_Celsius: soil temperature applied to every root at every time step (degree Celsius)
    :param nodules: if True, a new type of element - “nodule” - that feeds from the mobile hexose of a mother root will be simulated
    :param root_order_limitation: if True, lateral roots of higher orders will not be formed
    :param root_order_treshold: root order above which no lateral roots can be formed
    :param using_solver: if True, a solver will be used to compute C fluxes and concentrations
    :param printing_solver_outputs: if True, the intermediate calculations of the solver will be printed for each root element
    :param simulation_results_file: the name of the CSV file where outputs will be written
    :param recording_interval_in_days: the time interval of distinct recordings of the current simulation (useful for checking the outputs while the simulation is still running)
    :param recording_images: if True, every PlantGL graph will be recorded as an image
    :param root_images_directory: the name of the folder where root images will be registered
    :param z_classification: if True, root variables will be intercepted and summed within distinct z-layers of the soil
    :param z_min: the upper z-coordinate in the soil, at which we start to compute root data
    :param z_max: the lower z-coordinate in the soil, at which we stop to compute root data
    :param z_interval: the thickness of each soil layer, in which root data will be individually computed
    :param z_classification_file: the name of the CSV file where z-classified root data will be registered
    :param printing_sum: if True, more variables summed over the whole root system will be printed.
    :param recording_sum: if True, variables summed over the whole root system will be registered in a file
    :param printing_warnings: if True, warning messages will be printed, if any
    :param recording_g: if True, the root MTG will be recorded as a pickle file for each time step
    :param g_directory: the name of the folder where MTG files are recorded
    :param recording_g_properties: if True, all the properties of all the root elements of the root MTG will be registered in a file
    :param g_properties_directory: the name of the folder where MTG properties are recorded
    :param random: if True, stochastic data will be simulated (e.g. variations of angles and diameters)
    :param plotting: if True, a PlantGL graph is generated and displayed at each time step
    :param scenario_id: indicates the scenario identifier to which the current printing relates (useful when running different scenarios in parallel)
    :param displayed_property: name of the property to be displayed on the PlantGL graph
    :param displayed_vmin: the minimum value of the scale used to display the property in the PlantGL graph
    :param displayed_vmax: the maximum value of the scale used to display the property in the PlantGL graph
    :param log_scale: if True, the scale of values displayed in the PlantGL Graph will be in log-scale
    :param cmap: name of the color map to be used when displaying the property in the PlantGL graph
    :param root_hairs_display: if True, root hairs density will be displayed on the PlantGL graph
    :param x_center: x-coordinate of the center of the PlantGL graph
    :param y_center: y-coordinate of the center of the PlantGL graph
    :param z_center: z-coordinate of the center of the PlantGL graph
    :param z_cam: z-coordinate of the camera looking at the center of the PlantGL graph
    :param camera_distance: distance between the camera and the center of the PlantGL graph
    :param step_back_coefficient: if this coefficient is not 0, the distance between the camera and the center of the PlantGL graph will be proportionally increased at every time step (useful when the MTG gets too large)
    :param camera_rotation: if True, the camera will rotate around the center of the PlantGL graph
    :param n_rotation_points: number of intermediates positions of the camera when rotating around the center of the PlantGL graph before coming back to the same position
    :return:
    """

    # We convert the time step in seconds:
    time_step_in_seconds = time_step_in_days * 60. * 60. * 24.
    # We calculate the number of steps necessary to reach the end of the simulation period:
    if simulation_period_in_days == 0. or time_step_in_days == 0.:
        print("WATCH OUT: No simulation was done, as time input was 0.")
        n_steps = 0
    else:
        n_steps = trunc(simulation_period_in_days / time_step_in_days)

    # We initialize empty variables at t=0:
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
    total_living_root_hairs_surface_series = []
    total_living_root_struct_mass_series = []
    total_dead_root_struct_mass_series = []
    total_root_hairs_mass_series = []
    total_sucrose_root_series = []
    total_hexose_root_series = []
    total_hexose_reserve_series = []
    total_hexose_soil_series = []
    total_mucilage_soil_series = []
    total_cells_soil_series = []

    total_sucrose_root_deficit_series = []
    total_hexose_root_deficit_series = []
    total_hexose_reserve_deficit_series = []
    total_hexose_soil_deficit_series = []
    total_mucilage_soil_deficit_series = []
    total_cells_soil_deficit_series = []

    total_respiration_series = []
    total_respiration_root_growth_series = []
    total_respiration_root_maintenance_series = []
    total_structural_mass_production_series = []
    total_hexose_production_from_phloem_series = []
    total_sucrose_loading_in_phloem_series = []
    total_hexose_mobilization_from_reserve_series = []
    total_hexose_immobilization_as_reserve_series = []
    total_hexose_exudation_series = []
    total_phloem_hexose_exudation_series = []
    total_hexose_uptake_series = []
    total_phloem_hexose_uptake_series = []
    total_mucilage_secretion_series = []
    total_cells_release_series = []
    total_net_hexose_exudation_series = []
    total_net_rhizodeposition_series = []
    total_hexose_degradation_series = []
    total_mucilage_degradation_series = []
    total_cells_degradation_series = []
    C_in_the_root_soil_system_series = []
    C_cumulated_in_the_degraded_pool_series = []
    C_cumulated_in_the_gaz_phase_series = []
    global_sucrose_deficit_series = []
    tip_C_hexose_root_series = []
    soil_temperature_series = []

    # We create an empty list that will contain the results of z classification:
    z_dictionary_series = []

    if recording_images:
        # We define the directory of root images if it doesn't exist:
        if not os.path.exists(root_images_directory):
            # Then we create it:
            os.mkdir(root_images_directory)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(root_images_directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g:
        # We define the directory "MTG_files" doesn't exist:
        if not os.path.exists(g_directory):
            # Then we create it:
            os.mkdir(g_directory)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(g_directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g_properties:
        # We define the directory "MTG_properties" doesn't exist:
        if not os.path.exists(g_properties_directory):
            # Then we create it:
            os.mkdir(g_properties_directory)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(g_properties_directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    # READING THE INPUT FILE:
    # -----------------------
    if input_file != "None" and not forcing_constant_inputs:  # and (constant_sucrose_input_rate <= 0 or constant_soil_temperature_in_Celsius <= 0):
        # # We first define the path and the file to read as a .csv:
        # PATH = os.path.join('.', input_file)
        # # Then we read the file and copy it in a dataframe "df":
        # input_frame = pd.read_csv(PATH, sep=',')
        # We use the function 'formatted inputs' to create a table containing the input data (soil temperature and sucrose input)
        # for each required step, depending on the chosen time step:
        input_frame = tools.formatted_inputs(original_input_file=input_file,
                                             final_input_file=os.path.join(outputs_directory, 'updated_input_file.csv'),
                                             original_time_step_in_days=input_file_time_step_in_days,
                                             final_time_step_in_days=time_step_in_days,
                                             simulation_period_in_days=simulation_period_in_days,
                                             do_not_execute_if_file_with_suitable_size_exists=False)

        # We then initialize the step and time according to the first line of the inputs dataframe:
        initial_step_number = input_frame['step_number'].loc[0]
        initial_time_in_days = input_frame['initial_time_in_days'].loc[0]
    else:
        # We then initialize the step and time according to the first line of the inputs dataframe:
        initial_step_number = 0
        initial_time_in_days = 0.

    step = initial_step_number

    # If we want to save all results at specified time intervals, we can create a list corresponding time steps:
    recording_steps_list = list(range(initial_step_number - 1, initial_step_number + n_steps,
                                      round(recording_interval_in_days / time_step_in_days)))

    # RECORDING THE INITIAL STATE OF THE MTG:
    # ---------------------------------------

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
        sc = tools.plot_mtg(g, prop_cmap=displayed_property, lognorm=log_scale, vmin=displayed_vmin,
                            vmax=displayed_vmax, cmap=cmap,
                            root_hairs_display=root_hairs_display,
                            width=width,
                            height=height,
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
        sc = tools.plot_mtg(g, prop_cmap=displayed_property, lognorm=log_scale, vmin=displayed_vmin,
                            vmax=displayed_vmax, cmap=cmap,
                            root_hairs_display=root_hairs_display,
                            width=width,
                            height=height,
                            x_center=x_center,
                            y_center=y_center,
                            z_center=z_center,
                            x_cam=x_camera,
                            y_cam=0,
                            z_cam=z_camera)
        # We move the camera further from the root system:
        x_camera = x_cam + x_cam * step_back_coefficient * (step - initial_step_number)
        z_camera = z_cam + z_cam * step_back_coefficient * (step - initial_step_number)

    # We finally display the MTG on PlantGL and possibly record it:
    if plotting:
        pgl.Viewer.display(sc)
        if recording_images:
            # pgl.Viewer.frameGL.setSize(width, height)
            image_name = os.path.join(root_images_directory, 'root%.5d.png')
            pgl.Viewer.saveSnapshot(image_name % step)

    # For integrating root variables on the z axis:
    # ----------------------------------------------
    if z_classification:
        z_dictionary = model.classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
        z_dictionary["time_in_days"] = 0.0
        z_dictionary_series.append(z_dictionary)

    # For recording the initial MTG to load it later on:
    # --------------------------------------------------
    g_file_name = os.path.join(g_directory, 'root%.5d.pckl')
    with open(g_file_name % step, 'wb') as output_file:
        pickle.dump(g, output_file, protocol=2)

    # For recording the properties of g in a csv file at each time step :
    # -------------------------------------------------------------------
    if recording_g_properties:
        prop_file_name = os.path.join(g_properties_directory, 'root%.5d.csv')
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
        total_root_hairs_mass_series.append(dictionary["total_root_hairs_mass"])
        total_living_root_surface_series.append(dictionary["total_living_root_surface"])
        total_dead_root_surface_series.append(dictionary["total_dead_root_surface"])
        total_living_root_hairs_surface_series.append(dictionary["total_living_root_hairs_surface"])
        total_sucrose_root_series.append(dictionary["total_sucrose_root"])
        total_hexose_root_series.append(dictionary["total_hexose_root"])
        total_hexose_reserve_series.append(dictionary["total_hexose_reserve"])
        total_hexose_soil_series.append(dictionary["total_hexose_soil"])
        total_mucilage_soil_series.append(dictionary["total_mucilage_soil"])
        total_cells_soil_series.append(dictionary["total_cells_soil"])

        total_sucrose_root_deficit_series.append(dictionary["total_sucrose_root_deficit"])
        total_hexose_root_deficit_series.append(dictionary["total_hexose_root_deficit"])
        total_hexose_reserve_deficit_series.append(dictionary["total_hexose_reserve_deficit"])
        total_hexose_soil_deficit_series.append(dictionary["total_hexose_soil_deficit"])
        total_mucilage_soil_deficit_series.append(dictionary["total_mucilage_soil_deficit"])
        total_cells_soil_deficit_series.append(dictionary["total_cells_soil_deficit"])

        total_respiration_series.append(dictionary["total_respiration"])
        total_respiration_root_growth_series.append(dictionary["total_respiration_root_growth"])
        total_respiration_root_maintenance_series.append(dictionary["total_respiration_root_maintenance"])
        total_structural_mass_production_series.append(dictionary["total_structural_mass_production"])
        total_hexose_production_from_phloem_series.append(dictionary["total_hexose_production_from_phloem"])
        total_sucrose_loading_in_phloem_series.append(dictionary["total_sucrose_loading_in_phloem"])
        total_hexose_mobilization_from_reserve_series.append(dictionary["total_hexose_mobilization_from_reserve"])
        total_hexose_immobilization_as_reserve_series.append(dictionary["total_hexose_immobilization_as_reserve"])
        total_hexose_exudation_series.append(dictionary["total_hexose_exudation"])
        total_phloem_hexose_exudation_series.append(dictionary["total_phloem_hexose_exudation"])
        total_hexose_uptake_series.append(dictionary["total_hexose_uptake_from_soil"])
        total_phloem_hexose_uptake_series.append(dictionary["total_phloem_hexose_uptake_from_soil"])
        total_mucilage_secretion_series.append(dictionary["total_mucilage_secretion"])
        total_cells_release_series.append(dictionary["total_cells_release"])
        total_net_rhizodeposition_series.append(dictionary["total_net_rhizodeposition"])
        total_net_hexose_exudation_series.append(dictionary["total_net_hexose_exudation"])
        total_hexose_degradation_series.append(dictionary["total_hexose_degradation"])
        total_mucilage_degradation_series.append(dictionary["total_mucilage_degradation"])
        total_cells_degradation_series.append(dictionary["total_cells_degradation"])

        C_in_the_root_soil_system_series.append(dictionary["C_in_the_root_soil_system"])
        C_cumulated_in_the_degraded_pool += dictionary["C_degraded_in_the_soil"]
        C_cumulated_in_the_degraded_pool_series.append(C_cumulated_in_the_degraded_pool)
        C_cumulated_in_the_gaz_phase += dictionary["C_respired_by_roots"]
        C_cumulated_in_the_gaz_phase_series.append(C_cumulated_in_the_gaz_phase)

        # Values that are integrative for the whole root system have been stored as properties in node 0 of the root system:
        global_sucrose_deficit_series.append(g.node(0).global_sucrose_deficit)
        tip_C_hexose_root_series.append(g.node(0).C_hexose_root)

        # Adding soil temperature:
        soil_temperature_series.append("NA")

        # Initializing the amount of C in the root_soil_CO2 system:
        previous_C_in_the_system = dictionary["C_in_the_root_soil_system"]
        theoretical_cumulated_C_in_the_system = previous_C_in_the_system

    # ------------------------------------------------------------------------------------------------------------------
    # We create an internal function for saving the sum properties, the MTG file and the z_classification_file (if any)
    # over the course of the simulation:
    def recording_attempt():

        # In any case, we record the MTG file:
        g_file_name = os.path.join(g_directory, 'root%.5d.pckl')
        with open(g_file_name % step, 'wb') as output:
            pickle.dump(g, output, protocol=2)
        print("The MTG file corresponding to the root system has been recorded.")

        # We can record all the results in a CSV file:
        if recording_sum:
            # We create a data_frame from the vectors generated in the main program up to this point:
            data_frame = pd.DataFrame({"Final time (days)": time_in_days_series,
                                       "Sucrose input (mol of sucrose)": sucrose_input_series,
                                       "Root structural mass (g)": total_living_root_struct_mass_series,
                                       "Root necromass (g)": total_dead_root_struct_mass_series,
                                       "Root length (m)": total_living_root_length_series,
                                       "Root surface (m2)": total_living_root_surface_series,
                                       "Total root hairs mass (g)": total_root_hairs_mass_series,
                                       "Total living root hairs surface (m2)": total_living_root_hairs_surface_series,
                                       "Sucrose in the root (mol of sucrose)": total_sucrose_root_series,
                                       "Hexose in the mobile pool of the roots (mol of hexose)": total_hexose_root_series,
                                       "Hexose in the reserve pool of the roots (mol of hexose)": total_hexose_reserve_series,
                                       "Hexose in the soil (mol of hexose)": total_hexose_soil_series,
                                       "Mucilage in the soil (mol of hexose)": total_mucilage_soil_series,
                                       "Cells in the soil (mol of equivalent-hexose)": total_cells_soil_series,

                                       "Deficit of sucrose in the root (mol of sucrose)": total_sucrose_root_deficit_series,
                                       "Deficit of hexose in the mobile pool of the roots (mol of hexose)": total_hexose_root_deficit_series,
                                       "Deficit of hexose in the reserve pool of the roots (mol of hexose)": total_hexose_reserve_deficit_series,
                                       "Deficit of hexose in the soil (mol of hexose)": total_hexose_soil_deficit_series,
                                       "Deficit of mucilage in the soil (mol of hexose)": total_mucilage_soil_deficit_series,
                                       "Deficit of cells in the soil (mol of hexose)": total_cells_soil_deficit_series,

                                       "CO2 originating from root growth (mol of C)": total_respiration_root_growth_series,
                                       "CO2 originating from root maintenance (mol of C)": total_respiration_root_maintenance_series,
                                       "Structural mass produced (g)": total_structural_mass_production_series,
                                       "Hexose unloaded from phloem (mol of hexose)": total_hexose_production_from_phloem_series,
                                       "Sucrose reloaded in the phloem (mol of hexose)": total_sucrose_loading_in_phloem_series,
                                       "Hexose mobilized from reserve (mol of hexose)": total_hexose_mobilization_from_reserve_series,
                                       "Hexose stored as reserve (mol of hexose)": total_hexose_immobilization_as_reserve_series,
                                       "Hexose emitted in the soil from parenchyma (mol of hexose)": total_hexose_exudation_series,
                                       "Hexose emitted in the soil from phloem (mol of hexose)": total_phloem_hexose_exudation_series,
                                       "Hexose taken up from the soil by parenchyma (mol of hexose)": total_hexose_uptake_series,
                                       "Hexose taken up from the soil by phloem (mol of hexose)": total_phloem_hexose_uptake_series,
                                       "Mucilage secreted in the soil (mol of hexose)": total_mucilage_secretion_series,
                                       "Cells release in the soil (mol of equivalent-hexose)": total_cells_release_series,
                                       "Total net rhizodeposition (mol of hexose)": total_net_rhizodeposition_series,
                                       "Hexose degraded in the soil (mol of hexose)": total_hexose_degradation_series,
                                       "Mucilage degraded in the soil (mol of hexose)": total_mucilage_degradation_series,
                                       "Cells degraded in the soil (mol of equivalent-hexose)": total_cells_degradation_series,

                                       "Cumulated amount of C present in the root-soil system (mol of C)": C_in_the_root_soil_system_series,
                                       "Cumulated amount of C that has been degraded in the soil (mol of C)": C_cumulated_in_the_degraded_pool_series,
                                       "Cumulated amount of C that has been respired by roots (mol of C)": C_cumulated_in_the_gaz_phase_series,
                                       "Final deficit in sucrose of the whole root system (mol of sucrose)": global_sucrose_deficit_series,

                                       "Concentration of hexose in the main root tip (mol of hexose per g)": tip_C_hexose_root_series,

                                       "Soil temperature (degree Celsius)": soil_temperature_series
                                       },
                                      # We re-order the columns:
                                      columns=["Final time (days)",
                                               "Sucrose input (mol of sucrose)",
                                               "Final deficit in sucrose of the whole root system (mol of sucrose)",
                                               "Cumulated amount of C present in the root-soil system (mol of C)",
                                               "Cumulated amount of C that has been respired by roots (mol of C)",
                                               "Cumulated amount of C that has been degraded in the soil (mol of C)",
                                               "Root structural mass (g)",
                                               "Root necromass (g)",
                                               "Root length (m)",
                                               "Root surface (m2)",
                                               "Total root hairs mass (g)",
                                               "Total living root hairs surface (m2)",
                                               "Sucrose in the root (mol of sucrose)",
                                               "Hexose in the mobile pool of the roots (mol of hexose)",
                                               "Hexose in the reserve pool of the roots (mol of hexose)",
                                               "Hexose in the soil (mol of hexose)",
                                               "Mucilage in the soil (mol of hexose)",
                                               "Cells in the soil (mol of equivalent-hexose)",
                                               "Deficit of sucrose in the root (mol of sucrose)",
                                               "Deficit of hexose in the mobile pool of the roots (mol of hexose)",
                                               "Deficit of hexose in the reserve pool of the roots (mol of hexose)",
                                               "Deficit of hexose in the soil (mol of hexose)",
                                               "Deficit of mucilage in the soil (mol of hexose)",
                                               "Deficit of cells in the soil (mol of hexose)",
                                               "CO2 originating from root growth (mol of C)",
                                               "CO2 originating from root maintenance (mol of C)",
                                               "Structural mass produced (g)",
                                               "Hexose unloaded from phloem (mol of hexose)",
                                               "Sucrose reloaded in the phloem (mol of hexose)",
                                               "Hexose mobilized from reserve (mol of hexose)",
                                               "Hexose stored as reserve (mol of hexose)",
                                               "Hexose emitted in the soil from parenchyma (mol of hexose)",
                                               "Hexose emitted in the soil from phloem (mol of hexose)",
                                               "Hexose taken up from the soil by parenchyma (mol of hexose)",
                                               "Hexose taken up from the soil by phloem (mol of hexose)",
                                               "Mucilage secreted in the soil (mol of hexose)",
                                               "Cells release in the soil (mol of equivalent-hexose)",
                                               "Total net rhizodeposition (mol of hexose)",
                                               "Hexose degraded in the soil (mol of hexose)",
                                               "Mucilage degraded in the soil (mol of hexose)",
                                               "Cells degraded in the soil (mol of equivalent-hexose)",
                                               "Concentration of hexose in the main root tip (mol of hexose per g)",
                                               "Soil temperature (degree Celsius)"
                                               ])
            # We save the data_frame in a CSV file:
            try:
                # In case the results file is not opened, we simply re-write it:
                data_frame.to_csv(os.path.join(outputs_directory, simulation_results_file),
                                  na_rep='NA', index=False, header=True)
                print("The main results have been written in the file", simulation_results_file, " .")
            except Exception as ex:
                # Otherwise we write the data in a new result file as back-up option:
                data_frame.to_csv(os.path.join(outputs_directory, 'simulation_results_BACKUP.csv'),
                                  na_rep='NA', index=False, header=True)
                print("")
                print("WATCH OUT: "
                      "The main results have been written in the alternative file 'simulation_results_BACKUP.csv'.")

        # We create another data frame that contains the results classified by z intervals:
        if z_classification:
            # We create a data_frame from the vectors generated in the main program up to this point:
            # data_frame_z = pd.DataFrame.from_dict(z_dictionary_series)
            data_frame_z = pd.DataFrame.from_dict(z_dictionary_series)
            # We save the data_frame in a CSV file:
            data_frame_z.to_csv(os.path.join(outputs_directory, 'z_classification.csv'),
                                na_rep='NA', index=False, header=True)
            print("The data classified by layers has been written in the file 'z_classification.csv'.")

        return

    # ------------------------------------------------------------------------------------------------------------------

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # The code will try to run the following code until it is finished or an error has been raised:
    try:

        # An iteration is done for each time step:
        for step in range(initial_step_number, initial_step_number + n_steps):

            # We calculate the current time in hours:
            current_time_in_hours = step * time_step_in_days * 24.

            # DEFINING THE INPUT OF CARBON TO THE ROOTS FOR THIS TIME STEP:
            # --------------------------------------------------------------
            if forcing_constant_inputs:
                sucrose_input_rate = constant_sucrose_input_rate
            else:
                sucrose_input_rate = input_frame.loc[step - initial_step_number, 'sucrose_input_rate']

            # DEFINING THE TEMPERATURE OF THE SOIL FOR THIS TIME STEP:
            # --------------------------------------------------------
            if constant_soil_temperature_in_Celsius > 0 and forcing_constant_inputs:  # or input_file == "None":
                soil_temperature = constant_soil_temperature_in_Celsius
            else:
                soil_temperature = input_frame.loc[step - initial_step_number, 'soil_temperature_in_degree_Celsius']

            # CALCULATING AN EQUIVALENT OF THERMAL TIME:
            # -------------------------------------------

            # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
            temperature_time_adjustment = model.temperature_modification(temperature_in_Celsius=soil_temperature,
                                                                         process_at_T_ref=1,
                                                                         T_ref=param.root_growth_T_ref,
                                                                         A=param.root_growth_A,
                                                                         B=param.root_growth_B,
                                                                         C=param.root_growth_C)

            # STARTING THE ACTUAL SIMULATION:
            # --------------------------------
            print("")
            print("(SCENARIO {})".format(scenario_id))
            print("From t =", "{:.2f}".format(Decimal((step) * time_step_in_days)), "days to t =",
                  "{:.2f}".format(Decimal((step+1) * time_step_in_days)), "days:")
            print("------------------------------------")
            print("   Soil temperature is", "{:.2f}".format(Decimal(soil_temperature)), "degree Celsius.")
            print("   The input rate of sucrose to the root for time=", "{:.2f}".format(Decimal(current_time_in_hours)),
                  "h is",
                  "{:.2E}".format(Decimal(sucrose_input_rate)), "mol of sucrose per second, i.e.",
                  "{:.2E}".format(Decimal(sucrose_input_rate * 60. * 60. * 24.)), "mol of sucrose per day.")

            print("   The root system initially includes", len(g) - 1, "root elements.")

            # CASE 1: WE REPRODUCE THE GROWTH WITHOUT CONSIDERATIONS OF LOCAL CONCENTRATIONS
            # -------------------------------------------------------------------------------

            if ArchiSimple:

                # The input of C (mol of C) from shoots is calculated from the input of sucrose
                # assuming that only a fraction of it st used for root growth (paramteter ArchiSimple_C_fraction):
                C_input_for_growth = sucrose_input_rate * 12 * time_step_in_seconds * ArchiSimple_C_fraction
                # We convert this input of C into an input of "biomass" as considered in ArchiSimple:
                struct_mass_input = C_input_for_growth / param.struct_mass_C_content

                # We reset to 0 all growth-associated C costs:
                model.reinitializing_growth_variables(g)

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

                # We proceed to the segmentation of the whole root system
                # (NOTE: segmentation should always occur AFTER actual growth):
                model.segmentation_and_primordia_formation(g, time_step_in_seconds, printing_warnings=printing_warnings,
                                                           soil_temperature_in_Celsius=soil_temperature, random=random,
                                                           ArchiSimple=ArchiSimple,
                                                           nodules=nodules,
                                                           root_order_limitation=root_order_limitation,
                                                           root_order_treshold=root_order_treshold)

            else:

                # CASE 2: WE PERFORM THE COMPLETE MODEL WITH C BALANCE IN EACH ROOT ELEMENT
                # --------------------------------------------------------------------------

                # 2a - ROOT GROWTH
                # ================

                # We reset to 0 all growth-associated C costs and initialize the initial dimensions or masses:
                model.reinitializing_growth_variables(g)

                # We calculate the potential growth of the root system without consideration of the amount of available hexose:
                model.potential_growth(g, time_step_in_seconds=time_step_in_seconds,
                                         radial_growth=radial_growth,
                                         soil_temperature_in_Celsius=soil_temperature,
                                         ArchiSimple=False)

                # We calculate the actual growth based on the amount of hexose remaining in the roots, and we record
                # the corresponding consumption of hexose in the root:
                model.actual_growth_and_corresponding_respiration(g, time_step_in_seconds=time_step_in_seconds,
                                                                    soil_temperature_in_Celsius=soil_temperature,
                                                                    printing_warnings=printing_warnings)

                # 2b - NEW GEOMETRY
                # =================

                # We proceed to the segmentation of the whole root system
                # (NOTE: segmentation should always occur AFTER actual growth):
                model.segmentation_and_primordia_formation(g, time_step_in_seconds,
                                                             soil_temperature_in_Celsius=soil_temperature,
                                                             random=random,
                                                             nodules=nodules,
                                                             root_order_limitation=root_order_limitation,
                                                             root_order_treshold=root_order_treshold)

                # We update the distance from tip for each root element in each root axis:
                model.update_distance_from_tip(g)

                # We update the surfaces and the volume for each root element in each root axis:
                model.update_surfaces_and_volumes(g)

                # 2c - SPECIFIC ROOT HAIR DYNAMICS
                # ================================

                # We modifiy root hairs characteristics according to their specific dynamics:
                model.root_hairs_dynamics(g, time_step_in_seconds=time_step_in_seconds,
                                          soil_temperature_in_Celsius=soil_temperature,
                                          printing_warnings=printing_warnings)

                # 2d - CARBON EXCHANGE
                # ====================

                # We now proceed to all the exchanges of C for each root element in each root axis
                # (NOTE: the supply of sucrose from the shoot is excluded in this calculation):
                tip_C_hexose_root = \
                    model.C_exchange_and_balance_in_roots_and_at_the_root_soil_interface(
                        g,
                        time_step_in_seconds=time_step_in_seconds,
                        soil_temperature_in_Celsius=soil_temperature,
                        using_solver=using_solver,
                        printing_solver_outputs=printing_solver_outputs,
                        printing_warnings=printing_warnings)
                # NOTE: we also use this function to record specifically the concentration of hexose in the root apex
                # of the primary root!

                # Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
                model.shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=sucrose_input_rate,
                                                         time_step_in_seconds=time_step_in_seconds,
                                                         printing_warnings=printing_warnings)
                # WARNING: The function "shoot_sucrose_supply_and_spreading" must be called AFTER the function "balance",
                # otherwise the deficit in sucrose may be counted twice!!!
                # TODO: check this affirmation about the position of shoot_sucrose_supply_and_spreading

                # # OPTIONAL: checking of possible anomalies in the root system:
                # model.control_of_anomalies(g)

            # PLOTTING THE MTG:
            # ------------------

            # If the rotation of the camera around the root system is required:
            if camera_rotation:
                x_cam = x_coordinates[index_camera]
                y_cam = y_coordinates[index_camera]
                z_cam = z_coordinates[index_camera]
                sc = tools.plot_mtg(g, prop_cmap=displayed_property, lognorm=log_scale, vmin=displayed_vmin,
                                    vmax=displayed_vmax, cmap=cmap,
                                    root_hairs_display=root_hairs_display,
                                    width=width,
                                    height=height,
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
                sc = tools.plot_mtg(g, prop_cmap=displayed_property, lognorm=log_scale, vmin=displayed_vmin,
                                    vmax=displayed_vmax, cmap=cmap,
                                    root_hairs_display=root_hairs_display,
                                    width=width,
                                    height=height,
                                    x_center=x_center,
                                    y_center=y_center,
                                    z_center=z_center,
                                    x_cam=x_camera,
                                    y_cam=0,
                                    z_cam=z_camera)
                # We move the camera further from the root system:
                x_camera = x_cam + x_cam * step_back_coefficient * (step - initial_step_number)
                z_camera = z_cam + z_cam * step_back_coefficient * (step - initial_step_number)

            # We finally display the MTG on PlantGL and possibly record it:
            if plotting:
                pgl.Viewer.display(sc)
                # If needed, we wait for a few seconds so that the graph is well positioned:
                time.sleep(0.5)
                if recording_images:
                    image_name = os.path.join(root_images_directory, 'root%.5d.png')
                    pgl.Viewer.saveSnapshot(image_name % step)

            # For integrating root variables on the z axis:
            # ----------------------------------------------
            if z_classification:
                z_dictionary = model.classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
                z_dictionary["time_in_days"] = time_step_in_days * step
                z_dictionary_series.append(z_dictionary)

            # For recording the MTG at each time step to load it later on:
            # ------------------------------------------------------------
            if recording_g:
                g_file_name = os.path.join(g_directory, 'root%.5d.pckl')
                with open(g_file_name % step, 'wb') as output:
                    pickle.dump(g, output, protocol=2)

            # For recording the properties of g in a csv file at each time step:
            # ------------------------------------------------------------------
            if recording_g_properties:
                prop_file_name = os.path.join(g_properties_directory, 'root%.5d.csv')
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
                time_in_days_series.append(time_step_in_days * (step+1))
                sucrose_input_series.append(sucrose_input_rate * time_step_in_seconds)
                total_living_root_length_series.append(dictionary["total_living_root_length"])
                total_dead_root_length_series.append(dictionary["total_dead_root_length"])
                total_living_root_struct_mass_series.append(dictionary["total_living_root_struct_mass"])
                total_dead_root_struct_mass_series.append(dictionary["total_dead_root_struct_mass"])
                total_root_hairs_mass_series.append(dictionary["total_root_hairs_mass"])
                total_living_root_surface_series.append(dictionary["total_living_root_surface"])
                total_dead_root_surface_series.append(dictionary["total_dead_root_surface"])
                total_living_root_hairs_surface_series.append(dictionary["total_living_root_hairs_surface"])
                total_sucrose_root_series.append(dictionary["total_sucrose_root"])
                total_hexose_root_series.append(dictionary["total_hexose_root"])
                total_hexose_reserve_series.append(dictionary["total_hexose_reserve"])
                total_hexose_soil_series.append(dictionary["total_hexose_soil"])
                total_mucilage_soil_series.append(dictionary["total_mucilage_soil"])
                total_cells_soil_series.append(dictionary["total_cells_soil"])

                total_sucrose_root_deficit_series.append(dictionary["total_sucrose_root_deficit"])
                total_hexose_root_deficit_series.append(dictionary["total_hexose_root_deficit"])
                total_hexose_reserve_deficit_series.append(dictionary["total_hexose_reserve_deficit"])
                total_hexose_soil_deficit_series.append(dictionary["total_hexose_soil_deficit"])
                total_mucilage_soil_deficit_series.append(dictionary["total_mucilage_soil_deficit"])
                total_cells_soil_deficit_series.append(dictionary["total_cells_soil_deficit"])

                total_respiration_series.append(dictionary["total_respiration"])
                total_respiration_root_growth_series.append(dictionary["total_respiration_root_growth"])
                total_respiration_root_maintenance_series.append(dictionary["total_respiration_root_maintenance"])
                total_structural_mass_production_series.append(dictionary["total_structural_mass_production"])
                total_hexose_production_from_phloem_series.append(dictionary["total_hexose_production_from_phloem"])
                total_sucrose_loading_in_phloem_series.append(dictionary["total_sucrose_loading_in_phloem"])
                total_hexose_mobilization_from_reserve_series.append(dictionary["total_hexose_mobilization_from_reserve"])
                total_hexose_immobilization_as_reserve_series.append(dictionary["total_hexose_immobilization_as_reserve"])
                total_hexose_exudation_series.append(dictionary["total_hexose_exudation"])
                total_phloem_hexose_exudation_series.append(dictionary["total_phloem_hexose_exudation"])
                total_hexose_uptake_series.append(dictionary["total_hexose_uptake_from_soil"])
                total_phloem_hexose_uptake_series.append(dictionary["total_phloem_hexose_uptake_from_soil"])
                total_mucilage_secretion_series.append(dictionary["total_mucilage_secretion"])
                total_cells_release_series.append(dictionary["total_cells_release"])
                total_net_rhizodeposition_series.append(dictionary["total_net_rhizodeposition"])
                total_net_hexose_exudation_series.append(dictionary["total_net_hexose_exudation"])
                total_hexose_degradation_series.append(dictionary["total_hexose_degradation"])
                total_mucilage_degradation_series.append(dictionary["total_mucilage_degradation"])
                total_cells_degradation_series.append(dictionary["total_cells_degradation"])

                C_in_the_root_soil_system_series.append(dictionary["C_in_the_root_soil_system"])
                C_cumulated_in_the_degraded_pool += dictionary["C_degraded_in_the_soil"]
                C_cumulated_in_the_degraded_pool_series.append(C_cumulated_in_the_degraded_pool)
                C_cumulated_in_the_gaz_phase += dictionary["C_respired_by_roots"]
                C_cumulated_in_the_gaz_phase_series.append(C_cumulated_in_the_gaz_phase)

                soil_temperature_series.append(soil_temperature)

                # In case of ArchiSimple:
                if ArchiSimple:
                    # Some lists have not been built, so we assign 0 to lists with proper size here
                    global_sucrose_deficit_series = ['0'] * len(time_in_days_series)
                    tip_C_hexose_root_series = ['0'] * len(time_in_days_series)
                else:
                    # Otherwise, these lists are appended with the properties stored in element 0 of the root system:
                    global_sucrose_deficit_series.append(g.node(0).global_sucrose_deficit)
                    tip_C_hexose_root_series.append(tip_C_hexose_root)

                    # CHECKING CARBON BALANCE:
                    ##########################

                    current_C_in_the_system = dictionary["C_in_the_root_soil_system"] \
                                              + C_cumulated_in_the_gaz_phase + C_cumulated_in_the_degraded_pool
                    theoretical_current_C_in_the_system = (previous_C_in_the_system
                                                           + sucrose_input_rate * time_step_in_seconds * 12.)
                    theoretical_cumulated_C_in_the_system += sucrose_input_rate * time_step_in_seconds * 12.

                    if abs(current_C_in_the_system - theoretical_current_C_in_the_system) / current_C_in_the_system > 1e-9:
                        print("!!! ERROR ON CARBON BALANCE: the current amount of C in the system is",
                              "{:.2E}".format(Decimal(current_C_in_the_system)), "but it should be",
                              "{:.2E}".format(Decimal(theoretical_current_C_in_the_system)), "mol of C")
                        print("This corresponds to a net disappearance of C of",
                              "{:.2E}".format(Decimal(theoretical_current_C_in_the_system - current_C_in_the_system)),
                              "mol of C, and the cumulated difference since the start of the simulation and the current one is",
                              "{:.2E}".format(
                                  Decimal(theoretical_cumulated_C_in_the_system - current_C_in_the_system)),
                              "mol of C.")

                        # We reinitialize the "previous" amount of C in the system with the current one for the next time step:
                    previous_C_in_the_system = current_C_in_the_system

            print("   The root system finally includes", len(g) - 1, "root elements.")
            print("(SCENARIO {})".format(scenario_id))

            # If the current iteration correspond to the time where one full time interval for recording has been reached:
            if step in recording_steps_list:
                # Then we record the current simulation results:
                print("Recording the simulation results obtained so far (until time t = {:.2f}".format(
                    Decimal((step+1) * time_step_in_days)), "days)...")
                recording_attempt()

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # At the end of the simulation (or just before an error is about to interrupt the program!):
    # -------------------------------------------------------------------------------------------
    finally:
        print("")
        print("The program has stopped at final time t = {:.2f}".format(Decimal((step+1) * time_step_in_days)), "days.")
        recording_attempt()

        # # For preventing the interpreter to close all windows at the end of the program, we ask the user to press a key
        # # before closing all the windows:
        # ask_before_end = input()

    return
