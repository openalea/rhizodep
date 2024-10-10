import os
import numpy as np
from openalea.mtg import *
import openalea.plantgl.all as pgl
import rhizodep.parameters as param
import rhizodep.tools as tools
import pickle
import time

def initiate_mycorrhizal_fungus():
    """
    This function initiates a new fungus MTG.
    :return: the new fungus
    """
    # We initiate a fungus as an empty MTG:
    fungus = MTG()

    # We set the initial properties:
    fungus.struct_mass = 0.
    fungus.root_exchange_surface = 0.
    fungus.C_hexose_fungus = 0.
    fungus.Deficit_hexose_fungus = 0.
    fungus.overall_infection_severity = 0.

    fungus.hexose_input_rate_from_root = 0.
    fungus.hexose_consumption_for_growth =0.
    fungus.struct_mass_increment = 0.
    fungus.surface_increment_by_expansion = 0.

    return fungus

def reinitializing_fungus_growth(fungus):
    """
    This function resets the values of different variables within a given fungus.
    :param fungus:
    :return:
    """

    fungus.hexose_consumption_for_growth = 0.
    fungus.struct_mass_increment = 0.

    fungus.initial_struct_mass = fungus.struct_mass
    fungus.initial_exchange_surface = fungus.root_exchange_surface

    return

def root_infection_by_fungus(root_MTG, fungus, step, time_step_in_seconds):
    """
    This function computes the possible new infection of all root elements of a given root MTG by the fungus
    and updates the infection severity and the new exchange severity between the fungus and the root elements.
    :param root_MTG:
    :param fungus:
    :param step:
    :param time_step_in_seconds:
    :return:
    """

    g = root_MTG
    # We initialize different counters on the whole root system for characterizing the fungal infection:
    sum_of_max_exchange_surfaces = 0.
    sum_of_infected_exchange_surfaces = 0.
    total_root_length = 0.
    total_infected_root_length = 0.
    total_infected_elements_number = 0.

    initial_fungus_exchange_surface = fungus.root_exchange_surface

    # We go through the root MTG to look at each possibility of infection:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)

        np.random.seed(vid*step)

        # We increment the total surface and length of the root system:
        total_root_length += n.length
        try:
            sum_of_max_exchange_surfaces += n.cortical_parenchyma_surface + n.epidermis_surface_without_hairs
        except:
            sum_of_max_exchange_surfaces = sum_of_max_exchange_surfaces

        # We try to access the infection status of the current root element:
        try:
            # If the status already exist, we just access it (if it's None, then the code will jump to "except"):
            initial_fungal_infection_severity = n.fungal_infection_severity + 0.
        except:
            # If it doesn't exist, we consider that no infection has (yet) occured:
            n.fungal_infection_severity = 0.
            initial_fungal_infection_severity = 0.

        # CASE 1: If the root element has already been infected and has not reached the maximum level of infection:
        if n.fungal_infection_severity > 0. and n.fungal_infection_severity <= 1.:

            # Then we consider that the surface of exchange between this root element and the fungus will be increased
            # proportionally to the increment of surface of the whole fungus calculated previously:
            exchange_surface = n.fungus_exchange_surface \
                               * (1 + fungus.surface_increment_by_expansion / fungus.root_exchange_surface)

            # We consider the maximal exchange surface of the fungus with this root element:
            max_exchange_surface = n.cortical_parenchyma_surface + n.epidermis_surface_without_hairs
            # We make sure that the increase in exchange surface is limited to the maximal possible surface:
            if exchange_surface > max_exchange_surface:
                n.fungus_exchange_surface = max_exchange_surface
                n.fungal_infection_severity = 1.
            else:
                n.fungus_exchange_surface = exchange_surface
                n.fungal_infection_severity = exchange_surface / max_exchange_surface

        # CASE 2: If the root element has not been infected yet, we consider the possibility of a new infection:
        else:
            # If the distance from the root tip is high enough to allow infection and if the root element is not dead:
            if n.distance_from_tip > param.min_distance_for_fungus_infection and n.length > 0. \
                    and n.type != "Dead" and n.type != "Just_dead":

                # We start by considering that the probability of infection decreases with the age of the root:
                infection_probability = param.initial_fungus_infection_risk \
                                        * time_step_in_seconds / n.thermal_time_since_cells_formation
                print("    > For root element",vid,"infection probability is", infection_probability)

                # Then, we consider the possibility of infection from an already-infected adjacent root element.
                # 1) We try to access the segment preceding the current element:
                previous_index = g.Father(vid, EdgeType='<')
                # If there is no father element on this axis:
                if previous_index is None:
                    # Then we try to move to the mother root, if any:
                    previous_index = g.Father(vid, EdgeType='+')
                # If we have managed to find a parent element:
                if previous_index is not None:
                    parent = g.node(previous_index)
                    try:
                        # And if the parent element has already been infected by the fungus:
                        if parent.fungal_infection_severity > 0.:
                            # Then the probability of infection of the targeted root element is increased:
                            infection_probability += parent.fungal_infection_severity * time_step_in_seconds *1e-5
                    except:
                        parent.fungal_infection_severity = 0.

                # 2) We try to access the segment succeeding to the current element:
                try:
                    next_index = g.Sons(vid, EdgeType='<')[0]
                except:
                    next_index = None
                # If there is no son element on this axis:
                if next_index is None:
                    # Then we try to move to the daughter root, if any:
                    try:
                        next_index = g.Sons(vid, EdgeType='+')[0]
                    except:
                        next_index =  None
                # If we have managed to find a son element:
                if next_index is not None:
                    son = g.node(next_index)
                    try:
                        # And if the parent element has already been infected by the fungus:
                        if son.fungal_infection_severity > 0.:
                            # Then the probability of infection of the targeted root element is increased:
                            infection_probability += son.fungal_infection_severity * time_step_in_seconds *1e-5
                    except:
                        son.fungal_infection_severity = 0.

                # Eventually, we make a random test to see whether infection should occur at this time step or not:
                random_result = np.random.random_sample()
                # If the random test leads to positive result as compared with the probability number:
                if random_result < infection_probability:
                    # Then the targeted root element becomes infected!
                    # We consider the maximal exchange surface of the fungus with this root element:
                    max_exchange_surface = n.cortical_parenchyma_surface + n.epidermis_surface_without_hairs
                    # And we initialize the exchange surface between the root element and the fungus:
                    n.fungus_exchange_surface = max_exchange_surface * param.initial_infected_surface_fraction
                    n.fungal_infection_severity = n.fungus_exchange_surface / max_exchange_surface

                    # fungus.root_exchange_surface += n.fungus_exchange_surface

                else:
                    n.fungus_exchange_surface = 0.
                    n.fungal_infection_severity = 0.
            else:
                n.fungus_exchange_surface = 0.

        # We reset the initial fungal infection severity:
        initial_fungal_infection_severity = 0.

        # We eventually increment the total infected length, infected surface and number of infected elements:
        if n.fungal_infection_severity >0.:
            sum_of_infected_exchange_surfaces += n.fungus_exchange_surface
            total_infected_elements_number += 1
            total_infected_root_length += n.length

    # Finally, we print the infection results on the whole root system:
    print("> The total infected root surface is", "{:.2E}".format(sum_of_infected_exchange_surfaces), "m2 over",
          "{:.2E}".format(sum_of_max_exchange_surfaces), "m2 of possible exchange surface.")
    print("> The infected length is", "{:.2E}".format(total_infected_root_length), "m over",
          "{:.2E}".format(total_root_length), "m of root length.")
    print("> The number of infected elements is", "{:d}".format(int(total_infected_elements_number)),"over", "{:d}".format(g.nb_vertices()))

    if (fungus.root_exchange_surface + fungus.surface_increment_by_expansion) > sum_of_infected_exchange_surfaces:
        # print("")
        # print("ACH! GROSS PROBLEM !!!!")
        # print("")
        extra_surface = fungus.root_exchange_surface + fungus.surface_increment_by_expansion - sum_of_infected_exchange_surfaces
        fungus.soil_exchange_surface = extra_surface
        fungus.root_exchange_surface = sum_of_infected_exchange_surfaces
        fungus.overall_infection_severity = sum_of_infected_exchange_surfaces / sum_of_max_exchange_surfaces
    else:
        fungus.root_exchange_surface = sum_of_infected_exchange_surfaces
        fungus.overall_infection_severity = sum_of_infected_exchange_surfaces / sum_of_max_exchange_surfaces

    return g

def root_fungus_exchange_rate(root_MTG, fungus):
    """
    This function computes the new exchange rate of hexose between each root element and the fungus,
     and updates the corresponding hexose cost in the root element.
    :param root_MTG:
    :param fungus:
    :return:
    """

    g = root_MTG
    for vid in g.vertices_iter(scale=1):
        root_element = g.node(vid)

        # We first verify that the root element and the fungus are in fact in contact:
        if root_element.fungal_infection_severity > 0.:

            exchange_surface = root_element.fungus_exchange_surface
            C_hexose_root = root_element.C_hexose_root
            C_hexose_fungus = fungus.C_hexose_fungus

            # We calculate the rate of net hexose transfer from the root to the fungus, depending on their respective
            # hexose concentration:
            exchanged_hexose_rate = param.fungus_permeability * (C_hexose_root - C_hexose_fungus) * exchange_surface

            # We record the rate of exchange in both the root element and the fungus, assuming that all hexose taken
            # by the fungus is used for growth:
            # TODO: WATCH OUT - we should include also a cost for maintenance of the fungus and for mycodeposition.
            root_element.hexose_consumption_rate_by_fungus = exchanged_hexose_rate
            root_element.hexose_consumption_by_growth_rate += exchanged_hexose_rate
            fungus.hexose_input_rate_from_root += exchanged_hexose_rate
        else:
            exchanged_hexose_rate = 0.

    return

def fungus_expansion(fungus, time_step_in_seconds):
    """
    This function calculates the new surface of the fungus based on the available hexose,
    which will help to propagate the infection within the root system in the function "root_infction_by_fungus".
    :param fungus:
    :param time_step_in_seconds:
    :return:
    """

    # We calculate the current amount of hexose available for growth, including the new input of hexose:
    available_hexose_for_growth = (fungus.C_hexose_fungus - param.C_hexose_fungus_min) * fungus.struct_mass \
                                  + fungus.hexose_input_rate_from_root * time_step_in_seconds \
                                  - fungus.Deficit_hexose_fungus

    # The exchange surface between the root system and the fungus might have been already increased by the infection
    # of new root elements. If so, we include the corresponding C cost:
    if fungus.root_exchange_surface > fungus.initial_exchange_surface:
        mass_increment_by_infection = (fungus.root_exchange_surface - fungus.initial_exchange_surface) * \
                         param.fungus_surface_to_mass_ratio
        fungus.hexose_consumption_for_growth += mass_increment_by_infection * param.hyphal_struct_mass_C_content \
                                                / (param.fungus_yield_growth * 6.)
    else:
        mass_increment_by_infection = 0.

    # If there is not enough hexose available for growth compared to what has been used for the infection of new
    # root elements:
    if fungus.hexose_consumption_for_growth > available_hexose_for_growth:
        # Then we stop here, and the C balance will then adjust the Deficit in hexose later on:
        remaining_hexose_for_growth = 0.
        mass_increment_by_expansion = 0.
        surface_increment_by_expansion = 0.
        print(">>> WATCH OUT! The fungal infection is costing too much carbon compared to what can be taken!")
    else:
        remaining_hexose_for_growth = available_hexose_for_growth - fungus.hexose_consumption_for_growth
        mass_increment_by_expansion = remaining_hexose_for_growth * param.fungus_yield_growth * 6. \
                                      / param.hyphal_struct_mass_C_content
        surface_increment_by_expansion = mass_increment_by_expansion * param.fungus_surface_to_mass_ratio

    # Eventually, we update the total consumption of hexose for growth, the structural mass of the fungus
    # and the increment in surface:
    fungus.hexose_consumption_for_growth += remaining_hexose_for_growth
    fungus.struct_mass_increment = mass_increment_by_infection + mass_increment_by_expansion
    fungus.struct_mass += mass_increment_by_infection + mass_increment_by_expansion
    fungus.surface_increment_by_expansion = surface_increment_by_expansion

    return

def fungus_mass_balance(fungus, time_step_in_seconds):

    """
    This function performs a mass balance on the fungus and resets the concentration of hexose and/or the deficit.
    :param fungus:
    :param time_step_in_seconds:
    :return:
    """

    if fungus.struct_mass >0. and fungus.hexose_input_rate_from_root != 0.:
        print(">>> Before fungal growth, its concentration is", fungus.C_hexose_fungus)
        fungus.C_hexose_fungus += \
            (fungus.hexose_input_rate_from_root * time_step_in_seconds - fungus.hexose_consumption_for_growth - fungus.Deficit_hexose_fungus) \
            / fungus.struct_mass
        if fungus.C_hexose_fungus < 0.:
            fungus.Deficit_hexose_fungus = fungus.C_hexose_fungus
            fungus.C_hexose_fungus = 0.
        else:
            fungus.Deficit_hexose_fungus = 0.
        print(">>> After fungal growth, its concentration is", fungus.C_hexose_fungus)

    return

def mycorrhizal_interaction(root_MTG, fungus, step, time_step_in_seconds):
    """
    This function calls successively all other mycorrhizal functions to have the fungus infecting different root
    elements, take some hexose and extends.
    :param root_MTG:
    :param fungus:
    :param step:
    :param time_step_in_seconds:
    :return:
    """

    # We reinitialize growth-related variables and initial mass and surface of the fungus:
    reinitializing_fungus_growth(fungus)

    # We compute the new infection of the fungus over the whole root MTG - this modifies the surface
    # of exchange between the roots and the fungus, and the severity of infection of each element:
    root_infection_by_fungus(root_MTG, fungus=fungus, step=step,
                                         time_step_in_seconds=time_step_in_seconds)
    # NOTE: this function increases the surface, but not the mass of the fungus. It does not register the cost of growth.

    # We then calculate the new rate of carbon exchange between each root element and the fungus f:
    root_fungus_exchange_rate(root_MTG, fungus=fungus)

    # We compute the increment of mass and surface of the fungus and the cost in hexose:
    fungus_expansion(fungus=fungus, time_step_in_seconds=time_step_in_seconds)

    # Eventually, we perform a carbon balance on the fungus to adjust its concentration:
    fungus_mass_balance(fungus, time_step_in_seconds)

    return

########################################################################################################################
########################################################################################################################

# MAIN SIMULATION:
#-----------------

# # 1. We load an existing root MTG:
# ##################################
#
# inputs_dir_path = 'C://Users//frees//rhizodep//simulations//running_scenarios//inputs'
# outputs_dir_path = 'C://Users//frees//rhizodep//simulations//running_scenarios//outputs'
# ROOT_MTG_FILE = 'initial_root_MTG_0099.pckl'
# root_images_directory = 'C://Users//frees//rhizodep//simulations//running_scenarios//outputs//root_images'
#
# # We get the targeted root MTG:
# filename = os.path.join(inputs_dir_path, ROOT_MTG_FILE)
# if not os.path.exists(filename):
#     print("!!! ERROR: the file", ROOT_MTG_FILE,"could not be found in the folder", inputs_dir_path, "!!!")
#     print("The scenarios stops here!")
#     simulation_allowed=False
# else:
#     # We load the MTG file and name it "g":
#     f = open(filename, 'rb')
#     g = pickle.load(f)
#     f.close()
#     print("The MTG", ROOT_MTG_FILE,"has been loaded!")
#     # And by precaution we save the initial MTG in the outputs:
#     g_file_name = os.path.join(outputs_dir_path, 'initial_root_MTG.pckl')
#     with open(g_file_name, 'wb') as output:
#         pickle.dump(g, output, protocol=2)
#     print("The initial MTG file has been saved in the outputs.")
#
# # We perform the scenarios:
#
# f = initiate_mycorrhizal_fungus()
#
# final_step=100
# for step in range(1,final_step+1):
#     print("For time step", step, ":")
#
#     # print("    Considering infection evolution...")
#     root_infection_by_fungus(g, fungus=f, step=step, time_step_in_seconds=3600)
#
#     # print("    Considering exchange with the root system...")
#     root_fungus_exchange_rate(g,fungus=f)
#
#     # print("    Considering fungal growth...")
#     fungus_mass_growth(fungus=f, time_step_in_seconds = 3600)
#
#     print("")
#
#     # We print the plot:
#     sc = tools.plot_mtg(g, prop_cmap='C_hexose_root', lognorm=True, vmin=1e-6,
#                         vmax=1e-3, cmap='jet',
#                         root_hairs_display=True,
#                         width=1200,
#                         height=1200,
#                         x_center=0.,
#                         y_center=0.,
#                         z_center=-0.1,
#                         x_cam=0.4,
#                         y_cam=0,
#                         z_cam=0.2)
#
#     # We finally display the MTG on PlantGL and possibly record it:
#     pgl.Viewer.display(sc)
#     # If needed, we wait for a few seconds so that the graph is well positioned:
#     time.sleep(0.5)
#
#     image_name = os.path.join(root_images_directory, 'root%.5d.png')
#     pgl.Viewer.saveSnapshot(image_name % (step + 1))

