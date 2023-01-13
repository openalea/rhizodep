#  -*- coding: utf-8 -*-

"""
    rhizodep.model
    ~~~~~~~~~~~~~

    The module :mod:`rhizodep.model` defines the equations of root functioning.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

#TODO: Check the calculation of "Structural mass produced (g)"

import os
import numpy as np
import pandas as pd
from math import sqrt, pi, floor, exp
from decimal import Decimal

from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order

import rhizodep.parameters as param

# FUNCTIONS FOR CALCULATING PROPERTIES ON THE MTG
#################################################

# Defining the surface of each root element in contact with the soil:
# -------------------------------------------------------------------
def surfaces_and_volumes(g, element, radius, length):
    """
    The function "surfaces_and_volumes" computes different surfaces (m2) and volumes (m3) of a root element,
    based on the properties radius (m) and length (m).
    :param g: the investigated MTG
    :param element: the investigated node of the MTG
    :param radius: the radius of the root element (m)
    :param length: the length of the root element (m)
    :return: a dictionary containing the calculated surfaces and volumes of the given element
    """

    n = element
    vid = n.index()
    number_of_children = n.nb_children()

    # CALCULATIONS OF EXTERNAL SURFACE AND VOLUME:
    # If the root element corresponds to an apex or a segment without lateral roots:
    if number_of_children == 0 or number_of_children == 1:
        external_surface = 2 * pi * radius * length
        volume = pi * radius ** 2 * length
    # Otherwise there is one or more lateral roots branched on the root segment:
    else:
        # So we sum all the sections of the lateral roots branched on the root segment:
        sum_ramif_sections = 0
        for child_vid in g.Sons(vid, EdgeType='+'):
            son = g.node(child_vid)
            # We avoid to remove the section of the sphere of a nodule:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if son.type != "Root_nodule":
                sum_ramif_sections += pi * son.radius ** 2
        # And we subtract this sum of sections from the external area of the main cylinder:
        external_surface = 2 * pi * radius * length - sum_ramif_sections
        volume = pi * radius ** 2 * length

    # SPECIAL CASE FOR NODULE:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if n.type == "Root_nodule":
        # We consider the surface and volume of a sphere:
        external_surface = 4 * pi * radius ** 2
        volume = 4 / 3. * pi * radius ** 3

    # CALCULATIONS OF THE TOTAL EXCHANGE SURFACE OF PHLOEM VESSELS:
    phloem_surface = param.phloem_surfacic_fraction * external_surface

    # CALCULATIONS OF THE TOTAL EXCHANGE SURFACE BETWEEN ROOT SYMPLASM AND APOPLASM:
    # We consider that the total surface of exchange with the symplasm corresponds to the fraction of symplasm
    # inside the root cylinder, which is proportional to the external surfaceof that cylinder
    # (we therefore EXCLUDE root hairs surface):
    symplasm_surface = param.symplasm_surfacic_fraction * external_surface

    # CREATION OF A DICTIONARY THAT WILL BE USED TO RECORD THE OUTPUTS:
    dictionary = {"external_surface": external_surface,
                  "volume": volume,
                  "phloem_surface": phloem_surface,
                  "symplasm_surface": symplasm_surface
                  }

    return dictionary

# Defining the distance of a vertex from the tip:
# -----------------------------------------------
def dist_to_tip(g):
    """
    The function "dist_to_tip" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
    Note that the dist-to-tip of an apex is defined as its length (and not as 0).
    :param g: the investigated MTG
    :return: the MTG with an updated property 'dist_to_tip'
    """

    # We initialize an empty dictionary for to_tips:
    to_tips = {}
    # We use the property "length" of each vertex based on the function "length":
    length = g.property('length')

    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        # We define its direct successor as son:
        son_id = g.Successor(vid)
        son = g.node(son_id)

        # We record the initial dist_to_tip as the "former" one (to be used by other functions):
        n.former_dist_to_tip = n.dist_to_tip

        # We try to get the value of dist_to_tip for the neighbouring root element located closer to the apex of the root:
        try:
            # We calculate the new distance from the tip by adding its length to the distance of the successor:
            n.dist_to_tip = son.dist_to_tip + n.length
        except:
            # If there is no successor because the element is an apex or a root nodule:
            # Then we simply define the distance to the tip as the length of the element:
            n.dist_to_tip = n.length

    # We return a modified version of the MTG "g" with the updated property "dist_to_tip":
    return g

# Calculation of the length of a root element intercepted between two z coordinates:
# ----------------------------------------------------------------------------------
def sub_length_z(x1, y1, z1, x2, y2, z2, z_first_layer, z_second_layer):
    # We make sure that the z coordinates are ordered in the right way:
    min_z = min(z1, z2)
    max_z = max(z1, z2)
    z_start = min(z_first_layer, z_second_layer)
    z_end = max(z_first_layer, z_second_layer)

    # For each row, if at least a part of the root segment formed between point 1 and 2 is included between z_start and z_end:
    if min_z < z_end and max_z >= z_start:
        # The z value to start from is defined as:
        z_low = max(z_start, min_z)
        # The z value to stop at is defined as:
        z_high = min(z_end, max_z)

        # If Z2 is different from Z1:
        if max_z > min_z:
            # Then we calculate the (x,y) coordinates of both low and high points between which length will be computed:
            x_low = (x2 - x1) * (z_low - z1) / (z2 - z1) + x1
            y_low = (y2 - y1) * (z_low - z1) / (z2 - z1) + y1
            x_high = (x2 - x1) * (z_high - z1) / (z2 - z1) + x1
            y_high = (y2 - y1) * (z_high - z1) / (z2 - z1) + y1
            # Geometrical explanation:
            # *************************
            # # The root segment between Start-point (X1, Y1, Z1) and End-Point (X2, Y2, Z2)
            # draws a line in the 3D space which is characterized by the following parametric equation:
            # { x = (X2-X1)*t + X1, y = (Y2-Y1)*t + Y1, z = (Z2-Z1)*t + Z1}
            # To find the coordinates x and y of a new point of coordinate z on this line, we have to solve this system of equations,
            # knowing that: z = (Z2-Z1)*t + Z1, which gives t = (z - Z1)/(Z2-Z1), and therefore:
            # x = (X2-X1)*(z - Z1)/(Z2-Z1) + X1
            # y = (Y2-Y1)*(z - Z1)/(Z2-Z1) + Y1
        # Otherwise, the calculation is much easier, since the whole segment is included in the plan x-y:
        else:
            x_low = x1
            y_low = y1
            x_high = x2
            y_high = y2

        # In every case, the length between the low and high points is computed as:
        inter_length = ((x_high - x_low) ** 2 + (y_high - y_low) ** 2 + (z_high - z_low) ** 2) ** 0.5
    # Otherwise, the root element is not included between z_first_layer and z_second_layer, and intercepted length is 0:
    else:
        inter_length = 0

    # We return the computed length:
    return inter_length

# Integration of root variables within different z_intervals:
# -----------------------------------------------------------
def classifying_on_z(g, z_min=0., z_max=1., z_interval=0.1):
    # We initialize empty dictionaries:
    included_length = {}
    dictionary_length = {}
    dictionary_struct_mass = {}
    dictionary_root_necromass = {}
    dictionary_surface = {}
    dictionary_net_hexose_exudation = {}
    dictionary_total_rhizodeposition = {}
    dictionary_hexose_degradation = {}
    final_dictionary = {}

    # For each interval of z values to be considered:
    for z_start in np.arange(z_min, z_max, z_interval):

        # We create the names of the new properties of the MTG to be computed, based on the current z interval:
        name_length_z = "length_" + str(round(z_start, 3)) + "-" + str(round(z_start + z_interval, 3)) + "_m"
        name_struct_mass_z = "struct_mass_" + str(round(z_start, 3)) + "-" + str(round(z_start + z_interval, 3)) + "_m"
        name_root_necromass_z = "root_necromass_" + str(round(z_start, 3)) + "-" + str(
            round(z_start + z_interval, 3)) + "_m"
        name_surface_z = "surface_" + str(round(z_start, 3)) + "-" + str(round(z_start + z_interval, 3)) + "_m"
        name_net_hexose_exudation_z = "net_hexose_exudation_" + str(round(z_start, 3)) + "-" + str(
            round(z_start + z_interval, 3)) + "_m"
        name_total_rhizodeposition_z = "total_rhizodeposition_" + str(round(z_start, 3)) + "-" + str(
            round(z_start + z_interval, 3)) + "_m"
        name_hexose_degradation_z = "hexose_degradation_" + str(round(z_start, 3)) + "-" + str(
            round(z_start + z_interval, 3)) + "_m"

        # We (re)initialize total values:
        total_included_length = 0
        total_included_struct_mass = 0
        total_included_root_necromass = 0
        total_included_surface = 0
        total_included_net_hexose_exudation = 0
        total_included_rhizodeposition = 0
        total_included_hexose_degradation = 0

        # We cover all the vertices in the MTG:
        for vid in g.vertices_iter(scale=1):
            # n represents the vertex:
            n = g.node(vid)

            # We make sure that the vertex has a positive length:
            if n.length > 0.:
                # We calculate the fraction of the length of this vertex that is included in the current range of z value:
                fraction_length = sub_length_z(x1=n.x1, y1=n.y1, z1=-n.z1, x2=n.x2, y2=n.y2, z2=-n.z2,
                                               z_first_layer=z_start,
                                               z_second_layer=z_start + z_interval) / n.length
                included_length[vid] = fraction_length * n.length
            else:
                # Otherwise, the fraction length and the length included in the range are set to 0:
                fraction_length = 0.
                included_length[vid] = 0.

            # We summed different variables based on the fraction of the length included in the z interval:
            total_included_length += n.length * fraction_length
            total_included_struct_mass += (n.struct_mass + n.root_hairs_struct_mass) * fraction_length
            if n.type == "Dead" or n.type == "Just_dead":
                total_included_root_necromass += (n.struct_mass + n.root_hairs_struct_mass) * fraction_length
            total_included_surface += n.external_surface * fraction_length
            total_included_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake) * fraction_length
            total_included_rhizodeposition += (n.hexose_exudation - n.hexose_uptake
                                               + n.mucilage_secretion + n.cells_release) * fraction_length
            total_included_hexose_degradation += n.hexose_degradation * fraction_length

        # We record the summed values for this interval of z in several dictionaries:
        dictionary_length[name_length_z] = total_included_length
        dictionary_struct_mass[name_struct_mass_z] = total_included_struct_mass
        dictionary_root_necromass[name_root_necromass_z] = total_included_root_necromass
        dictionary_surface[name_surface_z] = total_included_surface
        dictionary_net_hexose_exudation[name_net_hexose_exudation_z] = total_included_net_hexose_exudation
        dictionary_total_rhizodeposition[name_total_rhizodeposition_z] = total_included_rhizodeposition
        dictionary_hexose_degradation[name_hexose_degradation_z] = total_included_hexose_degradation

        # We also create a new property of the MTG that corresponds to the fraction of length of each node in the z interval:
        g.properties()[name_length_z] = included_length

    # Finally, we merge all dictionaries into a single one that will be returned by the function:
    final_dictionary = {}
    for d in [dictionary_length, dictionary_struct_mass, dictionary_root_necromass, dictionary_surface,
              dictionary_net_hexose_exudation,
              dictionary_total_rhizodeposition,
              dictionary_hexose_degradation]:
        final_dictionary.update(d)

    return final_dictionary

# Integration of root variables within different z_intervals:
# -----------------------------------------------------------
def recording_MTG_properties(g, file_name='g_properties.csv'):
    """
    This function records the properties of each node of the MTG "g" in a csv file.
    """

    # We define and reorder the list of all properties of the MTG:
    list_of_properties = list(g.properties().keys())
    list_of_properties.sort()

    # We create an empty list of node indices:
    node_index = []
    # We create an empty list that will contain the properties of each node:
    g_properties = []

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # Initializing an empty list of properties for the current node:
        node_properties = []
        # Adding the index at the beginning of the list:
        node_properties.append(vid)
        # n represents the vertex:
        n = g.node(vid)
        # For each possible property:
        for property in list_of_properties:
            # We add the value of this property to the list:
            node_properties.append(getattr(n, property, "NA"))
        # Finally, we add the new node's properties list as a new item in g_properties:
        g_properties.append(node_properties)
    # We create a list containing the headers of the dataframe:
    column_names = ['node_index']
    column_names.extend(list_of_properties)
    # We create the final dataframe:
    data_frame = pd.DataFrame(g_properties, columns=column_names)
    # We record the dataframe as a csv file:
    data_frame.to_csv(file_name, na_rep='NA', index=False, header=True)

    return

# Modification of a process according to soil temperature:
# --------------------------------------------------------
def temperature_modification(temperature_in_Celsius, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
    """
    This function calculates how the value of a process should be modified according to soil temperature (in degrees Celsius).
    Parameters correspond the the value of the process at reference temperature T_ref (process_at_T_ref),
    to two empirical coefficients A and B, and to a coefficient C used to switch between different formalisms.
    If C=0 and B=1, then the relationship corresponds to a classical linear increase with temperature (thermal time).
    If C=1, A=0 and B>1, then the relationship corresponds to a classical exponential increase with temperature (Q10).
    If C=1, A<0 and B>0, then the relationship corresponds to bell-shaped curve, close to the one from Parent et al. (2010).
    """

    # We initialize the value of the temperature-modified process:
    modified_process = 0.

    # We avoid unwanted cases:
    if C != 0 and C != 1:
        print("The modification of the process at T =", temperature_in_Celsius, "only works for C=0 or C=1!")
        print("The modified process has been set to 0.")
        modified_process = 0.
        return 0.
    elif C == 1:
        if (A * (temperature_in_Celsius - T_ref) + B) < 0.:
            print("The modification of the process at T =", temperature_in_Celsius,
                  "is unstable with this set of parameters!")
            print("The modified process has been set to 0.")
            modified_process = 0.
            return modified_process

    # We compute a temperature-modified process, correspond to a Q10-modified relationship,
    # based on the work of Tjoelker et al. (2001):
    modified_process = process_at_T_ref * (A * (temperature_in_Celsius - T_ref) + B) ** (1 - C) \
                       * (A * (temperature_in_Celsius - T_ref) + B) ** (C * (temperature_in_Celsius - T_ref) / 10.)

    if modified_process < 0.:
        modified_process = 0.

    return modified_process

# Adding a new root element with pre-defined properties:
# ------------------------------------------------------
def ADDING_A_CHILD(mother_element, edge_type='+', label='Apex', type='Normal_root_before_emergence',
                   root_order=1,
                   angle_down=45., angle_roll=0., length=0., radius=0.,
                   identical_properties=True, nil_properties=False):
    """
    This function creates a new child element on the mother element, based on the function add_child.
    When called, this allows to automatically define standard properties without defining them in the main code.
    :param edge_type:
    :return:
    """

    # If nil_properties = True, then we set most of the properties of the new element to 0:
    if nil_properties:

        new_child = mother_element.add_child(edge_type=edge_type,
                                             # Characteristics:
                                             # -----------------
                                             label=label,
                                             type=type,
                                             root_order=root_order,
                                             # Authorizations and C requirements:
                                             # -----------------------------------
                                             lateral_root_emergence_possibility='Impossible',
                                             emergence_cost=0.,
                                             # Geometry and topology:
                                             # -----------------------
                                             angle_down=angle_down,
                                             angle_roll=angle_roll,
                                             # The length of the primordium is set to 0:
                                             length=length,
                                             radius=radius,
                                             original_radius=radius,
                                             potential_length=0.,
                                             theoretical_radius=radius,
                                             potential_radius=radius,
                                             initial_length=0.,
                                             initial_radius=radius,
                                             external_surface=0.,
                                             volume=0.,
                                             dist_to_ramif=0.,
                                             dist_to_tip=0.,
                                             former_dist_to_tip=0.,
                                             actual_elongation=0.,
                                             # Quantities and concentrations:
                                             # -------------------------------
                                             struct_mass=0.,
                                             initial_struct_mass=0.,
                                             C_hexose_root=0.,
                                             C_hexose_reserve=0.,
                                             C_hexose_soil=0.,
                                             C_mucilage_soil=0.,
                                             C_cells_soil=0.,
                                             C_sucrose_root=0.,
                                             Deficit_hexose_root=0.,
                                             Deficit_hexose_soil=0.,
                                             Deficit_mucilage_soil=0.,
                                             Deficit_cells_soil=0.,
                                             Deficit_sucrose_root=0.,
                                             # Root hairs:
                                             #------------
                                             root_hair_radius = param.root_hair_radius,
                                             root_hair_length = 0.,
                                             actual_length_with_hairs=0.,
                                             living_root_hairs_number = 0.,
                                             dead_root_hairs_number = 0.,
                                             total_root_hairs_number = 0.,
                                             actual_time_since_root_hairs_emergence_started = 0.,
                                             thermal_time_since_root_hairs_emergence_started = 0.,
                                             actual_time_since_root_hairs_emergence_stopped = 0.,
                                             thermal_time_since_root_hairs_emergence_stopped = 0.,
                                             all_root_hairs_formed = False,
                                             root_hairs_lifespan = param.root_hairs_lifespan,
                                             root_hairs_external_surface = 0.,
                                             living_root_hairs_external_surface = 0.,
                                             root_hairs_volume = 0.,
                                             root_hairs_struct_mass = 0.,
                                             root_hairs_struct_mass_produced = 0.,
                                             initial_living_root_hairs_struct_mass = 0.,
                                             living_root_hairs_struct_mass = 0.,
                                             # Fluxes:
                                             # --------
                                             resp_maintenance=0.,
                                             resp_growth=0.,
                                             struct_mass_produced=0.,
                                             hexose_growth_demand=0.,
                                             hexose_consumption_by_growth=0.,
                                             hexose_possibly_required_for_elongation=0.,
                                             hexose_production_from_phloem=0.,
                                             sucrose_loading_in_phloem=0.,
                                             hexose_mobilization_from_reserve=0.,
                                             hexose_immobilization_as_reserve=0.,
                                             hexose_exudation=0.,
                                             hexose_uptake=0.,
                                             mucilage_secretion=0.,
                                             cells_release=0.,
                                             total_rhizodeposition=0.,
                                             hexose_degradation=0.,
                                             mucilage_degradation=0.,
                                             cells_degradation=0.,
                                             specific_net_exudation=0.,
                                             # Time indications:
                                             # ------------------
                                             growth_duration=param.GDs * (2 * radius) ** 2,
                                             life_duration=param.LDs * 2. * radius * param.root_tissue_density,
                                             actual_time_since_primordium_formation=0.,
                                             actual_time_since_emergence=0.,
                                             actual_potential_time_since_emergence=0.,
                                             actual_time_since_growth_stopped=0.,
                                             actual_time_since_death=0.,
                                             thermal_time_since_primordium_formation=0.,
                                             thermal_time_since_emergence=0.,
                                             thermal_potential_time_since_emergence=0.,
                                             thermal_time_since_growth_stopped=0.,
                                             thermal_time_since_death=0.
                                             )

    # Otherwise, if identical_properties=True, then we copy most of the properties of the mother element in the new element:
    elif identical_properties:

        new_child = mother_element.add_child(edge_type=edge_type,
                                             # Characteristics:
                                             # -----------------
                                             label=label,
                                             type=type,
                                             root_order=root_order,
                                             # Authorizations and C requirements:
                                             # -----------------------------------
                                             lateral_root_emergence_possibility='Impossible',
                                             emergence_cost=0.,
                                             # Geometry and topology:
                                             # -----------------------
                                             angle_down=angle_down,
                                             angle_roll=angle_roll,
                                             # The length of the primordium is set to 0:
                                             length=length,
                                             radius=mother_element.radius,
                                             original_radius=mother_element.radius,
                                             potential_length=length,
                                             theoretical_radius=mother_element.theoretical_radius,
                                             potential_radius=mother_element.potential_radius,
                                             initial_length=length,
                                             initial_radius=mother_element.radius,
                                             external_surface=0.,
                                             volume=0.,
                                             dist_to_ramif=mother_element.dist_to_ramif,
                                             dist_to_tip=mother_element.dist_to_tip,
                                             former_dist_to_tip=mother_element.former_dist_to_tip,
                                             actual_elongation=mother_element.actual_elongation,
                                             # Quantities and concentrations:
                                             # -------------------------------
                                             struct_mass=mother_element.struct_mass,
                                             initial_struct_mass=mother_element.initial_struct_mass,
                                             C_hexose_root=mother_element.C_hexose_root,
                                             C_hexose_reserve=mother_element.C_hexose_reserve,
                                             C_hexose_soil=mother_element.C_hexose_soil,
                                             C_mucilage_soil=mother_element.C_mucilage_soil,
                                             C_cells_soil=mother_element.C_cells_soil,
                                             C_sucrose_root=mother_element.C_sucrose_root,
                                             Deficit_hexose_root=mother_element.Deficit_hexose_root,
                                             Deficit_hexose_soil=mother_element.Deficit_hexose_soil,
                                             Deficit_mucilage_soil=mother_element.Deficit_mucilage_soil,
                                             Deficit_cells_soil=mother_element.Deficit_cells_soil,
                                             Deficit_sucrose_root=mother_element.Deficit_sucrose_root,
                                             # Root hairs:
                                             # ------------
                                             root_hair_radius=mother_element.root_hair_radius,
                                             root_hair_length=mother_element.root_hair_length,
                                             actual_length_with_hairs=mother_element.actual_length_with_hairs,
                                             living_root_hairs_number=mother_element.living_root_hairs_number,
                                             dead_root_hairs_number=mother_element.dead_root_hairs_number,
                                             total_root_hairs_number=mother_element.total_root_hairs_number,
                                             actual_time_since_root_hairs_emergence_started=mother_element.actual_time_since_root_hairs_emergence_started,
                                             thermal_time_since_root_hairs_emergence_started=mother_element.thermal_time_since_root_hairs_emergence_started,
                                             actual_time_since_root_hairs_emergence_stopped=mother_element.actual_time_since_root_hairs_emergence_stopped,
                                             thermal_time_since_root_hairs_emergence_stopped=mother_element.thermal_time_since_root_hairs_emergence_stopped,
                                             all_root_hairs_formed=mother_element.all_root_hairs_formed,
                                             root_hairs_lifespan=mother_element.root_hairs_lifespan,
                                             root_hairs_external_surface=mother_element.root_hairs_external_surface,
                                             living_root_hairs_external_surface=mother_element.living_root_hairs_external_surface,
                                             root_hairs_volume=mother_element.root_hairs_volume,
                                             root_hairs_struct_mass=mother_element.root_hairs_struct_mass,
                                             root_hairs_struct_mass_produced=mother_element.root_hairs_struct_mass_produced,
                                             living_root_hairs_struct_mass=mother_element.living_root_hairs_struct_mass,
                                             initial_living_root_hairs_struct_mass=mother_element.initial_living_root_hairs_struct_mass,
                                             # Fluxes:
                                             # -------
                                             resp_maintenance=mother_element.resp_maintenance,
                                             resp_growth=mother_element.resp_growth,
                                             struct_mass_produced=mother_element.struct_mass_produced,
                                             hexose_growth_demand=mother_element.hexose_growth_demand,
                                             hexose_possibly_required_for_elongation=mother_element.hexose_possibly_required_for_elongation,
                                             hexose_production_from_phloem=mother_element.hexose_production_from_phloem,
                                             sucrose_loading_in_phloem=mother_element.sucrose_loading_in_phloem,
                                             hexose_mobilization_from_reserve=mother_element.hexose_mobilization_from_reserve,
                                             hexose_immobilization_as_reserve=mother_element.hexose_immobilization_as_reserve,
                                             hexose_exudation=mother_element.hexose_exudation,
                                             hexose_uptake=mother_element.hexose_uptake,
                                             mucilage_secretion=mother_element.mucilage_secretion,
                                             cells_release=mother_element.cells_release,
                                             total_rhizodeposition=mother_element.total_rhizodeposition,
                                             hexose_degradation=mother_element.hexose_degradation,
                                             mucilage_degradation=mother_element.mucilage_degradation,
                                             cells_degradation=mother_element.cells_degradation,
                                             hexose_consumption_by_growth=mother_element.hexose_consumption_by_growth,
                                             specific_net_exudation=mother_element.specific_net_exudation,
                                             # Time indications:
                                             # ------------------
                                             growth_duration=mother_element.growth_duration,
                                             life_duration=mother_element.life_duration,
                                             actual_time_since_primordium_formation=mother_element.actual_time_since_primordium_formation,
                                             actual_time_since_emergence=mother_element.actual_time_since_emergence,
                                             actual_time_since_growth_stopped=mother_element.actual_time_since_growth_stopped,
                                             actual_time_since_death=mother_element.actual_time_since_death,

                                             thermal_time_since_primordium_formation=mother_element.thermal_time_since_primordium_formation,
                                             thermal_time_since_emergence=mother_element.thermal_time_since_emergence,
                                             thermal_potential_time_since_emergence=mother_element.thermal_potential_time_since_emergence,
                                             thermal_time_since_growth_stopped=mother_element.thermal_time_since_growth_stopped,
                                             thermal_time_since_death=mother_element.thermal_time_since_death
                                             )

    return new_child

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "POTENTIAL GROWTH"
########################################################################################################################
########################################################################################################################

# Function for calculating root elongation:
#------------------------------------------

def elongated_length(initial_length=0., radius=0., C_hexose_root=1,
                     elongation_time_in_seconds=0.,
                     ArchiSimple=True, printing_warnings=False, soil_temperature_in_Celsius=20):
    """

    :param initial_length:
    :param radius:
    :param C_hexose_root:
    :param elongation_time_in_seconds:
    :param ArchiSimple:
    :param printing_warnings:
    :return:
    """

    # If we keep the classical ArchiSimple rule:
    if ArchiSimple:
        # Then the elongation is calculated following the rules of Pages et al. (2014):
        elongation = param.EL * 2. * radius * elongation_time_in_seconds
    else:
        # Otherwise, we additionally consider a limitation of the elongation according to the local concentration of hexose,
        # based on a Michaelis-Menten formalism:
        if C_hexose_root > 0.:
            elongation = param.EL * 2. * radius * C_hexose_root / (
                    param.Km_elongation + C_hexose_root) * elongation_time_in_seconds
        else:
            elongation = 0.

    # We calculate the new potential length corresponding to this elongation:
    new_length = initial_length + elongation
    if new_length < initial_length:
        print("!!! ERROR: There is a problem of elongation, with the initial length", initial_length,
              " and the radius", radius, "and the elongation time", elongation_time_in_seconds)
    return new_length

# Formation of a root primordium at the apex of the mother root:
#---------------------------------------------------------------

def primordium_formation(g, apex, elongation_rate=0., time_step_in_seconds=1. * 60. * 60. * 24.,
                         soil_temperature_in_Celsius=20, random=False,
                         root_order_limitation=False, root_order_treshold=2):
    """

    :param apex:
    :param elongation_rate:
    :param time_step_in_seconds:
    :param random:
    :return:
    """

    # NOTE: This function has to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
    # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)

    # OPERATING PRIMORDIUM FORMATION:
    # --------------------------------
    # We initialize the new_apex that will be returned by the function:
    new_apex = []

    # VERIFICATION: We make sure that no lateral root has already been form on the present apex.
    # We calculate the number of children of the apex (it should be 0!):
    n_children = len(apex.children())
    # If there is at least one children, it means that there is already one lateral primordium or root on the apex:
    if n_children >= 1:
        # Then we don't add any primordium and simply return the unaltered apex:
        new_apex.append(apex)
        return new_apex

    # If the order of the current apex is too high, we forbid the formation of a new primordium of higher order:
    if root_order_limitation and apex.root_order + 1 > root_order_treshold:
        # Then we don't add any primordium and simply return the unaltered apex:
        new_apex.append(apex)
        return new_apex

    # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
    # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
    # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014).
    # We also set the root angles depending on random:
    if random:
        # The seed used to generate random values is defined according to a parameter random_choice and the index of the apex:
        np.random.seed(param.random_choice * apex.index())
        potential_radius = np.random.normal((apex.radius - param.Dmin / 2.) * param.RMD + param.Dmin / 2.,
                                            ((
                                                         apex.radius - param.Dmin / 2.) * param.RMD + param.Dmin / 2.) * param.CVDD)
        apex_angle_roll = abs(np.random.normal(120, 10))
        if apex.root_order == 1:
            primordium_angle_down = abs(np.random.normal(45, 10))
        else:
            primordium_angle_down = abs(np.random.normal(70, 10))
        primordium_angle_roll = abs(np.random.normal(5, 5))
    else:
        potential_radius = (apex.radius - param.Dmin / 2) * param.RMD + param.Dmin / 2.
        apex_angle_roll = 120
        if apex.root_order == 1:
            primordium_angle_down = 45
        else:
            primordium_angle_down = 70
        primordium_angle_roll = 5

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # We add a condition, i.e. potential_radius should not be larger than the one from the mother root:
    if potential_radius > apex.radius:
        potential_radius = apex.radius

    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:
    if apex.dist_to_ramif > param.IPD and potential_radius >= param.Dmin and potential_radius <= apex.radius:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # The distance that the tip of the apex has covered since the actual primordium formation is calculated:
        elongation_since_last_ramif = apex.dist_to_ramif - param.IPD

        # A specific rolling angle is attributed to the parent apex:
        apex.angle_roll = apex_angle_roll

        # We verify that the apex has actually elongated:
        if apex.actual_elongation > 0:
            # Then the actual time since the primordium must have been formed is precisely calculated
            # according to the actual growth of the parent apex since primordium formation,
            # taking into account the actual growth rate of the parent defined as
            # apex.actual_elongation / time_step_in_seconds
            actual_time_since_formation = elongation_since_last_ramif / elongation_rate
        else:
            actual_time_since_formation = 0.

        # And we add the primordium of a possible new lateral root:
        ramif = ADDING_A_CHILD(mother_element=apex, edge_type='+', label='Apex', type='Normal_root_before_emergence',
                               root_order=apex.root_order + 1,
                               angle_down=primordium_angle_down,
                               angle_roll=primordium_angle_roll,
                               length=0.,
                               radius=potential_radius,
                               identical_properties=False,
                               nil_properties=True)
        # We specify the exact time since formation:
        ramif.actual_time_since_primordium_formation = actual_time_since_formation
        ramif.thermal_time_since_primordium_formation = actual_time_since_formation * temperature_time_adjustment
        # And the new distance between the parent apex and the last ramification is redefined,
        # by taking into account the actual elongation of apex since the child formation:
        apex.dist_to_ramif = elongation_since_last_ramif
        # # We also put in memory the index of the child:
        # apex.lateral_primordium_index = ramif.index()
        # We add the apex and its ramif in the list of apices returned by the function:
        new_apex.append(apex)
        new_apex.append(ramif)

    return new_apex

# Function for calculating the amount of C to be used in neighbouring elements for sustaining root elongation:
#-------------------------------------------------------------------------------------------------------------
def calculating_C_supply_for_elongation(g, element):
    """

    :param element:
    :return:
    """

    n = element

    # We initialize each amount of hexose available for growth:
    n.hexose_possibly_required_for_elongation = 0.
    n.struct_mass_contributing_to_elongation = 0.

    # We initialize empty lists:
    list_of_elongation_supporting_elements = []
    list_of_elongation_supporting_elements_hexose = []
    list_of_elongation_supporting_elements_mass = []

    # We then calculate the length of an apical zone of a fixed length which can provide the amount of hexose required for growth:
    growing_zone_length = param.growing_zone_factor * n.radius
    # We calculate the corresponding volume to which this length should correspond based on the diameter of this apex:
    supplying_volume = growing_zone_length * n.radius ** 2 * pi

    # We start counting the hexose at the apex:
    index = n.index()
    current_element = n

    # We initialize a temporary variable that will be used as a counter:
    remaining_volume = supplying_volume

    # As long the remaining volume is not zero:
    while remaining_volume > 0:

        # If the volume of the current element is lower than the remaining volume:
        if remaining_volume > current_element.volume:
            # We make sure to include in the list of supplying elements only elements with a positive length
            # (e.g. NOT the elements of length 0 that support seminal or adventitious roots):
            if current_element.length > 0.:
                # We add to the amount of hexose available all the hexose in the current element
                # (EXCLUDING sugars in the living root hairs):!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                hexose_contribution = current_element.C_hexose_root * current_element.struct_mass
                n.hexose_possibly_required_for_elongation += hexose_contribution
                n.struct_mass_contributing_to_elongation += current_element.struct_mass
                # We record the index of the contributing element:
                list_of_elongation_supporting_elements.append(index)
                # We record the amount of hexose that the current element can provide:
                list_of_elongation_supporting_elements_hexose.append(hexose_contribution)
                # We record the structural mass from which the current element contributes:
                list_of_elongation_supporting_elements_mass.append(current_element.struct_mass)
                # We subtract the volume of the current element to the remaining volume:
                remaining_volume = remaining_volume - current_element.volume

            # And we try to move the index to the segment preceding the current element:
            index_attempt = g.Father(index, EdgeType='<')
            # If there is no father element on this axis:
            if index_attempt is None:
                # Then we try to move to the mother root, if any:
                index_attempt = g.Father(index, EdgeType='+')
                # If there is no such root:
                if index_attempt is None:
                    # print "!!! ERROR: For element", n.index(),"of type", n.type, "there is no possibility to move higher than the element", index
                    # Then we exit the loop here:
                    break
            # We set the new index:
            index = index_attempt
            # We define the new element to consider according to the new index:
            current_element = g.node(index)
        # Otherwise, this is the last preceding element to consider:
        else:
            # We finally add to the amount of hexose available for elongation a part of the hexose of the current element:
            hexose_contribution = current_element.C_hexose_root * current_element.struct_mass \
                                  * remaining_volume / current_element.volume
            n.hexose_possibly_required_for_elongation += hexose_contribution
            n.struct_mass_contributing_to_elongation += current_element.struct_mass \
                                                        * remaining_volume / current_element.volume
            # We record the index of the contributing element:
            list_of_elongation_supporting_elements.append(index)
            # We record the amount of hexose that the current element can provide:
            list_of_elongation_supporting_elements_hexose.append(hexose_contribution)
            # We record the structural mass from which the current element contributes:
            list_of_elongation_supporting_elements_mass.append(
                current_element.struct_mass * remaining_volume / current_element.volume)
            # And the remaining volume to consider is set to 0:
            remaining_volume = 0.
            # And we exit the loop here:
            break

    # We record the average concentration in hexose of the whole zone of hexose supply contributing to elongation:
    if n.struct_mass_contributing_to_elongation > 0.:
        n.growing_zone_C_hexose_root = n.hexose_possibly_required_for_elongation / n.struct_mass_contributing_to_elongation
    else:
        print("!!! ERROR: the mass contributing to elongation in element", n.index(), "of type", n.type, "is",
              n.struct_mass_contributing_to_elongation,
              "g, and its structural mass is", n.struct_mass, "g!")
        n.growing_zone_C_hexose_root = 0.

    n.list_of_elongation_supporting_elements = list_of_elongation_supporting_elements
    n.list_of_elongation_supporting_elements_hexose = list_of_elongation_supporting_elements_hexose
    n.list_of_elongation_supporting_elements_mass = list_of_elongation_supporting_elements_mass

    return list_of_elongation_supporting_elements, \
           list_of_elongation_supporting_elements_hexose, \
           list_of_elongation_supporting_elements_mass

# Function calculating the potential development of an apex:
#-----------------------------------------------------------
def potential_apex_development(g, apex, time_step_in_seconds=1. * 60. * 60. * 24., ArchiSimple=False,
                               soil_temperature_in_Celsius=20, printing_warnings=False):
    """

    :param apex:
    :param time_step_in_seconds:
    :param ArchiSimple:
    :param printing_warnings:
    :return:
    """

    # We initialize an empty list in which the modified apex will be added:
    new_apex = []
    # We record the current radius and length prior to growth as the initial radius and length:
    apex.initial_radius = apex.radius
    apex.initial_length = apex.length
    # We initialize the properties "potential_radius" and "potential_length" returned by the function:
    apex.potential_radius = apex.radius
    apex.potential_length = apex.length

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
    # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)

    # CASE 1: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING SEMINAL OR ADVENTITIOUS ROOT
    # -----------------------------------------------------------------------------------------------------
    # If the seminal root has not emerged yet:
    if apex.type == "Seminal_root_before_emergence" or apex.type == "Adventitious_root_before_emergence":
        # If the time elapsed since the last emergence of seminal root is higher than the prescribed interval time:
        if (apex.thermal_time_since_primordium_formation \
            + time_step_in_seconds * temperature_time_adjustment) >= apex.emergence_delay_in_thermal_time:
            # The potential time elapsed since seminal root's possible emergence is calculated:
            apex.thermal_potential_time_since_emergence = \
                apex.thermal_time_since_primordium_formation + time_step_in_seconds * temperature_time_adjustment \
                - apex.emergence_delay_in_thermal_time
            # If the apex could have emerged sooner:
            if apex.thermal_potential_time_since_emergence > time_step_in_seconds * temperature_time_adjustment:
                # The time since emergence is reduced to the time elapsed during this time step:
                apex.thermal_potential_time_since_emergence = time_step_in_seconds * temperature_time_adjustment
            # We record the different elements that can contribute to the C supply necessary for growth,
            # and we calculate a mean concentration of hexose in this supplying zone:
            calculating_C_supply_for_elongation(g, apex)
            # The corresponding potential elongation of the apex is calculated:
            apex.potential_length = elongated_length(initial_length=apex.initial_length, radius=apex.initial_radius,
                                                     C_hexose_root=apex.growing_zone_C_hexose_root,
                                                     elongation_time_in_seconds=apex.thermal_potential_time_since_emergence,
                                                     ArchiSimple=ArchiSimple,
                                                     soil_temperature_in_Celsius=soil_temperature_in_Celsius)
            # Last, if ArchiSimple has been chosen as the growth model:
            if ArchiSimple:
                # Then we automatically allow the root to emerge, without consideration of C limitation:
                apex.type = "Normal_root_after_emergence"
        # In any case, the time since primordium formation is incremented, as usual:
        apex.actual_time_since_primordium_formation += time_step_in_seconds
        apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
        # And the new element returned by the function corresponds to the potentially emerging apex:
        new_apex.append(apex)
        # And the function returns this new apex and stops here:
        return new_apex

    # CASE 2: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING NORMAL LATERAL ROOT
    # ---------------------------------------------------------------------------------------------
    if apex.type == "Normal_root_before_emergence":
        # If the time since primordium formation is higher than the delay of emergence:
        if apex.thermal_time_since_primordium_formation + time_step_in_seconds * temperature_time_adjustment > param.emergence_delay:
            # The time since primordium formation is incremented:
            apex.actual_time_since_primordium_formation += time_step_in_seconds
            apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
            # The potential time elapsed at the end of this time step since the emergence is calculated:
            apex.thermal_potential_time_since_emergence = apex.thermal_time_since_primordium_formation - param.emergence_delay
            # If the apex could have emerged sooner:
            if apex.thermal_potential_time_since_emergence > time_step_in_seconds * temperature_time_adjustment:
                # The time since emergence is equal to the time elapsed during this time step (since it must have emerged at this time step):
                apex.thermal_potential_time_since_emergence = time_step_in_seconds * temperature_time_adjustment
            # We record the different element that can contribute to the C supply necessary for growth,
            # and we calculate a mean concentration of hexose in this supplying zone:
            calculating_C_supply_for_elongation(g, apex)
            # The corresponding elongation of the apex is calculated:
            apex.potential_length = elongated_length(initial_length=apex.initial_length, radius=apex.initial_radius,
                                                     C_hexose_root=apex.growing_zone_C_hexose_root,
                                                     elongation_time_in_seconds=apex.thermal_potential_time_since_emergence,
                                                     ArchiSimple=ArchiSimple,
                                                     soil_temperature_in_Celsius=soil_temperature_in_Celsius)

            # If ArchiSimple has been chosen as the growth model:
            if ArchiSimple:
                apex.type = "Normal_root_after_emergence"
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # Otherwise, we control the actual emergence of this primordium through the management of the parent:
            else:
                # We select the parent on which the primordium has been formed:
                vid = apex.index()
                index_parent = g.Father(vid, EdgeType='+')
                parent = g.node(index_parent)
                # # The possibility of emergence of a lateral root from the parent is recorded inside the parent:
                # parent.lateral_root_emergence_possibility = "Possible"
                # parent.lateral_primordium_index = apex.index()
                # And the new element returned by the function corresponds to the potentially emerging apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
        # Otherwise, the time since primordium formation is simply incremented:
        else:
            apex.actual_time_since_primordium_formation += time_step_in_seconds
            apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
            # And the new element returned by the function corresponds to the modified apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

    # CASE 3: THE APEX BELONGS TO AN AXIS THAT HAS ALREADY EMERGED:
    # --------------------------------------------------------------
    # IF THE APEX CAN CONTINUE GROWING:
    if apex.thermal_time_since_emergence + time_step_in_seconds * temperature_time_adjustment < apex.growth_duration:
        # The times are incremented:
        apex.actual_time_since_primordium_formation += time_step_in_seconds
        apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
        apex.actual_time_since_emergence += time_step_in_seconds
        apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
        # We record the different element that can contribute to the C supply necessary for growth,
        # and we calculate a mean concentration of hexose in this supplying zone:
        calculating_C_supply_for_elongation(g, apex)
        # The corresponding potential elongation of the apex is calculated:
        apex.potential_length = elongated_length(initial_length=apex.length, radius=apex.radius,
                                                 C_hexose_root=apex.growing_zone_C_hexose_root,
                                                 elongation_time_in_seconds=time_step_in_seconds * temperature_time_adjustment,
                                                 ArchiSimple=ArchiSimple,
                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius)
        # And the new element returned by the function corresponds to the modified apex:
        new_apex.append(apex)
        # And the function returns this new apex and stops here:
        return new_apex

    # OTHERWISE, THE APEX HAD TO STOP:
    else:
        # IF THE APEX HAS NOT REACHED ITS LIFE DURATION:
        if apex.thermal_time_since_growth_stopped + time_step_in_seconds * temperature_time_adjustment < apex.life_duration:
            # IF THE APEX HAS ALREADY BEEN STOPPED AT A PREVIOUS TIME STEP:
            if apex.type == "Stopped" or apex.type == "Just_stopped":
                # The time since growth stopped is simply increased by one time step:
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
                # The type is (re)declared "Stopped":
                apex.type = "Stopped"
                # The times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

            # OTHERWISE, THE APEX HAS TO STOP DURING THIS TIME STEP:
            else:
                # The type is declared "Just stopped":
                apex.type = "Just_stopped"
                # Then the exact time since growth stopped is calculated:
                apex.thermal_time_since_growth_stopped = apex.thermal_time_since_emergence + time_step_in_seconds * temperature_time_adjustment - apex.growth_duration
                apex.actual_time_since_growth_stopped = apex.thermal_time_since_growth_stopped / temperature_time_adjustment

                # We record the different element that can contribute to the C supply necessary for growth,
                # and we calculate a mean concentration of hexose in this supplying zone:
                calculating_C_supply_for_elongation(g, apex)
                # And the potential elongation of the apex before growth stopped is calculated:
                apex.potential_length = elongated_length(initial_length=apex.length, radius=apex.radius,
                                                         C_hexose_root=apex.growing_zone_C_hexose_root,
                                                         elongation_time_in_seconds=time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped,
                                                         ArchiSimple=ArchiSimple,
                                                         soil_temperature_in_Celsius=soil_temperature_in_Celsius)
                # VERIFICATION:
                if time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped < 0.:
                    print("!!! ERROR: The apex", apex.index(), "has stopped since",
                          apex.actual_time_since_growth_stopped,
                          "seconds; the time step is", time_step_in_seconds)
                    print("We set the potential length of this apex equal to its initial length.")
                    apex.potential_length = apex.initial_length

                # The times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

        # OTHERWISE, THE APEX MUST BE DEAD:
        else:
            # IF THE APEX HAS ALREADY DIED AT A PREVIOUS TIME STEP:
            if apex.type == "Dead" or apex.type == "Just_dead":
                # The type is (re)declared "Dead":
                apex.type = "Dead"
                # And the times are simply incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.actual_time_since_death += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_death += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # OTHERWISE, THE APEX HAS TO DIE DURING THIS TIME STEP:
            else:
                # Then the apex is declared "Just dead":
                apex.type = "Just_dead"
                # The exact time since the apex died is calculated:
                apex.thermal_time_since_death = apex.thermal_time_since_growth_stopped + time_step_in_seconds * temperature_time_adjustment - apex.life_duration
                apex.actual_time_since_death = apex.thermal_time_since_death / temperature_time_adjustment
                # And the other times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

    # VERIFICATION: If the apex does not match any of the cases listed above:
    print("!!! ERROR: No case found for defining growth of apex", apex.index(), "of type", apex.type)
    new_apex.append(apex)
    return new_apex

# Function calculating the potential development of a root segment:
#------------------------------------------------------------------
def potential_segment_development(g, segment, time_step_in_seconds=60. * 60. * 24., radial_growth="Possible",
                                  ArchiSimple=False, soil_temperature_in_Celsius=20):
    """

    """

    # We initialize an empty list that will contain the new segment to be returned:
    new_segment = []
    # We record the current radius and length prior to growth as the initial radius and length:
    segment.initial_radius = segment.radius
    segment.initial_length = segment.length
    # We initialize the properties "potential_radius" and "potential_length":
    segment.theoretical_radius = segment.radius
    segment.potential_radius = segment.radius
    segment.potential_length = segment.length

    # CASE 1: THE SEGMENT IS A NODULE:
    # ################################

    if segment.type == "Root_nodule":
        # We consider the amount of hexose available in the nodule AND in the parent segment
        # (EXCLUDING the amount of hexose in living rot hairs):!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        index_parent = g.Father(segment.index(), EdgeType='+')
        parent = g.node(index_parent)
        segment.hexose_available_for_thickening = parent.C_hexose_root * parent.struct_mass \
                                                  + segment.C_hexose_root * segment.struct_mass
        # We calculate an average concentration of hexose that will help to regulate nodule growth:
        C_hexose_regulating_nodule_growth = segment.hexose_available_for_thickening / (
                parent.struct_mass + segment.struct_mass)
        # We modulate the relative increase in radius by the amount of C available in the nodule:
        thickening_rate = param.relative_nodule_thickening_rate_max \
                          * C_hexose_regulating_nodule_growth / (
                                  param.Km_nodule_thickening + C_hexose_regulating_nodule_growth)
        # We calculate a coefficient that will modify the rate of thickening according to soil temperature
        # assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
        thickening_rate = thickening_rate * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)
        segment.theoretical_radius = segment.radius * (1 + thickening_rate * time_step_in_seconds)
        if segment.theoretical_radius > param.nodule_max_radius:
            segment.potential_radius = param.nodule_max_radius
        else:
            segment.potential_radius = segment.theoretical_radius
        # We add the modified segment to the list of new segments, and we quit the function here:
        new_segment.append(segment)
        return new_segment

    # CASE 2: THE SEGMENT IS NOT A NODULE:
    ######################################

    # We initialize internal variables:
    son_section = 0.
    sum_of_lateral_sections = 0.
    number_of_actual_children = 0.
    death_count = 0.
    list_of_times_since_death = []

    # We define the amount of hexose available for thickening
    # (EXCLUDING the amount of hexose in living root hairs): !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    segment.hexose_available_for_thickening = segment.C_hexose_root * segment.struct_mass

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # ------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
    # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)

    # CALCULATING DEATH OR RADIAL GROWTH:
    # -----------------------------------

    # For each child of the segment:
    for child in segment.children():

        # Then we add one child to the actual number of children:
        number_of_actual_children += 1

        if child.radius < 0. or child.potential_radius < 0.:
            print("!!! ERROR: the radius of the element", child.index(), "is negative!")
        # If the child belongs to the same axis:
        if child.properties()['edge_type'] == '<':
            # Then we record the THEORETICAL section of this child:
            son_section = child.theoretical_radius ** 2 * pi
            # # Then we record the section of this child:
            # son_section = child.radius * child.radius * pi
        # Otherwise if the child is the element of a lateral root AND if this lateral root has already emerged
        # AND the lateral element is not a nodule:
        elif child.properties()['edge_type'] == '+' and child.length > 0. and child.type != "Root_nodule":
            # We add the POTENTIAL section of this child to a sum of lateral sections:
            sum_of_lateral_sections += child.theoretical_radius ** 2 * pi
            # # We add the section of this child to a sum of lateral sections:
            # sum_of_lateral_sections += child.radius ** 2 * pi

        # If this child has just died or was already dead:
        if child.type == "Just_dead" or child.type == "Dead":
            # Then we add one dead child to the death count:
            death_count += 1
            # And we record the exact time since death:
            list_of_times_since_death.append(child.actual_time_since_death)

    # If each child in the list of children has been recognized as dead or just dead:
    if death_count == number_of_actual_children:
        # If the investigated segment was already declared dead at the previous time step:
        if segment.type == "Just_dead" or segment.type == "Dead":
            # Then we transform its status into "Dead"
            segment.type = "Dead"
        else:
            # Then the segment has to die:
            segment.type = "Just_dead"
    # Otherwise, at least one of the children axis is not dead, so the father segment should not be dead

    # REGULATION OF RADIAL GROWTH BY AVAILABLE CARBON:
    # ------------------------------------------------
    # If the radial growth is possible:
    if radial_growth == "Possible":
        # The radius of the root segment is defined according to the pipe model.
        # In ArchiSimp9, the radius is increased by considering the sum of the sections of all the children,
        # by adding a fraction (SGC) of this sum of sections to the current section of the parent segment,
        # and by calculating the new radius that corresponds to this new section of the parent:
        segment.theoretical_radius = sqrt(son_section / pi + param.SGC * sum_of_lateral_sections / pi)
        # However, if the net difference is below 0.1% of the initial radius:
        if (segment.theoretical_radius - segment.initial_radius) <= 0.001 * segment.initial_radius:
            # Then the potential radius is set to the initial radius:
            segment.theoretical_radius = segment.initial_radius
        # If we consider simple ArchiSimple rules:
        if ArchiSimple:
            # Then the potential radius to form is equal to the theoretical one determined by geometry:
            segment.potential_radius = segment.theoretical_radius
        # Otherwise, if we don't strictly follow simple ArchiSimple rules and if there can be an increase in radius:
        elif segment.length > 0. and segment.theoretical_radius > segment.radius:
            # We calculate the maximal increase in radius that can be achieved over this time step,
            # based on a Michaelis-Menten formalism that regulates the maximal rate of increase
            # according to the amount of hexose available:
            thickening_rate = param.relative_root_thickening_rate_max \
                              * segment.C_hexose_root / (param.Km_thickening + segment.C_hexose_root)
            # We calculate a coefficient that will modify the rate of thickening according to soil temperature
            # assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
            thickening_rate = thickening_rate * temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=param.root_growth_T_ref,
                A=param.root_growth_A,
                B=param.root_growth_B,
                C=param.root_growth_C)
            # The maximal possible new radius according to this regulation is therefore:
            new_radius_max = (1 + thickening_rate * time_step_in_seconds) * segment.initial_radius
            # If the potential new radius is higher than the maximal new radius:
            if segment.theoretical_radius > new_radius_max:
                # Then potential thickening is limited up to the maximal new radius:
                segment.potential_radius = new_radius_max
            # Otherwise, the potential radius to achieve is equal to the theoretical one:
            else:
                segment.potential_radius = segment.theoretical_radius
        # And if the segment corresponds to one of the elements of length 0 supporting one seminal or adventitious root:
        if segment.type == "Support_for_seminal_root" or segment.type == "Support_for_adventitious_root":
            # Then the radius is directly increased, as this element will not be considered in the function calculating actual growth:
            segment.radius = segment.potential_radius

    # We increase the various time variables:
    segment.actual_time_since_primordium_formation += time_step_in_seconds
    segment.actual_time_since_emergence += time_step_in_seconds
    segment.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
    segment.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment

    if segment.type == "Stopped" or segment.type == "Just_stopped":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
    if segment.type == "Just_dead":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
        # We check that the list of times_since_death is not empty:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if list_of_times_since_death:
            segment.actual_time_since_death = min(list_of_times_since_death)
        else:
            segment.actual_time_since_death = 0.
        segment.thermal_time_since_death = segment.actual_time_since_death * temperature_time_adjustment
    if segment.type == "Dead":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
        segment.actual_time_since_death += time_step_in_seconds
        segment.thermal_time_since_death += time_step_in_seconds * temperature_time_adjustment

    new_segment.append(segment)
    return new_segment

# Simulation of potential root growth for all root elements:
#-----------------------------------------------------------
# We define a class "Simulate" which is used to simulate the development of apices and segments on the whole MTG "g":
class Simulate_potential_growth(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        """ Simulate on MTG. """
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex" or "Segment", from the tip to the base:
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = next(root_gen)
        self.apices_list = [g.node(v) for v in pre_order(g, root) if g.label(v) == 'Apex']
        self.segments_list = [g.node(v) for v in post_order(g, root) if g.label(v) == 'Segment']

    def step(self, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth="Possible", ArchiSimple=False,
             soil_temperature_in_Celsius=20):
        list_of_apices = list(self.apices_list)
        list_of_segments = list(self.segments_list)

        # For each apex in the list of apices:
        for apex in list_of_apices:
            # We define the new list of apices with the function apex_development:
            new_apices = [potential_apex_development(self.g, apex, time_step_in_seconds=time_step_in_seconds,
                                                     ArchiSimple=ArchiSimple,
                                                     soil_temperature_in_Celsius=soil_temperature_in_Celsius)]

            # We add these new apices to apex:
            self.apices_list.extend(new_apices)
        # For each segment in the list of segments:
        for segment in list_of_segments:
            # We define the new list of apices with the function apex_development:
            new_segments = [potential_segment_development(self.g, segment, time_step_in_seconds=time_step_in_seconds,
                                                          radial_growth=radial_growth, ArchiSimple=ArchiSimple,
                                                          soil_temperature_in_Celsius=soil_temperature_in_Celsius)]

            # We add these new apices to apex:
            self.segments_list.extend(new_segments)


# Function that calculates the potential growth of the whole MTG at a given time step:
#-------------------------------------------------------------------------------------
def potential_growth(g, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth="Possible", ArchiSimple=False,
                     soil_temperature_in_Celsius=20):
    # We simulate the development of all apices and segments in the MTG:
    simulator = Simulate_potential_growth(g)
    simulator.step(time_step_in_seconds=time_step_in_seconds, radial_growth=radial_growth, ArchiSimple=ArchiSimple,
                   soil_temperature_in_Celsius=soil_temperature_in_Celsius)

    return

# Function that divides a root into segment and generates root primordia:
#------------------------------------------------------------------------
def segmentation_and_primordium_formation(g, apex, time_step_in_seconds=1. * 60. * 60. * 24.,
                                          soil_temperature_in_Celsius=20, ArchiSimple=False, random=True,
                                          nodules=True, root_order_limitation=False, root_order_treshold=2):
    # NOTE: This function is supposed to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    # Optional - We can add random geometry, or not:
    if random:
        # The seed used to generate random values is defined according to a parameter random_choice and the index of the apex:
        np.random.seed(param.random_choice * apex.index())
        angle_mean = 0
        angle_var = 5
        segment_angle_down = np.random.normal(angle_mean, angle_var)
        segment_angle_roll = np.random.normal(angle_mean, angle_var)
        apex_angle_down = np.random.normal(angle_mean, angle_var)
        apex_angle_roll = np.random.normal(angle_mean, angle_var)
    else:
        segment_angle_down = 0
        segment_angle_roll = 0
        apex_angle_down = 0
        apex_angle_roll = 0

    # We initialize the new_apex that will be returned by the function:
    new_apex = []
    # We record the initial geometrical features and total amounts,
    # knowing that the concentrations will not be changed by the segmentation:
    initial_root_order=apex.root_order
    initial_length = apex.length
    initial_dist_to_ramif = apex.dist_to_ramif
    initial_elongation = apex.actual_elongation
    initial_elongation_rate = apex.actual_elongation_rate
    initial_struct_mass = apex.struct_mass
    initial_resp_maintenance = apex.resp_maintenance
    initial_resp_growth = apex.resp_growth
    initial_struct_mass_produced = apex.struct_mass_produced
    initial_initial_struct_mass = apex.initial_struct_mass
    initial_initial_living_root_hairs_struct_mass = apex.initial_living_root_hairs_struct_mass
    # Note: this is not an error, we also need to record the initial structural mass before growth!
    initial_hexose_exudation = apex.hexose_exudation
    initial_hexose_uptake = apex.hexose_uptake
    initial_mucilage_secretion = apex.mucilage_secretion
    initial_cells_release = apex.cells_release
    initial_total_rhizodeposition = apex.total_rhizodeposition
    initial_hexose_degradation = apex.hexose_degradation
    initial_mucilage_degradation = apex.mucilage_degradation
    initial_cells_degradation = apex.cells_degradation
    initial_hexose_growth_demand = apex.hexose_growth_demand
    initial_hexose_consumption_by_growth = apex.hexose_consumption_by_growth
    initial_hexose_production_from_phloem = apex.hexose_production_from_phloem
    initial_sucrose_loading_in_phloem = apex.sucrose_loading_in_phloem
    initial_hexose_mobilization_from_reserve = apex.hexose_mobilization_from_reserve
    initial_hexose_immobilization_as_reserve = apex.hexose_immobilization_as_reserve
    initial_Deficit_sucrose_root = apex.Deficit_sucrose_root
    initial_Deficit_hexose_root = apex.Deficit_hexose_root
    initial_Deficit_hexose_soil = apex.Deficit_hexose_soil
    initial_Deficit_mucilage_soil = apex.Deficit_mucilage_soil
    initial_Deficit_cells_soil = apex.Deficit_cells_soil

    # We record the type of the apex, as it may correspond to an apex that has stopped (or even died):
    initial_type = apex.type
    initial_lateral_root_emergence_possibility = apex.lateral_root_emergence_possibility

    # If the length of the apex is smaller than the defined length of a root segment:
    if apex.length <= param.segment_length:

        # We assume that the growth functions that may have been called previously have only modified radius and length,
        # but not the struct_mass and the total amounts present in the root element.
        # We modify the geometrical features of the present element according to the new length and radius:
        apex.volume = surfaces_and_volumes(g, apex, apex.radius, apex.potential_length)["volume"]
        apex.struct_mass = apex.volume * param.root_tissue_density

        # We simply call the function primordium_formation to check whether a primordium should have been formed
        # (Note: we assume that the segment length is always smaller than the inter-branching distance IBD,
        # so that in this case, only 0 or 1 primordium may have been formed - the function is called only once):
        new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                                             root_order_limitation=root_order_limitation,
                                             root_order_treshold=root_order_treshold))

    # Otherwise, we have to calculate the number of entire segments within the apex.
    else:

        # If the final length of the apex does not correspond to an entire number of segments:
        if apex.length / param.segment_length - floor(apex.length / param.segment_length) > 0.:
            # Then the total number of segments to be formed is:
            n_segments = floor(apex.length / param.segment_length)
        else:
            # Otherwise, the number of segments to be formed is decreased by 1,
            # so that the last element corresponds to an apex with a positive length:
            n_segments = floor(apex.length / param.segment_length) - 1
        n_segments = int(n_segments)

        # We develop each new segment, except the last one, by transforming the current apex into a segment
        # and by adding a new apex after it, in an iterative way for (n-1) segments:
        for i in range(1, n_segments):
            # We define the length of the present element as the constant length of a segment:
            apex.length = param.segment_length
            # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
            apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - param.segment_length * i)
            # We modify the geometrical features of the present element according to the new length:
            apex.volume = surfaces_and_volumes(g, apex, apex.radius, apex.length)["volume"]
            apex.struct_mass = apex.volume * param.root_tissue_density

            # We calculate the mass fraction that the segment represents compared to the whole element prior to segmentation:
            mass_fraction = apex.struct_mass / initial_struct_mass

            # We modify the variables representing total amounts according to this mass fraction:
            apex.resp_maintenance = initial_resp_maintenance * mass_fraction
            apex.resp_growth = initial_resp_growth * mass_fraction

            apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
            apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction
            apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction

            apex.hexose_exudation = initial_hexose_exudation * mass_fraction
            apex.hexose_uptake = initial_hexose_uptake * mass_fraction
            apex.mucilage_secretion = initial_mucilage_secretion * mass_fraction
            apex.cells_release = initial_cells_release * mass_fraction
            apex.total_rhizodeposition = initial_total_rhizodeposition * mass_fraction
            apex.hexose_degradation = initial_hexose_degradation * mass_fraction
            apex.mucilage_degradation = initial_mucilage_degradation * mass_fraction
            apex.cells_degradation = initial_cells_degradation * mass_fraction
            apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
            apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

            apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
            apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
            apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
            apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

            apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
            apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
            apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
            apex.Deficit_mucilage_soil = initial_Deficit_mucilage_soil * mass_fraction
            apex.Deficit_cells_soil = initial_Deficit_cells_soil * mass_fraction
            # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
            new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                                 time_step_in_seconds=time_step_in_seconds,
                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                                                 random=random,
                                                 root_order_limitation=root_order_limitation,
                                                 root_order_treshold=root_order_treshold))
            # The current element that has been elongated up to segment_length is now considered as a segment:
            apex.label = 'Segment'

            # And we add a new apex after this segment, initially of length 0, that is now the new element called "apex":
            apex = ADDING_A_CHILD(mother_element=apex, edge_type='<', label='Apex',
                                  type=apex.type,
                                  root_order=initial_root_order,
                                  angle_down=segment_angle_down,
                                  angle_roll=segment_angle_roll,
                                  length=0.,
                                  radius=apex.radius,
                                  identical_properties=True,
                                  nil_properties=False)
            apex.actual_elongation = param.segment_length * i

        # Finally, we do this operation one last time for the last segment:
        # We define the length of the present element as the constant length of a segment:
        apex.length = param.segment_length
        apex.potential_length = apex.length
        apex.initial_length = apex.length
        # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
        apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - param.segment_length * n_segments)
        # We modify the geometrical features of the present element according to the new length:
        apex.volume = surfaces_and_volumes(g, apex, apex.radius, apex.length)["volume"]
        apex.struct_mass = apex.volume * param.root_tissue_density
        # We modify the variables representing total amounts according to the mass fraction:
        mass_fraction = apex.struct_mass / initial_struct_mass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
        apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction
        apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.mucilage_secretion = initial_mucilage_secretion * mass_fraction
        apex.cells_release = initial_cells_release * mass_fraction
        apex.total_rhizodeposition = initial_total_rhizodeposition * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.mucilage_degradation = initial_mucilage_degradation * mass_fraction
        apex.cells_degradation = initial_cells_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

        apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
        apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
        apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
        apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        apex.Deficit_mucilage_soil = initial_Deficit_mucilage_soil * mass_fraction
        apex.Deficit_cells_soil = initial_Deficit_cells_soil * mass_fraction
        # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
        new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                                             root_order_limitation=root_order_limitation,
                                             root_order_treshold=root_order_treshold))
        # The current element that has been elongated up to segment_length is now considered as a segment:
        apex.label = 'Segment'

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # We add the possibility of a nodule formation on the segment that is closest to the apex:
        if nodules and len(apex.children()) < 2 and np.random.random() < param.nodule_formation_probability:
            nodule_formation(g, mother_element=apex)

        # And we define the new, final apex after the new defined segment, with a new length defined as:
        new_length = initial_length - n_segments * param.segment_length
        apex = ADDING_A_CHILD(mother_element=apex, edge_type='<', label='Apex',
                              type=apex.type,
                              root_order=initial_root_order,
                              angle_down=segment_angle_down,
                              angle_roll=segment_angle_roll,
                              length=new_length,
                              radius=apex.radius,
                              identical_properties=True,
                              nil_properties=False)
        apex.potential_length = new_length
        apex.initial_length = new_length
        apex.dist_to_ramif = initial_dist_to_ramif
        apex.actual_elongation = initial_elongation

        # We modify the geometrical features of the new apex according to the defined length:
        apex.volume = surfaces_and_volumes(g, apex, apex.radius, apex.length)["volume"]
        apex.struct_mass = apex.volume * param.root_tissue_density
        # We modify the variables representing total amounts according to the new struct_mass:
        mass_fraction = apex.struct_mass / initial_struct_mass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
        apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction
        apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.mucilage_secretion = initial_mucilage_secretion * mass_fraction
        apex.cells_release = initial_cells_release * mass_fraction
        apex.total_rhizodeposition = initial_total_rhizodeposition * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.mucilage_degradation = initial_mucilage_degradation * mass_fraction
        apex.cells_degradation = initial_cells_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

        apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
        apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
        apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
        apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        apex.Deficit_mucilage_soil = initial_Deficit_mucilage_soil * mass_fraction
        apex.Deficit_cells_soil = initial_Deficit_cells_soil * mass_fraction
        # And we call the function primordium_formation to check whether a primordium should have been formed:
        new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                                             root_order_limitation=root_order_limitation,
                                             root_order_treshold=root_order_treshold))
        # And we add the last apex present at the end of the elongated axis:
        new_apex.append(apex)

    return new_apex

# Simulation of segmentation and primordia formation for all root elements:
#--------------------------------------------------------------------------
# We define a class "Simulate_segmentation_and_primordia_formation" which is used to simulate the segmentation of apices
# and the apparition of primordium for a given MTG:
class Simulate_segmentation_and_primordia_formation(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        """ Simulate on MTG. """
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex":
        self._apices = [g.node(v) for v in g.vertices_iter(scale=1) if g.label(v) == 'Apex']

    def step(self, time_step_in_seconds, soil_temperature_in_Celsius=20, random=True, nodules=False,
             root_order_limitation=False, root_order_treshold=2):
        # We define "apices_list" as the list of all apices in g:
        apices_list = list(self._apices)
        # For each apex in the list of apices:
        for apex in apices_list:
            if apex.type == "Normal_root_after_emergence" and apex.length > 0.:  # Is it needed? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # We define the new list of apices with the function apex_development:
                new_apex = segmentation_and_primordium_formation(self.g, apex, time_step_in_seconds,
                                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                 random=random,
                                                                 nodules=nodules,
                                                                 root_order_limitation=root_order_limitation,
                                                                 root_order_treshold=root_order_treshold)
                # We add these new apices to apex:
                self._apices.extend(new_apex)


# Function that creates new segments and priomordia in "g":
#----------------------------------------------------------
def segmentation_and_primordia_formation(g, time_step_in_seconds=1. * 60. * 60. * 24.,
                                         soil_temperature_in_Celsius=20,
                                         random=True, printing_warnings=False,
                                         nodules=False,
                                         root_order_limitation=False,
                                         root_order_treshold=2):
    # We simulate the segmentation of all apices:
    simulator = Simulate_segmentation_and_primordia_formation(g)
    simulator.step(time_step_in_seconds, soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                   nodules=nodules,
                   root_order_limitation=root_order_limitation, root_order_treshold=root_order_treshold)

    return

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "ACTUAL GROWTH AND ASSOCIATED RESPIRATION"
########################################################################################################################
########################################################################################################################

# Actual elongation, radial growth and growth respiration of root elements:
#--------------------------------------------------------------------------
def actual_growth_and_corresponding_respiration(g, time_step_in_seconds, soil_temperature_in_Celsius=20,
                                                printing_warnings=False):
    """
    This function defines how a segment, an apex and possibly an emerging root primordium will grow according to the amount
    of hexose present in the segment, taking into account growth respiration based on the model of Thornley and Cannell
    (2000). The calculation is based on the values of potential_radius, potential_length, lateral_root_emergence_possibility
    and emergence_cost defined in each element by the module "POTENTIAL GROWTH".
    The function returns the MTG "g" with modified values of radius and length of each element, the possibility of the
    emergence of lateral roots, and the cost of growth in terms of hexose demand.
    """

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
    # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)

    # PROCEEDING TO ACTUAL GROWTH:
    # -----------------------------

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)
    # We cover all the vertices in the MTG, from the tips to the base:
    for vid in post_order(g, root):

        # n represents the current root element:
        n = g.node(vid)

        # AVOIDANCE OF UNWANTED CASES:
        # -----------------------------
        # We make sure that the element is not dead:
        if n.type == "Dead" or n.type == "Just_dead" or n.type == "Support_for_seminal_root" or n.type == "Support_for_adventitious_root":
            # In such case, we just pass to the next element in the iteration:
            continue

        # We make sure that there is a potential growth for this element:
        if n.potential_length <= n.initial_length and n.potential_radius <= n.initial_radius:
            # In such case, we just pass to the next element in the iteration:
            continue

        # INITIALIZATION AND CALCULATIONS OF POTENTIAL GROWTH DEMAND IN HEXOSE:
        # ----------------------------------------------------------------------

        # WARNING: All growth related variables should have been initialized by another module at the beginning of the time step!!!

        # We calculate the initial volume of the element:
        initial_volume = surfaces_and_volumes(g, n, n.initial_radius, n.initial_length)["volume"]
        # We calculate the potential volume of the element based on the potential radius and potential length:
        potential_volume = surfaces_and_volumes(g, n, n.potential_radius, n.potential_length)["volume"]
        # We calculate the number of moles of hexose required for growth, including the respiration cost according to
        # the yield growth included in the model of Thornley and Cannell (2000), where root_tissue_density is the dry structural
        # weight per volume (g m-3) and struct_mass_C_content is the amount of C per gram of dry structural mass (mol_C g-1):
        n.hexose_growth_demand = (potential_volume - initial_volume) \
                                 * param.root_tissue_density * param.struct_mass_C_content / param.yield_growth * 1 / 6.
        # We verify that this potential growth demand is positive:
        if n.hexose_growth_demand < 0.:
            print("!!! ERROR: a negative growth demand of", n.hexose_growth_demand,
                  "was calculated for the element", n.index(), "of class", n.label)
            print("The initial volume is", initial_volume, "the potential volume is", potential_volume)
            print("The initial length was", n.initial_length, "and the potential length was", n.potential_length)
            print("The initial radius was", n.initial_radius, "and the potential radius was", n.potential_radius)
            n.hexose_growth_demand = 0.
            # In such case, we just pass to the next element in the iteration:
            continue

        # CALCULATIONS OF THE AMOUNT OF HEXOSE AVAILABLE FOR GROWTH:
        # ---------------------------------------------------------

        # We initialize each amount of hexose available for growth:
        hexose_possibly_required_for_elongation = 0.
        hexose_available_for_thickening = 0.

        # If elongation is possible:
        if n.potential_length > n.length:
            hexose_possibly_required_for_elongation = n.hexose_possibly_required_for_elongation
            list_of_elongation_supporting_elements = n.list_of_elongation_supporting_elements
            list_of_elongation_supporting_elements_hexose = n.list_of_elongation_supporting_elements_hexose
            list_of_elongation_supporting_elements_mass = n.list_of_elongation_supporting_elements_mass

        # If radial growth is possible:
        if n.potential_radius > n.radius:
            # We only consider the amount of hexose immediately available in the element that can increase in radius:
            hexose_available_for_thickening = n.hexose_available_for_thickening

        # In case no hexose is available at all:
        if (hexose_possibly_required_for_elongation + hexose_available_for_thickening) <= 0.:
            # Then we move to the next element in the main loop:
            continue

        # We initialize the temporary variable "remaining_hexose" that computes the amount of hexose left for growth:
        remaining_hexose_for_elongation = hexose_possibly_required_for_elongation
        remaining_hexose_for_thickening = hexose_available_for_thickening

        # ACTUAL ELONGATION IS FIRST CONSIDERED:
        # ---------------------------------------

        # We calculate the maximal possible length of the root element according to all the hexose available for elongation:
        volume_max = initial_volume + hexose_possibly_required_for_elongation * 6. \
                     / (param.root_tissue_density * param.struct_mass_C_content) * param.yield_growth
        length_max = volume_max / (pi * n.initial_radius ** 2)

        # If the element can elongate:
        if n.potential_length > n.initial_length:

            # CALCULATING ACTUAL ELONGATION:
            # If elongation is possible but is limited by the amount of hexose available:
            if n.potential_length >= length_max:
                # Elongation is limited using all the amount of hexose available:
                n.length = length_max
            # Otherwise, elongation can be done up to the full potential:
            else:
                # Elongation is done up to the full potential:
                n.length = n.potential_length
            # The corresponding new volume is calculated:
            volume_after_elongation = (pi * n.initial_radius ** 2) * n.length
            # The overall cost of elongation is calculated as:
            hexose_consumption_by_elongation = \
                1. / 6. * (volume_after_elongation - initial_volume) \
                * param.root_tissue_density * param.struct_mass_C_content / param.yield_growth

            # If there has been an actual elongation:
            if n.length > n.initial_length:

                # REGISTERING THE COSTS FOR ELONGATION:
                # We cover each of the elements that have provided hexose for sustaining the elongation of element n:
                for i in range(0, len(list_of_elongation_supporting_elements)):
                    index = list_of_elongation_supporting_elements[i]
                    supplying_element = g.node(index)
                    # We define the actual contribution of the current element based on total hexose consumption by growth
                    # of element n and the relative contribution of the current element to the pool of the available hexose:
                    hexose_actual_contribution_to_elongation = hexose_consumption_by_elongation \
                                                               * list_of_elongation_supporting_elements_hexose[i] \
                                                               / hexose_possibly_required_for_elongation
                    # The amount of hexose used for growth in this element is increased:
                    supplying_element.hexose_consumption_by_growth += hexose_actual_contribution_to_elongation
                    # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                    supplying_element.resp_growth += hexose_actual_contribution_to_elongation \
                                                     * (1 - param.yield_growth) * 6.

        # ACTUAL RADIAL GROWTH IS THEN CONSIDERED:
        # -----------------------------------------

        # If the radius of the element can increase:
        if n.potential_radius > n.initial_radius:

            # CALCULATING ACTUAL THICKENING:
            # We calculate the increase in volume that can be achieved with the amount of hexose available:
            possible_radial_increase_in_volume = \
                remaining_hexose_for_thickening * 6. * param.yield_growth \
                / (param.root_tissue_density * param.struct_mass_C_content)
            # We calculate the maximal possible volume based on the volume of the new cylinder after elongation
            # and the increase in volume that could be achieved by consuming all the remaining hexose:
            volume_max = surfaces_and_volumes(g, n, n.initial_radius, n.length)["volume"] \
                         + possible_radial_increase_in_volume
            # We then calculate the corresponding new possible radius corresponding to this maximum volume:
            if n.type == "Root_nodule":
                # If the element corresponds to a nodule, then it we calculate the radius of a theoretical sphere:
                possible_radius = (3. / (4. * pi)) ** (1. / 3.)
            else:
                # Otherwise, we calculate the radius of a cylinder:
                possible_radius = sqrt(volume_max / (n.length * pi))
            if possible_radius < 0.9999 * n.initial_radius:  # We authorize a difference of 0.01% due to calculation errors!
                print("!!! ERROR: the calculated new radius of element", n.index(), "is lower than the initial one!")
                print("The possible radius was", possible_radius, "and the initial radius was", n.initial_radius)

            # If the maximal radius that can be obtained is lower than the potential radius suggested by the potential growth module:
            if possible_radius <= n.potential_radius:
                # Then radial growth is limited and there is no remaining hexose after radial growth:
                n.radius = possible_radius
                hexose_actual_contribution_to_thickening = remaining_hexose_for_thickening
                remaining_hexose_for_thickening = 0.
            else:
                # Otherwise, radial growth is done up to the full potential and the remaining hexose is calculated:
                n.radius = n.potential_radius
                net_increase_in_volume = surfaces_and_volumes(g, n, n.radius, n.length)["volume"] \
                                         - surfaces_and_volumes(g, n, n.initial_radius, n.length)["volume"]
                # net_increase_in_volume = pi * (n.radius ** 2 - n.initial_radius ** 2) * n.length
                # We then calculate the remaining amount of hexose after thickening:
                hexose_actual_contribution_to_thickening = \
                    1. / 6. * net_increase_in_volume \
                    * param.root_tissue_density * param.struct_mass_C_content / param.yield_growth

            # REGISTERING THE COSTS FOR THICKENING:
            #--------------------------------------
            fraction_of_available_hexose_in_the_element = \
                (n.C_hexose_root * n.initial_struct_mass) / hexose_available_for_thickening
            # The amount of hexose used for growth in this element is increased:
            n.hexose_consumption_by_growth += \
                (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)
            # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
            n.resp_growth += \
                (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) \
                * (1 - param.yield_growth) * 6.
            if n.type == "Root_nodule":
                index_parent = g.Father(n.index(), EdgeType='+')
                parent = g.node(index_parent)
                fraction_of_available_hexose_in_the_element = \
                    (parent.C_hexose_root * parent.initial_struct_mass) / hexose_available_for_thickening
                # The amount of hexose used for growth in this element is increased:
                parent.hexose_consumption_by_growth += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)
                # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                parent.resp_growth += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) \
                    * (1 - param.yield_growth) * 6.

        # RECORDING THE ACTUAL STRUCTURAL MODIFICATIONS:
        # -----------------------------------------------

        # The new volume and surfaces of the element is automatically calculated:
        n.external_surface = surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
        n.volume = surfaces_and_volumes(g, n, n.radius, n.length)["volume"]
        n.phloem_surface = surfaces_and_volumes(g, n, n.radius, n.length)["phloem_surface"]
        n.symplasm_surface = surfaces_and_volumes(g, n, n.radius, n.length)["symplasm_surface"]
        # The new dry structural struct_mass of the element is calculated from its new volume:
        n.struct_mass = n.volume * param.root_tissue_density
        n.struct_mass_produced = (n.volume - initial_volume) * param.root_tissue_density

        # Verification: we check that no negative length or struct_mass have been generated!
        if n.volume < 0:
            print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length,
                  "and a mass of", n.struct_mass)
            # We then reset all the geometrical values to their initial values:
            n.length = n.initial_length
            n.radius = n.initial_radius
            n.struct_mass = n.initial_struct_mass
            n.struct_mass_produced = 0.
            n.external_surface = n.initial_surface
            n.volume = initial_volume

        # MODIFYING LOCAL OR GLOBAL PROPERTIES:
        # --------------------------------------
        # If there has been an actual elongation:
        if n.length > n.initial_length:

            # If the elongated apex corresponded to any primordium that has been allowed to emerge:
            if n.type == "Seminal_root_before_emergence" \
                    or n.type == "Adventitious_root_before_emergence" \
                    or n.type == "Normal_root_before_emergence":
                # We now consider the apex to have emerged:
                n.type = "Normal_root_after_emergence"
                # The exact time since emergence is recorded:
                n.thermal_time_since_emergence = n.thermal_potential_time_since_emergence
                n.actual_time_since_emergence = n.thermal_time_since_emergence / temperature_time_adjustment
                # The actual elongation rate is calculated:
                n.actual_elongation = n.length - n.initial_length
                n.actual_elongation_rate = n.actual_elongation / n.actual_time_since_emergence
                # Note: at this stage, no sugar has been allocated to the emerging primordium itself!
                # # The parent is the base of the root system:
                # parent = g.node(1)
                # # The possibility of emergence of a lateral root from the parent is forbidden again:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # parent.lateral_root_emergence_possibility = "Impossible"
            elif n.type == "Normal_root_after_emergence":
                # The actual elongation rate is calculated:
                n.actual_elongation = n.length - n.initial_length
                n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

            # The distance to the last ramification is increased:
            n.dist_to_ramif += n.actual_elongation

    return g

# Function calculating a satisfaction coefficient for the growth of the whole root system:
#-----------------------------------------------------------------------------------------
def satisfaction_coefficient(g, struct_mass_input):
    # We initialize the sum of individual demands for struct_mass:
    sum_struct_mass_demand = 0.
    SC = 0.

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):
        # n represents the current root element:
        n = g.node(vid)

        # We calculate the initial volume of the element:
        initial_volume = surfaces_and_volumes(g, n, n.initial_radius, n.initial_length)["volume"]
        # We calculate the potential volume of the element based on the potential radius and potential length:
        potential_volume = surfaces_and_volumes(g, n, n.potential_radius, n.potential_length)["volume"]

        # The growth demand of the element in struct_mass is calculated:
        n.growth_demand_in_struct_mass = (potential_volume - initial_volume) * param.root_tissue_density
        sum_struct_mass_demand += n.growth_demand_in_struct_mass

    # We make sure that the structural mass input is not negative, as this case does not work with ArchiSimple:
    if struct_mass_input < 0.:
        struct_mass_input = 0.

    # We calculate the overall satisfaction coefficient SC described by Pages et al. (2014):
    if sum_struct_mass_demand <= 0:
        print("!!! ERROR: The total growth demand calculated for ArchiSimple was nil or negative. "
              "The satisfaction coefficient of ArchiSimple has been set to 1.")
        SC = 1.
    else:
        SC = struct_mass_input / sum_struct_mass_demand

    return SC

# Function performing the growth of each element based on the potential growth and the satisfaction coefficient SC:
#------------------------------------------------------------------------------------------------------------------
def ArchiSimple_growth(g, SC, time_step_in_seconds, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    SC is the satisfaction coefficient for growth calculated on the whole root system.
    """

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
    # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)

    # PERFORMING ARCHISIMPLE GROWTH:
    # -------------------------------

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):

        # n represents the current root element:
        n = g.node(vid)

        # We make sure that the element is not dead and has not already been stopped at the previous time step:
        if n.type == "Dead" or n.type == "Just_dead" or n.type == "Stopped":
            # Then we pass to the next element in the iteration:
            continue
        # We make sure that the root elements at the basis that support adventitious root are not considered:
        if n.type == "Support_for_seminal_root" or n.type == "Support_for_adventitious_root":
            # Then we pass to the next element in the iteration:
            continue

        # We perform each type of growth according to the satisfaction coefficient SC:
        if SC > 1.:
            relative_growth_increase = 1.
        elif SC < 0:
            print("!!! ERROR: Satisfaction coefficient was negative!!! We set it to 0.")
            relative_growth_increase = 0.
        else:
            relative_growth_increase = SC

        # WARNING: This approach is not an exact C balance on the root system! The relative reduction of growth caused
        # by SC should not be the same between elongation and radial growth!
        n.length += (n.potential_length - n.initial_length) * relative_growth_increase
        n.actual_elongation = n.length - n.initial_length

        # We calculate the actual elongation rate of this element:
        if (n.thermal_potential_time_since_emergence > 0) and (
                n.thermal_potential_time_since_emergence < time_step_in_seconds):
            n.actual_elongation_rate = n.actual_elongation / (
                    n.thermal_potential_time_since_emergence / temperature_time_adjustment)
        else:
            n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

        n.radius += (n.potential_radius - n.initial_radius) * relative_growth_increase
        # The volume of the element is automatically calculated:
        n.volume = surfaces_and_volumes(g, n, n.radius, n.length)["volume"]
        # The new dry structural struct_mass of the element is calculated from its new volume:
        n.struct_mass = n.volume * param.root_tissue_density

        # In case where the root element corresponds to an apex, the distance to the last ramification is increased:
        if n.label == "Apex":
            n.dist_to_ramif += n.actual_elongation

        # VERIFICATION:
        if n.length < 0 or n.struct_mass < 0:
            print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length,
                  "and a mass of", n.struct_mass)

    return g

# Function for reinitializing all growth-related variables at the beginning or end of a time step:
#-------------------------------------------------------------------------------------------------
def reinitializing_growth_variables(g):

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We set to 0 the growth-related variables:
        n.hexose_consumption_by_growth = 0.
        n.hexose_possibly_required_for_elongation = 0.
        n.resp_growth = 0.
        n.struct_mass_produced = 0.
        n.root_hairs_struct_mass_produced = 0.
        n.hexose_growth_demand = 0.
        n.actual_elongation = 0.
        n.actual_elongation_rate = 0.

        # We make sure that the initial values of length, radius and struct_mass are correctly initialized:
        n.initial_length = n.length
        n.initial_radius = n.radius
        n.potential_radius = n.radius
        n.theoretical_radius = n.radius
        n.initial_struct_mass = n.struct_mass
        n.initial_living_root_hairs_struct_mass = n.living_root_hairs_struct_mass
        n.initial_surface = n.external_surface

    return

# Root hairs dynamics:
# --------------------
def root_hairs_dynamics(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                       soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_hairs_dynamics" computes the evolution of the density and average length of root hairs
    along each root axis.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue

        # We also exclude nodules and dead elements from this computation:
        if n.type == "Just_dead" or n.type == "Dead" or n.type=="Nodule":
            continue

        # We calculate the equivalent of a thermal time for the current time step:
        temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                               process_at_T_ref=1,
                                                               T_ref=param.root_growth_T_ref,
                                                               A=param.root_growth_A,
                                                               B=param.root_growth_B,
                                                               C=param.root_growth_C)
        elapsed_thermal_time = time_step_in_seconds * temperature_time_adjustment

        # We keep in memory the initial mass of the living root hairs:
        initial_living_root_hairs_struct_mass = n.living_root_hairs_struct_mass

        # We calculate the total number of (newly formed) root hairs (if any) and update their age:
        #------------------------------------------------------------------------------------------
        # CASE 1 - If the current element is completely included within the actual growing zone of the root at the root
        # tip, the root hairs cannot be formed yet:
        if n.dist_to_tip <= param.growing_zone_factor * n.radius:
            # We stop here with the calculations and move to the next element:
            continue
        # CASE 2 - If all root hairs have already been formed:
        if n.all_root_hairs_formed:
            # Then we simply increase the time since root hairs emergence started:
            n.actual_time_since_root_hairs_emergence_started += time_step_in_seconds
            n.thermal_time_since_root_hairs_emergence_started += elapsed_thermal_time
            n.actual_time_since_root_hairs_emergence_stopped += time_step_in_seconds
            n.thermal_time_since_root_hairs_emergence_stopped += elapsed_thermal_time
        # CASE 3 - If the theoretical growing zone limit is located somewhere within the root element:
        elif n.dist_to_tip - n.length < param.growing_zone_factor * n.radius:
            # We first record the previous length of the root hair zone within the element:
            initial_length_with_hairs = n.actual_length_with_hairs
            # Then the new length of the root hair zone is calculated:
            n.actual_length_with_hairs = n.dist_to_tip - param.growing_zone_factor * n.radius
            net_increase_in_root_hairs_length = n.actual_length_with_hairs - initial_length_with_hairs
            # The corresponding number of root hairs is calculated:
            n.total_root_hairs_number = param.root_hairs_density * n. radius * n.actual_length_with_hairs
            # The time since root hair formation started is then calculated, using the recent increase in the length
            # of the current root hair zone and the elongation rate the corresponding root tip. The latter is
            # calculated using the difference between the new dist_to_tip of the element and the previous one:
            elongation_rate_in_actual_time = (n.dist_to_tip - n.former_dist_to_tip) / time_step_in_seconds
            elongation_rate_in_thermal_time = (n.dist_to_tip - n.former_dist_to_tip) / elapsed_thermal_time
            # SUBCASE 3.1 - If root hairs had not emerged at the previous time step:
            if elongation_rate_in_actual_time > 0. and initial_length_with_hairs <= 0.:
                # We increase the time since root hairs emerged by only the fraction of the time step corresponding to the growth of hairs:
                n.actual_time_since_root_hairs_emergence_started += \
                    time_step_in_seconds - net_increase_in_root_hairs_length / elongation_rate_in_actual_time
                n.thermal_time_since_root_hairs_emergence_started += \
                    elapsed_thermal_time - net_increase_in_root_hairs_length / elongation_rate_in_thermal_time
            # SUBCASE 3.2 - the hairs had already started to grow:
            else:
                # Consequently, the full time elapsed during this time step can be added to the age:
                n.actual_time_since_root_hairs_emergence_started += time_step_in_seconds
                n.thermal_time_since_root_hairs_emergence_started += elapsed_thermal_time
        # CASE 4 - the element is now "full" with root hairs as the limit of root elongation is located further down:
        else:
            # The actual time since root hairs emergence started is first increased:
            n.actual_time_since_root_hairs_emergence_started += time_step_in_seconds
            n.thermal_time_since_root_hairs_emergence_started += elapsed_thermal_time
            # We then record the previous length of the root hair zone within the root element:
            initial_length_with_hairs = n.actual_length_with_hairs
            # And the new length of the root hair zone is necessarily the full length of the root element:
            n.actual_length_with_hairs = n.length
            net_increase_in_root_hairs_length = n.actual_length_with_hairs - initial_length_with_hairs
            # The total number of hairs is defined according to the radius and total length of the element:
            n.total_root_hairs_number = param.root_hairs_density * n. radius * n.length
            # The elongation of the corresponding root tip is calculated as the difference between the new
            # dist_to_tip of the element and the previous one:
            elongation_rate_in_actual_time = (n.dist_to_tip - n.former_dist_to_tip) / time_step_in_seconds
            elongation_rate_in_thermal_time = (n.dist_to_tip - n.former_dist_to_tip) / elapsed_thermal_time
            # The actual time since root hairs emergence has stopped is then calculated:
            if elongation_rate_in_actual_time > 0.:
                n.actual_time_since_root_hairs_emergence_stopped += \
                    time_step_in_seconds - net_increase_in_root_hairs_length / elongation_rate_in_actual_time
                n.thermal_time_since_root_hairs_emergence_stopped += \
                    elapsed_thermal_time - net_increase_in_root_hairs_length / elongation_rate_in_thermal_time
            else:
                n.actual_time_since_root_hairs_emergence_stopped += time_step_in_seconds
                n.thermal_time_since_root_hairs_emergence_stopped += elapsed_thermal_time
            # At this stage, all root hairs that could be formed have been formed, so we record this:
            n.all_root_hairs_formed = True

        # We now calculate the number of living and dead root hairs:
        #-----------------------------------------------------------
        # Root hairs are dying when the time since they emerged is higher than their lifespan.
        # If the time since root hairs emergence started is lower than the lifespan,
        # no root hair should be dead:
        if n.thermal_time_since_root_hairs_emergence_started <= n.root_hairs_lifespan:
            n.dead_root_hairs_number = 0.
        elif n.thermal_time_since_root_hairs_emergence_stopped > n.root_hairs_lifespan:
            # Otherwise, if the time since root hairs emergence stopped is higher than the lifespan,
            # all the root hairs of the root element must now be dead:
            n.dead_root_hairs_number = n.total_root_hairs_number
        else:
            # In the intermediate case, there are currently both dead and living root hairs on the root element:
            # We assume that there is a linear decrease of root hair age between the first hair that has emerged
            # and the last one that has emerged:
            time_since_first_death = n.thermal_time_since_root_hairs_emergence_started - n.root_hairs_lifespan
            dead_fraction =  time_since_first_death / (n.thermal_time_since_root_hairs_emergence_started
                                                       - n.thermal_time_since_root_hairs_emergence_stopped)
            n.dead_root_hairs_number = n.total_root_hairs_number * dead_fraction

        # In all cases, the number of the living root hairs is then calculated by difference with the total hair number:
        n.living_root_hairs_number = n.total_root_hairs_number - n.dead_root_hairs_number

        # We calculate the new average root hairs length, if needed:
        # ----------------------------------------------------------
        # If the root hairs had not reached their maximal length:
        if n.root_hair_length < param.root_hair_max_length:
            # The new potential root hairs length is calculated according to the elongation rate,
            # corrected by temperature and modulated by the concentration of hexose (in the same way as for root
            # elongation) available in the root hair zone on the root element:
            new_length = n.root_hair_length + param.root_hairs_elongation_rate * param.root_hair_radius \
                         * n.C_hexose_root * (n.actual_length_with_hairs / n.length) \
                         / (param.Km_elongation + n.C_hexose_root) * elapsed_thermal_time
            # If the new calculated length is higher than the maximal length:
            if new_length > param.root_hair_max_length:
                # We set the root hairs length to the maximal length:
                n.root_hair_length = param.root_hair_max_length
            else:
                # Otherwise, we record the new calculated length:
                n.root_hair_length = new_length

        # We finally calculate the total external surface (m2), volume (m3) and mass (g) of root hairs:
        #----------------------------------------------------------------------------------------------
        # In the calculation of surface, we consider the root hair to be a cylinder, and include the lateral section,
        # but exclude the section of the cylinder at the tip:
        n.root_hairs_external_surface = ((param.root_hair_radius * 2 * pi) * n.root_hair_length) * n.total_root_hairs_number
        n.root_hairs_volume = (param.root_hair_radius ** 2 * pi) * n.root_hair_length * n.total_root_hairs_number
        n.root_hairs_struct_mass = n.root_hairs_volume * param.root_tissue_density
        if n.total_root_hairs_number > 0.:
            n.living_root_hairs_external_surface = n.root_hairs_external_surface * n.living_root_hairs_number \
                                              / n.total_root_hairs_number
            n.living_root_hairs_struct_mass = n.root_hairs_struct_mass * n.living_root_hairs_number \
                                              / n.total_root_hairs_number
        else:
            n.living_root_hairs_external_surface = 0.
            n.living_root_hairs_struct_mass = 0.
        
        # We calculate the mass of hairs that has been effectively produced:
        n.root_hairs_struct_mass_produced = n.living_root_hairs_struct_mass - initial_living_root_hairs_struct_mass
        # If the difference of root hair mass with the initial mass is negative (it can happen when root hairs die),
        # we set the production to zero:
        if n.root_hairs_struct_mass_produced < 0.:
            n.root_hairs_struct_mass_produced = 0.
            
        # We add the cost of producing the new living root hairs (if any) to the hexose consumption by growth:
        n.hexose_consumption_by_growth += n.root_hairs_struct_mass_produced \
                                          * param.struct_mass_C_content / param.yield_growth / 6.
        n.resp_growth += n.root_hairs_struct_mass_produced \
                         * param.struct_mass_C_content * (1 - param.yield_growth) * 6.

    return

# Formation of root nodules:
#---------------------------
def nodule_formation(g, mother_element,
                     time_step_in_seconds=1. * 60. * 60. * 24.,
                     soil_temperature_in_Celsius=20,
                     random=True):
    """
    This function simulates the formation of one nodule on a root mother element. The nodule is considered as a special
    lateral root segment that has no apex connected to it.
    """

    # We add a lateral root element called "nodule" on the mother element:
    nodule = ADDING_A_CHILD(mother_element, edge_type='+', label='Segment', type='Root_nodule',
                            root_order=mother_element.root_order+1,
                            angle_down=90, angle_roll=0, length=0, radius=0,
                            identical_properties=False, nil_properties=True)
    nodule.type = "Root_nodule"
    # nodule.length=mother_element.radius
    # nodule.radius=mother_element.radius/10.
    nodule.length = mother_element.radius
    nodule.radius = mother_element.radius
    nodule.original_radius = nodule.radius
    dict = surfaces_and_volumes(g, element=nodule, radius=nodule.radius, length=nodule.length)
    nodule.external_surface = dict['external_surface']
    nodule.volume = dict['volume']
    nodule.phloem_surface = dict['phloem_surface']
    nodule.symplasm_surface = dict['symplasm_surface']
    nodule.struct_mass = nodule.volume * param.root_tissue_density * param.struct_mass_C_content

    # print("Nodule", nodule.index(), "has been formed!")

    return nodule

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "SUCROSE SUPPLY FROM THE SHOOTS"
########################################################################################################################
########################################################################################################################

# Calculation of the total amount of sucrose and structural struct_mass in the root system:
# -----------------------------------------------------------------------------------------
def total_root_sucrose_and_living_struct_mass(g):
    """
    This function computes the total amount of sucrose of the root system (in mol of sucrose),
    and the total dry structural mass of the root system (in g of dry structural mass).
    :param g: the investigated MTG
    :return: total_sucrose_root(mol of sucrose), total_struct_mass (g of dry structural mass)
    """

    # We initialize the values to 0:
    total_sucrose_root = 0.
    total_living_struct_mass = 0.

    # We cover all the vertices in the MTG, whether they are dead or not:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # We increment the total amount of sucrose in the root system, including dead root elements (if any):
        total_sucrose_root += n.C_sucrose_root * (n.struct_mass + n.living_root_hairs_struct_mass) \
                              - n.Deficit_sucrose_root
        # We only select the elements that have a positive struct_mass and are not dead
        # (if they have just died, we still include them in the balance):
        if n.struct_mass > 0. and n.type != "Dead" and n.type != "Just_dead":
            # We calculate the total living struct_mass by summing all the local struct_masses:
            total_living_struct_mass += n.struct_mass + n.living_root_hairs_struct_mass
    # We return a list of two numeric values:
    return total_sucrose_root, total_living_struct_mass


# Calculating the net input of sucrose by the aerial parts into the root system:
# ------------------------------------------------------------------------------
def shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=1e-9, time_step_in_seconds=1. * 60. * 60. * 24.,
                                       printing_warnings=False):
    """
    This function calculates the new root sucrose concentration (mol of sucrose per gram of dry root structural mass)
    AFTER the supply of sucrose from the shoot.
    """

    # The input of sucrose over this time step is calculated
    # from the sucrose transport rate provided as input of the function:
    sucrose_input = sucrose_input_rate * time_step_in_seconds

    # We calculate the remaining amount of sucrose in the root system,
    # based on the current sucrose concentration and structural mass of each root element:
    total_sucrose_root, total_living_struct_mass = total_root_sucrose_and_living_struct_mass(g)
    # Note: The total sucrose is the total amount of sucrose present in the root system, including local deficits
    # but excluding the possible global deficit of the whole root system, which is considered below.
    # The total struct_mass corresponds to the structural mass of the living roots & hairs
    # (including those that have just died and can still release sucrose!).

    # We use a global variable recorded outside this function that corresponds to the possible deficit of sucrose
    # (in moles of sucrose) of the whole root system calculated at the previous time_step:
    global_sucrose_deficit = g.property('global_sucrose_deficit')[g.root]
    # Note that this value is stored in the base node of the root system.
    if global_sucrose_deficit > 0.:
        print("!!! Before homogenizing sucrose concentration, the global deficit in sucrose was", global_sucrose_deficit)
    # The new average sucrose concentration in the root system is calculated as:
    C_sucrose_root_after_supply = (total_sucrose_root + sucrose_input - global_sucrose_deficit) \
                                  / total_living_struct_mass
    # This new concentration includes the amount of sucrose from element that have just died,
    # but excludes the mass of these dead elements!

    if C_sucrose_root_after_supply >= 0.:
        new_C_sucrose_root = C_sucrose_root_after_supply
        # We reset the global variable global_sucrose_deficit:
        g.property('global_sucrose_deficit')[g.root] = 0.
    else:
        new_sucrose_deficit = - C_sucrose_root_after_supply * total_living_struct_mass
        # We record the general deficit in sucrose:
        g.property('global_sucrose_deficit')[g.root] = new_sucrose_deficit
        print("!!! After homogenizing sucrose concentration, the deficit in sucrose is",
              new_sucrose_deficit)
        # We defined the new concentration of sucrose as 0:
        new_C_sucrose_root = 0.

    # We go through the MTG to modify the sugars concentrations:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)
        # If the element has not emerged yet, it doesn't contain any sucrose yet;
        # if has died, it should not contain any sucrose anymore:
        if n.length <= 0. or n.type == "Dead" or n.type=="Just_dead":
            n.C_sucrose_root = 0.
        else:
            # The local sucrose concentration in the root is calculated from the new sucrose concentration calculated above:
            n.C_sucrose_root = new_C_sucrose_root
        # AND BECAUSE THE LOCAL DEFICITS OF SUCROSE HAVE BEEN ALREADY INCLUDED IN THE TOTAL SUCROSE CALCULATION,
        # WE RESET ALL LOCAL DEFICITS TO 0:
        n.Deficit_sucrose_root = 0.

    return g

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "EXCHANGE BETWEEN SUCROSE AND HEXOSE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def exchange_with_phloem(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                         soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "exchange_with_phloem" simulates the process of sucrose unloading from phloem over time
    (in seconds) and its immediate conversion into hexose, for a given root element with an external surface (m2).
    It also simulates the process of sucrose loading (if any) that works in the other way.
    It returns the variable hexose_production_from_phloem (in mol of hexose) and sucrose_loading_in_phloem (in mol of sucrose),
    considering that 2 mol of hexose are produced for 1 mol of sucrose.
    The unloading of sucrose is represented as an active process with a substrate-limited relationship
    (Michaelis-Menten function), where unloading_coeff (in mol m-2 s-1) is the maximal amount of sucrose unloading
    and Km_unloading (in mol per gram of root structural struct_mass) represents the sucrose concentration
    for which the rate of hexose production is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the unloading coefficient and the production of hexose:
        n.unloading_coeff = 0.
        n.hexose_production_from_phloem = 0.
        n.sucrose_loading_in_phloem = 0.
        n.net_sucrose_unloading = 0.

        # We verify that the element does not correspond to a primordium that has not emerged:
        if n.length <= 0.:
            continue
        # If the element is dead, we consider that there is no possible exchange:
        if n.type == "Dead" or n.type=="Just_dead":
            continue

        # We calculate the current external surface of the element:
        n.phloem_surface = surfaces_and_volumes(g, n, n.radius, n.length)["phloem_surface"]

        # We assume that phloem's permeability per m2 of surface is identical everywhere along the root:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        n.phloem_permeability = param.phloem_permeability

        # We deal with special cases:
        #---------------------------

        # We verify that the concentration of sucrose is not negative:
        if n.C_sucrose_root <= 0.:
            if printing_warnings:
                print("WARNING: No phloem unloading occured for node", n.index(),
                      "because root sucrose concentration was", n.C_sucrose_root,
                      "mol/g.")
            n.phloem_permeability = 0.
        if n.type == "Stopped" or n.type == "Just_stopped":
            # If the element has stopped its growth, we decrease its unloading coefficient:
            n.phloem_permeability = n.phloem_permeability / 50. #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # If the node corresponds to the base of the root system and if an adventious root can emerge:
        if n.type=="Base_of_the_root_system": #and n.lateral_root_emergence_possibility == "Possible":
            n.phloem_permeability = n.phloem_permeability * 10.

        # We correct unloading rate according to soil temperature:
        n.phloem_permeability = n.phloem_permeability * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.phloem_permeability_T_ref,
            A=param.phloem_permeability_A,
            B=param.phloem_permeability_B,
            C=param.phloem_permeability_C)

        n.hexose_production_from_phloem = 2. * n.phloem_permeability * (n.C_sucrose_root - n.C_hexose_root / 2.) \
                                          * n.phloem_surface * time_step_in_seconds
        # We make sure that hexose production can't become negative:
        if n.hexose_production_from_phloem < 0.:
            n.hexose_production_from_phloem = 0.

        # Loading:
        # --------

        # We correct the max loading rate according to the distance from the tip in the middle of the segment:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n.dist_to_tip < n.length:
            print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
        else:
            n.max_loading_rate = param.surfacic_loading_rate_reference * (
                    1. - 1. / (1. + (n.dist_to_tip-n.length/2.) / n.original_radius) ** param.gamma_loading)

        # We correct loading according to soil temperature:
        n.max_loading_rate = n.max_loading_rate * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.max_loading_rate_T_ref,
            A=param.max_loading_rate_A,
            B=param.max_loading_rate_B,
            C=param.max_loading_rate_C)

        # We verify that the concentration of hexose in root is not nil or negative:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No phloem loading occured for node", n.index(),
                      "because root hexose concentration was", n.C_hexose_root,
                      "mol/g.")
            n.max_loading_rate = 0.

        # If there is a demand for growth, we decrease the loading rate:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n.hexose_growth_demand > 0.:
            n.max_loading_rate  = 0.

        # We calculate the potential production of sucrose from hexose (in mol) according to the Michaelis-Menten function:
        n.sucrose_loading_in_phloem = 0.5 * n.max_loading_rate * n.phloem_surface * n.C_hexose_root \
                                      / (param.Km_loading + n.C_hexose_root) * time_step_in_seconds
        # We make sure that hexose production can't become negative:
        if n.sucrose_loading_in_phloem < 0.:
            n.sucrose_loading_in_phloem = 0.

        n.net_sucrose_unloading = n.hexose_production_from_phloem / 2. - n.sucrose_loading_in_phloem

    return g

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "EXCHANGE BETWEEN MOBILE HEXOSE AND RESERVE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def exchange_with_reserve(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                          soil_temperature_in_Celsius=20, printing_warnings=False):
    """

    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the unloading coefficient and the production of hexose:
        n.hexose_mobilization_from_reserve = 0.
        n.hexose_immobilization_as_reserve = 0.
        n.net_hexose_immobilization = 0.

        # We verify that the element does not correspond to a primordium that has not emerged:
        if n.length <= 0.:
            continue
        # We verify that the concentration of sucrose and hexose in root are not negative:
        if n.C_hexose_root < 0. or n.C_hexose_reserve < 0.:
            if printing_warnings:
                print("WARNING: No exchange with phloem occurred for node", n.index(),
                      "because root sucrose concentration was", n.C_sucrose_root,
                      "mol/g, root hexose concentration was", n.C_hexose_root,
                      "mol/g, and hexose reserve concentration was", n.C_hexose_reserve)
            continue

        # If the element was already dead at the beginning of the time step, we don't consider it
        # (and if it has just died, we set its max reserve concentration to 0, see below):
        if n.type == "Dead":
            continue

        # If the element corresponds to a root nodule, we don't consider any reserve pool:
        if n.type == "Root_nodule":
            continue

        # CALCULATION OF THE MAXIMAL CONCENTRATION OF HEXOSE IN THE RESERVE POOL:
        if n.type == "Just_dead":
            # If the element has just died, all reserve is emptied over this time step:
            n.hexose_mobilization_from_reserve = n.C_hexose_reserve * (n.struct_mass + n.living_root_hairs_struct_mass)
            n.C_hexose_reserve = 0.
            # And the immobilization rate remains 0.
            # And we move to the next element:
            continue

        # The maximal concentration in the reserve is defined:
        n.C_hexose_reserve_max = param.C_hexose_reserve_max

        # We correct max loading and unloading rates according to soil temperature:
        corrected_max_mobilization_rate = param.max_mobilization_rate \
                                          * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                     process_at_T_ref=1,
                                                                     T_ref=param.max_mobilization_rate_T_ref,
                                                                     A=param.max_mobilization_rate_A,
                                                                     B=param.max_mobilization_rate_B,
                                                                     C=param.max_mobilization_rate_C)
        corrected_max_immobilization_rate = param.max_immobilization_rate \
                                            * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                       process_at_T_ref=1,
                                                                       T_ref=param.max_immobilization_rate_T_ref,
                                                                       A=param.max_immobilization_rate_A,
                                                                       B=param.max_immobilization_rate_B,
                                                                       C=param.max_immobilization_rate_C)

        # CALCULATIONS OF THEORETICAL MOBILIZATION / IMMOBILIZATION RATES:
        # We calculate the potential mobilization of hexose from reserve (in mol) according to the Michaelis-Menten function:
        n.hexose_mobilization_from_reserve = corrected_max_mobilization_rate * n.C_hexose_reserve \
                                             / (param.Km_mobilization + n.C_hexose_reserve) \
                                             * time_step_in_seconds \
                                             * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
        # We calculate the potential immobilization of hexose as reserve (in mol) according to the Michaelis-Menten function:
        if n.C_hexose_root < param.C_hexose_root_min_for_reserve:
            # If the concentration of mobile hexose is already too low, there is no immobilization:
            n.hexose_immobilization_as_reserve = 0.
        else:
            n.hexose_immobilization_as_reserve = corrected_max_immobilization_rate * n.C_hexose_root \
                                                 / (param.Km_immobilization + n.C_hexose_root) \
                                                 * time_step_in_seconds \
                                                 * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)

        # CARBON BALANCE AND ADJUSTMENTS:
        # We control the balance on the reserve by calculating the new theoretical concentration in the reserve pool:
        C_hexose_reserve_new = (n.C_hexose_reserve * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                                + n.hexose_immobilization_as_reserve - n.hexose_mobilization_from_reserve) \
                               / (n.struct_mass + n.living_root_hairs_struct_mass)

        # If the new concentration is lower than the minimal concentration:
        if C_hexose_reserve_new < param.C_hexose_reserve_min:
            # The amount of hexose that can be mobilized is lowered, while the amount of hexose immobilized is not modified:
            n.hexose_mobilization_from_reserve = n.hexose_mobilization_from_reserve \
                                                 - (param.C_hexose_reserve_min - C_hexose_reserve_new) \
                                                 * (n.struct_mass + n.living_root_hairs_struct_mass)
            # And we set the concentration to the minimal concentration:
            n.C_hexose_reserve = param.C_hexose_reserve_min
        # Otherwise, if the concentration in the reserve is higher than the maximal one:
        elif C_hexose_reserve_new > n.C_hexose_reserve_max:
            # The mobilized amount is not modified and the maximal amount of hexose that can be immobilized is reduced:
            n.hexose_immobilization_as_reserve = n.hexose_immobilization_as_reserve \
                                                 - (C_hexose_reserve_new - n.C_hexose_reserve_max) \
                                                 * (n.struct_mass + n.living_root_hairs_struct_mass)
            # And we limit the concentration to the maximal concentration:
            n.C_hexose_reserve = n.C_hexose_reserve_max
        # Else, the mobilization and immobilization processes are unchanged, and we record the final concentration as the expected one:
        else:
            n.C_hexose_reserve = C_hexose_reserve_new

        # We make sure that insignificant changes will not be kept in memory:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n.hexose_immobilization_as_reserve < 1e-20:
            n.hexose_immobilization_as_reserve=0.
        if n.hexose_mobilization_from_reserve < 1e-20:
            n.hexose_mobilization_from_reserve = 0.
        if n.C_hexose_reserve < 1e-20:
            n.C_hexose_reserve = 0.

        # In any case, the net balance is recorded:
        n.net_hexose_immobilization = n.hexose_immobilization_as_reserve - n.hexose_mobilization_from_reserve

    return

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "ROOT MAINTENANCE"
########################################################################################################################
########################################################################################################################

# Function calculating maintenance respiration:
#----------------------------------------------
def maintenance_respiration(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                            soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "maintenance" calculates the amount resp_maintenance (mol of CO2) corresponding to the consumption
    of a part of the local hexose pool to cover the costs of maintenance processes, i.e. any biological process in the
    root that is NOT linked to the actual growth of the root. The calculation is derived from the model of Thornley and
    Cannell (2000), who initially used this formalism to describe the residual maintenance costs that could not be
    accounted for by known processes. The local amount of CO2 respired for maintenance is calculated as a
    Michaelis-Menten function of the local concentration of hexose "C_hexose_root" (in mol of hexose per gram of root
    structural struct_mass. "g" represents the MTG describing the root system, "resp_maintenance__max" (mol of CO2 per gram
    of root structural struct_mass per second) is the maximal rate of maintenance respiration, and "Km_maintenance" (mol of
    hexose per gram of root structural struct_mass) represents the hexose concentration for which the rate of respiration is
    equal to half of its maximum. "struct_mass" is the root structural struct_mass (g) and time is expressed in seconds.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the maintenance respiration:
        n.resp_maintenance = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We consider that dead elements cannot respire (unless over the first time step following death,
        # i.e. when the type is "Just_dead"):
        if n.type == "Dead":
            continue
        # We also check whether the concentration of hexose in root is positive or not:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No maintenance occurred for node", n.index(),
                      "because root hexose concentration was", n.C_hexose_root, "mol/g.")
            continue

        # We correct the maximal respiration according to soil temperature:
        corrected_resp_maintenance_max = param.resp_maintenance_max \
                                         * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                    process_at_T_ref=1,
                                                                    T_ref=param.resp_maintenance_max_T_ref,
                                                                    A=param.resp_maintenance_max_A,
                                                                    B=param.resp_maintenance_max_B,
                                                                    C=param.resp_maintenance_max_C)

        # We calculate the number of moles of CO2 generated by maintenance respiration over the time_step:
        n.resp_maintenance = corrected_resp_maintenance_max \
                             * n.C_hexose_root / (param.Km_maintenance + n.C_hexose_root) \
                             * (n.struct_mass + n.living_root_hairs_struct_mass) * time_step_in_seconds

        if n.resp_maintenance < 0.:
            print("!!! ERROR: a negative maintenance respiration was calculated for the element", n.index())
            n.resp_maintenance = 0.

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "RHIZODEPOSITION"
########################################################################################################################
########################################################################################################################

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def root_hexose_exudation(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                          soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_hexose_exudation" computes the net amount (in mol of hexose) of hexose accumulated
    outside the root over time (in seconds), without considering any degradation process of hexose
    outside the root or hexose uptake by the root.
    Exudation corresponds to the difference between the efflux of hexose from the root
    to the soil by a passive diffusion. The efflux by diffusion is calculated from the product of the root external
    surface (m2), the permeability coefficient (g m-2) and the gradient of hexose concentration (mol of hexose
    per gram of dry root structural struct_mass).
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the exudation of hexose in the element:
        n.hexose_exudation = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue

        # We also check whether the concentration of hexose in root is positive or not:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No hexose exudation occurred for node", n.index(),
                      "because root hexose concentration was", n.C_hexose_root, "mol/g.")
            continue
        # NOTE : We consider that even dead elements can continue exuding hexose while there is hexose inside the root!

        # We calculate the permeability coefficient P according to the distance of the element from the apex:
        # OPTION 1: Pmax is the same everywhere
        # n.permeability_coeff = param.Pmax_apex
        # OPTION 2 (Personeni et al. 2007):
        # n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip*100) ** gamma_exudation
        # OPTION 3 (Personeni et al. 2007 modified, with special cases considered):
        if n.dist_to_tip < n.length:
            print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
        else:
            n.permeability_coeff = param.Pmax_apex \
                                   / (1 + (n.dist_to_tip-n.length/2.) / n.original_radius) ** param.gamma_exudation
        # if n.lateral_root_emergence_possibility == "Possible":
        #     n.permeability_coeff = n.permeability_coeff * 10.

        # We calculate the total surface of exchange between symplasm and apoplasm in the root cortex + epidermis:
        n.symplasm_surface = surfaces_and_volumes(g, n, n.radius, n.length)["symplasm_surface"]

        # We correct the permeability coefficient according to soil temperature:
        corrected_permeability_coeff = n.permeability_coeff \
                                       * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                  process_at_T_ref=1,
                                                                  T_ref=param.permeability_coeff_T_ref,
                                                                  A=param.permeability_coeff_A,
                                                                  B=param.permeability_coeff_B,
                                                                  C=param.permeability_coeff_C)

        # Then exudation is calculated as an efflux by diffusion, even for dead root elements:
        n.hexose_exudation = (n.symplasm_surface + n.living_root_hairs_external_surface) \
                             * corrected_permeability_coeff \
                             * (n.C_hexose_root - n.C_hexose_soil) * time_step_in_seconds
        if n.hexose_exudation < 0.:
            if printing_warnings:
                print("WARNING: a negative exudation flux was calculated for the element", n.index(),
                      "; exudation flux has therefore been set up to zero!")
            n.hexose_exudation = 0.

        # NOTE : We consider that dead elements can also liberate hexose in the soil, until they are empty.

    return g

# Uptake of hexose from the soil by the root:
# -------------------------------------------
def root_hexose_uptake(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                       soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_hexose_uptake" computes the amount (in mol of hexose) of hexose taken up by roots from the soil.
    This influx of hexose is represented as an active process with a substrate-limited
    relationship (Michaelis-Menten function), where uptake_rate_max (in mol) is the maximal influx, and Km_uptake
    (in mol per gram of root structural struct_mass) represents the hexose concentration for which
    hexose_degradation is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the uptake of hexose in the element:
        n.hexose_uptake = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_hexose_soil <= 0.:
            if printing_warnings:
                print("WARNING: No uptake of hexose from the soil occurred for node", n.index(),
                      "because soil hexose concentration was", n.C_hexose_soil, "mol/g.")
            continue
        # We consider that dead elements cannot take up any hexose from the soil:
        if n.type == "Just_dead" or n.type == "Dead":
            continue

        # We correct the maximal uptake rate according to soil temperature:
        corrected_uptake_rate_max = param.uptake_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.uptake_rate_max_T_ref,
            A=param.uptake_rate_max_A,
            B=param.uptake_rate_max_B,
            C=param.uptake_rate_max_C)

        # We calculate the total surface of exchange between symplasm and apoplasm in the root cortex + epidermis:
        n.symplasm_surface = surfaces_and_volumes(g, n, n.radius, n.length)["symplasm_surface"]
        # The uptake of hexose by the root from the soil is calculated:
        n.hexose_uptake = (n.symplasm_surface + n.living_root_hairs_external_surface) \
                          * corrected_uptake_rate_max \
                          * n.C_hexose_soil / (param.Km_uptake + n.C_hexose_soil) * time_step_in_seconds

    return g

# Mucilage secretion:
# ------------------
def root_mucilage_secretion(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                       soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_mucilage_secretion" computes the amount (in mol of hexose) of mucilage secreted by roots and released
    into the soil.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the secretion of mucilage:
        n.mucilage_secretion = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We also check whether the concentration of hexose in root is positive or not:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No mucilage secretion occurred for node", n.index(),
                      "because root hexose concentration was", n.C_hexose_soil, "mol/g.")
            continue
        # We consider that dead elements cannot secrete any mucilage:
        if n.type == "Just_dead" or n.type == "Dead":
            continue

        # We modified the maximal secretion rate according to the distance to the apex, similarly to
        # what has been done for hexose exudation:
        if n.dist_to_tip < n.length:
            print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
        else:
            n.secretion_rate_max = param.secretion_rate_max \
                                   / (1 + (n.dist_to_tip-n.length/2.)/ n.original_radius) ** param.gamma_secretion

        # We correct the maximal secretion rate according to soil temperature
        # (This corresponds to a bell-shape where the maximum is obtained at 27 degree Celsius,
        # as suggested by Morré et al. (1967) for maize mucilage secretion):
        corrected_secretion_rate_max = n.secretion_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.secretion_rate_max_T_ref,
            A=param.secretion_rate_max_A,
            B=param.secretion_rate_max_B,
            C=param.secretion_rate_max_C)

        # We calculate the total surface of exchange between the root and the soil, including living root hairs:
        n.external_surface = surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
        # The secretion of mucilage by the root is then calculated:
        n.mucilage_secretion \
            = (n.external_surface + n.living_root_hairs_external_surface) * corrected_secretion_rate_max \
              * n.C_hexose_root / (param.Km_secretion + n.C_hexose_root) * time_step_in_seconds

    return g

# Release of root cells:
# ----------------------
def root_cells_release(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                       soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_cells_release" computes the amount (in equivalent mol of hexose) of epidermal root cells released
    into the soil.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the release of root cells:
        n.cells_release = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # # We also check whether the concentration of hexose in root is positive or not:
        # if n.C_hexose_root <= 0.:
        #     if printing_warnings:
        #         print("WARNING: No mucilage secretion occurred for node", n.index(),
        #               "because root hexose concentration was", n.C_hexose_soil, "mol/g.")
        #     continue
        # We consider that dead elements cannot release any cells anymore:
        if n.type == "Just_dead" or n.type == "Dead":
            continue

        # We modify the maximal surfacic release rate according to the mean distance to the tip (in the middle of the
        # root element), assuming that the release decreases linearily with the distance to the tip, until reaching 0
        # when the this distance becomes higher than the growing zone length:
        if n.dist_to_tip < param.growing_zone_factor*n.radius:
            average_distance = n.dist_to_tip - n.length/2.
            reduction = (param.growing_zone_factor*n.radius - average_distance) / (param.growing_zone_factor*n.radius)
            n.cells_surfacic_release =  param.surfacic_cells_release_rate * reduction
        # In the special case where the end of the growing zone is located somewhere in the middle of the root element:
        elif n.dist_to_tip - n.length < param.growing_zone_factor*n.radius:
            average_distance = (n.dist_to_tip - n.length) + (param.growing_zone_factor*n.radius
                                                             - (n.dist_to_tip - n.length))/2.
            reduction = (param.growing_zone_factor*n.radius - average_distance) / (param.growing_zone_factor*n.radius)
            n.cells_surfacic_release = param.surfacic_cells_release_rate * reduction
        else:
            n.cells_surfacic_release = 0.

        # We correct the release rate according to soil temperature:
        corrected_cells_surfacic_release = n.cells_surfacic_release * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.surfacic_cells_release_rate_T_ref,
            A=param.surfacic_cells_release_rate_A,
            B=param.surfacic_cells_release_rate_B,
            C=param.surfacic_cells_release_rate_C)

        # We also regulate the surface release rate according to the potential accumulation of cells around the root:
        # the rate is maximal when no cells are around, and linearily decreases with the concentration of cells in the
        # soil, until reaching 0 when the concentration is equal or higher than the maximal concentration in the soil:
        corrected_cells_surfacic_release = corrected_cells_surfacic_release \
                                           * (param.C_cells_soil_max - n.C_cells_soil) / param.C_cells_soil_max
        if corrected_cells_surfacic_release < 0.:
            corrected_cells_surfacic_release = 0.

        # We calculate the total surface of exchange between the root and the soil:
        n.external_surface = surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
        # The release of cells by the root is then calculated according to this surface:
        n.cells_release \
            = n.external_surface * corrected_cells_surfacic_release * time_step_in_seconds

    return g

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "SOIL TRANSFORMATION"
########################################################################################################################
########################################################################################################################

# Degradation of hexose in the soil (microbial consumption):
# ----------------------------------------------------------
def soil_hexose_degradation(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                            soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "hexose_degradation" computes the decrease of the concentration of hexose outside the root (in mol of
    hexose per gram of root structural mass) over time (in seconds). It mimics the uptake of hexose by rhizosphere
    microorganisms, and is therefore described using a substrate-limited function (Michaelis-Menten). g represents the
    MTG describing the root system, degradation_rate_max is the maximal degradation of hexose (mol m-2), and Km_degradation
    (mol per gram of root structural mass) represents the hexose concentration for which the rate of hexose_degradation
    is equal to half of its maximum. The surface of the symplasm rather than the external surface of the root element
    is taken into account here, similarly to what is done for exudation or re-uptake of hexose by the root.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the degradation of hexose:
        n.hexose_degradation = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0.:
            continue

        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_hexose_soil <= 0.:
            if printing_warnings:
                print("WARNING: No degradation in the soil occurred for node", n.index(),
                      "because soil hexose concentration was", n.C_hexose_soil, "mol/g.")
            continue

        # We correct the maximal degradation rate according to soil temperature:
        corrected_hexose_degradation_rate_max = param.hexose_degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.hexose_degradation_rate_max_T_ref,
            A=param.hexose_degradation_rate_max_A,
            B=param.hexose_degradation_rate_max_B,
            C=param.hexose_degradation_rate_max_C)

        # We calculate the total surface of exchange between symplasm and apoplasm in the root cortex + epidermis:
        n.symplasm_surface = surfaces_and_volumes(g, n, n.radius, n.length)["symplasm_surface"]
        # hexose_degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.hexose_degradation = (n.symplasm_surface + n.living_root_hairs_external_surface) \
                               * corrected_hexose_degradation_rate_max * n.C_hexose_soil \
                               / (param.Km_hexose_degradation + n.C_hexose_soil) * time_step_in_seconds

    return g

# Degradation of mucilage in the soil (microbial consumption):
# ------------------------------------------------------------
def soil_mucilage_degradation(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                            soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "soil_mucilage_degradation" computes the decrease of the concentration of mucilage outside the root
    (in mol of equivalent-hexose per gram of root structural mass) over time (in seconds). g represents the
    MTG describing the root system, mucilage_degradation_rate_max is the maximal degradation of mucilage (mol m-2 s-1),
    and Km_mucilage_degradation (mol per gram of root structural mass) represents the mucilage concentration for which
    the rate of mucilage_degradation is equal to half of its maximum. Only the external surface of the root element
    is taken into account here, similarly to what is done for mucilage secretion.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the degradation of hexose:
        n.mucilage_degradation = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0.:
            continue

        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_mucilage_soil <= 0.:
            if printing_warnings:
                print("WARNING: No degradation in the soil occurred for node", n.index(),
                      "because soil hexose concentration was", n.C_mucilage_soil, "mol/g.")
            continue

        # We correct the maximal degradation rate according to soil temperature:
        corrected_mucilage_degradation_rate_max = param.mucilage_degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.mucilage_degradation_rate_max_T_ref,
            A=param.mucilage_degradation_rate_max_A,
            B=param.mucilage_degradation_rate_max_B,
            C=param.mucilage_degradation_rate_max_C)

        # We calculate the total external surface of the root element, i.e. the surface of the epidermis,
        # including living root hairs:
        n.external_surface = surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
        # Mucilage degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.mucilage_degradation = (n.external_surface + n.living_root_hairs_external_surface) \
                                 * corrected_mucilage_degradation_rate_max \
                                 * n.C_mucilage_soil / (param.Km_mucilage_degradation + n.C_mucilage_soil) \
                                 * time_step_in_seconds

    return g

# Degradation of root cells released in the soil (microbial consumption):
# -----------------------------------------------------------------------
def soil_cells_degradation(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                            soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "soil_cells_degradation" computes the decrease of the concentration of cells outside the root
    (in mol of equivalent-hexose per gram of root structural mass) over time (in seconds). g represents the
    MTG describing the root system, cells_degradation_rate_max is the maximal degradation of cells (mol m-2 s-1),
    and Km_cells_degradation (mol per gram of root structural mass) represents the cells concentration for which
    the rate of cells degradation is equal to half of its maximum. Only the external surface of the root element
    is taken into account here, similarly to what is done for root cells release.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the degradation of hexose:
        n.cells_degradation = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0.:
            continue

        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_cells_soil <= 0.:
            if printing_warnings:
                print("WARNING: No degradation in the soil occurred for node", n.index(),
                      "because soil hexose concentration was", n.C_cells_soil, "mol/g.")
            continue

        # We correct the maximal degradation rate according to soil temperature:
        corrected_cells_degradation_rate_max = param.cells_degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.cells_degradation_rate_max_T_ref,
            A=param.cells_degradation_rate_max_A,
            B=param.cells_degradation_rate_max_B,
            C=param.cells_degradation_rate_max_C)

        # We calculate the total external surface of the root element, i.e. the surface of the epidermis:
        n.external_surface = surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
        # Cells degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.cells_degradation = n.external_surface * corrected_cells_degradation_rate_max * n.C_cells_soil \
                                 / (param.Km_cells_degradation + n.C_cells_soil) * time_step_in_seconds

    return g

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MAIN PROGRAM:
########################################################################################################################
########################################################################################################################

# Performing a complete C balance on each root element:
#-------------------------------------------------------
def balance(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We exclude root elements that have not emerged yet:
        if n.length <= 0.:
            continue

        # BALANCE ON HEXOSE AT THE SOIL/ROOT INTERFACE:
        # We calculate the new concentration of hexose in the soil according to hexose degradation, exudation and uptake:
        n.C_hexose_soil = (n.C_hexose_soil * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                           - n.Deficit_hexose_soil
                           - n.hexose_degradation
                           + n.hexose_exudation
                           - n.hexose_uptake) \
                          / (n.struct_mass + n.living_root_hairs_struct_mass)
        # We reset the deficit to 0:
        n.Deficit_hexose_soil = 0.
        if n.C_hexose_soil < 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit of soil hexose for element", n.index(),
                      "that corresponds to", n.Deficit_hexose_soil,
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_hexose_soil = -n.C_hexose_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
            # And we set the concentration to 0:
            n.C_hexose_soil = 0.

        # BALANCE ON MUCILAGE AT THE SOIL/ROOT INTERFACE:
        # We calculate the new concentration of mucilage (mol of equivalent-hexose per gram of dry structural mass)
        # in the soil:
        n.C_mucilage_soil = (n.C_mucilage_soil * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                             - n.Deficit_mucilage_soil
                             + n.mucilage_secretion
                             - n.mucilage_degradation) \
                            / (n.struct_mass + n.living_root_hairs_struct_mass)
        # We reset the deficit to 0:
        n.Deficit_mucilage_soil = 0.
        if n.C_mucilage_soil < 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit of soil mucilage for element", n.index(),
                      "that corresponds to", n.C_mucilage_soil * (n.struct_mass + n.living_root_hairs_struct_mass),
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_mucilage_soil = -n.C_mucilage_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
            # And we set the concentration to 0:
            n.C_mucilage_soil = 0.

        # BALANCE ON RELEASED CELLS AT THE SOIL/ROOT INTERFACE:
        # We calculate the new concentration of cells (mol of equivalent-hexose per gram of dry structural mass)
        # at the root/soil interface:
        n.C_cells_soil = (n.C_cells_soil * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                          - n.Deficit_cells_soil
                          + n.cells_release
                          - n.cells_degradation) \
                         / (n.struct_mass + n.living_root_hairs_struct_mass)
        # We reset the deficit to 0:
        n.Deficit_cells_soil = 0.
        if n.C_cells_soil < 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit of released cells for element", n.index(),
                      "that corresponds to", n.C_cells_soil * (n.struct_mass + n.living_root_hairs_struct_mass),
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            # We define a positive deficit (mol of C) based on the negative concentration:
            n.Deficit_cells_soil = -n.C_cells_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
            # And we set the concentration to 0:
            n.C_cells_soil = 0.

        # BALANCE ON HEXOSE IN THE ROOT CYTOPLASM:
        # We calculate the new concentration of hexose in the root cytoplasm:
        n.C_hexose_root = (n.C_hexose_root * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                           - n.Deficit_hexose_root
                           - n.hexose_exudation + n.hexose_uptake
                           - n.mucilage_secretion
                           - n.cells_release
                           - n.resp_maintenance / 6. - n.hexose_consumption_by_growth
                           + n.hexose_production_from_phloem - 2. * n.sucrose_loading_in_phloem
                           + n.hexose_mobilization_from_reserve - n.hexose_immobilization_as_reserve) \
                          / (n.struct_mass + n.living_root_hairs_struct_mass)
        # We reset the deficit to 0:
        n.Deficit_hexose_root = 0.
        if n.C_hexose_root < 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit of root hexose for element", n.index(),
                      "that corresponds to", n.Deficit_hexose_root,
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_hexose_root = - n.C_hexose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
            # And we set the concentration to 0:
            n.C_hexose_root = 0.

        # BALANCE ON SUCROSE IN THE ROOT:
        # We calculate the new concentration of sucrose in the root according to exchange with phloem:
        # (NOTE: The deficit in sucrose is not included in this balance, since it is included in the function shoot_supply before)
        n.C_sucrose_root = (n.C_sucrose_root * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                            + n.sucrose_loading_in_phloem
                            - n.hexose_production_from_phloem / 2.) \
                           / (n.struct_mass + n.living_root_hairs_struct_mass)
        if n.C_sucrose_root < 0:
            # We define a positive deficit (mol of sucrose) based on the negative concentration:
            n.Deficit_sucrose_root = -n.C_sucrose_root * n.struct_mass
            # And we set the concentration to 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit in root sucrose for element", n.index(),
                      "that corresponds to", n.Deficit_sucrose_root,
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            n.C_sucrose_root = 0.
        # PLEASE NOTE: The global (if any) deficit in sucrose is only used by the function "shoot_sucrose_and_spreading"
        # when defining the new homogeneous concentration of sucrose within the root system,
        # and when performing a true carbon balance of the root system in "summing".

        # CREATING NEW PROPERTIES:
        # We calculate the net exudation of hexose (in mol of hexose):
        n.net_hexose_exudation = n.hexose_exudation - n.hexose_uptake
        # We calculate the total rhizodeposition (in mol of hexose):
        n.total_rhizodeposition = n.net_hexose_exudation + n.mucilage_secretion + n.cells_release
        # We calculate the total biomass of each element, including the structural mass and all sugars:
        n.biomass = (n.struct_mass + n.living_root_hairs_struct_mass) \
                    + (n.C_hexose_root * 6 * 12.01 + n.C_hexose_reserve * 6 * 12.01) \
                    * (n.struct_mass + n.living_root_hairs_struct_mass) \
                    + (n.C_sucrose_root * 12 * 12.01) * n.struct_mass
        # We calculate a net rate of exudation, in gram of C per gram of dry structural mass per day:
        n.net_hexose_exudation_rate_per_day_per_gram \
            = (n.net_hexose_exudation / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 \
              / (n.struct_mass  + n.living_root_hairs_struct_mass)
        # We calculate a net rate of exudation, in gram of C per cm of root per day:
        n.net_hexose_exudation_rate_per_day_per_cm \
            = (n.net_hexose_exudation / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100
        # We calculate a rate of mucilage secretion, in gram of C per cm of root per day:
        n.mucilage_secretion_rate_per_day_per_cm \
            = (n.mucilage_secretion / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100
        # We calculate a total rate of rhizodeposition, in gram of C per cm of root per day:
        n.total_rhizodeposition_rate_per_day_per_cm \
            = (n.total_rhizodeposition / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100

    # OPTIONAL: If the element corresponds to the apex of the primary root:
        if n.radius == param.D_ini / 2. and n.label == "Apex":
            # Then the function will give its specific concentration of mobile hexose:
            tip_C_hexose_root = n.C_hexose_root
    return tip_C_hexose_root

# Calculation of total amounts and dimensions of the root system:
# ---------------------------------------------------------------
def summing(g, printing_total_length=True, printing_total_struct_mass=True, printing_all=False):
    """
    This function computes a number of general properties summed over the whole MTG.
    :param g: the investigated MTG
    :param printing_total_length: a Boolean defining whether total_length should be printed on the screen or not
    :param printing_total_struct_mass: a Boolean defining whether total_struct_mass should be printed on the screen or not
    :param printing_all: a Boolean defining whether all properties should be printed on the screen or not
    :return: a dictionary containing the numerical value of each property integrated over the whole MTG
    """

    # We initialize the values to 0:
    total_length = 0.
    total_dead_length = 0.
    total_struct_mass = 0.
    total_root_hairs_mass = 0.
    total_dead_struct_mass = 0.
    total_surface = 0.
    total_dead_surface = 0.
    total_living_root_hairs_surface = 0.
    total_sucrose_root = 0.
    total_hexose_root = 0.
    total_hexose_reserve = 0.
    total_hexose_soil = 0.
    total_mucilage_soil = 0.
    total_cells_soil = 0.
    total_sucrose_root_deficit = 0.
    total_hexose_root_deficit = 0.
    total_hexose_soil_deficit = 0.
    total_mucilage_soil_deficit = 0.
    total_cells_soil_deficit = 0.
    total_respiration = 0.
    total_respiration_root_growth = 0.
    total_respiration_root_maintenance = 0.
    total_struct_mass_produced = 0.
    total_hexose_production_from_phloem = 0.
    total_sucrose_loading_in_phloem = 0.
    total_hexose_immobilization_as_reserve = 0.
    total_hexose_mobilization_from_reserve = 0.
    total_hexose_exudation = 0.
    total_hexose_uptake = 0.
    total_net_hexose_exudation = 0.
    total_mucilage_secretion = 0.
    total_cells_release = 0.
    total_rhizodeposition = 0.
    total_hexose_degradation = 0.
    total_mucilage_degradation = 0.
    total_cells_degradation = 0.

    C_in_the_root_soil_system = 0.
    C_degraded = 0.
    C_respired_by_roots = 0.

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # If the current element has no length, there is no need to include it in the balance:
        if n.length <= 0.:
            continue

        # Calculations of total mass, length and surface:
        if n.type == "Dead" or n.type == "Just_dead":
            # Note: we only include dead root hairs in the necromass balance when the root element itself has died!
            total_dead_struct_mass += n.struct_mass + n.root_hairs_struct_mass
            total_dead_length += n.length
            total_dead_surface += surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
        else:
            total_length += n.length
            total_struct_mass += n.struct_mass + n.root_hairs_struct_mass
            total_surface += surfaces_and_volumes(g, n, n.radius, n.length)["external_surface"]
            # Note that living root hairs are NOT included in the total surface of living roots!
            # Additionaly, we calculate the total surface of living root hairs:
            total_living_root_hairs_surface +=  n.living_root_hairs_external_surface

        # Additionaly, we calculate the total mass of root hairs (both alive and dead) in the root system
        # (but note that root hairs have already been included in the total structural mass calculated above):
        total_root_hairs_mass += n.root_hairs_struct_mass

        total_sucrose_root += n.C_sucrose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_hexose_root += n.C_hexose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_hexose_reserve += n.C_hexose_reserve * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_hexose_soil += n.C_hexose_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_mucilage_soil += n.C_mucilage_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_cells_soil += n.C_cells_soil * (n.struct_mass + n.living_root_hairs_struct_mass)

        total_sucrose_root_deficit += n.Deficit_sucrose_root
        total_hexose_root_deficit += n.Deficit_hexose_root
        total_hexose_soil_deficit += n.Deficit_hexose_soil
        total_mucilage_soil_deficit += n.Deficit_mucilage_soil
        total_cells_soil_deficit += n.Deficit_cells_soil

        total_respiration += n.resp_maintenance + n.resp_growth
        total_respiration_root_growth += n.resp_growth
        total_respiration_root_maintenance += n.resp_maintenance
        total_struct_mass_produced += n.struct_mass_produced + n.root_hairs_struct_mass_produced
        total_hexose_production_from_phloem += n.hexose_production_from_phloem
        total_sucrose_loading_in_phloem += n.sucrose_loading_in_phloem
        total_hexose_immobilization_as_reserve += n.hexose_immobilization_as_reserve
        total_hexose_mobilization_from_reserve += n.hexose_mobilization_from_reserve
        total_hexose_exudation += n.hexose_exudation
        total_hexose_uptake += n.hexose_uptake
        total_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake)
        total_mucilage_secretion += n.mucilage_secretion
        total_cells_release += n.cells_release
        total_rhizodeposition += n.total_rhizodeposition
        total_hexose_degradation += n.hexose_degradation
        total_mucilage_degradation += n.mucilage_degradation
        total_cells_degradation += n.cells_degradation

    # We add to the sum of local deficits in sucrose the possible global deficit in sucrose used in shoot_supply function:
    total_sucrose_root_deficit += g.property('global_sucrose_deficit')[g.root]

    # CARBON BALANCE:
    # --------------
    # We check that the carbon balance is correct (in moles of C):
    C_in_the_root_soil_system = (total_struct_mass + total_dead_struct_mass) * param.struct_mass_C_content \
                                + (total_sucrose_root - total_sucrose_root_deficit) * 12. \
                                + (total_hexose_root - total_hexose_root_deficit) * 6. \
                                + (total_hexose_reserve) * 6. \
                                + (total_hexose_soil - total_hexose_soil_deficit) * 6. \
                                + (total_mucilage_soil - total_mucilage_soil_deficit) * 6. \
                                + (total_cells_soil - total_cells_soil_deficit) * 6.
    C_degraded = (total_hexose_degradation + total_mucilage_degradation + total_cells_degradation) * 6.
    C_respired_by_roots = total_respiration

    if printing_total_length:
        print("   New state of the root system:")
        print("      The current total root length is",
              "{:.1f}".format(Decimal(total_length * 100)), "cm.")
    if printing_total_struct_mass:
        print("      The current total root structural mass (including the mass of associated root hairs) is",
              "{:.2E}".format(Decimal(total_struct_mass)), "g, i.e.",
              "{:.2E}".format(Decimal(total_struct_mass * param.struct_mass_C_content)), "mol of C.")
    if printing_all:
        print("      The current total dead root structural mass (including the mass of associated root hairs) is",
              "{:.2E}".format(Decimal(total_dead_struct_mass)), "g, i.e.",
              "{:.2E}".format(Decimal(total_dead_struct_mass * param.struct_mass_C_content)), "mol of C.")
        print("      The current amount of sucrose in the roots (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_sucrose_root - total_sucrose_root_deficit)), "mol of sucrose, i.e.",
              "{:.2E}".format(Decimal((total_sucrose_root - total_sucrose_root_deficit) * 12)), "mol of C.")
        print("      The current amount of mobile hexose in the roots (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_hexose_root - total_hexose_root_deficit)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal((total_hexose_root - total_hexose_root_deficit) * 6)), "mol of C.")
        print("      The current amount of hexose stored as reserve in the roots is",
              "{:.2E}".format(Decimal(total_hexose_reserve)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_hexose_reserve * 6)), "mol of C.")
        print("      The current amount of hexose in the soil (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_hexose_soil - total_hexose_soil_deficit)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal((total_hexose_soil - total_hexose_soil_deficit) * 6)), "mol of C.")
        print("      The total amount of CO2 respired by roots over this time step was",
              "{:.2E}".format(Decimal(total_respiration)), "mol of C, including",
              "{:.2E}".format(Decimal(total_respiration_root_growth)), "mol of C for growth and",
              "{:.2E}".format(Decimal(total_respiration_root_maintenance)), "mol of C for maintenance.")
        print("      The total net amount of hexose exuded by roots over this time step was",
              "{:.2E}".format(Decimal(total_net_hexose_exudation)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_net_hexose_exudation * 6)), "mol of C.")
        print("      The total net rhizodeposition over this time step was",
              "{:.2E}".format(Decimal(total_rhizodeposition)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_rhizodeposition * 6)), "mol of C.")
        print("      The total amount of hexose degraded in the soil over this time step was",
              "{:.2E}".format(Decimal(total_hexose_degradation)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_hexose_degradation * 6)), "mol of C.")

    dictionary = {"total_living_root_length": total_length,
                  "total_dead_root_length": total_dead_length,
                  "total_living_root_struct_mass": total_struct_mass,
                  "total_dead_root_struct_mass": total_dead_struct_mass,
                  "total_root_hairs_mass": total_root_hairs_mass,
                  "total_living_root_surface": total_surface,
                  "total_dead_root_surface": total_dead_surface,
                  "total_living_root_hairs_surface": total_living_root_hairs_surface,
                  "total_sucrose_root": total_sucrose_root,
                  "total_hexose_root": total_hexose_root,
                  "total_hexose_reserve": total_hexose_reserve,
                  "total_hexose_soil": total_hexose_soil,
                  "total_mucilage_soil": total_mucilage_soil,
                  "total_cells_soil": total_cells_soil,
                  "total_sucrose_root_deficit": total_sucrose_root_deficit,
                  "total_hexose_root_deficit": total_hexose_root_deficit,
                  "total_hexose_soil_deficit": total_hexose_soil_deficit,
                  "total_mucilage_soil_deficit": total_mucilage_soil_deficit,
                  "total_cells_soil_deficit": total_cells_soil_deficit,
                  "total_respiration": total_respiration,
                  "total_respiration_root_growth": total_respiration_root_growth,
                  "total_respiration_root_maintenance": total_respiration_root_maintenance,
                  "total_structural_mass_production": total_struct_mass_produced,
                  "total_hexose_production_from_phloem": total_hexose_production_from_phloem,
                  "total_sucrose_loading_in_phloem": total_sucrose_loading_in_phloem,
                  "total_hexose_immobilization_as_reserve": total_hexose_immobilization_as_reserve,
                  "total_hexose_mobilization_from_reserve": total_hexose_mobilization_from_reserve,
                  "total_hexose_exudation": total_hexose_exudation,
                  "total_hexose_uptake": total_hexose_uptake,
                  "total_mucilage_secretion": total_mucilage_secretion,
                  "total_total_rhizodeposition": total_rhizodeposition,
                  "total_cells_release": total_cells_release,
                  "total_hexose_degradation": total_hexose_degradation,
                  "total_mucilage_degradation": total_mucilage_degradation,
                  "total_cells_degradation": total_cells_degradation,
                  "total_net_hexose_exudation": total_net_hexose_exudation,
                  "total_rhizodeposition": total_rhizodeposition,
                  "C_in_the_root_soil_system": C_in_the_root_soil_system,
                  "C_degraded_in_the_soil": C_degraded,
                  "C_respired_by_roots": C_respired_by_roots
                  }

    return dictionary

# Control of anomalies in the MTG:
# --------------------------------
def control_of_anomalies(g):
    """
    The function contol_of_anomalies checks for the presence of elements with negative measurable properties (e.g. length, concentrations).
    """

    # CHECKING THAT UNEMERGED ROOT ELEMENTS DO NOT CONTAIN CARBON:
    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        if n.length <= 0.:
            if n.C_sucrose_root != 0.:
                print("")
                print("??? ERROR: for element", n.index(), " of length", n.length,
                      "m, the concentration of root sucrose is", n.C_sucrose_root)
            if n.C_hexose_root != 0.:
                print("")
                print("??? ERROR: for element", n.index(), " of length", n.length,
                      "m, the concentration of root hexose is", n.C_hexose_root)
            if n.C_hexose_soil != 0.:
                print("")
                print("??? ERROR: for element", n.index(), " of length", n.length,
                      "m, the concentration of soil hexose is", n.C_hexose_soil)

    return

# Initialization of the root system:
#-----------------------------------
def initiate_mtg(random=True,
                 initial_segment_length=1e-3,
                 initial_C_sucrose_root=1e-4,
                 initial_C_hexose_root=1e-4):

    # We create a new MTG called g:
    g = MTG()

    # Properties shared by the whole root system (stored in the first element at the base of the root system):
    # --------------------------------------------------------------------------------------------------------
    # We initiate the global variable that corresponds to a possible general deficit in sucrose of the whole root system:
    g.add_property('global_sucrose_deficit')
    g.property('global_sucrose_deficit')[g.root] = 0.

    # We first add one initial element:
    # ---------------------------------
    id_segment = g.add_component(g.root, label='Segment')
    segment = g.node(id_segment)

    # Characteristics:
    # -----------------
    segment.type = "Base_of_the_root_system"
    # By definition, we set the order of the primary root to 1:
    segment.root_order = 1

    # Authorizations and C requirements:
    # -----------------------------------
    segment.lateral_root_emergence_possibility = 'Impossible'
    segment.emergence_cost = 0.

    # Geometry and topology:
    # -----------------------

    base_radius = param.D_ini / 2.

    segment.angle_down = 0
    segment.angle_roll = 0
    segment.length = initial_segment_length
    segment.radius = base_radius
    segment.original_radius = base_radius
    segment.initial_length = initial_segment_length
    segment.initial_radius = base_radius

    segment.root_hair_radius = param.root_hair_radius
    segment.root_hair_length = 0.
    segment.actual_length_with_hairs = 0.
    segment.living_root_hairs_number = 0.
    segment.dead_root_hairs_number = 0.
    segment.total_root_hairs_number = 0.

    segment.actual_time_since_root_hairs_emergence_started = 0.
    segment.thermal_time_since_root_hairs_emergence_started = 0.
    segment.actual_time_since_root_hairs_emergence_stopped= 0.
    segment.thermal_time_since_root_hairs_emergence_stopped = 0.
    segment.all_root_hairs_formed = False
    segment.root_hairs_lifespan = param.root_hairs_lifespan
    segment.root_hairs_external_surface = 0.
    segment.root_hairs_volume = 0.
    segment.living_root_hairs_external_surface = 0.
    segment.root_hairs_struct_mass = 0.
    segment.root_hairs_struct_mass_produced = 0.
    segment.living_root_hairs_struct_mass = 0.

    surface_dictionary = surfaces_and_volumes(g, segment, segment.radius, segment.length)
    segment.external_surface = surface_dictionary["external_surface"]
    segment.volume = surface_dictionary["volume"]
    segment.phloem_surface = surface_dictionary["phloem_surface"]
    segment.symplasm_surface = surface_dictionary["symplasm_surface"]

    segment.dist_to_tip = segment.length
    segment.former_dist_to_tip = segment.length
    segment.dist_to_ramif = 0.
    segment.actual_elongation = segment.length
    segment.actual_elongation_rate = 0

    # Quantities and concentrations:
    # -------------------------------
    segment.struct_mass = segment.volume * param.root_tissue_density
    segment.initial_struct_mass = segment.struct_mass
    segment.initial_living_root_hairs_struct_mass = segment.living_root_hairs_struct_mass
    # We define the initial sugar concentrations:
    segment.C_sucrose_root = initial_C_sucrose_root
    segment.C_hexose_root = initial_C_hexose_root
    segment.C_hexose_reserve = 0.
    segment.C_hexose_soil = 0.
    segment.C_mucilage_soil = 0.
    segment.C_cells_soil = 0.
    segment.Deficit_sucrose_root = 0.
    segment.Deficit_hexose_root = 0.
    segment.Deficit_hexose_soil = 0.
    segment.Deficit_mucilage_soil = 0.
    segment.Deficit_cells_soil = 0.

    # Fluxes:
    # --------
    segment.resp_maintenance = 0.
    segment.resp_growth = 0.
    segment.hexose_growth_demand = 0.
    segment.hexose_possibly_required_for_elongation = 0.
    segment.hexose_consumption_by_growth = 0.
    segment.hexose_production_from_phloem = 0.
    segment.sucrose_loading_in_phloem = 0.
    segment.hexose_mobilization_from_reserve = 0.
    segment.hexose_immobilization_as_reserve = 0.
    segment.hexose_exudation = 0.
    segment.hexose_uptake = 0.
    segment.mucilage_secretion = 0.
    segment.cells_release = 0.
    segment.total_rhizodeposition = 0.
    segment.hexose_degradation = 0.
    segment.mucilage_degradation = 0.
    segment.cells_degradation = 0.
    segment.specific_net_exudation = 0.

    # Time indications:
    # ------------------
    segment.growth_duration = param.GDs * (2. * base_radius) ** 2 * param.main_roots_growth_extender #WATCH OUT!!! we artificially multiply growth duration for seminal and adventious roots!!!!!!!!!!!!!!!!!!!!!!
    segment.life_duration = param.LDs * (2. * base_radius) * param.root_tissue_density
    segment.actual_time_since_primordium_formation = 0.
    segment.actual_time_since_emergence = 0.
    segment.actual_time_since_growth_stopped = 0.
    segment.actual_time_since_death = 0.
    segment.thermal_time_since_primordium_formation = 0.
    segment.thermal_time_since_emergence = 0.
    segment.thermal_potential_time_since_emergence = 0.
    segment.thermal_time_since_growth_stopped = 0.
    segment.thermal_time_since_death = 0.

    # ADDING THE PRIMORDIA OF ALL POSSIBLE SEMINAL ROOTS:
    #----------------------------------------------------
    # If there is more than one seminal root (i.e. roots already formed in the seed):
    if param.n_seminal_roots > 1:

        # We read additional parameters that are stored in a CSV file, with one column containing the delay for each
        # emergence event, and the second column containing the number of seminal roots that have to emerge at each event:
        seminal_inputs_path = os.path.join("C:/Users/frees/rhizodep/src/rhizodep","seminal_roots_inputs.csv")
        seminal_inputs_file = pd.read_csv(seminal_inputs_path)

        # For each event of seminal roots emergence:
        for i in range(0, len(seminal_inputs_file.emergence_delay_in_thermal_time)):
            # For each seminal root that can emerge at this emergence event:
            for j in range(0,seminal_inputs_file.number_of_seminals_per_event[i]):

                # We make sure that the seminal roots will have different random insertion angles:
                np.random.seed(param.random_choice + i*j)

                # Then we form one supporting segment of length 0 + one primordium of seminal root.
                # We add one new segment without any length on the same axis as the base:
                segment = ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Segment',
                                         type='Support_for_seminal_root',
                                         root_order=1,
                                         angle_down=0,
                                         angle_roll=abs(np.random.normal(180, 180)),
                                         length=0.,
                                         radius=base_radius,
                                         identical_properties=False,
                                         nil_properties=True)

                # We define the radius of a seminal root according to the parameter Di:
                if random:
                    radius_seminal = abs(np.random.normal(param.D_ini / 2. * param.D_sem_to_D_ini_ratio,
                                         param.D_ini / 2. * param.D_sem_to_D_ini_ratio * param.CVDD))

                # And we add one new primordium of seminal root on the previously defined segment:
                apex_seminal = ADDING_A_CHILD(mother_element=segment, edge_type='+', label='Apex',
                                                   type='Seminal_root_before_emergence',
                                                   root_order=1,
                                                   angle_down=abs(np.random.normal(60, 10)),
                                                   angle_roll=5,
                                                   length=0.,
                                                   radius=radius_seminal,
                                                   identical_properties=False,
                                                   nil_properties=True)
                apex_seminal.original_radius = radius_seminal
                apex_seminal.initial_radius = radius_seminal
                apex_seminal.growth_duration = param.GDs * (2. * radius_seminal) ** 2 * param.main_roots_growth_extender
                # WATCH OUT!!! we artificially multiply growth duration for seminal and adventious roots!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                apex_seminal.life_duration = param.LDs * (2. * radius_seminal) * param.root_tissue_density

                # We defined the delay of emergence for the new primordium:
                apex_seminal.emergence_delay_in_thermal_time = seminal_inputs_file.emergence_delay_in_thermal_time[i]

    # ADDING THE PRIMORDIA OF ALL POSSIBLE ADVENTIOUS ROOTS:
    # ------------------------------------------------------
    # If there should be more than one main root (i.e. adventitious roots formed at the basis):
    if param.n_adventitious_roots > 0:

        # We read additional parameters from a table, with one column containing the delay for each emergence event,
        # and the second column containing the number of adventitious roots that have to emerge at each event.
        # We try to access an already-existing CSV file:
        adventitious_inputs_path = os.path.join("C:/Users/frees/rhizodep/src/rhizodep","adventitious_roots_inputs.csv")
        # If the file doesn't exist, we construct a new table using the specified parameters:
        if not os.path.exists(adventitious_inputs_path):
            print("NOTE: there was no CSV file describing the apparitions of adventitious roots!")
            print("We therefore built a table according to the parameters 'n_adventitious_roots' and 'ER'")
            # We initialize an empty data frame:
            adventitious_inputs_file = pd.DataFrame()
            # We define a list that will contain the successive thermal times corresponding to root emergence:
            list_time = [param.starting_time_for_adventitious_roots_emergence + x * 1/param.ER
                         for x in range(0, param.n_adventitious_roots)]
            # We define another list containing only "1" as the number of roots to be emerged for each event:
            list_number = np.ones(param.n_adventitious_roots)
            # We assigned the two lists to the dataframe, and record it:
            adventitious_inputs_file['emergence_delay_in_thermal_time'] = list_time
            adventitious_inputs_file['number_of_adventitious_roots_per_event'] = list_number
            adventitious_inputs_file.to_csv(os.path.join("C:/Users/frees/rhizodep/src/rhizodep", "adventitious_roots_inputs.csv"),
                                            na_rep='NA', index=False, header=True)
        else:
            adventitious_inputs_file = pd.read_csv(adventitious_inputs_path)

        # For each event of adventitious roots emergence:
        for i in range(0, len(adventitious_inputs_file.emergence_delay_in_thermal_time)):
            # For each adventitious root that can emerge at this emergence event:
            for j in range(0, int(adventitious_inputs_file.number_of_adventitious_roots_per_event[i])):

                # We make sure that the adventitious roots will have different random insertion angles:
                np.random.seed(param.random_choice + i * j*3)

                # Then we form one supporting segment of length 0 + one primordium of seminal root.
                # We add one new segment without any length on the same axis as the base:
                segment = ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Segment',
                                         type='Support_for_adventitious_root',
                                         root_order=1,
                                         angle_down=0,
                                         angle_roll=abs(np.random.normal(0, 180)),
                                         length=0.,
                                         radius=base_radius,
                                         identical_properties=False,
                                         nil_properties=True)

                # We define the radius of a adventitious root according to the parameter Di:
                if random:
                    radius_adventitious = abs(np.random.normal(param.D_ini / 2. * param.D_adv_to_D_ini_ratio,
                                                          param.D_ini / 2. * param.D_adv_to_D_ini_ratio *
                                                          param.CVDD))

                # And we add one new primordium of adventitious root on the previously defined segment:
                apex_adventitious = ADDING_A_CHILD(mother_element=segment, edge_type='+', label='Apex',
                                              type='Adventitious_root_before_emergence',
                                              root_order=1,
                                              angle_down=abs(np.random.normal(60, 10)),
                                              angle_roll=5,
                                              length=0.,
                                              radius=radius_adventitious,
                                              identical_properties=False,
                                              nil_properties=True)
                apex_adventitious.original_radius = radius_adventitious
                apex_adventitious.initial_radius = radius_adventitious
                apex_adventitious.growth_duration = param.GDs * (2. * radius_adventitious) ** 2 \
                                                    * param.main_roots_growth_extender
                # WATCH OUT!!! we artificially multiply growth duration for seminal and adventious roots!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                apex_adventitious.life_duration = param.LDs * (2. * radius_adventitious) * param.root_tissue_density

                # We defined the delay of emergence for the new primordium:
                apex_adventitious.emergence_delay_in_thermal_time \
                    = adventitious_inputs_file.emergence_delay_in_thermal_time[i]

    # FINAL APEX CONFIGURATION AT THE END OF THE MAIN ROOT:
    #------------------------------------------------------
    apex = ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Apex',
                          type='Normal_root_after_emergence',
                          root_order=1,
                          angle_down=0,
                          angle_roll=0,
                          length=0.,
                          radius=base_radius,
                          identical_properties=False,
                          nil_properties=True)
    apex.original_radius = apex.radius
    apex.initial_radius = apex.radius
    apex.growth_duration = param.GDs * (2. * base_radius) ** 2 * param.main_roots_growth_extender
    #WATCH OUT!!! we artificially multiply growth duration for seminal and adventious roots!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    apex.life_duration = param.LDs * (2. * base_radius) * param.root_tissue_density

    apex.volume = surfaces_and_volumes(g, apex, apex.radius, apex.length)["volume"]
    apex.struct_mass = apex.volume * param.root_tissue_density
    apex.initial_struct_mass = apex.struct_mass
    apex.initial_living_root_hairs_struct_mass = apex.living_root_hairs_struct_mass

    return g
