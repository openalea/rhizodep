#  -*- coding: utf-8 -*-

"""
    This script 'model' contains all functions necessary for running the model RhizoDep.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import numpy as np
import pandas as pd
from math import sqrt, pi, floor, exp, isnan
from decimal import Decimal
from scipy.integrate import solve_ivp

from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order

from . import parameters as param

# To display more than 5 columns when printing a Panda datframe:
pd.set_option('display.max_columns',20)

# FUNCTIONS FOR CALCULATING PROPERTIES ON THE MTG
#################################################

# Defining the volume and external surface of a given root element:
# -----------------------------------------------------------------
def volume_and_external_surface_from_radius_and_length(g, element, radius, length):
    """
    This function computes the volume (m3) of a root element and its external surface (excluding possible root hairs)
    based on the properties radius (m) and length (m) - and possibly on its type.
    :param g: the investigated MTG
    :param element: the investigated node of the MTG
    :param radius: the radius of the element (m)
    :param length: the length of the element (m)
    :return: a dictionary containing the volume and the external surface of the given element
    """

    # READING THE VALUES:
    #--------------------
    n = element
    vid = n.index()
    number_of_children = n.nb_children()

    # CALCULATIONS OF EXTERNAL SURFACE AND VOLUME
    #############################################

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
            # We avoid to remove the section of the sphere of a nodule:
            if son.type != "Root_nodule":
                sum_ramif_sections += pi * son.radius ** 2
        # And we subtract this sum of sections from the external area of the main cylinder:
        external_surface = 2 * pi * radius * length - sum_ramif_sections
        volume = pi * radius ** 2 * length
        # NOTE: we consider that there is no "dead" volume in the cylinder, even when a lateral root occupies part of
        # the mother root element. The volume of the daughter root included within the mother root element is assumed to
        # belong to the mother root element.

    # SPECIAL CASE FOR NODULE:
    ##########################
    
    if n.type == "Root_nodule":
        # We consider the surface and volume of a sphere:
        external_surface = 4 * pi * radius ** 2
        volume = 4 / 3. * pi * radius ** 3

    # CREATION OF A DICTIONARY THAT WILL BE USED TO RECORD THE OUTPUTS:
    ###################################################################
    dictionary = {"volume": volume, "external_surface": external_surface}

    # NOTE: the volume and external surface of the element have not been updated at this stage!
    return dictionary

# Defining specific surfaces of exchange within a root element:
# -------------------------------------------------------------
def specific_surfaces(element):
    """
    This function estimates the differents surfaces within the root section, e.g. from phloem vessels, stelar parenchyma, 
    cortical parenchyma and epidermal parenchyma.
    :param element: the root element to be considered
    :return: the updated root element
    """

    # READING THE VALUES:
    # --------------------
    n = element
    external_surface = n.external_surface

    # PHLOEM VESSELS:
    # We assume that the total surface of the phloem vessels is proportional to the external surface:
    phloem_surface = param.phloem_surfacic_fraction * external_surface

    # STELAR PARENCHYMA:
    stelar_parenchyma_surface = param.stelar_parenchyma_surfacic_fraction * external_surface

    # CORTICAL PARENCHYMA:
    cortical_parenchyma_surface = param.cortical_parenchyma_surfacic_fraction * external_surface

    # EPIDERMIS (WITHOUT ROOT HAIRS !!!):
    epidermis_surface_without_hairs = param.epidermal_parenchyma_surfacic_fraction * external_surface

    # RECORDING THE VALUES:
    # ----------------------
    n.phloem_surface = phloem_surface
    n.stelar_parenchyma_surface = stelar_parenchyma_surface
    n.cortical_parenchyma_surface = cortical_parenchyma_surface
    n.epidermis_surface_without_hairs = epidermis_surface_without_hairs

    return n

# Function for calculating, any given distance from root tip, the relative conductances of the endo- and exodermis:
#------------------------------------------------------------------------------------------------------------------
def endodermis_and_exodermis_conductances_as_a_function_of_x(distance_from_tip,
                                                             starting_distance_endodermis,
                                                             ending_distance_endodermis,
                                                             starting_distance_exodermis,
                                                             ending_distance_exodermis):
    """
    This simple function caclulates what should be the relative conductance of the endodermis and exodermis barriers,
    based on the distance from root tip.
    :param distance_from_tip: the distance from root tip (meter)
    :param starting_distance_endodermis: the distance at which the endodermis starts to mature (meter)
    :param ending_distance_endodermis: the distance at which the endodermis stops to mature (meter)
    :param starting_distance_exodermis: the distance at which the exodermis starts to mature (meter)
    :param ending_distance_exodermis: the distance at which the exodermis stops to mature (meter)
    :return: a dictionary containing conductance_endodermis and conductance_exodermis
    """

    # ENDODERMIS:
    # Above the starting distance, we consider that the conductance rapidly decreases as the endodermis is formed:
    if distance_from_tip > starting_distance_endodermis:
        # # OPTION 1: Conductance decreases as y = x0/x
        # conductance_endodermis = starting_distance_endodermis / distance_from_tip
        # OPTION 2: Conductance linearly decreases with x, up to reaching 0:
        conductance_endodermis = 1 - (distance_from_tip - starting_distance_endodermis) \
                                 / (ending_distance_endodermis - starting_distance_endodermis)
        if conductance_endodermis < 0.:
            conductance_endodermis = 0.
    # Below the starting distance, the conductance is necessarily maximal:
    else:
        conductance_endodermis = 1

    # EXODERMIS:
    # Above the starting distance, we consider that the conductance rapidly decreases as the exodermis is formed:
    if distance_from_tip > starting_distance_exodermis:
        # # OPTION 1: Conductance decreases as y = x0/x
        # conductance_exodermis = starting_distance_exodermis / distance_from_tip
        # OPTION 2: Conductance linearly decreases with x, up to reaching 0:
        conductance_exodermis = 1 - (distance_from_tip - starting_distance_exodermis) \
                                 / (ending_distance_exodermis - starting_distance_exodermis)
        if conductance_exodermis < 0.:
            conductance_exodermis = 0.
        # Below the starting distance, the conductance is necessarily maximal:
    else:
        conductance_exodermis = 1

    # We create a dictionary containing the values of the two conductances:
    dictionary = {"conductance_endodermis": conductance_endodermis, "conductance_exodermis": conductance_exodermis}

    return dictionary

# Function that integrates length-dependent values of endodermis and exodermis conductances between two points:
#--------------------------------------------------------------------------------------------------------------
def root_barriers_length_integrator(length_start,
                                    length_stop,
                                    number_of_length_steps,
                                    starting_distance_endodermis,
                                    ending_distance_endodermis,
                                    starting_distance_exodermis,
                                    ending_distance_exodermis):
    """
    This function calculates the mean conductance of endodermis and exodermis along a specified length, by sub-dividing
    this length into little subsegments, and by eventually summing their individual, weighted conductances.
    :param length_start: the position along the root where calculations start
    :param length_stop: the position along the root where calculations stop
    :param number_of_length_steps: the number of intermediate positions to compute along the path to get a good estimation of the mean conductances
    :param starting_distance_endodermis: the distance at which the endodermis starts to mature (meter)
    :param ending_distance_endodermis: the distance at which the endodermis stops to mature (meter)
    :param starting_distance_exodermis: the distance at which the exodermis starts to mature (meter)
    :param ending_distance_exodermis: the distance at which the exodermis stops to mature (meter)
    :return: a dictionary containing conductance_endodermis and conductance_exodermis
    """

    # We calculate the length step, by which we will progressively increase length between start and stop:
    length_step = (length_stop - length_start) / number_of_length_steps

    # We initialize the value to be computed:
    integrated_value_endodermis = 0.
    integrated_value_exodermis = 0.
   # We initialize the progressive_length:
    progressive_length = length_start + length_step / 2.

    # We cover the whole distance between length_start and length_stop, by the number of steps specified.
    # For each new sublength:
    for i in range(0, number_of_length_steps):
        # The new conductances are calculated in the middle of the current sub-length:
        dict_cond = endodermis_and_exodermis_conductances_as_a_function_of_x(progressive_length,
                                                                             starting_distance_endodermis,
                                                                             ending_distance_endodermis,
                                                                             starting_distance_exodermis,
                                                                             ending_distance_exodermis)
        # The integrated values for endodermis and exodermis conductances are increased:
        integrated_value_endodermis += dict_cond['conductance_endodermis'] / number_of_length_steps
        integrated_value_exodermis += dict_cond['conductance_exodermis'] / number_of_length_steps

        # We move the length to the next length step:
        progressive_length += length_step

    # We create a dictionary containing the values of the two conductances:
    dictionary = {"conductance_endodermis": integrated_value_endodermis, "conductance_exodermis": integrated_value_exodermis}

    return dictionary

def transport_barriers(g, n, computation_with_age=True, computation_with_distance_to_tip=False):
    """
    This function computes the actual relative conductances of cell walls, endodermis and exodermis for a given root
    element, based on either the distance to root tip or the age of the root segment.
    :param g: the root MTG to work on
    :param n: the root element where calculations will be made
    :param computation_with_age: if True, the calculations are done according to the age of the root element
    :param computation_with_distance_to_tip: if True, the calculations are done according to the distance of the root element from the tip
    :return: the updated element n with the new relative conductances
    """

    # READING THE VALUES:
    #--------------------
    vid = n.index()
    number_of_children = n.nb_children()
    length = n.length
    radius = n.radius
    distance_from_tip = n.distance_from_tip
    age = n.thermal_time_since_cells_formation

    # CELL WALLS RESISTANCE INCREASED IN THE MERISTEMATIC ZONE:
    # ---------------------------------------------------------

    meristem_zone_length = param.meristem_limite_zone_factor * radius
    relative_conductance_at_meristem = param.relative_conductance_at_meristem
    # We assume that the relative conductance of cell walls is either homogeneously reduced over the length of the
    # meristem zone, or is maximal elsewhere, i.e. equal to 1.
    # If the current element encompasses a part of the meristem zone:
    if (distance_from_tip - length) < meristem_zone_length:
        # Then we calculate the fraction of the length of the current element where the meristem is present:
        fraction_of_meristem_zone = 1 - (distance_from_tip - length) / meristem_zone_length
        # And the relative conductance of the cell walls in the whole element is a linear combination of the meristem
        # zone and the non-meristem zone:
        relative_conductance_walls = relative_conductance_at_meristem * fraction_of_meristem_zone \
                                     + 1. * (1 - fraction_of_meristem_zone)
    else:
        # Otherwise, the relative conductance of the cell walls is considered to be 1 by definition.
        relative_conductance_walls = 1.

    # BARRIERS OF ENDODERMIS & EXODERMIS:
    #------------------------------------

    # OPTION 1 - The formation of transport barriers is dictated by root segment age:
    if computation_with_age:

        # # WITH LINEAR EVOLUTION:
        # start_endo = param.start_thermal_time_for_endodermis_formation
        # end_endo = param.end_thermal_time_for_endodermis_formation
        # if age <= start_endo:
        #     relative_conductance_endodermis = 1.
        # elif age < end_endo:
        #     relative_conductance_endodermis = 1 - (age - start_endo) / (end_endo - start_endo)
        # else:
        #     relative_conductance_endodermis = 0.
        # # And the conductance of exodermis is also either 1, 0 or inbetween:
        # start_exo = param.start_thermal_time_for_exodermis_formation
        # end_exo = param.end_thermal_time_for_exodermis_formation
        # if age <= start_exo:
        #     relative_conductance_exodermis = 1.
        # elif age < end_endo:
        #     relative_conductance_exodermis = 1 - (age - start_exo) / (end_exo - start_exo)
        # else:
        #     relative_conductance_exodermis = 0.

        # WITH GOMPERTZ CONTINUOUS EVOLUTION:
        # Note: As the transition between 100% conductance and 0% for both endodermis and exodermis is described by a
        # Gompertz function involving a double exponential, we avoid unnecessary long calculations when the content of
        # the exponential is too high/low:
        # if param.endodermis_b - param.endodermis_c * age > 1000:
        #     relative_conductance_endodermis = param.endodermis_a / 100.
        # else:
        #     relative_conductance_endodermis = (100 - param.endodermis_a * np.exp(-np.exp(param.endodermis_b - param.endodermis_c * age)))/100.
        #     print("Conductance at the endodermis is", relative_conductance_endodermis)
        # if param.exodermis_b - param.exodermis_c * age > 1000:
        #     relative_conductance_exodermis = param.exodermis_a / 100.
        # else:
        #     relative_conductance_exodermis = (100 - param.exodermis_a * np.exp(-np.exp(param.exodermis_b/(60.*60.*24.) - param.exodermis_c * age)))/100.

        relative_conductance_endodermis = (100 - param.endodermis_a * np.exp(-np.exp(param.endodermis_b / (60. * 60. * 24.) - param.endodermis_c * age))) / 100.
        relative_conductance_exodermis = (100 - param.exodermis_a * np.exp(-np.exp(param.exodermis_b / (60. * 60. * 24.) - param.exodermis_c * age))) / 100.

    # OPTION 2 - The formation of transport barriers is dictated by the distance to root tip:
    if computation_with_distance_to_tip:

        # We define the distances from apex where barriers start/end:
        start_distance_endodermis = param.start_distance_for_endodermis_factor * radius
        end_distance_endodermis = param.end_distance_for_endodermis_factor * radius
        start_distance_exodermis = param.start_distance_for_exodermis_factor * radius
        end_distance_exodermis = param.end_distance_for_exodermis_factor * radius

        # We call a function that integrates the values of relative conductances between the beginning and the end of the
        # root element, knowing the evolution of the conductances with x (the distance from root tip):
        dict_cond = root_barriers_length_integrator(length_start = distance_from_tip - length,
                                                    length_stop = distance_from_tip,
                                                    number_of_length_steps = 10,
                                                    starting_distance_endodermis = start_distance_endodermis,
                                                    ending_distance_endodermis = end_distance_endodermis,
                                                    starting_distance_exodermis = start_distance_exodermis,
                                                    ending_distance_exodermis = end_distance_exodermis)
        relative_conductance_endodermis = dict_cond['conductance_endodermis']
        relative_conductance_exodermis = dict_cond['conductance_exodermis']

    # SPECIAL CASE: # We now consider a special case where the endodermis and/or exodermis barriers are temporarily
    # opened because of the emergence of a lateral root.

    # If there are more than one child, then it means there are lateral roots:
    if number_of_children > 1:
        # We define two maximal thermal durations, above which the barriers are not considered to be affected anymore:
        t_max_endo = param.max_thermal_time_since_endodermis_disruption
        t_max_exo = param.max_thermal_time_since_exodermis_disruption
        # We initialize empty lists of new conductances:
        possible_conductances_endo = []
        possible_conductances_exo = []
        # We cover all possible lateral roots emerging from the current element:
        for child_vid in g.Sons(vid, EdgeType='+'):
            # We get the lateral root as "son":
            son = g.node(child_vid)
            # We register the time since this lateral root emerged, and its current length:
            t = son.thermal_time_since_emergence
            lateral_length = son.length
            # If there is a lateral root and this has not emerged yet:
            if lateral_length <=0.:
                # Then we move to the next possible lateral root (otherwise, the loop stops and conductance remains unaltered):
                continue
            # ENDODERMIS: If the lateral root has emerged recently, its endodermis barrier has been diminished as soon
            # as the lateral started to elongate:
            t_since_endodermis_was_disrupted = t
            if t_since_endodermis_was_disrupted < t_max_endo:
                # We increase the relative conductance of endodermis according to the age of the lateral root,
                # considering that the conductance starts at 1 and linearily decreases with time until reaching 0.
                # However, if the barrier was not completely formed initially, we should not set it to zero, and therefore
                # define the new conductance as the maximal value between the original conductance and the new one:
                new_conductance = max(relative_conductance_endodermis,
                                      (t_max_endo - t_since_endodermis_was_disrupted) / t_max_endo)
                possible_conductances_endo.append(new_conductance)

            # EXODERMIS: If the lateral root has emerged recently, its exodermis barrier may have been diminished,
            # provided that the length of the lateral root is actually higher than the radius of the mother root
            # (i.e. that the lateral root tip has actually crossed the exodermis of the mother root):
            if lateral_length >= radius:
                # We approximate the time since the exodermis was disrupted, considering the the lateral root has
                # elongated at a constant speed:
                t_since_exodermis_was_disrupted = t * (lateral_length - radius) / lateral_length
                # If this time is small enough, the exodermis barrier may have been compromised:
                if t_since_exodermis_was_disrupted < t_max_exo:
                    # We increase the relative conductance of exodermis according to the time elapsed since the lateral
                    # root crossed the exodermis, considering that the conductance starts at 1 and linearily decreases
                    # with time until reaching 0. However, if the barrier was not completely formed initially, we should
                    # not set it to zero, and we therefore define the new conductance as the maximal value between the
                    # original conductance and the new one:
                    new_conductance = max(relative_conductance_exodermis,
                                          (t_max_exo - t_since_exodermis_was_disrupted) / t_max_exo)
                    possible_conductances_exo.append(new_conductance)

        # Now that we have covered all lateral roots, we limit the conductance of the barriers of the mother root
        # element by choosing the least limiting lateral root (only active if the lists did not remain empty):
        if possible_conductances_endo:
            relative_conductance_endodermis = max(possible_conductances_endo)
        if possible_conductances_exo:
            relative_conductance_exodermis = max(possible_conductances_exo)

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record the new conductances of cell walls, endodermis and exodermis:
    n.relative_conductance_walls = relative_conductance_walls
    n.relative_conductance_endodermis = relative_conductance_endodermis
    n.relative_conductance_exodermis = relative_conductance_exodermis

    return n

# Defining the different surfaces and volumes for all root elements:
# ------------------------------------------------------------------
def update_surfaces_and_volumes(g):

    """
    This function go through each root element and updates their surfaces, transport barriers and volume.
    :param g: the root MTG to be considered
    :return: the updated root MTG
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We intialize a value for the exchange surface:
        n.total_exchange_surface_with_soil_solution = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0.:
            continue
        
        # We call the function that automatically calculates the volume and external surface:
        surfaces_and_volumes_dict = volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)
        # We compute the total volume of the element:
        n.volume = surfaces_and_volumes_dict["volume"]
        # We calculate the current external surface of the element:
        n.external_surface = surfaces_and_volumes_dict["external_surface"]

        # We call the function that automatically updates the other surfaces of within the cells (ex: cortical symplast):
        specific_surfaces(n)

        # We call the function that automatically updates the transport barriers (i.e. endodermis and exodermis):
        transport_barriers(g, n, computation_with_age=True)

        # We update the surfaces of exchange between the soil solution and the accessible root symplast.
        # We first read the values from the current root element:
        S_epid = n.epidermis_surface_without_hairs
        S_hairs = n.living_root_hairs_external_surface
        S_cortex = n.cortical_parenchyma_surface
        S_stele = n.stelar_parenchyma_surface
        S_vessels = n.phloem_surface
        cond_walls = n.relative_conductance_walls
        cond_exo = n.relative_conductance_exodermis
        cond_endo = n.relative_conductance_endodermis
        # We then calculate the total surface of exchange between symplasm and apoplasm in the root parenchyma,
        # modulated by the conductance of cell walls (reduced in the meristematic zone) and the conductances of
        # endodermis and exodermis barriers (when these barriers are mature, conductance is expected to be 0 in general,
        # and part of the symplasm is not accessible anymore to the soil solution).
        S_exch_without_phloem = (S_epid + S_hairs) + cond_walls * (cond_exo * S_cortex + cond_endo * S_stele)
        S_exch_phloem = cond_walls * cond_exo * cond_endo * S_vessels
        # We finally record these exchange surfaces within n:
        n.non_vascular_exchange_surface_with_soil_solution = S_exch_without_phloem
        n.phloem_exchange_surface_with_soil_solution = S_exch_phloem
        n.total_exchange_surface_with_soil_solution = S_exch_without_phloem + S_exch_phloem
        n.root_exchange_surface_per_cm = (S_exch_without_phloem + S_exch_phloem) / (n.length*100)

    return g

# Defining the distance of a vertex from the tip for the whole root system:
# -------------------------------------------------------------------------
def update_distance_from_tip(g):
    """
    The function "distance_from_tip" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
    Note that the dist-to-tip of an apex is defined as its length (and not as 0).
    :param g: the investigated MTG
    :return: the MTG with an updated property 'distance_from_tip'
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

        # We record the initial distance_from_tip as the "former" one (to be used by other functions):
        n.former_distance_from_tip = n.distance_from_tip

        # We try to get the value of distance_from_tip for the neighbouring root element located closer to the apex of the root:
        try:
            # We calculate the new distance from the tip by adding its length to the distance of the successor:
            n.distance_from_tip = son.distance_from_tip + n.length
        except:
            # If there is no successor because the element is an apex or a root nodule:
            # Then we simply define the distance to the tip as the length of the element:
            n.distance_from_tip = n.length

    # We return a modified version of the MTG "g" with the updated property "distance_from_tip":
    return g

# Calculating the growth duration of a given root apex:
# -----------------------------------------------------
def calculate_growth_duration(radius, index, root_order, ArchiSimple=False):
    """
    This function computes the growth duration of a given apex, based on its radius and root order. If ArchiSimple
    option is activated, the function will calculate the duration proportionally to the square radius of the apex.
    Otherwise, the duration is set from a probability test, largely independent from the radius of the apex.
    :param radius: the radius of the apex element from which we compute the growth duration
    :param index: the index of the apex element, used for setting a new random seed for this element
    :param ArchiSimple: if True, the original rule set by ArchiSimple will be applied to compute the growth duration
    :return: the growth duration of the apex (s)
    """

    # If we only want to apply original ArchiSimple rules:
    if ArchiSimple:
        # Then the growth duration of the apex is proportional to the square diameter of the apex:
        growth_duration = param.GDs * (2. * radius) ** 2
    # Otherwise, we define the growth duration as a fixed value, randomly chosen between three possibilities:
    else:
        # We first define the seed of random, depending on the index of the apex:
        np.random.seed(param.random_choice * index)
        # We then generate a random float number between 0 and 1, which will determine whether growth duration is low, medium or high:
        random_result= np.random.random_sample()
        # CASE 1: The apex corresponds to a seminal or adventitious root
        if root_order ==1:
            growth_duration = param.GD_highest
        else:
            # If we select random zoning, then the growth duration will be drawn from a range, for three different cases
            # (from most likely to less likely):
            if param.GD_by_frequency:
                # CASE 2: Most likely, the growth duration will be low for a lateral root
                if random_result < param.GD_prob_low:
                    # We draw a random growth-duration in the lower range:
                    growth_duration = np.random.uniform(0., param.GD_low)
                # CASE 3: Occasionnaly, the growth duration may be a bit higher for a lateral root
                if random_result < param.GD_prob_medium:
                    # We draw a random growth-duration in the lower range:
                    growth_duration = np.random.uniform(param.GD_low, param.GD_medium)
                # CASE 3: Occasionnaly, the growth duration may be a bit higher for a lateral root
                else:
                    # We draw a random growth-duration in the lower range:
                    growth_duration = np.random.uniform(param.GD_medium, param.GD_high)
            # If random zoning has not been selected, a constant duration is selected for each probabibility range:
            else:
                # CASE 2: Most likely, the growth duration will be low for a lateral root
                if random_result < param.GD_prob_low:
                    growth_duration = param.GD_low
                # CASE 3: Occasionally, the growth duration of the lateral root may be significantly higher
                elif random_result < param.GD_prob_medium:
                    growth_duration = param.GD_medium
                # CASE 4: Exceptionally, the growth duration of the lateral root is as high as that from a seminal root,
                # as long as the radius of the lateral root is high enough (i.e. twice as high as the minimal possible radius)
                elif radius > 2 * param.Dmin/2.:
                    growth_duration = param.GD_highest

    # We return a modified version of the MTG "g" with the updated property "distance_from_tip":
    return growth_duration


# Calculation of the length of a root element intercepted between two z coordinates:
# ----------------------------------------------------------------------------------
def sub_length_z(x1, y1, z1, x2, y2, z2, z_up, z_down):
    """
    This function computes the length (m) between two points in space that is included within a horizontal layer
    defined by z_up and z_down coordinates.
    :param x1: x-coordinate of point 1
    :param y1: y-coordinate of point 1
    :param z1: z-coordinate of point 1
    :param x2: x-coordinate of point 2
    :param y2: y-coordinate of point 2
    :param z2: z-coordinate of point 2
    :param z_up: z-coordinate of the upper horizontal surface
    :param z_down: z-coordinate of the lower horizontal surface
    :return: the computed length.
    """
    # We make sure that the z coordinates are ordered in the right way:
    min_z = min(z1, z2)
    max_z = max(z1, z2)
    z_start = min(z_up, z_down)
    z_end = max(z_up, z_down)

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
    # Otherwise, the root element is not included between z_up and z_down, and intercepted length is 0:
    else:
        inter_length = 0

    # We return the computed length:
    return inter_length


# Integration of root variables within different z_intervals:
# -----------------------------------------------------------
def classifying_on_z(g, z_min=0., z_max=1., z_interval=0.1):
    """
    This function calculates different quantities of the MTG distributed among different z layers.
    :param g: the MTG for which we calculate properties
    :param z_min: the z-coordinate of the upper horizontal surface
    :param z_max: the z-coordinate of the lower horizontal surface
    :param z_interval: the thickness of each z-layer
    :return: a dictionary containing the distribution of key properties within the different z-layers
    """
    # We initialize empty dictionaries:
    included_length = {}
    dictionary_length = {}
    dictionary_struct_mass = {}
    dictionary_root_necromass = {}
    dictionary_surface = {}
    dictionary_net_hexose_exudation = {}
    dictionary_total_net_rhizodeposition = {}
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
        name_total_net_rhizodeposition_z = "total_net_rhizodeposition_" + str(round(z_start, 3)) + "-" + str(
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
                                               z_up=z_start,
                                               z_down=z_start + z_interval) / n.length
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
            total_included_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake_from_soil) * fraction_length
            total_included_rhizodeposition += (n.hexose_exudation - n.hexose_uptake_from_soil
                                               + n.mucilage_secretion + n.cells_release) * fraction_length
            total_included_hexose_degradation += n.hexose_degradation * fraction_length

        # We record the summed values for this interval of z in several dictionaries:
        dictionary_length[name_length_z] = total_included_length
        dictionary_struct_mass[name_struct_mass_z] = total_included_struct_mass
        dictionary_root_necromass[name_root_necromass_z] = total_included_root_necromass
        dictionary_surface[name_surface_z] = total_included_surface
        dictionary_net_hexose_exudation[name_net_hexose_exudation_z] = total_included_net_hexose_exudation
        dictionary_total_net_rhizodeposition[name_total_net_rhizodeposition_z] = total_included_rhizodeposition
        dictionary_hexose_degradation[name_hexose_degradation_z] = total_included_hexose_degradation

        # We also create a new property of the MTG that corresponds to the fraction of length of each node in the z interval:
        g.properties()[name_length_z] = included_length

    # Finally, we merge all dictionaries into a single one that will be returned by the function:
    final_dictionary = {}
    for d in [dictionary_length, dictionary_struct_mass, dictionary_root_necromass, dictionary_surface,
              dictionary_net_hexose_exudation,
              dictionary_total_net_rhizodeposition,
              dictionary_hexose_degradation]:
        final_dictionary.update(d)

    return final_dictionary


# Recording the properties of each node of a MTG in a CSV file:
# -------------------------------------------------------------
def recording_MTG_properties(g, file_name='g_properties.csv', list_of_properties=[]):
    """
    This function records the properties of each node of the MTG "g" inside a csv file.
    :param g: the MTG where properties are recorded
    :param file_name: the name of the csv file where properties of each node will be recorded
    :param list_of_properties: a list containing the names of the specific properties to be recorded
                              (if the list is empty, all the properties of the MTG will be recorded)
    :return: [no return]
    """

    if list_of_properties==[]:
        # We define and reorder the list of all properties of the MTG by alphabetical order:
        list_of_properties = list(g.properties().keys())
        list_of_properties.sort(key=str.lower)

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
            # node_properties.append(getattr(n, property, "NA"))
            try:
                value = g.properties()[property][vid]
            except:
                # print("!!! ERROR: the property", property,"could not be assessed for the element", vid)
                value = "NA"
            node_properties.append(value)
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
    Parameters correspond to the value of the process at reference temperature T_ref (process_at_T_ref),
    to two empirical coefficients A and B, and to a coefficient C used to switch between different formalisms.
    If C=0 and B=1, then the relationship corresponds to a classical linear increase with temperature (thermal time).
    If C=1, A=0 and B>1, then the relationship corresponds to a classical exponential increase with temperature (Q10).
    If C=1, A<0 and B>0, then the relationship corresponds to bell-shaped curve, close to the one from Parent et al. (2010).
    :param temperature_in_Celsius: the temperature for which the new value of the process will be calculated
    :param process_at_T_ref: the reference value of the process at the reference temperature
    :param T_ref: the reference temperature
    :param A: parameter A (may be equivalent to the coefficient of linear increase)
    :param B: parameter B (may be equivalent to the Q10 value)
    :param C: parameter C (either 0 or 1)
    :return: the new value of the process
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
                   root_order=1, angle_down=45., angle_roll=0., length=0., radius=0.,
                   identical_properties=True, nil_properties=False):
    """
    This function creates a new child element on the mother element, based on the function add_child.
    When called, this allows to automatically define standard properties without defining them in the main code.
    :param mother_element: the node of the MTG on which the child element will be created
    :param edge_type: the type of relationship between the mother element and its child ('+' or '<')
    :param label: label of the child element
    :param type: type of the child element
    :param root_order: root_order of the child element
    :param angle_down: angle_down of the child element
    :param angle_roll: angle_roll of the child element
    :param length: length of the child element
    :param radius: radius of the child element
    :param identical_properties: if True, the main properties of the child will be identical to those of the mother
    :param nil_properties: if True, the main properties of the child will be 0
    :return: the new child element
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
                                             total_exchange_surface_with_soil_solution=0.,
                                             volume=0.,
                                             dist_to_ramif=0.,
                                             distance_from_tip=0.,
                                             former_distance_from_tip=0.,
                                             actual_elongation=0.,
                                             actual_elongation_rate=0.,
                                             # Quantities and concentrations:
                                             # -------------------------------
                                             struct_mass=0.,
                                             initial_struct_mass=0.,
                                             C_hexose_root=0.,
                                             C_hexose_reserve=0.,
                                             C_hexose_soil=0.,
                                             Cs_mucilage_soil=0.,
                                             Cs_cells_soil=0.,
                                             C_sucrose_root=0.,
                                             Deficit_sucrose_root=0.,
                                             Deficit_hexose_root=0.,
                                             Deficit_hexose_reserve=0.,
                                             Deficit_hexose_soil=0.,
                                             Deficit_mucilage_soil=0.,
                                             Deficit_cells_soil=0.,
                                             Deficit_sucrose_root_rate=0.,
                                             Deficit_hexose_root_rate=0.,
                                             Deficit_hexose_reserve_rate=0.,
                                             Deficit_hexose_soil_rate=0.,
                                             Deficit_mucilage_soil_rate=0.,
                                             Deficit_cells_soil_rate=0.,
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
                                             hexose_consumption_by_growth_rate=0.,
                                             hexose_possibly_required_for_elongation=0.,
                                             hexose_production_from_phloem=0.,
                                             sucrose_loading_in_phloem=0.,
                                             hexose_mobilization_from_reserve=0.,
                                             hexose_immobilization_as_reserve=0.,
                                             hexose_exudation=0.,
                                             hexose_uptake_from_soil=0.,
                                             phloem_hexose_exudation=0.,
                                             phloem_hexose_uptake_from_soil=0.,
                                             mucilage_secretion=0.,
                                             cells_release=0.,
                                             total_net_rhizodeposition=0.,
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
                                             actual_time_since_cells_formation=0.,
                                             actual_potential_time_since_emergence=0.,
                                             actual_time_since_growth_stopped=0.,
                                             actual_time_since_death=0.,
                                             thermal_time_since_primordium_formation=0.,
                                             thermal_time_since_emergence=0.,
                                             thermal_time_since_cells_formation=0.,
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
                                             total_exchange_surface_with_soil_solution=0.,
                                             volume=0.,
                                             dist_to_ramif=mother_element.dist_to_ramif,
                                             distance_from_tip=mother_element.distance_from_tip,
                                             former_distance_from_tip=mother_element.former_distance_from_tip,
                                             actual_elongation=mother_element.actual_elongation,
                                             actual_elongation_rate=mother_element.actual_elongation_rate,
                                             # Quantities and concentrations:
                                             # -------------------------------
                                             struct_mass=mother_element.struct_mass,
                                             initial_struct_mass=mother_element.initial_struct_mass,
                                             C_hexose_root=mother_element.C_hexose_root,
                                             C_hexose_reserve=mother_element.C_hexose_reserve,
                                             C_hexose_soil=mother_element.C_hexose_soil,
                                             Cs_mucilage_soil=mother_element.Cs_mucilage_soil,
                                             Cs_cells_soil=mother_element.Cs_cells_soil,
                                             C_sucrose_root=mother_element.C_sucrose_root,
                                             Deficit_sucrose_root=mother_element.Deficit_sucrose_root,
                                             Deficit_hexose_root=mother_element.Deficit_hexose_root,
                                             Deficit_hexose_reserve=mother_element.Deficit_hexose_reserve,
                                             Deficit_hexose_soil=mother_element.Deficit_hexose_soil,
                                             Deficit_mucilage_soil=mother_element.Deficit_mucilage_soil,
                                             Deficit_cells_soil=mother_element.Deficit_cells_soil,
                                             Deficit_sucrose_root_rate=mother_element.Deficit_sucrose_root_rate,
                                             Deficit_hexose_root_rate=mother_element.Deficit_hexose_root_rate,
                                             Deficit_hexose_reserve_rate=mother_element.Deficit_hexose_reserve_rate,
                                             Deficit_hexose_soil_rate=mother_element.Deficit_hexose_soil_rate,
                                             Deficit_mucilage_soil_rate=mother_element.Deficit_mucilage_soil_rate,
                                             Deficit_cells_soil_rate=mother_element.Deficit_cells_soil_rate,

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
                                             hexose_uptake_from_soil=mother_element.hexose_uptake_from_soil,
                                             phloem_hexose_exudation=mother_element.phloem_hexose_exudation,
                                             phloem_hexose_uptake_from_soil=mother_element.phloem_hexose_uptake_from_soil,
                                             mucilage_secretion=mother_element.mucilage_secretion,
                                             cells_release=mother_element.cells_release,
                                             total_net_rhizodeposition=mother_element.total_net_rhizodeposition,
                                             hexose_degradation=mother_element.hexose_degradation,
                                             mucilage_degradation=mother_element.mucilage_degradation,
                                             cells_degradation=mother_element.cells_degradation,
                                             hexose_consumption_by_growth=mother_element.hexose_consumption_by_growth,
                                             hexose_consumption_by_growth_rate=mother_element.hexose_consumption_by_growth_rate,
                                             specific_net_exudation=mother_element.specific_net_exudation,
                                             # Time indications:
                                             # ------------------
                                             growth_duration=mother_element.growth_duration,
                                             life_duration=mother_element.life_duration,
                                             actual_time_since_primordium_formation=mother_element.actual_time_since_primordium_formation,
                                             actual_time_since_emergence=mother_element.actual_time_since_emergence,
                                             actual_time_since_cells_formation=mother_element.actual_time_since_cells_formation,
                                             actual_time_since_growth_stopped=mother_element.actual_time_since_growth_stopped,
                                             actual_time_since_death=mother_element.actual_time_since_death,

                                             thermal_time_since_primordium_formation=mother_element.thermal_time_since_primordium_formation,
                                             thermal_time_since_emergence=mother_element.thermal_time_since_emergence,
                                             thermal_time_since_cells_formation=mother_element.thermal_time_since_cells_formation,
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
                     ArchiSimple=True, printing_warnings=False,
                     growth_reduction = 1.,
                     soil_temperature_in_Celsius=20):
    """
    This function computes a new length (m) based on the elongation process described by ArchiSimple and regulated by
    the available concentration of hexose.
    :param initial_length: the initial length (m)
    :param radius: radius (m)
    :param C_hexose_root: the concentration of hexose available for elongation (mol of hexose per gram of strctural mass)
    :param elongation_time_in_seconds: the period of elongation (s)
    :param ArchiSimple: if True, a classical ArchiSimple elongation without hexose regulation will be computed
    :param printing_warnings: if True, all warning messages will be displayed in the console
    :param growth_reduction: a factor reducing the rate of elongation (dimensionless, between 0 and 1)
    :param soil_temperature_in_Celsius: the soil temperature perceived by the root (in degrees Celsius)
    :return: the new elongated length
    """

    # If we keep the classical ArchiSimple rule:
    if ArchiSimple:
        # Then the elongation is calculated following the rules of Pages et al. (2014):
        elongation = param.EL * growth_reduction * 2. * radius * elongation_time_in_seconds
    else:
        # Otherwise, we additionally consider a limitation of the elongation according to the local concentration of hexose,
        # based on a Michaelis-Menten formalism:
        if C_hexose_root > param.C_hexose_min_for_elongation:
            elongation = (param.EL * growth_reduction * 2. * radius
                          * C_hexose_root / (param.Km_elongation + C_hexose_root) * elongation_time_in_seconds)
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
                         soil_temperature_in_Celsius=20, random=True,
                         root_order_limitation=False, root_order_treshold=2, simple_growth_duration=True):
    """
    This function considers the formation of a primordium on a root apex, and, if possible, creates this new element
    of length 0.
    :param g: the MTG to work on
    :param apex: the apex on which the primordium may be created
    :param elongation_rate: the rate of elongation of the apex over the time step during which the primordium may be created (m s-1)
    :param time_step_in_seconds: the time step during which the primordium may be created (s)
    :param soil_temperature_in_Celsius: the temperature of root growth (degree Celsius)
    :param random: if True, randomness of angles will be considered
    :param root_order_limitation: if True, primordia of high root order will not be formed
    :param root_order_treshold: the root order above which primordium formation may be forbidden.
    :return: the new primordium
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
                                            ((apex.radius - param.Dmin / 2.) * param.RMD + param.Dmin / 2.) * param.CVDD)
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

    # We add a condition, i.e. potential_radius should not be larger than the one from the mother root:
    if potential_radius > apex.radius:
        potential_radius = apex.radius

    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:
    if apex.dist_to_ramif > param.IPD and potential_radius >= param.Dmin and potential_radius <= apex.radius:
        # The distance that the tip of the apex has covered since the actual primordium formation is calculated:
        elongation_since_last_ramif = apex.dist_to_ramif - param.IPD

        # A specific rolling angle is attributed to the parent apex:
        apex.angle_roll = apex_angle_roll

        # We verify that the apex has actually elongated:
        if apex.actual_elongation > 0 and elongation_rate >0.:
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
        # We specifically recomputes the growth duration:
        ramif.growth_duration = calculate_growth_duration(radius=ramif.radius, index=ramif.index(),
                                                          root_order=ramif.root_order, ArchiSimple=simple_growth_duration)
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
    This function computes the list of root elements that can supply C as hexose for sustaining the elongation
    of a given element, as well as their structural mass and their amount of available hexose.
    :param g: the MTG corresponding to the root system
    :param element: the element for which we calculate the possible supply of C for its elongation
    :return: three lists containing the indices of elements, their hexose amount (mol of hexose) and their structural mass (g).
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
                # (EXCLUDING sugars in the living root hairs):
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
    This function considers a root apex, i.e. the terminal root element of a root axis (including the primordium of a
    root that has not emerged yet), and calculates its potential elongation, without actually elongating the apex or
    forming any new root primordium in the standard case (only when ArchiSimple option is set to True). Aging of the
    apex is also considered.
    :param g: the root MTG to which the apex belongs
    :param apex: the apex to be considered
    :param time_step_in_seconds: the time step over which elongation is considered
    :param ArchiSimple: a Boolean (True/False) expliciting whether original rules for ArchiSimple should be kept or not
    :param soil_temperature_in_Celsius: the actual temperature experienced by the root apex
    :param printing_warnings: a Boolean (True/False) expliciting whether Warnings should be printed in the console
    :return: the updated apex
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
                                                     growth_reduction=1-param.friction_coefficient,
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
                                                     growth_reduction=1-param.friction_coefficient,
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
                # The possibility of emergence of a lateral root from the parent is recorded inside the parent:
                parent.lateral_root_emergence_possibility = "Possible"
                parent.lateral_primordium_index = apex.index()
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
                                                 growth_reduction=1-param.friction_coefficient,
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
                apex.actual_time_since_cells_formation += time_step_in_seconds
                apex.thermal_time_since_cells_formation += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

            # OTHERWISE, THE APEX HAS TO STOP DURING THIS TIME STEP:
            else:
                # The type is declared "Just stopped":
                apex.type = "Just_stopped"
                # Then the exact time since growth stopped is calculated:
                apex.thermal_time_since_growth_stopped = apex.thermal_time_since_emergence \
                                                         + time_step_in_seconds * temperature_time_adjustment \
                                                         - apex.growth_duration
                apex.actual_time_since_growth_stopped = apex.thermal_time_since_growth_stopped / temperature_time_adjustment

                # We record the different element that can contribute to the C supply necessary for growth,
                # and we calculate a mean concentration of hexose in this supplying zone:
                calculating_C_supply_for_elongation(g, apex)
                # And the potential elongation of the apex before growth stopped is calculated:
                apex.potential_length = elongated_length(initial_length=apex.length, radius=apex.radius,
                                                         C_hexose_root=apex.growing_zone_C_hexose_root,
                                                         elongation_time_in_seconds=time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped,
                                                         ArchiSimple=ArchiSimple,
                                                         growth_reduction=1-param.friction_coefficient,
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
                apex.actual_time_since_cells_formation += time_step_in_seconds
                apex.thermal_time_since_cells_formation += time_step_in_seconds * temperature_time_adjustment
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
                apex.actual_time_since_cells_formation += time_step_in_seconds
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.actual_time_since_death += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_cells_formation += time_step_in_seconds * temperature_time_adjustment
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
                apex.thermal_time_since_death = apex.thermal_time_since_growth_stopped \
                                                + time_step_in_seconds * temperature_time_adjustment - apex.life_duration
                apex.actual_time_since_death = apex.thermal_time_since_death / temperature_time_adjustment
                # And the other times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.actual_time_since_cells_formation += time_step_in_seconds
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_cells_formation += time_step_in_seconds * temperature_time_adjustment
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
def potential_segment_development(g, segment, time_step_in_seconds=60. * 60. * 24., radial_growth=True,
                                  ArchiSimple=False, soil_temperature_in_Celsius=20):
    """
    This function considers a root segment, i.e. a root element that can thicken but not elongate, and calculates its
    potential increase in radius according to the pipe model (possibly regulated by C availability), and its possible death.
    :param g: the root MTG to which the segment belongs
    :param segment: the segment to be considered
    :param time_step_in_seconds: the time step over which elongation is considered
    :param radial_growth: a Boolean (True/False) expliciting whether radial growth should be considered or not
    :param ArchiSimple: a Boolean (True/False) expliciting whether original rules for ArchiSimple should be kept or not
    :param soil_temperature_in_Celsius: the actual temperature experienced by the root segment
    :return: the updated segment
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
    # NOTE: a nodule is considered here as a tumor which grows radially by feeding from root hexose, but does not produce
    # new root axes.

    if segment.type == "Root_nodule":
        # We consider the amount of hexose available in the nodule AND in the parent segment
        # (EXCLUDING the amount of hexose in living rot hairs):
        index_parent = g.Father(segment.index(), EdgeType='+')
        parent = g.node(index_parent)
        segment.hexose_available_for_thickening = parent.C_hexose_root * parent.struct_mass \
                                                  + segment.C_hexose_root * segment.struct_mass
        # We calculate an average concentration of hexose that will help to regulate nodule growth:
        C_hexose_regulating_nodule_growth = segment.hexose_available_for_thickening / (
                parent.struct_mass + segment.struct_mass)
        # We modulate the relative increase in radius by the amount of C available in the nodule:
        thickening_rate = param.relative_nodule_thickening_rate_max \
                          * C_hexose_regulating_nodule_growth \
                          / (param.Km_nodule_thickening + C_hexose_regulating_nodule_growth)
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
    # (EXCLUDING the amount of hexose in living root hairs):
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


    # CHECKING WHETHER THE APEX OF THE ROOT AXIS HAS STOPPED GROWING:
    # ---------------------------------------------------------------

    # We look at the apex of the axis to which the segment belongs (i.e. we get the last element of the axis):
    index_apex = g.Axis(segment.index())[-1]
    apex = g.node(index_apex)
    # print("For segment", segment.index(), "the terminal index is", index_apex, "and has the type", apex.type)
    if apex.label != "Apex":
        print("ERROR: when trying to access the terminal apex of the axis of the segment", segment.index(),
              "we obtained the element", index_apex," that is a", apex.label, "!!!")
    # If the apex of the root axis has stopped growing, we propagrate this information to the segment,
    # unless the segment is just a supporting element of a lateral seminal or adventitious axis:
    if segment.type != "Support_for_seminal_root" and segment.type != "Support_for_adventitious_root":
        if apex.type == "Just_stopped":
            segment.type = "Just_stopped"
        elif apex.type == "Stopped":
            segment.type = "Stopped"

    # CHECKING POSSIBLE ROOT SEGMENT DEATH (and computing intermediate variable for radial growth):
    # ---------------------------------------------------------------------------------------------

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
        # Otherwise, if the segment does not correspond to a supporting element without length:
        elif segment.type != "Support_for_seminal_root" and segment.type != "Support_for_adventitious_root":
            # Then the segment has to die:
            segment.type = "Just_dead"
    # Otherwise, at least one of the children axis is not dead, so the father segment should not be dead

    # REGULATION OF RADIAL GROWTH BY AVAILABLE CARBON:
    # ------------------------------------------------
    # If the radial growth is possible:
    if radial_growth:
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
        elif segment.length > 0. and segment.theoretical_radius > segment.radius \
                and segment.C_hexose_root > param.C_hexose_min_for_thickening:
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
                # print(" >>> Radial growth has been limited by the maximal tickening rate!")
            # Otherwise, the potential radius to achieve is equal to the theoretical one:
            else:
                segment.potential_radius = segment.theoretical_radius
        # And if the segment corresponds to one of the elements of length 0 supporting one seminal or adventitious root:
        if segment.type == "Support_for_seminal_root" or segment.type == "Support_for_adventitious_root":
            # Then the radius is directly increased, as this element will not be considered in the function calculating actual growth:
            segment.radius = segment.potential_radius

    # UPDATING THE DIFFERENT TIMES:
    #------------------------------

    # We increase the various time variables:
    segment.actual_time_since_primordium_formation += time_step_in_seconds
    segment.actual_time_since_emergence += time_step_in_seconds
    segment.actual_time_since_cells_formation += time_step_in_seconds
    segment.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
    segment.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
    segment.thermal_time_since_cells_formation += time_step_in_seconds * temperature_time_adjustment

    if segment.type == "Just_stopped":
        segment.actual_time_since_growth_stopped = apex.actual_time_since_growth_stopped
        segment.thermal_time_since_growth_stopped = apex.actual_time_since_growth_stopped * temperature_time_adjustment
    if segment.type == "Stopped":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
    if segment.type == "Just_dead":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
        # AVOIDING PROBLEMS - We check that the list of times_since_death is not empty:
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
# We define a class "Simulate" that is used to simulate the development of apices and segments on the whole MTG "g":
class Simulate_potential_growth(object):
    # We initiate the object with a list of root apices:
    def __init__(self, g):
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex" or "Segment", from the tip to the base:
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = next(root_gen)
        self.apices_list = [g.node(v) for v in pre_order(g, root) if g.label(v) == 'Apex']
        self.segments_list = [g.node(v) for v in post_order(g, root) if g.label(v) == 'Segment']

    def step(self, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth=True, ArchiSimple=False,
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
def potential_growth(g, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth=True, ArchiSimple=False,
                     soil_temperature_in_Celsius=20):
    """
    This function covers the whole root MTG and computes the potential growth of segments and apices.
    :param g: the root MTG to be considered
    :param time_step_in_seconds: the time step over which potential growth is computed
    :param radial_growth: a Boolean (True/False) expliciting whether radial growth should be considered or not
    :param ArchiSimple: a Boolean (True/False) expliciting whether original rules for ArchiSimple should be kept or not
    :param soil_temperature_in_Celsius: the same, homogeneous temperature experienced by each root element of the MTG
    :return:
    """
    # We simulate the development of all apices and segments in the MTG:
    simulator = Simulate_potential_growth(g)
    simulator.step(time_step_in_seconds=time_step_in_seconds, radial_growth=radial_growth, ArchiSimple=ArchiSimple,
                   soil_temperature_in_Celsius=soil_temperature_in_Celsius)
    return


# Function that divides a root into segment and generates root primordia:
#------------------------------------------------------------------------
def segmentation_and_primordium_formation(g, apex, time_step_in_seconds=1. * 60. * 60. * 24.,
                                          soil_temperature_in_Celsius=20, simple_growth_duration=True, random=True,
                                          nodules=True, root_order_limitation=False, root_order_treshold=2):
    # NOTE: This function is supposed to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    """
    This function transforms an elongated root apex into a list of segments and a terminal, smaller apex. A primordium
    of a lateral root can be formed on the new segment in some cases, depending on the distance to tip and the root orders.
    :param g: the root MTG to which the apex belongs
    :param apex: the root apex to be segmented
    :param time_step_in_seconds: the time step over which segmentation may have occured
    :param soil_temperature_in_Celsius: the temperature experienced by the root element
    :param simple_growth_duration: a Boolean (True/False) expliciting whether original rules for ArchiSimple should be kept or not
    :param random: a Boolean (True/False) expliciting whether random orientations can be defined for the new elements
    :param nodules: a Boolean (True/False) expliciting whether nodules could be formed or not
    :param root_order_limitation: a Boolean (True/False) expliciting whether lateral roots should be prevented above a certain root order
    :param root_order_treshold: the root order above which new lateral roots cannot be formed
    :return:
    """

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
    # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.root_growth_T_ref,
                                                           A=param.root_growth_A,
                                                           B=param.root_growth_B,
                                                           C=param.root_growth_C)

    # ADJUSTING ROOT ANGLES FOR THE FUTURE NEW SEGMENTS:
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

    # RECORDING THE VALUES OF THE APEX BEFORE SEGMENTATION (e.g. the values that will be altered by the segmentation!):
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
    initial_struct_mass_produced = apex.struct_mass_produced
    initial_resp_maintenance = apex.resp_maintenance
    initial_resp_growth = apex.resp_growth

    initial_root_hairs_struct_mass = apex.root_hairs_struct_mass
    initial_root_hairs_struct_mass_produced = apex.root_hairs_struct_mass_produced
    initial_living_root_hairs_struct_mass = apex.living_root_hairs_struct_mass
    initial_living_root_hairs_number = apex.living_root_hairs_number
    initial_dead_root_hairs_number = apex.dead_root_hairs_number
    initial_total_root_hairs_number = apex.total_root_hairs_number
    initial_living_root_hairs_external_surface = apex.living_root_hairs_external_surface

    # NOTE: The following is not an error, we also need to record the "initial" structural mass and surfaces before growth!
    initial_initial_struct_mass = apex.initial_struct_mass
    initial_initial_living_root_hairs_struct_mass = apex.initial_living_root_hairs_struct_mass
    initial_initial_external_surface = apex.initial_external_surface
    initial_initial_living_root_hairs_external_surface = apex.initial_living_root_hairs_external_surface

    initial_hexose_exudation = apex.hexose_exudation
    initial_hexose_uptake_from_soil = apex.hexose_uptake_from_soil
    initial_phloem_hexose_exudation = apex.phloem_hexose_exudation
    initial_phloem_hexose_uptake_from_soil = apex.phloem_hexose_uptake_from_soil
    initial_mucilage_secretion = apex.mucilage_secretion
    initial_cells_release = apex.cells_release
    initial_total_net_rhizodeposition = apex.total_net_rhizodeposition
    initial_hexose_degradation = apex.hexose_degradation
    initial_mucilage_degradation = apex.mucilage_degradation
    initial_cells_degradation = apex.cells_degradation
    initial_hexose_growth_demand = apex.hexose_growth_demand
    initial_hexose_consumption_by_growth = apex.hexose_consumption_by_growth
    initial_hexose_consumption_by_growth_rate = apex.hexose_consumption_by_growth_rate
    initial_hexose_production_from_phloem = apex.hexose_production_from_phloem
    initial_sucrose_loading_in_phloem = apex.sucrose_loading_in_phloem
    initial_hexose_mobilization_from_reserve = apex.hexose_mobilization_from_reserve
    initial_hexose_immobilization_as_reserve = apex.hexose_immobilization_as_reserve
    initial_Deficit_sucrose_root = apex.Deficit_sucrose_root
    initial_Deficit_hexose_root = apex.Deficit_hexose_root
    initial_Deficit_hexose_reserve = apex.Deficit_hexose_reserve
    initial_Deficit_hexose_soil = apex.Deficit_hexose_soil
    initial_Deficit_mucilage_soil = apex.Deficit_mucilage_soil
    initial_Deficit_cells_soil = apex.Deficit_cells_soil

    initial_Deficit_sucrose_root_rate = apex.Deficit_sucrose_root_rate
    initial_Deficit_hexose_root_rate = apex.Deficit_hexose_root_rate
    initial_Deficit_hexose_reserve_rate = apex.Deficit_hexose_reserve_rate
    initial_Deficit_hexose_soil_rate = apex.Deficit_hexose_soil_rate
    initial_Deficit_mucilage_soil_rate = apex.Deficit_mucilage_soil_rate
    initial_Deficit_cells_soil_rate = apex.Deficit_cells_soil_rate

    # We record the type of the apex, as it may correspond to an apex that has stopped (or even died):
    initial_type = apex.type
    initial_lateral_root_emergence_possibility = apex.lateral_root_emergence_possibility

    # CASE 1: NO SEGMENTATION IS NECESSARY
    #-------------------------------------
    # If the length of the apex is smaller than the defined length of a root segment:
    if apex.length <= param.segment_length:

        # CONSIDERING POSSIBLE PRIMORDIUM FORMATION:
        # We simply call the function primordium_formation to check whether a primordium should have been formed
        # (Note: we assume that the segment length is always smaller than the inter-branching distance IBD,
        # so that in this case, only 0 or 1 primordium may have been formed - the function is called only once):
        new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                                             root_order_limitation=root_order_limitation,
                                             root_order_treshold=root_order_treshold,
                                             simple_growth_duration=simple_growth_duration))

        # If there has been an actual elongation of the root apex:
        if apex.actual_elongation_rate > 0.:
            # # RECALCULATING DIMENSIONS:
            # # We assume that the growth functions that may have been called previously have only modified radius and length,
            # # but not the struct_mass and the total amounts present in the root element.
            # # We modify the geometrical features of the present element according to the new length and radius:
            # apex.volume = volume_and_external_surface_from_radius_and_length(g, apex, apex.radius, apex.length)["volume"]
            # apex.struct_mass = apex.volume * param.root_tissue_density

            # CALCULATING THE AVERAGE TIME SINCE ROOT CELLS FORMED:
            # We update the time since root cells have formed, based on the elongation rate and on the principle that
            # cells appear at the very end of the root tip, and then age. In this case, the average age of the element is
            # calculated as the mean value between the incremented age of the part that was already formed and has not
            # elongated, and the average age of the new part formed.
            # The age of the initially-existing part is simply incremented by the time step:
            Age_of_non_elongated_part = apex.actual_time_since_cells_formation + time_step_in_seconds
            # The age of the cells at the top of the elongated part is by definition equal to:
            Age_of_elongated_part_up = apex.actual_elongation / apex.actual_elongation_rate
            # The age of the cells at the root tip is by definition 0:
            Age_of_elongated_part_down = 0
            # The average age of the elongated part is calculated as the mean value between both extremities:
            Age_of_elongated_part = 0.5 * (Age_of_elongated_part_down + Age_of_elongated_part_up)
            # Eventually, the average age of the whole new element is calculated from the age of both parts:
            apex.actual_time_since_cells_formation = (apex.initial_length * Age_of_non_elongated_part \
                                                      + (apex.length - apex.initial_length) * Age_of_elongated_part) \
                                                     / apex.length
            # We also calculate the thermal age of root cells according to the previous thermal time of
            # initially-exisiting cells:
            Thermal_age_of_non_elongated_part = apex.thermal_time_since_cells_formation + time_step_in_seconds * \
                                                temperature_time_adjustment
            Thermal_age_of_elongated_part = Age_of_elongated_part * temperature_time_adjustment
            apex.thermal_time_since_cells_formation = (apex.initial_length * Thermal_age_of_non_elongated_part \
                                                       + (apex.length - apex.initial_length) * Thermal_age_of_elongated_part) / apex.length

    # CASE 2: THE APEX HAS TO BE SEGMENTED
    #-------------------------------------
    # Otherwise, we have to calculate the number of entire segments within the apex.
    else:

        # CALCULATION OF THE NUMBER OF ROOT SEGMENTS TO BE FORMED:
        # If the final length of the apex does not correspond to an entire number of segments:
        if apex.length / param.segment_length - floor(apex.length / param.segment_length) > 0.:
            # Then the total number of segments to be formed is:
            n_segments = floor(apex.length / param.segment_length)
        else:
            # Otherwise, the number of segments to be formed is decreased by 1,
            # so that the last element corresponds to an apex with a positive length:
            n_segments = floor(apex.length / param.segment_length) - 1
        n_segments = int(n_segments)

        # We need to calculate the final length of the terminal apex:
        final_apex_length = initial_length - n_segments * param.segment_length

        # FORMATION OF THE NEW SEGMENTS
        # We develop each new segment, except the last one, by transforming the current apex into a segment
        # and by adding a new apex after it, in an iterative way for (n-1) segments:
        for i in range(1, n_segments + 1):

            # We define the length of the present element (still called "apex") as the constant length of a segment:
            apex.length = param.segment_length
            # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
            apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - param.segment_length * i)
            # We modify the geometrical features of the present element according to the new length:
            apex.volume = volume_and_external_surface_from_radius_and_length(g, apex, apex.radius, apex.length)["volume"]
            apex.struct_mass = apex.volume * param.root_tissue_density

            # We calculate the mass fraction that the segment represents compared to the whole element prior to segmentation:
            mass_fraction = apex.struct_mass / initial_struct_mass

            # We modify the variables representing total amounts according to this mass fraction:
            apex.resp_maintenance = initial_resp_maintenance * mass_fraction
            apex.resp_growth = initial_resp_growth * mass_fraction

            apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
            apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction
            apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction
            apex.initial_external_surface = initial_initial_external_surface * mass_fraction
            apex.initial_living_root_hairs_external_surface = initial_initial_living_root_hairs_external_surface \
                                                              * mass_fraction

            apex.root_hairs_struct_mass = initial_root_hairs_struct_mass * mass_fraction
            apex.root_hairs_struct_mass_produced = initial_root_hairs_struct_mass_produced * mass_fraction
            apex.living_root_hairs_struct_mass = initial_living_root_hairs_struct_mass * mass_fraction
            apex.living_root_hairs_number = initial_living_root_hairs_number * mass_fraction
            apex.dead_root_hairs_number = initial_dead_root_hairs_number * mass_fraction
            apex.total_root_hairs_number = initial_total_root_hairs_number * mass_fraction
            apex.living_root_hairs_external_surface = initial_living_root_hairs_external_surface * mass_fraction

            apex.hexose_exudation = initial_hexose_exudation * mass_fraction
            apex.hexose_uptake_from_soil = initial_hexose_uptake_from_soil * mass_fraction
            apex.phloem_hexose_exudation = initial_phloem_hexose_exudation * mass_fraction
            apex.phloem_hexose_uptake_from_soil = initial_phloem_hexose_uptake_from_soil * mass_fraction
            apex.mucilage_secretion = initial_mucilage_secretion * mass_fraction
            apex.cells_release = initial_cells_release * mass_fraction
            apex.total_net_rhizodeposition = initial_total_net_rhizodeposition * mass_fraction
            apex.hexose_degradation = initial_hexose_degradation * mass_fraction
            apex.mucilage_degradation = initial_mucilage_degradation * mass_fraction
            apex.cells_degradation = initial_cells_degradation * mass_fraction
            apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
            apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
            apex.hexose_consumption_by_growth_rate = initial_hexose_consumption_by_growth_rate * mass_fraction

            apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
            apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
            apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
            apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

            apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
            apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
            apex.Deficit_hexose_reserve = initial_Deficit_hexose_reserve * mass_fraction
            apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
            apex.Deficit_mucilage_soil = initial_Deficit_mucilage_soil * mass_fraction
            apex.Deficit_cells_soil = initial_Deficit_cells_soil * mass_fraction

            apex.Deficit_sucrose_root_rate = initial_Deficit_sucrose_root_rate * mass_fraction
            apex.Deficit_hexose_root_rate = initial_Deficit_hexose_root_rate * mass_fraction
            apex.Deficit_hexose_reserve_rate = initial_Deficit_hexose_reserve_rate * mass_fraction
            apex.Deficit_hexose_soil_rate = initial_Deficit_hexose_soil_rate * mass_fraction
            apex.Deficit_mucilage_soil_rate = initial_Deficit_mucilage_soil_rate * mass_fraction
            apex.Deficit_cells_soil_rate = initial_Deficit_cells_soil_rate * mass_fraction

            # CALCULATING THE TIME SINCE ROOT CELLS FORMATION IN THE NEW SEGMENT:
            # We update the time since root cells have formed, based on the elongation rate and on the principle that
            # cells appear at the very end of the root tip, and then age.

            # CASE 1: The first new segment contains a part that was already formed at the previous time step
            if i==1:
                # In this case, the average age of the element is calculated as the mean value between the incremented
                # age of the part that was already formed and has not elongated, and the average age of the new part to
                # complete the length of the segment.
                # The age of the initially-existing part is simply incremented by the time step:
                Age_of_non_elongated_part = apex.actual_time_since_cells_formation + time_step_in_seconds
                # The age of the cells at the top of the elongated part is by definition equal to:
                Age_of_elongated_part_up = apex.actual_elongation / apex.actual_elongation_rate
                # The age of the cells at the bottom of the new segment is defined according to the length below it:
                Age_of_elongated_part_down = (final_apex_length + (n_segments - 1) * param.segment_length) \
                                             / apex.actual_elongation_rate
                # The age of the elongated part of the new segment is calculated as the mean value between both extremities:
                Age_of_elongated_part = 0.5 * (Age_of_elongated_part_down + Age_of_elongated_part_up)
                # Eventually, the average age of the whole new segment is calculated from the age of both parts:
                apex.actual_time_since_cells_formation = (apex.initial_length * Age_of_non_elongated_part \
                                                          + (apex.length-apex.initial_length) * Age_of_elongated_part) \
                                                         / apex.length
                # We also calculate the thermal age of root cells according to the previous thermal time of initially-exisiting cells:
                Thermal_age_of_non_elongated_part = apex.thermal_time_since_cells_formation + time_step_in_seconds * temperature_time_adjustment
                Thermal_age_of_elongated_part = Age_of_elongated_part * temperature_time_adjustment
                apex.thermal_time_since_cells_formation = (apex.initial_length * Thermal_age_of_non_elongated_part \
                                                          + (apex.length-apex.initial_length) * Thermal_age_of_elongated_part) / apex.length

            # CASE 2: The new segment is only made of new root cells formed during this time step
            else:
                # The age of the cells at the top of the segment is defined according to the elongated length up to it:
                Age_up = (final_apex_length + (n_segments - i) * param.segment_length) / apex.actual_elongation_rate
                # The age of the cells at the bottom of the segment is defined according to the elongated length up to it:
                Age_down = (final_apex_length + (n_segments -1 - i) * param.segment_length) / apex.actual_elongation_rate
                apex.actual_time_since_cells_formation = 0.5 * (Age_down + Age_up)
                apex.thermal_time_since_cells_formation = apex.actual_time_since_cells_formation * temperature_time_adjustment

            # CONSIDERING POSSIBLE PRIMORDIUM FORMATION:
            # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
            new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                                 time_step_in_seconds=time_step_in_seconds,
                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                                                 random=random,
                                                 root_order_limitation=root_order_limitation,
                                                 root_order_treshold=root_order_treshold,
                                                 simple_growth_duration=simple_growth_duration))

            # The current element that has been elongated up to segment_length is now considered as a segment:
            apex.label = 'Segment'

            # If the segment is not the last one on the elongated axis:
            if i < n_segments:
                # Then we also add a new element, initially of length 0, which we call "apex" and which will correspond
                # to the next segment to be defined in the loop:
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
            else:
                # Otherwise, the loop will stop now, and we will add the terminal apex hereafter.
                # NODULE OPTION:
                # We add the possibility of a nodule formation on the segment that is closest to the apex:
                if nodules and len(apex.children()) < 2 and np.random.random() < param.nodule_formation_probability:
                    nodule_formation(g, mother_element=apex)  # WATCH OUT: here, "apex" still corresponds to the last segment!

        # FORMATION OF THE TERMINAL APEX:
        # And we define the new, final apex after the last defined segment, with a new length defined as:
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
        apex.volume = volume_and_external_surface_from_radius_and_length(g, apex, apex.radius, apex.length)["volume"]
        apex.struct_mass = apex.volume * param.root_tissue_density
        # We modify the variables representing total amounts according to the new struct_mass:
        mass_fraction = apex.struct_mass / initial_struct_mass

        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction

        apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
        apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction
        apex.initial_external_surface = initial_initial_external_surface * mass_fraction
        apex.initial_living_root_hairs_external_surface = initial_initial_living_root_hairs_external_surface * mass_fraction

        apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction

        apex.root_hairs_struct_mass = initial_root_hairs_struct_mass * mass_fraction
        apex.root_hairs_struct_mass_produced = initial_root_hairs_struct_mass_produced * mass_fraction
        apex.living_root_hairs_struct_mass = initial_living_root_hairs_struct_mass * mass_fraction
        apex.living_root_hairs_number = initial_living_root_hairs_number * mass_fraction
        apex.dead_root_hairs_number = initial_dead_root_hairs_number * mass_fraction
        apex.total_root_hairs_number = initial_total_root_hairs_number * mass_fraction
        apex.living_root_hairs_external_surface = initial_living_root_hairs_external_surface * mass_fraction

        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake_from_soil = initial_hexose_uptake_from_soil * mass_fraction
        apex.phloem_hexose_exudation = initial_phloem_hexose_exudation * mass_fraction
        apex.phloem_hexose_uptake_from_soil = initial_phloem_hexose_uptake_from_soil * mass_fraction
        apex.mucilage_secretion = initial_mucilage_secretion * mass_fraction
        apex.cells_release = initial_cells_release * mass_fraction
        apex.total_net_rhizodeposition = initial_total_net_rhizodeposition * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.mucilage_degradation = initial_mucilage_degradation * mass_fraction
        apex.cells_degradation = initial_cells_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
        apex.hexose_consumption_by_growth_rate = initial_hexose_consumption_by_growth_rate * mass_fraction
        apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
        apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
        apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
        apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_reserve = initial_Deficit_hexose_reserve * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        apex.Deficit_mucilage_soil = initial_Deficit_mucilage_soil * mass_fraction
        apex.Deficit_cells_soil = initial_Deficit_cells_soil * mass_fraction

        apex.Deficit_sucrose_root_rate = initial_Deficit_sucrose_root_rate * mass_fraction
        apex.Deficit_hexose_root_rate = initial_Deficit_hexose_root_rate * mass_fraction
        apex.Deficit_hexose_reserve_rate = initial_Deficit_hexose_reserve_rate * mass_fraction
        apex.Deficit_hexose_soil_rate = initial_Deficit_hexose_soil_rate * mass_fraction
        apex.Deficit_mucilage_soil_rate = initial_Deficit_mucilage_soil_rate * mass_fraction
        apex.Deficit_cells_soil_rate = initial_Deficit_cells_soil_rate * mass_fraction

        # CALCULATING THE TIME SINCE ROOT CELLS FORMATION IN THE NEW SEGMENT:
        # We update the time since root cells have formed, based on the elongation rate:
        Age_up = final_apex_length / apex.actual_elongation_rate
        Age_down = 0
        apex.actual_time_since_cells_formation = 0.5 * (Age_down + Age_up)
        apex.thermal_time_since_cells_formation = apex.actual_time_since_cells_formation * temperature_time_adjustment

        # And we call the function primordium_formation to check whether a primordium should have been formed:
        new_apex.append(primordium_formation(g, apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                                             root_order_limitation=root_order_limitation,
                                             root_order_treshold=root_order_treshold,
                                             simple_growth_duration=simple_growth_duration))

        # Finally, we add the last apex present at the end of the elongated axis:
        new_apex.append(apex)

    return new_apex


# Simulation of segmentation and primordia formation for all root elements:
#--------------------------------------------------------------------------
# We define a class "Simulate_segmentation_and_primordia_formation" which is used to simulate the segmentation of apices
# and the apparition of primordium for a given MTG:
class Simulate_segmentation_and_primordia_formation(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex":
        self._apices = [g.node(v) for v in g.vertices_iter(scale=1) if g.label(v) == 'Apex']

    def step(self, time_step_in_seconds, soil_temperature_in_Celsius=20, simple_growth_duration=True, random=True,
             nodules=False, root_order_limitation=False, root_order_treshold=2):
        # We define "apices_list" as the list of all apices in g:
        apices_list = list(self._apices)
        # For each apex in the list of apices that have emerged with a positive length:
        for apex in apices_list:
            if apex.type == "Normal_root_after_emergence" and apex.length > 0.:
                # We define the new list of apices with the function apex_development:
                new_apex = segmentation_and_primordium_formation(self.g,
                                                                 apex,
                                                                 time_step_in_seconds=time_step_in_seconds,
                                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                 simple_growth_duration=simple_growth_duration,
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
                                         simple_growth_duration=True,
                                         random=True, printing_warnings=False,
                                         nodules=False,
                                         root_order_limitation=False,
                                         root_order_treshold=2):
    """
    This function considers segmentation and primordia formation across the whole root MTG.
    :param g: the root MTG to be considered
    :param time_step_in_seconds: the time step over which segmentation may have occured
    :param soil_temperature_in_Celsius: the same, homogeneous temperature experienced by the whole root system
    :param random: a Boolean (True/False) expliciting whether random orientations can be defined for the new elements
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :param nodules: a Boolean (True/False) expliciting whether nodules could be formed or not
    :param root_order_limitation: a Boolean (True/False) expliciting whether lateral roots should be prevented above a certain root order
    :param root_order_treshold: the root order above which new lateral roots cannot be formed
    :return:
    """
    # We simulate the segmentation of all apices:
    simulator = Simulate_segmentation_and_primordia_formation(g)
    simulator.step(time_step_in_seconds, soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                   simple_growth_duration=simple_growth_duration, random=random, nodules=nodules,
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
    emergence of lateral roots, and the cost of growth in terms of hexose consumption.
    :param g: the root MTG to be considered
    :param time_step_in_seconds: the time step over which growth is considered
    :param soil_temperature_in_Celsius: the same, homogeneous temperature experienced by the whole root system
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: g, the updated MTG
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
        initial_volume = volume_and_external_surface_from_radius_and_length(g, n, n.initial_radius, n.initial_length)["volume"]
        # We calculate the potential volume of the element based on the potential radius and potential length:
        potential_volume = volume_and_external_surface_from_radius_and_length(g, n, n.potential_radius, n.potential_length)["volume"]
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
        elif n.hexose_growth_demand == 0.:
            continue

        # CALCULATIONS OF THE AMOUNT OF HEXOSE AVAILABLE FOR GROWTH:
        # ---------------------------------------------------------

        # We initialize each amount of hexose available for growth:
        hexose_available_for_elongation = 0.
        hexose_available_for_thickening = 0.

        # If elongation is possible:
        if n.potential_length > n.length:
            # NEW - We calculate the actual amount of hexose that can be used for growth according to the treshold
            # concentration below which no growth process is authorized:
            hexose_available_for_elongation \
                = n.hexose_possibly_required_for_elongation \
                  - n.struct_mass_contributing_to_elongation * param.C_hexose_min_for_elongation
            if hexose_available_for_elongation <0.:
                hexose_available_for_elongation = 0.
            list_of_elongation_supporting_elements = n.list_of_elongation_supporting_elements
            list_of_elongation_supporting_elements_hexose = n.list_of_elongation_supporting_elements_hexose
            list_of_elongation_supporting_elements_mass = n.list_of_elongation_supporting_elements_mass

        # If radial growth is possible:
        if n.potential_radius > n.radius:
            # We only consider the amount of hexose immediately available in the element that can increase in radius.
            # NEW - We calculate the actual amount of hexose that can be used for thickening according to the treshold
            # concentration below which no thickening is authorized:
            hexose_available_for_thickening = n.hexose_available_for_thickening \
                                              - n.struct_mass * param.C_hexose_min_for_thickening
            if hexose_available_for_thickening <0.:
                hexose_available_for_thickening = 0.

        # In case no hexose is available at all:
        if (hexose_available_for_elongation + hexose_available_for_thickening) <= 0.:
            # Then we move to the next element in the main loop:
            continue

        # We initialize the temporary variable "remaining_hexose" that computes the amount of hexose left for growth:
        remaining_hexose_for_elongation = hexose_available_for_elongation
        remaining_hexose_for_thickening = hexose_available_for_thickening

        # ACTUAL ELONGATION IS FIRST CONSIDERED:
        # ---------------------------------------

        # We calculate the maximal possible length of the root element according to all the hexose available for elongation:
        volume_max = initial_volume + hexose_available_for_elongation * 6. \
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
                    # of element n and the relative contribution of the current element to the pool of the potentially available hexose:
                    hexose_actual_contribution_to_elongation = hexose_consumption_by_elongation \
                                                               * list_of_elongation_supporting_elements_hexose[i] \
                                                               / n.hexose_possibly_required_for_elongation
                    # The amount of hexose used for growth in this element is increased:
                    supplying_element.hexose_consumption_by_growth += hexose_actual_contribution_to_elongation
                    supplying_element.hexose_consumption_by_growth_rate += hexose_actual_contribution_to_elongation / time_step_in_seconds
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
            volume_max = volume_and_external_surface_from_radius_and_length(g, n, n.initial_radius, n.length)["volume"] \
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
                print("---- Radial growth has been limited by the availability of hexose!")
            else:
                # Otherwise, radial growth is done up to the full potential and the remaining hexose is calculated:
                n.radius = n.potential_radius
                net_increase_in_volume = volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)["volume"] \
                                         - volume_and_external_surface_from_radius_and_length(g, n, n.initial_radius, n.length)["volume"]
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
            n.hexose_consumption_by_growth_rate += \
                (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) /time_step_in_seconds
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
                parent.hexose_consumption_by_growth_rate += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)/time_step_in_seconds
                # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                parent.resp_growth += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) \
                    * (1 - param.yield_growth) * 6.

        # RECORDING THE ACTUAL STRUCTURAL MODIFICATIONS:
        # -----------------------------------------------

        # The new volume and surfaces of the element is automatically calculated:
        n.external_surface = volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)["external_surface"]
        n.volume = volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)["volume"]
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
            n.external_surface = n.initial_external_surface
            n.volume = initial_volume

        # MODIFYING SPECIFIC PROPERTIES AFTER ELONGATION:
        # -----------------------------------------------
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
                # if n.type == "Adventitious_root_before_emergence":
                #     print("> A new adventitious root has emerged, starting from element", n.index(), "!")
            elif n.type == "Normal_root_after_emergence":
                # The actual elongation rate is calculated:
                n.actual_elongation = n.length - n.initial_length
                n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

            # # IF THE AGE OF CELLS NEEDS TO BE UPDATED AT THIS STAGE:
            # # In both cases, we calculate a new, average age of the root cells in the elongated apex, based on the
            # # principle that cells appear at the very end of the root tip, and then age. In this case, the average age
            # # of the element that has elongated is calculated as the mean value between the incremented age of the part
            # # that was already formed and the average age of the new part that has been created:
            # initial_time_since_cells_formation = n.actual_time_since_cells_formation
            # Age_non_elongated = initial_time_since_cells_formation + time_step_in_seconds
            # Age_elongated =  0.5 * n.actual_elongation / n.actual_elongation_rate
            # n.actual_time_since_cells_formation = (n.initial_length * Age_non_elongated + n.actual_elongation * Age_elongated) / n.length
            # n.thermal_time_since_cells_formation += (n.actual_time_since_cells_formation - initial_time_since_cells_formation) * temperature_time_adjustment

            # The distance to the last ramification is increased:
            n.dist_to_ramif += n.actual_elongation

    return g


# Function calculating a satisfaction coefficient for the growth of the whole root system:
# -----------------------------------------------------------------------------------------
def satisfaction_coefficient(g, struct_mass_input):
    """
    This function computes a general "satisfaction coefficient" SC for the whole root system according to ArchiSimple
    rules, i.e. it compares the available C for root growth and the need for C associated to the potential growth of all
    root elements. If SC >1, there won't be any growth limitation by C, otherwise, the growth of each element will be
    reduced proportionally to SC.
    :param g: the root MTG to be considered
    :param struct_mass_input: the available input of "biomass" to be used for growth
    :return: the satisfaction coefficient SC
    """
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
        initial_volume = volume_and_external_surface_from_radius_and_length(g, n, n.initial_radius, n.initial_length)["volume"]
        # We calculate the potential volume of the element based on the potential radius and potential length:
        potential_volume = volume_and_external_surface_from_radius_and_length(g, n, n.potential_radius, n.potential_length)["volume"]

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
# ------------------------------------------------------------------------------------------------------------------
def ArchiSimple_growth(g, SC, time_step_in_seconds, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the growth of the root system according to ArchiSimple's rules.
    :param g: the root MTG to be considered
    :param SC: the satisfaction coefficient (i.e. the ratio of C offer and C demand)
    :param time_step_in_seconds: the time step over which growth is considered
    :param soil_temperature_in_Celsius: the same, homogeneous temperature experienced by the whole root system
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: g, the updated root MTG
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

        # We initialize the initial volume of the root element:
        initial_volume = n.volume

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
        n.volume = volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)["volume"]
        # The new dry structural struct_mass of the element is calculated from its new volume:
        n.struct_mass = n.volume * param.root_tissue_density
        n.struct_mass_produced = (n.volume - initial_volume) * param.root_tissue_density

        # In case where the root element corresponds to an apex, the distance to the last ramification is increased:
        if n.label == "Apex":
            n.dist_to_ramif += n.actual_elongation

        # VERIFICATION:
        if n.length < 0 or n.struct_mass < 0:
            print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length,
                  "and a mass of", n.struct_mass)

    return g


# Function for reinitializing all growth-related variables at the beginning or end of a time step:
# -------------------------------------------------------------------------------------------------
def reinitializing_growth_variables(g):
    """
    This function re-initializes different growth-related variables (e.g. potential growth variables).
    :param g: the root MTG to be considered
    :return: [the MTG has been updated with new values for the growth-related variables]
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We set to 0 the growth-related variables:
        n.hexose_consumption_by_growth = 0.
        n.hexose_consumption_by_growth_rate = 0.
        n.hexose_possibly_required_for_elongation = 0.
        n.resp_growth = 0.
        n.struct_mass_produced = 0.
        n.root_hairs_struct_mass_produced = 0.
        n.hexose_growth_demand = 0.
        n.actual_elongation = 0.
        n.actual_elongation_rate = 0.

        n.hexose_consumption_rate_by_fungus = 0.

        # We make sure that the initial values of length, radius and struct_mass are correctly initialized:
        n.initial_length = n.length
        n.initial_radius = n.radius
        n.potential_radius = n.radius
        n.theoretical_radius = n.radius
        n.initial_struct_mass = n.struct_mass
        n.initial_living_root_hairs_struct_mass = n.living_root_hairs_struct_mass
        n.initial_external_surface = n.external_surface
        n.initial_living_root_hairs_external_surface = n.living_root_hairs_external_surface

    return


# Root hairs dynamics:
# --------------------
def root_hairs_dynamics(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                        soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the evolution of the density and average length of root hairs along each root,
    and specifies which hairs are alive or dead.
    :param g: the root MTG to be considered
    :param time_step_in_seconds: the time step over which growth is considered
    :param soil_temperature_in_Celsius: the same, homogeneous temperature experienced by the whole root system
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return:
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue

        # We also exclude nodules and dead elements from this computation:
        if n.type == "Just_dead" or n.type == "Dead" or n.type == "Nodule":
            continue

        # # WE ALSO AVOID ROOT APICES - EVEN IF IN THEORY ROOT HAIRS MAY ALSO APPEAR ON THEM:
        # if n.label == "Apex":
        #     continue
        # # Even if root hairs should have already emerge on that root apex, they will appear in the next step (or in a few steps)
        # # when the element becomes a segment.

        # We calculate the equivalent of a thermal time for the current time step:
        temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                               process_at_T_ref=1,
                                                               T_ref=param.root_growth_T_ref,
                                                               A=param.root_growth_A,
                                                               B=param.root_growth_B,
                                                               C=param.root_growth_C)
        elapsed_thermal_time = time_step_in_seconds * temperature_time_adjustment

        # We keep in memory the initial total mass of root hairs (possibly including dead hairs):
        initial_root_hairs_struct_mass = n.root_hairs_struct_mass

        # We calculate the total number of (newly formed) root hairs (if any) and update their age:
        # ------------------------------------------------------------------------------------------
        # CASE 1 - If the current element is completely included within the actual growing zone of the root at the root
        # tip, the root hairs cannot have formed yet:
        if n.distance_from_tip <= param.growing_zone_factor * n.radius:
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
        elif n.distance_from_tip - n.length < param.growing_zone_factor * n.radius:
            # We first record the previous length of the root hair zone within the element:
            initial_length_with_hairs = n.actual_length_with_hairs
            # Then the new length of the root hair zone is calculated:
            n.actual_length_with_hairs = n.distance_from_tip - param.growing_zone_factor * n.radius
            net_increase_in_root_hairs_length = n.actual_length_with_hairs - initial_length_with_hairs
            # The corresponding number of root hairs is calculated:
            n.total_root_hairs_number = param.root_hairs_density * n.radius * n.actual_length_with_hairs
            # The time since root hair formation started is then calculated, using the recent increase in the length
            # of the current root hair zone and the elongation rate of the corresponding root tip. The latter is
            # calculated using the difference between the new distance_from_tip of the element and the previous one:
            elongation_rate_in_actual_time = (n.distance_from_tip - n.former_distance_from_tip) / time_step_in_seconds
            elongation_rate_in_thermal_time = (n.distance_from_tip - n.former_distance_from_tip) / elapsed_thermal_time
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
            n.total_root_hairs_number = param.root_hairs_density * n.radius * n.length
            # The elongation of the corresponding root tip is calculated as the difference between the new
            # distance_from_tip of the element and the previous one:
            elongation_rate_in_actual_time = (n.distance_from_tip - n.former_distance_from_tip) / time_step_in_seconds
            elongation_rate_in_thermal_time = (n.distance_from_tip - n.former_distance_from_tip) / elapsed_thermal_time
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
        # -----------------------------------------------------------
        # Root hairs are dying when the time since they emerged is higher than their lifespan.
        # If the time since root hairs emergence started is lower than the lifespan,
        # no root hair should be dead:
        if n.thermal_time_since_root_hairs_emergence_started <= n.root_hairs_lifespan:
            n.dead_root_hairs_number = 0.
        # Otherwise, if the time since root hairs emergence stopped is higher than the lifespan:
        elif n.thermal_time_since_root_hairs_emergence_stopped > n.root_hairs_lifespan:
            # Then all the root hairs of the root element must now be dead:
            n.dead_root_hairs_number = n.total_root_hairs_number
        # In the intermediate case, there are currently both dead and living root hairs on the root element:
        else:
            # We assume that there is a linear decrease of root hair age between the first hair that has emerged
            # and the last one that has emerged:
            time_since_first_death = n.thermal_time_since_root_hairs_emergence_started - n.root_hairs_lifespan
            dead_fraction = time_since_first_death / (n.thermal_time_since_root_hairs_emergence_started
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
        # ----------------------------------------------------------------------------------------------
        # In the calculation of surface, we consider the root hair to be a cylinder, and include the lateral section,
        # but exclude the section of the cylinder at the tip:
        n.root_hairs_external_surface = ((param.root_hair_radius * 2 * pi) * n.root_hair_length) \
                                        * n.total_root_hairs_number
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

        # We calculate the mass of hairs that has been effectively produced, including from root hairs that may have died since then:
        #----------------------------------------------------------------------------------------------------------------------------
        # We calculate the new production as the difference between initial and final mass:
        n.root_hairs_struct_mass_produced = n.root_hairs_struct_mass - initial_root_hairs_struct_mass

        # We add the cost of producing the new living root hairs (if any) to the hexose consumption by growth:
        hexose_consumption = n.root_hairs_struct_mass_produced * param.struct_mass_C_content / param.yield_growth / 6.
        n.hexose_consumption_by_growth += hexose_consumption
        n.hexose_consumption_by_growth_rate += hexose_consumption / time_step_in_seconds
        n.resp_growth += hexose_consumption * 6. * (1 - param.yield_growth)

    return


# Formation of root nodules:
# ---------------------------
def nodule_formation(g, mother_element):
    """
    This function simulates the formation of one nodule on a root mother element. The nodule is considered as a special
    lateral root segment that has no apex connected and which cannot generate root primordium.
    :param g: the root MTG to be considered
    :param mother_element: the mother element on which a new nodule element will be formed
    :return:
    """

    # We add a lateral root element called "nodule" on the mother element:
    nodule = ADDING_A_CHILD(mother_element, edge_type='+', label='Segment', type='Root_nodule',
                            root_order=mother_element.root_order + 1,
                            angle_down=90, angle_roll=0, length=0, radius=0,
                            identical_properties=False, nil_properties=True)
    nodule.type = "Root_nodule"
    # nodule.length=mother_element.radius
    # nodule.radius=mother_element.radius/10.
    nodule.length = mother_element.radius
    nodule.radius = mother_element.radius
    nodule.original_radius = nodule.radius
    dict = volume_and_external_surface_from_radius_and_length(g, element=nodule, radius=nodule.radius, length=nodule.length)
    nodule.external_surface = dict['external_surface']
    nodule.volume = dict['volume']
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
    and the total dry structural mass of the root system (in gram of dry structural mass).
    :param g: the investigated MTG
    :return: total_sucrose_root (mol of sucrose), total_struct_mass (gram of dry structural mass)
    """

    # We initialize the values to 0:
    total_sucrose_root = 0.
    total_living_struct_mass = 0.

    # We cover all the vertices in the MTG, whether they are dead or not:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        if n.length <=0.:
            continue
        else:
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
    AFTER supplying sucrose from the shoot.
    :param g: the root MTG to be considered
    :param sucrose_input_rate: the rate of sucrose supply from the shoots into the roots (mol of sucrose per second)
    :param time_step_in_seconds: the time step over which the supply is considered
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: g, the updated MTG
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
        n.Deficit_sucrose_root_rate = 0.

    return g

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "EXCHANGE BETWEEN SUCROSE AND HEXOSE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def exchange_with_phloem_rate(g, n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function simulates the rate of sucrose unloading from phloem and its immediate conversion into hexose.
    The hexose pool is assumed to correspond to the symplastic compartment of root cells from the stelar and cortical
    parenchyma and from the epidermis. The function also simulates the opposite process of sucrose loading,
    considering that 2 mol of hexose are produced for 1 mol of sucrose.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    #--------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    length = n.length
    struct_mass = n.struct_mass
    original_radius = n.original_radius
    distance_from_tip = n.distance_from_tip
    C_sucrose_root = n.C_sucrose_root
    C_hexose_root = n.C_hexose_root
    hexose_consumption_by_growth_rate = n.hexose_consumption_by_growth_rate

    # We also calculate the relevant surface of the element:
    exchange_surface = n.phloem_surface

    # We initialize the rates that will be computed:
    hexose_production_from_phloem_rate = 0.
    sucrose_loading_in_phloem_rate  = 0.

    # We check whether unloading should be described by a diffusion process or a Michaelis-Menten kinetics:
    unloading_by_diffusion = param.unloading_by_diffusion

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    #------------------------------------------
    # We initialize the possibility of exchange with phloem:
    possible_exchange_with_phloem = True
    # We consider all the cases where no net exchange should be allowed:
    if length <= 0. or exchange_surface <=0. or type == "Just_dead" or type == "Dead":
        possible_exchange_with_phloem = False

    if possible_exchange_with_phloem:

        # SUCROSE UNLOADING:
        #-------------------
        # We initially assume that phloem unloading capacity is identical everywhere along the root:
        if unloading_by_diffusion:
            phloem_permeability = param.phloem_permeability
        else:
            max_unloading_rate = param.max_unloading_rate

        # We initialize a second condition:
        unloading_allowed = True

        # We forbid any sucrose unloading if there is already a global deficit of sucrose:
        global_sucrose_deficit = g.property('global_sucrose_deficit')[g.root]
        if global_sucrose_deficit > 0.:
            if unloading_by_diffusion:
                phloem_permeability = 0.
                unloading_allowed = False
            else:
                max_unloading_rate = 0.
                unloading_allowed = False
            if printing_warnings:
                print("WARNING: No phloem unloading occured for node", n.index(),
                      "because there was a global deficit of sucrose!")

        # We verify that the concentration of sucrose is not negative or inferior to the one of hexose:
        if C_sucrose_root <= C_hexose_root/2.:
            if unloading_by_diffusion:
                # If such, we forbid any sucrose unloading:
                phloem_permeability = 0.
                unloading_allowed = False
            elif C_sucrose_root <= 0.:
                max_unloading_rate = 0.
                unloading_allowed = False
            if printing_warnings:
                print("WARNING: No phloem unloading occured for node", n.index(),
                      "because root sucrose concentration was", n.C_sucrose_root, "mol/g.")

        # ADJUSTING THE UNLOADING OF PHLOEM ALONG THE ROOT:
        # We now correct the phloem unloading capacity according to the amount of hexose that has been used for growth:
        if unloading_allowed:
            # We use a reference value for the rate of hexose consumption by growth, independent from mass or surface.
            # The unloading capacity will then linearily increase with the rate of hexose consumption for growth, knowing 
            # that if the actual consumption rate equals the reference value, then the unloading capacity is doubled.
            reference_consumption_rate = param.reference_rate_of_hexose_consumption_by_growth
            if unloading_by_diffusion:
                # print(">>> Before adjustement, unloading with permeability is:", phloem_permeability * (C_sucrose_root - C_hexose_root / 2.))
                phloem_permeability = phloem_permeability \
                                      * (1 + hexose_consumption_by_growth_rate / reference_consumption_rate)
                # Note: As the permeability relates to phloem surface, and therefore to the size of the element,
                # the calculation above will result in a total increase of the permeability (gDW per second) according to
                # the actual, total consumption rate for growth (mol per second). There is therefore no need to relate the
                # reference rate to the mass, length or surface of the element.
                # print(">>> After adjustement, unloading with permeability is:", phloem_permeability * (C_sucrose_root - C_hexose_root / 2.))
            else:
                # print(">>> Before adjustement, unloading without permeability is:", max_unloading_rate)
                max_unloading_rate = max_unloading_rate \
                                      * (1 + hexose_consumption_by_growth_rate / reference_consumption_rate)
                # print(">>> After adjustement, unloading without permeability is:", max_unloading_rate)

            # We now correct the unloading rate according to soil temperature:
            temperature_modifier = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                            process_at_T_ref=1,
                                                            T_ref=param.phloem_unloading_T_ref,
                                                            A=param.phloem_unloading_A,
                                                            B=param.phloem_unloading_B,
                                                            C=param.phloem_unloading_C)
            if unloading_by_diffusion:
                phloem_permeability = phloem_permeability * temperature_modifier
            else:
                max_unloading_rate = max_unloading_rate * temperature_modifier

            # Eventually, we compute the unloading rate and the automatical conversion into hexose:
            if unloading_by_diffusion:
                hexose_production_from_phloem_rate = 2. * phloem_permeability * (C_sucrose_root - C_hexose_root / 2.) \
                                                     * exchange_surface
                # print("With diffusion, the overall phloem unloading rate is", phloem_permeability * (C_sucrose_root - C_hexose_root / 2.))
            else:
                hexose_production_from_phloem_rate = 2. * max_unloading_rate * C_sucrose_root * exchange_surface \
                                                     / (param.Km_unloading + C_sucrose_root )
                # print("Without diffusion, the overall phloem unloading rate is", max_unloading_rate * C_sucrose_root / (param.Km_unloading + C_sucrose_root ))

        # AVOIDING PROBLEMS - We make sure that hexose production can't become negative:
        if hexose_production_from_phloem_rate < 0.:
                print("!!!ERROR!!!  A negative sucrose unloading rate was computed for element", n.index(),
                      "; we therefore set the loading rate to 0!")
                hexose_production_from_phloem_rate = 0.

        # SUCROSE LOADING:
        # ----------------
<<<<<<< HEAD
        # # We correct the max loading rate according to the distance from the tip in the middle of the segment.
        # max_loading_rate = param.surfacic_loading_rate_reference \
        #     * (1. - 1. / (1. + ((distance_from_tip-length/2.) / original_radius) ** param.gamma_loading))
        max_loading_rate = param.max_loading_rate
=======
>>>>>>> 1bf87af11d8a55bd3ec614e424a06656216f58ba

        max_loading_rate = param.max_loading_rate
        # We correct loading according to soil temperature:
        max_loading_rate = max_loading_rate \
                           * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                      process_at_T_ref=1,
                                                      T_ref=param.max_loading_rate_T_ref,
                                                      A=param.max_loading_rate_A,
                                                      B=param.max_loading_rate_B,
                                                      C=param.max_loading_rate_C)

        # We verify that the concentration of hexose in root is not nil or negative:
        if C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No phloem loading occured for node", n.index(),
                      "because root hexose concentration was", C_hexose_root,
                      "mol/g.")
            max_loading_rate = 0.

        # If there is a demand for growth, we set the loading rate to 0:
        if hexose_consumption_by_growth_rate > 0.:
            max_loading_rate  = 0.

        # We calculate the potential production of sucrose from root hexose (in mol) according to the Michaelis-Menten function:
        sucrose_loading_in_phloem_rate = 0.5 * max_loading_rate * exchange_surface * C_hexose_root \
                                      / (param.Km_loading + C_hexose_root)

        # AVOIDING PROBLEMS - We make sure that hexose production can't become negative:
        if sucrose_loading_in_phloem_rate < 0.:
            print("!!!ERROR!!!  A negative sucrose loading rate was computed for element", n.index(),
                  "; we therefore set the loading rate to 0!")
            sucrose_loading_in_phloem_rate = 0.

    # RECORDING THE RESULTS:
    #-----------------------
    # Eventually, we record all new values  in the element n, including the difference between unloading and loading:
    n.hexose_production_from_phloem_rate = hexose_production_from_phloem_rate
    n.sucrose_loading_in_phloem_rate = sucrose_loading_in_phloem_rate
    n.net_sucrose_unloading_rate = hexose_production_from_phloem_rate / 2. - sucrose_loading_in_phloem_rate

    return n

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "EXCHANGE BETWEEN MOBILE HEXOSE AND RESERVE"
########################################################################################################################
########################################################################################################################

# Net exchange between labile hexose and reserve pool:
# --------------------------------------------------------------------------
def exchange_with_reserve_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function simulates the rate of hexose immobilization into the reserve pool of a given root element n,
    and the rate of hexose mobilization from this reserve.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    length = n.length
    C_hexose_root = n.C_hexose_root
    C_hexose_reserve = n.C_hexose_reserve
    struct_mass = n.struct_mass
    living_root_hairs_struct_mass = n.living_root_hairs_struct_mass

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rates:
    hexose_mobilization_from_reserve_rate = 0.
    hexose_immobilization_as_reserve_rate = 0.

    # We initialize the possibility of exchange with reserve:
    possible_exchange_with_reserve = True

    # We verify that the element does not correspond to a primordium that has not emerged:
    if length <= 0.:
        possible_exchange_with_reserve = False
    # We verify that the concentration of hexose are not negative:
    if C_hexose_root < 0. or C_hexose_reserve < 0.:
        if printing_warnings:
            print("WARNING: No exchange with reserve occurred for node", n.index(),
                  "because root sucrose concentration was", C_sucrose_root,
                  "mol/g, root hexose concentration was", C_hexose_root,
                  "mol/g, and hexose reserve concentration was", C_hexose_reserve)
        possible_exchange_with_reserve = False

    # If the element corresponds to a root nodule, we don't consider any reserve pool:
    if type == "Root_nodule":
        possible_exchange_with_reserve = False

    # NOTE: If the element has died, special cases are included below in the calculations.

    if possible_exchange_with_reserve:

        # The maximal concentration in the reserve is defined here:
        C_hexose_reserve_max = param.C_hexose_reserve_max

        # We correct maximum rates according to soil temperature:
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

        # CALCULATIONS OF THEORETICAL MOBILIZATION RATE:
        #-----------------------------------------------
        if C_hexose_reserve <= param.C_hexose_reserve_min:
            hexose_mobilization_from_reserve_rate = 0.
        else:
            # We calculate the potential mobilization of hexose from reserve (in mol) according to the Michaelis-Menten function:
            hexose_mobilization_from_reserve_rate = corrected_max_mobilization_rate * C_hexose_reserve \
                                               / (param.Km_mobilization + C_hexose_reserve) \
                                               * (struct_mass + living_root_hairs_struct_mass)

        # CALCULATIONS OF THEORETICAL IMMOBILIZATION RATE:
        # -----------------------------------------------
        # If the concentration of mobile hexose is already too low or the concentration if the reserve pool too high,
        # or if the element has died:
        if C_hexose_root <= param.C_hexose_root_min_for_reserve or C_hexose_reserve >= param.C_hexose_reserve_max \
                or type == "Just_dead" or type =="Dead":
            # Then there is no possible immobilization in the reserve:
            hexose_immobilization_as_reserve_rate = 0.
        else:
            # We calculate the potential immobilization of hexose as reserve (in mol) according to the
            # Michaelis-Menten function:
            hexose_immobilization_as_reserve_rate = corrected_max_immobilization_rate * C_hexose_root \
                                                 / (param.Km_immobilization + C_hexose_root) \
                                                 * (struct_mass + living_root_hairs_struct_mass)

    # RECORDING THE RESULTS:
    # -----------------------
    # Eventually, we record all new values in the element n, including the net difference of the two opposite rates:
    n.hexose_immobilization_as_reserve_rate = hexose_immobilization_as_reserve_rate
    n.hexose_mobilization_from_reserve_rate = hexose_mobilization_from_reserve_rate
    n.net_hexose_immobilization_rate = hexose_immobilization_as_reserve_rate - hexose_mobilization_from_reserve_rate

    return n

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "ROOT MAINTENANCE"
########################################################################################################################
########################################################################################################################

# Function calculating maintenance respiration:
#----------------------------------------------
def maintenance_respiration_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function calculates the rate of respiration (mol of CO2 per second) corresponding to the consumption
    of a part of the local hexose pool to cover the costs of maintenance processes, i.e. any biological process in the
    root that is NOT linked to the actual growth of the root. The calculation is derived from the model of Thornley and
    Cannell (2000), who initially used this formalism to describe the residual maintenance costs that could not be
    accounted for by known processes. The local amount of CO2 respired for maintenance is calculated relatively
    to the structural dry mass of the element n and is regulated by a Michaelis-Menten function of the local
    concentration of hexose.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    C_hexose_root = n.C_hexose_root
    struct_mass = n.struct_mass
    living_root_hairs_struct_mass = n.living_root_hairs_struct_mass

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate:
    resp_maintenance_rate = 0.

    # We initialize the possibility of maintenance:
    possible_maintenance = True
    # We consider that dead elements cannot respire (unless over the first time step following death,
    # i.e. when the type is "Just_dead"):
    if type == "Dead":
        possible_maintenance = False
    # We also check whether the concentration of hexose in root is positive or not:
    if C_hexose_root <= 0.:
        possible_maintenance = False
        if printing_warnings:
            print("WARNING: No maintenance occurred for node", n.index(),
                  "because root hexose concentration was", C_hexose_root, "mol/g.")

    if possible_maintenance:

        # We correct the maximal respiration rate according to soil temperature:
        corrected_resp_maintenance_max = param.resp_maintenance_max \
                                         * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                    process_at_T_ref=1,
                                                                    T_ref=param.resp_maintenance_max_T_ref,
                                                                    A=param.resp_maintenance_max_A,
                                                                    B=param.resp_maintenance_max_B,
                                                                    C=param.resp_maintenance_max_C)

        # CALCULATIONS OF MAINTENANCE RATE:
        #----------------------------------

        # We calculate the rate of maintenance respiration:
        resp_maintenance_rate = corrected_resp_maintenance_max * C_hexose_root \
                                / (param.Km_maintenance + C_hexose_root) \
                                * (struct_mass + living_root_hairs_struct_mass)

    # RECORDING THE RESULTS:
    # -----------------------
    # Eventually, we record all new values in the element n:
    n.resp_maintenance_rate = resp_maintenance_rate

    return n

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "NET RHIZODEPOSITION"
########################################################################################################################
########################################################################################################################

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def root_sugars_exudation_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False,
                               exudation_at_root_tips_only=False):
    """
    This function computes the rate of hexose exudation (mol of hexose per seconds) for a given root element.
    Exudation corresponds to an efflux of hexose from the root to the soil by a passive diffusion. This efflux
    is calculated from the product of the exchange surface (m2) with the soil solution, a permeability coefficient (g m-2)
    and the gradient of hexose concentration between cells and soil (mol of hexose per gram of dry root structural mass).
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :param exudation_at_root_tips_only: a Boolean (True/False) determining whether exudation is only allowed at root tips
                                        (i.e. within the meristem and root elongation zone) or not
    :return: the updated root element n
    """

    # READING THE VALUES:
    # -------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    length = n.length
    original_radius = n.original_radius
    distance_from_tip = n.distance_from_tip

    C_sucrose_root = n.C_sucrose_root
    C_hexose_root = n.C_hexose_root
    C_hexose_soil = n.C_hexose_soil

    S_epid =  n.epidermis_surface_without_hairs
    S_hairs = n.living_root_hairs_external_surface
    S_cortex = n.cortical_parenchyma_surface
    S_stele = n.stelar_parenchyma_surface
    S_vessels = n.phloem_surface

    cond_walls = n.relative_conductance_walls
    cond_exo = n.relative_conductance_exodermis
    cond_endo = n.relative_conductance_endodermis

    # We calculate the total surface of exchange between symplasm and apoplasm in the root parenchyma, modulated by the 
    # conductance of cell walls (reduced in the meristematic zone) and the conductances of endodermis and exodermis 
    # barriers (when these barriers are mature, conductance is expected to be 0 in general, and part of the symplasm is 
    # not accessible anymore to the soil solution:
    non_vascular_exchange_surface = (S_epid + S_hairs) + cond_walls * (cond_exo * S_cortex + cond_endo * S_stele)
    vascular_exchange_surface = cond_walls * cond_exo * cond_endo * S_vessels

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate and the permeability coefficient of the element n:
    hexose_exudation_rate = 0.
    phloem_hexose_exudation_rate = 0.
    corrected_permeability_coeff = 0.

    # We initialize the possibility of exudation:
    possible_exudation = True
    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0 or non_vascular_exchange_surface <=0.:
        # If not, we forbid exudation:
        possible_exudation = False
    # We check whether the concentration of hexose in root is positive or not:
    if C_hexose_root <= 0.:
        # If not, we forbid exudation:
        possible_exudation = False
        if printing_warnings:
            print("WARNING: No hexose exudation occurred for node", n.index(),
                  "because root hexose concentration was", C_hexose_root, "mol/g.")
    # We also check the possibility that exudation is only allowed at root tips.
    # If the distance from root tip is higher than the prescribed root elongation zone:
    if exudation_at_root_tips_only and n.distance_from_tip > n.radius * param.growing_zone_factor :
        # Then we forbid exudation:
        possible_exudation = False

    if possible_exudation:

        # CALCULATION OF THE PERMEABILITY COEFFICIENT:
        #---------------------------------------------
        # We correct the maximal permeability according to soil temperature:
        corrected_P_max_apex = param.Pmax_apex \
                                       * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                  process_at_T_ref=1,
                                                                  T_ref=param.permeability_coeff_T_ref,
                                                                  A=param.permeability_coeff_A,
                                                                  B=param.permeability_coeff_B,
                                                                  C=param.permeability_coeff_C)

        # # MODIFICATION OF THE PERMEABILITY WITH THE DISTANCE FROM ROOT TIP:
        # # The description of the evolution of P along the root is adapted from Personeni et al. (2007):
        # if distance_from_tip < length:
        #     print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
        # else:
        #     corrected_permeability_coeff = corrected_P_max_apex \
        #                          / (1 + (distance_from_tip - length / 2.) / original_radius) ** param.gamma_exudation
        corrected_permeability_coeff = corrected_P_max_apex

        # CALCULATIONS OF EXUDATION RATE:
        #--------------------------------
        # We calculate the rate of hexose exudation, even for dead root elements:
        hexose_exudation_rate = corrected_permeability_coeff * (C_hexose_root - C_hexose_soil) * non_vascular_exchange_surface
        # NOTE : We consider that dead elements still liberate hexose in the soil, until they are empty.
        if hexose_exudation_rate < 0.:
            if printing_warnings:
                print("WARNING: a negative hexose exudation flux was calculated for the element", n.index(),
                      "; hexose exudation flux has therefore been set to zero!")
            hexose_exudation_rate = 0.

        # We also include the direct exchange between soil solution and phloem vessels, when these are in direct
        # contact with the soil solution (e.g. in the meristem, or if a lateral roots disturbs all the barriers):
        phloem_hexose_exudation_rate = corrected_permeability_coeff * (2 * C_sucrose_root - C_hexose_soil) \
                                       * vascular_exchange_surface

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record all new values in the element n:
    n.hexose_exudation_rate = hexose_exudation_rate
    n.phloem_hexose_exudation_rate = phloem_hexose_exudation_rate
    n.total_exudation_rate = hexose_exudation_rate + phloem_hexose_exudation_rate
    n.permeability_coeff = corrected_permeability_coeff

    return n

# Uptake of hexose from the soil by the root:
# -------------------------------------------
def root_sugars_uptake_from_soil_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the rate of hexose uptake by roots from the soil. This influx of hexose is represented
    as an active process with a substrate-limited relationship (Michaelis-Menten function) depending on the hexose
    concentration in the soil.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    length = n.length
    C_hexose_soil = n.C_hexose_soil

    S_epid = n.epidermis_surface_without_hairs
    S_hairs = n.living_root_hairs_external_surface
    S_cortex = n.cortical_parenchyma_surface
    S_stele = n.stelar_parenchyma_surface
    S_vessels = n.phloem_surface

    cond_walls = n.relative_conductance_walls
    cond_exo = n.relative_conductance_exodermis
    cond_endo = n.relative_conductance_endodermis

    # We calculate the total surface of exchange between symplasm and apoplasm in the root parenchyma, modulated by the
    # conductance of cell walls (reduced in the meristematic zone) and the conductances of endodermis and exodermis
    # barriers (when these barriers are mature, conductance is expected to be 0 in general, and part of the symplasm is
    # not accessible anymore to the soil solution:
    non_vascular_exchange_surface = (S_epid + S_hairs) + cond_walls * (cond_exo * S_cortex + cond_endo * S_stele)
    vascular_exchange_surface = cond_walls * cond_exo * cond_endo * S_vessels

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # -----------------------------------------
    # We initialize the rate of the element n:
    hexose_uptake_from_soil_rate = 0.
    phloem_hexose_uptake_from_soil_rate = 0.

    # We initialize the possibility of uptake:
    possible_uptake = True
    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0 or non_vascular_exchange_surface <= 0.:
        possible_uptake = False
    # We check whether the concentration of hexose in root is positive or not:
    if C_hexose_soil <= 0.:
        possible_uptake = False
        if printing_warnings:
            print("WARNING: No hexose uptake occurred for node", n.index(),
                  "because soil hexose concentration was", C_hexose_soil, "mol/g.")
    # We consider that dead elements cannot take up any hexose from the soil:
    if n.type == "Just_dead" or n.type == "Dead":
        possible_uptake = False

    if possible_uptake:

        # CALCULATION OF THE MAXIMAL UPTAKE:
        # ----------------------------------
        # We correct the maximal uptake rate according to soil temperature:
        corrected_uptake_rate_max = param.uptake_rate_max \
                                    * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                               process_at_T_ref=1,
                                                               T_ref=param.uptake_rate_max_T_ref,
                                                               A=param.uptake_rate_max_A,
                                                               B=param.uptake_rate_max_B,
                                                               C=param.uptake_rate_max_C)

        # CALCULATIONS OF UPTAKE RATE:
        # ----------------------------
        # We calculate the rate of hexose uptake:
        hexose_uptake_from_soil_rate = corrected_uptake_rate_max * non_vascular_exchange_surface \
                             * C_hexose_soil / (param.Km_uptake + C_hexose_soil)

        # NEW: we also include the direct exchange between soil solution and phloem vessels, when these are in direct
        # contact with the soil solution (e.g. in the meristem, or if a lateral roots disturbs all the barriers):
        phloem_hexose_uptake_from_soil_rate = corrected_uptake_rate_max * vascular_exchange_surface \
                             * C_hexose_soil / (param.Km_uptake + C_hexose_soil)
        # NOTE: We consider that the uptake rate though the surface of the phloem vessels in contact with
        # the solution is the same as through the surface of the parenchyma cells where the mobile pool is kept!

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record the new rate:
    n.hexose_uptake_from_soil_rate = hexose_uptake_from_soil_rate
    n.phloem_hexose_uptake_from_soil_rate = phloem_hexose_uptake_from_soil_rate
    n.total_hexose_uptake_from_soil_rate = hexose_uptake_from_soil_rate + phloem_hexose_uptake_from_soil_rate

    return n

# Mucilage secretion:
# ------------------
def mucilage_secretion_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the rate of mucilage secretion (in mol of equivalent hexose per second) for a given root element n.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    length = n.length
    original_radius = n.original_radius
    distance_from_tip = n.distance_from_tip
    C_hexose_root = n.C_hexose_root
    Cs_mucilage_soil = n.Cs_mucilage_soil

    # We calculate the total surface of exchange with the soil for mucilage secretion:
    exchange_surface = n.external_surface + n.living_root_hairs_external_surface

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate of the element n:
    mucilage_secretion_rate = 0.

    # We initialize the possibility of mucilage secretion:
    possible_secretion = True
    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0 or exchange_surface <=0.:
        possible_secretion = False
    # We check whether the concentration of hexose in root is positive or not:
    if C_hexose_root <= 0.:
        possible_secretion = False
        if printing_warnings:
            print("WARNING: No mucilage secretion occurred for node", n.index(),
                  "because root hexose concentration was", C_hexose_root, "mol/g.")
    # We consider that dead elements cannot secrete any mucilage:
    if n.type == "Dead":
        possible_secretion = False
    # We consider that elements from an axis that has stopped growing (e.g. apices) do not secrete any mucilage anymore:
    if n.type == "Stopped":
        possible_secretion = False

    if possible_secretion:

        # CALCULATION OF THE MAXIMAL RATE OF SECRETION:
        #---------------------------------------------
        # We correct the maximal secretion rate according to soil temperature
        # (This could to a bell-shape where the maximum is obtained at 27 degree Celsius,
        # as suggested by Morré et al. (1967) for maize mucilage secretion):
        corrected_secretion_rate_max = param.secretion_rate_max \
                                       * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                  process_at_T_ref=1,
                                                                  T_ref=param.secretion_rate_max_T_ref,
                                                                  A=param.secretion_rate_max_A,
                                                                  B=param.secretion_rate_max_B,
                                                                  C=param.secretion_rate_max_C)

        # We modify the maximal secretion rate according to the distance to the apex, similarly to
        # what has been done for hexose exudation:
        if distance_from_tip < length:
            print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
            corrected_secretion_rate_max = 0.
        else:
            corrected_secretion_rate_max = corrected_secretion_rate_max \
                                           / (1 + (distance_from_tip - length / 2.) / original_radius) ** param.gamma_secretion

        # We also regulate the rate according to the potential accumulation of mucilage around the root:
        # the rate is maximal when no mucilage accumulates around, and linearily decreases with the concentration
        # of mucilage at the soil-root interface, until reaching 0 when the concentration is equal or higher than
        # the maximal concentration soil (NOTE: Cs_mucilage_soil is expressed in mol of equivalent hexose per m2 of
        # external surface):
        corrected_secretion_rate_max = corrected_secretion_rate_max \
                                           * (param.Cs_mucilage_soil_max - Cs_mucilage_soil) / param.Cs_mucilage_soil_max

        # We verify that the rate is not negative:
        if corrected_secretion_rate_max < 0.:
            corrected_secretion_rate_max = 0.

        # CALCULATIONS OF SECRETION RATE:
        #--------------------------------
        # We calculate the rate of secretion according to a Michaelis-Menten formalis:
        mucilage_secretion_rate = corrected_secretion_rate_max * exchange_surface \
                                  * C_hexose_root / (param.Km_secretion + C_hexose_root)

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record all new values in the element n:
    n.mucilage_secretion_rate = mucilage_secretion_rate

    return n

# Release of root cells:
# ----------------------
def cells_release_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the rate of release of epidermal or cap root cells (in mol of equivalent hexose per second)
    into the soil for a given root element. The rate of release linearily decreases with increasing length from the tip,
    until reaching 0 at the end of the elongation zone. The external concentration of root cells (mol of equivalent-
    hexose per m2) is also linearily decreasing the release of cells at the interface.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    type = n.type
    length = n.length
    radius = n.radius
    distance_from_tip = n.distance_from_tip
    C_hexose_root = n.C_hexose_root
    Cs_cells_soil = n.Cs_cells_soil

    # We calculate the total surface of exchange with the soil for mucilage secretion:
    exchange_surface = n.external_surface

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate of the element n:
    cells_release_rate = 0.

    # We initialize the possibility of cells release:
    possible_cells_release = True

    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0 or exchange_surface <=0.:
        possible_cells_release = False

    # Then we also make sure that there is some hexose available in the root to produce the cells:
    if C_hexose_root <=0.:
        possible_cells_degradation = False
        if printing_warnings:
            print("WARNING: No cells release occurred for node", n.index(),
                  "because root hexose concentration was", C_hexose_root, "mol/g.")

    # We consider that dead root elements can't release cells anymore:
    if n.type == "Just_dead" or n.type == "Dead":
        possible_cells_release = False

    # We also consider that if the root axis has stopped elongating, no cells are released anymore:
    if n.type == "Stopped":
        possible_cells_release = False

    if possible_cells_release:

        # CALCULATION OF THE MAXIMAL RATE OF CELLS RELEASE:
        #--------------------------------------------------

        # We modify the maximal surfacic release rate according to the mean distance to the tip (in the middle of the
        # root element), assuming that the release decreases linearily with the distance to the tip, until reaching 0
        # when the this distance becomes higher than the growing zone length:
        if distance_from_tip < param.growing_zone_factor * radius:
            average_distance = distance_from_tip - length / 2.
            reduction = (param.growing_zone_factor * radius - average_distance) \
                        / (param.growing_zone_factor * radius)
            cells_surfacic_release = param.surfacic_cells_release_rate * reduction
        # In the special case where the end of the growing zone is located somewhere on the root element:
        elif distance_from_tip - length < param.growing_zone_factor * radius:
            average_distance = (distance_from_tip - length) + (param.growing_zone_factor * radius
                                                             - (distance_from_tip - length)) / 2.
            reduction = (param.growing_zone_factor * radius - average_distance) \
                        / (param.growing_zone_factor * radius)
            cells_surfacic_release = param.surfacic_cells_release_rate * reduction
        # Otherwise, there is no cells release:
        else:
            cells_surfacic_release = 0.

        # We correct the release rate according to soil temperature:
        corrected_cells_surfacic_release = cells_surfacic_release \
                                           * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                      process_at_T_ref=1,
                                                                      T_ref=param.surfacic_cells_release_rate_T_ref,
                                                                      A=param.surfacic_cells_release_rate_A,
                                                                      B=param.surfacic_cells_release_rate_B,
                                                                      C=param.surfacic_cells_release_rate_C)

        # We also regulate the surface release rate according to the potential accumulation of cells around the root:
        # the rate is maximal when no cells are around, and linearily decreases with the concentration of cells
        # in the soil, until reaching 0 when the concentration is equal or higher than the maximal concentration in the
        # soil (NOTE: Cs_cells_soil is expressed in mol of equivalent hexose per m2 of external surface):
        corrected_cells_surfacic_release = corrected_cells_surfacic_release \
                                           * (param.Cs_cells_soil_max - Cs_cells_soil) / param.Cs_cells_soil_max

        # We verify that cells release is not negative:
        if corrected_cells_surfacic_release < 0.:
            corrected_cells_surfacic_release = 0.

        # CALCULATIONS OF CELLS RELEASE RATE:
        #------------------------------------

        # The release of cells by the root is then calculated according to this surface:
        cells_release_rate = exchange_surface * corrected_cells_surfacic_release

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record all new values in the element n:
    n.cells_release_rate = cells_release_rate

    return n

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "SOIL TRANSFORMATION"
########################################################################################################################
########################################################################################################################

# Degradation of hexose in the soil (microbial consumption):
# ----------------------------------------------------------
def hexose_degradation_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the rate of hexose "consumption" (in mol of hexose per seconds) at the soil-root interface
    for a given root element. It mimics the uptake of hexose by rhizosphere microorganisms, and is therefore described
    using a substrate-limited function (Michaelis-Menten).
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    length = n.length
    C_hexose_soil = n.C_hexose_soil

    S_epid = n.epidermis_surface_without_hairs
    S_hairs = n.living_root_hairs_external_surface
    S_cortex = n.cortical_parenchyma_surface
    S_stele = n.stelar_parenchyma_surface

    cond_walls = n.relative_conductance_walls
    cond_exo = n.relative_conductance_exodermis
    cond_endo = n.relative_conductance_endodermis

    # We calculate the total surface of exchange, identical to that of root hexose exudation and root hexose uptake,
    # assuming that degradation may occur in the apoplasm in contact with soil solution, within the root itself...

    non_vascular_exchange_surface = (S_epid + S_hairs) + cond_walls * (cond_exo * S_cortex + cond_endo * S_stele)

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate of the element n:
    hexose_degradation_rate = 0.

    # We initialize the possibility of hexose degradation:
    possible_hexose_degradation = True
    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0:
        possible_hexose_degradation = False
    # We check whether the concentration of hexose in soil is positive or not:
    if C_hexose_soil <= 0.:
        possible_hexose_degradation = False
        if printing_warnings:
            print("WARNING: No hexose degradation occurred for node", n.index(),
                  "because soil hexose concentration was", C_hexose_soil, "mol/g.")

    if possible_hexose_degradation:

        # CALCULATION OF THE MAXIMAL RATE OF HEXOSE DEGRADATION:
        # ------------------------------------------------------

        # We correct the maximal degradation rate according to soil temperature:
        corrected_hexose_degradation_rate_max = param.hexose_degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.hexose_degradation_rate_max_T_ref,
            A=param.hexose_degradation_rate_max_A,
            B=param.hexose_degradation_rate_max_B,
            C=param.hexose_degradation_rate_max_C)

        # CALCULATIONS OF SOIL HEXOSE DEGRADATION RATE:
        # ---------------------------------------------
        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of hexose
        # in the soil:
        hexose_degradation_rate = corrected_hexose_degradation_rate_max * non_vascular_exchange_surface \
                                       * C_hexose_soil / (param.Km_hexose_degradation + C_hexose_soil)

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record all new values in the element n:
    n.hexose_degradation_rate = hexose_degradation_rate

    return n

# Degradation of mucilage in the soil (microbial consumption):
# ------------------------------------------------------------
def mucilage_degradation_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the rate of mucilage degradation outside the root (in mol of equivalent-hexose per second)
    for a given root element. Only the external surface of the root element is taken into account here, similarly to
    what is done for mucilage secretion.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    length = n.length
    Cs_mucilage_soil = n.Cs_mucilage_soil
    # We calculate the total surface of exchange, identical to that of root mucilage secretion:
    external_surface = n.external_surface
    living_root_hairs_external_surface = n.living_root_hairs_external_surface
    exchange_surface = external_surface + living_root_hairs_external_surface

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate of the element n:
    mucilage_degradation_rate = 0.

    # We initialize the possibility of mucilage degradation:
    possible_mucilage_degradation = True
    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0 or exchange_surface <=0.:
        possible_mucilage_degradation = False
    # We check whether the concentration of mucilage in soil is positive or not:
    if Cs_mucilage_soil <= 0.:
        possible_mucilage_degradation = False
        if printing_warnings:
            print("WARNING: No mucilage degradation occurred for node", n.index(),
                  "because soil mucilage concentration was", Cs_mucilage_soil, "mol/g.")

    if possible_mucilage_degradation:

        # CALCULATION OF THE MAXIMAL RATE OF MUCILAGE DEGRADATION:
        # -------------------------------------------------------

        # We correct the maximal degradation rate according to soil temperature:
        corrected_mucilage_degradation_rate_max = param.mucilage_degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.mucilage_degradation_rate_max_T_ref,
            A=param.mucilage_degradation_rate_max_A,
            B=param.mucilage_degradation_rate_max_B,
            C=param.mucilage_degradation_rate_max_C)

        # CALCULATIONS OF SOIL MUCILAGE DEGRADATION RATE:
        # ---------------------------------------------
        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of mucilage
        # in the soil:
        mucilage_degradation_rate = corrected_mucilage_degradation_rate_max * exchange_surface \
                                       * Cs_mucilage_soil / (param.Km_mucilage_degradation + Cs_mucilage_soil)

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record all new values in the element n:
    n.mucilage_degradation_rate = mucilage_degradation_rate

    return n

# Degradation of root cells released in the soil (microbial consumption):
# -----------------------------------------------------------------------
def cells_degradation_rate(n, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    This function computes the rate of root cells degradation outside the root (in mol of equivalent-hexose per second)
    for a given root element. Only the external surface of the root element is taken into account as the exchange
    surface, similarly to what is done for root cells release.
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :return: the updated root element n
    """

    # READING THE VALUES:
    # --------------------
    # We read the values of interest in the element n only once, so that it's not called too many times:
    length = n.length
    C_hexose_root = n.C_hexose_root
    Cs_cells_soil = n.Cs_cells_soil
    # We calculate the total surface of exchange, identical to that of root cells release:
    exchange_surface = n.external_surface

    # CONSIDERING CASES THAT SHOULD BE AVOIDED:
    # ------------------------------------------
    # We initialize the rate of the element n:
    cells_degradation_rate = 0.

    # We initialize the possibility of cells degradation:
    possible_cells_degradation = True
    # First, we ensure that the element has a positive length and surface of exchange:
    if length <= 0 or exchange_surface <=0.:
        possible_cells_degradation = False
    # We check whether the concentration of cells in soil is positive or not:
    if Cs_cells_soil <= 0.:
        possible_cells_degradation = False
        if printing_warnings:
            print("WARNING: No cells degradation occurred for node", n.index(),
                  "because soil cells concentration was", Cs_cells_soil, "mol/g.")

    if possible_cells_degradation:

        # CALCULATION OF THE MAXIMAL RATE OF CELLS DEGRADATION:
        # -----------------------------------------------------
        # We correct the maximal degradation rate according to soil temperature:
        corrected_cells_degradation_rate_max = param.cells_degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=param.cells_degradation_rate_max_T_ref,
            A=param.cells_degradation_rate_max_A,
            B=param.cells_degradation_rate_max_B,
            C=param.cells_degradation_rate_max_C)

        # CALCULATIONS OF SOIL CELLS DEGRADATION RATE:
        # --------------------------------------------
        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of root cells
        # in the soil:
        cells_degradation_rate = corrected_cells_degradation_rate_max * exchange_surface \
                                       * Cs_cells_soil / (param.Km_cells_degradation + Cs_cells_soil)

    # RECORDING THE RESULTS:
    # ----------------------
    # Eventually, we record all new values in the element n:
    n.cells_degradation_rate = cells_degradation_rate

    return n

########################################################################################################################

########################################################################################################################
########################################################################################################################
# MAIN PROGRAM:
########################################################################################################################
########################################################################################################################


def calculating_all_growth_independent_fluxes(g, n, soil_temperature_in_Celsius, printing_warnings,
                                              exudation_at_root_tips_only=False):
    """
    This function simply calls all the fluxes-related (not growth-dependent) functions of the model and runs them on a 
    given element n.
    :param g: the root MTG to be considered
    :param n: the root element to be considered
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :param exudation_at_root_tips_only: a Boolean (True/False) determining whether exudation is only allowed at root tips
                                        (i.e. within the meristem and root elongation zone) or not
    :return: the updated root element n
    """

    # Unloading of sucrose from phloem and conversion of sucrose into hexose, or reloading of sucrose:
    exchange_with_phloem_rate(g, n, soil_temperature_in_Celsius, printing_warnings)
    # Net immobilization of hexose into the reserve pool:
    exchange_with_reserve_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Maintenance respiration:
    maintenance_respiration_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Transfer of hexose from the root to the soil:
    root_sugars_exudation_rate(n, soil_temperature_in_Celsius, printing_warnings,
                               exudation_at_root_tips_only=exudation_at_root_tips_only)
    # Transfer of hexose from the soil to the root:
    root_sugars_uptake_from_soil_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Consumption of hexose in the soil:
    hexose_degradation_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Secretion of mucilage into the soil:
    mucilage_secretion_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Consumption of mucilage in the soil:
    mucilage_degradation_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Release of root cells into the soil:
    cells_release_rate(n, soil_temperature_in_Celsius, printing_warnings)
    # Consumption of root cells in the soil:
    cells_degradation_rate(n, soil_temperature_in_Celsius, printing_warnings)

    return n

def calculating_amounts_from_rates(n, time_step_in_seconds):
    """
    This function simply integrates the values of several fluxes over a given time step for a given root element n.
    :param n: the root element to be considered
    :param time_step_in_seconds: the time step over which the amounts will be calculated
    :return: the updated root element n
    """

    n.hexose_production_from_phloem = n.hexose_production_from_phloem_rate * time_step_in_seconds
    n.sucrose_loading_in_phloem = n.sucrose_loading_in_phloem_rate * time_step_in_seconds
    n.hexose_immobilization_as_reserve = n.hexose_immobilization_as_reserve_rate * time_step_in_seconds
    n.hexose_mobilization_from_reserve = n.hexose_mobilization_from_reserve_rate * time_step_in_seconds
    n.resp_maintenance = n.resp_maintenance_rate * time_step_in_seconds
    n.hexose_exudation = n.hexose_exudation_rate * time_step_in_seconds
    n.phloem_hexose_exudation = n.phloem_hexose_exudation_rate * time_step_in_seconds
    n.hexose_uptake_from_soil = n.hexose_uptake_from_soil_rate * time_step_in_seconds
    n.phloem_hexose_uptake_from_soil = n.phloem_hexose_uptake_from_soil_rate * time_step_in_seconds
    n.hexose_degradation = n.hexose_degradation_rate * time_step_in_seconds
    n.mucilage_secretion = n.mucilage_secretion_rate * time_step_in_seconds
    n.mucilage_degradation = n.mucilage_degradation_rate * time_step_in_seconds
    n.cells_release = n.cells_release_rate * time_step_in_seconds
    n.cells_degradation = n.cells_degradation_rate * time_step_in_seconds

    return n

def calculating_extra_variables(n, time_step_in_seconds):
    """
    This function computes additional variables on a given root element n, which may be of interest for the outputs.
    :param n: the root element to be considered
    :param time_step_in_seconds: the time step over which extra variables will be calculated
    :return: the updated root element n
    """
    # We calculate the net surfacic flux of phloem unloading (in mol of sucrose per m2 per second):
    # n.phloem_surfacic_unloading_rate = n.net_sucrose_unloading_rate / n.phloem_surface
    n.phloem_surfacic_unloading_rate = n.hexose_production_from_phloem_rate / 2. / n.phloem_surface
    # print("For element", n.index(), "the surfacic phloem unloading rate is", "{:.2E}".format(n.phloem_surfacic_unloading_rate), "mol of sucrose per m2 per second.")
    # We calculate the net rhizodeposition rates expressed in gC per cm per day:
    n.net_unloading_rate_per_day_per_cm = \
        n.net_sucrose_unloading_rate * 24. * 60. * 60. * 12. * 12.01 / n.length / 100
    # We calculate the net exudation of hexose (in mol of hexose):
    n.net_hexose_exudation = n.hexose_exudation + n.phloem_hexose_exudation - n.hexose_uptake_from_soil - n.phloem_hexose_uptake_from_soil
    # We calculate the total net rhizodeposition (i.e. subtracting the uptake of hexose from soil by roots),
    # expressed in mol of hexose:
    n.total_net_rhizodeposition \
        = n.net_hexose_exudation + n.mucilage_secretion + n.cells_release
    # We calculate the total biomass of each element, including the structural mass and all sugars:
    n.biomass = n.struct_mass + (n.C_hexose_root * 6 * 12.01 \
                                 + n.C_hexose_reserve * 6 * 12.01 + n.C_sucrose_root * 12 * 12.01) * n.struct_mass
    # We calculate a net rate of exudation, in gram of C per gram of dry structural mass per day:
    n.net_hexose_exudation_rate_per_day_per_gram \
        = (n.net_hexose_exudation / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.struct_mass
    # We calculate a net rate of exudation, in gram of C per cm of root per day:
    n.net_hexose_exudation_rate_per_day_per_cm \
        = (n.net_hexose_exudation / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100
    # We calculate the net rhizodeposition rates expressed in gC per cm per day:
    n.net_rhizodeposition_rate_per_day_per_cm = \
        (n.total_net_rhizodeposition / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100
    # We calculate the secretion rate expressed in gC per cm per day:
    n.mucilage_secretion_rate_per_day_per_cm = \
        (n.mucilage_secretion / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100
    # We calculate the cells release rate expressed in gC per cm per day:
    n.cells_release_rate_per_day_per_cm = \
        (n.cells_release / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100

    return n

def calculating_time_derivatives_of_the_amount_in_each_pool(n):
    """
    This function calculates the time derivative (dQ/dt) of the amount in each pool, for a given root element, based on 
    a carbon balance.
    :param n: the root element to be considered
    :return: a dictionary "y_derivatives" containing the values of net evolution rates for each pool.
    """

    # We initialize an empty dictionary which will contain the different fluxes:
    y_derivatives = {}

    # We calculate the derivative of the amount of sucrose for this element
    # (NOTE: we don't consider here the transfer of sucrose from other elements through the phloem):
    y_derivatives['sucrose_root'] = \
        - n.hexose_production_from_phloem_rate / 2. \
        - n.phloem_hexose_exudation_rate / 2. \
        + n.sucrose_loading_in_phloem_rate \
        + n.phloem_hexose_uptake_from_soil_rate / 2. \
        - n.Deficit_sucrose_root_rate

    # We calculate the derivative of the amount of hexose in the root reserve pool:
    y_derivatives['hexose_reserve'] = \
        + n.hexose_immobilization_as_reserve_rate - n.hexose_mobilization_from_reserve_rate \
        - n.Deficit_hexose_reserve_rate

    # Before calculating the evolution of the amount of hexose root, we make sure that all the terms in the equation are
    # known. In particular, we set the consumption rate by mycorrhizal fungus to 0 in case the interaction with the
    # fungus has not been considered so far.
    try:
        n.hexose_consumption_rate_by_fungus += 0.
    except:
        n.hexose_consumption_rate_by_fungus = 0.

    # We calculate the derivative of the amount of hexose in the mobile pool of the root:
    y_derivatives['hexose_root'] = \
        - n.hexose_exudation_rate + n.hexose_uptake_from_soil_rate \
        - n.mucilage_secretion_rate \
        - n.cells_release_rate \
        - n.resp_maintenance_rate / 6. \
        - n.hexose_consumption_by_growth_rate - n.hexose_consumption_rate_by_fungus \
        + n.hexose_production_from_phloem_rate - 2. * n.sucrose_loading_in_phloem_rate \
        + n.hexose_mobilization_from_reserve_rate - n.hexose_immobilization_as_reserve_rate \
        - n.Deficit_hexose_root_rate

    # We calculate the derivative of the amount of hexose in the soil pool:
    y_derivatives['hexose_soil'] = \
        - n.hexose_degradation_rate \
        + n.hexose_exudation_rate - n.hexose_uptake_from_soil_rate \
        + n.phloem_hexose_exudation_rate - n.phloem_hexose_uptake_from_soil_rate \
        - n.Deficit_hexose_soil_rate

    # We calculate the derivative of the amount of mucilage in the soil pool:
    y_derivatives['mucilage_soil'] = \
        + n.mucilage_secretion_rate \
        - n.mucilage_degradation_rate \
        - n.Deficit_mucilage_soil_rate

    # We calculate the derivative of the amount of root cells in the soil pool:
    y_derivatives['cells_soil'] = \
        + n.cells_release_rate \
        - n.cells_degradation_rate \
        - n.Deficit_cells_soil_rate

    return y_derivatives

def adjusting_pools_and_deficits(n, time_step_in_seconds, printing_warnings=False):
    """
    This function adjusts possibly negative concentrations by setting them to 0 and by recording the corresponding deficit.
    WATCH OUT: This function must be called once the concentrations have already been calculated with the previous deficits!
    :param n: the root element where calculations are made
    :param time_step_in_seconds: the time step over which deficits are included
    :param printing_warnings: if True, warning messages will be printed
    :return: the updated element n
    """

    # We calculate possible deficits to be included in the next time step:
    # ---------------------------------------------------------------------

    # We reset the local deficits to 0 (as the deficits are supposed to have been included in calculations before calling this function!):
    n.Deficit_sucrose_root = 0.
    n.Deficit_hexose_root = 0.
    n.Deficit_hexose_reserve = 0.
    n.Deficit_hexose_soil = 0.
    n.Deficit_mucilage_soil = 0.
    n.Deficit_cells_soil = 0.

    # We also reset the rates at which local deficits are included in the calculations:
    n.Deficit_sucrose_root_rate = 0.
    n.Deficit_hexose_root_rate = 0.
    n.Deficit_hexose_reserve_rate = 0.
    n.Deficit_hexose_soil_rate = 0.
    n.Deficit_mucilage_soil_rate = 0.
    n.Deficit_cells_soil_rate = 0.

    # Looking at sucrose:
    if n.C_sucrose_root < 0:
        # We define a positive deficit (mol of sucrose) based on the negative concentration:
        n.Deficit_sucrose_root = -n.C_sucrose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
        n.Deficit_sucrose_root_rate = n.Deficit_sucrose_root / time_step_in_seconds
        # And we set the concentration to 0:
        if printing_warnings:
            print("WARNING: After balance, there is a deficit in root sucrose for element", n.index(),
                  "that corresponds to", n.Deficit_sucrose_root,
                  "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        # And we set the concentration to zero:
        n.C_sucrose_root = 0.
    # PLEASE NOTE: The global (if any) deficit in sucrose is only used by the function "shoot_sucrose_and_spreading"
    # when defining the new homogeneous concentration of sucrose within the root system, and when performing a true
    # carbon balance of the root system in "summing".

    # Looking at hexose in the mobile pool of the root:
    if n.C_hexose_root < 0:
        if printing_warnings:
            print("WARNING: After balance, there is a deficit of root hexose for element", n.index(),
                  "that corresponds to", n.Deficit_hexose_root,
                  "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        # We define a positive deficit (mol of hexose) based on the negative concentration:
        n.Deficit_hexose_root = - n.C_hexose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
        n.Deficit_hexose_root_rate = n.Deficit_hexose_root / time_step_in_seconds
        # And we set the concentration to 0:
        n.C_hexose_root = 0.

    # Looking at hexose in the reserve pool:
    if n.C_hexose_reserve < 0:
        if printing_warnings:
            print("WARNING: After balance, there is a deficit of reserve hexose for element", n.index(),
                  "that corresponds to", n.Deficit_hexose_reserve,
                  "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        # We define a positive deficit (mol of hexose) based on the negative concentration:
        n.Deficit_hexose_reserve = - n.C_hexose_reserve * (n.struct_mass + n.living_root_hairs_struct_mass)
        n.Deficit_hexose_reserve_rate = n.Deficit_hexose_reserve / time_step_in_seconds
        # And we set the concentration to 0:
        n.C_hexose_reserve = 0.

    # Looking at hexose in the soil:
    if n.C_hexose_soil < 0:
        if printing_warnings:
            print("WARNING: After balance, there is a deficit of soil hexose for element", n.index(),
                  "that corresponds to", n.Deficit_hexose_soil,
                  "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        # We define a positive deficit (mol of hexose) based on the negative concentration:
        n.Deficit_hexose_soil = -n.C_hexose_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
        n.Deficit_hexose_soil_rate = n.Deficit_hexose_soil / time_step_in_seconds
        # And we set the concentration to 0:
        n.C_hexose_soil = 0.

    # Looking at mucilage in the soil:
    if n.Cs_mucilage_soil < 0:
        if printing_warnings:
            print("WARNING: After balance, there is a deficit of soil mucilage for element", n.index(),
                  "that corresponds to", n.Deficit_mucilage_soil,
                  "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        # We define a positive deficit (mol of equivalent-hexose) based on the negative concentration:
        n.Deficit_mucilage_soil = -n.Cs_mucilage_soil * (n.external_surface + n.living_root_hairs_external_surface)
        n.Deficit_mucilage_soil_rate = n.Deficit_mucilage_soil / time_step_in_seconds
        # And we set the concentration to 0:
        n.Cs_mucilage_soil = 0.

    # Looking at the root cells in the soil:
    if n.Cs_cells_soil < 0:
        if printing_warnings:
            print("WARNING: After balance, there is a deficit of root cells in the soil for element", n.index(),
                  "that corresponds to", n.Deficit_cells_soil,
                  "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        # We define a positive deficit (mol of hexose-equivalent) based on the negative concentration:
        n.Deficit_cells_soil = -n.Cs_cells_soil * (n.external_surface + n.living_root_hairs_external_surface)
        n.Deficit_cells_soil_rate = n.Deficit_cells_soil / time_step_in_seconds
        # And we set the concentration to 0:
        n.Cs_cells_soil = 0.

    # To avoid registering much too low values:
    for value in [n.C_hexose_root, n.C_hexose_reserve, n.C_hexose_soil, n.Cs_mucilage_soil, n.Cs_cells_soil,
                  n.Deficit_hexose_root, n.Deficit_hexose_reserve, n.Deficit_hexose_soil, n.Deficit_mucilage_soil, n.Deficit_cells_soil]:
        if value < 1e-20:
            value = 0.
    return n

# We create a class containing the system of differential equations to be solved with a solver:
#----------------------------------------------------------------------------------------------
class Differential_Equation_System(object):

    def __init__(self, g, n, time_step_in_seconds=1. * (60. * 60. * 24.), soil_temperature_in_Celsius=20,
                 printing_warnings=False, printing_solver_outputs=False):
        """
        This class is used to solve a system of differential equations corresponding to the evolution of the amounts
        in each pool for a given root element n.
        :param n: the root element n
        :param time_step_in_seconds: the time step over which new amounts/concentrations will be updated
        :param soil_temperature_in_Celsius: the temperature sensed by the root element n
        :param printing_warnings: if True, warning messages related to processes will be printed
        :param printing_solver_outputs: if True, the successive steps of the solver will be printed
        """
        self.g = g
        self.n = n
        self.time_step_in_seconds = time_step_in_seconds
        self.soil_temperature_in_Celsius = soil_temperature_in_Celsius
        self.printing_warnings = printing_warnings
        self.printing_solver_outputs = printing_solver_outputs

        # We initialize an empty list for the initial conditions of each variable:
        self.initial_conditions = []  #: the initial conditions of the compartments

        # # We initialize a timer that will be used to calculate the period elapsed over each micro time step of the solver:
        # self.initial_time_solver = 0.

        # We define the list of variables for which derivatives will be integrated, which correspond here to
        # the quantities in each pool (NOTE: we voluntarily exclude the sucrose pool in the phloem):
        self.variables_in_the_system = ['hexose_root',
                                        'hexose_reserve',
                                        'hexose_soil',
                                        'mucilage_soil',
                                        'cells_soil'
                                        ]

        # We define a second list of variables for which derivatives will not be be integrated, but for which we want to
        # record the evolution and the final state considering the successive variations of concentrations at each micro
        # time step within a time step:
        self.variables_not_in_the_system = ['hexose_production_from_phloem',
                                            'sucrose_loading_in_phloem',
                                            'resp_maintenance',
                                            'hexose_immobilization_as_reserve',
                                            'hexose_mobilization_from_reserve',
                                            'hexose_exudation',
                                            'phloem_hexose_exudation',
                                            'hexose_uptake_from_soil',
                                            'phloem_hexose_uptake_from_soil',
                                            'mucilage_secretion',
                                            'cells_release',
                                            'hexose_degradation',
                                            'mucilage_degradation',
                                            'cells_degradation']

        # We initialize an index and a dictionary:
        index = 0 # The index will be used to attribute the right property in the list corresponding to y
        y_mapping = {} # y_mapping is a dictionnary associating a unique index for each variable in y
        # We create a mapping, so that there is a dictionary containing the link between the variable name and the index in the list:
        for var in self.variables_in_the_system:
            # For the new key [var] in the dictionary, we add the corresponding index:
            y_mapping[var] = index
            # We also add a 0 in the list corresponding to the initial conditions of y:
            self.initial_conditions.append(0)
            index += 1
        # We also add the amounts exchanged between pools that we want to keep in memory:
        for var in self.variables_not_in_the_system:
            y_mapping[var] = index
            self.initial_conditions.append(0)
            index += 1

        # We keep this information in the "self":
        self.y_variables_mapping = y_mapping

        # We create a time grid of the tutorial that will be used by the solver (NOTE: here it only contains 0 and the
        # final time at the end of the time step, but we could add more intermediate values):
        self.time_grid_in_seconds = np.array([0.0, self.time_step_in_seconds])

    def _C_fluxes_within_each_segment_derivatives(self, t, y):
        """
        This internal function computes the derivative of the vector y (containing the amounts in the different pools) at `t`.
        :return: The derivatives of `y` at `t`.
        :rtype: list [float]
        """

        # We create a vector of 0 for each variable in y, for initializing the vector 'y_derivatives':
        y_derivatives = np.zeros_like(y)

        # We assign to the pool amounts in n the values in y:
        #----------------------------------------------------

        # We cover all the variables that need to be computed over the time step:
        for variable_name in self.y_variables_mapping.keys():
            # And we attribute the corresponding value in y to the corresppnding property in the root element n:
            setattr(self.n, variable_name, y[self.y_variables_mapping[variable_name]])
            # Now n has new amounts corresponding to the values calculated in y! But these amounts could be negative.

        # AVOIDING NEGATIVE AMOUNTS: We check that no pool has a negative amount, otherwise, we set it to zero!
        for variable_name in self.variables_in_the_system:
            pool_amount = getattr(self.n, variable_name)
            if pool_amount < 0.:
                # We forbids the amount to be negative and reset it to 0:
                setattr(self.n, variable_name, 0)
                # NOTE: As the corresponding exchanges between pools that led to this negative value have been recorded,
                # this reset to 0 does not eventually affect the C balance, as deficits will be defined and recorded
                # at the end of the time step!

        # We update the concentrations in each pool:
        #-------------------------------------------

        # 1) Hexose concentrations are expressed relative to the structural mass of the root element (unlike mucilage or cells!):
        mass = (self.n.initial_struct_mass + self.n.initial_living_root_hairs_struct_mass)
        # We make sure that the mass is positive:
        if isnan(mass) or mass <=0.:
                # If the initial mass is not positive, it could be that a new axis just emerged, so its initial mass was 0.
                # In this case, we set the concentrations to 0, as no fluxes (except Deficit) should be calculated before
                # growth is finished and sucrose is supplied there!
                self.n.C_hexose_root = 0.
                self.n.C_hexose_reserve = 0.
                self.n.C_hexose_soil = 0.
                # In the case the element did not emerge (i.e. its age is higher that the normal tutorial time step),
                # there must have been a problem:
                if self.n.actual_time_since_emergence > param.time_step_in_days * (24. * 60. * 60.):
                    print("!!! For element", self.n.index(),
                          "the initial mass before updating concentrations in the solver was 0 or NA!")
        else:
            # Otherwise, new concentrations are calculated considering the new amounts in each pool and the initial mass of the element:
            self.n.C_hexose_root = (self.n.hexose_root) / mass
            self.n.C_hexose_reserve = (self.n.hexose_reserve) / mass
            self.n.C_hexose_soil = (self.n.hexose_soil) / mass
            # At this point, concentrations could be negative!

        # 2) Mucilage concentration and root cells concentration in the soil are expressed relative to the external
        # surface of roots:
        surface = (self.n.initial_external_surface + self.n.initial_living_root_hairs_external_surface)
        if isnan(surface) or surface <=0.:
            # If the initial surface is not positive, it could be that a new axis just emerged.
            # In this case, we set the concentrations to 0, as no fluxes (except Deficit) should be calculated before
            # growth is finished and sucrose is supplied there!
            self.n.Cs_mucilage_soil = 0.
            self.n.Cs_cells_soil = 0.
            # In the case the element did not emerge (i.e. its age is higher that the normal tutorial time step),
            # there must have been a problem:
            if self.n.actual_time_since_emergence > param.time_step_in_days * (24. * 60. * 60.):
                print("!!! For element", self.n.index(),
                      "the initial surface before updating concentrations in the solver was 0 or NA!")
        else:
            self.n.Cs_mucilage_soil = (self.n.mucilage_soil) / surface
            self.n.Cs_cells_soil = (self.n.cells_soil) / surface
            # At this point, concentrations could be negative!

        #  Calculation of all C-related fluxes:
        # -------------------------------------
        # We call an external function that computes all the fluxes for given element n, based on the new concentrations:
        calculating_all_growth_independent_fluxes(self.g, self.n, self.soil_temperature_in_Celsius, self.printing_warnings,
                                                  self.exudation_at_root_tips_only)

        # Calculation of time derivatives:
        # ---------------------------------
        # We get a list of all derivatives of the amount in each pool, i.e. their rate of evolution over time (dQ/dt):
        time_derivatives = calculating_time_derivatives_of_the_amount_in_each_pool(self.n)
        # # And we record the new time derivatives. We cover all the variables in the differential equations system,
        # and assign the corresponding derivative over time:
        for variable_name in self.variables_in_the_system:
            y_derivatives[self.y_variables_mapping[variable_name]] = time_derivatives[variable_name]

        # # We also record other fluxes over time. We cover all the variables NOT in the differential equations system,
        # which need to be computed over the time step:
        for variable_name in self.variables_not_in_the_system:
            # We assign to 'y_derivatives' the actual rate that has been calculated above by the function
            # 'calculating_all_growth_independent_fluxes':
            variable_name_rate = variable_name + '_rate'
            y_derivatives[self.y_variables_mapping[variable_name]] = getattr(self.n, variable_name_rate)

        # The function returns the actual list of derivatives over time, both for pools and specific exchanged amounts:
        return y_derivatives

    def _update_initial_conditions(self):
        """
        This internal function simply computes the initial conditions in y, calculating the amount in each pool from its
        initial concentration.
        :return:
        """

        # We calculate the structural mass to which concentrations are related:
        mass = (self.n.initial_struct_mass + self.n.initial_living_root_hairs_struct_mass)
        if isnan(mass):
            print("!!! For element", self.n.index(), "the initial mass before updating the initial conditions in the solver was NA!")
            mass=0.
        # We calculate the external surface to which some concentrations are related:
        surface = (self.n.initial_external_surface + self.n.initial_living_root_hairs_external_surface)
        if isnan(surface):
            print("!!! For element", self.n.index(),
                  "the initial external surface before updating the initial conditions in the solver was NA!")
            surface=0.

        # We define the list of variables for which concentrations is related to surface, not mass:
        surface_related_concentrations = ['mucilage_soil', 'cells_soil']

        # We cover all the variables in the differential equations system and calculate the new amount in the pool,
        # based on the concentration in this pool:
        for variable_name in self.variables_in_the_system:
            # If the concentration is related to surface (and not mass):
            if variable_name in surface_related_concentrations:
                # We create the name corresponding to the concentration property in the root element:
                concentration_name = "Cs_" + variable_name
                # Then the amount in the pool is calculated from the concentration and the surface:
                self.initial_conditions[self.y_variables_mapping[variable_name]] = getattr(self.n, concentration_name) \
                                                                                   * surface
            else:
                # Otherwise the amount in the pool is calculated from the concentration and the mass:
                concentration_name = "C_" + variable_name
                self.initial_conditions[self.y_variables_mapping[variable_name]] = getattr(self.n, concentration_name) \
                                                                                   * mass

        # We cover all the variables NOT in the differential equations system:
        for variable_name in self.variables_not_in_the_system:
            # The initial amounts corresponding to the exchange is necessarily 0 at the beginning of the time step:
            self.initial_conditions[self.y_variables_mapping[variable_name]] = 0.

        return

    def _update_final_conditions(self, sol):
        """
        This function updates the root element n with all the quantities computed by the solver, using the last values
        in the time list of y and assigning them to the corresponding variables in the root element n.
        :param sol: the solution returned by the function 'solve_ivp'
        :return:
        """

        # We cover all the variables to be computed (i.e. both the amounts in each pool and the exchanged amounts):
        for compartment_name, compartment_index in self.y_variables_mapping.items():
            # We assign to the property in n the new value of the amount after running the solver, which is the last
            # value of the time series of the corresponging variable in the new y given by the outputs of the solver:
            setattr(self.n, compartment_name, sol.y[compartment_index][-1])

        # We also update accordingly the rates of exchange between each pool.
        # We cover all the variables NOT in the differential equations system but transformed by the solver:
        for variable_name in self.variables_not_in_the_system:
            # We create the corresponding name of the rate variable:
            variable_name_rate = variable_name + '_rate'
            # We get the value of the exchanged amount that has been updated:
            exchanged_amount = getattr(self.n, variable_name)
            # We calculate the overall mean rate of exchange between the whole time step:
            new_mean_rate = exchanged_amount / self.time_step_in_seconds
            # We assign this value of rate to the corresponding variable in n:
            setattr(self.n, variable_name_rate, new_mean_rate)

        return

    def run(self):
        """
        This internal function actually solves the system of differential equations for a given root element.
        Within a given run, it first updates the new amounts because of the function '_update_initial_conditions',
        then the new concentrations and fluxes because of the function '_C_fluxes_within_each_segment_derivatives'
        computed by the solver 'solve_ivp', an external function from SciPy, which generates new amounts in each pool.
        'solve_ivp' determines the temporal resolution by itself, depending on the integration method and the desired
        accuracy of the solution.
        """

        # We first initialize y before running the solver:
        self._update_initial_conditions()

        # We call the solver, which will repeat calculations at different times within the time step and progressively
        # transform all the variables stored in y:
        sol = solve_ivp(fun=self._C_fluxes_within_each_segment_derivatives,
                        t_span=self.time_grid_in_seconds, # Note that you could impose to the solver a number of different time points to be solved here!
                        y0=self.initial_conditions,
                        # method='BDF',
                        method='LSODA',
                        # t_eval=np.array([self.time_step_in_seconds]), # Select this to get only the final quantities at the end of the time step
                        # t_eval=np.linspace(0, self.time_step_in_seconds, 10), # Select this to get time points regularly distributed within the time step
                        t_eval=None, # Select t_eval=None to get automatical time points within the current macro time step
                        min_step=60, # OPTIONAL: defines the minimal micro time step (0 by default)
                        dense_output=False # OPTIONAL: defines whether the solution should be continuous or not (False by default)
                        )

        # OPTIONAL: We can print the results of the different iterations at some of the micro time steps!
        if self.printing_solver_outputs:
            try:
                # print(self.n.type, "-", self.n.label, "-", self.n.index())
                solver_times = pd.DataFrame(sol.t)
                solver_times.columns=['Time']
                solver_y = pd.DataFrame(sol.y)
                solver_y = solver_y.transpose()
                solver_y.columns = self.y_variables_mapping.keys()
                solver_results = pd.concat([solver_times, solver_y], axis=1)
                print(solver_results)
                print("")
            except:
                print("   > PROBLEM: Not able to interpret the solver results! Here was the message given by the solver:")
                print(sol.message)
                print("")

        # We make sure that only the last value at the end of the time step is kept in memory to update each amount:
        self._update_final_conditions(sol)
        # # NOTE: To better understand the purpose of _update_final_conditions, we can compare what sol.y gives and what
        # # has been eventually recorded in n, using as an example the amount of root hexose at the apex of the primary root:
        # if self.n.label=="Apex" and self.n.radius==param.D_ini/2.:
        #     y_value = sol.y[0,-1]
        #     n_value = self.n.hexose_root
        #     print(">>>> For the first apex, here is the final pool of root hexose in y after solving:", y_value)
        #     print(">>>> For the first apex, here is the pool of root hexose in n after solving:", n_value)
        #     print(">>>> The difference between the first and the second is:", y_value - n_value)

        return

# Performing a complete C balance on each root element:
#------------------------------------------------------
# NOTE: The function calls all processes of C exchange between pools on each root element and performs a new C balance,
# with or without using a solver:
def C_exchange_and_balance_in_roots_and_at_the_root_soil_interface(g,
                                                                   time_step_in_seconds=1. * (60. * 60. * 24.),
                                                                   soil_temperature_in_Celsius=20,
                                                                   using_solver=False,
                                                                   printing_solver_outputs=False,
                                                                   printing_warnings=False,
                                                                   exudation_at_root_tips_only=False):
    """
    :param g: the root MTG to be considered
    :param time_step_in_seconds: the time step over which the balance is done
    :param soil_temperature_in_Celsius: the temperature experience by the root element n
    :param using_solver: a Boolean (True/False) expliciting whether the system should be solved by a solver or using finite differences
    :param printing_solver_outputs: a Boolean (True/False) expliciting whether the outputs of the solver should be displayed or not
    :param printing_warnings: a Boolean (True/False) expliciting whether warning messages should be printed in the console
    :param exudation_at_root_tips_only: a Boolean (True/False) determining whether exudation is only allowed at root tips
                                        (i.e. within the meristem and root elongation zone) or not
    :return: the function updates the properties of the MTG g and returns the concentration of hexose in the apex of the primary root
    """

    # We initialize a tip concentration:
    tip_C_hexose_root = -1

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # First, we ensure that the element has a positive length:
        if n.length <= 0.:
            continue

        # # IF NOT DONE ELSEWHERE - We calculate an average flux of C consumption due to growth:
        # n.hexose_consumption_by_growth_rate = n.hexose_consumption_by_growth / time_step_in_seconds

        # OPTION 1: WITHOUT SOLVER
        ##########################

        if not using_solver:

            # We simply calculate all related fluxes and the new concentrations based on the initial conditions
            # at the beginning of the time step.

            # Calculating all new fluxes and related quantities:
            #---------------------------------------------------

            # We calculate all C-related fluxes (independent of the time step):
            calculating_all_growth_independent_fluxes(g, n, soil_temperature_in_Celsius, printing_warnings,
                                                      exudation_at_root_tips_only)

            # We then calculate the quantities that correspond to these fluxes (dependent of the time step):
            calculating_amounts_from_rates(n, time_step_in_seconds)

            # Calculating the new variations of the quantities in each pool over time:
            #-------------------------------------------------------------------------
            # We call a dictionary containing the time derivative (dQ/dt) of the amount present in each pool,
            # based on the C balance:
            y_time_derivatives = calculating_time_derivatives_of_the_amount_in_each_pool(n)

            # Calculating new concentrations based on C balance:
            #---------------------------------------------------

            # WATCH OUT: Below, the possible deficits are not included, since they have been already taken into account
            # as rates in the function "calculating_time_derivatives_of_the_amount_in_each_pool(n)" called above!

            # We calculate the new concentration of sucrose in the root according to sucrose conversion into hexose:
            sucrose_root_derivative = y_time_derivatives["sucrose_root"] * time_step_in_seconds
            n.C_sucrose_root = (n.C_sucrose_root * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                                + sucrose_root_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)

            # We calculate the new concentration of hexose in the root cytoplasm:
            hexose_root_derivative = y_time_derivatives["hexose_root"] * time_step_in_seconds
            n.C_hexose_root = (n.C_hexose_root * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                                + hexose_root_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)

            # We calculate the new concentration of hexose in the reserve:
            hexose_reserve_derivative = y_time_derivatives["hexose_reserve"] * time_step_in_seconds
            n.C_hexose_reserve = (n.C_hexose_reserve * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                                + hexose_reserve_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)

            # We calculate the new concentration of hexose in the soil:
            hexose_soil_derivative = y_time_derivatives["hexose_soil"] * time_step_in_seconds
            n.C_hexose_soil = (n.C_hexose_soil * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
                                + hexose_soil_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)

            # We calculate the new concentration of hexose in the soil:
            mucilage_soil_derivative = y_time_derivatives["mucilage_soil"] * time_step_in_seconds
            n.Cs_mucilage_soil = (n.Cs_mucilage_soil * (n.initial_external_surface
                                                        + n.initial_living_root_hairs_external_surface)
                                  + mucilage_soil_derivative) / (n.external_surface + n.living_root_hairs_external_surface)

            # We calculate the new concentration of cells in the soil:
            cells_soil_derivative = y_time_derivatives["cells_soil"] * time_step_in_seconds
            n.Cs_cells_soil = (n.Cs_cells_soil * (n.initial_external_surface
                                                        + n.initial_living_root_hairs_external_surface)
                                  + cells_soil_derivative) / (n.external_surface + n.living_root_hairs_external_surface)

        # OPTION 2: WITH SOLVER
        #######################
        # We use a numeric solver to calculate the best equilibrium between pools over time, so that the fluxes
        # still correspond to plausible values even when the time step is too large:
        if using_solver:

            if printing_solver_outputs:
                print("Considering for the solver the element", n.index(),"of length", n.length, "...")

            # We use the class corresponding to the system of differential equations and its resolution:
            System = Differential_Equation_System(g, n,
                                                  time_step_in_seconds,
                                                  soil_temperature_in_Celsius,
                                                  printing_warnings=printing_warnings,
                                                  printing_solver_outputs=printing_solver_outputs)
            System.run()
            # At this stage, the amounts in each pool pool have been updated by the solver, as well as the amount
            # exchanged between pools corresponding to specific processes.

            # We recalculate the new final concentrations based on (i) quantities at the end of the solver,
            # and (ii) struct_mass (instead of initial_struct_mass as used in the solver):
            n.C_hexose_root = n.hexose_root / (n.struct_mass + n.living_root_hairs_struct_mass)
            n.C_hexose_reserve = n.hexose_reserve / (n.struct_mass + n.living_root_hairs_struct_mass)
            n.C_hexose_soil = n.hexose_soil / (n.struct_mass + n.living_root_hairs_struct_mass)
            n.Cs_mucilage_soil = n.mucilage_soil / (n.external_surface + n.living_root_hairs_external_surface)
            n.Cs_cells_soil = n.cells_soil / (n.external_surface + n.living_root_hairs_external_surface)

            # SPECIAL CASE FOR SUCROSE:
            # As sucrose pool was not included in the solver, we update C_sucrose_root based on the mean exchange rates
            # calculated over the time_step.
            # We calculate the initial amount of sucrose in the root element:
            initial_sucrose_root_amount = n.C_sucrose_root \
                                          * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
            # We call again a dictionary containing the time derivative (dQ/dt) of the amount present in each pool, and
            # in particular sucrose, based on the balance between different rates (NOTE: the transfer of sucrose
            # between elements through the phloem is not considered at this stage!):
            y_time_derivatives = calculating_time_derivatives_of_the_amount_in_each_pool(n)
            estimated_sucrose_root_derivative = y_time_derivatives['sucrose_root'] * time_step_in_seconds
            # Eventually, the new concentration of sucrose in the root element is calculated:
            n.C_sucrose_root = (initial_sucrose_root_amount + estimated_sucrose_root_derivative) \
              / (n.struct_mass + n.living_root_hairs_struct_mass)

        # WITH BOTH OPTIONS, AFTER ALL PROCESSES HAVE BEEN COMPUTED:
        ############################################################

        # Updating concentrations and deficits:
        #--------------------------------------
        # We make sure that new concentration in each pool is not negative - otherwise we set it to 0 and record the
        # corresponding deficit to balance the next calculation of the concentration:
        adjusting_pools_and_deficits(n, time_step_in_seconds, printing_warnings)

        #  Calculation of additional variables:
        # -------------------------------------
        calculating_extra_variables(n, time_step_in_seconds)
        
        # SPECIAL CASE: we record the property of the apex of the primary root
        #---------------------------------------------------------------------
        # If the element corresponds to the apex of the primary root:
        if n.radius == param.D_ini / 2. and n.label == "Apex":
            # Then the function will give its specific concentration of mobile hexose:
            tip_C_hexose_root = n.C_hexose_root

    # We return the concentration of hexose in the apex of the primary root:
    return tip_C_hexose_root

# Calculation of total amounts and dimensions of the root system:
# ---------------------------------------------------------------
def summing_and_possibly_homogenizing(g,
                                      printing_total_length=True, printing_total_struct_mass=True, printing_all=False,
                                      homogenizing_root_sugar_concentrations=False,
                                      homogenizing_soil_concentrations=False,
                                      renewal_of_soil_solution=False,
                                      time_step_in_seconds=60.*60.):
    """
    This function computes a number of general properties summed over the whole MTG, and possibly modifies
    concentrations inside or outside the roots to account for artificial rehomogenization.
    :param g: the investigated MTG
    :param printing_total_length: a Boolean (True/False) defining whether total_length should be printed on the screen or not
    :param printing_total_struct_mass: a Boolean (True/False) defining whether total_struct_mass should be printed on the screen or not
    :param printing_all: a Boolean (True/False) defining whether all properties should be printed on the screen or not
    :param homogenizing_root_sugar_concentrations: a Boolean (True/False) allowing the concentration of root mobile hexose
                                                   and root reserve hexose to be rehomogenized, so that the concentrations
                                                   are uniform within the whole root system
    :param homogenizing_soil_concentrations: a Boolean (True/False) allowing the concentration of rhizodeposits
                                             to be rehomogenized, so that the concentrations at the soil-root
                                             interface are uniform within the whole root system
    :param renewal_of_soil_solution: a Boolean (True/False), which, together with homogenizing_soil_concentrations,
                                     allows the root-soil interface to be artificially "cleaned"
    :return: a dictionary containing the numerical value of each property integrated over the whole MTG
    """

    # We initialize the values to 0:
    total_length = 0.
    total_dead_length = 0.
    total_struct_mass = 0.
    total_living_struct_mass = 0.
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
    total_hexose_reserve_deficit = 0.
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
    total_hexose_uptake_from_soil = 0.
    total_phloem_hexose_exudation = 0.
    total_phloem_hexose_uptake_from_soil = 0.
    total_net_hexose_exudation = 0.
    total_mucilage_secretion = 0.
    total_cells_release = 0.
    total_net_rhizodeposition = 0.
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

        # INCREMEMENTATION OF TOTAL AMOUNTS:
        # Calculations of total mass, length and surface:
        if n.type == "Dead" or n.type == "Just_dead":
            # Note: we only include dead root hairs in the necromass balance when the root element itself has died!
            total_dead_struct_mass += n.struct_mass + n.root_hairs_struct_mass
            total_dead_length += n.length
            total_dead_surface += volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)["external_surface"]
        else:
            total_length += n.length
            total_struct_mass += n.struct_mass + n.root_hairs_struct_mass
            total_living_struct_mass += n.struct_mass + n.living_root_hairs_struct_mass
            total_surface += volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)["external_surface"]
            # # NOTE: living root hairs are NOT included in the total external surface of living roots!
            # NOTE: living root hairs are included in the total exchange surface with soil solution!
            # Additionaly, we calculate the total surface of living root hairs:
            total_living_root_hairs_surface +=  n.living_root_hairs_external_surface

        # Additionaly, we calculate the total mass of root hairs (both alive and dead) in the root system
        # (but note that root hairs have already been included in the total structural mass calculated above):
        total_root_hairs_mass += n.root_hairs_struct_mass

        # We increase the total amounts in each pool, based on the concentrations of the pool and its mass or surface
        # (NOTE: we don't include Deficits here, as they are considered further below):
        total_sucrose_root += n.C_sucrose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_hexose_root += n.C_hexose_root * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_hexose_reserve += n.C_hexose_reserve * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_hexose_soil += n.C_hexose_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
        total_mucilage_soil += n.Cs_mucilage_soil * (n.external_surface + n.living_root_hairs_external_surface)
        total_cells_soil += n.Cs_cells_soil * (n.external_surface + n.living_root_hairs_external_surface)

        # We increase the possible deficit of each pool:
        total_sucrose_root_deficit += n.Deficit_sucrose_root
        total_hexose_root_deficit += n.Deficit_hexose_root
        total_hexose_reserve_deficit += n.Deficit_hexose_reserve
        total_hexose_soil_deficit += n.Deficit_hexose_soil
        total_mucilage_soil_deficit += n.Deficit_mucilage_soil
        total_cells_soil_deficit += n.Deficit_cells_soil

        # We also increase the quantities exchanged between pools because of specific processes:
        total_respiration += n.resp_maintenance + n.resp_growth
        total_respiration_root_growth += n.resp_growth
        total_respiration_root_maintenance += n.resp_maintenance
        total_struct_mass_produced += n.struct_mass_produced + n.root_hairs_struct_mass_produced
        total_hexose_production_from_phloem += n.hexose_production_from_phloem
        total_sucrose_loading_in_phloem += n.sucrose_loading_in_phloem
        total_hexose_immobilization_as_reserve += n.hexose_immobilization_as_reserve
        total_hexose_mobilization_from_reserve += n.hexose_mobilization_from_reserve
        total_hexose_exudation += n.hexose_exudation
        total_hexose_uptake_from_soil += n.hexose_uptake_from_soil
        total_phloem_hexose_exudation += n.phloem_hexose_exudation
        total_phloem_hexose_uptake_from_soil += n.phloem_hexose_uptake_from_soil
        total_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake_from_soil)
        total_mucilage_secretion += n.mucilage_secretion
        total_cells_release += n.cells_release
        total_net_rhizodeposition += n.total_net_rhizodeposition
        total_hexose_degradation += n.hexose_degradation
        total_mucilage_degradation += n.mucilage_degradation
        total_cells_degradation += n.cells_degradation

    # CONSIDERING THE POSSIBLE REHOMOGENIZING OF INTERNAL SUGAR CONCENTRATIONS ALONG THE ROOTS:
    # -----------------------------------------------------------------------------------------
    if homogenizing_root_sugar_concentrations:

        # FOR ROOT HEXOSE CONCENTRATION: 
        # the concentration is already homogenized by 'shoot_sucrose_supply_and_spreading'

        # FOR ROOT HEXOSE CONCENTRATION:
        # We calculate the new homogenized concentration to apply everywhere along the living roots:
        new_C_hexose_root = (total_hexose_root - total_hexose_root_deficit) / total_living_struct_mass
        # => Note: the total_struct_mass includes living root hairs mass!
        # If the newly calculated concentration is positive:
        if new_C_hexose_root >= 0.:
            # Then we apply the same concentration and the corresponding deficit everywhere along the living roots:
            g.properties()["C_hexose_root"] \
                = {vid: new_C_hexose_root for vid in g.properties()["C_hexose_root"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            g.properties()["Deficit_hexose_root"] \
                = {vid: 0. for vid in g.properties()["Deficit_hexose_root"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            g.properties()["Deficit_hexose_root_rate"] \
                = {vid: 0. for vid in g.properties()["Deficit_hexose_root_rate"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            # If a deficit had been registered:
            if total_hexose_root_deficit > 0.:
                # Then the deficit is now integrated in the new homogenized concentration,
                # and we cancel the registered total deficit:
                total_hexose_root = new_C_hexose_root * total_living_struct_mass
                total_hexose_root_deficit = 0.
            # Otherwise, there is no need for correction of the total amounts.
        else:
            # Otherwise, we set the concentration to 0 everywhere along the living roots:
            g.properties()["C_hexose_root"] \
                = {vid: 0. for vid in g.properties()["C_hexose_root"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            # We calculate the deficit of each relevant root element relatively to its mass ratio:
            g.properties()["Deficit_hexose_root"] \
                = {vid: -new_C_hexose_root * (g.node(vid).struct_mass + g.node(vid).living_root_hairs_struct_mass)
                   for vid in g.vertices_iter(scale=1) if
                   g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            g.properties()["Deficit_hexose_root_rate"] \
                = {vid: -new_C_hexose_root * total_living_struct_mass / time_step_in_seconds
                   for vid in g.vertices_iter(scale=1) if
                   g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            # If a positive total amount had been registered:
            if total_hexose_root > 0.:
                # Then it is now integrated in the new total deficit,
                # and we cancel the registered total amount:
                total_hexose_root_deficit = -new_C_hexose_root * total_living_struct_mass
                total_hexose_root = 0.
                
        # FOR RESERVE HEXOSE CONCENTRATION:
        # We calculate the new homogenized concentration to apply everywhere along the living roots:
        new_C_hexose_reserve = (total_hexose_reserve - total_hexose_reserve_deficit) / total_living_struct_mass
        # => Note: the total_struct_mass includes living root hairs mass!
        # If the newly calculated concentration is positive:
        if new_C_hexose_reserve >= 0.:
            # Then we apply the same concentration and the corresponding deficit everywhere along the living roots:
            g.properties()["C_hexose_reserve"] \
                = {vid: new_C_hexose_reserve for vid in g.properties()["C_hexose_reserve"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            g.properties()["Deficit_hexose_reserve"] \
                = {vid: 0. for vid in g.properties()["Deficit_hexose_reserve"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            g.properties()["Deficit_hexose_reserve_rate"] \
                = {vid: 0. for vid in g.properties()["Deficit_hexose_reserve_rate"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            # If a deficit had been registered:
            if total_hexose_reserve_deficit > 0.:
                # Then the deficit is now integrated in the new homogenized concentration,
                # and we cancel the registered total deficit:
                total_hexose_reserve = new_C_hexose_reserve * total_living_struct_mass
                total_hexose_reserve_deficit = 0.
            # Otherwise, there is no need for correction of the total amounts.
        else:
            # Otherwise, we set the concentration to 0 everywhere along the living roots:
            g.properties()["C_hexose_reserve"] \
                = {vid: 0. for vid in g.properties()["C_hexose_reserve"].keys()
                   if g.node(vid).type != "Just_dead" and g.node(vid).type != "Dead"}
            # We calculate the deficit of each relevant root element relatively to its mass ratio:
            g.properties()["Deficit_hexose_reserve"] \
                = {vid: -new_C_hexose_reserve * (g.node(vid).struct_mass + g.node(vid).living_root_hairs_struct_mass)
                   for vid in g.vertices_iter(scale=1) if g.node(vid).type!="Just_dead" and g.node(vid).type!="Dead"}
            g.properties()["Deficit_hexose_reserve_rate"] \
                = {vid: -new_C_hexose_reserve * total_living_struct_mass / time_step_in_seconds
                   for vid in g.vertices_iter(scale=1) if g.node(vid).type!="Just_dead" and g.node(vid).type!="Dead"}
            # If a positive total amount had been registered:
            if total_hexose_reserve > 0.:
                # Then it is now integrated in the new total deficit,
                # and we cancel the registered total amount:
                total_hexose_reserve_deficit = -new_C_hexose_reserve * total_living_struct_mass
                total_hexose_reserve = 0.
    
    # CONSIDERING THE POSSIBLE REHOMOGENIZING OF RHIZODEPOSITS CONCENTRATIONS ALONG THE ROOTS:
    #-----------------------------------------------------------------------------------------
    if homogenizing_soil_concentrations:

        # FOR SOIL HEXOSE CONCENTRATION:
        # If the soil solution is to be renewed:
        if renewal_of_soil_solution:
            # Then the concentration is set to 0:
            new_C_hexose_soil = 0.
            # And we correspondingly increase the total amount that has "disappeared" from the root interface:
            total_hexose_degradation += total_hexose_soil - total_hexose_soil_deficit
        else:
            # Otherwise, we calculate the new homogenized concentration to apply everywhere along the roots:
            new_C_hexose_soil = (total_hexose_soil - total_hexose_soil_deficit) / total_living_struct_mass
            # => Note: the total_struct_mass includes living root hairs mass!
        # If the newly calculated concentration is positive:
        if new_C_hexose_soil >= 0.:
            # Then we apply the same concentration and the corresponding deficit everywhere along the roots:
            g.properties()["C_hexose_soil"] \
                = {vid: new_C_hexose_soil for vid in g.properties()["C_hexose_soil"].keys()}
            g.properties()["Deficit_hexose_soil"] \
                = {vid: 0. for vid in g.properties()["Deficit_hexose_soil"].keys()}
            g.properties()["Deficit_hexose_soil_rate"] \
                = {vid: 0. for vid in g.properties()["Deficit_hexose_soil_rate"].keys()}
            # If a deficit had been registered:
            if total_hexose_soil_deficit > 0.:
                # Then the deficit is now integrated in the new homogenized concentration,
                # and we cancel the registered total deficit:
                total_hexose_soil = new_C_hexose_soil * total_living_struct_mass
                total_hexose_soil_deficit = 0.
            # Otherwise, there is no need for correction of the total amounts.
        else:
            # Otherwise, we set the concentration to 0 everywhere along the roots:
            g.properties()["C_hexose_soil"] \
                = {vid: 0. for vid in g.properties()["C_hexose_soil"].keys()}
            # We calculate the deficit of each relevant root element relatively to its mass ratio:
            g.properties()["Deficit_hexose_soil"] \
                = {vid: -new_C_hexose_soil * (g.node(vid).struct_mass + g.node(vid).living_root_hairs_struct_mass)
                   for vid in g.vertices_iter(scale=1) if g.node(vid).type!="Just_dead" and g.node(vid).type!="Dead"}
            g.properties()["Deficit_hexose_soil_rate"] \
                = {vid: g.node(vid).Deficit_hexose_soil / time_step_in_seconds
                   for vid in g.vertices_iter(scale=1) if g.node(vid).type!="Just_dead" and g.node(vid).type!="Dead"}
            # If a positive total amount had been registered:
            if total_hexose_soil > 0.:
                # Then it is now integrated in the new total deficit, and we cancel the registered total amount:
                total_hexose_soil_deficit = -new_C_hexose_soil * total_living_struct_mass
                total_hexose_soil = 0.

        # FOR SOIL MUCILAGE CONCENTRATION:
        # If the soil solution is to be renewed:
        if renewal_of_soil_solution:
            # Then the concentration is set to 0:
            new_Cs_mucilage_soil = 0.
            # And we correspondingly increase the total amount that has "disappeared" from the root interface:
            total_mucilage_degradation += total_mucilage_soil - total_mucilage_soil_deficit
        else:
            # We calculate the new homogenized concentration to apply everywhere along the roots:
            new_Cs_mucilage_soil = (total_mucilage_soil - total_mucilage_soil_deficit) \
                                   / (total_surface + total_living_root_hairs_surface)
            # => Note: the total_surface does NOT include living root hairs surface!
        # If the newly calculated concentration is positive:
        if new_Cs_mucilage_soil >= 0.:
            # Then we apply the same concentration and the corresponding deficit everywhere along the roots:
            g.properties()["Cs_mucilage_soil"] \
                = {vid: new_Cs_mucilage_soil for vid in g.properties()["Cs_mucilage_soil"].keys()}
            g.properties()["Deficit_mucilage_soil"] \
                = {vid: 0. for vid in g.properties()["Deficit_mucilage_soil"].keys()}
            g.properties()["Deficit_mucilage_soil_rate"] \
                = {vid: 0. for vid in g.properties()["Deficit_mucilage_soil_rate"].keys()}
            # If a deficit had been registered:
            if total_mucilage_soil_deficit > 0.:
                # Then the deficit is now integrated in the new homogenized concentration,
                # and we cancel the registered total deficit:
                total_mucilage_soil = new_Cs_mucilage_soil * (total_surface + total_living_root_hairs_surface)
                total_mucilage_soil_deficit = 0.
            # Otherwise, there is no need for correction of the total amounts.
        else:
            # Otherwise, we simply set the concentration to 0 everywhere along the roots:
            g.properties()["Cs_mucilage_soil"] \
                = {vid: 0. for vid in g.properties()["Cs_mucilage_soil"].keys()}
            # We calculate the deficit of each relevant root element relatively to its surface ratio:
            g.properties()["Deficit_mucilage_soil"] \
                = {vid: -new_Cs_mucilage_soil * (g.node(vid).external_surface + g.node(vid).living_root_hairs_external_surface)
                   for vid in g.vertices_iter(scale=1)}
            g.properties()["Deficit_mucilage_soil_rate"] \
                = {vid: g.node(vid).Deficit_mucilage_soil / time_step_in_seconds
                   for vid in g.vertices_iter(scale=1)}
            # If a positive total amount had been registered:
            if total_mucilage_soil > 0.:
                # Then it is now integrated in the new total deficit,
                # and we cancel the registered total amount:
                total_mucilage_soil_deficit = -new_Cs_mucilage_soil * (total_surface + total_living_root_hairs_surface)
                total_mucilage_soil = 0.

        # FOR SOIL CELLS CONCENTRATION:
        # If the soil solution is to be renewed:
        if renewal_of_soil_solution:
            # Then the concentration is set to 0:
            new_Cs_cells_soil = 0.
            # And we correspondingly increase the total amount that has "disappeared" from the root interface:
            total_cells_degradation += total_cells_soil - total_cells_soil_deficit
        else:
            # We calculate the new homogenized concentration to apply everywhere along the roots:
            new_Cs_cells_soil = (total_cells_soil - total_cells_soil_deficit) \
                                   / (total_surface + total_living_root_hairs_surface)
            # => Note: the total_surface does NOT include living root hairs surface!
        # If the newly calculated concentration is positive:
        if new_Cs_cells_soil >= 0.:
            # Then we apply the same concentration and the corresponding deficit everywhere along the roots:
            g.properties()["Cs_cells_soil"] \
                = {vid: new_Cs_cells_soil for vid in g.properties()["Cs_cells_soil"].keys()}
            g.properties()["Deficit_cells_soil"] \
                = {vid: 0. for vid in g.properties()["Deficit_cells_soil"].keys()}
            g.properties()["Deficit_cells_soil_rate"] \
                = {vid: 0. for vid in g.properties()["Deficit_cells_soil_rate"].keys()}
            # If a deficit had been registered:
            if total_cells_soil_deficit > 0.:
                # Then the deficit is now integrated in the new homogenized concentration,
                # and we cancel the registered total deficit:
                total_cells_soil = new_Cs_cells_soil * (total_surface + total_living_root_hairs_surface)
                total_cells_soil_deficit = 0.
            # Otherwise, there is no need for correction of the total amounts.
        else:
            # Otherwise, we simply set the concentration to 0 everywhere along the roots:
            g.properties()["Cs_cells_soil"] \
                = {vid: 0. for vid in g.properties()["Cs_cells_soil"].keys()}
            # We calculate the deficit of each relevant root element relatively to its surface ratio:
            g.properties()["Deficit_cells_soil"] \
                = {vid: -new_Cs_cells_soil * (g.node(vid).external_surface + g.node(vid).living_root_hairs_external_surface)
                   for vid in g.vertices_iter(scale=1)}
            g.properties()["Deficit_cells_soil_rate"] \
                = {vid: g.node(vid).Deficit_cells_soil / time_step_in_seconds
                   for vid in g.vertices_iter(scale=1)}
            # If a positive total amount had been registered:
            if total_cells_soil > 0.:
                # Then it is now integrated in the new total deficit,
                # and we cancel the registered total amount:
                total_cells_soil_deficit = -new_Cs_cells_soil * (total_surface + total_living_root_hairs_surface)
                total_cells_soil = 0.

    # CORRECTING TOTAL SUCROSE DEFICIT:
    #----------------------------------
    
    # We add to the sum of local deficits in sucrose the possible global deficit in sucrose used in shoot_supply function:
    total_sucrose_root_deficit += g.property('global_sucrose_deficit')[g.root]

    # CARBON BALANCE:
    # --------------
    # We check that the carbon balance is correct (in moles of C):
    C_in_the_root_soil_system = (total_struct_mass + total_dead_struct_mass) * param.struct_mass_C_content \
                                + (total_sucrose_root - total_sucrose_root_deficit) * 12. \
                                + (total_hexose_root - total_hexose_root_deficit) * 6. \
                                + (total_hexose_reserve - total_hexose_reserve_deficit) * 6. \
                                + (total_hexose_soil - total_hexose_soil_deficit) * 6. \
                                + (total_mucilage_soil - total_mucilage_soil_deficit) * 6. \
                                + (total_cells_soil - total_cells_soil_deficit) * 6.
    C_degraded = (total_hexose_degradation + total_mucilage_degradation + total_cells_degradation) * 6.
    C_respired_by_roots = total_respiration

    # For avoiding problems: we avoid carrying too low values for some variables.
    if total_hexose_reserve_deficit < 1e-25:
        total_hexose_reserve_deficit = 0.

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
        print("      The current amount of hexose stored as reserve in the roots (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_hexose_reserve  - total_hexose_reserve_deficit)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal((total_hexose_reserve - total_hexose_reserve_deficit) * 6)), "mol of C.")
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
              "{:.2E}".format(Decimal(total_net_rhizodeposition)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_net_rhizodeposition * 6)), "mol of C.")
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
                  "total_hexose_reserve_deficit": total_hexose_reserve_deficit,
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
                  "total_phloem_hexose_exudation": total_phloem_hexose_exudation,
                  "total_hexose_uptake_from_soil": total_hexose_uptake_from_soil,
                  "total_phloem_hexose_uptake_from_soil": total_phloem_hexose_uptake_from_soil,
                  "total_mucilage_secretion": total_mucilage_secretion,
                  "total_cells_release": total_cells_release,
                  "total_hexose_degradation": total_hexose_degradation,
                  "total_mucilage_degradation": total_mucilage_degradation,
                  "total_cells_degradation": total_cells_degradation,
                  "total_net_hexose_exudation": total_net_hexose_exudation,
                  "total_net_rhizodeposition": total_net_rhizodeposition,
                  "C_in_the_root_soil_system": C_in_the_root_soil_system,
                  "C_degraded_in_the_soil": C_degraded,
                  "C_respired_by_roots": C_respired_by_roots
                  }

    return dictionary

# Control of anomalies in the MTG:
# --------------------------------
def control_of_anomalies(g):
    """
    The function contol_of_anomalies checks for the presence of elements with negative measurable properties
    (e.g. length, concentrations).
    :param g: the root MTG to be considered
    :return: [nothing, only warnings are displayed]
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
                 simple_growth_duration=True,
                 initial_segment_length=1e-3,
                 initial_apex_length=0.,
                 initial_C_sucrose_root=0.,
                 initial_C_hexose_root=0.,
                 input_file_path="C:/Users/frees/rhizodep/src/rhizodep/",
                 forcing_seminal_roots_events=True,
                 seminal_roots_events_file="seminal_roots_inputs.csv",
                 forcing_adventitious_roots_events=True,
                 adventitious_roots_events_file="adventitious_roots_inputs.csv"):

    """
    This functions generates a root MTG from scratch, containing only one segment of a specific length,
    terminated by an apex (preferably of length 0).
    :param random: if True, random angles and radii will be used
    :param simple_growth_duration: if True, the classical rule of ArchiSimple will be used to determine the growth
                                   duration of roots (otherwise, a probabilistic rule will be used)
    :param initial_segment_length: the length of the initial segment to create
    :param initial_apex_length: the length of the initial apex to create
    :param initial_C_sucrose_root: the initial sucrose concentration in the root axis to create
    :param initial_C_hexose_root: the initial mobile hexose concentration in the root axis to create
    :param input_file_path: the path of the folder containing the files setting the emergence of seminal roots and adventitious roots
    :param forcing_seminal_roots_events: if True, the emergence of future seminal roots will be set according to a regular time interval
    :param seminal_roots_events_file: the name of the file where information of the delay of emergence
                                      between each seminal root is detailed (if not forcing a regular interval of seminal root emergence)
    :param forcing_adventitious_roots_events: if True, the emergence of future adventitious roots will be set according to a regular time interval
    :param adventitious_roots_events_file: the name of the file where information of the delay of emergence
                                           between each adventitious root is detailed (if not forcing a regular interval of seminal root emergence)
    :return:
    """

    # We create a new MTG called g:
    g = MTG()

    # Properties shared by the whole root system (stored in the first element at the base of the root system):
    # --------------------------------------------------------------------------------------------------------
    # We initiate the global variable that corresponds to a possible general deficit in sucrose of the whole root system:
    g.add_property('global_sucrose_deficit')
    g.property('global_sucrose_deficit')[g.root] = 0.

    # We first add one initial element:a
    # ---------------------------------
    id_segment = g.add_component(g.root, label='Segment')
    base_segment = g.node(id_segment)

    # Characteristics:
    # -----------------
    base_segment.type = "Base_of_the_root_system"
    # By definition, we set the order of the primary root to 1:
    base_segment.root_order = 1

    # Authorizations and C requirements:
    # -----------------------------------
    base_segment.lateral_root_emergence_possibility = 'Impossible'
    base_segment.emergence_cost = 0.

    # Geometry and topology:
    # -----------------------

    base_radius = param.D_ini / 2.

    # base_segment.angle_down = 180
    base_segment.angle_down = 0
    base_segment.angle_roll = 0
    base_segment.length = initial_segment_length
    base_segment.radius = base_radius
    base_segment.original_radius = base_radius
    base_segment.initial_length = initial_segment_length
    base_segment.initial_radius = base_radius

    base_segment.root_hair_radius = param.root_hair_radius
    base_segment.root_hair_length = 0.
    base_segment.actual_length_with_hairs = 0.
    base_segment.living_root_hairs_number = 0.
    base_segment.dead_root_hairs_number = 0.
    base_segment.total_root_hairs_number = 0.

    base_segment.actual_time_since_root_hairs_emergence_started = 0.
    base_segment.thermal_time_since_root_hairs_emergence_started = 0.
    base_segment.actual_time_since_root_hairs_emergence_stopped= 0.
    base_segment.thermal_time_since_root_hairs_emergence_stopped = 0.
    base_segment.all_root_hairs_formed = False
    base_segment.root_hairs_lifespan = param.root_hairs_lifespan
    base_segment.root_hairs_external_surface = 0.
    base_segment.root_hairs_volume = 0.
    base_segment.living_root_hairs_external_surface = 0.
    base_segment.initial_living_root_hairs_external_surface = 0.
    base_segment.root_hairs_struct_mass = 0.
    base_segment.root_hairs_struct_mass_produced = 0.
    base_segment.living_root_hairs_struct_mass = 0.

    surface_dictionary = volume_and_external_surface_from_radius_and_length(g, base_segment, base_segment.radius, base_segment.length)
    base_segment.external_surface = surface_dictionary["external_surface"]
    base_segment.initial_external_surface = base_segment.external_surface
    base_segment.volume = surface_dictionary["volume"]

    base_segment.distance_from_tip = base_segment.length
    base_segment.former_distance_from_tip = base_segment.length
    base_segment.dist_to_ramif = 0.
    base_segment.actual_elongation = base_segment.length
    base_segment.actual_elongation_rate = 0

    # Quantities and concentrations:
    # -------------------------------
    base_segment.struct_mass = base_segment.volume * param.root_tissue_density
    base_segment.initial_struct_mass = base_segment.struct_mass
    base_segment.initial_living_root_hairs_struct_mass = base_segment.living_root_hairs_struct_mass
    # We define the initial concentrations:
    base_segment.C_sucrose_root = initial_C_sucrose_root
    base_segment.C_hexose_root = initial_C_hexose_root
    base_segment.C_hexose_reserve = 0.
    base_segment.C_hexose_soil = 0.
    base_segment.Cs_mucilage_soil = 0.
    base_segment.Cs_cells_soil = 0.

    # We calculate the possible deficits:
    base_segment.Deficit_sucrose_root = 0.
    base_segment.Deficit_hexose_root = 0.
    base_segment.Deficit_hexose_reserve = 0.
    base_segment.Deficit_hexose_soil = 0.
    base_segment.Deficit_mucilage_soil = 0.
    base_segment.Deficit_cells_soil = 0.
    base_segment.Deficit_sucrose_root_rate = 0.
    base_segment.Deficit_hexose_root_rate = 0.
    base_segment.Deficit_hexose_reserve_rate = 0.
    base_segment.Deficit_hexose_soil_rate = 0.
    base_segment.Deficit_mucilage_soil_rate = 0.
    base_segment.Deficit_cells_soil_rate = 0.

    # Amounts related to the processes:
    # ---------------------------------
    base_segment.struct_mass_produced = 0.
    base_segment.resp_maintenance = 0.
    base_segment.resp_growth = 0.
    base_segment.hexose_growth_demand = 0.
    base_segment.hexose_possibly_required_for_elongation = 0.
    base_segment.hexose_consumption_by_growth = 0.
    base_segment.hexose_production_from_phloem = 0.
    base_segment.sucrose_loading_in_phloem = 0.
    base_segment.hexose_mobilization_from_reserve = 0.
    base_segment.hexose_immobilization_as_reserve = 0.
    base_segment.hexose_exudation = 0.
    base_segment.hexose_uptake_from_soil = 0.
    base_segment.phloem_hexose_exudation = 0.
    base_segment.phloem_hexose_uptake_from_soil = 0.
    base_segment.mucilage_secretion = 0.
    base_segment.cells_release = 0.
    base_segment.total_net_rhizodeposition = 0.
    base_segment.hexose_degradation = 0.
    base_segment.mucilage_degradation = 0.
    base_segment.cells_degradation = 0.
    base_segment.specific_net_exudation = 0.

    # Rates:
    #-------
    base_segment.resp_maintenance_rate = 0.
    base_segment.resp_growth_rate = 0.
    base_segment.hexose_growth_demand_rate = 0.
    base_segment.hexose_consumption_by_growth_rate = 0.
    base_segment.hexose_production_from_phloem_rate = 0.
    base_segment.sucrose_loading_in_phloem_rate = 0.
    base_segment.hexose_mobilization_from_reserve_rate = 0.
    base_segment.hexose_immobilization_as_reserve_rate = 0.
    base_segment.hexose_exudation_rate = 0.
    base_segment.hexose_uptake_from_soil_rate = 0.
    base_segment.phloem_hexose_exudation_rate = 0.
    base_segment.phloem_hexose_uptake_from_soil_rate = 0.
    base_segment.mucilage_secretion_rate = 0.
    base_segment.cells_release_rate = 0.
    base_segment.hexose_degradation_rate = 0.
    base_segment.mucilage_degradation_rate = 0.
    base_segment.cells_degradation_rate = 0.

    # Time indications:
    # ------------------
    base_segment.growth_duration = calculate_growth_duration(radius=base_radius, index=id_segment, root_order=1,
                                                             ArchiSimple=simple_growth_duration)
    base_segment.life_duration = param.LDs * (2. * base_radius) * param.root_tissue_density
    base_segment.actual_time_since_primordium_formation = 0.
    base_segment.actual_time_since_emergence = 0.
    base_segment.actual_time_since_cells_formation = 0.
    base_segment.actual_time_since_growth_stopped = 0.
    base_segment.actual_time_since_death = 0.
    base_segment.thermal_time_since_primordium_formation = 0.
    base_segment.thermal_time_since_emergence = 0.
    base_segment.thermal_time_since_cells_formation = 0.
    base_segment.thermal_potential_time_since_emergence = 0.
    base_segment.thermal_time_since_growth_stopped = 0.
    base_segment.thermal_time_since_death = 0.

    segment = base_segment

    # ADDING THE PRIMORDIA OF ALL POSSIBLE SEMINAL ROOTS:
    #----------------------------------------------------
    # If there is more than one seminal root (i.e. roots already formed in the seed):
    if param.n_seminal_roots > 1 or not forcing_seminal_roots_events:

        # We read additional parameters that are stored in a CSV file, with one column containing the delay for each
        # emergence event, and the second column containing the number of seminal roots that have to emerge at each event:
        # We try to access an already-existing CSV file:
        seminal_inputs_path = os.path.join(input_file_path, seminal_roots_events_file)
        # If the file doesn't exist, we construct a new table using the specified parameters:
        if not os.path.exists(seminal_inputs_path) or forcing_seminal_roots_events:
            print("NOTE: no CSV file describing the apparitions of seminal roots can be used!")
            print("=> We therefore built a table according to the parameters 'n_seminal_roots' and 'ER'.")
            print("")
            # We initialize an empty data frame:
            seminal_inputs_file = pd.DataFrame()
            # We define a list that will contain the successive thermal times corresponding to root emergence:
            list_time = [x * 1/param.ER for x in range(1, param.n_seminal_roots)]
            # We define another list containing only "1" as the number of roots to be emerged for each event:
            list_number = np.ones(param.n_seminal_roots-1, dtype='int8')
            # We assigned the two lists to the dataframe, and record it:
            seminal_inputs_file['emergence_delay_in_thermal_time'] = list_time
            seminal_inputs_file['number_of_seminal_roots_per_event'] = list_number
            # seminal_inputs_file.to_csv(os.path.join(input_file_path, seminal_roots_events_file),
            #                                 na_rep='NA', index=False, header=True)
        else:
            seminal_inputs_file = pd.read_csv(seminal_inputs_path)

        # For each event of seminal roots emergence:
        for i in range(0, len(seminal_inputs_file.emergence_delay_in_thermal_time)):
            # For each seminal root that can emerge at this emergence event:
            for j in range(0,seminal_inputs_file.number_of_seminal_roots_per_event[i]):

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
                # apex_seminal.growth_duration = param.GDs * (2. * radius_seminal) ** 2 * param.main_roots_growth_extender
                apex_seminal.growth_duration = calculate_growth_duration(radius=radius_seminal,
                                                                         index=apex_seminal.index(),
                                                                         root_order=1,
                                                                         ArchiSimple=simple_growth_duration)
                apex_seminal.life_duration = param.LDs * (2. * radius_seminal) * param.root_tissue_density

                # We defined the delay of emergence for the new primordium:
                apex_seminal.emergence_delay_in_thermal_time = seminal_inputs_file.emergence_delay_in_thermal_time[i]

    # ADDING THE PRIMORDIA OF ALL POSSIBLE ADVENTIOUS ROOTS:
    # ------------------------------------------------------
    # If there should be more than one main root (i.e. adventitious roots formed at the basis):
    if param.n_adventitious_roots > 0 or not forcing_adventitious_roots_events:

        # We read additional parameters from a table, with one column containing the delay for each emergence event,
        # and the second column containing the number of adventitious roots that have to emerge at each event.
        # We try to access an already-existing CSV file:
        adventitious_inputs_path = os.path.join(input_file_path, adventitious_roots_events_file)
        # If the file doesn't exist, we construct a new table using the specified parameters:
        if not os.path.exists(adventitious_inputs_path) or forcing_adventitious_roots_events:
            print("NOTE: no CSV file describing the apparitions of adventitious roots can be used!")
            print("=> We therefore built a table according to the parameters 'n_adventitious_roots' and 'ER'.")
            print("")
            # We initialize an empty data frame:
            adventitious_inputs_file = pd.DataFrame()
            # We define a list that will contain the successive thermal times corresponding to root emergence:
            list_time = [param.starting_time_for_adventitious_roots_emergence + x * 1/param.ER
                         for x in range(0, param.n_adventitious_roots)]
            # We define another list containing only "1" as the number of roots to be emerged for each event:
            list_number = np.ones(param.n_adventitious_roots, dtype='int8')
            # We assigned the two lists to the dataframe, and record it:
            adventitious_inputs_file['emergence_delay_in_thermal_time'] = list_time
            adventitious_inputs_file['number_of_adventitious_roots_per_event'] = list_number
            # adventitious_inputs_file.to_csv(os.path.join(input_file_path, adventitious_roots_events_file),
            #                                 na_rep='NA', index=False, header=True)
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
                # apex_adventitious.growth_duration = param.GDs * (2. * radius_adventitious) ** 2 * param.main_roots_growth_extender
                apex_adventitious.growth_duration = calculate_growth_duration(radius=radius_adventitious,
                                                                              index=apex_adventitious.index(),
                                                                              root_order=1,
                                                                              ArchiSimple=simple_growth_duration)
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
                          length=initial_apex_length,
                          radius=base_radius,
                          identical_properties=False,
                          nil_properties=True)
    apex.original_radius = apex.radius
    apex.initial_radius = apex.radius
    # apex.growth_duration = param.GDs * (2. * base_radius) ** 2 * param.main_roots_growth_extender
    apex.growth_duration = calculate_growth_duration(radius=base_radius,
                                                     index=apex.index(),
                                                     root_order=1,
                                                     ArchiSimple=simple_growth_duration)
    apex.life_duration = param.LDs * (2. * base_radius) * param.root_tissue_density

    if initial_apex_length <=0.:
        apex.C_sucrose_root=0.
        apex.C_hexose_root=0.
    else:
        apex.C_sucrose_root=initial_C_sucrose_root
        apex.C_hexose_root=initial_C_hexose_root
    apex.C_hexose_soil=0.
    apex.Cs_mucilage_soil=0.
    apex.Cs_cells_soil=0.

    apex.volume = volume_and_external_surface_from_radius_and_length(g, apex, apex.radius, apex.length)["volume"]
    apex.struct_mass = apex.volume * param.root_tissue_density
    apex.initial_struct_mass = apex.struct_mass
    apex.initial_living_root_hairs_struct_mass = apex.living_root_hairs_struct_mass
    apex.initial_external_surface = apex.external_surface
    apex.initial_living_root_hairs_external_surface = apex.living_root_hairs_external_surface

    return g