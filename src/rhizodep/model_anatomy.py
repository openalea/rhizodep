import os
import numpy as np
import pandas as pd
from math import sqrt, pi, floor, exp, isnan
from decimal import Decimal
from scipy.integrate import solve_ivp
from dataclasses import dataclass, asdict
import inspect as ins
from functools import partial

from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order

import rhizodep.parameters as param

class RootAnatomy:
    def __init__(self, g, time_step_in_seconds):
        self.g = g
        self.time_steps_in_seconds = time_step_in_seconds

    def volume_and_external_surface_from_radius_and_length(self, g, element, radius, length):
        """
        This function computes the volume (m3) of a root element and its external surface (excluding possible root hairs)
        based on the properties radius (m) and length (m) and possibly on its type.
        :param g: the investigated MTG
        :param element: the investigated node of the MTG
        :return: a dictionary containing the volume and the external surface of the given element
        """

        # READING THE VALUES:
        # --------------------
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
                # TODO: Should the section of a son nodule also be removed from the section of the mother element?
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
    def specific_surfaces(self, element):
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

        # TODO: Improve the calculation of phloem surface and stellar/cortical/epidermal surfaces, e.g. taking into
        #  account an age factor?

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
    # ------------------------------------------------------------------------------------------------------------------
    def endodermis_and_exodermis_conductances_as_a_function_of_x(self, distance_from_tip,
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
    # --------------------------------------------------------------------------------------------------------------
    def root_barriers_length_integrator(self, length_start,
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
            dict_cond = self.endodermis_and_exodermis_conductances_as_a_function_of_x(progressive_length,
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
        dictionary = {"conductance_endodermis": integrated_value_endodermis,
                      "conductance_exodermis": integrated_value_exodermis}

        return dictionary

    def transport_barriers(self, g, n, computation_with_age=True, computation_with_distance_to_tip=False):
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
        # --------------------
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
        # TODO: Consider a progressive reduction of the conductance of cell walls with high root cells age?
        # If the current element encompasses part of all of the meristem zone:
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
        # ------------------------------------

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
            if param.endodermis_b - param.endodermis_c * age > 1000:
                relative_conductance_endodermis = 1.
            else:
                relative_conductance_endodermis = (100 - param.endodermis_a * np.exp(
                    -np.exp(param.endodermis_b - param.endodermis_c * age))) / 100.
            if param.exodermis_b - param.exodermis_c * age > 1000:
                relative_conductance_exodermis = 1.
            else:
                relative_conductance_exodermis = (100 - param.exodermis_a * np.exp(
                    -np.exp(param.exodermis_b - param.exodermis_c * age))) / 100.

            # OPTION 2 - The formation of transport barriers is dictated by the distance to root tip:
        if computation_with_distance_to_tip:
            # We define the distances from apex where barriers start/end:
            start_distance_endodermis = param.start_distance_for_endodermis_factor * radius
            end_distance_endodermis = param.end_distance_for_endodermis_factor * radius
            start_distance_exodermis = param.start_distance_for_exodermis_factor * radius
            end_distance_exodermis = param.end_distance_for_exodermis_factor * radius

            # We call a function that integrates the values of relative conductances between the beginning and the end of the
            # root element, knowing the evolution of the conductances with x (the distance from root tip):
            dict_cond = self.root_barriers_length_integrator(length_start=distance_from_tip - length,
                                                        length_stop=distance_from_tip,
                                                        number_of_length_steps=10,
                                                        starting_distance_endodermis=start_distance_endodermis,
                                                        ending_distance_endodermis=end_distance_endodermis,
                                                        starting_distance_exodermis=start_distance_exodermis,
                                                        ending_distance_exodermis=end_distance_exodermis)
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
                if lateral_length <= 0.:
                    # Then we move to the next possible lateral root (otherwise, the loop stops and conductance remains unaltered):
                    continue
                # ENDODERMIS: If the lateral root has emerged recently, its endodermis barrier has been diminished as soon
                # as the lateral started to elongate:
                t_since_endodermis_was_disrupted = t
                if t_since_endodermis_was_disrupted < t_max_endo:
                    # We increase the relative conductance of endodermis according to the age of the lateral root,
                    # considering that the barrier starts at 1 and linearily decreases with time until reaching 0. However,
                    # if the barrier was not completely formed initially, we should not set it to zero, and therefore
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
                        # root crossed the exodermis, considering that the barrier starts at 1 and linearily decreases with
                        # time until reaching 0. However, if the barrier was not completely formed initially, we should not
                        # set it to zero, and we therefore define the new conductance as the maximal value between the
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
    def update_surfaces_and_volumes(self, g):

        """
        This function go through each root element and updates their surfaces, transport barriers and volume.
        :param g: the root MTG to be considered
        :return: the updated root MTG
        """

        # We cover all the vertices in the MTG:
        for vid in g.vertices_iter(scale=1):

            # n represents the vertex:
            n = g.node(vid)

            # First, we ensure that the element has a positive length:
            if n.length <= 0.:
                continue

            # We call the function that automatically calculates the volume and external surface:
            surfaces_and_volumes_dict = self.volume_and_external_surface_from_radius_and_length(g, n, n.radius, n.length)
            # We compute the total volume of the element:
            n.volume = surfaces_and_volumes_dict["volume"]
            # We calculate the current external surface of the element:
            n.external_surface = surfaces_and_volumes_dict["external_surface"]

            # We call the function that automatically updates the other surfaces of within the cells (ex: cortical symplast):
            self.specific_surfaces(n)

            # We call the function that automatically updates the transport barriers (i.e. endodermis and exodermis):
            self.transport_barriers(g, n)

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

        return g

    # Defining the distance of a vertex from the tip for the whole root system:
    # -------------------------------------------------------------------------

    def update_local_anatomy(self):
        # We update the surfaces and the volume for each root element in each root axis:
        self.update_surfaces_and_volumes()

        # 2c - SPECIFIC ROOT HAIR DYNAMICS
        # ================================

        # We modifiy root hairs characteristics according to their specific dynamics:
        self.root_hairs_dynamics(soil_temperature_in_Celsius=soil_temperature,
                                  printing_warnings=printing_warnings)
