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


class SoilModel:
    def __init__(self, g, time_step_in_seconds):
        self.g = g
        self.time_step_in_seconds = time_step_in_seconds

        # TODO TP hardcode
        self.soil_temperature = 20.0

    # MODULE "SOIL TRANSFORMATION"

    # Degradation of hexose in the soil (microbial consumption):
    # ----------------------------------------------------------

    # We create a class containing the system of differential equations to be solved with a solver:
    # ----------------------------------------------------------------------------------------------

    def run_exchanges_and_balance(self):
        pass

    def hexose_degradation_rate(self, n, soil_temperature_in_Celsius=20, printing_warnings=False):
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
            corrected_hexose_degradation_rate_max = param.hexose_degradation_rate_max * self.temperature_modification(
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
    def mucilage_degradation_rate(self, n, soil_temperature_in_Celsius=20, printing_warnings=False):
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
        if length <= 0 or exchange_surface <= 0.:
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
            corrected_mucilage_degradation_rate_max = param.mucilage_degradation_rate_max * self.temperature_modification(
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
    def cells_degradation_rate(self, n, soil_temperature_in_Celsius=20, printing_warnings=False):
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
        if length <= 0 or exchange_surface <= 0.:
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
            corrected_cells_degradation_rate_max = param.cells_degradation_rate_max * self.temperature_modification(
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

    # TODO FOR TRISTAN: Consider adding similar functions for describing N mineralization/organization in the soil?

