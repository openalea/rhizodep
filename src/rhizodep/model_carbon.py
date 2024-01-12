#  -*- coding: utf-8 -*-

"""
    rhizodep.model
    ~~~~~~~~~~~~~

    The module :mod:`rhizodep.model` defines the equations of root functioning.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

# TODO: Check the calculation of "Structural mass produced (g)"
# TODO: Update ArchiSimple option (e.g. with the new way to represent Seminal/Adventitious roots?)
# TODO: Add a general upper scale for the whole root system, in which the properties about phloem sucrose could be stored?
# TODO: Introduce explicit threshold concentration for limiting processes, rather than "0"
# TODO: Consider giving priority to root maintenance, and trigger senescence when root maintenance is not ensured
# TODO: Watch the calculation of surface and volume for the apices - if they correspond to cones, the mass balance for
#  segmentation may not be correct! A "reverse" function for calculating length/radius from the volume should be used.

# TODO: POSSIBLE FUTURE DEVELOPMENT - Modify the way the apex and elongation zone are discretized:
#  instead of using a tip that is elongated and segmented over time, define a meristem object and an elongation zone
#  object of fixed length, followed by a segmentation zone of variable length that is actually elongating. This would
#  help to get a better spatial description of fluxes and concentration at the root tip, without the influence of the
#  length of the tip (the larger the tip, the lower the internal and external concentrations....).

# TODO: POSSIBLE FUTURE DEVELOPMENT - Use two distinct structures for computing fluxes and geometry:
#  instead of using a unique structure of the root system for growth, C exchange and topology/geometry, we would define
#  a first "metabolic" structure with elements of variable length (smaller at the apex, larger at the base of each root
#  axis), which would be used to calculate growth, topology and fluxes, without consideration of the spatial position of
#  each root element. A converter would then be used to translate this structure into another "geometric" structure with
#  elements of fixed (?) length and specified orientation in space. The concentrations and fluxes in an element of the
#  new structure would be calculated as a linear combination of the concentrations and fluxes in the corresponding
#  elements of the first structure. This second structure would be used to display a plausible geometry and to project
#  the root system into a soil grid. New concentrations at the interface of the roots would then be calculated, and the
#  converter would be used in a reverse way to switch back to the first, metabolic structure at the next time step. This
#  new development would be useful only if the metabolic structure has less elements in total than the geometric one.

# TODO: POSSIBLE FUTURE DEVELOPMENT - Create a new ID of each root element, so that the ID does not change over time
#  (unlike the index of elements in the MTG, which cannot be easily predicted or interpreted):
#  We could label the elongating apex as a special case (A) for a given axis, while segments' numerotation
#  would start close to the mother root and increase towards the apex. In this way the apex keeps the same ID regardless
#  of the elongation of the axis and the segmentation, while a given segment also keeps its own ID regardless of
#  segmentation. Each lateral axis would be identified based on the supporting element (if two laterals emerge from the
#  same segment, they are numbered successively depending on their orientation in space ?).
#  Example:
#  - R(0)-A always designates the apex of the primary root.
#  - R(0-S3-1)-S2 represents the second segment of the axis R(0-S3-1), which corresponds to the first lateral emerging
#  from the third segment of axis R0.
#  - R((0-S3-1)-S2-1)-A represents the apex of the first lateral root emerging from the segment described above.
#  Check what is used in L-system and whether we are not reinventing the wheel!

# TODO: POSSIBLE FUTURE DEVELOPMENT - Create a way to detect oscillatory events (e.g. when a concentration is either nil
#  or very high from one step to another) and to avoid them (by blocking the consumption of the particular pool for example).

import numpy as np
import pandas as pd
from math import isnan
from scipy.integrate import solve_ivp
from dataclasses import dataclass, asdict
import inspect as ins
from functools import partial

import rhizodep.parameters as param


@dataclass
class RootCarbonModel:
    """
    Root carbon balance model originating from Rhizodep model.py

    forked :
        https://forgemia.inra.fr/tristan.gerault/rhizodep/-/commits/rhizodep_2022?ref_type=heads
    base_commit :
        92a6f7ad927ffa0acf01aef645f9297a4531878c
    """

    def __init__(self, g, time_step_in_seconds: int):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step_in_seconds: time step of the simulation (s)
        :return:
        """
        self.g = g
        self.props = self.g.properties()
        self.time_steps_in_seconds = time_step_in_seconds

        # TODO clarify names to be sure we talk about concentrations or amounts
        self.states = """
            C_sucrose_root
            sucrose_root
            C_hexose_root
            hexose_root

            C_hexose_soil
            hexose_soil

            type
            struct_mass
            living_root_hairs_struct_mass
            length
            radius
            original_radius
            exchange_surface
            non_vascular_exchange_surface
            distance_from_tip

            hexose_production_from_phloem
            phloem_hexose_exudation
            sucrose_loading_in_phloem
            phloem_hexose_uptake_from_soil
            hexose_immobilization_as_reserve
            hexose_mobilization_from_reserve
            hexose_exudation
            hexose_uptake_from_soil
            mucilage_secretion
            cells_release
            resp_maintenance
            hexose_consumption_by_growth
            sucrose_loading_in_phloem
            hexose_mobilization_from_reserve,
            hexose_immobilization_as_reserve

            deficit_sucrose_root
            deficit_hexose_reserve
            deficit_hexose_root
        """.split()

        for name in self.states:
            setattr(self, name, self.props[name])

        self.inputs = {
            # Common
            "soil": [
                "soil_temperature_in_Celsius"
                "Cs_mucilage_soil"
                "Cs_cells_soil"
            ],
            "growth": [
                "hexose_consumption_by_growth"
            ]
        }

    def store_functions_call(self):
        # Storing function calls
        #
        # Local and plant scale processes...
        self.process_param = asdict(ProcessCarbon())
        self.process_methods = [partial(getattr(self, func), **self.process_param)
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_args = [
            [partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
            for func in dir(self) if
            (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_names = [func[8:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'process' in func)]

        # Local and plant scale update...
        self.update_param = asdict(UpdateCarbon())
        self.update_methods = [partial(getattr(self, func), **self.update_param)
                               for func in dir(self) if
                               (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_args = [
            [partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
            for func in dir(self) if
            (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_names = [func[7:] for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'update' in func)]

        # Local deficits update...
        self.deficit_methods = [partial(getattr(self, func), **self.update_param)
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'deficit' in func)]
        self.deficit_args = [
            [partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
            for func in dir(self) if
            (callable(getattr(self, func)) and '__' not in func and 'deficit' in func)]
        self.deficit_names = [func for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'deficit' in func)]

    def exchanges_and_balance(self):

        # 2d - CARBON EXCHANGE
        # ====================
        # TODO initialize new elements based on struct mass property at the same time?
        # TODO concentrations update related to struct mass update
        # We now proceed to all the exchanges of C for each root element in each root axis
        # (NOTE: the supply of sucrose from the shoot is excluded in this calculation):
        self.props.update(self.prc_resolution())
        self.props.update(self.upd_resolution())

        # Computing the deficits after this balance
        self.props.update(self.dfc_resolution())
        # Then bring concentrations back to 0 if negative
        # NOTE : a bit raw
        self.props.update(self.forbid_negatives())

        # Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
        self.shoot_sucrose_supply_and_spreading()

        # # OPTIONAL: checking of possible anomalies in the root system:
        self.control_of_anomalies()

    def prc_resolution(self):
        return dict(zip([name for name in self.process_names],
                        map(self.dict_mapper, *(self.process_methods, self.process_args))))

    def upd_resolution(self):
        return dict(
            zip([name for name in self.update_names], map(self.dict_mapper, *(self.update_methods, self.update_args))))

    def dfc_resolution(self):
        return dict(zip([name for name in self.deficit_names],
                        map(self.dict_mapper, *(self.deficit_methods, self.deficit_args))))

    def forbid_negatives(self):
        return dict(zip(["hexose_root", "hexose_reserve", "sucrose_root"],
                        map(self.dict_mapper,
                            *([partial(max, 0.)] * 3, [self.hexose_root, self.hexose_reserve, self.sucrose_root]))))

    def dict_mapper(self, fcn, args):
        return dict(zip(args[0](), map(fcn, *(d().values() for d in args))))

    def get_up_to_date(self, prop):
        return getattr(self, prop)

    # Modification of a process according to soil temperature:
    # --------------------------------------------------------
    def temperature_modification(self, temperature_in_Celsius, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
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

    # Calculation of the total amount of sucrose and structural struct_mass in the root system:
    # -----------------------------------------------------------------------------------------
    def total_root_sucrose_and_living_struct_mass(self):
        """
        This function computes the total amount of sucrose of the root system (in mol of sucrose),
        and the total dry structural mass of the root system (in g of dry structural mass).
        :param g: the investigated MTG
        :return: total_sucrose_root (mol of sucrose), total_struct_mass (g of dry structural mass)
        (untouched)
        """
        # We initialize the values to 0:
        self.total_sucrose_root[1] = 0.
        self.total_living_struct_mass[1] = 0.

        # We cover all the vertices in the MTG, whether they are dead or not:
        for vid in self.g.vertices_iter(scale=1):
            if self.length <= 0.:
                continue
            else:
                # We increment the total amount of sucrose in the root system, including dead root elements (if any):
                self.total_sucrose_root += self.C_sucrose_root[vid] * (
                            self.struct_mass[vid] + self.living_root_hairs_struct_mass[vid]) \
                                           - self.Deficit_sucrose_root[vid]

                # We only select the elements that have a positive struct_mass and are not dead
                # (if they have just died, we still include them in the balance):
                if self.struct_mass[vid] > 0. and self.type[vid] != "Dead" and self.type[vid] != "Just_dead":
                    # We calculate the total living struct_mass by summing all the local struct_masses:
                    self.total_living_struct_mass += self.struct_mass + self.living_root_hairs_struct_mass

    # Calculating the net input of sucrose by the aerial parts into the root system:
    # ------------------------------------------------------------------------------
    def shoot_sucrose_supply_and_spreading(self):
        """
        This function calculates the new root sucrose concentration (mol of sucrose per gram of dry root structural mass)
        AFTER supplying sucrose from the shoot.
        (nearly untouched)
        """

        # We calculate the remaining amount of sucrose in the root system,
        # based on the current sucrose concentration and structural mass of each root element:
        self.total_root_sucrose_and_living_struct_mass()
        # Note: The total sucrose is the total amount of sucrose present in the root system, including local deficits
        # but excluding the possible global deficit of the whole root system, which is considered below.
        # The total struct_mass corresponds to the structural mass of the living roots & hairs
        # (including those that have just died and can still release sucrose!).

        # We use a global variable recorded outside this function that corresponds to the possible deficit of sucrose
        # (in moles of sucrose) of the whole root system calculated at the previous time_step:
        # Note that this value is stored in the base node of the root system.
        if self.global_sucrose_deficit[1] > 0.:
            print("!!! Before homogenizing sucrose concentration, the global deficit in sucrose was",
                  self.global_sucrose_deficit[1])
        # The new average sucrose concentration in the root system is calculated as:
        C_sucrose_root_after_supply = (self.total_sucrose_root[1] + (
                    self.sucrose_input_rate * self.time_steps_in_seconds) - self.global_sucrose_deficit[1]) \
                                      / self.total_living_struct_mass[1]
        # This new concentration includes the amount of sucrose from element that have just died,
        # but excludes the mass of these dead elements!

        if C_sucrose_root_after_supply >= 0.:
            new_C_sucrose_root = C_sucrose_root_after_supply
            # We reset the global variable global_sucrose_deficit:
            self.global_sucrose_deficit[1] = 0.
        else:
            # We record the general deficit in sucrose:
            self.global_sucrose_deficit[1] = - C_sucrose_root_after_supply * self.total_living_struct_mass[1]

            print("!!! After homogenizing sucrose concentration, the deficit in sucrose is",
                  self.global_sucrose_deficit[1])
            # We defined the new concentration of sucrose as 0:
            new_C_sucrose_root = 0.

        # We go through the MTG to modify the sugars concentrations:
        for vid in self.g.vertices_iter(scale=1):
            # If the element has not emerged yet, it doesn't contain any sucrose yet;
            # if has died, it should not contain any sucrose anymore:
            if self.length[vid] <= 0. or self.type[vid] == "Dead" or self.type[vid] == "Just_dead":
                self.C_sucrose_root[vid] = 0.
            else:
                # The local sucrose concentration in the root is calculated from the new sucrose concentration calculated above:
                self.C_sucrose_root[vid] = new_C_sucrose_root
            # AND BECAUSE THE LOCAL DEFICITS OF SUCROSE HAVE BEEN ALREADY INCLUDED IN THE TOTAL SUCROSE CALCULATION,
            # WE RESET ALL LOCAL DEFICITS TO 0:
            self.Deficit_sucrose_root[vid] = 0.
            self.Deficit_sucrose_root_rate[vid] = 0.

    def actualize_concentrations_struct_mass(self):
        return

    # Unloading of sucrose from the phloem and conversion of sucrose into hexose:

    # --------------------------------------------------------------------------
    # These 3 functions simulate the rate of sucrose unloading from phloem and its immediate conversion into hexose.
    # The hexose pool is assumed to correspond to the symplastic compartment of root cells from the stelar and cortical
    # parenchyma and from the epidermis. The function also simulates the opposite process of sucrose loading,
    # considering that 2 mol of hexose are produced for 1 mol of sucrose.

    def process_hexose_diffusion_from_phloem(self, length, exchange_surface, C_sucrose_root, C_hexose_root,
                                             hexose_consumption_by_growth_rate, soil_temperature_in_Celsius, **kwargs):
        # We consider all the cases where no net exchange should be allowed:
        if length <= 0. or exchange_surface <= 0. or type == "Just_dead" or type == "Dead":
            return 0

        else:
            global_sucrose_deficit = self.g.property('global_sucrose_deficit')[self.g.root]
            if (global_sucrose_deficit > 0.) or (C_sucrose_root <= C_hexose_root / 2.):
                return 0
            else:
                phloem_permeability = kwargs["phloem_permeability"] \
                                      * (1 + hexose_consumption_by_growth_rate / kwargs[
                    "reference_rate_of_hexose_consumption_by_growth"])
                phloem_permeability *= self.temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                     process_at_T_ref=1,
                                                                     T_ref=kwargs["phloem_unloading_T_ref"],
                                                                     A=kwargs["phloem_unloading_A"],
                                                                     B=kwargs["phloem_unloading_B"],
                                                                     C=kwargs["phloem_unloading_C"])
                return max(2. * phloem_permeability * (C_sucrose_root - C_hexose_root / 2.) * exchange_surface, 0)

    def process_hexose_active_production_from_phloem(self, length, exchange_surface, C_sucrose_root, C_hexose_root,
                                                     hexose_consumption_by_growth_rate, soil_temperature_in_Celsius,
                                                     **kwargs):
        # We consider all the cases where no net exchange should be allowed:
        if length <= 0. or exchange_surface <= 0. or type == "Just_dead" or type == "Dead":
            return 0

        else:
            global_sucrose_deficit = self.g.property('global_sucrose_deficit')[self.g.root]
            if (global_sucrose_deficit > 0.) or (C_sucrose_root <= C_hexose_root / 2.):
                return 0
            else:
                max_unloading_rate = kwargs["max_unloading_rate"] \
                                     * (1 + hexose_consumption_by_growth_rate / kwargs[
                    "reference_rate_of_hexose_consumption_by_growth"])
                max_unloading_rate *= self.temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                    process_at_T_ref=1,
                                                                    T_ref=kwargs["phloem_unloading_T_ref"],
                                                                    A=kwargs["phloem_unloading_A"],
                                                                    B=kwargs["phloem_unloading_B"],
                                                                    C=kwargs["phloem_unloading_C"])
                return max(2. * max_unloading_rate * C_sucrose_root * exchange_surface / (
                            kwargs["Km_unloading"] + C_sucrose_root), 0)

    def process_sucrose_loading(self, soil_temperature_in_Celsius, exchange_surface, C_hexose_root, **kwargs):
        # # We correct the max loading rate according to the distance from the tip in the middle of the segment.
        # max_loading_rate = param.surfacic_loading_rate_reference \
        #     * (1. - 1. / (1. + ((distance_from_tip-length/2.) / original_radius) ** param.gamma_loading))
        # TODO: Reconsider the way the variation of the max loading rate along the root axis has been described!

        # We correct loading according to soil temperature:
        max_loading_rate = kwargs["max_loading_rate"] \
                           * self.temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=param.max_loading_rate_T_ref,
                                                           A=param.max_loading_rate_A,
                                                           B=param.max_loading_rate_B,
                                                           C=param.max_loading_rate_C)
        if C_hexose_root <= 0.:
            return 0.
        else:
            return max(0.5 * max_loading_rate * exchange_surface * C_hexose_root / (param.Km_loading + C_hexose_root),
                       0.)

    # Unloading of sucrose from the phloem and conversion of sucrose into hexose:
    # --------------------------------------------------------------------------

    def process_mobilization_from_hexose(self, length, C_hexose_root, C_hexose_reserve, type, struct_mass,
                                         living_root_hairs_struct_mass, soil_temperature_in_Celsius, **kwargs):
        # CALCULATIONS OF THEORETICAL IMMOBILIZATION RATE:
        # We verify that the element does not correspond to a primordium that has not emerged:
        if length <= 0. or C_hexose_root < 0. or C_hexose_reserve < 0. or type == "Root_nodule":
            return 0.
        else:
            # We correct maximum rates according to soil temperature:
            corrected_max_mobilization_rate = kwargs["max_mobilization_rate"] * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["max_mobilization_rate_T_ref"],
                A=kwargs["max_mobilization_rate_A"],
                B=kwargs["max_mobilization_rate_B"],
                C=kwargs["max_mobilization_rate_C"])
            if C_hexose_reserve <= kwargs["C_hexose_reserve_min"]:
                return 0.
            else:
                return corrected_max_mobilization_rate * C_hexose_reserve / (
                        kwargs["Km_mobilization"] + C_hexose_reserve) * (struct_mass + living_root_hairs_struct_mass)

    def process_immobilization_as_reserve(self, C_hexose_root, C_hexose_reserve, type, soil_temperature_in_Celsius,
                                          struct_mass, living_root_hairs_struct_mass, **kwargs):
        # CALCULATIONS OF THEORETICAL IMMOBILIZATION RATE:
        if C_hexose_root <= kwargs["C_hexose_root_min_for_reserve"] or C_hexose_reserve >= kwargs[
            "C_hexose_reserve_max"] \
                or type == "Just_dead" or type == "Dead":
            return 0.
        else:
            corrected_max_immobilization_rate = kwargs["max_immobilization_rate"] \
                                                * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["max_immobilization_rate_T_ref"],
                A=kwargs["max_immobilization_rate_A"],
                B=kwargs["max_immobilization_rate_B"],
                C=kwargs["max_immobilization_rate_C"])

            return corrected_max_immobilization_rate * C_hexose_root / (param.Km_immobilization + C_hexose_root) * (
                    struct_mass + living_root_hairs_struct_mass)

    # Function calculating maintenance respiration:
    # ----------------------------------------------

    # This function calculates the rate of respiration (mol of CO2 per second) corresponding to the consumption
    # of a part of the local hexose pool to cover the costs of maintenance processes, i.e. any biological process in the
    # root that is NOT linked to the actual growth of the root. The calculation is derived from the model of Thornley and
    # Cannell (2000), who initially used this formalism to describe the residual maintenance costs that could not be
    # accounted for by known processes. The local amount of CO2 respired for maintenance is calculated relatively
    # to the structural dry mass of the element n and is regulated by a Michaelis-Menten function of the local
    # concentration of hexose.

    def process_maintenance_respiration(self, type, C_hexose_root, soil_temperature_in_Celsius, struct_mass,
                                        living_root_hairs_struct_mass, **kwargs):

        # TODO FOR TRISTAN: Consider expliciting a respiration cost associated to the exchange of N in the root (e.g. cost for uptake, xylem loading)?

        # CONSIDERING CASES THAT SHOULD BE AVOIDED:
        # We consider that dead elements cannot respire (unless over the first time step following death,
        # i.e. when the type is "Just_dead"):
        if type == "Dead" or C_hexose_root <= 0.:
            return 0.
        else:
            # We correct the maximal respiration rate according to soil temperature:
            corrected_resp_maintenance_max = kwargs["resp_maintenance_max"] \
                                             * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["resp_maintenance_max_T_ref"],
                A=kwargs["resp_maintenance_max_A"],
                B=kwargs["resp_maintenance_max_B"],
                C=kwargs["resp_maintenance_max_C"])

            return corrected_resp_maintenance_max * C_hexose_root / (kwargs["Km_maintenance"] + C_hexose_root) * (
                    struct_mass + living_root_hairs_struct_mass)

    # Exudation of hexose from the root into the soil:
    # ------------------------------------------------
    # This function computes the rate of hexose exudation (mol of hexose per seconds) for a given root element.
    # Exudation corresponds to an efflux of hexose from the root to the soil by a passive diffusion. This efflux
    # is calculated from the product of the exchange surface (m2) with the soil solution, a permeability coefficient (g m-2)
    # and the gradient of hexose concentration between cells and soil (mol of hexose per gram of dry root structural mass).

    # TODO : report to anatomy module
    # We calculate the total surface of exchange between symplasm and apoplasm in the root parenchyma, modulated by the
    # conductance of cell walls (reduced in the meristematic zone) and the conductances of endodermis and exodermis
    # barriers (when these barriers are mature, conductance is expected to be 0 in general, and part of the symplasm is
    # not accessible anymore to the soil solution:
    # non_vascular_exchange_surface = (S_epid + S_hairs) + cond_walls * (cond_exo * S_cortex + cond_endo * S_stele)
    # vascular_exchange_surface = cond_walls * cond_exo * cond_endo * S_vessels

    def process_hexose_exudation(self, length, non_vascular_exchange_surface, C_hexose_root,
                                 soil_temperature_in_Celsius,
                                 C_hexose_soil, **kwargs):
        if length <= 0 or non_vascular_exchange_surface <= 0. or C_hexose_root <= 0.:
            return 0.
        else:
            corrected_P_max_apex = kwargs["Pmax_apex"] \
                                   * self.temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                   process_at_T_ref=1,
                                                                   T_ref=kwargs["permeability_coeff_T_ref"],
                                                                   A=kwargs["permeability_coeff_A"],
                                                                   B=kwargs["permeability_coeff_B"],
                                                                   C=kwargs["permeability_coeff_C"])
            # if distance_from_tip < length:
            #     print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
            # else:
            #     corrected_permeability_coeff = corrected_P_max_apex \
            #                          / (1 + (distance_from_tip - length / 2.) / original_radius) ** param.gamma_exudation
            corrected_permeability_coeff = corrected_P_max_apex

            return max(corrected_permeability_coeff * (C_hexose_root - C_hexose_soil) * non_vascular_exchange_surface,
                       0)

    def process_phloem_hexose_exudation(self, length, non_vascular_exchange_surface, C_hexose_root,
                                        soil_temperature_in_Celsius, C_sucrose_root, C_hexose_soil,
                                        vascular_exchange_surface, **kwargs):
        if length <= 0 or non_vascular_exchange_surface <= 0. or C_hexose_root <= 0.:
            return 0.
        else:
            corrected_P_max_apex = kwargs["Pmax_apex"] \
                                   * self.temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                   process_at_T_ref=1,
                                                                   T_ref=kwargs["permeability_coeff_T_ref"],
                                                                   A=kwargs["permeability_coeff_A"],
                                                                   B=kwargs["permeability_coeff_B"],
                                                                   C=kwargs["permeability_coeff_C"])
            # if distance_from_tip < length:
            #     print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
            # else:
            #     corrected_permeability_coeff = corrected_P_max_apex \
            #                          / (1 + (distance_from_tip - length / 2.) / original_radius) ** param.gamma_exudation
            corrected_permeability_coeff = corrected_P_max_apex
            return corrected_permeability_coeff * (2 * C_sucrose_root - C_hexose_soil) * vascular_exchange_surface

    # Uptake of hexose from the soil by the root:
    # -------------------------------------------
    # This function computes the rate of hexose uptake by roots from the soil. This influx of hexose is represented
    # as an active process with a substrate-limited relationship (Michaelis-Menten function) depending on the hexose
    # concentration in the soil.

    def process_hexose_uptake_from_soil(self, length, non_vascular_exchange_surface, C_hexose_soil, type,
                                        soil_temperature_in_Celsius, **kwargs):
        if length <= 0 or non_vascular_exchange_surface <= 0. or C_hexose_soil <= 0. or type == "Just_dead" or type == "Dead":
            return 0.
        else:
            corrected_uptake_rate_max = kwargs["uptake_rate_max"] \
                                        * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["uptake_rate_max_T_ref"],
                A=kwargs["uptake_rate_max_A"],
                B=kwargs["uptake_rate_max_B"],
                C=kwargs["uptake_rate_max_C"])

            return corrected_uptake_rate_max * non_vascular_exchange_surface \
                * C_hexose_soil / (kwargs["Km_uptake"] + C_hexose_soil)

    def process_phloem_hexose_uptake_from_soil(self, length, vascular_exchange_surface, C_hexose_soil, type,
                                               soil_temperature_in_Celsius, **kwargs):
        if length <= 0 or vascular_exchange_surface <= 0. or C_hexose_soil <= 0. or type == "Just_dead" or type == "Dead":
            return 0.
        else:
            corrected_uptake_rate_max = kwargs["uptake_rate_max"] \
                                        * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["uptake_rate_max_T_ref"],
                A=kwargs["uptake_rate_max_A"],
                B=kwargs["uptake_rate_max_B"],
                C=kwargs["uptake_rate_max_C"])

            return corrected_uptake_rate_max * vascular_exchange_surface * C_hexose_soil / (
                    kwargs["Km_uptake"] + C_hexose_soil)

    # Mucilage secretion:
    # ------------------
    # This function computes the rate of mucilage secretion (in mol of equivalent hexose per second) for a given root element n.
    def process_mucilage_secretion(self, length, exchange_surface, C_hexose_root, type, distance_from_tip,
                                   soil_temperature_in_Celsius, original_radius, Cs_mucilage_soil, **kwargs):
        # First, we ensure that the element has a positive length and surface of exchange:
        if length <= 0 or exchange_surface <= 0. or C_hexose_root <= 0. or type == "Dead" or type == "Stopped" or distance_from_tip < length:
            return 0.
        else:
            # We correct the maximal secretion rate according to soil temperature
            # (This could to a bell-shape where the maximum is obtained at 27 degree Celsius,
            # as suggested by Morré et al. (1967) for maize mucilage secretion):
            corrected_secretion_rate_max = kwargs["secretion_rate_max"] * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["secretion_rate_max_T_ref"],
                A=kwargs["secretion_rate_max_A"],
                B=kwargs["secretion_rate_max_B"],
                C=kwargs["secretion_rate_max_C"]
                # We also regulate the rate according to the potential accumulation of mucilage around the root:
                # the rate is maximal when no mucilage accumulates around, and linearily decreases with the concentration
                # of mucilage at the soil-root interface, until reaching 0 when the concentration is equal or higher than
                # the maximal concentration soil (NOTE: Cs_mucilage_soil is expressed in mol of equivalent hexose per m2 of
                # external surface):
            ) / (
                                                   (1 + (distance_from_tip - length / 2.) / original_radius) ** kwargs[
                                               "gamma_secretion"]
                                           ) * (
                                                   param.Cs_mucilage_soil_max - Cs_mucilage_soil) / kwargs[
                                               "Cs_mucilage_soil_max"]
            # TODO: Validate this linear decrease until reaching the max surfacic density.

            return max(corrected_secretion_rate_max * exchange_surface * C_hexose_root / (
                    kwargs["Km_secretion"] + C_hexose_root), 0.)

    # Release of root cells:
    # ----------------------
    # This function computes the rate of release of epidermal or cap root cells (in mol of equivalent hexose per second)
    # into the soil for a given root element. The rate of release linearily decreases with increasing length from the tip,
    # until reaching 0 at the end of the elongation zone. The external concentration of root cells (mol of equivalent-
    # hexose per m2) is also linearily decreasing the release of cells at the interface.

    def process_cells_release(self, length, exchange_surface, C_hexose_root, type, distance_from_tip, radius,
                              soil_temperature_in_Celsius, Cs_cells_soil, **kwargs):
        # First, we ensure that the element has a positive length and surface of exchange:
        if length <= 0 or exchange_surface <= 0. or C_hexose_root <= 0. or type == "Just_dead" or type == "Dead" or type == "Stopped":
            return 0.
        else:
            # We modify the maximal surfacic release rate according to the mean distance to the tip (in the middle of the
            # root element), assuming that the release decreases linearily with the distance to the tip, until reaching 0
            # when the this distance becomes higher than the growing zone length:
            if distance_from_tip < kwargs["growing_zone_factor"] * radius:
                average_distance = distance_from_tip - length / 2.
                reduction = (kwargs["growing_zone_factor"] * radius - average_distance) \
                            / (kwargs["growing_zone_factor"] * radius)
                cells_surfacic_release = kwargs["surfacic_cells_release_rate"] * reduction
            # In the special case where the end of the growing zone is located somewhere on the root element:
            elif distance_from_tip - length < kwargs["growing_zone_factor"] * radius:
                average_distance = (distance_from_tip - length) + (kwargs["growing_zone_factor"] * radius
                                                                   - (distance_from_tip - length)) / 2.
                reduction = (kwargs["growing_zone_factor"] * radius - average_distance) \
                            / (kwargs["growing_zone_factor"] * radius)
                cells_surfacic_release = kwargs["surfacic_cells_release_rate"] * reduction
            # Otherwise, there is no cells release:
            else:
                return 0.

            # We correct the release rate according to soil temperature:
            corrected_cells_surfacic_release = cells_surfacic_release * self.temperature_modification(
                temperature_in_Celsius=soil_temperature_in_Celsius,
                process_at_T_ref=1,
                T_ref=kwargs["surfacic_cells_release_rate_T_ref"],
                A=kwargs["surfacic_cells_release_rate_A"],
                B=kwargs["surfacic_cells_release_rate_B"],
                C=kwargs["surfacic_cells_release_rate_C"])

            # We also regulate the surface release rate according to the potential accumulation of cells around the root:
            # the rate is maximal when no cells are around, and linearily decreases with the concentration of cells
            # in the soil, until reaching 0 when the concentration is equal or higher than the maximal concentration in the
            # soil (NOTE: Cs_cells_soil is expressed in mol of equivalent hexose per m2 of external surface):
            corrected_cells_surfacic_release = corrected_cells_surfacic_release * (
                    kwargs["Cs_cells_soil_max"] - Cs_cells_soil) / kwargs["Cs_cells_soil_max"]
            # TODO: Validate this linear decrease until reaching the max surfacic density.

            # The release of cells by the root is then calculated according to this surface:
            # TODO: Are we sure that cells release is not dependent on C availability?
            return max(exchange_surface * corrected_cells_surfacic_release, 0.)

    # These methods calculate the time derivative (dQ/dt) of the amount in each pool, for a given root element, based on
    # a C balance.
    # TODO account for struct mass evolution in the update. When is it updated?
    # TODO FOR TRISTAN: Consider adding N balance here (to be possibly used in the solver).
    def update_sucrose_root(self, sucrose_root, struct_mass, hexose_production_from_phloem,
                            phloem_hexose_exudation, sucrose_loading_in_phloem,
                            phloem_hexose_uptake_from_soil, deficit_sucrose_root, **kwargs):
        return sucrose_root + (self.time_steps_in_seconds / struct_mass) * (
                - hexose_production_from_phloem / 2.
                - phloem_hexose_exudation / 2.
                + sucrose_loading_in_phloem
                + phloem_hexose_uptake_from_soil / 2.
                - deficit_sucrose_root)

    def update_hexose_reserve(self, hexose_reserve, struct_mass, hexose_immobilization_as_reserve,
                              hexose_mobilization_from_reserve, deficit_hexose_reserve, **kwargs):
        return hexose_reserve * (self.time_steps_in_seconds / struct_mass) * (
                hexose_immobilization_as_reserve
                - hexose_mobilization_from_reserve
                - deficit_hexose_reserve)

    def update_hexose_root(self, hexose_root, struct_mass, hexose_exudation, hexose_uptake_from_soil,
                           mucilage_secretion, cells_release, resp_maintenance,
                           hexose_consumption_by_growth, hexose_production_from_phloem,
                           sucrose_loading_in_phloem, hexose_mobilization_from_reserve,
                           hexose_immobilization_as_reserve, deficit_hexose_root, **kwargs):
        return hexose_root * (self.time_steps_in_seconds / struct_mass) * (
                - hexose_exudation
                + hexose_uptake_from_soil
                - mucilage_secretion
                - cells_release
                - resp_maintenance / 6.
                - hexose_consumption_by_growth
                + hexose_production_from_phloem
                - 2. * sucrose_loading_in_phloem
                + hexose_mobilization_from_reserve
                - hexose_immobilization_as_reserve
                - deficit_hexose_root)

    # # TODO report to soil
    # def calculating_time_derivatives_of_the_amount_in_each_pool(n):
    #
    #     # We calculate the derivative of the amount of hexose in the soil pool:
    #     y_derivatives['hexose_soil'] = \
    #         - n.hexose_degradation_rate \
    #         + n.hexose_exudation_rate - n.hexose_uptake_from_soil_rate \
    #         + n.phloem_hexose_exudation_rate - n.phloem_hexose_uptake_from_soil_rate \
    #         - n.Deficit_hexose_soil_rate
    #
    #     # We calculate the derivative of the amount of mucilage in the soil pool:
    #     y_derivatives['mucilage_soil'] = \
    #         + n.mucilage_secretion_rate \
    #         - n.mucilage_degradation_rate \
    #         - n.Deficit_mucilage_soil_rate
    #
    #     # We calculate the derivative of the amount of root cells in the soil pool:
    #     y_derivatives['cells_soil'] = \
    #         + n.cells_release_rate \
    #         - n.cells_degradation_rate \
    #         - n.Deficit_cells_soil_rate
    #
    #     return y_derivatives

    # This function adjusts possibly negative concentrations by setting them to 0 and by recording the corresponding deficit.
    # WATCH OUT: This function must be called once the concentrations have already been calculated with the previous deficits!
    def deficit_sucrose_root(self, C_sucrose_root, struct_mass, living_root_hairs_struct_mass):
        if C_sucrose_root < 0:
            return - C_sucrose_root * (struct_mass + living_root_hairs_struct_mass) / self.time_steps_in_seconds
        else:
            # TODO : or None could be more efficient?
            return 0.

    def deficit_hexose_reserve(self, hexose_reserve, struct_mass, living_root_hairs_struct_mass):
        if hexose_reserve < 0:
            return - hexose_reserve * (struct_mass + living_root_hairs_struct_mass) / self.time_steps_in_seconds
        else:
            return 0.

    def deficit_hexose_root(self, hexose_reserve, struct_mass, living_root_hairs_struct_mass):
        if hexose_reserve < 0:
            return - hexose_reserve * (struct_mass + living_root_hairs_struct_mass) / self.time_steps_in_seconds
        else:
            return 0.

        # TODO map partial max(0, ) to some properties after deficit storage.

        # def adjusting_pools_and_deficits(self, n, time_step_in_seconds, printing_warnings=False):
        # TODO : report to soil model
        # # Looking at hexose in the soil:
        # if n.C_hexose_soil < 0:
        #     if printing_warnings:
        #         print("WARNING: After balance, there is a deficit of soil hexose for element", n.index(),
        #               "that corresponds to", n.Deficit_hexose_soil,
        #               "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        #     # We define a positive deficit (mol of hexose) based on the negative concentration:
        #     n.Deficit_hexose_soil = -n.C_hexose_soil * (n.struct_mass + n.living_root_hairs_struct_mass)
        #     n.Deficit_hexose_soil_rate = n.Deficit_hexose_soil / time_step_in_seconds
        #     # And we set the concentration to 0:
        #     n.C_hexose_soil = 0.
        #
        # # Looking at mucilage in the soil:
        # if n.Cs_mucilage_soil < 0:
        #     if printing_warnings:
        #         print("WARNING: After balance, there is a deficit of soil mucilage for element", n.index(),
        #               "that corresponds to", n.Deficit_mucilage_soil,
        #               "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        #     # We define a positive deficit (mol of equivalent-hexose) based on the negative concentration:
        #     n.Deficit_mucilage_soil = -n.Cs_mucilage_soil * (n.external_surface + n.living_root_hairs_external_surface)
        #     n.Deficit_mucilage_soil_rate = n.Deficit_mucilage_soil / time_step_in_seconds
        #     # And we set the concentration to 0:
        #     n.Cs_mucilage_soil = 0.
        #
        # # Looking at the root cells in the soil:
        # if n.Cs_cells_soil < 0:
        #     if printing_warnings:
        #         print("WARNING: After balance, there is a deficit of root cells in the soil for element", n.index(),
        #               "that corresponds to", n.Deficit_cells_soil,
        #               "; the concentration has been set to 0 and the deficit will be included in the next balance.")
        #     # We define a positive deficit (mol of hexose-equivalent) based on the negative concentration:
        #     n.Deficit_cells_soil = -n.Cs_cells_soil * (n.external_surface + n.living_root_hairs_external_surface)
        #     n.Deficit_cells_soil_rate = n.Deficit_cells_soil / time_step_in_seconds
        #     # And we set the concentration to 0:
        #     n.Cs_cells_soil = 0.
        return

    def control_of_anomalies(self):
        """
        The function contol_of_anomalies checks for the presence of elements with negative measurable properties (e.g. length, concentrations).
        (untouched)
        """

        # CHECKING THAT UNEMERGED ROOT ELEMENTS DO NOT CONTAIN CARBON:
        # We cover all the vertices in the MTG:
        for vid in self.g.vertices_iter(scale=1):
            # n represents the vertex:
            n = self.g.node(vid)
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

    # TODO adapt to class structure
    class Differential_Equation_System(object):

        # TODO FOR TRISTAN: Consider adding N-related variables here to be computed together with C fluxes...

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
            index = 0  # The index will be used to attribute the right property in the list corresponding to y
            y_mapping = {}  # y_mapping is a dictionnary associating a unique index for each variable in y
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

            # We create a time grid of the simulation that will be used by the solver (NOTE: here it only contains 0 and the
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
            # ----------------------------------------------------

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
            # -------------------------------------------

            # 1) Hexose concentrations are expressed relative to the structural mass of the root element (unlike mucilage or cells!):
            mass = (self.n.initial_struct_mass + self.n.initial_living_root_hairs_struct_mass)
            # We make sure that the mass is positive:
            if isnan(mass) or mass <= 0.:
                # If the initial mass is not positive, it could be that a new axis just emerged, so its initial mass was 0.
                # In this case, we set the concentrations to 0, as no fluxes (except Deficit) should be calculated before
                # growth is finished and sucrose is supplied there!
                self.n.C_hexose_root = 0.
                self.n.C_hexose_reserve = 0.
                self.n.C_hexose_soil = 0.
                # In the case the element did not emerge (i.e. its age is higher that the normal simulation time step),
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
            if isnan(surface) or surface <= 0.:
                # If the initial surface is not positive, it could be that a new axis just emerged.
                # In this case, we set the concentrations to 0, as no fluxes (except Deficit) should be calculated before
                # growth is finished and sucrose is supplied there!
                self.n.Cs_mucilage_soil = 0.
                self.n.Cs_cells_soil = 0.
                # In the case the element did not emerge (i.e. its age is higher that the normal simulation time step),
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
            self.calculating_all_growth_independent_fluxes(self.g, self.n, self.soil_temperature_in_Celsius,
                                                           self.printing_warnings)

            # Calculation of time derivatives:
            # ---------------------------------
            # We get a list of all derivatives of the amount in each pool, i.e. their rate of evolution over time (dQ/dt):
            time_derivatives = self.calculating_time_derivatives_of_the_amount_in_each_pool(self.n)
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
                print("!!! For element", self.n.index(),
                      "the initial mass before updating the initial conditions in the solver was NA!")
                mass = 0.
            # We calculate the external surface to which some concentrations are related:
            surface = (self.n.initial_external_surface + self.n.initial_living_root_hairs_external_surface)
            if isnan(surface):
                print("!!! For element", self.n.index(),
                      "the initial external surface before updating the initial conditions in the solver was NA!")
                surface = 0.

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
                    self.initial_conditions[self.y_variables_mapping[variable_name]] = getattr(self.n,
                                                                                               concentration_name) \
                                                                                       * surface
                else:
                    # Otherwise the amount in the pool is calculated from the concentration and the mass:
                    concentration_name = "C_" + variable_name
                    self.initial_conditions[self.y_variables_mapping[variable_name]] = getattr(self.n,
                                                                                               concentration_name) \
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
                            t_span=self.time_grid_in_seconds,
                            # Note that you could impose to the solver a number of different time points to be solved here!
                            y0=self.initial_conditions,
                            # method='BDF',
                            method='LSODA',
                            # t_eval=np.array([self.time_step_in_seconds]), # Select this to get only the final quantities at the end of the time step
                            # t_eval=np.linspace(0, self.time_step_in_seconds, 10), # Select this to get time points regularly distributed within the time step
                            t_eval=None,
                            # Select t_eval=None to get automatical time points within the current macro time step
                            min_step=60,  # OPTIONAL: defines the minimal micro time step (0 by default)
                            dense_output=False
                            # OPTIONAL: defines whether the solution should be continuous or not (False by default)
                            )

            # OPTIONAL: We can print the results of the different iterations at some of the micro time steps!
            if self.printing_solver_outputs:
                try:
                    # print(self.n.type, "-", self.n.label, "-", self.n.index())
                    solver_times = pd.DataFrame(sol.t)
                    solver_times.columns = ['Time']
                    solver_y = pd.DataFrame(sol.y)
                    solver_y = solver_y.transpose()
                    solver_y.columns = self.y_variables_mapping.keys()
                    solver_results = pd.concat([solver_times, solver_y], axis=1)
                    print(solver_results)
                    print("")
                except:
                    print(
                        "   > PROBLEM: Not able to interpret the solver results! Here was the message given by the solver:")
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
    # ------------------------------------------------------
    # NOTE: The function alls all processes of C exchange between pools on each root element and performs a new C balance,
    # with or without using a solver:
    # def C_exchange_and_balance_in_roots_and_at_the_root_soil_interface(self,
    #                                                                    time_step_in_seconds=1. * (60. * 60. * 24.),
    #                                                                    using_solver=False,
    #                                                                    printing_solver_outputs=False,
    #                                                                    printing_warnings=False):
    #     """
    #
    #     :param g:
    #     :param time_step_in_seconds: the time step over which the balance is done
    #     :param soil_temperature_in_Celsius:
    #     :param using_solver:
    #     :param printing_solver_outputs:
    #     :param printing_warnings:
    #     :return:
    #     """
    #
    #     # TODO FOR TRISTAN: Consider updating this function with N fluxes, or doing a similar function separated from here.
    #     # TODO : map with conditions on length or struct mass?
    #     # We initialize a tip concentration:
    #     tip_C_hexose_root = -1
    #
    #     # We cover all the vertices in the MTG:
    #     for vid in self.g.vertices_iter(scale=1):
    #
    #         # n represents the vertex:
    #         n = self.g.node(vid)
    #
    #         # First, we ensure that the element has a positive length:
    #         if n.length <= 0.:
    #             continue
    #
    #         # # IF NOT DONE ELSEWHERE - We calculate an average flux of C consumption due to growth:
    #         # n.hexose_consumption_by_growth_rate = n.hexose_consumption_by_growth / time_step_in_seconds
    #
    #         # OPTION 1: WITHOUT SOLVER
    #         ##########################
    #
    #         if not using_solver:
    #             # TODO C : Separate flows precisely
    #             # We simply calculate all related fluxes and the new concentrations based on the initial conditions
    #             # at the beginning of the time step.
    #
    #             # Calculating all new fluxes and related quantities:
    #             # ---------------------------------------------------
    #
    #             # We calculate all C-related fluxes (independent of the time step):
    #             #DONE self.calculating_all_growth_independent_fluxes(self.g, n, self.soil_temperature, printing_warnings)
    #
    #             # We then calculate the quantities that correspond to these fluxes (dependent of the time step):
    #             # REPORTED IN UPDATE self.calculating_amounts_from_fluxes(n, time_step_in_seconds)
    #
    #             # Calculating the new variations of the quantities in each pool over time:
    #             # -------------------------------------------------------------------------
    #             # We call a dictionary containing the time derivative (dQ/dt) of the amount present in each pool,
    #             # based on the C balance:
    #             # DONE y_time_derivatives = self.calculating_time_derivatives_of_the_amount_in_each_pool(n)
    #
    #             # Calculating new concentrations based on C balance:
    #             # ---------------------------------------------------
    #
    #             # WATCH OUT: Below, the possible deficits are not included, since they have been already taken into account
    #             # as rates in the function "calculating_time_derivatives_of_the_amount_in_each_pool(n)" called above!
    #
    #             # TODO : get insights from bellow for mass actualization in update methods
    #
    #             # # We calculate the new concentration of sucrose in the root according to sucrose conversion into hexose:
    #             # sucrose_root_derivative = y_time_derivatives["sucrose_root"] * time_step_in_seconds
    #             # n.C_sucrose_root = (n.C_sucrose_root * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
    #             #                     + sucrose_root_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)
    #             #
    #             # # We calculate the new concentration of hexose in the root cytoplasm:
    #             # hexose_root_derivative = y_time_derivatives["hexose_root"] * time_step_in_seconds
    #             # n.C_hexose_root = (n.C_hexose_root * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
    #             #                    + hexose_root_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)
    #             #
    #             # # We calculate the new concentration of hexose in the reserve:
    #             # hexose_reserve_derivative = y_time_derivatives["hexose_reserve"] * time_step_in_seconds
    #             # n.C_hexose_reserve = (n.C_hexose_reserve * (
    #             #             n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
    #             #                       + hexose_reserve_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)
    #
    #             # TODO : report to soil
    #             # # We calculate the new concentration of hexose in the soil:
    #             # hexose_soil_derivative = y_time_derivatives["hexose_soil"] * time_step_in_seconds
    #             # n.C_hexose_soil = (n.C_hexose_soil * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
    #             #                    + hexose_soil_derivative) / (n.struct_mass + n.living_root_hairs_struct_mass)
    #             #
    #             # # We calculate the new concentration of hexose in the soil:
    #             # mucilage_soil_derivative = y_time_derivatives["mucilage_soil"] * time_step_in_seconds
    #             # n.Cs_mucilage_soil = (n.Cs_mucilage_soil * (n.initial_external_surface
    #             #                                             + n.initial_living_root_hairs_external_surface)
    #             #                       + mucilage_soil_derivative) / (
    #             #                                  n.external_surface + n.living_root_hairs_external_surface)
    #             #
    #             # # We calculate the new concentration of cells in the soil:
    #             # cells_soil_derivative = y_time_derivatives["cells_soil"] * time_step_in_seconds
    #             # n.Cs_cells_soil = (n.Cs_cells_soil * (n.initial_external_surface
    #             #                                       + n.initial_living_root_hairs_external_surface)
    #             #                    + cells_soil_derivative) / (
    #             #                               n.external_surface + n.living_root_hairs_external_surface)
    #             pass
    #
    #         # OPTION 2: WITH SOLVER
    #         #######################
    #         # We use a numeric solver to calculate the best equilibrium between pools over time, so that the fluxes
    #         # still correspond to plausible values even when the time step is too large:
    #         if using_solver:
    #
    #             if printing_solver_outputs:
    #                 print("Considering for the solver the element", n.index(), "of length", n.length, "...")
    #
    #             # We use the class corresponding to the system of differential equations and its resolution:
    #             System = self.Differential_Equation_System(self.g, n,
    #                                                   time_step_in_seconds,
    #                                                   self.soil_temperature_in_Celsius,
    #                                                   printing_warnings=printing_warnings,
    #                                                   printing_solver_outputs=printing_solver_outputs)
    #             System.run()
    #             # At this stage, the amounts in each pool pool have been updated by the solver, as well as the amount
    #             # exchanged between pools corresponding to specific processes.
    #
    #             # We recalculate the new final concentrations based on (i) quantities at the end of the solver,
    #             # and (ii) struct_mass (instead of initial_struct_mass as used in the solver):
    #             n.C_hexose_root = n.hexose_root / (n.struct_mass + n.living_root_hairs_struct_mass)
    #             n.C_hexose_reserve = n.hexose_reserve / (n.struct_mass + n.living_root_hairs_struct_mass)
    #             n.C_hexose_soil = n.hexose_soil / (n.struct_mass + n.living_root_hairs_struct_mass)
    #             n.Cs_mucilage_soil = n.mucilage_soil / (n.external_surface + n.living_root_hairs_external_surface)
    #             n.Cs_cells_soil = n.cells_soil / (n.external_surface + n.living_root_hairs_external_surface)
    #
    #             # SPECIAL CASE FOR SUCROSE:
    #             # As sucrose pool was not included in the solver, we update C_sucrose_root based on the mean exchange rates
    #             # calculated over the time_step.
    #             # We calculate the initial amount of sucrose in the root element:
    #             initial_sucrose_root_amount = n.C_sucrose_root \
    #                                           * (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass)
    #             # We call again a dictionary containing the time derivative (dQ/dt) of the amount present in each pool, and
    #             # in particular sucrose, based on the balance between different rates (NOTE: the transfer of sucrose
    #             # between elements through the phloem is not considered at this stage!):
    #             y_time_derivatives = self.calculating_time_derivatives_of_the_amount_in_each_pool(n)
    #             estimated_sucrose_root_derivative = y_time_derivatives['sucrose_root'] * time_step_in_seconds
    #             # Eventually, the new concentration of sucrose in the root element is calculated:
    #             n.C_sucrose_root = (initial_sucrose_root_amount + estimated_sucrose_root_derivative) \
    #                                / (n.struct_mass + n.living_root_hairs_struct_mass)
    #
    #         # WITH BOTH OPTIONS, AFTER ALL PROCESSES HAVE BEEN COMPUTED:
    #         ############################################################
    #
    #         # Updating concentrations and deficits:
    #         # --------------------------------------
    #         # We make sure that new concentration in each pool is not negative - otherwise we set it to 0 and record the
    #         # corresponding deficit to balance the next calculation of the concentration:
    #         # DONE self.adjusting_pools_and_deficits(n, time_step_in_seconds, printing_warnings)
    #
    #         #  Calculation of additional variables:
    #         # -------------------------------------
    #         # MOVED TO TOOLS self.calculating_extra_variables(n, time_step_in_seconds)
    #
    #     #     # SPECIAL CASE: we record the property of the apex of the primary root
    #     #     # ---------------------------------------------------------------------
    #     #     # If the element corresponds to the apex of the primary root:
    #     #     if n.radius == param.D_ini / 2. and n.label == "Apex":
    #     #         # Then the function will give its specific concentration of mobile hexose:
    #     #         tip_C_hexose_root = n.C_hexose_root
    #     #
    #     # # We return the concentration of hexose in the apex of the primary root:
    #     # return tip_C_hexose_root
