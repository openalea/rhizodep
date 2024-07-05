#  -*- coding: utf-8 -*-

"""
    rhizodep.model_growth
    ~~~~~~~~~~~~~

    The module :mod:`rhizodep.model_growth` defines the equations of root architectured growth.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import numpy as np
import pandas as pd
from math import sqrt, pi, floor
from dataclasses import dataclass

from openalea.mtg import *
from openalea.mtg.traversal import post_order
from openalea.mtg import turtle as turt

from metafspm.component import Model, declare
from metafspm.component_factory import *


family = "growth"


@dataclass
class RootGrowthModel(Model):
    """
    DESCRIPTION
    -----------
    Root growth model originating from Rhizodep shoot.py

    forked :
        https://forgemia.inra.fr/tristan.gerault/rhizodep/-/commits/rhizodep_2022?ref_type=heads
    base_commit :
        92a6f7ad927ffa0acf01aef645f9297a4531878c
    """

    family = family

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---
    # FROM SOIL MODEL
    soil_temperature: float = declare(default=15, unit="°C", unit_comment="", description="soil temperature in contact with roots", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    root_tissue_density: float = declare(default=0.10 * 1e6, unit="g.m-3", unit_comment="of structural mass", description="root_tissue_density", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---
    type: str = declare(default="Normal_root_after_emergence", unit="", unit_comment="", description="Example segment type provided by root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="", edit_by="user")
    radius: float = declare(default=3.5e-4, unit="m", unit_comment="", description="Example root segment radius", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="intensive", edit_by="user")
    z1: float = declare(default=0., unit="m", unit_comment="", description="Depth of the segment tip computed by plantGL, colar side", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="intensive", edit_by="user")
    z2: float = declare(default=0., unit="m", unit_comment="", description="Depth of the segment tip computed by plantGL, apex side", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="intensive", edit_by="user")
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    struct_mass: float = declare(default=1.35e-4, unit="g", unit_comment="", description="Example root segment structural mass", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    #TODO parameter!
    initial_struct_mass: float = declare(default=1.35e-4, unit="g", unit_comment="", description="Same as struct_mass but corresponds to the previous time step; it is intended to record the variation", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    living_root_hairs_struct_mass: float = declare(default=0., unit="g", unit_comment="", description="Example root segment living root hairs structural mass", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    root_hair_length: float = declare(default=1.e-3, unit="m", unit_comment="", description="Example root hair length", 
                                                    min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between 0.5 and 1.3 mm.", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    total_root_hairs_number: float = declare(default=30 * (1.6e-4 / 3.5e-4) * 3.e-3 * 1e3, unit="adim", unit_comment="", description="Example root hairs number on segment external surface", 
                                                    min_value="", max_value="", value_comment="30 * (1.6e-4 / radius) * length * 1e3", references=" According to the work of Gahoonia et al. (1997), the root hair density is about 30 hairs per mm for winter wheat, for a root radius of about 0.16 mm.", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    hexose_consumption_by_growth: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Hexose consumption rate by growth is coupled to a root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    distance_from_tip: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example distance from tip", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="intensive", edit_by="user")
    volume: float = declare(default=1e-9, unit="m3", unit_comment="", description="Initial volume of the collar element", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    struct_mass_produced: float = declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="extensive", edit_by="user")
    thermal_time_since_emergence: float = declare(default=0, unit="°C", unit_comment="", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="intensive", edit_by="user")
                                                    

    # --- INITIALIZES MODEL PARAMETERS ---
    # Segment initialization
    D_ini: float = declare(default=0.8e-3, unit="m", unit_comment="", description="Initial tip diameter of the primary root", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hair_radius: float = declare(default=12 * 1e-6 /2., unit="m", unit_comment="", description="Average radius of root hair", 
                                                    min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair diameter is relatively constant for different genotypes of wheat and barley, i.e. 12 microns", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_lifespan: float = declare(default=46 * (60. * 60.), unit="s", unit_comment="time equivalent at temperature of T_ref", description="Average lifespan of a root hair", 
                                                    min_value="", max_value="", value_comment="", references="According to the data from McElgunn and Harrison (1969), the lifespan of wheat root hairs is 40-55h, depending on the temperature. For a temperature of 20 degree Celsius, the linear regression from this data gives 46h.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_density: float = declare(default=30 * 1e3 / (0.16 / 2. * 1e-3), unit=".m-2", unit_comment="number of hairs par meter of root per meter of root radius", description="Average density of root hairs", 
                                                    min_value="", max_value="", value_comment="", references="According to the data from McElgunn and Harrison (1969), the elongation rate for wheat root hairs is about 0.080 mm h-1.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hair_max_length: float = declare(default=1 * 1e-3, unit="m", unit_comment="", description="Average maximal length of a root hair", 
                                                    min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between 0.5 and 1.3 mm.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_elongation_rate: float = declare(default=0.080 * 1e-3 / (60. * 60.) /(12 * 1e-6 /2.), unit=".s-1", unit_comment="in meter per second per meter of root radius", description="Average elongation rate of root hairs", 
                                                    min_value="", max_value="", value_comment="", references="According to the data from McElgunn and Harrison (1969), the elongation rate for wheat root hairs is about 0.080 mm h-1.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    LDs: float = declare(default=4000. * (60. * 60. * 24.) * 1000 * 1e-6, unit="s.m-1..g-1.m-3", unit_comment="time equivalent at temperature of T_ref", description="Average lifespan of a root hair", 
                                                    min_value="", max_value="", value_comment="", references="5000 day mm-1 g-1 cm3 (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    ER: float = declare(default=0.2 / (60. * 60. * 24.), unit=".s-1", unit_comment="time equivalent at temperature of T_ref", description="Emission rate of adventitious roots", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    n_seminal_roots: int = declare(default=5, unit="adim", unit_comment="", description="Maximal number of roots emerging from the base (including primary and seminal roots)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    n_adventitious_roots: int = declare(default=10, unit="adim", unit_comment="", description="Maximal number of roots emerging from the base (including primary and seminal roots)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    random_choice: float = declare(default=8, unit="adim", unit_comment="", description="We set the random seed, so that the same simulation can be repeted with the same seed:", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    D_sem_to_D_ini_ratio: float = declare(default=0.95, unit="adim", unit_comment="", description="Proportionality coefficient between the tip diameter of a seminal root and D_ini", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    CVDD: float = declare(default=0.2, unit="adim", unit_comment="", description="Relative variation of the daughter root diameter", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    starting_time_for_adventitious_roots_emergence: float = declare(default=(60. * 60. * 24.) * 9., unit="s", unit_comment="time equivalent at temperature of T_ref", description="Time when adventitious roots start to successively emerge", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    D_adv_to_D_ini_ratio: float = declare(default=0.8, unit="adim", unit_comment="", description="Proportionality coefficient between the tip diameter of an adventitious root and D_ini ", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # Temperature
    process_at_T_ref: float = declare(default=1., unit="adim", unit_comment="", description="Proportion of maximal process intensity occuring at T_ref", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    T_ref: float = declare(default=0., unit="°C", unit_comment="", description="the reference temperature", 
                                                    min_value="", max_value="", value_comment="", references="We assume that relative growth is 0 at T_ref=0 degree Celsius, and linearily increases to reach 1 at 20 degree.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    A: float = declare(default=1/20, unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    B: float = declare(default=0, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    C: float = declare(default=0, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # C supply for elongation
    growing_zone_factor: float = declare(default=8 * 2., unit="adim", unit_comment="", description="Proportionality factor between the radius and the length of the root apical zone in which C can sustain root elongation", 
                                                    min_value="", max_value="", value_comment="", references="According to illustrations by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap, meristem and elongation zones is about 8 times the diameter of the tip.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # potential development
    emergence_delay: float = declare(default=3.27 * (60. * 60. * 24.), unit="s", unit_comment="time equivalent at temperature of T_ref", description="Delay of emergence of the primordium", 
                                                    min_value="", max_value="", value_comment="", references="emergence_delay = 3 days (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    EL: float = declare(default=6.5e-4, unit="s-1", unit_comment="meters of root per meter of radius per second equivalent to T_ref_growth", description="Slope of the elongation rate = f(tip diameter) ", 
                                                    min_value="", max_value="", value_comment="", references="EL = 5 mm mm-1 day-1 (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Km_elongation: float = declare(default=1250 * 1e-6 / 6., unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for root elongation", 
                                                    min_value="", max_value="", value_comment="", references="According to Barillot et al. (2016b): Km for root growth is 1250 umol C g-1 for sucrose. According to Gauthier et al (2020): Km for regulation of the RER by sucrose concentration in hz = 100-150 umol C g-1", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    relative_nodule_thickening_rate_max: float = declare(default=20. / 100. / (24. * 60. * 60.), unit="s-1", unit_comment="", description="Maximal rate of relative increase in nodule radius", 
                                                    min_value="", max_value="", value_comment="", references="We consider that the radius can't increase by more than 20% every day (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Km_nodule_thickening: float = declare(default=1250 * 1e-6 / 6. * 100, unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for nodule thickening", 
                                                    min_value="", max_value="", value_comment="Km_elongation * 100", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    nodule_max_radius: float = declare(default=0.8e-3 * 20., unit="m", unit_comment="", description="Maximal radius of nodule", 
                                                    min_value="", max_value="", value_comment="Dini * 10", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    SGC: float = declare(default=0.0, unit="adim", unit_comment="", description="Proportionality coefficient between the section area of the segment and the sum of distal section areas", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    relative_root_thickening_rate_max: float = declare(default=5. / 100. / (24. * 60. * 60.), unit="s-1", unit_comment="", description="Maximal rate of relative increase in root radius", 
                                                    min_value="", max_value="", value_comment="", references="We consider that the radius can't increase by more than 5% every day (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Km_thickening: float = declare(default=1250 * 1e-6 / 6., unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for root thickening", 
                                                    min_value="", max_value="", value_comment="Km_elongation", references="We assume that the Michaelis-Menten constant for thickening is the same as for root elongation. (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # actual growth
    struct_mass_C_content: float = declare(default=0.44 / 12.01, unit="mol.g-1", unit_comment="of carbon", description="C content of structural mass", 
                                                    min_value="", max_value="", value_comment="", references="We assume that the structural mass contains 44% of C. (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    yield_growth: float = declare(default=0.8, unit="adim", unit_comment="mol of CO2 per mol of C used for structural mass", description="Growth yield", 
                                                    min_value="", max_value="", value_comment="", references="We use the range value (0.75-0.85) proposed by Thornley and Cannell (2000)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # Segmentation and primordium formation
    segment_length: float = declare(default=3. / 1000., unit="m", unit_comment="", description="Length of a segment", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    nodule_formation_probability: float = declare(default=0.5, unit="m", unit_comment="", description="Probability (between 0 and 1) of nodule formation for each apex that elongates", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Dmin: float = declare(default=0.122 / 1000., unit="m", unit_comment="", description="Minimal threshold tip diameter (i.e. the diameter of the finest observable roots)", 
                                                    min_value="", max_value="", value_comment="", references="Dmin=0.05 mm (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    RMD: float = declare(default=0.57, unit="adim", unit_comment="", description="Average ratio of the diameter of the daughter root to that of the mother root", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    IPD: float = declare(default=0.00474, unit="m", unit_comment="", description="Inter-primordia distance", 
                                                    min_value="", max_value="", value_comment="", references="IPD = 7.6 mm (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    new_root_tissue_density: float = declare(default=0.10 * 1e6, unit="g.m3", unit_comment="of structural mass", description="root_tissue_density", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # Growth durations
    GDs: float = declare(default=800 * (60. * 60. * 24.) * 1000. ** 2., unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="Coefficient of growth duration", 
                                                    min_value="", max_value="", value_comment="", references="Reference: GDs=930. day mm-2 (Pagès et Picon-Cochard, 2014)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    main_roots_growth_extender: float = declare(default=100., unit="s.s-1", unit_comment="", description="Coefficient of growth duration extension, by which the theoretical growth duration is multiplied for seminal and adventitious roots", 
                                                    min_value="", max_value="", value_comment="", references="Reference: GDs=400. day mm-2 ()", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_highest: float = declare(default=60 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="For seminal and adventitious roots, a longer growth duration is applied", 
                                                    min_value="", max_value="", value_comment="Expected growth duration of a seminal root", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_high: float = declare(default=6 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="The growth duration has a probability of [1-GD_prob_medium] to equal GD_high", 
                                                    min_value="", max_value="", value_comment="", references="Estimated the longest observed lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_medium: float = declare(default=0.70 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="The growth duration has a probability of [GD_prob_medium - GD_prob_low] to equal GD_medium", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the medium lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_low: float = declare(default=0.25 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="The growth duration has a probability of [GD_prob_low] to equal GD_low", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the shortest lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_by_frequency: bool = declare(default=False, unit="adim", unit_comment="", description="As an alternative to using a single value of growth duration depending on diameter, we offer the possibility to rather define the growth duration as a random choice between three values (low, medium and high), depending on their respective probability", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_prob_low: float = declare(default=0.50, unit="adim", unit_comment="", description="Probability for low growth duration", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the shortest lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_prob_medium: float = declare(default=0.85, unit="adim", unit_comment="Probability for medium growth duration", description="Coefficient of growth duration", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the medium lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # --- USER ORIENTED PARAMETERS FOR SIMULATION ---
    # initiate MTG
    random: bool = declare(default=True, unit="adim", unit_comment="", description="Allow random processes in growth", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    ArchiSimple: bool = declare(default=False, unit="adim", unit_comment="", description="Allow growth according to the original Archisimple model", 
                                                    min_value="", max_value="", value_comment="", references="(Pagès et al., 2014)", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    initial_segment_length: float = declare(default=1e-3, unit="m", unit_comment="", description="Initial segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    initial_apex_length: float = declare(default=1e-4, unit="m", unit_comment="", description="Initial apex length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    initial_C_hexose_root: float = declare(default=1e-4, unit="mol.g-1", unit_comment="", description="Initial hexose concentration of root segments", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    input_file_path: str = declare(default="C:/Users/frees/rhizodep/src/rhizodep/", unit="m", unit_comment="", description="Filepath for input files", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    forcing_seminal_roots_events: bool = declare(default=False, unit="m", unit_comment="", description="a Boolean expliciting if seminal root events should be forced", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    seminal_roots_events_file: str = declare(default="seminal_roots_inputs.csv", unit="m", unit_comment="", description="Filepath pointing to input table to plan seminal root emergence event", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    forcing_adventitious_roots_events: bool = declare(default=False, unit="m", unit_comment="", description="a Boolean expliciting if adventicious root events should be forced", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    adventitious_roots_events_file: str = declare(default="adventitious_roots_inputs.csv", unit="adim", unit_comment="", description="Filepath pointing to input table to plan adventitious root emergence event", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    radial_growth: str = declare(default="Possible", unit="adim", unit_comment="", description="equivalent to a Boolean expliciting whether radial growth should be considered or not", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    nodules: bool = declare(default=False, unit="adim", unit_comment="", description="a Boolean expliciting whether nodules could be formed or not", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_order_limitation: bool = declare(default=False, unit="adim", unit_comment="", description="a Boolean expliciting whether lateral roots should be prevented above a certain root order", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_order_treshold: int = declare(default=2, unit="adim", unit_comment="", description="the root order above which new lateral roots cannot be formed", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="simulation_parameter", by="model_growth", state_variable_type="", edit_by="user")

    def __init__(self, g=None, time_step_in_seconds: int=3600, **scenario: dict):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param time_step_in_seconds: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        
        if g is None:
            self.g = self.initiate_mtg()
        else:
            self.g = g

        self.props = self.g.properties()
        self.time_step_in_seconds = time_step_in_seconds
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step_in_seconds, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        for name in self.state_variables:
            # We do this to avoid having the initiate_mtg rewritten each time a new state variable is defined.
            if name not in self.props.keys():
                self.props.setdefault(name, {})
                self.props[name].update({key: getattr(self, name) for key in self.vertices})
            setattr(self, name, self.props[name])

    def initiate_mtg(self):
        """
        This functions generates a root MTG from nothing, containing only one segment of a specific length,
        terminated by an apex (preferably of length 0).
        :return: g: an initiated root MTG
        """

        # TODO# FOR TRISTAN: Add all initial N-related variables that need to be explicited before N fluxes computation.

        # We create a new MTG called g:
        g = MTG()

        # We first add one initial element:
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

        base_radius = self.D_ini / 2.

        base_segment.angle_down = 0
        base_segment.angle_roll = 0
        base_segment.length = self.initial_segment_length
        base_segment.radius = base_radius
        base_segment.original_radius = base_radius
        base_segment.initial_length = self.initial_segment_length
        base_segment.initial_radius = base_radius

        base_segment.root_hair_radius = self.root_hair_radius
        base_segment.root_hair_length = 0.
        base_segment.actual_length_with_hairs = 0.
        base_segment.living_root_hairs_number = 0.
        base_segment.dead_root_hairs_number = 0.
        base_segment.total_root_hairs_number = 0.

        base_segment.actual_time_since_root_hairs_emergence_started = 0.
        base_segment.thermal_time_since_root_hairs_emergence_started = 0.
        base_segment.actual_time_since_root_hairs_emergence_stopped = 0.
        base_segment.thermal_time_since_root_hairs_emergence_stopped = 0.
        base_segment.all_root_hairs_formed = False
        base_segment.root_hairs_lifespan = self.root_hairs_lifespan

        base_segment.root_hairs_struct_mass = 0.
        base_segment.root_hairs_struct_mass_produced = 0.
        base_segment.living_root_hairs_struct_mass = 0.
        base_segment.distance_from_tip = base_segment.length
        base_segment.former_distance_from_tip = base_segment.length
        base_segment.dist_to_ramif = 0.
        base_segment.actual_elongation = base_segment.length
        base_segment.actual_elongation_rate = 0

        base_segment.volume = self.volume

        base_segment.struct_mass = base_segment.volume * self.new_root_tissue_density
        base_segment.initial_struct_mass = base_segment.struct_mass
        base_segment.initial_living_root_hairs_struct_mass = base_segment.living_root_hairs_struct_mass

        # We define the initial concentrations:
        base_segment.C_hexose_root = self.initial_C_hexose_root

        # Fluxes:
        # --------
        base_segment.resp_growth = 0.
        base_segment.hexose_growth_demand = 0.
        base_segment.hexose_possibly_required_for_elongation = 0.
        base_segment.hexose_consumption_by_growth_amount = 0.

        # Rates:
        # -------
        base_segment.resp_growth_rate = 0.
        base_segment.hexose_growth_demand_rate = 0.
        base_segment.hexose_consumption_by_growth = 0.

        # Time indications:
        # ------------------
        base_segment.growth_duration = self.GDs * (2. * base_radius) ** 2 * self.main_roots_growth_extender #WATCH OUT!!! we artificially multiply growth duration for seminal and adventious roots!
        # base_segment.growth_duration = self.calculate_growth_duration(radius=base_radius, index=id_segment, root_order=1)
        base_segment.life_duration = self.LDs * (2. * base_radius) * self.new_root_tissue_density
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
        # ----------------------------------------------------
        # If there is more than one seminal root (i.e. roots already formed in the seed):
        if self.n_seminal_roots > 1 or not self.forcing_seminal_roots_events:

            # We read additional parameters that are stored in a CSV file, with one column containing the delay for each
            # emergence event, and the second column containing the number of seminal roots that have to emerge at each event:
            # We try to access an already-existing CSV file:
            seminal_inputs_path = os.path.join(self.input_file_path, self.seminal_roots_events_file)
            # If the file doesn't exist, we construct a new table using the specified parameters:
            if not os.path.exists(seminal_inputs_path) or self.forcing_seminal_roots_events:
                print("NOTE: no CSV file describing the apparitions of seminal roots can be used!")
                print("=> We therefore built a table according to the parameters 'n_seminal_roots' and 'ER'.")
                print("")
                # We initialize an empty data frame:
                seminal_inputs_file = pd.DataFrame()
                # We define a list that will contain the successive thermal times corresponding to root emergence:
                list_time = [x * 1 / self.ER for x in range(1, self.n_seminal_roots)]

                # We define another list containing only "1" as the number of roots to be emerged for each event:
                list_number = np.ones(self.n_seminal_roots - 1, dtype='int8')
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
                for j in range(0, seminal_inputs_file.number_of_seminal_roots_per_event[i]):

                    # We make sure that the seminal roots will have different random insertion angles:
                    np.random.seed(self.random_choice + i * j)

                    # Then we form one supporting segment of length 0 + one primordium of seminal root.
                    # We add one new segment without any length on the same axis as the base:
                    segment = self.ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Segment',
                                                               type='Support_for_seminal_root',
                                                               root_order=1,
                                                               angle_down=0,
                                                               angle_roll=abs(np.random.normal(180, 180)),
                                                               length=0.,
                                                               radius=base_radius,
                                                               identical_properties=False,
                                                               nil_properties=True)

                    # We define the radius of a seminal root according to the parameter Di:
                    if self.random:
                        radius_seminal = abs(np.random.normal(self.D_ini / 2. * self.D_sem_to_D_ini_ratio,
                                                              self.D_ini / 2. * self.D_sem_to_D_ini_ratio * self.CVDD))
                    else:
                        radius_seminal = self.D_ini / 2. * self.D_sem_to_D_ini_ratio

                    # And we add one new primordium of seminal root on the previously defined segment:
                    apex_seminal = self.ADDING_A_CHILD(mother_element=segment, edge_type='+', label='Apex',
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
                    apex_seminal.growth_duration = self.GDs * (2. * radius_seminal) ** 2 * self.main_roots_growth_extender
                    # apex_seminal.growth_duration = self.calculate_growth_duration(radius=radius_seminal,
                    #                                                                            index=apex_seminal.index(),
                    #                                                                            root_order=1)
                    apex_seminal.life_duration = self.LDs * (2. * radius_seminal) * apex_seminal.root_tissue_density

                    # We defined the delay of emergence for the new primordium:
                    apex_seminal.emergence_delay_in_thermal_time = seminal_inputs_file.emergence_delay_in_thermal_time[i]

        # ADDING THE PRIMORDIA OF ALL POSSIBLE ADVENTIOUS ROOTS:
        # ------------------------------------------------------
        # If there should be more than one main root (i.e. adventitious roots formed at the basis):
        if self.n_adventitious_roots > 0 or not self.forcing_adventitious_roots_events:

            # We read additional parameters from a table, with one column containing the delay for each emergence event,
            # and the second column containing the number of adventitious roots that have to emerge at each event.
            # We try to access an already-existing CSV file:
            adventitious_inputs_path = os.path.join(self.input_file_path, self.adventitious_roots_events_file)
            # If the file doesn't exist, we construct a new table using the specified parameters:
            if not os.path.exists(adventitious_inputs_path) or self.forcing_adventitious_roots_events:
                print("NOTE: no CSV file describing the apparitions of adventitious roots can be used!")
                print("=> We therefore built a table according to the parameters 'n_adventitious_roots' and 'ER'.")
                print("")
                # We initialize an empty data frame:
                adventitious_inputs_file = pd.DataFrame()
                # We define a list that will contain the successive thermal times corresponding to root emergence:
                list_time = [self.starting_time_for_adventitious_roots_emergence + x * 1 / self.ER
                             for x in range(0, self.n_adventitious_roots)]
                # We define another list containing only "1" as the number of roots to be emerged for each event:
                list_number = np.ones(self.n_adventitious_roots, dtype='int8')
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
                    np.random.seed(self.random_choice + i * j * 3)

                    # Then we form one supporting segment of length 0 + one primordium of seminal root.
                    # We add one new segment without any length on the same axis as the base:
                    segment = self.ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Segment',
                                                               type='Support_for_adventitious_root',
                                                               root_order=1,
                                                               angle_down=0,
                                                               angle_roll=abs(np.random.normal(0, 180)),
                                                               length=0.,
                                                               radius=base_radius,
                                                               identical_properties=False,
                                                               nil_properties=True)

                    # We define the radius of a adventitious root according to the parameter Di:
                    if self.random:
                        radius_adventitious = abs(np.random.normal(self.D_ini / 2. * self.D_adv_to_D_ini_ratio,
                                                                   self.D_ini / 2. * self.D_adv_to_D_ini_ratio *
                                                                   self.CVDD))
                    else:
                        radius_adventitious = self.D_ini / 2. * self.D_adv_to_D_ini_ratio

                    # And we add one new primordium of adventitious root on the previously defined segment:
                    apex_adventitious = self.ADDING_A_CHILD(mother_element=segment, edge_type='+',
                                                                         label='Apex',
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
                    apex_adventitious.growth_duration = self.GDs * (2. * radius_adventitious) ** 2 * self.main_roots_growth_extender
                    # apex_adventitious.growth_duration = self.calculate_growth_duration(
                    #     radius=radius_adventitious,
                    #     index=apex_adventitious.index(),
                    #     root_order=1)
                    apex_adventitious.life_duration = self.LDs * (2. * radius_adventitious) * apex_adventitious.root_tissue_density

                    # We defined the delay of emergence for the new primordium:
                    apex_adventitious.emergence_delay_in_thermal_time \
                        = adventitious_inputs_file.emergence_delay_in_thermal_time[i]

        # FINAL APEX CONFIGURATION AT THE END OF THE MAIN ROOT:
        # ------------------------------------------------------
        apex = self.ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Apex',
                                                type='Normal_root_after_emergence',
                                                root_order=1,
                                                angle_down=0,
                                                angle_roll=0,
                                                length=self.initial_apex_length,
                                                radius=base_radius,
                                                identical_properties=False,
                                                nil_properties=True)
        apex.original_radius = apex.radius
        apex.initial_radius = apex.radius
        apex.growth_duration = self.GDs * (2. * base_radius) ** 2 * self.main_roots_growth_extender
        # apex.growth_duration = self.calculate_growth_duration(radius=base_radius,
        #                                                                    index=apex.index(),
        #                                                                    root_order=1)
        apex.life_duration = self.LDs * (2. * base_radius) * apex.root_tissue_density

        if self.initial_apex_length <= 0.:
            apex.C_hexose_root = 0.
        else:
            apex.C_hexose_root = self.initial_C_hexose_root

        apex.volume = self.volume
        apex.struct_mass = apex.volume * apex.root_tissue_density
        apex.initial_struct_mass = apex.struct_mass
        apex.initial_living_root_hairs_struct_mass = apex.living_root_hairs_struct_mass

        return g

    # SUBDIVISIONS OF THE SCHEDULING LOOP
    # -----------------------------------
    @stepinit
    def reinitializing_growth_variables(self):
        """
        This function re-initializes different growth-related variables (e.g. potential growth variables).

        :return:
        """
        # We cover all the vertices in the MTG:
        for vid in self.g.vertices_iter(scale=1):
            # n represents the vertex:
            n = self.g.node(vid)

            # We set to 0 the growth-related variables:
            n.hexose_consumption_by_growth_amount = 0.
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
        return

    # Function that calculates the potential growth of the whole MTG at a given time step:
    @potential
    @state
    def potential_growth(self):
        """
        This function covers the whole root MTG and computes the potential growth of segments and apices.
        :return:
        """
        # We simulate the development of all apices and segments in the MTG:
        for vid in self.g.vertices_iter(scale=1):
            n = self.g.node(vid)
            if n.label == "Apex":
                self.potential_apex_development(apex=n)
            elif n.label == "Segment":
                self.potential_segment_development(segment=n)

    # Function calculating the potential development of an apex:
    def potential_apex_development(self, apex):
        """
        This function considers a root apex, i.e. the terminal root element of a root axis (including the primordium of a
        root that has not emerged yet), and calculates its potential elongation, without actually elongating the apex or
        forming any new root primordium in the standard case (only when ArchiSimple option is set to True). Aging of the
        apex is also considered.
        :param apex: the apex to be considered
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
        temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=apex.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)

        # CASE 1: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING SEMINAL OR ADVENTITIOUS ROOT
        # -----------------------------------------------------------------------------------------------------
        # If the seminal root has not emerged yet:
        if apex.type == "Seminal_root_before_emergence" or apex.type == "Adventitious_root_before_emergence":
            # If the time elapsed since the last emergence of seminal root is higher than the prescribed interval time:
            if ( apex.thermal_time_since_primordium_formation + self.time_step_in_seconds * temperature_time_adjustment) >= apex.emergence_delay_in_thermal_time:
                # The potential time elapsed since seminal root's possible emergence is calculated:
                apex.thermal_potential_time_since_emergence = apex.thermal_time_since_primordium_formation + self.time_step_in_seconds * temperature_time_adjustment \
                                                              - apex.emergence_delay_in_thermal_time
                # If the apex could have emerged sooner:
                if apex.thermal_potential_time_since_emergence > self.time_step_in_seconds * temperature_time_adjustment:
                    # The time since emergence is reduced to the time elapsed during this time step:
                    apex.thermal_potential_time_since_emergence = self.time_step_in_seconds * temperature_time_adjustment

                # We record the different elements that can contribute to the C supply necessary for growth,
                # and we calculate a mean concentration of hexose in this supplying zone:
                self.calculating_supply_for_elongation(element=apex)
                # The corresponding potential elongation of the apex is calculated:
                apex.potential_length = self.elongated_length(element=apex, initial_length=apex.initial_length,
                                                              radius=apex.initial_radius,
                                                              C_hexose_root=apex.growing_zone_C_hexose_root,
                                                              elongation_time_in_seconds=apex.thermal_potential_time_since_emergence)
                # Last, if ArchiSimple has been chosen as the growth model:
                if self.ArchiSimple:
                    # Then we automatically allow the root to emerge, without consideration of C limitation:
                    apex.type = "Normal_root_after_emergence"
            # In any case, the time since primordium formation is incremented, as usual:
            apex.actual_time_since_primordium_formation += self.time_step_in_seconds
            apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
            # And the new element returned by the function corresponds to the potentially emerging apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

        # CASE 2: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING NORMAL LATERAL ROOT
        # ---------------------------------------------------------------------------------------------
        if apex.type == "Normal_root_before_emergence":
            # If the time since primordium formation is higher than the delay of emergence:
            if apex.thermal_time_since_primordium_formation + self.time_step_in_seconds * temperature_time_adjustment > self.emergence_delay:
                # The time since primordium formation is incremented:
                apex.actual_time_since_primordium_formation += self.time_step_in_seconds
                apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
                # The potential time elapsed at the end of this time step since the emergence is calculated:
                apex.thermal_potential_time_since_emergence = apex.thermal_time_since_primordium_formation - self.emergence_delay
                # If the apex could have emerged sooner:
                if apex.thermal_potential_time_since_emergence > self.time_step_in_seconds * temperature_time_adjustment:
                    # The time since emergence is equal to the time elapsed during this time step (since it must have emerged at this time step):
                    apex.thermal_potential_time_since_emergence = self.time_step_in_seconds * temperature_time_adjustment
                # We record the different element that can contribute to the C supply necessary for growth,
                # and we calculate a mean concentration of hexose in this supplying zone:
                self.calculating_supply_for_elongation(element=apex)
                # The corresponding elongation of the apex is calculated:
                apex.potential_length = self.elongated_length(element=apex, initial_length=apex.initial_length,
                                                              radius=apex.initial_radius,
                                                              C_hexose_root=apex.growing_zone_C_hexose_root,
                                                              elongation_time_in_seconds=apex.thermal_potential_time_since_emergence)

                # If ArchiSimple has been chosen as the growth model:
                if self.ArchiSimple:
                    apex.type = "Normal_root_after_emergence"
                    new_apex.append(apex)
                    # And the function returns this new apex and stops here:
                    return new_apex
                # Otherwise, we control the actual emergence of this primordium through the management of the parent:
                else:
                    # We select the parent on which the primordium has been formed:
                    vid = apex.index()
                    index_parent = self.g.Father(vid, EdgeType='+')
                    parent = self.g.node(index_parent)
                    # The possibility of emergence of a lateral root from the parent is recorded inside the parent:
                    parent.lateral_root_emergence_possibility = "Possible"
                    parent.lateral_primordium_index = apex.index()
                    # And the new element returned by the function corresponds to the potentially emerging apex:
                    new_apex.append(apex)
                    # And the function returns this new apex and stops here:
                    return new_apex
            # Otherwise, the time since primordium formation is simply incremented:
            else:
                apex.actual_time_since_primordium_formation += self.time_step_in_seconds
                apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
                # And the new element returned by the function corresponds to the modified apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

        # CASE 3: THE APEX BELONGS TO AN AXIS THAT HAS ALREADY EMERGED:
        # --------------------------------------------------------------
        # IF THE APEX CAN CONTINUE GROWING:
        if apex.thermal_time_since_emergence + self.time_step_in_seconds * temperature_time_adjustment < apex.growth_duration:
            # The times are incremented:
            apex.actual_time_since_primordium_formation += self.time_step_in_seconds
            apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
            apex.actual_time_since_emergence += self.time_step_in_seconds
            apex.thermal_time_since_emergence += self.time_step_in_seconds * temperature_time_adjustment
            # We record the different element that can contribute to the C supply necessary for growth,
            # and we calculate a mean concentration of hexose in this supplying zone:
            self.calculating_supply_for_elongation(element=apex)
            # The corresponding potential elongation of the apex is calculated:
            apex.potential_length = self.elongated_length(element=apex, initial_length=apex.length, radius=apex.radius,
                                                          C_hexose_root=apex.growing_zone_C_hexose_root,
                                                          elongation_time_in_seconds=self.time_step_in_seconds * temperature_time_adjustment)
            # And the new element returned by the function corresponds to the modified apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

        # OTHERWISE, THE APEX HAD TO STOP:
        else:
            # IF THE APEX HAS NOT REACHED ITS LIFE DURATION:
            if apex.thermal_time_since_growth_stopped + self.time_step_in_seconds * temperature_time_adjustment < apex.life_duration:
                # IF THE APEX HAS ALREADY BEEN STOPPED AT A PREVIOUS TIME STEP:
                if apex.type == "Stopped" or apex.type == "Just_stopped":
                    # The time since growth stopped is simply increased by one time step:
                    apex.actual_time_since_growth_stopped += self.time_step_in_seconds
                    apex.thermal_time_since_growth_stopped += self.time_step_in_seconds * temperature_time_adjustment
                    # The type is (re)declared "Stopped":
                    apex.type = "Stopped"
                    # The times are incremented:
                    apex.actual_time_since_primordium_formation += self.time_step_in_seconds
                    apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
                    apex.actual_time_since_emergence += self.time_step_in_seconds
                    apex.thermal_time_since_emergence += self.time_step_in_seconds * temperature_time_adjustment
                    apex.actual_time_since_cells_formation += self.time_step_in_seconds
                    apex.thermal_time_since_cells_formation += self.time_step_in_seconds * temperature_time_adjustment
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
                                                             + self.time_step_in_seconds * temperature_time_adjustment \
                                                             - apex.growth_duration
                    apex.actual_time_since_growth_stopped = apex.thermal_time_since_growth_stopped / temperature_time_adjustment

                    # We record the different element that can contribute to the C supply necessary for growth,
                    # and we calculate a mean concentration of hexose in this supplying zone:
                    self.calculating_supply_for_elongation(element=apex)
                    # And the potential elongation of the apex before growth stopped is calculated:
                    apex.potential_length = self.elongated_length(element=apex, initial_length=apex.length, radius=apex.radius,
                                                                  C_hexose_root=apex.growing_zone_C_hexose_root,
                                                                  elongation_time_in_seconds=self.time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped)
                    # VERIFICATION:
                    if self.time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped < 0.:
                        print("!!! ERROR: The apex", apex.index(), "has stopped since",
                              apex.actual_time_since_growth_stopped,
                              "seconds; the time step is", self.time_step_in_seconds)
                        print("We set the potential length of this apex equal to its initial length.")
                        apex.potential_length = apex.initial_length

                    # The times are incremented:
                    apex.actual_time_since_primordium_formation += self.time_step_in_seconds
                    apex.actual_time_since_emergence += self.time_step_in_seconds
                    apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_emergence += self.time_step_in_seconds * temperature_time_adjustment
                    apex.actual_time_since_cells_formation += self.time_step_in_seconds
                    apex.thermal_time_since_cells_formation += self.time_step_in_seconds * temperature_time_adjustment
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
                    apex.actual_time_since_primordium_formation += self.time_step_in_seconds
                    apex.actual_time_since_emergence += self.time_step_in_seconds
                    apex.actual_time_since_cells_formation += self.time_step_in_seconds
                    apex.actual_time_since_growth_stopped += self.time_step_in_seconds
                    apex.actual_time_since_death += self.time_step_in_seconds
                    apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_emergence += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_cells_formation += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_growth_stopped += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_death += self.time_step_in_seconds * temperature_time_adjustment
                    # The new element returned by the function corresponds to this apex:
                    new_apex.append(apex)
                    # And the function returns this new apex and stops here:
                    return new_apex
                # OTHERWISE, THE APEX HAS TO DIE DURING THIS TIME STEP:
                else:
                    # Then the apex is declared "Just dead":
                    apex.type = "Just_dead"
                    # The exact time since the apex died is calculated:
                    apex.thermal_time_since_death = apex.thermal_time_since_growth_stopped + self.time_step_in_seconds * temperature_time_adjustment - apex.life_duration
                    apex.actual_time_since_death = apex.thermal_time_since_death / temperature_time_adjustment
                    # And the other times are incremented:
                    apex.actual_time_since_primordium_formation += self.time_step_in_seconds
                    apex.actual_time_since_emergence += self.time_step_in_seconds
                    apex.actual_time_since_cells_formation += self.time_step_in_seconds
                    apex.actual_time_since_growth_stopped += self.time_step_in_seconds
                    apex.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_emergence += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_cells_formation += self.time_step_in_seconds * temperature_time_adjustment
                    apex.thermal_time_since_growth_stopped += self.time_step_in_seconds * temperature_time_adjustment
                    # The new element returned by the function corresponds to this apex:
                    new_apex.append(apex)
                    # And the function returns this new apex and stops here:
                    return new_apex

    # Function for calculating root elongation:
    def elongated_length(self, element, initial_length: float, radius: float, C_hexose_root: float, elongation_time_in_seconds: float):
        """
        This function computes a new length (m) based on the elongation process described by ArchiSimple and regulated by
        the available concentration of hexose.
        :param initial_length: the initial length (m)
        :param radius: radius (m)
        :param C_hexose_root: the concentration of hexose available for elongation (mol of hexose per gram of strctural mass)
        :param elongation_time_in_seconds: the period of elongation (s)
        :return: the new elongated length
        """
        
        # If we keep the classical ArchiSimple rule:
        if self.ArchiSimple:
            # Then the elongation is calculated following the rules of Pages et al. (2014):
            elongation = self.EL * 2. * radius * elongation_time_in_seconds
        else:
            # Otherwise, we additionally consider a limitation of the elongation according to the local concentration of hexose,
            # based on a Michaelis-Menten formalism:
            if C_hexose_root > 0.:
                elongation = self.EL * 2. * radius * C_hexose_root / (
                        self.Km_elongation + C_hexose_root) * elongation_time_in_seconds
            else:
                elongation = 0.

        # We calculate the new potential length corresponding to this elongation:
        new_length = initial_length + elongation
        if new_length < initial_length:
            print("!!! ERROR: There is a problem of elongation, with the initial length", initial_length,
                  " and the radius", radius, "and the elongation time", elongation_time_in_seconds)
        return new_length

    # Function for calculating the amount of C to be used in neighbouring elements for sustaining root elongation:
    def calculating_supply_for_elongation(self, element):
        """
        This function computes the list of root elements that can supply C as hexose for sustaining the elongation
        of a given element, as well as their structural mass and their amount of available hexose.
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
        growing_zone_length = self.growing_zone_factor * n.radius
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
                    # TODO: Should the C from root hairs be used for helping roots to grow?
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
                index_attempt = self.g.Father(index, EdgeType='<')
                # If there is no father element on this axis:
                if index_attempt is None:
                    # Then we try to move to the mother root, if any:
                    index_attempt = self.g.Father(index, EdgeType='+')
                    # If there is no such root:
                    if index_attempt is None:
                        # Then we exit the loop here:
                        break
                # We set the new index:
                index = index_attempt
                # We define the new element to consider according to the new index:
                current_element = self.g.node(index)
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

    # Function calculating the potential development of a root segment:
    def potential_segment_development(self, segment):
        """
        This function considers a root segment, i.e. a root element that can thicken but not elongate, and calculates its
        potential increase in radius according to the pipe model (possibly regulated by C availability), and its possible death.
        :param segment: the segment to be considered
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
            # TODO: Should the C from root hairs be used for helping nodules to grow?
            index_parent = self.g.Father(segment.index(), EdgeType='+')
            parent = self.g.node(index_parent)
            segment.hexose_available_for_thickening = parent.C_hexose_root * parent.struct_mass \
                                                      + segment.C_hexose_root * segment.struct_mass
            # We calculate an average concentration of hexose that will help to regulate nodule growth:
            C_hexose_regulating_nodule_growth = segment.hexose_available_for_thickening / (
                    parent.struct_mass + segment.struct_mass)
            # We modulate the relative increase in radius by the amount of C available in the nodule:
            thickening_rate = self.relative_nodule_thickening_rate_max \
                              * C_hexose_regulating_nodule_growth \
                              / (self.Km_nodule_thickening + C_hexose_regulating_nodule_growth)
            # We calculate a coefficient that will modify the rate of thickening according to soil temperature
            # assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
            thickening_rate = thickening_rate * self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=segment.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)
            segment.theoretical_radius = segment.radius * (1 + thickening_rate * self.time_step_in_seconds)
            if segment.theoretical_radius > self.nodule_max_radius:
                segment.potential_radius = self.nodule_max_radius
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
        # TODO: Should the C from root hairs be used for helping nodules to grow?
        segment.hexose_available_for_thickening = segment.C_hexose_root * segment.struct_mass

        # CALCULATING AN EQUIVALENT OF THERMAL TIME:
        # ------------------------------------------

        # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
        # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
        temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=segment.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)

        # CHECKING WHETHER THE APEX OF THE ROOT AXIS HAS STOPPED GROWING:
        # ---------------------------------------------------------------

        # We look at the apex of the axis to which the segment belongs (i.e. we get the last element of all the Descendants):
        index_apex = self.g.Descendants(segment.index())[-1]
        apex = self.g.node(index_apex)
        # print("For segment", segment.index(), "the terminal index is", index_apex, "and has the type", apex.label)
        # Depending on the type of the apex, we adjust the type of the segment on the same axis:
        if apex.type == "Just_stopped":
            segment.type = "Just_stopped"
        elif apex.type == "Stopped":
            segment.type = "Stopped"

        # CHECKING POSSIBLE ROOT SEGMENT DEATH:
        # -------------------------------------

        # For each child of the segment:
        for child in segment.children():

            # Then we add one child to the actual number of children:
            number_of_actual_children += 1

            if child.radius < 0. or child.potential_radius < 0.:
                print("!!! ERROR: the radius of the element", child.index(), "is negative!")
            # If the child belongs to the same axis:
            if child.edge_type == '<':
                # Then we record the THEORETICAL section of this child:
                son_section = child.theoretical_radius ** 2 * pi
                # # Then we record the section of this child:
                # son_section = child.radius * child.radius * pi
            # Otherwise if the child is the element of a lateral root AND if this lateral root has already emerged
            # AND the lateral element is not a nodule:
            elif child.edge_type == '+' and child.length > 0. and child.type != "Root_nodule":
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
        if self.radial_growth == "Possible":
            # The radius of the root segment is defined according to the pipe model.
            # In ArchiSimp9, the radius is increased by considering the sum of the sections of all the children,
            # by adding a fraction (SGC) of this sum of sections to the current section of the parent segment,
            # and by calculating the new radius that corresponds to this new section of the parent:
            segment.theoretical_radius = sqrt(son_section / pi + self.SGC * sum_of_lateral_sections / pi)
            # However, if the net difference is below 0.1% of the initial radius:
            if (segment.theoretical_radius - segment.initial_radius) <= 0.001 * segment.initial_radius:
                # Then the potential radius is set to the initial radius:
                segment.theoretical_radius = segment.initial_radius
            # If we consider simple ArchiSimple rules:
            if self.ArchiSimple:
                # Then the potential radius to form is equal to the theoretical one determined by geometry:
                segment.potential_radius = segment.theoretical_radius
            # Otherwise, if we don't strictly follow simple ArchiSimple rules and if there can be an increase in radius:
            elif segment.length > 0. and segment.theoretical_radius > segment.radius:
                # We calculate the maximal increase in radius that can be achieved over this time step,
                # based on a Michaelis-Menten formalism that regulates the maximal rate of increase
                # according to the amount of hexose available:
                thickening_rate = self.relative_root_thickening_rate_max \
                                  * segment.C_hexose_root / (self.Km_thickening + segment.C_hexose_root)
                # We calculate a coefficient that will modify the rate of thickening according to soil temperature
                # assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
                thickening_rate = thickening_rate * self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=segment.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)
                # The maximal possible new radius according to this regulation is therefore:
                new_radius_max = (1 + thickening_rate * self.time_step_in_seconds) * segment.initial_radius
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

        # UPDATING THE DIFFERENT TIMES:
        # ------------------------------

        # We increase the various time variables:
        segment.actual_time_since_primordium_formation += self.time_step_in_seconds
        segment.actual_time_since_emergence += self.time_step_in_seconds
        segment.actual_time_since_cells_formation += self.time_step_in_seconds
        segment.thermal_time_since_primordium_formation += self.time_step_in_seconds * temperature_time_adjustment
        segment.thermal_time_since_emergence += self.time_step_in_seconds * temperature_time_adjustment
        segment.thermal_time_since_cells_formation += self.time_step_in_seconds * temperature_time_adjustment

        if segment.type == "Just_stopped":
            segment.actual_time_since_growth_stopped = apex.actual_time_since_growth_stopped
            segment.thermal_time_since_growth_stopped = apex.actual_time_since_growth_stopped * temperature_time_adjustment
        if segment.type == "Stopped":
            segment.actual_time_since_growth_stopped += self.time_step_in_seconds
            segment.thermal_time_since_growth_stopped += self.time_step_in_seconds * temperature_time_adjustment
        if segment.type == "Just_dead":
            segment.actual_time_since_growth_stopped += self.time_step_in_seconds
            segment.thermal_time_since_growth_stopped += self.time_step_in_seconds * temperature_time_adjustment
            # AVOIDING PROBLEMS - We check that the list of times_since_death is not empty:
            if list_of_times_since_death:
                segment.actual_time_since_death = min(list_of_times_since_death)
            else:
                segment.actual_time_since_death = 0.
            segment.thermal_time_since_death = segment.actual_time_since_death * temperature_time_adjustment
        if segment.type == "Dead":
            segment.actual_time_since_growth_stopped += self.time_step_in_seconds
            segment.thermal_time_since_growth_stopped += self.time_step_in_seconds * temperature_time_adjustment
            segment.actual_time_since_death += self.time_step_in_seconds
            segment.thermal_time_since_death += self.time_step_in_seconds * temperature_time_adjustment

        new_segment.append(segment)
        return new_segment

    # Actual elongation, radial growth and growth respiration of root elements:
    @actual
    @state
    def actual_growth_and_corresponding_respiration(self):
        """
        This function defines how a segment, an apex and possibly an emerging root primordium will grow according to the amount
        of hexose present in the segment, taking into account growth respiration based on the model of Thornley and Cannell
        (2000). The calculation is based on the values of potential_radius, potential_length, lateral_root_emergence_possibility
        and emergence_cost defined in each element by the module "POTENTIAL GROWTH".
        The function returns the MTG "g" with modified values of radius and length of each element, the possibility of the
        emergence of lateral roots, and the cost of growth in terms of hexose consumption.

        :return:
        """

        # TODO FOR TRISTAN: In a second step, you may consider how the emergence of primordia may depend on N availability in the root or in the soil.

        # PROCEEDING TO ACTUAL GROWTH:
        # -----------------------------

        # We have to cover each vertex from the apices up to the base one time:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)
        # We cover all the vertices in the MTG, from the tips to the base:
        for vid in post_order(self.g, root):

            # n represents the current root element:
            n = self.g.node(vid)

            # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
            # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
            temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=n.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)

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
            initial_volume = self.volume_from_radius_and_length(n, n.initial_radius, n.initial_length)
            # We calculate the potential volume of the element based on the potential radius and potential length:
            potential_volume = self.volume_from_radius_and_length(n, n.potential_radius, n.potential_length)
            # We calculate the number of moles of hexose required for growth, including the respiration cost according to
            # the yield growth included in the model of Thornley and Cannell (2000), where root_tissue_density is the dry structural
            # weight per volume (g m-3) and struct_mass_C_content is the amount of C per gram of dry structural mass (mol_C g-1):
            n.hexose_growth_demand = (potential_volume - initial_volume) \
                                     * n.root_tissue_density * self.struct_mass_C_content / self.yield_growth * 1 / 6.
            # We verify that this potential growth demand is positive:
            if n.hexose_growth_demand < 0.:
                print("!!! ERROR: a negative growth demand of", n.hexose_growth_demand,
                      "was calculated for the element", n.index(), "of class", n.label)
                print("The initial volume is", initial_volume, "the potential volume is", potential_volume)
                print("The initial length was", n.initial_length, "and the potential length was",
                      n.potential_length)
                print("The initial radius was", n.initial_radius, "and the potential radius was",
                      n.potential_radius)
                n.hexose_growth_demand = 0.
                # In such case, we just pass to the next element in the iteration:
                continue
            elif n.hexose_growth_demand == 0.:
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
                         / (n.root_tissue_density * self.struct_mass_C_content) * self.yield_growth
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
                volume_after_elongation = self.volume_from_radius_and_length(n, n.initial_radius, n.length)
                # The overall cost of elongation is calculated as:
                hexose_consumption_by_elongation = \
                    1. / 6. * (volume_after_elongation - initial_volume) \
                    * n.root_tissue_density * self.struct_mass_C_content / self.yield_growth

                # If there has been an actual elongation:
                if n.length > n.initial_length:

                    # REGISTERING THE COSTS FOR ELONGATION:
                    # We cover each of the elements that have provided hexose for sustaining the elongation of element n:
                    for i in range(0, len(list_of_elongation_supporting_elements)):
                        index = list_of_elongation_supporting_elements[i]
                        supplying_element = self.g.node(index)
                        # We define the actual contribution of the current element based on total hexose consumption by growth
                        # of element n and the relative contribution of the current element to the pool of the available hexose:
                        hexose_actual_contribution_to_elongation = hexose_consumption_by_elongation \
                                                                   * list_of_elongation_supporting_elements_hexose[
                                                                       i] / hexose_possibly_required_for_elongation
                        # The amount of hexose used for growth in this element is increased:
                        supplying_element.hexose_consumption_by_growth_amount += hexose_actual_contribution_to_elongation
                        supplying_element.hexose_consumption_by_growth += hexose_actual_contribution_to_elongation / self.time_step_in_seconds
                        # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                        supplying_element.resp_growth += hexose_actual_contribution_to_elongation * (1 - self.yield_growth) * 6.

            # ACTUAL RADIAL GROWTH IS THEN CONSIDERED:
            # -----------------------------------------

            # If the radius of the element can increase:
            if n.potential_radius > n.initial_radius:

                # CALCULATING ACTUAL THICKENING:
                # We calculate the increase in volume that can be achieved with the amount of hexose available:
                possible_radial_increase_in_volume = \
                    remaining_hexose_for_thickening * 6. * self.yield_growth \
                    / (n.root_tissue_density * self.struct_mass_C_content)
                # We calculate the maximal possible volume based on the volume of the new cylinder after elongation
                # and the increase in volume that could be achieved by consuming all the remaining hexose:
                volume_max = self.volume_from_radius_and_length(n, n.initial_radius, n.length) + possible_radial_increase_in_volume
                # We then calculate the corresponding new possible radius corresponding to this maximum volume:
                if n.type == "Root_nodule":
                    # If the element corresponds to a nodule, then it we calculate the radius of a theoretical sphere:
                    possible_radius = (3. / (4. * pi)) ** (1. / 3.)
                else:
                    # Otherwise, we calculate the radius of a cylinder:
                    possible_radius = sqrt(volume_max / (n.length * pi))
                if possible_radius < 0.9999 * n.initial_radius:  # We authorize a difference of 0.01% due to calculation errors!
                    print("!!! ERROR: the calculated new radius of element", n.index(),
                          "is lower than the initial one!")
                    print("The possible radius was", possible_radius, "and the initial radius was",
                          n.initial_radius)

                # If the maximal radius that can be obtained is lower than the potential radius suggested by the potential growth module:
                if possible_radius <= n.potential_radius:
                    # Then radial growth is limited and there is no remaining hexose after radial growth:
                    n.radius = possible_radius
                    hexose_actual_contribution_to_thickening = remaining_hexose_for_thickening
                    remaining_hexose_for_thickening = 0.
                else:
                    # Otherwise, radial growth is done up to the full potential and the remaining hexose is calculated:
                    n.radius = n.potential_radius
                    net_increase_in_volume = self.volume_from_radius_and_length(n, n.radius, n.length) \
                        - self.volume_from_radius_and_length(n, n.initial_radius, n.length)
                    # net_increase_in_volume = pi * (n.radius ** 2 - n.initial_radius ** 2) * n.length
                    # We then calculate the remaining amount of hexose after thickening:
                    hexose_actual_contribution_to_thickening = \
                        1. / 6. * net_increase_in_volume \
                        * n.root_tissue_density * self.struct_mass_C_content / self.yield_growth

                # REGISTERING THE COSTS FOR THICKENING:
                # --------------------------------------
                fraction_of_available_hexose_in_the_element = \
                    (n.C_hexose_root * n.initial_struct_mass) / hexose_available_for_thickening
                # The amount of hexose used for growth in this element is increased:
                n.hexose_consumption_by_growth_amount += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)
                n.hexose_consumption_by_growth += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) / self.time_step_in_seconds
                # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                n.resp_growth += \
                    (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) \
                    * (1 - self.yield_growth) * 6.
                if n.type == "Root_nodule":
                    index_parent = self.g.Father(n.index(), EdgeType='+')
                    parent = self.g.node(index_parent)
                    fraction_of_available_hexose_in_the_element = \
                        (parent.C_hexose_root * parent.initial_struct_mass) / hexose_available_for_thickening
                    # The amount of hexose used for growth in this element is increased:
                    parent.hexose_consumption_by_growth_amount += \
                        (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)
                    parent.hexose_consumption_by_growth += \
                        (
                                hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) / self.time_step_in_seconds
                    # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                    parent.resp_growth += \
                        (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) \
                        * (1 - self.yield_growth) * 6.

            # RECORDING THE ACTUAL STRUCTURAL MODIFICATIONS:
            # -----------------------------------------------
            # The new volume of the element is automatically calculated
            n.volume = self.volume_from_radius_and_length(n, n.radius, n.length)
            # The new dry structural struct_mass of the element is calculated from its new volume:
            n.struct_mass = n.volume * n.root_tissue_density
            n.struct_mass_produced = (n.volume - initial_volume) * n.root_tissue_density

            if n.struct_mass < n.initial_struct_mass and n.struct_mass_produced > 0.:
                print(f"!!! ERROR during initialisation for initial struct mass, no concentrations will be updated on {n.index()}")
                n.initial_struct_mass = n.struct_mass

            # Verification: we check that no negative length or struct_mass have been generated!
            if n.volume < 0:
                print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length,
                      "and a mass of", n.struct_mass)
                # We then reset all the geometrical values to their initial values:
                n.length = n.initial_length
                n.radius = n.initial_radius
                n.struct_mass = n.initial_struct_mass
                n.struct_mass_produced = 0.
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
                    n.actual_elongation_rate = n.actual_elongation / self.time_step_in_seconds

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

    # Function that creates new segments and priomordia in "g":
    # ----------------------------------------------------------
    @segmentation
    @state
    def segmentation_and_primordia_formation(self):
        """
        This function considers segmentation and primordia formation across the whole root MTG.
        :return:
        """
        # We simulate the segmentation of all apices:
        for vid in self.vertices:
            n = self.g.node(vid)
            # For each apex in the list of apices that have emerged with a positive length:
            if n.label == 'Apex' and n.type == "Normal_root_after_emergence" and n.length > 0.:
                self.segmentation_and_primordium_formation(apex=n)

        # We make sure that stored vertices are well updated with the new ones
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        self.post_growth_updating()

    def segmentation_and_primordium_formation(self, apex):
        """
        This function transforms an elongated root apex into a list of segments and a terminal, smaller apex. A primordium
        of a lateral root can be formed on the new segment in some cases, depending on the distance to tip and the root orders.
        :param apex: the root apex to be segmented
        :return:
        """

        # TODO: Simplify the readability of this function by looping on two sets of variables, the ones that remain identical
        #  within the segments, and those that are divided.

        # TODO# FOR TRISTAN: When working with a dynamic root structure, you will need to specify in this function "segmentation"
        #  the new quantitative variables that would need to be recalculated when segmenting a long root apex
        #  (ex: N amount will need to be divided, while N concentration will remain identical among the segments)

        # CALCULATING AN EQUIVALENT OF THERMAL TIME:
        # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
        # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
        temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=apex.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)

        # ADJUSTING ROOT ANGLES FOR THE FUTURE NEW SEGMENTS:
        # Optional - We can add random geometry, or not:
        if self.random:
            # The seed used to generate random values is defined according to a parameter random_choice and the index of the apex:
            np.random.seed(self.random_choice * apex.index())
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
        initial_root_order = apex.root_order
        initial_length = apex.length
        initial_dist_to_ramif = apex.dist_to_ramif
        initial_elongation = apex.actual_elongation
        initial_elongation_rate = apex.actual_elongation_rate
        initial_struct_mass = apex.struct_mass
        initial_struct_mass_produced = apex.struct_mass_produced
        initial_resp_growth = apex.resp_growth

        initial_root_hairs_struct_mass = apex.root_hairs_struct_mass
        initial_root_hairs_struct_mass_produced = apex.root_hairs_struct_mass_produced
        initial_living_root_hairs_struct_mass = apex.living_root_hairs_struct_mass
        initial_living_root_hairs_number = apex.living_root_hairs_number
        initial_dead_root_hairs_number = apex.dead_root_hairs_number
        initial_total_root_hairs_number = apex.total_root_hairs_number
        # TODO: the time-related variables associated to root hairs will not be changed here, but rather in the function 'root_hairs_dynamics' - check whether this is OK!

        # NOTE: The following is not an error, we also need to record the "initial" structural mass and surfaces before growth!
        initial_initial_struct_mass = apex.initial_struct_mass
        initial_initial_living_root_hairs_struct_mass = apex.initial_living_root_hairs_struct_mass
        initial_hexose_growth_demand = apex.hexose_growth_demand
        initial_hexose_consumption_by_growth_amount = apex.hexose_consumption_by_growth_amount
        initial_hexose_consumption_by_growth = apex.hexose_consumption_by_growth
        # TODO deficits old = new or * mass fraction when segmentation (Deficits, rates, fluxes)

        # We record the type of the apex, as it may correspond to an apex that has stopped (or even died):
        initial_type = apex.type
        initial_lateral_root_emergence_possibility = apex.lateral_root_emergence_possibility

        # CASE 1: NO SEGMENTATION IS NECESSARY
        # -------------------------------------
        # If the length of the apex is smaller than the defined length of a root segment:
        if apex.length <= self.segment_length:
            # CONSIDERING POSSIBLE PRIMORDIUM FORMATION:
            # We simply call the function primordium_formation to check whether a primordium should have been formed
            # (Note: we assume that the segment length is always smaller than the inter-branching distance IBD,
            # so that in this case, only 0 or 1 primordium may have been formed - the function is called only once):
            new_apex.append(self.primordium_formation(apex, elongation_rate=initial_elongation_rate))

            # If there has been an actual elongation of the root apex:
            if apex.actual_elongation_rate > 0.:
                # # RECALCULATING DIMENSIONS:
                # # We assume that the growth functions that may have been called previously have only modified radius and length,
                # # but not the struct_mass and the total amounts present in the root element.
                # # We modify the geometrical features of the present element according to the new length and radius:
                # apex.volume = volume_from_radius_and_length(g, apex, apex.radius, apex.length)["volume"]
                # apex.struct_mass = apex.volume * param.root_tissue_density

                # CALCULATING THE AVERAGE TIME SINCE ROOT CELLS FORMED:
                # We update the time since root cells have formed, based on the elongation rate and on the principle that
                # cells appear at the very end of the root tip, and then age. In this case, the average age of the element is
                # calculated as the mean value between the incremented age of the part that was already formed and has not
                # elongated, and the average age of the new part formed.
                # The age of the initially-existing part is simply incremented by the time step:
                Age_of_non_elongated_part = apex.actual_time_since_cells_formation + self.time_step_in_seconds
                # The age of the cells at the top of the elongated part is by definition equal to:
                Age_of_elongated_part_up = apex.actual_elongation / apex.actual_elongation_rate
                # The age of the cells at the root tip is by definition 0:
                Age_of_elongated_part_down = 0
                # The average age of the elongated part is calculated as the mean value between both extremities:
                Age_of_elongated_part = 0.5 * (Age_of_elongated_part_down + Age_of_elongated_part_up)
                # Eventually, the average age of the whole new element is calculated from the age of both parts:
                apex.actual_time_since_cells_formation = (apex.initial_length * Age_of_non_elongated_part
                                                          + (
                                                                  apex.length - apex.initial_length) * Age_of_elongated_part) \
                                                         / apex.length
                # We also calculate the thermal age of root cells according to the previous thermal time of
                # initially-exisiting cells:
                Thermal_age_of_non_elongated_part = apex.thermal_time_since_cells_formation + self.time_step_in_seconds * \
                                                    temperature_time_adjustment
                Thermal_age_of_elongated_part = Age_of_elongated_part * temperature_time_adjustment
                apex.thermal_time_since_cells_formation = (apex.initial_length * Thermal_age_of_non_elongated_part
                                                        + (apex.length - apex.initial_length) * Thermal_age_of_elongated_part) / apex.length

        # CASE 2: THE APEX HAS TO BE SEGMENTED
        # -------------------------------------
        # Otherwise, we have to calculate the number of entire segments within the apex.
        else:
            # CALCULATION OF THE NUMBER OF ROOT SEGMENTS TO BE FORMED:
            # If the final length of the apex does not correspond to an entire number of segments:
            if apex.length / self.segment_length - floor(apex.length / self.segment_length) > 0.:
                # Then the total number of segments to be formed is:
                n_segments = floor(apex.length / self.segment_length)
            else:
                # Otherwise, the number of segments to be formed is decreased by 1,
                # so that the last element corresponds to an apex with a positive length:
                n_segments = floor(apex.length / self.segment_length) - 1
            n_segments = int(n_segments)

            # We need to calculate the final length of the terminal apex:
            final_apex_length = initial_length - n_segments * self.segment_length

            # FORMATION OF THE NEW SEGMENTS
            # We develop each new segment, except the last one, by transforming the current apex into a segment
            # and by adding a new apex after it, in an iterative way for (n-1) segments:
            for i in range(1, n_segments + 1):

                # We define the length of the present element (still called "apex") as the constant length of a segment:
                apex.length = self.segment_length
                # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
                apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - self.segment_length * i)
                # We modify the geometrical features of the present element according to the new length:
                apex.volume = self.volume_from_radius_and_length(apex, apex.radius, apex.length)
                apex.struct_mass = apex.volume * apex.root_tissue_density

                # We calculate the mass fraction that the segment represents compared to the whole element prior to segmentation:
                mass_fraction = apex.struct_mass / initial_struct_mass

                # We modify the variables representing total amounts according to this mass fraction:
                apex.resp_growth = initial_resp_growth * mass_fraction

                apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
                apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction
                apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction

                apex.root_hairs_struct_mass = initial_root_hairs_struct_mass * mass_fraction
                apex.root_hairs_struct_mass_produced = initial_root_hairs_struct_mass_produced * mass_fraction
                apex.living_root_hairs_struct_mass = initial_living_root_hairs_struct_mass * mass_fraction
                apex.living_root_hairs_number = initial_living_root_hairs_number * mass_fraction
                apex.dead_root_hairs_number = initial_dead_root_hairs_number * mass_fraction
                apex.total_root_hairs_number = initial_total_root_hairs_number * mass_fraction
                apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
                apex.hexose_consumption_by_growth_amount = initial_hexose_consumption_by_growth_amount * mass_fraction
                apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

                # CALCULATING THE TIME SINCE ROOT CELLS FORMATION IN THE NEW SEGMENT:
                # We update the time since root cells have formed, based on the elongation rate and on the principle that
                # cells appear at the very end of the root tip, and then age.

                # CASE 1: The first new segment contains a part that was already formed at the previous time step
                if i == 1:
                    # In this case, the average age of the element is calculated as the mean value between the incremented
                    # age of the part that was already formed and has not elongated, and the average age of the new part to
                    # complete the length of the segment.
                    # The age of the initially-existing part is simply incremented by the time step:
                    Age_of_non_elongated_part = apex.actual_time_since_cells_formation + self.time_step_in_seconds
                    # The age of the cells at the top of the elongated part is by definition equal to:
                    Age_of_elongated_part_up = apex.actual_elongation / apex.actual_elongation_rate
                    # The age of the cells at the bottom of the new segment is defined according to the length below it:
                    Age_of_elongated_part_down = (final_apex_length + (n_segments - 1) * self.segment_length) \
                                                 / apex.actual_elongation_rate
                    # The age of the elongated part of the new segment is calculated as the mean value between both extremities:
                    Age_of_elongated_part = 0.5 * (Age_of_elongated_part_down + Age_of_elongated_part_up)
                    # Eventually, the average age of the whole new segment is calculated from the age of both parts:
                    apex.actual_time_since_cells_formation = (apex.initial_length * Age_of_non_elongated_part + (
                                                                      apex.length - apex.initial_length) * Age_of_elongated_part) \
                                                             / apex.length
                    # We also calculate the thermal age of root cells according to the previous thermal time of initially-exisiting cells:
                    Thermal_age_of_non_elongated_part = apex.thermal_time_since_cells_formation + self.time_step_in_seconds * temperature_time_adjustment
                    Thermal_age_of_elongated_part = Age_of_elongated_part * temperature_time_adjustment
                    apex.thermal_time_since_cells_formation = (apex.initial_length * Thermal_age_of_non_elongated_part + (
                                                                              apex.length - apex.initial_length) * Thermal_age_of_elongated_part) / apex.length

                # CASE 2: The new segment is only made of new root cells formed during this time step
                else:
                    # The age of the cells at the top of the segment is defined according to the elongated length up to it:
                    Age_up = (final_apex_length + (
                            n_segments - i) * self.segment_length) / apex.actual_elongation_rate
                    # The age of the cells at the bottom of the segment is defined according to the elongated length up to it:
                    Age_down = (final_apex_length + (
                            n_segments - 1 - i) * self.segment_length) / apex.actual_elongation_rate
                    apex.actual_time_since_cells_formation = 0.5 * (Age_down + Age_up)
                    apex.thermal_time_since_cells_formation = apex.actual_time_since_cells_formation * temperature_time_adjustment

                # CONSIDERING POSSIBLE PRIMORDIUM FORMATION:
                # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
                new_apex.append(self.primordium_formation(apex, elongation_rate=initial_elongation_rate))

                # The current element that has been elongated up to segment_length is now considered as a segment:
                apex.label = 'Segment'

                # If the segment is not the last one on the elongated axis:
                if i < n_segments:
                    # Then we also add a new element, initially of length 0, which we call "apex" and which will correspond
                    # to the next segment to be defined in the loop:
                    apex = self.ADDING_A_CHILD(mother_element=apex, edge_type='<', label='Apex',
                                               type=apex.type,
                                               root_order=initial_root_order,
                                               angle_down=segment_angle_down,
                                               angle_roll=segment_angle_roll,
                                               length=0.,
                                               radius=apex.radius,
                                               identical_properties=True,
                                               nil_properties=False)
                    apex.actual_elongation = self.segment_length * i
                else:
                    # Otherwise, the loop will stop now, and we will add the terminal apex hereafter.
                    # NODULE OPTION:
                    # We add the possibility of a nodule formation on the segment that is closest to the apex:
                    if self.nodules and len(
                            apex.children()) < 2 and np.random.random() < self.nodule_formation_probability:
                        self.nodule_formation(mother_element=apex)  # WATCH OUT: here, "apex" still corresponds to the last segment!

            # FORMATION OF THE TERMINAL APEX:
            # And we define the new, final apex after the last defined segment, with a new length defined as:
            new_length = initial_length - n_segments * self.segment_length
            apex = self.ADDING_A_CHILD(mother_element=apex, edge_type='<', label='Apex',
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
            apex.volume = self.volume_from_radius_and_length(apex, apex.radius, apex.length)
            apex.struct_mass = apex.volume * apex.root_tissue_density
            # We modify the variables representing total amounts according to the new struct_mass:
            mass_fraction = apex.struct_mass / initial_struct_mass

            apex.resp_growth = initial_resp_growth * mass_fraction

            apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
            apex.initial_living_root_hairs_struct_mass = initial_initial_living_root_hairs_struct_mass * mass_fraction

            apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction

            apex.root_hairs_struct_mass = initial_root_hairs_struct_mass * mass_fraction
            apex.root_hairs_struct_mass_produced = initial_root_hairs_struct_mass_produced * mass_fraction
            apex.living_root_hairs_struct_mass = initial_living_root_hairs_struct_mass * mass_fraction
            apex.living_root_hairs_number = initial_living_root_hairs_number * mass_fraction
            apex.dead_root_hairs_number = initial_dead_root_hairs_number * mass_fraction
            apex.total_root_hairs_number = initial_total_root_hairs_number * mass_fraction
            apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
            apex.hexose_consumption_by_growth_amount = initial_hexose_consumption_by_growth_amount * mass_fraction
            apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

            # CALCULATING THE TIME SINCE ROOT CELLS FORMATION IN THE NEW SEGMENT:
            # We update the time since root cells have formed, based on the elongation rate:
            Age_up = final_apex_length / apex.actual_elongation_rate
            Age_down = 0
            apex.actual_time_since_cells_formation = 0.5 * (Age_down + Age_up)
            apex.thermal_time_since_cells_formation = apex.actual_time_since_cells_formation * temperature_time_adjustment

            # And we call the function primordium_formation to check whether a primordium should have been formed:
            new_apex.append(self.primordium_formation(apex, elongation_rate=initial_elongation_rate))

            # Finally, we add the last apex present at the end of the elongated axis:
            new_apex.append(apex)

        return new_apex

    # Formation of a root primordium at the apex of the mother root:
    # ---------------------------------------------------------------
    def primordium_formation(self, apex, elongation_rate=0.):
        """
        This function considers the formation of a primordium on a root apex, and, if possible, creates this new element
        of length 0.
        :param apex: the apex on which the primordium may be created
        :param elongation_rate: the rate of elongation of the apex over the time step during which the primordium may be created (m s-1)
        :return: the new primordium
        """

        # NOTE: This function has to be called AFTER the actual elongation of the apex has been done and the distance
        # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

        # CALCULATING AN EQUIVALENT OF THERMAL TIME:
        # -------------------------------------------

        # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
        # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
        temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=apex.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)

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
        if self.root_order_limitation and apex.root_order + 1 > self.root_order_treshold:
            # Then we don't add any primordium and simply return the unaltered apex:
            new_apex.append(apex)
            return new_apex

        # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
        # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
        # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014).
        # We also set the root angles depending on random:
        if self.random:
            # The seed used to generate random values is defined according to a parameter random_choice and the index of the apex:
            np.random.seed(self.random_choice * apex.index())
            potential_radius = np.random.normal((apex.radius - self.Dmin / 2.) * self.RMD + self.Dmin / 2.,
                                                ((apex.radius - self.Dmin / 2.) * self.RMD + self.Dmin / 2.) * self.CVDD)
            apex_angle_roll = abs(np.random.normal(120, 10))
            if apex.root_order == 1:
                primordium_angle_down = abs(np.random.normal(45, 10))
            else:
                primordium_angle_down = abs(np.random.normal(70, 10))
            primordium_angle_roll = abs(np.random.normal(5, 5))
        else:
            potential_radius = (apex.radius - self.Dmin / 2) * self.RMD + self.Dmin / 2.
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
        if apex.dist_to_ramif > self.IPD and potential_radius >= self.Dmin and potential_radius <= apex.radius:
            # The distance that the tip of the apex has covered since the actual primordium formation is calculated:
            elongation_since_last_ramif = apex.dist_to_ramif - self.IPD
            # TODO: Is the third condition really relevant?

            # A specific rolling angle is attributed to the parent apex:
            apex.angle_roll = apex_angle_roll

            # We verify that the apex has actually elongated:
            if apex.actual_elongation > 0 and elongation_rate > 0.:
                # Then the actual time since the primordium must have been formed is precisely calculated
                # according to the actual growth of the parent apex since primordium formation,
                # taking into account the actual growth rate of the parent defined as
                # apex.actual_elongation / time_step_in_seconds
                actual_time_since_formation = elongation_since_last_ramif / elongation_rate
            else:
                actual_time_since_formation = 0.

            # And we add the primordium of a possible new lateral root:
            ramif = self.ADDING_A_CHILD(mother_element=apex, edge_type='+', label='Apex',
                                        type='Normal_root_before_emergence',
                                        root_order=apex.root_order + 1,
                                        angle_down=primordium_angle_down,
                                        angle_roll=primordium_angle_roll,
                                        length=0.,
                                        radius=potential_radius,
                                        identical_properties=False,
                                        nil_properties=True)
            # We specifically recomputes the growth duration:
            ramif.growth_duration = self.GDs * (2. * ramif.radius) ** 2 * self.main_roots_growth_extender
            # ramif.growth_duration = self.calculate_growth_duration(radius=ramif.radius, index=ramif.index(),
            #                                                        root_order=ramif.root_order)
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

    @postsegmentation
    @state
    def root_hairs_dynamics(self):
        """
        This function computes the evolution of the density and average length of root hairs along each root,
        and specifies which hairs are alive or dead.
        :return:
        """
        # TODO FOR TRISTAN In a second step, consider playing on the density / max. length of root hairs depending on the availability of N in the soil (if relevant)?

        # We cover all the vertices in the MTG:
        for vid in self.g.vertices_iter(scale=1):
            # n represents the vertex:
            n = self.g.node(vid)

            # First, we ensure that the element has a positive length:
            if n.length <= 0:
                continue

            # We also exclude nodules and dead elements from this computation:
            if n.type == "Just_dead" or n.type == "Dead" or n.type == "Nodule":
                continue

            # # TODO: Check the consequences of avoiding apex in root hairs dynamics!
            # # WE ALSO AVOID ROOT APICES - EVEN IF IN THEORY ROOT HAIRS MAY ALSO APPEAR ON THEM:
            # if n.label == "Apex":
            #     continue
            # # Even if root hairs should have already emerge on that root apex, they will appear in the next step (or in a few steps)
            # # when the element becomes a segment.

            # We calculate the equivalent of a thermal time for the current time step:
            temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=n.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)
            elapsed_thermal_time = self.time_step_in_seconds * temperature_time_adjustment

            # We keep in memory the initial total mass of root hairs (possibly including dead hairs):
            initial_root_hairs_struct_mass = n.root_hairs_struct_mass

            # We calculate the total number of (newly formed) root hairs (if any) and update their age:
            # ------------------------------------------------------------------------------------------
            # CASE 1 - If the current element is completely included within the actual growing zone of the root at the root
            # tip, the root hairs cannot have formed yet:
            if n.distance_from_tip <= self.growing_zone_factor * n.radius:
                # We stop here with the calculations and move to the next element:
                continue
            # CASE 2 - If all root hairs have already been formed:
            if n.all_root_hairs_formed:
                # Then we simply increase the time since root hairs emergence started:
                n.actual_time_since_root_hairs_emergence_started += self.time_step_in_seconds
                n.thermal_time_since_root_hairs_emergence_started += elapsed_thermal_time
                n.actual_time_since_root_hairs_emergence_stopped += self.time_step_in_seconds
                n.thermal_time_since_root_hairs_emergence_stopped += elapsed_thermal_time
            # CASE 3 - If the theoretical growing zone limit is located somewhere within the root element:
            elif n.distance_from_tip - n.length < self.growing_zone_factor * n.radius:
                # We first record the previous length of the root hair zone within the element:
                initial_length_with_hairs = n.actual_length_with_hairs
                # Then the new length of the root hair zone is calculated:
                n.actual_length_with_hairs = n.distance_from_tip - self.growing_zone_factor * n.radius
                net_increase_in_root_hairs_length = n.actual_length_with_hairs - initial_length_with_hairs
                # The corresponding number of root hairs is calculated:
                n.total_root_hairs_number = self.root_hairs_density * n.radius * n.actual_length_with_hairs
                # The time since root hair formation started is then calculated, using the recent increase in the length
                # of the current root hair zone and the elongation rate of the corresponding root tip. The latter is
                # calculated using the difference between the new distance_from_tip of the element and the previous one:
                elongation_rate_in_actual_time = (n.distance_from_tip - n.former_distance_from_tip) / self.time_step_in_seconds
                elongation_rate_in_thermal_time = (n.distance_from_tip - n.former_distance_from_tip) / elapsed_thermal_time
                # SUBCASE 3.1 - If root hairs had not emerged at the previous time step:
                if elongation_rate_in_actual_time > 0. and initial_length_with_hairs <= 0.:
                    # We increase the time since root hairs emerged by only the fraction of the time step corresponding to the growth of hairs:
                    n.actual_time_since_root_hairs_emergence_started += \
                        self.time_step_in_seconds - net_increase_in_root_hairs_length / elongation_rate_in_actual_time
                    n.thermal_time_since_root_hairs_emergence_started += \
                        elapsed_thermal_time - net_increase_in_root_hairs_length / elongation_rate_in_thermal_time
                # SUBCASE 3.2 - the hairs had already started to grow:
                else:
                    # Consequently, the full time elapsed during this time step can be added to the age:
                    n.actual_time_since_root_hairs_emergence_started += self.time_step_in_seconds
                    n.thermal_time_since_root_hairs_emergence_started += elapsed_thermal_time
            # CASE 4 - the element is now "full" with root hairs as the limit of root elongation is located further down:
            else:
                # The actual time since root hairs emergence started is first increased:
                n.actual_time_since_root_hairs_emergence_started += self.time_step_in_seconds
                n.thermal_time_since_root_hairs_emergence_started += elapsed_thermal_time
                # We then record the previous length of the root hair zone within the root element:
                initial_length_with_hairs = n.actual_length_with_hairs
                # And the new length of the root hair zone is necessarily the full length of the root element:
                n.actual_length_with_hairs = n.length
                net_increase_in_root_hairs_length = n.actual_length_with_hairs - initial_length_with_hairs
                # The total number of hairs is defined according to the radius and total length of the element:
                n.total_root_hairs_number = self.root_hairs_density * n.radius * n.length
                # The elongation of the corresponding root tip is calculated as the difference between the new
                # distance_from_tip of the element and the previous one:
                elongation_rate_in_actual_time = (
                                                         n.distance_from_tip - n.former_distance_from_tip) / self.time_step_in_seconds
                elongation_rate_in_thermal_time = (
                                                          n.distance_from_tip - n.former_distance_from_tip) / elapsed_thermal_time
                # The actual time since root hairs emergence has stopped is then calculated:
                if elongation_rate_in_actual_time > 0.:
                    n.actual_time_since_root_hairs_emergence_stopped += \
                        self.time_step_in_seconds - net_increase_in_root_hairs_length / elongation_rate_in_actual_time
                    n.thermal_time_since_root_hairs_emergence_stopped += \
                        elapsed_thermal_time - net_increase_in_root_hairs_length / elongation_rate_in_thermal_time
                else:
                    n.actual_time_since_root_hairs_emergence_stopped += self.time_step_in_seconds
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
            if n.root_hair_length < self.root_hair_max_length:
                # The new potential root hairs length is calculated according to the elongation rate,
                # corrected by temperature and modulated by the concentration of hexose (in the same way as for root
                # elongation) available in the root hair zone on the root element:
                new_length = n.root_hair_length + self.root_hairs_elongation_rate * self.root_hair_radius \
                             * n.C_hexose_root * (n.actual_length_with_hairs / n.length) \
                             / (self.Km_elongation + n.C_hexose_root) * elapsed_thermal_time
                # If the new calculated length is higher than the maximal length:
                if new_length > self.root_hair_max_length:
                    # We set the root hairs length to the maximal length:
                    n.root_hair_length = self.root_hair_max_length
                else:
                    # Otherwise, we record the new calculated length:
                    n.root_hair_length = new_length

            # We finally calculate the total external surface (m2), volume (m3) and mass (g) of root hairs:
            # ----------------------------------------------------------------------------------------------
            # In the calculation of surface, we consider the root hair to be a cylinder, and include the lateral section,
            # but exclude the section of the cylinder at the tip:
            n.root_hairs_volume = (self.root_hair_radius ** 2 * pi) * n.root_hair_length * n.total_root_hairs_number
            n.root_hairs_struct_mass = n.root_hairs_volume * n.root_tissue_density
            if n.total_root_hairs_number > 0.:
                n.living_root_hairs_struct_mass = n.root_hairs_struct_mass * n.living_root_hairs_number \
                                                  / n.total_root_hairs_number
            else:
                n.living_root_hairs_struct_mass = 0.

            # We calculate the mass of hairs that has been effectively produced, including from root hairs that may have died since then:
            # ----------------------------------------------------------------------------------------------------------------------------
            # We calculate the new production as the difference between initial and final mass:
            n.root_hairs_struct_mass_produced = n.root_hairs_struct_mass - initial_root_hairs_struct_mass

            # We add the cost of producing the new living root hairs (if any) to the hexose consumption by growth:
            hexose_consumption = n.root_hairs_struct_mass_produced * self.struct_mass_C_content / self.yield_growth / 6.
            n.hexose_consumption_by_growth_amount += hexose_consumption
            n.hexose_consumption_by_growth += hexose_consumption / self.time_step_in_seconds
            n.resp_growth += hexose_consumption * 6. * (1 - self.yield_growth)

    # TODO UNUSED
    # Function calculating a satisfaction coefficient for the growth of the whole root system:
    # -----------------------------------------------------------------------------------------
    def satisfaction_coefficient(self, struct_mass_input):
        """
        This function computes a general "satisfaction coefficient" SC for the whole root system according to ArchiSimple
        rules, i.e. it compares the available C for root growth and the need for C associated to the potential growth of all
        root elements. If SC >1, there won't be any growth limitation by C, otherwise, the growth of each element will be
        reduced proportionally to SC.
        :param struct_mass_input: the available input of "biomass" to be used for growth
        :return: the satisfaction coefficient SC
        """
        # We initialize the sum of individual demands for struct_mass:
        sum_struct_mass_demand = 0.
        SC = 0.

        # We have to cover each vertex from the apices up to the base one time:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We cover all the vertices in the MTG:
        for vid in post_order(self.g, root):
            # n represents the current root element:
            n = self.g.node(vid)

            # We calculate the initial volume of the element:
            initial_volume = self.volume_from_radius_and_length(n, n.initial_radius, n.initial_length)
            # We calculate the potential volume of the element based on the potential radius and potential length:
            potential_volume = self.volume_from_radius_and_length(n, n.potential_radius, n.potential_length)

            # The growth demand of the element in struct_mass is calculated:
            n.growth_demand_in_struct_mass = (potential_volume - initial_volume) * n.root_tissue_density
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
    
    # TODO UNUSED
    # Function performing the growth of each element based on the potential growth and the satisfaction coefficient SC:
    # ------------------------------------------------------------------------------------------------------------------

    def ArchiSimple_growth(self, SC):
        """
        This function computes the growth of the root system according to ArchiSimple's rules.
        :param SC: the satisfaction coefficient (i.e. the ratio of C offer and C demand)
        :return: g, the updated root MTG
        """

        # TODO FOR TRISTAN: If one day you have time to lose, you may see whether you want to add an equivalent limitation of the elongation of all apices with the amount of N available in the root - knowing that this does not exist in ArchiSimple anyway.

        # We have to cover each vertex from the apices up to the base one time:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # PERFORMING ARCHISIMPLE GROWTH:
        # -------------------------------

        # We cover all the vertices in the MTG:
        for vid in post_order(self.g, root):

            # n represents the current root element:
            n = self.g.node(vid)

            # CALCULATING AN EQUIVALENT OF THERMAL TIME:
            # -------------------------------------------

            # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil
            # temperature assuming a linear relationship (this is equivalent as the calculation of "growth degree-days):
            temperature_time_adjustment = self.temperature_modification(process_at_T_ref=self.process_at_T_ref,
                                                                    soil_temperature=n.soil_temperature,
                                                                    T_ref=self.T_ref, A=self.A, B=self.B, C=self.C)

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
                    n.thermal_potential_time_since_emergence < self.time_step_in_seconds):
                n.actual_elongation_rate = n.actual_elongation / (
                        n.thermal_potential_time_since_emergence / temperature_time_adjustment)
            else:
                n.actual_elongation_rate = n.actual_elongation / self.time_step_in_seconds

            n.radius += (n.potential_radius - n.initial_radius) * relative_growth_increase
            # The volume of the element is automatically calculated:
            n.volume = self.volume_from_radius_and_length(n, n.radius, n.length)
            # The new dry structural struct_mass of the element is calculated from its new volume:
            n.struct_mass = n.volume * n.root_tissue_density

            # In case where the root element corresponds to an apex, the distance to the last ramification is increased:
            if n.label == "Apex":
                n.dist_to_ramif += n.actual_elongation

            # VERIFICATION:
            if n.length < 0 or n.struct_mass < 0:
                print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length,
                      "and a mass of", n.struct_mass)

    # Formation of root nodules:
    # ---------------------------
    def nodule_formation(self, mother_element):
        """
        This function simulates the formation of one nodule on a root mother element. The nodule is considered as a special
        lateral root segment that has no apex connected and which cannot generate root primordium.
        :param mother_element: the mother element on which a new nodule element will be formed
        :return:
        """

        # We add a lateral root element called "nodule" on the mother element:
        nodule = self.ADDING_A_CHILD(mother_element, edge_type='+', label='Segment', type='Root_nodule',
                                     root_order=mother_element.root_order + 1,
                                     angle_down=90, angle_roll=0, length=0, radius=0,
                                     identical_properties=False, nil_properties=True)
        nodule.type = "Root_nodule"
        # nodule.length=mother_element.radius
        # nodule.radius=mother_element.radius/10.
        nodule.length = mother_element.radius
        nodule.radius = mother_element.radius
        nodule.original_radius = nodule.radius
        nodule.volume = self.volume_from_radius_and_length(nodule, nodule.radius, nodule.length)
        nodule.struct_mass = nodule.volume * nodule.root_tissue_density * self.struct_mass_C_content

        # print("Nodule", nodule.index(), "has been formed!")

        return nodule

    @postsegmentation
    @state
    def update_distance_from_tip(self):
        """
        The function "distance_from_tip" computes the distance (in meter) of a given vertex from the apex
        of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
        Note that the dist-to-tip of an apex is defined as its length (and not as 0).
        :return: the MTG with an updated property 'distance_from_tip'
        """

        # We initialize an empty dictionary for to_tips:
        to_tips = {}
        # We use the property "length" of each vertex based on the function "length":
        length = self.g.property('length')

        # We define "root" as the starting point of the loop below:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root tips to the base:
        for vid in post_order(self.g, root):
            # We define the current root element as n:
            n = self.g.node(vid)
            # We define its direct successor as son:
            son_id = self.g.Successor(vid)
            son = self.g.node(son_id)

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

    # Adding a new root element with pre-defined properties:
    def ADDING_A_CHILD(self, mother_element, edge_type='+', label='Apex', type='Normal_root_before_emergence',
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

        # TODO# FOR TRISTAN: When working with a dynamic root structure, you will need to specify in this function
        #  "ADDING_A_CHILD" your new variables that will either be set to 0 (nil properties) or be equal to that of the mother
        #  element.

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
                                                 volume=0.,
                                                 root_tissue_density=self.new_root_tissue_density,
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
                                                 # Root hairs:
                                                 # ------------
                                                 root_hair_radius=self.root_hair_radius,
                                                 root_hair_length=0.,
                                                 actual_length_with_hairs=0.,
                                                 living_root_hairs_number=0.,
                                                 dead_root_hairs_number=0.,
                                                 total_root_hairs_number=0.,
                                                 actual_time_since_root_hairs_emergence_started=0.,
                                                 thermal_time_since_root_hairs_emergence_started=0.,
                                                 actual_time_since_root_hairs_emergence_stopped=0.,
                                                 thermal_time_since_root_hairs_emergence_stopped=0.,
                                                 all_root_hairs_formed=False,
                                                 root_hairs_lifespan=self.root_hairs_lifespan,
                                                 root_hairs_struct_mass=0.,
                                                 root_hairs_struct_mass_produced=0.,
                                                 initial_living_root_hairs_struct_mass=0.,
                                                 living_root_hairs_struct_mass=0.,
                                                 # Fluxes:
                                                 # --------
                                                 resp_growth=0.,
                                                 struct_mass_produced=0.,
                                                 hexose_growth_demand=0.,
                                                 hexose_consumption_by_growth_amount=0.,
                                                 hexose_consumption_by_growth=0.,
                                                 hexose_possibly_required_for_elongation=0.,
                                                 # Time indications:
                                                 # ------------------
                                                 soil_temperature=7.8, # TODO change
                                                 growth_duration=self.GDs * (2 * radius) ** 2,
                                                 life_duration=self.LDs * 2. * radius * self.new_root_tissue_density,
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
            
            return new_child

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
                                                 volume=0.,
                                                 root_tissue_density=mother_element.root_tissue_density,
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
                                                 root_hairs_struct_mass=mother_element.root_hairs_struct_mass,
                                                 root_hairs_struct_mass_produced=mother_element.root_hairs_struct_mass_produced,
                                                 living_root_hairs_struct_mass=mother_element.living_root_hairs_struct_mass,
                                                 initial_living_root_hairs_struct_mass=mother_element.initial_living_root_hairs_struct_mass,
                                                 # Fluxes:
                                                 # -------
                                                 resp_growth=mother_element.resp_growth,
                                                 struct_mass_produced=mother_element.struct_mass_produced,
                                                 hexose_growth_demand=mother_element.hexose_growth_demand,
                                                 hexose_possibly_required_for_elongation=mother_element.hexose_possibly_required_for_elongation,
                                                 hexose_consumption_by_growth_amount=mother_element.hexose_consumption_by_growth_amount,
                                                 hexose_consumption_by_growth=mother_element.hexose_consumption_by_growth,
                                                 # Time indications:
                                                 # ------------------
                                                 soil_temperature=mother_element.soil_temperature,
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

    def volume_from_radius_and_length(self, element, radius: float, length: float):
        """
        This function computes the volume (m3) of a root element
        based on the properties radius (m) and length (m) and possibly on its type.
        :param element: the investigated node of the MTG
        :param radius: radius of the element
        :param length: length of the element
        :return: the volume of the element
        """

        # If this is a regular root segment
        if element.type != "Root_nodule":
            # We consider the volume of a cylinder
            volume = pi * radius ** 2 * length
        else:
            # We consider the volume of a sphere:
            volume = 4 / 3. * pi * radius ** 3

        return volume


    # Calculating the growth duration of a given root apex:
    # -----------------------------------------------------
    def calculate_growth_duration(self, radius, index, root_order):
        """
        This function computes the growth duration of a given apex, based on its radius and root order. If ArchiSimple
        option is activated, the function will calculate the duration proportionally to the square radius of the apex.
        Otherwise, the duration is set from a probability test, largely independent from the radius of the apex.
        :param radius: the radius of the apex element from which we compute the growth duration
        :param index: the index of the apex element, used for setting a new random seed for this element
        :param root_order: order of the considered segment
        :return: the growth duration of the apex (s)
        """

        # If we only want to apply original ArchiSimple rules:
        if self.ArchiSimple:
            # Then the growth duration of the apex is proportional to the square diameter of the apex:
            growth_duration = self.GDs * (2. * radius) ** 2
        # Otherwise, we define the growth duration as a fixed value, randomly chosen between three possibilities:
        else:
            # We first define the seed of random, depending on the index of the apex:
            np.random.seed(self.random_choice * index)
            # We then generate a random float number between 0 and 1, which will determine whether growth duration is low, medium or high:
            random_result = np.random.random_sample()
            # CASE 1: The apex corresponds to a seminal or adventitious root
            if root_order == 1:
                growth_duration = self.GD_highest
            else:
                # If we select random zoning, then the growth duration will be drawn from a range, for three different cases
                # (from most likely to less likely):
                if self.GD_by_frequency:
                    # CASE 2: Most likely, the growth duration will be low for a lateral root
                    if random_result < self.GD_prob_low:
                        # We draw a random growth-duration in the lower range:
                        growth_duration = np.random.uniform(0., self.GD_low)
                    # CASE 3: Occasionnaly, the growth duration may be a bit higher for a lateral root
                    if random_result < self.GD_prob_medium:
                        # We draw a random growth-duration in the lower range:
                        growth_duration = np.random.uniform(self.GD_low, self.GD_medium)
                    # CASE 3: Occasionnaly, the growth duration may be a bit higher for a lateral root
                    else:
                        # We draw a random growth-duration in the lower range:
                        growth_duration = np.random.uniform(self.GD_medium, self.GD_high)
                # If random zoning has not been selected, a constant duration is selected for each probabibility range:
                else:
                    # CASE 2: Most likely, the growth duration will be low for a lateral root
                    if random_result < self.GD_prob_low:
                        growth_duration = self.GD_low
                    # CASE 3: Occasionally, the growth duration of the lateral root may be significantly higher
                    elif random_result < self.GD_prob_medium:
                        growth_duration = self.GD_medium
                    # CASE 4: Exceptionally, the growth duration of the lateral root is as high as that from a seminal root,
                    # as long as the radius of the lateral root is high enough (i.e. twice as high as the minimal possible radius)
                    elif radius > 2 * self.Dmin / 2.:
                        growth_duration = self.GD_highest

        # We return a modified version of the MTG "g" with the updated property "distance_from_tip":
        return growth_duration
