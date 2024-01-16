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
from dataclasses import dataclass, field, fields
import inspect as ins
from functools import partial
from math import pi


@dataclass
class RootCarbonModel:
    """
    Root carbon balance model originating from Rhizodep shoot.py
    TODO adapt differential equation system
    forked :
        https://forgemia.inra.fr/tristan.gerault/rhizodep/-/commits/rhizodep_2022?ref_type=heads
    base_commit :
        92a6f7ad927ffa0acf01aef645f9297a4531878c
    """
    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---
    # FROM SOIL MODEL
    soil_temperature_in_Celsius: float = field(default=15, metadata=dict(unit="°C", unit_comment="", description="soil temperature in contact with roots", value_comment="", references="", variable_type="input", by="model_soil", state_variable_type="", edit_by="user"))
    C_hexose_soil: float = field(default=1e-5, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Hexose concentration in soil", value_comment="", references="", variable_type="input", by="model_soil", state_variable_type="", edit_by="user"))
    Cs_mucilage_soil: float = field(default=1e-5, metadata=dict(unit="mol.g-1", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", value_comment="", references="", variable_type="input", by="model_soil", state_variable_type="", edit_by="user"))
    Cs_cells_soil: float = field(default=1e-5, metadata=dict(unit="mol.g-1", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", value_comment="", references="", variable_type="input", by="model_soil", state_variable_type="", edit_by="user"))

    # FROM GROWTH MODEL
    type: str = field(default="Normal_root_after_emergence", metadata=dict(unit="", unit_comment="", description="Example segment type provided by root growth model", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    radius: float = field(default=3.5e-4, metadata=dict(unit="m", unit_comment="", description="Example root segment radius", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    length: float = field(default=3.e-3, metadata=dict(unit="m", unit_comment="", description="Example root segment length", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    struct_mass: float = field(default=1.35e-4, metadata=dict(unit="g", unit_comment="", description="Example root segment structural mass", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    initial_struct_mass: float = field(default=1.35e-4, metadata=dict(unit="g", unit_comment="", description="Same as struct_mass but corresponds to the previous time step; it is intended to record the variation", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    living_root_hairs_struct_mass: float = field(default=0., metadata=dict(unit="g", unit_comment="", description="Example root segment living root hairs structural mass", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    hexose_consumption_by_growth: float = field(default=0., metadata=dict(unit="g", unit_comment="", description="Hexose consumption by growth is coupled to a root growth model", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    distance_from_tip: float = field(default=3.e-3, metadata=dict(unit="m", unit_comment="", description="Example distance from tip", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))

    # FROM SHOOT MODEL
    sucrose_input_rate: float = field(default=1e-6, metadata=dict(unit="mol.s-1", unit_comment="", description="Sucrose input rate in phloem at collar point", value_comment="", references="", variable_type="input", by="model_shoot", state_variable_type="", edit_by="user"))

    # FROM ANATOMY MODEL
    root_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", value_comment="", references="", variable_type="input", by="model_anatomy", state_variable_type="extensive", edit_by="user"))
    phloem_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between root parenchyma and apoplasmic xylem vessels.", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="input", edit_by="user"))
    apoplasmic_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface to account for exchanges between xylem + stele apoplasm and soil. We account for it through cylindrical surface, a pathway closing as soon as endodermis differentiates", value_comment="", references="", variable_type="input", by="model_anatomy", state_variable_type="extensive", edit_by="user"))

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES
    # Pools initial size
    C_sucrose_root: float = field(default=0.0100 / 12.01 / 12, metadata=dict(unit="mol.g-1", unit_comment="of sucrose", description="Sucrose concentration in root", value_comment="", references="0.0025 is a plausible value according to the results of Gauthier (2019, pers. communication), but here, we use a plausible sucrose concentration (10 mgC g-1) in roots according to various experimental results.", variable_type="state_variable", by="model_carbon", state_variable_type="intensive", edit_by="user"))
    C_hexose_root: float = field(default=1e-3, metadata=dict(unit="mol.g-1", unit_comment="of sucrose", description="Hexose concentration in root", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="intensive", edit_by="user"))
    C_hexose_reserve: float = field(default=1e-3 * 2., metadata=dict(unit="mol.g-1", unit_comment="of sucrose", description="Hexose reserve concentration in root", value_comment="C_hexose_root * 2",  references="We expect the reserve pool to be two times higher than the mobile one.", variable_type="state_variable", by="model_carbon", state_variable_type="intensive", edit_by="user"))

    # Transport Processes
    hexose_diffusion_from_phloem: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    hexose_active_production_from_phloem: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    sucrose_loading_in_phloem: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    hexose_exudation: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    phloem_hexose_exudation: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    hexose_uptake_from_soil: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    phloem_hexose_uptake_from_soil: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    mucilage_secretion: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of equivalent hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    cells_release: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of equivalent hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))

    # Metabolic Processes
    hexose_mobilization_from_reserve: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    hexose_immobilization_as_reserve: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    maintenance_respiration: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))

    # Deficits
    deficit_sucrose_root: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of sucrose", description="Sucrose deficit rate in root", value_comment="", references="Hypothesis of no initial deficit", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    deficit_hexose_reserve: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose reserve", description="Hexose reserve deficit rate in root", value_comment="", references="Hypothesis of no initial deficit", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    deficit_hexose_root: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="Hexose deficit rate in root", value_comment="", references="Hypothesis of no initial deficit", variable_type="state_variable", by="model_carbon", state_variable_type="extensive", edit_by="user"))

    # SUMMED STATE VARIABLES
    total_sucrose_root: float = field(default=0., metadata=dict(unit="mol", unit_comment="of sucrose", description="Summed sucrose root at root system level", value_comment="", references="", variable_type="plant_scale_state", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    total_living_struct_mass: float = field(default=0., metadata=dict(unit="g", unit_comment="", description="Summed structural mass at root system level", value_comment="", references="", variable_type="plant_scale_state", by="model_carbon", state_variable_type="extensive", edit_by="user"))
    global_sucrose_deficit: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of sucrose", description="Summed sucrose deficit at root system level", value_comment="", references="", variable_type="plant_scale_state", by="model_carbon", state_variable_type="extensive", edit_by="user"))

    # --- INITIALIZES MODEL PARAMETERS ---

    # Temperature
    process_at_T_ref: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="Proportion of maximal process intensity occuring at T_ref", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    phloem_unloading_T_ref: float = field(default=10, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We reuse the observed evolution of Frankenberger and Johanson (1983) on invertase activity in different soils with temperature from 10 to 100 degree Celsius which show an increase of about 5 times between 20 degrees and 50 degrees (maximum), assuming that the activity of invertase outside the phloem tissues is correlated to the unloading rate of sucrose from phloem.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    phloem_unloading_A: float = field(default=-0.04, metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="Frankenberger and Johanson (1983), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    phloem_unloading_B: float = field(default=2.9, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="Frankenberger and Johanson (1983), see T_ref", variable_type="parametyer", by="model_carbon", state_variable_type="", edit_by="user"))
    phloem_unloading_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="Frankenberger and Johanson (1983), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    max_loading_rate_T_ref: float = field(default=10, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We reuse the temperature-evolution used for phloem unloading, based on the work of Frankenberger and Johanson (1983) (phloem_unloading_T_ref)", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_loading_rate_A: float = field(default=-0.04, metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="Frankenberger and Johanson (1983), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_loading_rate_B: float = field(default=2.9, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="Frankenberger and Johanson (1983), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_loading_rate_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="Frankenberger and Johanson (1983), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    max_immobilization_rate_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="According to the work of Mohabir and John (1988) on starch synthesis in potatoe tubers based on labelled sucrose incorporation in disks of starch, the immobilization increased until 21.5 degree Celsius and then decreases again. We fitted the evolution of starch synthesis with temperature (8-30 degrees) to get the parameters estimation.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_immobilization_rate_A: float = field(default=-0.0521, metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="Mohabir and John (1988), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_immobilization_rate_B: float = field(default=0.861, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="Mohabir and John (1988), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_immobilization_rate_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="Mohabir and John (1988), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    max_mobilization_rate_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="max_immobilization_rate_T_ref", references="We assume that the mobilization obeys to the same evolution with temperature as the immobilization process.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_mobilization_rate_A: float = field(default=-0.0521, metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="max_immobilization_rate_A", references="We assume that the mobilization obeys to the same evolution with temperature as the immobilization process.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_mobilization_rate_B: float = field(default=0.861, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="max_immobilization_rate_B", references="We assume that the mobilization obeys to the same evolution with temperature as the immobilization process.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_mobilization_rate_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="max_immobilization_rate_C", references="We assume that the mobilization obeys to the same evolution with temperature as the immobilization process.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    resp_maintenance_max_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We fitted the parameters on the mean curve of maintenance respiration of whole-plant wheat obtained from Gifford (1995).", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    resp_maintenance_max_A: float = field(default=-0.0442, metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="Gifford (1995), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    resp_maintenance_max_B: float = field(default=1.55, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="Gifford (1995), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    resp_maintenance_max_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="Gifford (1995), see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    permeability_coeff_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We assume that the permeability does not directly depend on temperature, according to the contrasted results obtained by Wan et al. (2001) on poplar, Shen and Yan (2002) on crotalaria, Hill et al. (2007) on wheat, or Kaldy (2012) on a sea grass.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    permeability_coeff_A: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    permeability_coeff_B: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    permeability_coeff_C: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    uptake_rate_max_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    uptake_rate_max_A: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    uptake_rate_max_B: float = field(default=3.82, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="The value for B (Q10) is adapted from the work of Coody et al. (1986, SBB), who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    uptake_rate_max_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    secretion_rate_max_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We arbitrarily assume that the secretion of mucilage exponentially increases with soil temperature with a Q10 of 2, although we could not find any experimental evidence for this.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    secretion_rate_max_A: float = field(default=-0., metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    secretion_rate_max_B: float = field(default=2., metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    secretion_rate_max_C: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    surfacic_cells_release_rate_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="This corresponds to a bell-shape where the maximum is obtained at 31 degree Celsius, obtained by fitting the data from Clowes and Wadekar (1988) on Zea mays roots between 15 and 35 degree.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    surfacic_cells_release_rate_A: float = field(default=-0.187, metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    surfacic_cells_release_rate_B: float = field(default=2.48, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    surfacic_cells_release_rate_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="see T_ref", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    # C METABOLIC PROCESSES
    # kinetic parameters
    max_immobilization_rate: float = field(default=1.8e-9, metadata=dict(unit="mol.g-1.s-1", unit_comment="", description="Maximum immobilization rate of hexose in the reserve pool", value_comment="overwrite 8 * 1e-6 / 6. from According to Gauthier et al. (2020): the new maximum rate of starch synthesis for vegetative growth in the shoot", references="According to the work of Mohabir and John (1988) on starch synthesis in potatoe tubers based on labelled sucrose incorporation in disks of starch, the immobilization rate is about 1.8e-9 at the temperature of 20 degree Celsius, assuming that the potatoe starch content is 65.6% of dry matter and that the structural mass is 28.5% (data taken from the data of Jansen et al. (2001)).", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_mobilization_rate: float = field(default=1.8e-9, metadata=dict(unit="mol.g-1.s-1", unit_comment="", description="Maximum mobilization rate of hexose from the reserve pool", value_comment="", references="max_immobilization_rate", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    C_hexose_reserve_min: float = field(default=0., metadata=dict(unit="mol.g-1", unit_comment="", description="Minimum concentration of hexose in the reserve pool", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    expected_C_hexose_root: float = field(default=1.3, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Expected hexose concentration in root", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    expected_C_hexose_soil: float = field(default=1.3 / 100., metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Expected hexose concentration in soil", value_comment="expected_C_hexose_root / 100", references="We expect the soil concentration to be 2 orders of magnitude lower than the root concentration. (??)", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    expected_C_hexose_reserve: float = field(default=1.3 * 2., metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Expected hexose concentration in the reserve pool", value_comment="expected_C_hexose_root", references="We expect the reserve pool to be two times higher than the mobile one. (??)", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_mobilization: float = field(default=1.3 * 2. * 5., metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for hexose remobilization from the reserve", value_comment="expected_C_hexose_reserve", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    C_hexose_root_min_for_reserve: float = field(default=5e-3, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Minimum concentration of hexose in the mobile pool for immobilization", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    C_hexose_reserve_max: float = field(default=5e-3, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Maximum concentration of hexose in the reserve pool", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_immobilization: float = field(default=1.3, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for hexose immobilization in the reserve pool", value_comment="expected_C_hexose_root", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    resp_maintenance_max: float = field(default=5e-8, metadata=dict(unit="mol.g-1.s-1", unit_comment="of CO2", description="Maximal maintenance respiration rate", value_comment="overwritting 4.1e-6 / 6. * (0.44 / 60. / 14.01) * 5 from => According to Barillot et al. (2016): max. residual maintenance respiration rate is 4.1e-6 umol_C umol_N-1 s-1, i.e. 4.1e-6/60*0.44 mol_C g-1 s-1 assuming an average struct_C:N_tot molar ratio of root of 60 [cf simulations by CN-Wheat 47 in 2020] and a C content of struct_mass of 44%. According to the same simulations, total maintenance respiration is about 5 times more than residual maintenance respiration.", references="According to Gifford (1995): the total maintenance respiration rate of the whole plant of wheat is about 0.024 gC gC-1 day-1, i.e. 5.28 e-8 assuming that the C to which this rate is related represents 44% of the dry structural biomass.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_maintenance: float = field(default=1.67e3 * 1e-6 / 6., metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for maintenance respiration", value_comment="", references="According to Barillot et al. (2016): Km=1670 umol of C per gram for residual maintenance respiration (based on sucrose concentration!).", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    # C TRANSPORT PROCESSES
    # kinetic parameters
    phloem_permeability: float = field(default=2e-4, metadata=dict(unit="g.m-2.s-1", unit_comment="", description="Coefficient of permeability of unloading phloem", value_comment="cheating, 5.76 override", references="According to Ross-Eliott et al. (2017), an unloading flow of sucrose of 1.2e-13 mol of sucrose per second can be calculated for a sieve element of 3.6 µm and the length of the unloading zone of 350 µm, # assuming a phloem concentration of 0.5 mol/l. We calculated that this concentration corresponded to 5.3 e-6 mol/gDW, considering that the sieve element was filled with phloem sap, that the root diameter was 111 µm, and that root tissue density was 0.1 g cm-3. Calculating an exchange surface of the sieve tube of 4e-9 m2, we obtained a permeability coefficient of 5.76 gDW m-2 s-1 using the values of the flow, of the gradient of sugar concentration (assuming hexose concentration was 0) and of the exchange surface.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    reference_rate_of_hexose_consumption_by_growth: float = field(default=3e-14, metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="Coefficient of permeability of unloading phloem", value_comment="", references="Reference consumption rate of hexose for growth for a given root element (used to multiply the reference unloading rate when growth has consumed hexose)", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_unloading_rate: float = field(default=2e-7, metadata=dict(unit="mol.g-1.s-1", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_unloading: float = field(default=1000 * 1e-6 / 12., metadata=dict(unit="mol.g-1", unit_comment="of sucrose", description="Affinity constant for sucrose unloading", value_comment="", references="According to Barillot et al. (2016b), this value is 1000 umol C g-1", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    max_loading_rate: float = field(default=2e-7, metadata=dict(unit="mol.g-1.s-1", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_loading: float = field(default=1000 * 1e-6 / 12. * 2, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for sucrose loading", value_comment="Km_unloading", references="We expect the Km of loading to be equivalent as the Km of unloading, as it may correspond to the same transporter.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    expected_exudation_efflux: float = field(default=608 * 0.000001 / 12.01 / 6 / 3600 * 1 / (0.5 * 10), metadata=dict(unit="mol.m-2.s-1", unit_comment="of hexose", description="Expected exudation rate", value_comment="", references="According to Jones and Darrah (1992): the net efflux of C for a root of maize is 608 ug C g-1 root DW h-1, and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface. OR: expected_exudation_efflux = 5.2 / 12.01 / 6. * 1e-6 * 100. ** 2. / 3600. Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Pmax_apex: float = field(default=608 * 0.000001 / 12.01 / 6 / 3600 * 1 / (0.5 * 10) / (1.3 + 1.3 / 100), metadata=dict(unit="g.m-2.s-1", unit_comment="", description="Permeability coefficient", value_comment="expected_exudation_efflux / (expected_C_hexose_root - expected_C_hexose_soil)", references="We calculate the permeability according to the expected exudation flux and expected concentration gradient between cytosol and soil.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    uptake_rate_max: float = field(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1), metadata=dict(unit="mol.m-2.s-1", unit_comment="of hexose", description="Maximum rate of influx of hexose from soil to roots", value_comment="", references="According to Jones and Darrah (1996), the uptake rate measured for all sugars tested with an individual external concentration of 100 uM is equivalent to 277 nmol hexose mg-1 day-1, and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_uptake: float = field(default=1000 * 1e-6 / 12. * 2, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for hexose uptake", value_comment="Km_loading", references="We assume that the transporters able to reload sugars in the phloem and to take up sugars from the soil behave in a similar manner.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    secretion_rate_max: float = field(default=1e-5/(12.01*6)/(0.1*pi)*1e4/(60.*60.*24.), metadata=dict(unit="mol.m-2.s-1", unit_comment="of hexose", description="Maximum rate of mucilage secretion", value_comment="", references="According to the measurements of Paull et al. (1975) and Chaboud (1983) on maize and of Horst et al. (1982) on cowpea, the mucilage secretion rate at root tip evolves within 5 to 10 ugC per cm of root per day. We therefore chose 10 gC per cm per day as the maximal rate, and convert it in mol of hexose per m2 per second, assuming that the root tip is a cylinder of 1 mm diameter.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    gamma_secretion: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="Coefficient affecting the decrease of mucilage secretion with distance from the apex", value_comment="", references="We assume that the mucilage secretion rapidly decreases when moving away from the apex.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Cs_mucilage_soil_max: float = field(default=10., metadata=dict(unit="mol.m-2", unit_comment="of equivalent hexose", description="Maximal surfacic concentration of mucilage at the soil-root interface, above which no mucilage secretion is possible", value_comment="DO A REAL ESTIMATION!", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Km_secretion: float = field(default=1000 * 1e-6 / 12. * 2 / 2, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for hexose uptake", value_comment="Km_loading", references="We assume that the concentration of root hexose for which mucilage secretion is half of the maximal rate is two-times lower than the one for which phloem reloading is half of the maximal rate.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    growing_zone_factor: float = field(default=8 * 2., metadata=dict(unit="adim", unit_comment="", description="Proportionality factor between the radius and the length of the root apical zone in which C can sustain root elongation", value_comment="", references="According to illustrations by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap, meristem and elongation zones is about 8 times the diameter of the tip.", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    surfacic_cells_release_rate: float = field(default=1279/6476*(33.9*1e6*1e-18)*0.1*1000000*0.44/12.01/6/(24*60*60)/(0.8*0.001*pi*0.8*0.001), metadata=dict(unit="mol.m-2.s-1", unit_comment="mol of equivalent-hexose per m2 of lateral external surface per second", description="Average release of root cells", value_comment="", references="We used the measurements by Clowes and Wadekar 1988 on Zea mays root cap cells obtained at 20 degree Celsius, i.e. 1279 cells per day. We recalculated the amount of equivalent hexose by relating the number of cap cells produced per day to a volume knowing that the whole cap was made of 6476 cells and had a volume of 33.9 *10^6 micrometer^3. The volume was later converted into a mass assuming a density of 0.1 g cm-3. We then assumed that the surface of root cap was equivalent to the lateral surface of a cylinder of radius 0.8 mm and height 0.8 mm (meristem size = 0.79-0.81 mm).", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))
    Cs_cells_soil_max: float = field(default=10., metadata=dict(unit="mol.m-2", unit_comment="of equivalent hexose", description="Maximal surfacic concentration of root cells in soil, above which no release of cells is possible", value_comment="DO A REAL ESTIMATION!", references="", variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user"))

    def __init__(self, g, time_step_in_seconds: int, **scenario: dict):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step_in_seconds: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """
        self.g = g
        self.props = self.g.properties()
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        self.time_steps_in_seconds = time_step_in_seconds
        self.available_inputs = []

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)

        self.state_variables = [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]

        for name in self.state_variables:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
            # set default in mtg
            self.props[name].update({key: getattr(self, name) for key in self.vertices})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

        # Repeat the same process for total root system properties
        self.plant_scale_states = [f.name for f in fields(self) if f.metadata["variable_type"] == "plant_scale_state"]

        for name in self.plant_scale_states:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
            # set default in mtg
            self.props[name].update({1: getattr(self, name)})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

    def apply_scenario(self, **kwargs):
        """
        Method to superimpose default parameters in order to create a scenario.
        Use Model.documentation to discover model parameters and state variables.
        :param kwargs: mapping of existing variable to superimpose.
        """
        for changed_parameter, value in kwargs.items():
            if changed_parameter in dir(self):
                setattr(self, changed_parameter, value)

    def post_coupling_init(self):
        self.get_available_inputs()
        self.store_functions_call()
        self.check_if_coupled()

    def store_functions_call(self):
        # Storing function calls

        # Local and plant scale processes...
        self.process_methods = [getattr(self, func) for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_names = [func[8:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'process' in func)]

        # Local and plant scale update...
        self.update_methods = [getattr(self, func) for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                             for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_names = [func[7:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'update' in func)]

        # Local deficits update...
        self.deficit_methods = [getattr(self, func) for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'compute' in func)]
        self.deficit_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                             for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'compute' in func)]
        self.deficit_names = [func[8:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'compute' in func)]

    def check_if_coupled(self):
        # For all expected input...
        input_variables = [f.name for f in fields(self) if f.metadata["variable_type"] == "input"]
        for inpt in input_variables:
            # If variable type has not gone to dictionary as it is part of the coupling process
            # we use provided default value to create the dictionnary used in the rest of the model
            if type(getattr(self, inpt)) != dict:
                if inpt not in self.props.keys():
                    self.props.setdefault(inpt, {})
                # set default in mtg
                self.props[inpt].update({key: getattr(self, inpt) for key in self.vertices})
                # link mtg dict to self dict
                setattr(self, inpt, self.props[inpt])

    def get_available_inputs(self):
        for inputs in self.available_inputs:
            source_model = inputs["applier"]
            linker = inputs["linker"]
            for name, source_variables in linker.items():
                # if variables have to be summed
                if len(source_variables.keys()) > 1:
                    return setattr(self, name, dict(zip(getattr(source_model, "vertices"), [sum([getattr(source_model, source_name)[vid] * unit_conversion for source_name, unit_conversion in source_variables.items()]) for vid in getattr(source_model, "vertices")])))
                else:
                    return setattr(self, name, getattr(source_model, list(source_variables.keys())[0]))

    def run_exchanges_and_balance(self):
        """
        Groups and order carbon exchange processes and balance for the provided time step
        """
        # We have to renew this call at each time step to ensure model inputs are well updated
        self.get_available_inputs()

        # TODO print(self.struct_mass, self.initial_struct_mass to check is they are indeed different here)
        self.post_growth_updating()

        self.props.update(self.prc_resolution())
        self.props.update(self.upd_resolution())

        # Computing the deficits after this balance
        self.props.update(self.dfc_resolution())
        # Then bring concentrations back to 0 if negative
        # TODO NOTE : a bit raw
        self.props.update(self.forbid_negatives())

        # Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
        self.shoot_sucrose_supply_and_spreading()

        # # OPTIONAL: checking of possible anomalies in the root system:
        self.control_of_anomalies()

    def prc_resolution(self):
        return dict(zip([name for name in self.process_names], map(self.dict_mapper, *(self.process_methods, self.process_args))))

    def upd_resolution(self):
        return dict(zip([name for name in self.update_names], map(self.dict_mapper, *(self.update_methods, self.update_args))))

    def dfc_resolution(self):
        return dict(zip([name for name in self.deficit_names], map(self.dict_mapper, *(self.deficit_methods, self.deficit_args))))

    def forbid_negatives(self):
        return dict(zip(["C_hexose_root", "C_hexose_reserve", "C_sucrose_root"],
                        map(self.dict_mapper,
                            *([partial(max, 0.)] * 3, [[partial(self.get_up_to_date, "C_hexose_root")],
                                                       [partial(self.get_up_to_date, "C_hexose_reserve")],
                                                       [partial(self.get_up_to_date, "C_sucrose_root")]]))))

    def dict_mapper(self, fcn, args):
        return dict(zip(args[0](), map(fcn, *(d().values() for d in args))))

    def get_up_to_date(self, prop):
        return getattr(self, prop)

    def post_growth_updating(self):
        """
        Description :
            Extend property dictionnary uppon new element partionning and updates concentrations uppon structural_mass change
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.C_sucrose_root.keys()):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                for prop in self.state_variables:
                    # if intensive, equals to parent
                    if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                        getattr(self, prop).update({vid: getattr(self, prop)[parent]})
                    # if extensive, we need structural mass wise partitioning
                    else:
                        getattr(self, prop).update({vid: getattr(self, prop)[parent] * mass_fraction,
                                                    parent: getattr(self, prop)[parent] * (1-mass_fraction)})
            else:
                for prop in self.state_variables:
                    # if intensive, concentrations have to be updated based on new structural mass
                    if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                        getattr(self, prop).update({vid: getattr(self, prop)[vid] * (
                            self.initial_struct_mass[vid] / self.struct_mass[vid])})

    # Modification of a process according to soil temperature:
    # --------------------------------------------------------
    def temperature_modification(self, process_at_T_ref=1., soil_temperature=15, T_ref=0., A=-0.05, B=3., C=1.):
        """
        This function calculates how the value of a process should be modified according to soil temperature (in degrees Celsius).
        Parameters correspond to the value of the process at reference temperature T_ref (process_at_T_ref),
        to two empirical coefficients A and B, and to a coefficient C used to switch between different formalisms.
        If C=0 and B=1, then the relationship corresponds to a classical linear increase with temperature (thermal time).
        If C=1, A=0 and B>1, then the relationship corresponds to a classical exponential increase with temperature (Q10).
        If C=1, A<0 and B>0, then the relationship corresponds to bell-shaped curve, close to the one from Parent et al. (2010).
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
            print("The modification of the process at T =", soil_temperature, "only works for C=0 or C=1!")
            print("The modified process has been set to 0.")
            modified_process = 0.
            return 0.
        elif C == 1:
            if (A * (soil_temperature - T_ref) + B) < 0.:
                print("The modification of the process at T =", soil_temperature,
                      "is unstable with this set of parameters!")
                print("The modified process has been set to 0.")
                modified_process = 0.
                return modified_process

        # We compute a temperature-modified process, correspond to a Q10-modified relationship,
        # based on the work of Tjoelker et al. (2001):
        modified_process = process_at_T_ref * (A * (soil_temperature - T_ref) + B) ** (1 - C) \
                           * (A * (soil_temperature - T_ref) + B) ** (C * (soil_temperature - T_ref) / 10.)

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
            if self.length[vid] <= 0.:
                continue
            else:
                # We increment the total amount of sucrose in the root system, including dead root elements (if any):
                self.total_sucrose_root[1] += self.C_sucrose_root[vid] * (
                            self.struct_mass[vid] + self.living_root_hairs_struct_mass[vid]) \
                                           - self.deficit_sucrose_root[vid]

                # We only select the elements that have a positive struct_mass and are not dead
                # (if they have just died, we still include them in the balance):
                if self.struct_mass[vid] > 0. and self.type[vid] != "Dead" and self.type[vid] != "Just_dead":
                    # We calculate the total living struct_mass by summing all the local struct_masses:
                    self.total_living_struct_mass[1] += self.struct_mass[vid] + self.living_root_hairs_struct_mass[vid]

    # Calculating the net input of sucrose by the aerial parts into the root system:
    # ------------------------------------------------------------------------------
    def shoot_sucrose_supply_and_spreading(self):
        """
        This function calculates the new root sucrose concentration (mol of sucrose per gram of dry root structural mass)
        AFTER supplying sucrose from the shoot.
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
                    self.sucrose_input_rate[1] * self.time_steps_in_seconds) - self.global_sucrose_deficit[1]) \
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
            self.deficit_sucrose_root[vid] = 0.

    # Unloading of sucrose from the phloem and conversion of sucrose into hexose:

    # --------------------------------------------------------------------------
    # These 3 functions simulate the rate of sucrose unloading from phloem and its immediate conversion into hexose.
    # The hexose pool is assumed to correspond to the symplastic compartment of root cells from the stelar and cortical
    # parenchyma and from the epidermis. The function also simulates the opposite process of sucrose loading,
    # considering that 2 mol of hexose are produced for 1 mol of sucrose.

    def process_hexose_diffusion_from_phloem(self, length, phloem_exchange_surface, C_sucrose_root, C_hexose_root,
                                             hexose_consumption_by_growth, soil_temperature_in_Celsius):
        # We consider all the cases where no net exchange should be allowed:
        if length <= 0. or phloem_exchange_surface <= 0. or type == "Just_dead" or type == "Dead":
            return 0

        else:
            if (self.global_sucrose_deficit[1] > 0.) or (C_sucrose_root <= C_hexose_root / 2.):
                return 0
            else:
                phloem_permeability = self.phloem_permeability * (1 + hexose_consumption_by_growth /
                                                                  self.reference_rate_of_hexose_consumption_by_growth)
                phloem_permeability *= self.temperature_modification(soil_temperature=soil_temperature_in_Celsius,
                                                                     T_ref=self.phloem_unloading_T_ref,
                                                                     A=self.phloem_unloading_A,
                                                                     B=self.phloem_unloading_B,
                                                                     C=self.phloem_unloading_C)
                return max(2. * phloem_permeability * (C_sucrose_root - C_hexose_root / 2.) * phloem_exchange_surface, 0)

    def process_hexose_active_production_from_phloem(self, length, phloem_exchange_surface, C_sucrose_root, C_hexose_root,
                                                     hexose_consumption_by_growth, soil_temperature_in_Celsius):
        # We consider all the cases where no net exchange should be allowed:
        if length <= 0. or phloem_exchange_surface <= 0. or type == "Just_dead" or type == "Dead":
            return 0

        else:
            if (self.global_sucrose_deficit[1] > 0.) or (C_sucrose_root <= C_hexose_root / 2.):
                return 0
            else:
                max_unloading_rate = self.max_unloading_rate * (1 + hexose_consumption_by_growth /
                                                                self.reference_rate_of_hexose_consumption_by_growth)
                max_unloading_rate *= self.temperature_modification(soil_temperature=soil_temperature_in_Celsius,
                                                                    T_ref=self.phloem_unloading_T_ref,
                                                                    A=self.phloem_unloading_A,
                                                                    B=self.phloem_unloading_B,
                                                                    C=self.phloem_unloading_C)
                return max(2. * max_unloading_rate * C_sucrose_root * phloem_exchange_surface / (
                            self.Km_unloading + C_sucrose_root), 0)

    def process_sucrose_loading_in_phloem(self, phloem_exchange_surface, C_hexose_root, soil_temperature_in_Celsius):
        # # We correct the max loading rate according to the distance from the tip in the middle of the segment.
        # max_loading_rate = param.surfacic_loading_rate_reference \
        #     * (1. - 1. / (1. + ((distance_from_tip-length/2.) / original_radius) ** param.gamma_loading))
        # TODO: Reconsider the way the variation of the max loading rate along the root axis has been described!

        # We correct loading according to soil temperature:
        max_loading_rate = self.max_loading_rate * self.temperature_modification(soil_temperature=soil_temperature_in_Celsius,
                                                                                 T_ref=self.max_loading_rate_T_ref,
                                                                                 A=self.max_loading_rate_A,
                                                                                 B=self.max_loading_rate_B,
                                                                                 C=self.max_loading_rate_C)
        if C_hexose_root <= 0.:
            return 0.
        else:
            return max(0.5 * max_loading_rate * phloem_exchange_surface * C_hexose_root / (self.Km_loading + C_hexose_root), 0.)

    def process_hexose_mobilization_from_reserve(self, length, C_hexose_root, C_hexose_reserve, type, struct_mass,
                                         living_root_hairs_struct_mass, soil_temperature_in_Celsius):
        """
        TODO not used??
        CALCULATIONS OF THEORETICAL IMMOBILIZATION RATE
        We verify that the element does not correspond to a primordium that has not emerged:
        """
        if length <= 0. or C_hexose_root < 0. or C_hexose_reserve < 0. or type == "Root_nodule":
            return 0.
        else:
            # We correct maximum rates according to soil temperature:
            corrected_max_mobilization_rate = self.max_mobilization_rate * self.temperature_modification(
                                                                        soil_temperature=soil_temperature_in_Celsius,
                                                                        T_ref=self.max_mobilization_rate_T_ref,
                                                                        A=self.max_mobilization_rate_A,
                                                                        B=self.max_mobilization_rate_B,
                                                                        C=self.max_mobilization_rate_C)
            if C_hexose_reserve <= self.C_hexose_reserve_min:
                return 0.
            else:
                return corrected_max_mobilization_rate * C_hexose_reserve / (
                        self.Km_mobilization + C_hexose_reserve) * (struct_mass + living_root_hairs_struct_mass)

    def process_hexose_immobilization_as_reserve(self, C_hexose_root, C_hexose_reserve, type,
                                          struct_mass, living_root_hairs_struct_mass, soil_temperature_in_Celsius):
        # CALCULATIONS OF THEORETICAL IMMOBILIZATION RATE:
        if C_hexose_root <= self.C_hexose_root_min_for_reserve or C_hexose_reserve >= self.C_hexose_reserve_max \
                or type == "Just_dead" or type == "Dead":
            return 0.
        else:
            corrected_max_immobilization_rate = self.max_immobilization_rate \
                                                * self.temperature_modification(
                                                                        soil_temperature=soil_temperature_in_Celsius,
                                                                        T_ref=self.max_immobilization_rate_T_ref,
                                                                        A=self.max_immobilization_rate_A,
                                                                        B=self.max_immobilization_rate_B,
                                                                        C=self.max_immobilization_rate_C)

            return corrected_max_immobilization_rate * C_hexose_root / (self.Km_immobilization + C_hexose_root) * (
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

    def process_maintenance_respiration(self, type, C_hexose_root, struct_mass, living_root_hairs_struct_mass, soil_temperature_in_Celsius):

        # TODO FOR TRISTAN: Consider expliciting a respiration cost associated to the exchange of N in the root (e.g. cost for uptake, xylem loading)?

        # CONSIDERING CASES THAT SHOULD BE AVOIDED:
        # We consider that dead elements cannot respire (unless over the first time step following death,
        # i.e. when the type is "Just_dead"):
        if type == "Dead" or C_hexose_root <= 0.:
            return 0.
        else:
            # We correct the maximal respiration rate according to soil temperature:
            corrected_resp_maintenance_max = self.resp_maintenance_max * self.temperature_modification(
                                                                    soil_temperature=soil_temperature_in_Celsius,
                                                                    T_ref=self.resp_maintenance_max_T_ref,
                                                                    A=self.resp_maintenance_max_A,
                                                                    B=self.resp_maintenance_max_B,
                                                                    C=self.resp_maintenance_max_C)

            return corrected_resp_maintenance_max * C_hexose_root / (self.Km_maintenance + C_hexose_root) * (
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

    def process_hexose_exudation(self, length, root_exchange_surface, C_hexose_root, C_hexose_soil, soil_temperature_in_Celsius):
        if length <= 0 or root_exchange_surface <= 0. or C_hexose_root <= 0.:
            return 0.
        else:
            corrected_P_max_apex = self.Pmax_apex * self.temperature_modification(
                                                                   soil_temperature=soil_temperature_in_Celsius,
                                                                   T_ref=self.permeability_coeff_T_ref,
                                                                   A=self.permeability_coeff_A,
                                                                   B=self.permeability_coeff_B,
                                                                   C=self.permeability_coeff_C)
            # if distance_from_tip < length:
            #     print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
            # else:
            #     corrected_permeability_coeff = corrected_P_max_apex \
            #                          / (1 + (distance_from_tip - length / 2.) / original_radius) ** param.gamma_exudation
            corrected_permeability_coeff = corrected_P_max_apex

            return max(corrected_permeability_coeff * (C_hexose_root - C_hexose_soil) * root_exchange_surface,
                       0)

    def process_phloem_hexose_exudation(self, length, root_exchange_surface, C_hexose_root, C_sucrose_root,
                                        C_hexose_soil, apoplasmic_exchange_surface, soil_temperature_in_Celsius):
        if length <= 0 or root_exchange_surface <= 0. or C_hexose_root <= 0.:
            return 0.
        else:
            corrected_P_max_apex = self.Pmax_apex * self.temperature_modification(
                                                                   soil_temperature=soil_temperature_in_Celsius,
                                                                   T_ref=self.permeability_coeff_T_ref,
                                                                   A=self.permeability_coeff_A,
                                                                   B=self.permeability_coeff_B,
                                                                   C=self.permeability_coeff_C)
            # if distance_from_tip < length:
            #     print("!!!ERROR!!! The distance to tip is lower than the length of the root element", vid)
            # else:
            #     corrected_permeability_coeff = corrected_P_max_apex \
            #                          / (1 + (distance_from_tip - length / 2.) / original_radius) ** param.gamma_exudation
            corrected_permeability_coeff = corrected_P_max_apex
            return corrected_permeability_coeff * (2 * C_sucrose_root - C_hexose_soil) * apoplasmic_exchange_surface

    # Uptake of hexose from the soil by the root:
    # -------------------------------------------
    # This function computes the rate of hexose uptake by roots from the soil. This influx of hexose is represented
    # as an active process with a substrate-limited relationship (Michaelis-Menten function) depending on the hexose
    # concentration in the soil.

    def process_hexose_uptake_from_soil(self, length, root_exchange_surface, C_hexose_soil, type, soil_temperature_in_Celsius):
        if length <= 0 or root_exchange_surface <= 0. or C_hexose_soil <= 0. or type == "Just_dead" or type == "Dead":
            return 0.
        else:
            corrected_uptake_rate_max = self.uptake_rate_max * self.temperature_modification(
                                                                                soil_temperature=soil_temperature_in_Celsius,
                                                                                T_ref=self.uptake_rate_max_T_ref,
                                                                                A=self.uptake_rate_max_A,
                                                                                B=self.uptake_rate_max_B,
                                                                                C=self.uptake_rate_max_C)

            return corrected_uptake_rate_max * root_exchange_surface \
                * C_hexose_soil / (self.Km_uptake + C_hexose_soil)

    def process_phloem_hexose_uptake_from_soil(self, length, apoplasmic_exchange_surface, C_hexose_soil, type, soil_temperature_in_Celsius):
        if length <= 0 or apoplasmic_exchange_surface <= 0. or C_hexose_soil <= 0. or type == "Just_dead" or type == "Dead":
            return 0.
        else:
            corrected_uptake_rate_max = self.uptake_rate_max * self.temperature_modification(
                                                                                    soil_temperature=soil_temperature_in_Celsius,
                                                                                    T_ref=self.uptake_rate_max_T_ref,
                                                                                    A=self.uptake_rate_max_A,
                                                                                    B=self.uptake_rate_max_B,
                                                                                    C=self.uptake_rate_max_C)

            return corrected_uptake_rate_max * apoplasmic_exchange_surface * C_hexose_soil / (
                    self.Km_uptake + C_hexose_soil)

    # Mucilage secretion:
    # ------------------
    # This function computes the rate of mucilage secretion (in mol of equivalent hexose per second) for a given root element n.
    def process_mucilage_secretion(self, length, root_exchange_surface, C_hexose_root, type, distance_from_tip, radius, Cs_mucilage_soil, soil_temperature_in_Celsius):
        # First, we ensure that the element has a positive length and surface of exchange:
        if length <= 0 or root_exchange_surface <= 0. or C_hexose_root <= 0. or type == "Dead" or type == "Stopped" or distance_from_tip < length:
            return 0.
        else:
            # We correct the maximal secretion rate according to soil temperature
            # (This could to a bell-shape where the maximum is obtained at 27 degree Celsius,
            # as suggested by Morré et al. (1967) for maize mucilage secretion):
            corrected_secretion_rate_max = self.secretion_rate_max * self.temperature_modification(
                                                                                    soil_temperature=soil_temperature_in_Celsius,
                                                                                    T_ref=self.secretion_rate_max_T_ref,
                                                                                    A=self.secretion_rate_max_A,
                                                                                    B=self.secretion_rate_max_B,
                                                                                    C=self.secretion_rate_max_C
                # We also regulate the rate according to the potential accumulation of mucilage around the root:
                # the rate is maximal when no mucilage accumulates around, and linearily decreases with the concentration
                # of mucilage at the soil-root interface, until reaching 0 when the concentration is equal or higher than
                # the maximal concentration soil (NOTE: Cs_mucilage_soil is expressed in mol of equivalent hexose per m2 of
                # external surface):
            ) / (
                                                   (1 + (distance_from_tip - length / 2.) / radius) ** self.gamma_secretion
                                           ) * (
                                                   self.Cs_mucilage_soil_max - Cs_mucilage_soil) / self.Cs_mucilage_soil_max
            # TODO: Validate this linear decrease until reaching the max surfacic density.

            return max(corrected_secretion_rate_max * root_exchange_surface * C_hexose_root / (
                    self.Km_secretion + C_hexose_root), 0.)

    # Release of root cells:
    # ----------------------
    # This function computes the rate of release of epidermal or cap root cells (in mol of equivalent hexose per second)
    # into the soil for a given root element. The rate of release linearily decreases with increasing length from the tip,
    # until reaching 0 at the end of the elongation zone. The external concentration of root cells (mol of equivalent-
    # hexose per m2) is also linearily decreasing the release of cells at the interface.

    def process_cells_release(self, length, root_exchange_surface, C_hexose_root, type, distance_from_tip, radius, Cs_cells_soil, soil_temperature_in_Celsius):
        # First, we ensure that the element has a positive length and surface of exchange:
        if length <= 0 or root_exchange_surface <= 0. or C_hexose_root <= 0. or type == "Just_dead" or type == "Dead" or type == "Stopped":
            return 0.
        else:
            # We modify the maximal surfacic release rate according to the mean distance to the tip (in the middle of the
            # root element), assuming that the release decreases linearily with the distance to the tip, until reaching 0
            # when the this distance becomes higher than the growing zone length:
            if distance_from_tip < self.growing_zone_factor * radius:
                average_distance = distance_from_tip - length / 2.
                reduction = (self.growing_zone_factor * radius - average_distance) \
                            / (self.growing_zone_factor * radius)
                cells_surfacic_release = self.surfacic_cells_release_rate * reduction
            # In the special case where the end of the growing zone is located somewhere on the root element:
            elif distance_from_tip - length < self.growing_zone_factor * radius:
                average_distance = (distance_from_tip - length) + (self.growing_zone_factor * radius
                                                                   - (distance_from_tip - length)) / 2.
                reduction = (self.growing_zone_factor * radius - average_distance) \
                            / (self.growing_zone_factor * radius)
                cells_surfacic_release = self.surfacic_cells_release_rate * reduction
            # Otherwise, there is no cells release:
            else:
                return 0.

            # We correct the release rate according to soil temperature:
            corrected_cells_surfacic_release = cells_surfacic_release * self.temperature_modification(
                                                                    soil_temperature=soil_temperature_in_Celsius,
                                                                    T_ref=self.surfacic_cells_release_rate_T_ref,
                                                                    A=self.surfacic_cells_release_rate_A,
                                                                    B=self.surfacic_cells_release_rate_B,
                                                                    C=self.surfacic_cells_release_rate_C)

            # We also regulate the surface release rate according to the potential accumulation of cells around the root:
            # the rate is maximal when no cells are around, and linearily decreases with the concentration of cells
            # in the soil, until reaching 0 when the concentration is equal or higher than the maximal concentration in the
            # soil (NOTE: Cs_cells_soil is expressed in mol of equivalent hexose per m2 of external surface):
            corrected_cells_surfacic_release = corrected_cells_surfacic_release * (
                    self.Cs_cells_soil_max - Cs_cells_soil) / self.Cs_cells_soil_max
            # TODO: Validate this linear decrease until reaching the max surfacic density.

            # The release of cells by the root is then calculated according to this surface:
            # TODO: Are we sure that cells release is not dependent on C availability?
            return max(root_exchange_surface * corrected_cells_surfacic_release, 0.)

    # These methods calculate the time derivative (dQ/dt) of the amount in each pool, for a given root element, based on
    # a C balance.
    # TODO account for struct mass evolution in the update. When is it updated?
    # TODO FOR TRISTAN: Consider adding N balance here (to be possibly used in the solver).
    def update_sucrose_root(self, C_sucrose_root, struct_mass, hexose_diffusion_from_phloem,
                            hexose_active_production_from_phloem, phloem_hexose_exudation, sucrose_loading_in_phloem,
                            phloem_hexose_uptake_from_soil, deficit_sucrose_root):
        return C_sucrose_root + (self.time_steps_in_seconds / struct_mass) * (
                - hexose_diffusion_from_phloem / 2.
                - hexose_active_production_from_phloem / 2.
                - phloem_hexose_exudation / 2.
                + sucrose_loading_in_phloem
                + phloem_hexose_uptake_from_soil / 2.
                - deficit_sucrose_root)

    def update_hexose_reserve(self, C_hexose_reserve, struct_mass, hexose_immobilization_as_reserve,
                              hexose_mobilization_from_reserve, deficit_hexose_reserve):
        return C_hexose_reserve + (self.time_steps_in_seconds / struct_mass) * (
                hexose_immobilization_as_reserve
                - hexose_mobilization_from_reserve
                - deficit_hexose_reserve)

    def update_hexose_root(self, C_hexose_root, struct_mass, hexose_exudation, hexose_uptake_from_soil,
                           mucilage_secretion, cells_release, maintenance_respiration,
                           hexose_consumption_by_growth, hexose_diffusion_from_phloem,
                           hexose_active_production_from_phloem, sucrose_loading_in_phloem,
                           hexose_mobilization_from_reserve, hexose_immobilization_as_reserve, deficit_hexose_root):
        return C_hexose_root + (self.time_steps_in_seconds / struct_mass) * (
                - hexose_exudation
                + hexose_uptake_from_soil
                - mucilage_secretion
                - cells_release
                - maintenance_respiration / 6.
                - hexose_consumption_by_growth
                + hexose_diffusion_from_phloem
                + hexose_active_production_from_phloem
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
    def compute_deficit_sucrose_root(self, C_sucrose_root, struct_mass, living_root_hairs_struct_mass):
        if C_sucrose_root < 0:
            return - C_sucrose_root * (struct_mass + living_root_hairs_struct_mass) / self.time_steps_in_seconds
        else:
            # TODO : or None could be more efficient?
            return 0.

    def compute_deficit_hexose_reserve(self, C_hexose_reserve, struct_mass, living_root_hairs_struct_mass):
        if C_hexose_reserve < 0:
            return - C_hexose_reserve * (struct_mass + living_root_hairs_struct_mass) / self.time_steps_in_seconds
        else:
            return 0.

    def compute_deficit_hexose_root(self, C_hexose_root, struct_mass, living_root_hairs_struct_mass):
        if C_hexose_root < 0:
            return - C_hexose_root * (struct_mass + living_root_hairs_struct_mass) / self.time_steps_in_seconds
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
                if self.n.actual_time_since_emergence > self.time_step_in_days * (24. * 60. * 60.):
                    print("!!! For element", self.n.index(),
                          "the initial mass before updating concentrations in the solver was 0 or NA!")
            else:
                # Otherwise, new concentrations are calculated considering the new amounts in each pool and the initial mass of the element:
                # TODO find alternative as hexose_root content etc doesn't exist anymore
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
                if self.n.actual_time_since_emergence > self.time_step_in_days * (24. * 60. * 60.):
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
