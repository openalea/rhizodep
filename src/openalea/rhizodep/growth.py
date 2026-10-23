"""
rhizodep's root growth as a metafspm StructuralComponent: the port of the legacy RootGrowthModel
(openalea.rhizodep.legacy, frozen), with the same results (devplan_metafspm.md).

    table = pd.DataFrame([dict(plant="p0", model=MyModel, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": {"D_ini": 7e-4, "n_adventitious_roots": 50}})])

with MyModel.initiators = (RootGrowth,): each plant's seedling is built by RootGrowth.initiate_plant, its parameters
are the class defaults overridden by the plant's scenario.

The plant's structure, in the MPG (devplan §10, Q-R4):

    Plant
     ├─ Axis (RootAxis)  the primary root        GrowthUnit ─ Phytomer ─ Organ (RootInternode), its `<` chain of
     │                                           SubOrgans: the base segment, the zero-length supports, the apex
     ├─ Axis             each seminal root       its Organ's first SubOrgan, its apex, branched (+) on its support
     └─ Axis             each nodal root         the same, on its support

Every element is a SubOrgan (a RootSegment, or a RootApex for an axis' tip), carrying the variables below. The
zero-length supports are the legacy's (several axes branched at one point); they stay until the port is exact.

Numbering: `vertex_index` is the legacy's vertex id (the MTG's ids in creation order), so that the draws seeded from
it and the comparison with the legacy match; `parent_id` and `axis_apex_id` are in that numbering (the collar's
children, the elements borne by a support, have the base, 1, as parent_id).

Codes: `type` (TYPE_*), `lateral_root_emergence_possibility` (EMERGENCE_*), `axis_type` (AXIS_*), integers.
"""
import os
from dataclasses import dataclass, fields
from math import pi

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
from math import floor, sqrt

from openalea.metafspm.solve.decorator import actual, postsegmentation, potential, segmentation

SubOrgan = ScalesConfig.SubOrgan
NII, NIE, DESCRIPTOR, MASSIC = "NonInertialIntensive", "NonInertialExtensive", "descriptor", "massic_concentration"
BY = "model_growth"

# The legacy's element types (its type_* parameters)
TYPE_BASE = 1
TYPE_SEMINAL_SUPPORT = 2
TYPE_SEMINAL_BEFORE_EMERGENCE = 3
TYPE_ADVENTITIOUS_SUPPORT = 4
TYPE_ADVENTITIOUS_BEFORE_EMERGENCE = 5
TYPE_BEFORE_EMERGENCE = 6
TYPE_EMERGED = 7
TYPE_STOPPED = 8
TYPE_JUST_STOPPED = 9
TYPE_DEAD = 10
TYPE_JUST_DEAD = 11
TYPE_NODULE = 12

# lateral_root_emergence_possibility (the legacy's 'Impossible' / 'Possible')
EMERGENCE_IMPOSSIBLE, EMERGENCE_POSSIBLE = 0, 1
# axis_type (the legacy's 'seminal' / 'nodal' / 'lateral')
AXIS_SEMINAL, AXIS_NODAL, AXIS_LATERAL = 0, 1, 2


def doc(unit, description, unit_comment="", min_value="", max_value="", value_comment="", references="", DOI=""):
    """The documentation entries of a declaration."""
    return dict(unit=unit, unit_comment=unit_comment, description=description, min_value=min_value,
                max_value=max_value, value_comment=value_comment, references=references, DOI=DOI)


def power(values, exponent):
    """values ** exponent as the legacy's Python computes it (the C library's pow): numpy's ** squares an array by
    x * x, which differs from pow in the last bit for about 0.1 % of values; float_power calls pow."""
    return np.float_power(values, exponent)


def outputs(k):
    """The return annotation of a step with k outputs besides its own: (values, name, values, ...)."""
    return tuple[(np.ndarray,) + (str, np.ndarray) * k]


@dataclass(frozen=True)
class Resource:
    """
    A resource of growth (devplan §21): the declared variables that hold it, by role. The growth component limits
    elongation and thickening by its resources, takes their costs from the elements that supply them, and gives the
    elements it creates their concentration (copied from the mother, 0 in a primordium).

    concentration            its concentration in the element (an input, e.g. from a carbon or nitrogen model)
    zone                     its concentration in an apex's supply zone
    required                 its amount in an apex's supply zone
    available_for_thickening its amount available to an element's thickening
    demand                   its demand of an element's potential growth
    consumption_amount       its amount consumed by growth in the step, on the supplying elements
    consumption              its rate of consumption by growth, on the supplying elements
    minimum_for_elongation   the parameter of its concentration kept in the supply zone (None: none)
    minimum_for_thickening   the parameter of its concentration kept in the element (None: none)
    copied_by_new_elements   whether an element created by segmentation copies its mother's required amount,
                             demand and consumptions (the legacy's hexose), or starts them at 0
    costs_by_available       whether its elongation cost is shared between the suppliers by its amount available
                             (net of the minimum) rather than its amount required
    """
    concentration: str
    zone: str
    required: str
    available_for_thickening: str
    demand: str
    consumption_amount: str
    consumption: str
    minimum_for_elongation: str = None
    minimum_for_thickening: str = None
    copied_by_new_elements: bool = True
    costs_by_available: bool = False


HEXOSE = Resource(concentration="C_hexose_root", zone="growing_zone_C_hexose_root",
                  required="hexose_possibly_required_for_elongation",
                  available_for_thickening="hexose_available_for_thickening", demand="hexose_growth_demand",
                  consumption_amount="hexose_consumption_by_growth_amount", consumption="hexose_consumption_by_growth",
                  minimum_for_elongation="C_hexose_min_for_elongation",
                  minimum_for_thickening="C_hexose_min_for_thickening")


EVENTS = ("primordium_formation", "emergence", "cells_formation", "growth_stopped", "death")


class Clocks:
    """
    The elements' actual and thermal times since events (actual_time_since_<event>, thermal_time_since_<event>),
    copied from a step's arguments and advanced together: the actual time by the time step, the thermal one by the
    time step times the temperature factor.
    """

    def __init__(self, arguments: dict, events=EVENTS):
        self.actual = {event: np.array(arguments[f"actual_time_since_{event}"], dtype=float) for event in events}
        self.thermal = {event: np.array(arguments[f"thermal_time_since_{event}"], dtype=float) for event in events}

    def advance(self, selected, time_step, thermal_time_step, *events):
        for event in events:
            self.actual[event][selected] += time_step
            self.thermal[event][selected] += thermal_time_step[selected]

    def outputs(self) -> tuple:
        """("actual_time_since_<event>", values, "thermal_time_since_<event>", values, ...), for a step's return."""
        return tuple(item for event in self.actual for item in (f"actual_time_since_{event}", self.actual[event],
                                                                f"thermal_time_since_{event}", self.thermal[event]))


def temperature_modification(soil_temperature, process_at_T_ref, T_ref, A, B, C):
    """
    The factor of a process at *soil_temperature* (°C), the legacy's Model.temperature_modification: the process at
    T_ref times (A (T - T_ref) + B)^(1 - C) (A (T - T_ref) + B)^(C (T - T_ref) / 10); C = 0 and B = 1: a linear
    increase (thermal time); C = 1, A = 0, B > 1: a Q10; C = 1, A < 0, B > 0: a bell shape (Parent et al., 2010).
    0 when C is neither 0 nor 1, or when C = 1 and A (T - T_ref) + B < 0.
    """
    x = A * (soil_temperature - T_ref) + B
    modified = process_at_T_ref * power(x, 1 - C) * power(x, C * (soil_temperature - T_ref) / 10.)
    return np.where(((C != 0) & (C != 1)) | ((C == 1) & (x < 0.)), 0., np.maximum(modified, 0.))


@dataclass
class RootGrowth(StructuralComponent):
    """
    Root growth of rhizodep (Rees et al.; Gérault et al.): potential growth of the apices and segments from their
    hexose, actual growth limited by it, segmentation and lateral primordia, root hairs. The port of the legacy
    RootGrowthModel; its steps follow (devplan §8).
    """
    # --- Inputs ---
    soil_temperature: float = input_variable(**doc('°C', 'soil temperature in contact with roots'),
                                             by='model_soil', initialize=15, scale=SubOrgan)
    root_tissue_density: float = input_variable(**doc('g.m-3', 'root_tissue_density',
                                                      unit_comment='of structural mass'),
                                                by='model_anatomy', initialize=100000.0, scale=SubOrgan)
    C_hexose_root: float = input_variable(**doc('mol.g-1',
                                                'Hexose concentration of the element (given by a carbon model)',
                                                unit_comment='of hexose per gram of structural mass'),
                                          by='model_carbon', initialize=0., scale=SubOrgan, state_variable_type=MASSIC)

    # --- State variables, declared by the legacy ---
    type: int = state_variable(**doc('', 'Example segment type provided by root growth model'),
                               state_variable_type=NII, initialize=1, scale=SubOrgan, dtype='int')
    root_order: int = state_variable(**doc('',
                                           "Example root segment's axis order computed by the initiate_mtg method or "
                                           'provided as input'),
                                     state_variable_type=DESCRIPTOR, initialize=1, scale=SubOrgan, dtype='int')
    vertex_index: int = state_variable(**doc('dimensionless',
                                             'Unique vertex identifier stored for ease of value access'),
                                       state_variable_type=NII, initialize=1, scale=SubOrgan, dtype='int')
    parent_id: int = state_variable(**doc('dimensionless',
                                          "Unique vertex identifier of current element's parent stored for ease of "
                                          'value access',
                                          value_comment='-1 as default when there is no parent assigned'),
                                    state_variable_type=NII, initialize=-1, scale=SubOrgan, dtype='int')
    axis_apex_id: int = state_variable(**doc('dimensionless',
                                             'Vertex identifier of the terminal apex of the root axis bearing the '
                                             'considered segment',
                                             value_comment='-1 as default when not assigned'),
                                       state_variable_type=NII, initialize=-1, scale=SubOrgan, dtype='int')
    lateral_primordium_index: int = state_variable(**doc('dimensionless',
                                                         'Unique axis identifier stored for ease of value access'),
                                                   state_variable_type=NII, initialize=-1, scale=SubOrgan, dtype='int')
    step_elongating: int = state_variable(**doc('adim', 'Whether the element could elongate in the last actual growth '
                                                        '(the legacy\'s step_elongating_elements)'),
                                          state_variable_type=DESCRIPTOR, initialize=0, scale=SubOrgan, dtype='int')
    axis_type: int = state_variable(**doc('', 'Example axis type provided by root growth model'),
                                    state_variable_type=DESCRIPTOR, initialize=0, scale=SubOrgan, dtype='int')
    radius: float = state_variable(**doc('m', 'Example root segment radius'),
                                   state_variable_type=NII, initialize=0.00035, scale=SubOrgan)
    z1: float = state_variable(**doc('m', 'Depth of the segment tip computed by plantGL, colar side'),
                               state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    z2: float = state_variable(**doc('m', 'Depth of the segment tip computed by plantGL, apex side'),
                               state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    length: float = state_variable(**doc('m', 'Example root segment length'),
                                   state_variable_type=NIE, initialize=0.003, scale=SubOrgan)
    initial_length: float = state_variable(**doc('m', 'Example root segment length'),
                                           state_variable_type=NIE, initialize=0.003, scale=SubOrgan)
    struct_mass: float = state_variable(**doc('g', 'Example root segment structural mass'),
                                        state_variable_type=NIE, initialize=0.000135, scale=SubOrgan)
    initial_struct_mass: float = state_variable(**doc('g',
                                                      'Same as struct_mass but corresponds to the previous time step;'
                                                      ' it is intended to record the variation'),
                                                state_variable_type=NIE, initialize=0.000135, scale=SubOrgan)
    initial_living_root_hairs_struct_mass: float = state_variable(**doc('g',
                                                                        'Same as struct_mass but corresponds to the '
                                                                        'previous time step; it is intended to record'
                                                                        ' the variation'),
                                                                  state_variable_type=NIE, initialize=0.0,
                                                                  scale=SubOrgan)
    living_root_hairs_struct_mass: float = state_variable(**doc('g',
                                                                'Example root segment living root hairs structural '
                                                                'mass'),
                                                          state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    living_struct_mass: float = state_variable(**doc('g', 'Sum of segment and root hair living struct mass'),
                                               state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    root_hair_length: float = state_variable(**doc('m', 'Example root hair length',
                                                   references='According to the work of Gahoonia et al. (1997), the '
                                                              'root hair maximal length for wheat and barley evolves '
                                                              'between 0.5 and 1.3 mm.'),
                                             state_variable_type=NIE, initialize=0.001, scale=SubOrgan)
    total_root_hairs_number: float = state_variable(**doc('adim',
                                                          'Example root hairs number on segment external surface',
                                                          value_comment='30 * (1.6e-4 / radius) * length * 1e3',
                                                          references=' According to the work of Gahoonia et al. '
                                                                     '(1997), the root hair density is about 30 hairs'
                                                                     ' per mm for winter wheat, for a root radius of '
                                                                     'about 0.16 mm.'),
                                                    state_variable_type=NIE, initialize=41.142857142857146,
                                                    scale=SubOrgan)
    growing_zone_C_hexose_root: float = state_variable(**doc('mol.g-1', 'Hexose availability in the growing zone'),
                                                       state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_growth: float = state_variable(**doc('mol.s-1',
                                                               'Hexose consumption rate by growth is coupled to a '
                                                               'root growth model'),
                                                         state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_growth_amount: float = state_variable(**doc('mol.s-1',
                                                                      'Hexose consumption rate by growth is coupled '
                                                                      'to a root growth model'),
                                                                state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    resp_growth: float = state_variable(**doc('mol.s-1', 'respiration rate during growth', unit_comment='of C'),
                                        state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_fungus: float = state_variable(**doc('mol.s-1', 'Hexose consumption rate by fungus'),
                                                         state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    distance_from_tip: float = state_variable(**doc('m', 'Example distance from tip'),
                                              state_variable_type=NII, initialize=0.003, scale=SubOrgan)
    volume: float = state_variable(**doc('m3', 'Initial volume of the collar element'),
                                   state_variable_type=NIE, initialize=1e-09, scale=SubOrgan)
    struct_mass_produced: float = state_variable(**doc('g', '', unit_comment='of dry weight'),
                                                 state_variable_type=NIE, initialize=0, scale=SubOrgan)
    root_hairs_struct_mass_produced: float = state_variable(**doc('g', '', unit_comment='of dry weight'),
                                                            state_variable_type=NIE, initialize=0, scale=SubOrgan)
    thermal_time_since_emergence: float = state_variable(**doc('s', ''),
                                                         state_variable_type=NII, initialize=0, scale=SubOrgan)
    thermal_time_since_cells_formation: float = state_variable(**doc('s', 'Thermal time since tissue formation'),
                                                               state_variable_type=NII, initialize=0, scale=SubOrgan)
    actual_time_since_formation: float = state_variable(**doc('s', ''),
                                                        state_variable_type=NII, initialize=0, scale=SubOrgan)
    tissue_formation_time: float = state_variable(**doc('day', ''),
                                                  state_variable_type=NII, initialize=0, scale=SubOrgan)

    # --- State variables the legacy writes without declaring them ---
    actual_elongation: float = state_variable(**doc('m', 'Length gained by the element during the time step'),
                                              state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    actual_elongation_rate: float = state_variable(**doc('m.s-1',
                                                         'Elongation rate of the element during the time step'),
                                                   state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    actual_length_with_hairs: float = state_variable(**doc('m', 'Length of the element bearing root hairs'),
                                                     state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    actual_potential_time_since_emergence: float = state_variable(**doc('s',
                                                                        'Time since the potential emergence of the '
                                                                        'primordium'),
                                                                  state_variable_type=NII, initialize=0.0,
                                                                  scale=SubOrgan)
    actual_time_since_cells_formation: float = state_variable(**doc('s',
                                                                    "Time since the formation of the element's cells"),
                                                              state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    actual_time_since_death: float = state_variable(**doc('s', 'Time since the death of the element'),
                                                    state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    actual_time_since_emergence: float = state_variable(**doc('s', "Time since the emergence of the element's axis"),
                                                        state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    actual_time_since_growth_stopped: float = state_variable(**doc('s',
                                                                   "Time since the growth of the element's axis "
                                                                   'stopped'),
                                                             state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    actual_time_since_primordium_formation: float = state_variable(**doc('s',
                                                                         'Time since the formation of the primordium'),
                                                                   state_variable_type=NII, initialize=0.0,
                                                                   scale=SubOrgan)
    actual_time_since_root_hairs_emergence_started: float = state_variable(**doc('s',
                                                                                 'Time since root hairs started to '
                                                                                 'emerge on the element'),
                                                                           state_variable_type=NII, initialize=0.0,
                                                                           scale=SubOrgan)
    actual_time_since_root_hairs_emergence_stopped: float = state_variable(**doc('s',
                                                                                 'Time since root hairs stopped '
                                                                                 'emerging on the element'),
                                                                           state_variable_type=NII, initialize=0.0,
                                                                           scale=SubOrgan)
    all_root_hairs_formed: float = state_variable(**doc('adim',
                                                        '1 when every root hair of the element has formed, else 0'),
                                                  state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    angle_down: float = state_variable(**doc('°',
                                             "Insertion angle of the element relative to its parent's direction "
                                             '(pitch)'),
                                       state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    angle_roll: float = state_variable(**doc('°', "Roll of the element around its parent's direction"),
                                       state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    dead_root_hairs_number: float = state_variable(**doc('adim', 'Number of dead root hairs on the element'),
                                                   state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    dist_to_ramif: float = state_variable(**doc('m', 'Length elongated by the apex since its last lateral primordium'),
                                          state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    emergence_cost: float = state_variable(**doc('mol of hexose', "Hexose needed for a primordium's emergence"),
                                           state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    emergence_delay_in_thermal_time: float = state_variable(**doc('s',
                                                                  'Thermal time before a seminal or nodal primordium '
                                                                  'emerges'),
                                                            state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    former_distance_from_tip: float = state_variable(**doc('m', 'Distance from the tip before the last growth'),
                                                     state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    growth_duration: float = state_variable(**doc('s', 'Thermal time during which the apex can grow'),
                                            state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    hexose_available_for_thickening: float = state_variable(**doc('mol of hexose',
                                                                  "Hexose available for the element's radial growth"),
                                                            state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_growth_demand: float = state_variable(**doc('mol of hexose',
                                                       "Hexose needed for the element's potential growth"),
                                                 state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_growth_demand_rate: float = state_variable(**doc('mol of hexose.s-1',
                                                            "Rate of hexose needed for the element's potential growth"),
                                                      state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_possibly_required_for_elongation: float = state_variable(**doc('mol of hexose',
                                                                          'Hexose of the supply zone available to the'
                                                                          " apex's elongation"),
                                                                    state_variable_type=NIE, initialize=0.0,
                                                                    scale=SubOrgan)
    initial_radius: float = state_variable(**doc('m', 'Radius at the start of the time step'),
                                           state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    lateral_root_emergence_possibility: int = state_variable(**doc('adim',
                                                                   'Whether a lateral primordium may emerge on the '
                                                                   'element (EMERGENCE_* codes)'),
                                                             state_variable_type=DESCRIPTOR, initialize=0,
                                                             scale=SubOrgan, dtype='int')
    life_duration: float = state_variable(**doc('s', 'Thermal time the element lives after its axis stopped growing'),
                                          state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    living_root_hairs_number: float = state_variable(**doc('adim', 'Number of living root hairs on the element'),
                                                     state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    original_radius: float = state_variable(**doc('m', 'Radius of the element at its formation'),
                                            state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    potential_length: float = state_variable(**doc('m', 'Length of the element after its potential growth'),
                                             state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    potential_radius: float = state_variable(**doc('m', 'Radius of the element after its potential growth'),
                                             state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    resp_growth_rate: float = state_variable(**doc('mol of C.s-1', 'Growth respiration rate'),
                                             state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    resp_maintenance: float = state_variable(**doc('mol of C', 'Maintenance respiration'),
                                             state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    root_hairs_struct_mass: float = state_variable(**doc('g', "Structural mass of the element's root hairs"),
                                                   state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    root_hairs_volume: float = state_variable(**doc('m3', "Volume of the element's root hairs"),
                                              state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    struct_mass_contributing_to_elongation: float = state_variable(**doc('g',
                                                                         'Structural mass of the supply zone '
                                                                         "contributing to the apex's elongation"),
                                                                   state_variable_type=NIE, initialize=0.0,
                                                                   scale=SubOrgan)
    temperature_modification: float = state_variable(**doc('adim', "Temperature factor of the element's thermal times"),
                                                     state_variable_type=NII, initialize=1.0, scale=SubOrgan)
    theoretical_radius: float = state_variable(**doc('m', "Radius the element would have from its axis' first radius"),
                                               state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    thermal_potential_time_since_emergence: float = state_variable(**doc('s',
                                                                         'Thermal time since the potential emergence '
                                                                         'of the primordium'),
                                                                   state_variable_type=NII, initialize=0.0,
                                                                   scale=SubOrgan)
    thermal_time_since_death: float = state_variable(**doc('s', 'Thermal time since the death of the element'),
                                                     state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    thermal_time_since_growth_stopped: float = state_variable(**doc('s',
                                                                    "Thermal time since the growth of the element's "
                                                                    'axis stopped'),
                                                              state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    thermal_time_since_primordium_formation: float = state_variable(**doc('s',
                                                                          'Thermal time since the formation of the '
                                                                          'primordium'),
                                                                    state_variable_type=NII, initialize=0.0,
                                                                    scale=SubOrgan)
    thermal_time_since_root_hairs_emergence_started: float = state_variable(**doc('s',
                                                                                  'Thermal time since root hairs '
                                                                                  'started to emerge'),
                                                                            state_variable_type=NII, initialize=0.0,
                                                                            scale=SubOrgan)
    thermal_time_since_root_hairs_emergence_stopped: float = state_variable(**doc('s',
                                                                                  'Thermal time since root hairs '
                                                                                  'stopped emerging'),
                                                                            state_variable_type=NII, initialize=0.0,
                                                                            scale=SubOrgan)

    # --- Plant-scale state ---
    total_living_struct_mass: float = state_variable(**doc('g', 'Living structural mass of the root system',
                                                           unit_comment='of dry weight'),
                                                     state_variable_type='extensive', initialize=0.,
                                                     scale=ScalesConfig.Plant)

    # --- Parameters ---
    D_ini: float = parameter(**doc('m', 'Initial tip diameter of the primary root'), by=BY, default=0.0008)
    root_hair_radius: float = parameter(**doc('m', 'Average radius of root hair',
                                              references='According to the work of Gahoonia et al. (1997), the root '
                                                         'hair diameter is relatively constant for different '
                                                         'genotypes of wheat and barley, i.e. 12 microns'),
                                        by=BY, default=6e-06)
    root_hairs_lifespan: float = parameter(**doc('s', 'Average lifespan of a root hair',
                                                 unit_comment='time equivalent at temperature of T_ref',
                                                 references='According to the data from McElgunn and Harrison (1969),'
                                                            ' the lifespan of wheat root hairs is 40-55h, depending '
                                                            'on the temperature. For a temperature of 20 degree '
                                                            'Celsius, the linear regression from this data gives 46h.'),
                                           by=BY, default=165600.0)
    root_hairs_density: float = parameter(**doc('.m-2', 'Average density of root hairs',
                                                unit_comment='number of hairs par meter of root per meter of root '
                                                             'radius',
                                                references='According to the data from McElgunn and Harrison (1969), '
                                                           'the elongation rate for wheat root hairs is about 0.080 '
                                                           'mm h-1.'),
                                          by=BY, default=374999999.99999994)
    root_hair_max_length: float = parameter(**doc('m', 'Average maximal length of a root hair',
                                                  references='According to the work of Gahoonia et al. (1997), the '
                                                             'root hair maximal length for wheat and barley evolves '
                                                             'between 0.5 and 1.3 mm.'),
                                            by=BY, default=0.001)
    root_hairs_elongation_rate: float = parameter(**doc('.s-1', 'Average elongation rate of root hairs',
                                                        unit_comment='in meter per second per meter of root radius',
                                                        references='According to the data from McElgunn and Harrison '
                                                                   '(1969), the elongation rate for wheat root hairs '
                                                                   'is about 0.080 mm h-1.'),
                                                  by=BY, default=0.003703703703703704)
    LDs: float = parameter(**doc('s.m-1..g-1.m-3', 'Average lifespan of a root hair',
                                 unit_comment='time equivalent at temperature of T_ref',
                                 references='5000 day mm-1 g-1 cm3 (??)'),
                           by=BY, default=345600.0)
    ER: float = parameter(**doc('.s-1', 'Emission rate of adventitious roots',
                                unit_comment='time equivalent at temperature of T_ref'),
                          by=BY, default=2.3148148148148148e-06)
    n_seminal_roots: int = parameter(**doc('adim',
                                           'Maximal number of roots emerging from the base (including primary and '
                                           'seminal roots)'),
                                     by=BY, default=5)
    n_adventitious_roots: int = parameter(**doc('adim',
                                                'Maximal number of roots emerging from the base (including primary '
                                                'and seminal roots)'),
                                          by=BY, default=10)
    random_choice: int = parameter(**doc('adim',
                                         'We set the random seed, so that the same simulation can be repeted with the'
                                         ' same seed:'),
                                   by=BY, default=8)
    D_sem_to_D_ini_ratio: float = parameter(**doc('adim',
                                                  'Proportionality coefficient between the tip diameter of a seminal '
                                                  'root and D_ini'),
                                            by=BY, default=0.95)
    CVDD: float = parameter(**doc('adim', 'Relative variation of the daughter root diameter'), by=BY, default=0.2)
    starting_time_for_adventitious_roots_emergence: float = parameter(**doc('s',
                                                                            'Time when adventitious roots start to '
                                                                            'successively emerge',
                                                                            unit_comment='time equivalent at '
                                                                                         'temperature of T_ref'),
                                                                      by=BY, default=777600.0)
    D_adv_to_D_ini_ratio: float = parameter(**doc('adim',
                                                  'Proportionality coefficient between the tip diameter of an '
                                                  'adventitious root and D_ini '),
                                            by=BY, default=0.8)
    process_at_T_ref: float = parameter(**doc('adim', 'Proportion of maximal process intensity occuring at T_ref'),
                                        by=BY, default=1.0)
    T_ref: float = parameter(**doc('°C', 'the reference temperature',
                                   references='We assume that relative growth is 0 at T_ref=0 degree Celsius, and '
                                              'linearily increases to reach 1 at 20 degree.'),
                             by=BY, default=0.0)
    A: float = parameter(**doc('adim', 'parameter A (may be equivalent to the coefficient of linear increase)'),
                         by=BY, default=0.05)
    B: int = parameter(**doc('adim', 'parameter B (may be equivalent to the Q10 value)'), by=BY, default=0)
    C: int = parameter(**doc('adim', 'parameter C (either 0 or 1)'), by=BY, default=0)
    growing_zone_factor: float = parameter(**doc('adim',
                                                 'Proportionality factor between the radius and the length of the '
                                                 'root apical zone in which C can sustain root elongation',
                                                 references='According to illustrations by Kozlova et al. (2020), the'
                                                            ' length of the growing zone corresponding to the root '
                                                            'cap, meristem and elongation zones is about 8 times the '
                                                            'diameter of the tip.'),
                                           by=BY, default=21.0)
    emergence_delay: float = parameter(**doc('s', 'Delay of emergence of the primordium',
                                             unit_comment='time equivalent at temperature of T_ref',
                                             references='emergence_delay = 3 days (??)'),
                                       by=BY, default=282528.0)
    EL: float = parameter(**doc('s-1', 'Slope of the elongation rate = f(tip diameter) ',
                                unit_comment='meters of root per meter of radius per second equivalent to '
                                             'T_ref_growth',
                                references='EL = 5 mm mm-1 day-1 (??)'),
                          by=BY, default=0.00065)
    Km_elongation: float = parameter(**doc('mol.g-1', 'Affinity constant for root elongation',
                                           unit_comment='of hexose',
                                           references='According to Barillot et al. (2016b): Km for root growth is '
                                                      '1250 umol C g-1 for sucrose. According to Gauthier et al '
                                                      '(2020): Km for regulation of the RER by sucrose concentration '
                                                      'in hz = 100-150 umol C g-1'),
                                     by=BY, default=0.00020833333333333335)
    relative_nodule_thickening_rate_max: float = parameter(**doc('s-1',
                                                                 'Maximal rate of relative increase in nodule radius',
                                                                 references="We consider that the radius can't "
                                                                            'increase by more than 20% every day (??)'),
                                                           by=BY, default=2.3148148148148148e-06)
    Km_nodule_thickening: float = parameter(**doc('mol.g-1', 'Affinity constant for nodule thickening',
                                                  unit_comment='of hexose',
                                                  value_comment='Km_elongation * 100'),
                                            by=BY, default=0.020833333333333336)
    nodule_max_radius: float = parameter(**doc('m', 'Maximal radius of nodule', value_comment='Dini * 10'),
                                         by=BY, default=0.016)
    SGC: float = parameter(**doc('adim',
                                 'Proportionality coefficient between the section area of the segment and the sum of '
                                 'distal section areas'),
                           by=BY, default=0.0)
    relative_root_thickening_rate_max: float = parameter(**doc('s-1',
                                                               'Maximal rate of relative increase in root radius',
                                                               references="We consider that the radius can't increase"
                                                                          ' by more than 5% every day (??)'),
                                                         by=BY, default=5.787037037037037e-07)
    C_hexose_min_for_elongation: float = parameter(**doc('mol.g-1', 'Treshold hexose concentration for elongation',
                                                         unit_comment='in mol of hexose per g of structural mass',
                                                         references='?'),
                                                   by=BY, default=1e-05)
    C_hexose_min_for_thickening: float = parameter(**doc('mol.g-1', 'Treshold hexose concentration for thikening',
                                                         unit_comment='in mol of hexose per g of structural mass',
                                                         references='?'),
                                                   by=BY, default=1e-05)
    Km_thickening: float = parameter(**doc('mol.g-1', 'Affinity constant for root thickening',
                                           unit_comment='of hexose',
                                           value_comment='Km_elongation',
                                           references='We assume that the Michaelis-Menten constant for thickening is'
                                                      ' the same as for root elongation. (??)'),
                                     by=BY, default=0.00020833333333333335)
    struct_mass_C_content: float = parameter(**doc('mol.g-1', 'C content of structural mass',
                                                   unit_comment='of carbon',
                                                   references='We assume that the structural mass contains 44% of C. '
                                                              '(??)'),
                                             by=BY, default=0.03663613655287261)
    yield_growth: float = parameter(**doc('adim', 'Growth yield',
                                          unit_comment='mol of CO2 per mol of C used for structural mass',
                                          references='We use the range value (0.75-0.85) proposed by Thornley and '
                                                     'Cannell (2000)'),
                                    by=BY, default=0.8)
    segment_length: float = parameter(**doc('m', 'Length of a segment'), by=BY, default=0.003)
    nodule_formation_probability: float = parameter(**doc('m',
                                                          'Probability (between 0 and 1) of nodule formation for each'
                                                          ' apex that elongates'),
                                                    by=BY, default=0.5)
    Dmin: float = parameter(**doc('m',
                                  'Minimal threshold tip diameter (i.e. the diameter of the finest observable roots)',
                                  references='Dmin=0.05 mm (??)'),
                            by=BY, default=0.000122)
    RMD: float = parameter(**doc('adim',
                                 'Average ratio of the diameter of the daughter root to that of the mother root'),
                           by=BY, default=0.57)
    IPD: float = parameter(**doc('m', 'Inter-primordia distance', references='IPD = 7.6 mm (??)'),
                           by=BY, default=0.00474)
    new_root_tissue_density: float = parameter(**doc('g.m-3', 'root_tissue_density',
                                                     unit_comment='of structural mass'),
                                               by=BY, default=140000.0)
    GDs: float = parameter(**doc('s.m-2', 'Coefficient of growth duration',
                                 unit_comment='time equivalent at temperature of T_ref',
                                 references='Reference: GDs=930. day mm-2 (Pagès et Picon-Cochard, 2014)'),
                           by=BY, default=69120000000000.0)
    main_roots_growth_extender: float = parameter(**doc('s.s-1',
                                                        'Coefficient of growth duration extension, by which the '
                                                        'theoretical growth duration is multiplied for seminal and '
                                                        'adventitious roots',
                                                        references='Reference: GDs=400. day mm-2 ()'),
                                                  by=BY, default=100.0)
    GD_highest: float = parameter(**doc('s.m-2',
                                        'For seminal and adventitious roots, a longer growth duration is applied',
                                        unit_comment='time equivalent at temperature of T_ref',
                                        value_comment='Expected growth duration of a seminal root'),
                                  by=BY, default=5184000.0)
    GD_high: float = parameter(**doc('s.m-2',
                                     'The growth duration has a probability of [1-GD_prob_medium] to equal GD_high',
                                     unit_comment='time equivalent at temperature of T_ref',
                                     references='Estimated the longest observed lateral wheat roots observed in '
                                                'rhizoboxes (Rees et al., unpublished)'),
                               by=BY, default=518400.0)
    GD_medium: float = parameter(**doc('s.m-2',
                                       'The growth duration has a probability of [GD_prob_medium - GD_prob_low] to '
                                       'equal GD_medium',
                                       unit_comment='time equivalent at temperature of T_ref',
                                       references='Estimated from the medium lateral wheat roots observed in '
                                                  'rhizoboxes (Rees et al., unpublished)'),
                                 by=BY, default=60479.99999999999)
    GD_low: float = parameter(**doc('s.m-2', 'The growth duration has a probability of [GD_prob_low] to equal GD_low',
                                    unit_comment='time equivalent at temperature of T_ref',
                                    references='Estimated from the shortest lateral wheat roots observed in '
                                               'rhizoboxes (Rees et al., unpublished)'),
                              by=BY, default=21600.0)
    GD_by_frequency: bool = parameter(**doc('adim',
                                            'As an alternative to using a single value of growth duration depending '
                                            'on diameter, we offer the possibility to rather define the growth '
                                            'duration as a random choice between three values (low, medium and high),'
                                            ' depending on their respective probability'),
                                      by=BY, default=False)
    GD_prob_low: float = parameter(**doc('adim', 'Probability for low growth duration',
                                         references='Estimated from the shortest lateral wheat roots observed in '
                                                    'rhizoboxes (Rees et al., unpublished)'),
                                   by=BY, default=0.5)
    GD_prob_medium: float = parameter(**doc('adim', 'Coefficient of growth duration',
                                            unit_comment='Probability for medium growth duration',
                                            references='Estimated from the medium lateral wheat roots observed in '
                                                       'rhizoboxes (Rees et al., unpublished)'),
                                      by=BY, default=0.85)
    random: bool = parameter(**doc('adim', 'Allow random processes in growth'), by=BY, default=True)
    simple_growth_duration: bool = parameter(**doc('adim', 'Allow growth according to the original Archisimple model',
                                                   references='(Pagès et al., 2014)'),
                                             by=BY, default=False)
    initial_segment_length: float = parameter(**doc('m', 'Initial segment length'), by=BY, default=0.001)
    initial_apex_length: float = parameter(**doc('m', 'Initial apex length'), by=BY, default=0.0001)
    initial_C_hexose_root: float = parameter(**doc('mol.g-1', 'Initial hexose concentration of root segments'),
                                             by=BY, default=0.0001)
    input_file_path: str = parameter(**doc('m', 'Filepath for input files'), by=BY, default='inputs')
    forcing_seminal_roots_events: bool = parameter(**doc('m',
                                                         'a Boolean expliciting if seminal root events should be '
                                                         'forced'),
                                                   by=BY, default=False)
    seminal_roots_events_file: str = parameter(**doc('m',
                                                     'Filepath pointing to input table to plan seminal root emergence'
                                                     ' event'),
                                               by=BY, default='seminal_roots_inputs.csv')
    forcing_adventitious_roots_events: bool = parameter(**doc('m',
                                                              'a Boolean expliciting if adventicious root events '
                                                              'should be forced'),
                                                        by=BY, default=False)
    adventitious_roots_events_file: str = parameter(**doc('adim',
                                                          'Filepath pointing to input table to plan adventitious root'
                                                          ' emergence event'),
                                                    by=BY, default='adventitious_roots_inputs.csv')
    radial_growth: bool = parameter(**doc('adim',
                                          'equivalent to a Boolean expliciting whether radial growth should be '
                                          'considered or not'),
                                    by=BY, default=True)
    nodules: bool = parameter(**doc('adim', 'a Boolean expliciting whether nodules could be formed or not'),
                              by=BY, default=False)
    root_order_limitation: bool = parameter(**doc('adim',
                                                  'a Boolean expliciting whether lateral roots should be prevented '
                                                  'above a certain root order'),
                                            by=BY, default=False)
    root_order_treshold: int = parameter(**doc('adim', 'the root order above which new lateral roots cannot be formed'),
                                         by=BY, default=2)

    # Coupled, the other components' variables are shared between the elements once per call, as the legacy's
    # post_growth_updating does it, but conservatively (metafspm's repartition; devplan §19, g17): by the living
    # structural mass after the call, against the living mass at its start (partition_basis None). For an element
    # that only elongated it is the legacy's basis (its initial masses); for one segmentation split, the legacy's
    # initial_struct_mass is already shared out, which would lose amounts
    partition_weight = "living_struct_mass"
    repartition = "end_of_call"

    # What the legacy's new elements start with at 0 (RootGrowthModel.ADDING_A_CHILD(nil_properties=True)), whatever
    # the declared defaults
    NIL = ("potential_length", "initial_length", "volume", "dist_to_ramif", "distance_from_tip",
           "former_distance_from_tip", "actual_elongation", "actual_elongation_rate", "struct_mass",
           "initial_struct_mass", "root_hair_length", "actual_length_with_hairs", "living_root_hairs_number",
           "dead_root_hairs_number", "total_root_hairs_number", "actual_time_since_root_hairs_emergence_started",
           "thermal_time_since_root_hairs_emergence_started", "actual_time_since_root_hairs_emergence_stopped",
           "thermal_time_since_root_hairs_emergence_stopped", "all_root_hairs_formed", "root_hairs_struct_mass",
           "root_hairs_struct_mass_produced", "initial_living_root_hairs_struct_mass", "living_root_hairs_struct_mass",
           "resp_growth", "struct_mass_produced", "hexose_growth_demand", "hexose_consumption_by_growth_amount",
           "hexose_consumption_by_growth", "hexose_consumption_by_fungus", "hexose_possibly_required_for_elongation",
           "actual_time_since_primordium_formation", "actual_time_since_emergence",
           "actual_time_since_cells_formation", "actual_potential_time_since_emergence",
           "actual_time_since_growth_stopped", "actual_time_since_death", "thermal_time_since_primordium_formation",
           "thermal_time_since_emergence", "thermal_time_since_cells_formation",
           "thermal_potential_time_since_emergence", "thermal_time_since_growth_stopped", "thermal_time_since_death")

    # The legacy's pipe model compares child.edge_type, a method of the MTG's nodes, with '<' and '+': never equal, so
    # it sees no child's section and the theoretical radius falls back on the initial one (g16: no radial growth).
    # True reproduces it, until the reviewed fixes (devplan §6)
    legacy_pipe_model = True

    # The resources of growth (devplan §21); the first one is the carbon: growth respiration comes from its costs
    resources = (HEXOSE,)
    # Options that are the legacy's by default, and reviewed fixes otherwise (devplan §6): the supply zone of an apex
    # reaching the collar continues into the collar's other children by volume (Root_BRIDGES', g9), and an element
    # created as a primordium takes its mother's soil temperature ("mother") rather than 7.8 °C (g3)
    collar_supply = False
    new_element_soil_temperature = 7.8

    # The legacy's per-element copies of these parameters are the parameters themselves (never changed per element)
    ELEMENT_PARAMETERS = ("root_hair_radius", "root_hairs_lifespan")

    @classmethod
    def plant_parameters(cls, parameters) -> dict:
        """The class's defaults (parameters, and the inputs a scenario may set) overridden by *parameters*."""
        values = {}
        for f in fields(cls):
            kind = f.metadata.get("variable_type")
            if kind in ("parameter", "input") or f.name == "volume":
                values[f.name] = parameters.get(f.name, f.default)
        return values

    @classmethod
    def seedling(cls, parameters, stream=None) -> list:
        """
        The legacy's initial elements (RootGrowthModel.initiate_mtg, then its initiate_heterogeneous_variables and
        the post-growth they end with) for the plant's *parameters*: a list, in the legacy's creation order, of
        dicts with "vertex_index", "parent" (its vertex_index or None), "edge" ('<' / '+'), "apex" (a RootApex) and
        "values" ({state variable: value}). *stream*: the plant's random stream, seeded as the legacy seeds numpy.
        """
        defaults = {f.name: f.default for f in fields(cls) if f.metadata.get("variable_type") == "state_variable"
                    and f.metadata.get("scale") == SubOrgan}
        elements = []
        tissue_density = parameters["new_root_tissue_density"]
        stream = stream if stream is not None else np.random.RandomState()

        def add(parent, edge, apex, values):
            element = dict(vertex_index=len(elements) + 1, parent=parent, edge=edge, apex=apex,
                           values={**defaults, **dict.fromkeys(cls.NIL, 0.), **values})
            elements.append(element)
            return element

        def nil_child(parent, edge, apex, type_, angle_down, angle_roll, length, radius):
            return add(parent["vertex_index"], edge, apex,
                       cls._nil_values(parameters, type_, 1, angle_down, angle_roll, length, radius))

        def main_root_durations(radius):
            """The growth and life durations of a first-order root (with the legacy's extender)."""
            return dict(growth_duration=(parameters["GDs"] * (2. * radius) ** 2
                                         * parameters["main_roots_growth_extender"]),
                        life_duration=parameters["LDs"] * (2. * radius) * tissue_density)

        # The base segment
        base_radius = parameters["D_ini"] / 2.
        length = parameters["initial_segment_length"]
        initial_mass = parameters["volume"] * tissue_density
        segment = add(None, "/", False, dict(
            type=TYPE_BASE, root_order=1, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE, emergence_cost=0.,
            angle_down=0., angle_roll=0., length=length, radius=base_radius, original_radius=base_radius,
            initial_length=length, initial_radius=base_radius, distance_from_tip=length,
            former_distance_from_tip=length, dist_to_ramif=0., actual_elongation=length, actual_elongation_rate=0.,
            volume=parameters["volume"], struct_mass=initial_mass, initial_struct_mass=initial_mass,
            **main_root_durations(base_radius)))

        # The seminal, then the nodal (adventitious) roots, each on a zero-length support
        for kind in ("seminal", "adventitious"):
            number_of_roots = parameters[f"n_{kind}_roots"]
            forcing = parameters[f"forcing_{kind}_roots_events"]
            if not (number_of_roots > (1 if kind == "seminal" else 0) or not forcing):
                continue
            events = cls._emergence_events(parameters, kind)
            ratio = parameters["D_sem_to_D_ini_ratio" if kind == "seminal" else "D_adv_to_D_ini_ratio"]
            support_type, apex_type = ((TYPE_SEMINAL_SUPPORT, TYPE_SEMINAL_BEFORE_EMERGENCE) if kind == "seminal" else
                                       (TYPE_ADVENTITIOUS_SUPPORT, TYPE_ADVENTITIOUS_BEFORE_EMERGENCE))
            for i, (delay, roots_per_event) in enumerate(events):
                for j in range(int(roots_per_event)):
                    # the legacy's np.random.seed(random_choice + i j (· 3)), on the plant's stream
                    stream.seed(int(parameters["random_choice"]) + i * j * (1 if kind == "seminal" else 3))
                    segment = nil_child(segment, "<", False, support_type, 0,
                                        abs(stream.normal(180 if kind == "seminal" else 0, 180)), 0., base_radius)
                    mean = parameters["D_ini"] / 2. * ratio
                    radius = abs(stream.normal(mean, mean * parameters["CVDD"])) if parameters["random"] else mean
                    apex = nil_child(segment, "+", True, apex_type, abs(stream.normal(60, 10)), 5, 0., radius)
                    apex["values"].update(emergence_delay_in_thermal_time=delay, **main_root_durations(radius))

        # The primary root's apex
        apex = nil_child(segment, "<", True, TYPE_EMERGED, 0, 0, parameters["initial_apex_length"], base_radius)
        apex["values"].update(initial_struct_mass=initial_mass, **main_root_durations(base_radius))

        cls._initial_values(parameters, elements)
        return elements

    @classmethod
    def _nil_values(cls, parameters, type_, root_order, angle_down, angle_roll, length, radius) -> dict:
        """A new element's values, RootGrowthModel.ADDING_A_CHILD(nil_properties=True)."""
        tissue_density = parameters["new_root_tissue_density"]
        return {**dict.fromkeys(cls.NIL, 0.), **dict(
            type=type_, root_order=root_order, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE,
            emergence_cost=0., angle_down=angle_down, angle_roll=angle_roll, length=length, radius=radius,
            original_radius=radius, potential_length=0., theoretical_radius=radius, potential_radius=radius,
            initial_length=0., initial_radius=radius, volume=0., struct_mass=0., initial_struct_mass=0.,
            **{resource.concentration: 0. for resource in cls.resources},
            soil_temperature=7.8, root_tissue_density=tissue_density,                                   # g3
            growth_duration=parameters["GDs"] * (2 * radius) ** 2,
            life_duration=parameters["LDs"] * 2. * radius * tissue_density, temperature_modification=0.)}

    @classmethod
    def _emergence_events(cls, parameters, kind) -> list:
        """[(emergence delay in thermal time, number of roots)]: the CSV table of input_file_path when it exists and
        the events are not forced, else built from n_<kind>_roots and ER (RootGrowthModel.initiate_mtg)."""
        path = os.path.join(parameters["input_file_path"], parameters[f"{kind}_roots_events_file"])
        if os.path.exists(path) and not parameters[f"forcing_{kind}_roots_events"]:
            table = pd.read_csv(path)
            return list(zip(table.emergence_delay_in_thermal_time, table[f"number_of_{kind}_roots_per_event"]))
        if kind == "seminal":
            return [(x * 1 / parameters["ER"], 1) for x in range(1, int(parameters["n_seminal_roots"]))]
        start = parameters["starting_time_for_adventitious_roots_emergence"]
        return [(start + x * 1 / parameters["ER"], 1) for x in range(0, int(parameters["n_adventitious_roots"]))]

    @classmethod
    def _initial_values(cls, parameters, elements) -> None:
        """What the legacy derives at its initiation: the inputs everywhere, the elements' volume, masses,
        temperature factor, legacy parent, distance from tip and axis apex, the hexose of the apex."""
        by_index = {element["vertex_index"]: element for element in elements}
        children = {index: [] for index in by_index}
        for element in elements:
            if element["parent"] is not None:
                children[element["parent"]].append(element["vertex_index"])
        # The collar (RootGrowthModel.__init__): the zero-length segments with children are skipped, their children
        # that are not such segments have the base as parent
        zero_segment = {index: (not element["apex"]) and element["values"]["length"] == 0
                        for index, element in by_index.items()}
        collar_children = {child for index in by_index if zero_segment[index] and children[index]
                           for child in children[index] if not zero_segment[child]}
        factor = float(temperature_modification(parameters["soil_temperature"], parameters["process_at_T_ref"],
                                                parameters["T_ref"], parameters["A"], parameters["B"],
                                                parameters["C"]))
        for element in elements:
            values = element["values"]
            values.update(soil_temperature=parameters["soil_temperature"],
                          root_tissue_density=parameters["root_tissue_density"], C_hexose_root=0.)
            values["vertex_index"] = element["vertex_index"]
            values["parent_id"] = (1 if element["vertex_index"] in collar_children else (element["parent"] or -1))
            volume = (4 / 3. * pi * values["radius"] ** 3 if values["type"] == TYPE_NODULE else
                      pi * values["radius"] ** 2 * values["length"])
            values["volume"] = volume
            values["struct_mass"] = volume * parameters["new_root_tissue_density"]
            values["living_struct_mass"] = values["struct_mass"] + values["living_root_hairs_struct_mass"]
            values["temperature_modification"] = factor
        elements[0]["values"]["C_hexose_root"] = parameters["initial_C_hexose_root"]
        if parameters["initial_apex_length"] > 0.:
            elements[-1]["values"]["C_hexose_root"] = parameters["initial_C_hexose_root"]
        # Distance from tip and axis apex: children before parents, along the successors ('<')
        successor_of = {element["parent"]: element["vertex_index"] for element in elements if element["edge"] == "<"}
        for element in reversed(elements):
            values, successor = element["values"], successor_of.get(element["vertex_index"])
            if element["apex"] or values["type"] == TYPE_NODULE or successor is None:
                values["distance_from_tip"], values["axis_apex_id"] = values["length"], element["vertex_index"]
            else:
                successor_values = by_index[successor]["values"]
                values["distance_from_tip"] = successor_values["distance_from_tip"] + values["length"]
                values["axis_apex_id"] = successor_values["axis_apex_id"]
            values["former_distance_from_tip"] = values["distance_from_tip"]

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        """The legacy's seedling under *plant* (the class docstring's structure), its values on the MPG."""
        parameters = cls.plant_parameters(parameters)
        stream = np.random.RandomState()
        elements = cls.seedling(parameters, stream)
        g.property("legacy_random")[plant] = stream
        scales, labels = g.scales, g.labels
        vids = {}

        def new_organ():
            complex_ = plant
            for scale, label in ((scales.Axis, labels.Axis.Root), (scales.GrowthUnit, labels.GrowthUnit.Root),
                                 (scales.Phytomer, labels.Phytomer.Root), (scales.Organ, labels.Organ.RootInternode)):
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            return complex_

        for e in elements:
            label = labels.SubOrgan.RootApex if e["apex"] else labels.SubOrgan.RootSegment
            if e["parent"] is None:
                vid = g.add_component(new_organ(), **PropsConfig(scale=scales.SubOrgan, edge_type='/', label=label))
            elif e["edge"] == "<":
                vid = g.add_child(vids[e["parent"]], **PropsConfig(scale=scales.SubOrgan, edge_type='<', label=label))
            else:
                vid = g.add_component_with_topo(new_organ(), vids[e["parent"]],
                                                **PropsConfig(scale=scales.SubOrgan, edge_type='+', label=label))
            vids[e["vertex_index"]] = vid
            for name, value in e["values"].items():
                g.property(name)[vid] = value
        g.property("total_living_struct_mass")[plant] = float(sum(e["values"]["living_struct_mass"]
                                                                  for e in elements))

    # ── Shared helpers of the steps ─────────────────────────────────────────────────────────────────────────────

    def _apex(self) -> np.ndarray:
        """The RootApex nodes (bool per node), read from the MPG's labels once per topology."""
        ds = self.data_structure
        key = (ds.topology_version, ds.n_nodes())
        cache = self.__dict__.get("_apex_cache")
        if cache is None or cache[0] != key:
            g = ds._mtg
            label, apex = g.property("label"), g.labels.SubOrgan.RootApex
            cache = (key, np.array([label.get(int(v)) == apex for v in ds.entity_ids("node")], dtype=bool))
            self.__dict__["_apex_cache"] = cache
        return cache[1]

    @staticmethod
    def _node_of(vertex_index) -> np.ndarray:
        """Node index of each legacy vertex id (-1 where there is none)."""
        vertex_index = np.asarray(vertex_index, dtype=np.int64)
        lookup = np.full(int(vertex_index.max(initial=0)) + 2, -1, dtype=np.int64)
        lookup[vertex_index] = np.arange(vertex_index.size)
        return lookup

    def _parameter(self, name, selected=None):
        """A per-plant parameter's value on each element (on the *selected* ones), for the hooks."""
        values = self.parameter_values(name)
        return values if selected is None else values[selected]

    def _resource_values(self, role) -> dict:
        """{resource concentration: its variable of *role* (a Resource field) on every element}."""
        ds = self.data_structure
        return {resource.concentration: np.array(ds.get(getattr(resource, role)), dtype=float)
                for resource in self.resources}

    # ── Hooks (devplan §21): the legacy's by default; a subclass overrides them, e.g. for nitrogen ──────────────
    #
    # Each takes arrays (on every element, or on the selected ones) and returns arrays; the parameters it needs are
    # read with self._parameter. A hook may also write variables its subclass declares (self.data_structure.set).

    # Whether the last element of a supply zone (the fraction of it the zone takes) is subject to supply_included
    supply_gates_the_last_element = False

    def supply_included(self, length, contributions):
        """The elements a supply zone takes from when it walks through them (the legacy: those with a length).
        *contributions*: {resource concentration: its amount in each element}."""
        return length > 0.

    def elongated_length(self, selected, start_length, radius, time, zone):
        """RootGrowthModel.elongated_length on the *selected* apices: C-free with simple_growth_duration, else
        Michaelis-Menten on the supply zone's hexose above its threshold (no elongation below). *zone*: {resource
        concentration: its concentration in each apex's supply zone}."""
        EL = self._parameter("EL", selected)
        if self.simple_growth_duration:
            elongation = EL * 2. * radius * time
        else:
            C = zone["C_hexose_root"][selected]
            Km = self._parameter("Km_elongation", selected)
            with np.errstate(divide="ignore", invalid="ignore"):
                elongation = np.where(C > self._parameter("C_hexose_min_for_elongation", selected),
                                      EL * 2. * radius * C / (Km + C) * time, 0.)
        return start_length + elongation

    def synchronise_emergence(self, type, thermal_time_since_primordium_formation, emergence_delay_in_thermal_time):
        """The primordia's times and emergence delays before their emergence is tested (the legacy: unchanged; a
        shoot model may synchronise the nodal roots' emergence with its phytomers)."""
        return thermal_time_since_primordium_formation, emergence_delay_in_thermal_time

    def thickening_rate(self, selected):
        """The relative thickening rate (s-1, before the temperature factor) of the *selected* segments: Michaelis-
        Menten on their hexose."""
        C = self.data_structure.get("C_hexose_root")[selected]
        return (self._parameter("relative_root_thickening_rate_max", selected) * C
                / (self._parameter("Km_thickening", selected) + C))

    def nodule_thickening_rate(self, selected, regulating):
        """The relative thickening rate of the *selected* nodules, from *regulating*: {resource concentration: its
        concentration in the nodule and its bearer together}. Michaelis-Menten on hexose."""
        C = regulating["C_hexose_root"]
        return (self._parameter("relative_nodule_thickening_rate_max", selected) * C
                / (self._parameter("Km_nodule_thickening", selected) + C))

    # ── The supply zones ───────────────────────────────────────────────────────────────────────────────────────

    def _supply(self, supplied, radius, length, volume, struct_mass, growing_zone_factor):
        """
        The supply zones of the apices *supplied* (RootGrowthModel.calculating_supply_for_elongation), element by
        element in the legacy's order (the apices visited children first, each zone from the tip to the base):
        (owner, supplier, {resource concentration: contribution}, mass contribution).
        """
        ds = self.data_structure
        concentrations = {r.concentration: np.asarray(ds.get(r.concentration), dtype=float) for r in self.resources}
        amounts = {name: concentration * struct_mass for name, concentration in concentrations.items()}
        include = np.asarray(self.supply_included(length, amounts), dtype=bool)
        budget = growing_zone_factor * radius * power(radius, 2) * pi
        if self.collar_supply:
            return self._supply_with_collar(supplied, budget, length, volume, struct_mass, amounts, include)
        owner, supplier, mass = ds.path_contributions(budget, volume, struct_mass, where=supplied, include=include)
        contributions = {name: ds.path_contributions(budget, volume, amount, where=supplied, include=include)[2]
                         for name, amount in amounts.items()}
        if self.supply_gates_the_last_element:
            kept = include[supplier]
            mass = np.where(kept, mass, 0.)
            contributions = {name: np.where(kept, values, 0.) for name, values in contributions.items()}
        return owner, supplier, contributions, mass

    def _supply_with_collar(self, supplied, budget, length, volume, struct_mass, amounts, include):
        """The supply zones walked element by element, a zone reaching the base continuing into the collar's other
        children by volume (Root_BRIDGES' calculating_supply_for_elongation_opt, the collar_supply option)."""
        ds = self.data_structure
        parents = np.asarray(ds.parents())
        vertex_index = np.asarray(ds.get("vertex_index"))
        owner_plant = np.asarray(ds.owner("Plant"))
        # The collar's children (the legacy's collar_children): borne by a zero-length support, their legacy parent
        # the base; in the legacy's order (ascending ids)
        collar = (np.asarray(ds.get("parent_id")) == 1) & (parents >= 0) & (vertex_index[np.maximum(parents, 0)] != 1)
        collar = collar[np.argsort(vertex_index, kind="stable")]
        collar_children = np.argsort(vertex_index, kind="stable")[collar]
        names = list(amounts)
        events = []                                     # (owner, supplier, fraction of the supplier)
        visit = ds.order("post", convention="openalea")
        for apex in visit[supplied[visit]].tolist():
            remaining, current, previous = budget[apex], apex, apex
            while remaining > 0:
                if remaining > volume[current]:
                    if length[current] > 0.:
                        if include[current]:
                            events.append((apex, current, None))
                            remaining = remaining - volume[current]
                        if parents[current] < 0:            # the base: the collar's other children
                            children = [k for k in collar_children.tolist()
                                        if owner_plant[k] == owner_plant[apex] and k != previous and length[k] > 0.]
                            # Python's sum of floats (compensated since 3.12), as Root_BRIDGES sums them
                            total = sum([float(volume[k]) for k in children])
                            for child in children:
                                source = min(remaining * volume[child] / total, volume[child])
                                events.append((apex, child, source))
                            if remaining <= total:
                                remaining = 0.
                    if parents[current] < 0:
                        break
                    previous, current = current, parents[current]
                else:
                    events.append((apex, current, remaining))
                    remaining = 0.
                    break
        owner = np.array([e[0] for e in events], dtype=np.int64)
        supplier = np.array([e[1] for e in events], dtype=np.int64)
        contributions, mass = {name: np.empty(len(events)) for name in names}, np.empty(len(events))
        for k, (_, element, source) in enumerate(events):
            for name in names:
                contributions[name][k] = (amounts[name][element] if source is None
                                          else amounts[name][element] * source / volume[element])
            mass[k] = struct_mass[element] if source is None else struct_mass[element] * source / volume[element]
        kept = include[supplier]
        if self.supply_gates_the_last_element and kept.size:
            mass = np.where(kept, mass, 0.)
            contributions = {name: np.where(kept, values, 0.) for name, values in contributions.items()}
        return owner, supplier, contributions, mass

    # ── Potential growth (RootGrowthModel.potential_growth; devplan §14) ───────────────────────────────────────

    @potential
    def _initial_length(self, length, radius, struct_mass, living_root_hairs_struct_mass, soil_temperature,
                        parent_id, vertex_index, process_at_T_ref, T_ref, A, B, C) -> outputs(14):
        """The step's start: the fluxes zeroed, the initial and potential values copied, the temperature factor; a
        zero-length apex takes its parent's soil temperature (the collar's children, the base's: parent_id)."""
        ds = self.data_structure
        parents = np.asarray(ds.parents())
        base = self._node_of(vertex_index)[1]
        parent = np.where(np.asarray(parent_id) == 1, base, parents)
        soil = np.array(soil_temperature, dtype=float)
        zero_apex = self._apex() & (length == 0.)
        soil[zero_apex] = soil_temperature[parent[zero_apex]]
        factor = temperature_modification(soil, process_at_T_ref, T_ref, A, B, C)
        zero = np.zeros(length.shape)
        for resource in self.resources:                 # each resource's consumptions, demand and supply zone
            for role in ("consumption_amount", "consumption", "required", "demand"):
                ds.set(getattr(resource, role), zero)
        return (np.array(length), "initial_radius", np.array(radius), "potential_radius", np.array(radius),
                "theoretical_radius", np.array(radius), "potential_length", np.array(length),
                "initial_struct_mass", np.array(struct_mass),
                "initial_living_root_hairs_struct_mass", np.array(living_root_hairs_struct_mass),
                "soil_temperature", soil, "temperature_modification", factor,
                "resp_growth", zero, "struct_mass_produced", zero, "root_hairs_struct_mass_produced", zero,
                "actual_elongation", zero, "actual_elongation_rate", zero, "hexose_consumption_by_fungus", zero)

    @potential
    def _potential_length(self, type, length, radius, initial_length, initial_radius, potential_length, volume,
                          struct_mass, temperature_modification, emergence_delay_in_thermal_time,
                          growth_duration, life_duration, thermal_time_since_primordium_formation,
                          actual_time_since_primordium_formation, thermal_potential_time_since_emergence,
                          thermal_time_since_emergence, actual_time_since_emergence,
                          thermal_time_since_growth_stopped, actual_time_since_growth_stopped,
                          thermal_time_since_cells_formation, actual_time_since_cells_formation,
                          thermal_time_since_death, actual_time_since_death, struct_mass_contributing_to_elongation,
                          lateral_root_emergence_possibility, lateral_primordium_index, vertex_index,
                          emergence_delay, growing_zone_factor) -> outputs(16):
        """
        The apices (RootGrowthModel.potential_apex_development): the state machine as masked branches, the supply
        zone of those elongating (calculating_supply_for_elongation, by path windows), their potential length; a
        lateral primordium emerging marks its bearer (the last one visited when several emerge on one bearer).
        The branches are chosen on the state at the step's start (the arguments); each element is in one branch.
        The resources' supply zones (their zone and required variables) are written for each resource.
        """
        ds = self.data_structure
        time_step = float(self.time_step)
        temperature_factor = temperature_modification
        thermal_time_step = time_step * temperature_factor
        apex = self._apex()
        thermal_time_since_primordium_formation, emergence_delay_in_thermal_time = self.synchronise_emergence(
            type, np.array(thermal_time_since_primordium_formation), np.array(emergence_delay_in_thermal_time))
        clocks = Clocks(locals())                       # the times since events, from the arguments
        new_type = np.array(type)
        new_potential_length = np.array(potential_length)
        potential_time_since_emergence = np.array(thermal_potential_time_since_emergence)

        # The branches
        primordium = apex & ((type == TYPE_SEMINAL_BEFORE_EMERGENCE) | (type == TYPE_ADVENTITIOUS_BEFORE_EMERGENCE))
        lateral = apex & (type == TYPE_BEFORE_EMERGENCE)
        other = apex & ~primordium & ~lateral
        primordium_emerges = primordium & (thermal_time_since_primordium_formation + thermal_time_step
                                           >= emergence_delay_in_thermal_time)
        lateral_emerges = lateral & (thermal_time_since_primordium_formation + thermal_time_step > emergence_delay)
        growing = other & (thermal_time_since_emergence + thermal_time_step < growth_duration)
        living = other & ~growing & (thermal_time_since_growth_stopped + thermal_time_step < life_duration)
        stopped = living & ((type == TYPE_STOPPED) | (type == TYPE_JUST_STOPPED))
        just_stopped = living & ~stopped
        dying = other & ~growing & ~living
        dead = dying & ((type == TYPE_DEAD) | (type == TYPE_JUST_DEAD))
        just_dead = dying & ~dead

        # The supply zone of the apices that elongate (order-free: it reads what this loop does not change)
        supplied = primordium_emerges | lateral_emerges | growing | just_stopped
        owner, _, contributions, mass_contributions = self._supply(supplied, radius, length, volume, struct_mass,
                                                                   growing_zone_factor)
        mass = np.zeros(length.size)
        np.add.at(mass, owner, mass_contributions)
        contributing = np.array(struct_mass_contributing_to_elongation)
        contributing[supplied] = mass[supplied]
        required, zone = self._resource_values("required"), self._resource_values("zone")
        for resource in self.resources:
            name = resource.concentration
            amount = np.zeros(length.size)
            np.add.at(amount, owner, contributions[name])
            required[name][supplied] = amount[supplied]
            with np.errstate(divide="ignore", invalid="ignore"):
                zone[name][supplied] = np.where(mass[supplied] > 0., amount[supplied] / mass[supplied], 0.)
            ds.set(resource.required, required[name])
            ds.set(resource.zone, zone[name])

        def elongated(selected, start_length, start_radius, time):
            return self.elongated_length(selected, start_length[selected], start_radius[selected], time, zone)

        def capped(time, selected):
            return np.where(time > thermal_time_step[selected], thermal_time_step[selected], time)

        # Seminal and nodal primordia
        selected = primordium_emerges
        potential_time_since_emergence[selected] = capped(
            thermal_time_since_primordium_formation[selected] + thermal_time_step[selected]
            - emergence_delay_in_thermal_time[selected], selected)
        new_potential_length[selected] = elongated(selected, initial_length, initial_radius,
                                                   potential_time_since_emergence[selected])
        if self.simple_growth_duration:
            new_type[selected] = TYPE_EMERGED
        clocks.advance(primordium, time_step, thermal_time_step, "primordium_formation")

        # Lateral primordia
        clocks.advance(lateral, time_step, thermal_time_step, "primordium_formation")
        selected = lateral_emerges
        potential_time_since_emergence[selected] = capped(
            clocks.thermal["primordium_formation"][selected] - emergence_delay[selected], selected)
        new_potential_length[selected] = elongated(selected, initial_length, initial_radius,
                                                   potential_time_since_emergence[selected])
        possibility, primordium_index = np.array(lateral_root_emergence_possibility), np.array(
            lateral_primordium_index)
        if self.simple_growth_duration:
            new_type[selected] = TYPE_EMERGED
        else:
            visit = ds.order("post", convention="openalea")
            emerging = visit[selected[visit]]
            bearers = np.asarray(ds.parents())[emerging]
            last = bearers.size - 1 - np.unique(bearers[::-1], return_index=True)[1]   # last visited per bearer
            possibility[bearers[last]] = EMERGENCE_POSSIBLE
            primordium_index[bearers[last]] = vertex_index[emerging[last]]

        # Growing apices
        selected = growing
        clocks.advance(selected, time_step, thermal_time_step, "primordium_formation", "emergence")
        new_potential_length[selected] = elongated(selected, length, radius, thermal_time_step[selected])

        # Stopped
        selected = stopped
        new_type[selected] = TYPE_STOPPED
        clocks.advance(selected, time_step, thermal_time_step, "growth_stopped", "primordium_formation", "emergence",
                       "cells_formation")

        # Just stopped: the elongation until the growth duration
        selected = just_stopped
        new_type[selected] = TYPE_JUST_STOPPED
        stopped_since = thermal_time_since_emergence[selected] + thermal_time_step[selected] - growth_duration[selected]
        clocks.thermal["growth_stopped"][selected] = stopped_since
        with np.errstate(divide="ignore", invalid="ignore"):
            clocks.actual["growth_stopped"][selected] = stopped_since / temperature_factor[selected]
        remaining = thermal_time_step[selected] - stopped_since
        new_potential_length[selected] = np.where(remaining < 0., initial_length[selected],
                                                  elongated(selected, length, radius, remaining))
        clocks.advance(selected, time_step, thermal_time_step, "primordium_formation", "emergence", "cells_formation")

        # Dead
        selected = dead
        new_type[selected] = TYPE_DEAD
        clocks.advance(selected, time_step, thermal_time_step, "primordium_formation", "emergence", "cells_formation",
                       "growth_stopped", "death")

        # Just dead
        selected = just_dead
        new_type[selected] = TYPE_JUST_DEAD
        dead_since = thermal_time_since_growth_stopped[selected] + thermal_time_step[selected] - life_duration[selected]
        clocks.thermal["death"][selected] = dead_since
        with np.errstate(divide="ignore", invalid="ignore"):
            clocks.actual["death"][selected] = dead_since / temperature_factor[selected]
        clocks.advance(selected, time_step, thermal_time_step, "primordium_formation", "emergence", "cells_formation",
                       "growth_stopped")

        return (new_potential_length, "type", new_type,
                "thermal_potential_time_since_emergence", potential_time_since_emergence, *clocks.outputs(),
                "emergence_delay_in_thermal_time", emergence_delay_in_thermal_time,
                "struct_mass_contributing_to_elongation", contributing,
                "lateral_root_emergence_possibility", possibility, "lateral_primordium_index", primordium_index)

    @potential
    def _theoretical_radius(self, type, length, radius, initial_radius, theoretical_radius, potential_radius,
                            C_hexose_root, struct_mass, temperature_modification, axis_apex_id, vertex_index,
                            thermal_time_since_primordium_formation, actual_time_since_primordium_formation,
                            thermal_time_since_emergence, actual_time_since_emergence,
                            thermal_time_since_cells_formation, actual_time_since_cells_formation,
                            thermal_time_since_growth_stopped, actual_time_since_growth_stopped,
                            thermal_time_since_death, actual_time_since_death, SGC, C_hexose_min_for_thickening,
                            nodule_max_radius) -> outputs(13):
        """
        The segments (RootGrowthModel.potential_segment_development): the type of their axis apex (stopped), death
        when all their children are dead, the pipe model, children first (a fold up the tree), then the potential
        radius limited by hexose, the supports' radius, the times; the nodules' thickening.
        """
        ds = self.data_structure
        time_step = float(self.time_step)
        temperature_factor = temperature_modification
        thermal_time_step = time_step * temperature_factor
        clocks = Clocks(locals())                       # the times since events, from the arguments
        parents = np.asarray(ds.parents())
        segment = ~self._apex()
        nodule = segment & (type == TYPE_NODULE)
        normal = segment & ~nodule
        new_theoretical_radius = np.array(theoretical_radius)
        new_potential_radius, new_radius = np.array(potential_radius), np.array(radius)
        # Each resource's amount available to thickening: the element's; a nodule's and its bearer's
        available = {}
        for resource in self.resources:
            concentration = np.asarray(ds.get(resource.concentration), dtype=float)
            available[resource.concentration] = concentration * struct_mass
            if nodule.any():
                bearer = parents[nodule]
                available[resource.concentration][nodule] = (concentration[bearer] * struct_mass[bearer]
                                                             + concentration[nodule] * struct_mass[nodule])
            ds.set(resource.available_for_thickening, available[resource.concentration])

        # Nodules
        selected = nodule
        if selected.any():
            bearer = parents[selected]
            regulating = {name: values[selected] / (struct_mass[bearer] + struct_mass[selected])
                          for name, values in available.items()}
            rate = self.nodule_thickening_rate(selected, regulating)
            rate = rate * temperature_factor[selected]
            new_theoretical_radius[selected] = radius[selected] * (1 + rate * time_step)
            new_potential_radius[selected] = np.where(new_theoretical_radius[selected] > nodule_max_radius[selected],
                                                      nodule_max_radius[selected], new_theoretical_radius[selected])

        # The axis apex's stop
        new_type = np.array(type)
        node_of = self._node_of(vertex_index)
        apex_node = node_of[np.clip(np.asarray(axis_apex_id, dtype=np.int64), 0, node_of.size - 1)]
        apex_type = np.where(apex_node >= 0, type[np.maximum(apex_node, 0)], -1)
        new_type[normal & (apex_type == TYPE_JUST_STOPPED)] = TYPE_JUST_STOPPED
        new_type[normal & (apex_type == TYPE_STOPPED)] = TYPE_STOPPED

        # Children first: death, time since death, the pipe model
        columns = np.column_stack([new_theoretical_radius, new_type.astype(float),
                                   np.array(actual_time_since_death, dtype=float)])

        def update(level, out):
            nodes = level.nodes
            rows = out[nodes].copy()
            todo = normal[nodes]
            if not todo.any():
                return rows
            child_type = out[:, 1]
            dead_child = (child_type == TYPE_JUST_DEAD) | (child_type == TYPE_DEAD)
            n_children = level.children(np.ones(out.shape[0]), op="sum")
            n_dead = level.children(dead_child.astype(float), op="sum")
            first_death = level.children(out[:, 2], op="min", where=dead_child, fill=0.)
            own_type = rows[:, 1]
            all_dead = todo & (n_dead == n_children)
            dead_type = np.where((own_type == TYPE_JUST_DEAD) | (own_type == TYPE_DEAD), TYPE_DEAD, TYPE_JUST_DEAD)
            level_type = np.where(all_dead, dead_type, own_type)
            if self.radial_growth:
                if self.legacy_pipe_model:
                    successor_section = lateral_sections = np.zeros(nodes.size)
                else:
                    section = power(out[:, 0], 2) * pi
                    successor_section = level.children(section, op="sum", edge="<")
                    lateral_sections = level.children(section, op="sum", edge="+",
                                                      where=(length > 0.) & (child_type != TYPE_NODULE))
                pipe_radius = np.sqrt(successor_section / pi + SGC[nodes] * lateral_sections / pi)
                start_radius = initial_radius[nodes]
                pipe_radius = np.where((pipe_radius - start_radius) <= 0.001 * start_radius, start_radius, pipe_radius)
                rows[todo, 0] = pipe_radius[todo]
            rows[todo, 1] = level_type[todo]
            death = rows[:, 2]
            death = np.where(todo & (level_type == TYPE_JUST_DEAD), np.where(n_children > 0, first_death, 0.), death)
            death = np.where(todo & (level_type == TYPE_DEAD), death + time_step, death)
            rows[:, 2] = death
            return rows

        folded = ds.fold(update, columns, direction="up")
        new_theoretical_radius[normal] = folded[normal, 0]
        new_type[normal] = folded[normal, 1].astype(new_type.dtype)

        # The potential radius
        if self.radial_growth:
            if self.simple_growth_duration:
                new_potential_radius[normal] = new_theoretical_radius[normal]
            else:
                selected = (normal & (length > 0.) & (new_theoretical_radius > radius)
                            & (C_hexose_root > C_hexose_min_for_thickening))
                rate = self.thickening_rate(selected)
                rate = rate * temperature_factor[selected]
                largest = (1 + rate * time_step) * initial_radius[selected]
                new_potential_radius[selected] = np.where(new_theoretical_radius[selected] > largest, largest,
                                                          new_theoretical_radius[selected])
            support = normal & ((new_type == TYPE_SEMINAL_SUPPORT) | (new_type == TYPE_ADVENTITIOUS_SUPPORT))
            new_radius[support] = new_potential_radius[support]

        # The times
        clocks.advance(normal, time_step, thermal_time_step, "primordium_formation", "emergence", "cells_formation")
        selected = normal & (new_type == TYPE_JUST_STOPPED)
        apex_stopped_since = actual_time_since_growth_stopped[np.maximum(apex_node, 0)]
        clocks.actual["growth_stopped"][selected] = apex_stopped_since[selected]
        clocks.thermal["growth_stopped"][selected] = apex_stopped_since[selected] * temperature_factor[selected]
        selected = normal & np.isin(new_type, (TYPE_STOPPED, TYPE_JUST_DEAD, TYPE_DEAD))
        clocks.advance(selected, time_step, thermal_time_step, "growth_stopped")
        clocks.actual["death"][normal] = folded[normal, 2]
        selected = normal & (new_type == TYPE_JUST_DEAD)
        clocks.thermal["death"][selected] = clocks.actual["death"][selected] * temperature_factor[selected]
        selected = normal & (new_type == TYPE_DEAD)
        clocks.thermal["death"][selected] += thermal_time_step[selected]

        return (new_theoretical_radius, "type", new_type, "potential_radius", new_potential_radius,
                "radius", new_radius, *clocks.outputs())

    # ── Actual growth (RootGrowthModel.actual_growth_and_corresponding_respiration) ───────────────────────────

    @staticmethod
    def _volume(type, radius, length):
        """RootGrowthModel.volume_from_radius_and_length: a cylinder, a sphere for a nodule."""
        return np.where(type == TYPE_NODULE, 4 / 3. * pi * power(radius, 3), pi * power(radius, 2) * length)

    # Hooks of actual growth (devplan §21): the legacy's, by hexose alone

    def growth_demand(self, delta_volume, root_tissue_density) -> dict:
        """{resource concentration: its demand} for the potential growth of *delta_volume* (m3): the hexose of its
        structural carbon and of the respiration of its growth."""
        return {"C_hexose_root": delta_volume * root_tissue_density * self._parameter("struct_mass_C_content")
                / self._parameter("yield_growth") * 1 / 6.}

    def growth_continues(self, demand) -> np.ndarray:
        """The elements whose growth goes on, from their demands (the legacy: a positive hexose demand)."""
        return demand["C_hexose_root"] > 0.

    def can_grow(self, for_elongation, for_thickening) -> np.ndarray:
        """The elements with enough of their resources available to grow at all (the legacy: some hexose)."""
        return (for_elongation["C_hexose_root"] + for_thickening["C_hexose_root"]) > 0.

    def elongation_volume_max(self, initial_volume, for_elongation, root_tissue_density) -> np.ndarray:
        """The largest volume the resources available to elongation allow."""
        return (initial_volume + for_elongation["C_hexose_root"] * 6.
                / (root_tissue_density * self._parameter("struct_mass_C_content")) * self._parameter("yield_growth"))

    def elongation_costs(self, delta_volume, root_tissue_density) -> dict:
        """{resource concentration: its amount consumed} by an elongation of *delta_volume*."""
        return {"C_hexose_root": 1. / 6. * delta_volume * root_tissue_density
                * self._parameter("struct_mass_C_content") / self._parameter("yield_growth")}

    def thickening(self, type, initial_radius, length, potential_radius, root_tissue_density, for_thickening):
        """The radius thickening reaches (with the element's new *length*), limited by the resources available to
        it, and {resource concentration: its amount consumed}, on every element."""
        volume_of = self._volume
        C_content, yield_growth = self._parameter("struct_mass_C_content"), self._parameter("yield_growth")
        available = for_thickening["C_hexose_root"]
        increase = available * 6. * yield_growth / (root_tissue_density * C_content)
        volume_max = volume_of(type, initial_radius, length) + increase
        possible_radius = np.where(type == TYPE_NODULE, (3. / (4. * pi)) ** (1. / 3.),             # g5
                                   np.sqrt(volume_max / (length * pi)))
        fits = possible_radius <= potential_radius
        radius = np.where(fits, possible_radius, potential_radius)
        net_increase = volume_of(type, radius, length) - volume_of(type, initial_radius, length)
        cost = np.where(fits, available, 1. / 6. * net_increase * root_tissue_density * C_content / yield_growth)
        return radius, {"C_hexose_root": cost}

    @actual
    def _length(self, type, length, radius, initial_length, initial_radius, potential_length, potential_radius,
                volume, struct_mass, initial_struct_mass, root_tissue_density, temperature_modification,
                struct_mass_contributing_to_elongation, resp_growth, struct_mass_produced,
                thermal_potential_time_since_emergence, thermal_time_since_emergence, actual_time_since_emergence,
                actual_elongation, actual_elongation_rate, dist_to_ramif, growing_zone_factor,
                yield_growth) -> outputs(13):
        """
        Growth limited by the resources of the supply zone (elongation) and of the element (thickening), their cost
        taken from the suppliers in proportion to what each supplies, plus the growth respiration; the primordia
        that elongate emerge. The costs are scattered in the legacy's visiting order (children first), so that each
        element's sums are the legacy's, bit for bit. Each resource's demand and consumptions are written.
        """
        ds = self.data_structure
        time_step = float(self.time_step)
        volume_of = self._volume
        resources = self.resources
        concentrations = {r.concentration: np.asarray(ds.get(r.concentration), dtype=float) for r in resources}
        required = self._resource_values("required")
        with np.errstate(divide="ignore", invalid="ignore"):
            initial_volume = volume_of(type, initial_radius, initial_length)
            potential_volume = volume_of(type, potential_radius, potential_length)
            grows = ~np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD, TYPE_SEMINAL_SUPPORT, TYPE_ADVENTITIOUS_SUPPORT))
            grows &= ~((potential_length <= initial_length) & (potential_radius <= initial_radius))
            demand = self._resource_values("demand")
            computed = self.growth_demand(potential_volume - initial_volume, root_tissue_density)
            for name, values in computed.items():
                demand[name][grows] = np.where(values < 0., 0., values)[grows]
            grows &= self.growth_continues(demand)

            for_elongation, for_thickening = {}, {}
            elongates, thickens = grows & (potential_length > length), grows & (potential_radius > radius)
            for resource in resources:
                name = resource.concentration
                minimum = (self._parameter(resource.minimum_for_elongation) if resource.minimum_for_elongation
                           else 0.)
                for_elongation[name] = np.where(elongates, np.maximum(
                    required[name] - struct_mass_contributing_to_elongation * minimum, 0.), 0.)
                available = np.asarray(ds.get(resource.available_for_thickening), dtype=float)
                if resource.minimum_for_thickening:
                    available = np.maximum(available - struct_mass * self._parameter(resource.minimum_for_thickening),
                                           0.)
                for_thickening[name] = np.where(thickens, available, 0.)
            grows &= self.can_grow(for_elongation, for_thickening)

            # Elongation, limited by the supply zone's resources
            new_length = np.array(length)
            length_max = (self.elongation_volume_max(initial_volume, for_elongation, root_tissue_density)
                          / (pi * power(initial_radius, 2)))
            elongating = grows & (potential_length > initial_length)
            new_length[elongating] = np.where(potential_length[elongating] >= length_max[elongating],
                                              length_max[elongating], potential_length[elongating])
            elongation_cost = self.elongation_costs(volume_of(type, initial_radius, new_length) - initial_volume,
                                                    root_tissue_density)
            supplied = elongating & (new_length > initial_length)

            # Thickening, limited by the element's resources
            thickening = grows & (potential_radius > initial_radius)
            reached, thickening_cost = self.thickening(type, initial_radius, new_length, potential_radius,
                                                       root_tissue_density, for_thickening)
            new_radius = np.array(radius)
            new_radius[thickening] = reached[thickening]
            parents = np.asarray(ds.parents())
            nodule = thickening & (type == TYPE_NODULE)

        # The costs, in the legacy's order: per element visited (children first), its elongation's suppliers (tip
        # to base), then its own thickening, then a nodule's parent
        visit = ds.order("post", convention="openalea")
        rank = np.empty(visit.size, dtype=np.int64)
        rank[visit] = np.arange(visit.size)
        owner, supplier, contributions, _ = self._supply(supplied, radius, length, volume, struct_mass,
                                                         growing_zone_factor)
        thickened, nodules = np.flatnonzero(thickening), np.flatnonzero(nodule)
        event_owner = np.concatenate([owner, thickened, nodules])
        target = np.concatenate([supplier, thickened, parents[nodules]])
        kind = np.concatenate([np.zeros(owner.size), np.ones(thickened.size), np.full(nodules.size, 2.)])
        order = np.lexsort((np.arange(event_owner.size), kind, rank[event_owner]))
        target = target[order]
        respiration = np.array(resp_growth)
        for index, resource in enumerate(resources):
            name = resource.concentration
            denominator = for_elongation[name] if resource.costs_by_available else required[name]
            share = concentrations[name] * initial_struct_mass
            with np.errstate(divide="ignore", invalid="ignore"):
                elongation_events = np.where(denominator[owner] > 0., elongation_cost[name][owner]
                                             * contributions[name] / denominator[owner], 0.)
                own_share = np.where(for_thickening[name] > 0., thickening_cost[name]
                                     * (share / for_thickening[name]), 0.)
                parent_share = np.where(for_thickening[name] > 0., thickening_cost[name]
                                        * (share[parents] / for_thickening[name]), 0.)
            value = np.concatenate([elongation_events, own_share[thickened], parent_share[nodules]])[order]
            consumed = np.array(ds.get(resource.consumption_amount), dtype=float)
            consumption_rate = np.array(ds.get(resource.consumption), dtype=float)
            np.add.at(consumed, target, value)
            np.add.at(consumption_rate, target, value / time_step)
            if index == 0:                                  # the carbon: growth respiration
                np.add.at(respiration, target, value * (1 - yield_growth[target]) * 6.)
            ds.set(resource.consumption_amount, consumed)
            ds.set(resource.consumption, consumption_rate)
            ds.set(resource.demand, demand[name])

        # The new structure
        new_volume, new_mass = np.array(volume), np.array(struct_mass)
        produced, new_initial_mass = np.array(struct_mass_produced), np.array(initial_struct_mass)
        new_volume[grows] = volume_of(type, new_radius, new_length)[grows]
        new_mass[grows] = new_volume[grows] * root_tissue_density[grows]
        produced[grows] = (new_volume[grows] - initial_volume[grows]) * root_tissue_density[grows]
        shrinks = grows & (new_mass < initial_struct_mass) & (produced > 0.)
        new_initial_mass[shrinks] = new_mass[shrinks]

        # Emergence, elongation
        new_type = np.array(type)
        time_since_emergence = np.array(thermal_time_since_emergence)
        actual_since_emergence = np.array(actual_time_since_emergence)
        elongation, elongation_rate = np.array(actual_elongation), np.array(actual_elongation_rate)
        new_dist_to_ramif = np.array(dist_to_ramif)
        longer = grows & (new_length > initial_length)
        emerges = longer & np.isin(type, (TYPE_SEMINAL_BEFORE_EMERGENCE, TYPE_ADVENTITIOUS_BEFORE_EMERGENCE,
                                          TYPE_BEFORE_EMERGENCE))
        new_type[emerges] = TYPE_EMERGED
        time_since_emergence[emerges] = thermal_potential_time_since_emergence[emerges]
        with np.errstate(divide="ignore", invalid="ignore"):
            actual_since_emergence[emerges] = time_since_emergence[emerges] / temperature_modification[emerges]
            elongation[emerges] = new_length[emerges] - initial_length[emerges]
            elongation_rate[emerges] = elongation[emerges] / actual_since_emergence[emerges]
        emerged = longer & (type == TYPE_EMERGED)
        elongation[emerged] = new_length[emerged] - initial_length[emerged]
        elongation_rate[emerged] = elongation[emerged] / time_step
        new_dist_to_ramif[longer] += elongation[longer]

        return (new_length, "radius", new_radius, "volume", new_volume, "struct_mass", new_mass,
                "struct_mass_produced", produced, "initial_struct_mass", new_initial_mass,
                "resp_growth", respiration, "type", new_type,
                "thermal_time_since_emergence", time_since_emergence, "actual_time_since_emergence",
                actual_since_emergence, "actual_elongation", elongation, "actual_elongation_rate", elongation_rate,
                "dist_to_ramif", new_dist_to_ramif, "step_elongating", elongating.astype(int))

    # ── Segmentation and primordia (RootGrowthModel.segmentation_and_primordia_formation) ─────────────────────
    #
    # Sequential by nature: each apex creates elements whose legacy ids (vertex_index, the next free one of its
    # plant) seed the following draws, in the legacy's order (the apices elongated in actual growth, visited children
    # first). An MPG-style step: it reads and edits the MPG, as the legacy its MTG; the draws come from the plant's
    # stream, re-seeded where the legacy re-seeds numpy's global generator, so that even the legacy's draws that follow
    # a seeding without one (a nodule's) are the same.

    # What a new element copies from its mother (RootGrowthModel.ADDING_A_CHILD(identical_properties=True)), besides
    # the radius (its radius, original and initial radius)
    COPIED = ("theoretical_radius", "potential_radius", "root_tissue_density", "dist_to_ramif", "distance_from_tip",
              "former_distance_from_tip", "actual_elongation", "actual_elongation_rate", "struct_mass",
              "initial_struct_mass", "C_hexose_root", "root_hair_length", "actual_length_with_hairs",
              "living_root_hairs_number", "dead_root_hairs_number", "total_root_hairs_number",
              "actual_time_since_root_hairs_emergence_started", "thermal_time_since_root_hairs_emergence_started",
              "actual_time_since_root_hairs_emergence_stopped", "thermal_time_since_root_hairs_emergence_stopped",
              "all_root_hairs_formed", "root_hairs_struct_mass", "root_hairs_struct_mass_produced",
              "living_root_hairs_struct_mass", "initial_living_root_hairs_struct_mass", "resp_growth",
              "struct_mass_produced", "hexose_growth_demand", "hexose_possibly_required_for_elongation",
              "hexose_consumption_by_growth_amount", "hexose_consumption_by_growth", "hexose_consumption_by_fungus",
              "soil_temperature", "growth_duration", "life_duration", "actual_time_since_primordium_formation",
              "actual_time_since_emergence", "actual_time_since_cells_formation", "actual_time_since_growth_stopped",
              "actual_time_since_death", "thermal_time_since_primordium_formation", "thermal_time_since_emergence",
              "thermal_time_since_cells_formation", "thermal_potential_time_since_emergence",
              "thermal_time_since_growth_stopped", "thermal_time_since_death", "temperature_modification")
    # What segmentation shares between the segments by their mass
    SHARED = ("resp_growth", "initial_struct_mass", "initial_living_root_hairs_struct_mass", "struct_mass_produced",
              "root_hairs_struct_mass", "root_hairs_struct_mass_produced", "living_root_hairs_struct_mass",
              "living_root_hairs_number", "dead_root_hairs_number", "total_root_hairs_number", "hexose_growth_demand",
              "hexose_consumption_by_growth_amount", "hexose_consumption_by_growth", "hexose_consumption_by_fungus")
    def _inputs_of_new(self) -> tuple:
        """The inputs a new element gets from segmentation: the soil's and the anatomy's, each resource's
        concentration."""
        return ("soil_temperature", "root_tissue_density") + tuple(r.concentration for r in self.resources)
    SEGMENTATION_PARAMETERS = ("segment_length", "random_choice", "IPD", "Dmin", "RMD", "CVDD", "root_order_treshold",
                               "GDs", "LDs", "main_roots_growth_extender", "new_root_tissue_density", "GD_highest",
                               "GD_high", "GD_medium", "GD_low", "GD_prob_low", "GD_prob_medium",
                               "nodule_formation_probability", "struct_mass_C_content")

    @segmentation
    def _segmentation_and_primordia_formation(self):
        ds = self.data_structure
        g = self.mtg
        properties = g.properties()
        vids = np.asarray(ds.entity_ids("node"))
        visit = ds.order("post", convention="openalea")
        candidate = self._apex() & (np.asarray(ds.get("type")) == TYPE_EMERGED) & (
            np.asarray(ds.get("step_elongating")) == 1)
        apices = vids[visit[candidate[visit]]].tolist()
        self.__dict__["_step_new_apices"] = []
        self.__dict__["_new_inputs"] = {}
        if not apices:
            return
        values = {name: dict(zip(vids.tolist(), self.parameter_values(name).tolist()))
                  for name in self.SEGMENTATION_PARAMETERS + self.ELEMENT_PARAMETERS}
        plant_of = dict(zip(vids.tolist(), np.asarray(ds.entity_ids("Plant"))[ds.owner("Plant")].tolist()))
        index = np.asarray(ds.get("vertex_index"), dtype=np.int64)
        next_index = {}
        for plant, last in zip(plant_of.values(), index.tolist()):
            next_index[plant] = max(next_index.get(plant, 0), last + 1)
        streams = g.property("legacy_random")
        for apex in apices:
            plant = plant_of[apex]
            if plant not in streams:
                streams[plant] = np.random.RandomState()
            context = dict(parameters={name: values[name][apex] for name in values}, plant=plant, next_index=next_index,
                           stream=streams[plant])
            new = self._segment_apex(g, properties, apex, context)
            for v in new:
                if v not in self.__dict__["_step_new_apices"]:
                    self.__dict__["_step_new_apices"].append(v)

    def _new_element(self, g, properties, mother, edge, apex, values, context):
        """A new element under *mother* ('<' in its Organ; '+' a new Organ of its Axis, a lateral or a nodule), its
        legacy id the plant's next one, and *values* on the MPG (its inputs kept for _inputs_of_new_elements)."""
        scales, labels = g.scales, g.labels
        label = labels.SubOrgan.RootApex if apex else labels.SubOrgan.RootSegment
        if edge == "<":
            vid = g.add_child(mother, **PropsConfig(scale=scales.SubOrgan, edge_type='<', label=label))
        else:
            organ = g.add_component(g.complex_at_scale(mother, scales.Phytomer),
                                    **PropsConfig(scale=scales.Organ, edge_type='/', label=labels.Organ.RootInternode))
            vid = g.add_component_with_topo(organ, mother, **PropsConfig(scale=scales.SubOrgan, edge_type='+',
                                                                         label=label))
        plant = context["plant"]
        properties["vertex_index"][vid] = context["next_index"][plant]
        properties["parent_id"][vid] = properties["vertex_index"][mother]
        context["next_index"][plant] += 1
        inputs = {}
        for name, value in values.items():
            if name in self._inputs_of_new():
                inputs[name] = value
            elif name not in self.ELEMENT_PARAMETERS:
                properties[name][vid] = value
        self.__dict__["_new_inputs"][vid] = inputs
        return vid

    def _value(self, properties, name, vid):
        """An element's value, its pending one for a new element's input."""
        pending = self.__dict__["_new_inputs"].get(vid)
        if pending is not None and name in pending:
            return pending[name]
        return properties[name][vid]

    def _copy_child(self, g, properties, mother, type_, root_order, angle_down, angle_roll, length, context):
        """RootGrowthModel.ADDING_A_CHILD('<', label Apex, identical_properties=True)."""
        radius = properties["radius"][mother]
        values = {name: self._value(properties, name, mother) for name in self.COPIED}
        for resource in self.resources:
            values[resource.concentration] = self._value(properties, resource.concentration, mother)
            if not resource.copied_by_new_elements:
                for role in ("required", "demand", "consumption_amount", "consumption"):
                    values[getattr(resource, role)] = 0.
        values.update(type=type_, root_order=root_order, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE,
                      emergence_cost=0., angle_down=angle_down, angle_roll=angle_roll, length=length,
                      radius=radius, original_radius=radius, potential_length=length, initial_length=length,
                      initial_radius=radius, volume=0.)
        return self._new_element(g, properties, mother, "<", True, values, context)

    def _segment_apex(self, g, properties, apex, context) -> list:
        """RootGrowthModel.segmentation_and_primordium_formation: the elements it creates or returns."""
        parameters, stream = context["parameters"], context["stream"]
        time_step = float(self.time_step)
        temperature_factor = properties["temperature_modification"][apex]
        legacy_id = int(properties["vertex_index"][apex])
        if self.random:
            stream.seed(int(parameters["random_choice"]) * legacy_id)
            segment_angle_down = stream.normal(0, 5)
            segment_angle_roll = stream.normal(0, 5)
            stream.normal(0, 5)                    # the legacy's apex_angle_down and apex_angle_roll, unused
            stream.normal(0, 5)
        else:
            segment_angle_down = segment_angle_roll = 0
        new = []
        root_order = properties["root_order"][apex]
        initial_length = properties["length"][apex]
        initial_dist_to_ramif = properties["dist_to_ramif"][apex]
        initial_elongation = properties["actual_elongation"][apex]
        initial_elongation_rate = properties["actual_elongation_rate"][apex]
        shared = {name: properties[name][apex] for name in self.SHARED}
        # (g1) the legacy overwrites its initial_struct_mass and initial_living_root_hairs_struct_mass with the apex's
        # values before growth: the mass fraction is relative to the mass before growth, and the living root hairs'
        # mass is shared from their mass before growth
        initial_struct_mass = properties["initial_struct_mass"][apex]
        shared["living_root_hairs_struct_mass"] = properties["initial_living_root_hairs_struct_mass"][apex]
        segment_length = parameters["segment_length"]

        if initial_length <= segment_length:
            new += self._primordium_formation(g, properties, apex, initial_elongation_rate, context)
            rate = properties["actual_elongation_rate"][apex]
            if rate > 0.:
                age_non_elongated = properties["actual_time_since_cells_formation"][apex] + time_step
                age_up = properties["actual_elongation"][apex] / rate
                age_elongated = 0.5 * (0 + age_up)
                length, length0 = properties["length"][apex], properties["initial_length"][apex]
                properties["actual_time_since_cells_formation"][apex] = (length0 * age_non_elongated
                                                                + (length - length0) * age_elongated) / length
                thermal_non_elongated = (properties["thermal_time_since_cells_formation"][apex]
                                         + time_step * temperature_factor)
                thermal_elongated = age_elongated * temperature_factor
                properties["thermal_time_since_cells_formation"][apex] = (length0 * thermal_non_elongated
                                                                 + (length - length0) * thermal_elongated) / length
            return new

        if initial_length / segment_length - floor(initial_length / segment_length) > 0.:
            n_segments = floor(initial_length / segment_length)
        else:
            n_segments = floor(initial_length / segment_length) - 1
        n_segments = int(n_segments)
        final_apex_length = initial_length - n_segments * segment_length
        element = apex
        for i in range(1, n_segments + 1):
            properties["length"][element] = segment_length
            properties["dist_to_ramif"][element] = initial_dist_to_ramif - (initial_length - segment_length * i)
            self._share_mass(properties, element, segment_length, initial_struct_mass, shared)
            rate = properties["actual_elongation_rate"][element]
            if i == 1:
                age_non_elongated = properties["actual_time_since_cells_formation"][element] + time_step
                age_up = properties["actual_elongation"][element] / rate
                age_down = (final_apex_length + (n_segments - 1) * segment_length) / rate
                age_elongated = 0.5 * (age_down + age_up)
                length, length0 = properties["length"][element], properties["initial_length"][element]
                properties["actual_time_since_cells_formation"][element] = (length0 * age_non_elongated + (
                    length - length0) * age_elongated) / length
                thermal_non_elongated = (properties["thermal_time_since_cells_formation"][element]
                                         + time_step * temperature_factor)
                thermal_elongated = age_elongated * temperature_factor
                properties["thermal_time_since_cells_formation"][element] = (length0 * thermal_non_elongated + (
                    length - length0) * thermal_elongated) / length
            else:
                age_up = (final_apex_length + (n_segments - i) * segment_length) / rate
                age_down = (final_apex_length + (n_segments - 1 - i) * segment_length) / rate
                properties["actual_time_since_cells_formation"][element] = 0.5 * (age_down + age_up)
                properties["thermal_time_since_cells_formation"][element] = (
                    properties["actual_time_since_cells_formation"][element] * temperature_factor)
            new += self._primordium_formation(g, properties, element, initial_elongation_rate, context)
            properties["label"][element] = g.labels.SubOrgan.RootSegment
            if i < n_segments:
                element = self._copy_child(g, properties, element, properties["type"][element], root_order,
                                           segment_angle_down, segment_angle_roll, 0., context)
                properties["actual_elongation"][element] = segment_length * i
            elif (self.nodules and len(g.children(element)) < 2
                  and stream.random_sample() < parameters["nodule_formation_probability"]):
                self._nodule_formation(g, properties, element, context)

        new_length = initial_length - n_segments * segment_length
        element = self._copy_child(g, properties, element, properties["type"][element], root_order, segment_angle_down,
                                   segment_angle_roll, new_length, context)
        properties["potential_length"][element] = new_length
        properties["initial_length"][element] = new_length
        properties["dist_to_ramif"][element] = initial_dist_to_ramif
        properties["actual_elongation"][element] = initial_elongation
        self._share_mass(properties, element, new_length, initial_struct_mass, shared)
        age_up = final_apex_length / properties["actual_elongation_rate"][element]
        properties["actual_time_since_cells_formation"][element] = 0.5 * (0 + age_up)
        properties["thermal_time_since_cells_formation"][element] = (
            properties["actual_time_since_cells_formation"][element] * temperature_factor)
        new += self._primordium_formation(g, properties, element, initial_elongation_rate, context)
        new.append(element)
        return new

    def _share_mass(self, properties, element, length, initial_struct_mass, shared):
        """A segment's (or the new apex's) volume and mass, and its share of the apex's amounts by its mass."""
        radius = properties["radius"][element]
        volume = pi * radius ** 2 * length if properties["type"][element] != TYPE_NODULE else 4 / 3. * pi * radius ** 3
        properties["volume"][element] = volume
        properties["struct_mass"][element] = volume * self._value(properties, "root_tissue_density", element)
        fraction = properties["struct_mass"][element] / initial_struct_mass
        for name, value in shared.items():
            properties[name][element] = value * fraction

    def _primordium_formation(self, g, properties, apex, elongation_rate, context) -> list:
        """RootGrowthModel.primordium_formation: a lateral primordium on *apex*, when its distance to the last one,
        its radius and its order allow."""
        parameters, stream = context["parameters"], context["stream"]
        if len(g.children(apex)) >= 1:
            return [apex]
        root_order = properties["root_order"][apex]
        if self.root_order_limitation and root_order + 1 > parameters["root_order_treshold"]:
            return [apex]
        radius = properties["radius"][apex]
        Dmin = parameters["Dmin"]
        mean = (radius - Dmin / 2.) * parameters["RMD"] + Dmin / 2.
        if self.random:
            stream.seed(int(parameters["random_choice"]) * int(properties["vertex_index"][apex]))
            potential_radius = stream.normal((radius - Dmin / 2.) * parameters["RMD"] + Dmin / 2.,
                                          ((radius - Dmin / 2.) * parameters["RMD"] + Dmin / 2.) * parameters["CVDD"])
            apex_angle_roll = abs(stream.normal(120, 10))
            primordium_angle_down = abs(stream.normal(45, 10)) if root_order == 1 else abs(stream.normal(70, 10))
            primordium_angle_roll = abs(stream.normal(5, 5))
        else:
            potential_radius = (radius - Dmin / 2) * parameters["RMD"] + Dmin / 2.
            apex_angle_roll = 120
            primordium_angle_down = 45 if root_order == 1 else 70
            primordium_angle_roll = 5
        if potential_radius > radius:
            potential_radius = radius
        dist_to_ramif = properties["dist_to_ramif"][apex]
        if not (dist_to_ramif > parameters["IPD"] and potential_radius >= Dmin and potential_radius <= radius):
            return []
        elongation_since_last_ramif = dist_to_ramif - parameters["IPD"]
        properties["angle_roll"][apex] = apex_angle_roll
        if properties["actual_elongation"][apex] > 0 and elongation_rate > 0.:
            time_since_formation = elongation_since_last_ramif / elongation_rate
        else:
            time_since_formation = 0.
        values = self._nil_values(parameters, TYPE_BEFORE_EMERGENCE, root_order + 1, primordium_angle_down,
                                  primordium_angle_roll, 0., potential_radius)
        self._new_element_soil_temperature(properties, values, apex)
        ramif = self._new_element(g, properties, apex, "+", True, values, context)
        if self.simple_growth_duration:
            properties["growth_duration"][ramif] = (parameters["GDs"] * (2. * potential_radius) ** 2
                                                    * self.lateral_growth_duration_factor(properties, apex)
                                                    * parameters["main_roots_growth_extender"])
        else:
            properties["growth_duration"][ramif] = self._growth_duration(
                potential_radius, int(properties["vertex_index"][ramif]), root_order + 1, parameters, stream)
        properties["actual_time_since_primordium_formation"][ramif] = time_since_formation
        properties["thermal_time_since_primordium_formation"][ramif] = (time_since_formation
                                                               * properties["temperature_modification"][apex])
        properties["dist_to_ramif"][apex] = elongation_since_last_ramif
        return [apex, ramif]

    def _new_element_soil_temperature(self, properties, values, mother):
        """A primordium's soil temperature: the legacy's 7.8 °C (g3), or its mother's (new_element_soil_temperature
        = "mother", Root_BRIDGES')."""
        if self.new_element_soil_temperature == "mother":
            values["soil_temperature"] = self._value(properties, "soil_temperature", mother)
        else:
            values["soil_temperature"] = self.new_element_soil_temperature

    def lateral_growth_duration_factor(self, properties, apex):
        """Hook: the factor of a lateral primordium's growth duration with simple_growth_duration, from its bearing
        apex (*properties*: the MPG's; the legacy: 1)."""
        return 1.

    def _growth_duration(self, radius, legacy_id, root_order, parameters, stream):
        """RootGrowthModel.calculate_growth_duration (not simple_growth_duration), with its if/if (g4)."""
        stream.seed(int(parameters["random_choice"]) * legacy_id)
        draw = stream.random_sample()
        if root_order == 1:
            return parameters["GD_highest"]
        if self.GD_by_frequency:
            if draw < parameters["GD_prob_low"]:
                duration = stream.uniform(0., parameters["GD_low"])
            if draw < parameters["GD_prob_medium"]:
                duration = stream.uniform(parameters["GD_low"], parameters["GD_medium"])
            else:
                duration = stream.uniform(parameters["GD_medium"], parameters["GD_high"])
            return duration
        if draw < parameters["GD_prob_low"]:
            return parameters["GD_low"]
        if draw < parameters["GD_prob_medium"]:
            return parameters["GD_medium"]
        if radius > 2 * parameters["Dmin"] / 2.:
            return parameters["GD_highest"]
        raise ValueError(f"no growth duration for a radius of {radius} m (the legacy's calculate_growth_duration "
                         "leaves it unassigned, g4)")

    def _nodule_formation(self, g, properties, mother, context):
        """RootGrowthModel.nodule_formation: a nodule, a lateral segment without apex."""
        parameters = context["parameters"]
        values = self._nil_values(parameters, TYPE_NODULE, properties["root_order"][mother] + 1, 90, 0, 0, 0)
        self._new_element_soil_temperature(properties, values, mother)
        radius = properties["radius"][mother]
        values.update(length=radius, radius=radius, original_radius=radius)
        values["volume"] = 4 / 3. * pi * radius ** 3
        values["struct_mass"] = values["volume"] * values["root_tissue_density"] * parameters["struct_mass_C_content"]
        return self._new_element(g, properties, mother, "+", False, values, context)

    @segmentation
    def _soil_temperature(self, soil_temperature, root_tissue_density) -> outputs(1):
        """The inputs of the elements segmentation created: copied from their mother, or a primordium's (7.8 °C or
        its mother's, new_root_tissue_density, no resource); each resource's concentration written."""
        pending = self.__dict__.get("_new_inputs") or {}
        ds = self.data_structure
        values = {"soil_temperature": np.array(soil_temperature), "root_tissue_density": np.array(root_tissue_density)}
        values.update({r.concentration: np.array(ds.get(r.concentration), dtype=float) for r in self.resources})
        if pending:
            index = dict(zip(np.asarray(ds.entity_ids("node")).tolist(), range(values["soil_temperature"].size)))
            for vid, inputs in pending.items():
                for name, value in inputs.items():
                    values[name][index[vid]] = value
        self.__dict__["_new_inputs"] = {}
        for resource in self.resources:
            ds.set(resource.concentration, values[resource.concentration])
        return values["soil_temperature"], "root_tissue_density", values["root_tissue_density"]

    # ── After growth (RootGrowthModel.post_growth_updating, standalone: the repartition is metafspm's) ─────────

    @postsegmentation
    def _distance_from_tip(self, length, type, distance_from_tip, vertex_index) -> outputs(2):
        """Distance from the tip and the axis' apex, along the successors from the tips (a fold up the tree); the
        former distance kept."""
        apex = self._apex()
        stop = apex | (type == TYPE_NODULE)
        columns = np.column_stack([np.array(length, dtype=float), np.asarray(vertex_index, dtype=float)])

        def update(level, out):
            nodes = level.nodes
            successors = level.children(np.ones(out.shape[0]), op="sum", edge="<")
            son_distance = level.children(out[:, 0], op="sum", edge="<")
            son_apex = level.children(out[:, 1], op="sum", edge="<")
            tip = stop[nodes] | (successors == 0)
            rows = np.empty((nodes.size, 2))
            rows[:, 0] = np.where(tip, length[nodes], son_distance + length[nodes])
            rows[:, 1] = np.where(tip, vertex_index[nodes], son_apex)
            return rows

        folded = self.data_structure.fold(update, columns, direction="up")
        return (folded[:, 0], "axis_apex_id", folded[:, 1].astype(np.int64),
                "former_distance_from_tip", np.array(distance_from_tip))

    # Hooks of the root hairs (devplan §21)

    def root_hair_elongation(self, selected, root_hair_length, share_with_hairs, elapsed_thermal_time):
        """The new root hair length of the *selected* elements (whose hairs are shorter than their maximum), from
        the share of their length bearing hairs and the thermal time elapsed: Michaelis-Menten on their hexose,
        capped at the maximal length."""
        C = self.data_structure.get("C_hexose_root")[selected]
        maximum = self._parameter("root_hair_max_length", selected)
        new_length = (root_hair_length + self._parameter("root_hairs_elongation_rate", selected)
                      * self._parameter("root_hair_radius", selected) * C * share_with_hairs
                      / (self._parameter("Km_elongation", selected) + C) * elapsed_thermal_time)
        return np.where(new_length > maximum, maximum, new_length)

    def root_hair_costs(self, produced) -> dict:
        """{resource concentration: its amount consumed} by the root hairs' structural mass *produced*."""
        return {"C_hexose_root": produced * self._parameter("struct_mass_C_content")
                / self._parameter("yield_growth") / 6.}

    @postsegmentation
    def _root_hair_length(self, type, length, radius, struct_mass, distance_from_tip,
                          former_distance_from_tip, temperature_modification, root_tissue_density,
                          root_hair_length, actual_length_with_hairs, total_root_hairs_number, dead_root_hairs_number,
                          living_root_hairs_number, all_root_hairs_formed, root_hairs_struct_mass,
                          living_root_hairs_struct_mass, root_hairs_struct_mass_produced, root_hairs_volume,
                          actual_time_since_root_hairs_emergence_started,
                          thermal_time_since_root_hairs_emergence_started,
                          actual_time_since_root_hairs_emergence_stopped,
                          thermal_time_since_root_hairs_emergence_stopped, resp_growth, growing_zone_factor,
                          root_hairs_density, root_hair_max_length, root_hair_radius, root_hairs_lifespan,
                          yield_growth) -> outputs(14):
        """RootGrowthModel.root_hairs_dynamics_opt on the elements of the legacy's focus (a structural mass, a
        Segment or Apex label, an emerged, stopped, dead or nodule type; not a collar's zero-length support):
        emergence along the element beyond the growing zone, death after their lifespan, their elongation limited
        by hexose, their mass and its cost."""
        ds = self.data_structure
        time_step = float(self.time_step)
        growing_zone_length = growing_zone_factor * radius
        focus = (struct_mass > 0.) & np.isin(type, (TYPE_BASE, TYPE_EMERGED, TYPE_STOPPED, TYPE_JUST_STOPPED,
                                                    TYPE_DEAD, TYPE_JUST_DEAD, TYPE_NODULE))
        indptr, _ = ds.children()
        collar = ~self._apex() & (length == 0.) & (np.diff(np.asarray(indptr)) > 0)   # the legacy's collar_skip
        selected = (focus & ~collar & ~np.isin(type, (TYPE_JUST_DEAD, TYPE_DEAD, TYPE_NODULE))
                    & (distance_from_tip > growing_zone_length))
        adjustment = np.maximum(1e-3, temperature_modification)
        elapsed = time_step * adjustment
        selected &= elapsed > 0.
        out = {name: np.array(value, dtype=float) for name, value in dict(
            root_hair_length=root_hair_length, actual_length_with_hairs=actual_length_with_hairs,
            total_root_hairs_number=total_root_hairs_number, dead_root_hairs_number=dead_root_hairs_number,
            living_root_hairs_number=living_root_hairs_number, all_root_hairs_formed=all_root_hairs_formed,
            root_hairs_struct_mass=root_hairs_struct_mass, living_root_hairs_struct_mass=living_root_hairs_struct_mass,
            root_hairs_struct_mass_produced=root_hairs_struct_mass_produced, root_hairs_volume=root_hairs_volume,
            started=actual_time_since_root_hairs_emergence_started,
            thermal_started=thermal_time_since_root_hairs_emergence_started,
            stopped=actual_time_since_root_hairs_emergence_stopped,
            thermal_stopped=thermal_time_since_root_hairs_emergence_stopped, resp=resp_growth).items()}
        initial_mass = np.array(root_hairs_struct_mass, dtype=float)
        total = np.array(total_root_hairs_number, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            # All formed: they age
            all_formed = selected & (all_root_hairs_formed != 0)
            out["started"][all_formed] += time_step
            out["thermal_started"][all_formed] += elapsed[all_formed]
            out["stopped"][all_formed] += time_step
            out["thermal_stopped"][all_formed] += elapsed[all_formed]
            # Forming: along the part of the element beyond the growing zone
            forming = selected & (all_root_hairs_formed == 0) & (distance_from_tip - length < growing_zone_length)
            initial_with_hairs = np.array(actual_length_with_hairs, dtype=float)
            out["actual_length_with_hairs"][forming] = (distance_from_tip - growing_zone_length)[forming]
            increase = out["actual_length_with_hairs"] - initial_with_hairs
            total[forming] = (root_hairs_density * radius * out["actual_length_with_hairs"])[forming]
            rate_actual = (distance_from_tip - former_distance_from_tip) / time_step
            rate_thermal = (distance_from_tip - former_distance_from_tip) / elapsed
            first = forming & (rate_actual > 0.) & (initial_with_hairs <= 0.)
            out["started"][first] += (time_step - increase / rate_actual)[first]
            out["thermal_started"][first] += (elapsed - increase / rate_thermal)[first]
            later = forming & ~first
            out["started"][later] += time_step
            out["thermal_started"][later] += elapsed[later]
            # All formed now: along the whole element
            completing = selected & (all_root_hairs_formed == 0) & ~(distance_from_tip - length < growing_zone_length)
            out["started"][completing] += time_step
            out["thermal_started"][completing] += elapsed[completing]
            out["actual_length_with_hairs"][completing] = length[completing]
            increase = out["actual_length_with_hairs"] - initial_with_hairs
            total[completing] = (root_hairs_density * radius * length)[completing]
            moving = completing & (rate_actual > 0.)
            out["stopped"][moving] += (time_step - increase / rate_actual)[moving]
            out["thermal_stopped"][moving] += (elapsed - increase / rate_thermal)[moving]
            still = completing & ~moving
            out["stopped"][still] += time_step
            out["thermal_stopped"][still] += elapsed[still]
            out["all_root_hairs_formed"][completing] = 1.
            out["total_root_hairs_number"][forming | completing] = total[forming | completing]
            # Death after their lifespan
            dead = np.where(out["thermal_started"] <= root_hairs_lifespan, 0.,
                            np.where(out["thermal_stopped"] > root_hairs_lifespan, total,
                                     total * ((out["thermal_started"] - root_hairs_lifespan)
                                              / (out["thermal_started"] - out["thermal_stopped"]))))
            out["dead_root_hairs_number"][selected] = dead[selected]
            living = total - out["dead_root_hairs_number"]
            out["living_root_hairs_number"][selected] = living[selected]
            # Elongation, limited by the resources
            short = selected & (root_hair_length < root_hair_max_length)
            out["root_hair_length"][short] = self.root_hair_elongation(short, root_hair_length[short], (
                out["actual_length_with_hairs"] / length)[short], elapsed[short])
            volume = (power(root_hair_radius, 2) * pi) * out["root_hair_length"] * total
            out["root_hairs_volume"][selected] = volume[selected]
            mass = volume * root_tissue_density
            out["root_hairs_struct_mass"][selected] = mass[selected]
            out["living_root_hairs_struct_mass"][selected] = np.where(total > 0., mass * living / total, 0.)[selected]
            produced = mass - initial_mass
            out["root_hairs_struct_mass_produced"][selected] = produced[selected]
            costs = self.root_hair_costs(produced)
            for index, resource in enumerate(self.resources):
                cost = costs[resource.concentration]
                consumed = np.array(ds.get(resource.consumption_amount), dtype=float)
                consumption_rate = np.array(ds.get(resource.consumption), dtype=float)
                consumed[selected] += cost[selected]
                consumption_rate[selected] += (cost / time_step)[selected]
                ds.set(resource.consumption_amount, consumed)
                ds.set(resource.consumption, consumption_rate)
                if index == 0:                              # the carbon: growth respiration
                    out["resp"][selected] += (cost * 6. * (1 - yield_growth))[selected]
        return (out["root_hair_length"], "actual_length_with_hairs", out["actual_length_with_hairs"],
                "total_root_hairs_number", out["total_root_hairs_number"],
                "dead_root_hairs_number", out["dead_root_hairs_number"],
                "living_root_hairs_number", out["living_root_hairs_number"],
                "all_root_hairs_formed", out["all_root_hairs_formed"],
                "root_hairs_struct_mass", out["root_hairs_struct_mass"],
                "living_root_hairs_struct_mass", out["living_root_hairs_struct_mass"],
                "root_hairs_struct_mass_produced", out["root_hairs_struct_mass_produced"],
                "root_hairs_volume", out["root_hairs_volume"],
                "actual_time_since_root_hairs_emergence_started", out["started"],
                "thermal_time_since_root_hairs_emergence_started", out["thermal_started"],
                "actual_time_since_root_hairs_emergence_stopped", out["stopped"],
                "thermal_time_since_root_hairs_emergence_stopped", out["thermal_stopped"],
                "resp_growth", out["resp"])

    @postsegmentation
    def _living_struct_mass(self, struct_mass, living_root_hairs_struct_mass, vertex_index) -> outputs(1):
        """The living structural mass, and the plant's total, summed in the legacy's order (its vertices' ids)."""
        ds = self.data_structure
        living = struct_mass + living_root_hairs_struct_mass
        owner = np.asarray(ds.owner("Plant"))
        totals = np.zeros(np.asarray(ds.entity_ids("Plant")).size)
        for plant in np.unique(owner):
            nodes = np.flatnonzero(owner == plant)
            totals[plant] = np.sum(living[nodes[np.argsort(vertex_index[nodes], kind="stable")]])
        return living, "total_living_struct_mass", totals
