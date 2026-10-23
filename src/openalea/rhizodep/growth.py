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


def outputs(k):
    """The return annotation of a step with k outputs besides its own: (values, name, values, ...)."""
    return tuple[(np.ndarray,) + (str, np.ndarray) * k]


def temperature_modification(soil_temperature, process_at_T_ref, T_ref, A, B, C):
    """
    The factor of a process at *soil_temperature* (°C), the legacy's Model.temperature_modification: the process at
    T_ref times (A (T - T_ref) + B)^(1 - C) (A (T - T_ref) + B)^(C (T - T_ref) / 10); C = 0 and B = 1: a linear
    increase (thermal time); C = 1, A = 0, B > 1: a Q10; C = 1, A < 0, B > 0: a bell shape (Parent et al., 2010).
    0 when C is neither 0 nor 1, or when C = 1 and A (T - T_ref) + B < 0.
    """
    x = A * (soil_temperature - T_ref) + B
    modified = process_at_T_ref * x ** (1 - C) * x ** (C * (soil_temperature - T_ref) / 10.)
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

    # The legacy's per-element copies of these parameters are the parameters themselves (never changed per element)
    ELEMENT_PARAMETERS = ("root_hair_radius", "root_hairs_lifespan")

    @classmethod
    def plant_parameters(cls, parameters) -> dict:
        """The class's defaults (parameters, and the inputs a scenario may set) overridden by *parameters*."""
        p = {}
        for f in fields(cls):
            kind = f.metadata.get("variable_type")
            if kind in ("parameter", "input") or f.name == "volume":
                p[f.name] = parameters.get(f.name, f.default)
        return p

    @classmethod
    def seedling(cls, p, stream=None) -> list:
        """
        The legacy's initial elements (RootGrowthModel.initiate_mtg, then its initiate_heterogeneous_variables and
        the post-growth they end with) for the parameters *p*: a list, in the legacy's creation order, of dicts
        with "vertex_index", "parent" (its vertex_index or None), "edge" ('<' / '+'), "apex" (a RootApex) and
        "values" ({state variable: value}).
        """
        defaults = {f.name: f.default for f in fields(cls) if f.metadata.get("variable_type") == "state_variable"
                    and f.metadata.get("scale") == SubOrgan}
        elements = []
        nrtd = p["new_root_tissue_density"]

        def add(parent, edge, apex, values):
            element = dict(vertex_index=len(elements) + 1, parent=parent, edge=edge, apex=apex,
                           values={**defaults, **dict.fromkeys(cls.NIL, 0.), **values})
            elements.append(element)
            return element

        def nil_child(parent, edge, apex, type_, angle_down, angle_roll, length, radius):
            return add(parent["vertex_index"], edge, apex,
                       cls._nil_values(p, type_, 1, angle_down, angle_roll, length, radius))

        # The base segment
        r0 = p["D_ini"] / 2.
        length = p["initial_segment_length"]
        segment = add(None, "/", False, dict(
            type=TYPE_BASE, root_order=1, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE, emergence_cost=0.,
            angle_down=0., angle_roll=0., length=length, radius=r0, original_radius=r0, initial_length=length,
            initial_radius=r0, distance_from_tip=length, former_distance_from_tip=length, dist_to_ramif=0.,
            actual_elongation=length, actual_elongation_rate=0., volume=p["volume"],
            struct_mass=p["volume"] * nrtd, initial_struct_mass=p["volume"] * nrtd,
            growth_duration=p["GDs"] * (2. * r0) ** 2 * p["main_roots_growth_extender"],
            life_duration=p["LDs"] * (2. * r0) * nrtd))

        # The seminal, then the nodal (adventitious) roots, each on a zero-length support
        for kind in ("seminal", "adventitious"):
            n = p[f"n_{kind}_roots"]
            forcing = p[f"forcing_{kind}_roots_events"]
            if not (n > (1 if kind == "seminal" else 0) or not forcing):
                continue
            events = cls._emergence_events(p, kind)
            ratio = p["D_sem_to_D_ini_ratio" if kind == "seminal" else "D_adv_to_D_ini_ratio"]
            support_type, apex_type = ((TYPE_SEMINAL_SUPPORT, TYPE_SEMINAL_BEFORE_EMERGENCE) if kind == "seminal" else
                                       (TYPE_ADVENTITIOUS_SUPPORT, TYPE_ADVENTITIOUS_BEFORE_EMERGENCE))
            for i, (delay, number) in enumerate(events):
                for j in range(int(number)):
                    # the legacy's np.random.seed(random_choice + i j (· 3)), on the plant's stream
                    rng = stream if stream is not None else np.random.RandomState()
                    rng.seed(int(p["random_choice"]) + i * j * (1 if kind == "seminal" else 3))
                    segment = nil_child(segment, "<", False, support_type, 0,
                                        abs(rng.normal(180 if kind == "seminal" else 0, 180)), 0., r0)
                    mean = p["D_ini"] / 2. * ratio
                    radius = abs(rng.normal(mean, mean * p["CVDD"])) if p["random"] else mean
                    apex = nil_child(segment, "+", True, apex_type, abs(rng.normal(60, 10)), 5, 0., radius)
                    apex["values"].update(
                        growth_duration=p["GDs"] * (2. * radius) ** 2 * p["main_roots_growth_extender"],
                        life_duration=p["LDs"] * (2. * radius) * nrtd, emergence_delay_in_thermal_time=delay)

        # The primary root's apex
        apex = nil_child(segment, "<", True, TYPE_EMERGED, 0, 0, p["initial_apex_length"], r0)
        apex["values"].update(
            growth_duration=p["GDs"] * (2. * r0) ** 2 * p["main_roots_growth_extender"],
            life_duration=p["LDs"] * (2. * r0) * nrtd,
            initial_struct_mass=p["volume"] * nrtd)

        cls._initial_values(p, elements)
        return elements

    @classmethod
    def _nil_values(cls, p, type_, root_order, angle_down, angle_roll, length, radius) -> dict:
        """A new element's values, RootGrowthModel.ADDING_A_CHILD(nil_properties=True)."""
        nrtd = p["new_root_tissue_density"]
        return {**dict.fromkeys(cls.NIL, 0.), **dict(
            type=type_, root_order=root_order, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE,
            emergence_cost=0., angle_down=angle_down, angle_roll=angle_roll, length=length, radius=radius,
            original_radius=radius, potential_length=0., theoretical_radius=radius, potential_radius=radius,
            initial_length=0., initial_radius=radius, volume=0., struct_mass=0., initial_struct_mass=0.,
            C_hexose_root=0., soil_temperature=7.8, root_tissue_density=nrtd,
            growth_duration=p["GDs"] * (2 * radius) ** 2, life_duration=p["LDs"] * 2. * radius * nrtd,
            temperature_modification=0.)}

    @classmethod
    def _emergence_events(cls, p, kind) -> list:
        """[(emergence delay in thermal time, number of roots)]: the CSV table of input_file_path when it exists and
        the events are not forced, else built from n_<kind>_roots and ER (RootGrowthModel.initiate_mtg)."""
        path = os.path.join(p["input_file_path"], p[f"{kind}_roots_events_file"])
        if os.path.exists(path) and not p[f"forcing_{kind}_roots_events"]:
            table = pd.read_csv(path)
            return list(zip(table.emergence_delay_in_thermal_time, table[f"number_of_{kind}_roots_per_event"]))
        if kind == "seminal":
            return [(x * 1 / p["ER"], 1) for x in range(1, int(p["n_seminal_roots"]))]
        start = p["starting_time_for_adventitious_roots_emergence"]
        return [(start + x * 1 / p["ER"], 1) for x in range(0, int(p["n_adventitious_roots"]))]

    @classmethod
    def _initial_values(cls, p, elements) -> None:
        """What the legacy derives at its initiation: the inputs everywhere, the elements' volume, masses,
        temperature factor, legacy parent, distance from tip and axis apex, the hexose of the apex."""
        by_index = {e["vertex_index"]: e for e in elements}
        children = {e["vertex_index"]: [] for e in elements}
        for e in elements:
            if e["parent"] is not None:
                children[e["parent"]].append(e["vertex_index"])
        # The collar (RootGrowthModel.__init__): the zero-length segments with children are skipped, their children
        # that are not such segments have the base as parent
        zero_segment = {v: (not e["apex"]) and e["values"]["length"] == 0 for v, e in by_index.items()}
        collar_children = {c for v in by_index if zero_segment[v] and children[v] for c in children[v]
                           if not zero_segment[c]}
        factor = float(temperature_modification(p["soil_temperature"], p["process_at_T_ref"], p["T_ref"], p["A"],
                                                p["B"], p["C"]))
        for e in elements:
            v = e["values"]
            v.update(soil_temperature=p["soil_temperature"], root_tissue_density=p["root_tissue_density"],
                     C_hexose_root=0.)
            v["vertex_index"] = e["vertex_index"]
            v["parent_id"] = 1 if e["vertex_index"] in collar_children else (e["parent"] or -1)
            volume = (4 / 3. * pi * v["radius"] ** 3 if v["type"] == TYPE_NODULE else
                      pi * v["radius"] ** 2 * v["length"])
            v["volume"] = volume
            v["struct_mass"] = volume * p["new_root_tissue_density"]
            v["living_struct_mass"] = v["struct_mass"] + v["living_root_hairs_struct_mass"]
            v["temperature_modification"] = factor
        elements[0]["values"]["C_hexose_root"] = p["initial_C_hexose_root"]
        if p["initial_apex_length"] > 0.:
            elements[-1]["values"]["C_hexose_root"] = p["initial_C_hexose_root"]
        # Distance from tip and axis apex: children before parents, along the successors ('<')
        successor = {e["parent"]: e["vertex_index"] for e in elements if e["edge"] == "<"}
        for e in reversed(elements):
            v, son = e["values"], successor.get(e["vertex_index"])
            if e["apex"] or v["type"] == TYPE_NODULE or son is None:
                v["distance_from_tip"], v["axis_apex_id"] = v["length"], e["vertex_index"]
            else:
                s = by_index[son]["values"]
                v["distance_from_tip"], v["axis_apex_id"] = s["distance_from_tip"] + v["length"], s["axis_apex_id"]
            v["former_distance_from_tip"] = v["distance_from_tip"]

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        """The legacy's seedling under *plant* (the class docstring's structure), its values on the MPG."""
        p = cls.plant_parameters(parameters)
        stream = np.random.RandomState()
        elements = cls.seedling(p, stream)
        g.property("legacy_random")[plant] = stream
        s, labels = g.scales, g.labels
        vids = {}

        def new_organ():
            complex_ = plant
            for scale, label in ((s.Axis, labels.Axis.Root), (s.GrowthUnit, labels.GrowthUnit.Root),
                                 (s.Phytomer, labels.Phytomer.Root), (s.Organ, labels.Organ.RootInternode)):
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            return complex_

        for e in elements:
            label = labels.SubOrgan.RootApex if e["apex"] else labels.SubOrgan.RootSegment
            if e["parent"] is None:
                vid = g.add_component(new_organ(), **PropsConfig(scale=s.SubOrgan, edge_type='/', label=label))
            elif e["edge"] == "<":
                vid = g.add_child(vids[e["parent"]], **PropsConfig(scale=s.SubOrgan, edge_type='<', label=label))
            else:
                vid = g.add_component_with_topo(new_organ(), vids[e["parent"]],
                                                **PropsConfig(scale=s.SubOrgan, edge_type='+', label=label))
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

    def _elongated_length(self, initial_length, radius, C, time, EL, Km_elongation, C_hexose_min_for_elongation):
        """RootGrowthModel.elongated_length: C-free with simple_growth_duration, else Michaelis-Menten above the
        threshold (no elongation below)."""
        if self.simple_growth_duration:
            elongation = EL * 2. * radius * time
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                elongation = np.where(C > C_hexose_min_for_elongation,
                                      EL * 2. * radius * C / (Km_elongation + C) * time, 0.)
        return initial_length + elongation

    # ── Potential growth (RootGrowthModel.potential_growth; devplan §14) ───────────────────────────────────────

    @potential
    def _initial_length(self, length, radius, struct_mass, living_root_hairs_struct_mass, soil_temperature,
                        parent_id, vertex_index, process_at_T_ref, T_ref, A, B, C) -> outputs(18):
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
        return (np.array(length), "initial_radius", np.array(radius), "potential_radius", np.array(radius),
                "theoretical_radius", np.array(radius), "potential_length", np.array(length),
                "initial_struct_mass", np.array(struct_mass),
                "initial_living_root_hairs_struct_mass", np.array(living_root_hairs_struct_mass),
                "soil_temperature", soil, "temperature_modification", factor,
                "hexose_consumption_by_growth_amount", zero, "hexose_consumption_by_growth", zero,
                "hexose_possibly_required_for_elongation", zero, "resp_growth", zero, "struct_mass_produced", zero,
                "root_hairs_struct_mass_produced", zero, "hexose_growth_demand", zero, "actual_elongation", zero,
                "actual_elongation_rate", zero, "hexose_consumption_by_fungus", zero)

    @potential
    def _potential_length(self, type, length, radius, initial_length, initial_radius, potential_length, volume,
                          struct_mass, C_hexose_root, temperature_modification, emergence_delay_in_thermal_time,
                          growth_duration, life_duration, thermal_time_since_primordium_formation,
                          actual_time_since_primordium_formation, thermal_potential_time_since_emergence,
                          thermal_time_since_emergence, actual_time_since_emergence,
                          thermal_time_since_growth_stopped, actual_time_since_growth_stopped,
                          thermal_time_since_cells_formation, actual_time_since_cells_formation,
                          thermal_time_since_death, actual_time_since_death, growing_zone_C_hexose_root,
                          hexose_possibly_required_for_elongation, struct_mass_contributing_to_elongation,
                          lateral_root_emergence_possibility, lateral_primordium_index, vertex_index,
                          emergence_delay, growing_zone_factor, EL, Km_elongation,
                          C_hexose_min_for_elongation) -> outputs(17):
        """
        The apices (RootGrowthModel.potential_apex_development): the state machine as masked branches, the supply
        zone of those elongating (calculating_supply_for_elongation, by path windows), their potential length; a
        lateral primordium emerging marks its bearer (the last one visited when several emerge on one bearer).
        """
        ds = self.data_structure
        ts = float(self.time_step)
        apex = self._apex()
        T = temperature_modification
        tsT = ts * T
        type_ = np.array(type)
        ttpf, atpf = np.array(thermal_time_since_primordium_formation), np.array(actual_time_since_primordium_formation)
        tpte = np.array(thermal_potential_time_since_emergence)
        tte, ate = np.array(thermal_time_since_emergence), np.array(actual_time_since_emergence)
        ttgs, atgs = np.array(thermal_time_since_growth_stopped), np.array(actual_time_since_growth_stopped)
        ttcf, atcf = np.array(thermal_time_since_cells_formation), np.array(actual_time_since_cells_formation)
        ttd, atd = np.array(thermal_time_since_death), np.array(actual_time_since_death)
        potential = np.array(potential_length)

        # The branches, on the state at the step's start
        primordium = apex & ((type == TYPE_SEMINAL_BEFORE_EMERGENCE) | (type == TYPE_ADVENTITIOUS_BEFORE_EMERGENCE))
        lateral = apex & (type == TYPE_BEFORE_EMERGENCE)
        other = apex & ~primordium & ~lateral
        primordium_emerges = primordium & (thermal_time_since_primordium_formation + tsT
                                           >= emergence_delay_in_thermal_time)
        lateral_emerges = lateral & (thermal_time_since_primordium_formation + tsT > emergence_delay)
        growing = other & (thermal_time_since_emergence + tsT < growth_duration)
        living = other & ~growing & (thermal_time_since_growth_stopped + tsT < life_duration)
        stopped = living & ((type == TYPE_STOPPED) | (type == TYPE_JUST_STOPPED))
        just_stopped = living & ~stopped
        dying = other & ~growing & ~living
        dead = dying & ((type == TYPE_DEAD) | (type == TYPE_JUST_DEAD))
        just_dead = dying & ~dead

        # The supply zone of the apices that elongate (order-free: it reads what this loop does not change)
        supplied = primordium_emerges | lateral_emerges | growing | just_stopped
        budget = growing_zone_factor * radius * radius ** 2 * pi
        include = length > 0.
        hexose = ds.path_window(budget, volume, C_hexose_root * struct_mass, where=supplied, include=include)
        mass = ds.path_window(budget, volume, struct_mass, where=supplied, include=include)
        required, contributing = np.array(hexose_possibly_required_for_elongation), np.array(
            struct_mass_contributing_to_elongation)
        zone_C = np.array(growing_zone_C_hexose_root)
        required[supplied], contributing[supplied] = hexose[supplied], mass[supplied]
        with np.errstate(divide="ignore", invalid="ignore"):
            zone_C[supplied] = np.where(mass[supplied] > 0., hexose[supplied] / mass[supplied], 0.)
        elongation = (EL, Km_elongation, C_hexose_min_for_elongation)

        # Seminal and nodal primordia
        m = primordium_emerges
        tpte[m] = thermal_time_since_primordium_formation[m] + tsT[m] - emergence_delay_in_thermal_time[m]
        tpte[m] = np.where(tpte[m] > tsT[m], tsT[m], tpte[m])
        potential[m] = self._elongated_length(initial_length[m], initial_radius[m], zone_C[m], tpte[m],
                                              *(x[m] for x in elongation))
        if self.simple_growth_duration:
            type_[m] = TYPE_EMERGED
        atpf[primordium] += ts
        ttpf[primordium] += tsT[primordium]

        # Lateral primordia
        atpf[lateral] += ts
        ttpf[lateral] += tsT[lateral]
        m = lateral_emerges
        tpte[m] = ttpf[m] - emergence_delay[m]
        tpte[m] = np.where(tpte[m] > tsT[m], tsT[m], tpte[m])
        potential[m] = self._elongated_length(initial_length[m], initial_radius[m], zone_C[m], tpte[m],
                                              *(x[m] for x in elongation))
        possibility, primordium_index = np.array(lateral_root_emergence_possibility), np.array(
            lateral_primordium_index)
        if self.simple_growth_duration:
            type_[m] = TYPE_EMERGED
        else:
            visit = ds.order("post", convention="openalea")
            emerging = visit[m[visit]]
            bearers = np.asarray(ds.parents())[emerging]
            last = bearers.size - 1 - np.unique(bearers[::-1], return_index=True)[1]   # last visited per bearer
            possibility[bearers[last]] = EMERGENCE_POSSIBLE
            primordium_index[bearers[last]] = vertex_index[emerging[last]]

        # Growing apices
        m = growing
        atpf[m] += ts; ttpf[m] += tsT[m]; ate[m] += ts; tte[m] += tsT[m]
        potential[m] = self._elongated_length(length[m], radius[m], zone_C[m], tsT[m], *(x[m] for x in elongation))

        # Stopped
        m = stopped
        atgs[m] += ts; ttgs[m] += tsT[m]
        type_[m] = TYPE_STOPPED
        atpf[m] += ts; ttpf[m] += tsT[m]; ate[m] += ts; tte[m] += tsT[m]; atcf[m] += ts; ttcf[m] += tsT[m]

        # Just stopped: the elongation until the growth duration
        m = just_stopped
        type_[m] = TYPE_JUST_STOPPED
        ttgs[m] = thermal_time_since_emergence[m] + tsT[m] - growth_duration[m]
        with np.errstate(divide="ignore", invalid="ignore"):
            atgs[m] = ttgs[m] / T[m]
        remaining = tsT[m] - ttgs[m]
        potential[m] = np.where(remaining < 0., initial_length[m],
                                self._elongated_length(length[m], radius[m], zone_C[m], remaining,
                                                       *(x[m] for x in elongation)))
        atpf[m] += ts; ate[m] += ts; ttpf[m] += tsT[m]; tte[m] += tsT[m]; atcf[m] += ts; ttcf[m] += tsT[m]

        # Dead
        m = dead
        type_[m] = TYPE_DEAD
        for actual in (atpf, ate, atcf, atgs, atd):
            actual[m] += ts
        for thermal in (ttpf, tte, ttcf, ttgs, ttd):
            thermal[m] += tsT[m]

        # Just dead
        m = just_dead
        type_[m] = TYPE_JUST_DEAD
        ttd[m] = thermal_time_since_growth_stopped[m] + tsT[m] - life_duration[m]
        with np.errstate(divide="ignore", invalid="ignore"):
            atd[m] = ttd[m] / T[m]
        for actual in (atpf, ate, atcf, atgs):
            actual[m] += ts
        for thermal in (ttpf, tte, ttcf, ttgs):
            thermal[m] += tsT[m]

        return (potential, "type", type_, "thermal_potential_time_since_emergence", tpte,
                "thermal_time_since_primordium_formation", ttpf, "actual_time_since_primordium_formation", atpf,
                "thermal_time_since_emergence", tte, "actual_time_since_emergence", ate,
                "thermal_time_since_growth_stopped", ttgs, "actual_time_since_growth_stopped", atgs,
                "thermal_time_since_cells_formation", ttcf, "actual_time_since_cells_formation", atcf,
                "thermal_time_since_death", ttd, "actual_time_since_death", atd,
                "growing_zone_C_hexose_root", zone_C, "hexose_possibly_required_for_elongation", required,
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
                            relative_root_thickening_rate_max, Km_thickening, relative_nodule_thickening_rate_max,
                            Km_nodule_thickening, nodule_max_radius) -> outputs(14):
        """
        The segments (RootGrowthModel.potential_segment_development): the type of their axis apex (stopped), death
        when all their children are dead, the pipe model, children first (a fold up the tree), then the potential
        radius limited by hexose, the supports' radius, the times; the nodules' thickening.
        """
        ds = self.data_structure
        ts = float(self.time_step)
        T = temperature_modification
        tsT = ts * T
        apex = self._apex()
        parents = np.asarray(ds.parents())
        segment = ~apex
        nodule = segment & (type == TYPE_NODULE)
        normal = segment & ~nodule
        theoretical, potential_r, radius_ = np.array(theoretical_radius), np.array(potential_radius), np.array(radius)
        available = C_hexose_root * struct_mass

        # Nodules
        m = nodule
        if m.any():
            p = parents[m]
            available[m] = C_hexose_root[p] * struct_mass[p] + C_hexose_root[m] * struct_mass[m]
            regulating = available[m] / (struct_mass[p] + struct_mass[m])
            rate = relative_nodule_thickening_rate_max[m] * regulating / (Km_nodule_thickening[m] + regulating)
            rate = rate * T[m]
            theoretical[m] = radius[m] * (1 + rate * ts)
            potential_r[m] = np.where(theoretical[m] > nodule_max_radius[m], nodule_max_radius[m], theoretical[m])

        # The axis apex's stop
        type_ = np.array(type)
        node_of = self._node_of(vertex_index)
        apex_node = node_of[np.clip(np.asarray(axis_apex_id, dtype=np.int64), 0, node_of.size - 1)]
        apex_type = np.where(apex_node >= 0, type[np.maximum(apex_node, 0)], -1)
        type_[normal & (apex_type == TYPE_JUST_STOPPED)] = TYPE_JUST_STOPPED
        type_[normal & (apex_type == TYPE_STOPPED)] = TYPE_STOPPED

        # Children first: death, time since death, the pipe model
        radial, simple = self.radial_growth, self.simple_growth_duration
        columns = np.column_stack([theoretical, type_.astype(float), np.array(actual_time_since_death, dtype=float)])

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
            own = rows[:, 1]
            all_dead = todo & (n_dead == n_children)
            new_type = np.where(all_dead, np.where((own == TYPE_JUST_DEAD) | (own == TYPE_DEAD), TYPE_DEAD,
                                                   TYPE_JUST_DEAD), own)
            if radial:
                if self.legacy_pipe_model:
                    son = laterals = np.zeros(nodes.size)
                else:
                    section = out[:, 0] ** 2 * pi
                    son = level.children(section, op="sum", edge="<")
                    laterals = level.children(section, op="sum", edge="+",
                                              where=(length > 0.) & (child_type != TYPE_NODULE))
                th = np.sqrt(son / pi + SGC[nodes] * laterals / pi)
                th = np.where((th - initial_radius[nodes]) <= 0.001 * initial_radius[nodes], initial_radius[nodes], th)
                rows[todo, 0] = th[todo]
            rows[todo, 1] = new_type[todo]
            death = rows[:, 2]
            death = np.where(todo & (new_type == TYPE_JUST_DEAD), np.where(n_children > 0, first_death, 0.), death)
            death = np.where(todo & (new_type == TYPE_DEAD), death + ts, death)
            rows[:, 2] = death
            return rows

        folded = ds.fold(update, columns, direction="up")
        theoretical[normal] = folded[normal, 0]
        type_[normal] = folded[normal, 1].astype(type_.dtype)
        atd, ttd = np.array(actual_time_since_death), np.array(thermal_time_since_death)
        atd[normal] = folded[normal, 2]

        # The potential radius
        if radial:
            if simple:
                potential_r[normal] = theoretical[normal]
            else:
                m = normal & (length > 0.) & (theoretical > radius) & (C_hexose_root > C_hexose_min_for_thickening)
                rate = relative_root_thickening_rate_max[m] * C_hexose_root[m] / (Km_thickening[m] + C_hexose_root[m])
                rate = rate * T[m]
                largest = (1 + rate * ts) * initial_radius[m]
                potential_r[m] = np.where(theoretical[m] > largest, largest, theoretical[m])
            support = normal & ((type_ == TYPE_SEMINAL_SUPPORT) | (type_ == TYPE_ADVENTITIOUS_SUPPORT))
            radius_[support] = potential_r[support]

        # The times
        atpf, ttpf = np.array(actual_time_since_primordium_formation), np.array(thermal_time_since_primordium_formation)
        ate, tte = np.array(actual_time_since_emergence), np.array(thermal_time_since_emergence)
        atcf, ttcf = np.array(actual_time_since_cells_formation), np.array(thermal_time_since_cells_formation)
        atgs, ttgs = np.array(actual_time_since_growth_stopped), np.array(thermal_time_since_growth_stopped)
        m = normal
        for actual in (atpf, ate, atcf):
            actual[m] += ts
        for thermal in (ttpf, tte, ttcf):
            thermal[m] += tsT[m]
        m = normal & (type_ == TYPE_JUST_STOPPED)
        source = actual_time_since_growth_stopped[np.maximum(apex_node, 0)]
        atgs[m] = source[m]
        ttgs[m] = source[m] * T[m]
        m = normal & ((type_ == TYPE_STOPPED) | (type_ == TYPE_JUST_DEAD) | (type_ == TYPE_DEAD))
        atgs[m] += ts
        ttgs[m] += tsT[m]
        m = normal & (type_ == TYPE_JUST_DEAD)
        ttd[m] = atd[m] * T[m]
        m = normal & (type_ == TYPE_DEAD)
        ttd[m] += tsT[m]

        return (theoretical, "type", type_, "potential_radius", potential_r, "radius", radius_,
                "hexose_available_for_thickening", available,
                "actual_time_since_primordium_formation", atpf, "thermal_time_since_primordium_formation", ttpf,
                "actual_time_since_emergence", ate, "thermal_time_since_emergence", tte,
                "actual_time_since_cells_formation", atcf, "thermal_time_since_cells_formation", ttcf,
                "actual_time_since_growth_stopped", atgs, "thermal_time_since_growth_stopped", ttgs,
                "actual_time_since_death", atd, "thermal_time_since_death", ttd)

    # ── Actual growth (RootGrowthModel.actual_growth_and_corresponding_respiration) ───────────────────────────

    @staticmethod
    def _volume(type, radius, length):
        """RootGrowthModel.volume_from_radius_and_length: a cylinder, a sphere for a nodule."""
        return np.where(type == TYPE_NODULE, 4 / 3. * pi * radius ** 3, pi * radius ** 2 * length)

    @actual
    def _length(self, type, length, radius, initial_length, initial_radius, potential_length, potential_radius,
                volume, struct_mass, initial_struct_mass, root_tissue_density, C_hexose_root, temperature_modification,
                hexose_possibly_required_for_elongation, struct_mass_contributing_to_elongation,
                hexose_available_for_thickening, hexose_growth_demand, hexose_consumption_by_growth_amount,
                hexose_consumption_by_growth, resp_growth, struct_mass_produced,
                thermal_potential_time_since_emergence, thermal_time_since_emergence, actual_time_since_emergence,
                actual_elongation, actual_elongation_rate, dist_to_ramif, growing_zone_factor, struct_mass_C_content,
                yield_growth, C_hexose_min_for_elongation, C_hexose_min_for_thickening) -> outputs(16):
        """
        Growth limited by the hexose of the supply zone (elongation) and of the element (thickening), its cost
        taken from the suppliers in proportion to their hexose, plus the growth respiration; the primordia that
        elongate emerge. The costs are scattered in the legacy's visiting order (children first), so that each
        element's sums are the legacy's, bit for bit.
        """
        ds = self.data_structure
        ts = float(self.time_step)
        rtd, Cc, y = root_tissue_density, struct_mass_C_content, yield_growth
        volume_of = self._volume
        with np.errstate(divide="ignore", invalid="ignore"):
            initial_volume = volume_of(type, initial_radius, initial_length)
            potential_volume = volume_of(type, potential_radius, potential_length)
            grows = ~np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD, TYPE_SEMINAL_SUPPORT, TYPE_ADVENTITIOUS_SUPPORT))
            grows &= ~((potential_length <= initial_length) & (potential_radius <= initial_radius))
            demand = np.array(hexose_growth_demand)
            demand[grows] = ((potential_volume - initial_volume) * rtd * Cc / y * 1 / 6.)[grows]
            negative = grows & (demand < 0.)
            demand[negative] = 0.
            grows &= demand > 0.

            elongates = grows & (potential_length > length)
            for_elongation = np.where(elongates, np.maximum(hexose_possibly_required_for_elongation
                                                            - struct_mass_contributing_to_elongation
                                                            * C_hexose_min_for_elongation, 0.), 0.)
            thickens = grows & (potential_radius > radius)
            for_thickening = np.where(thickens, np.maximum(hexose_available_for_thickening
                                                           - struct_mass * C_hexose_min_for_thickening, 0.), 0.)
            grows &= (for_elongation + for_thickening) > 0.

            # Elongation, limited by the supply zone's hexose
            length_, radius_ = np.array(length), np.array(radius)
            volume_max = initial_volume + for_elongation * 6. / (rtd * Cc) * y
            length_max = volume_max / (pi * initial_radius ** 2)
            m = grows & (potential_length > initial_length)
            elongating = m
            length_[m] = np.where(potential_length[m] >= length_max[m], length_max[m], potential_length[m])
            elongation_cost = 1. / 6. * (volume_of(type, initial_radius, length_) - initial_volume) * rtd * Cc / y
            supplied = m & (length_ > initial_length)

            # Thickening, limited by the element's hexose
            m = grows & (potential_radius > initial_radius)
            increase = for_thickening * 6. * y / (rtd * Cc)
            volume_max = volume_of(type, initial_radius, length_) + increase
            possible = np.where(type == TYPE_NODULE, (3. / (4. * pi)) ** (1. / 3.), np.sqrt(volume_max / (length_ * pi)))
            fits = possible <= potential_radius
            radius_[m] = np.where(fits[m], possible[m], potential_radius[m])
            net = volume_of(type, radius_, length_) - volume_of(type, initial_radius, length_)
            thickening_cost = np.where(fits, for_thickening, 1. / 6. * net * rtd * Cc / y)
            own_share = thickening_cost * ((C_hexose_root * initial_struct_mass) / for_thickening)
            parents = np.asarray(ds.parents())
            nodule = m & (type == TYPE_NODULE)
            parent_share = thickening_cost * ((C_hexose_root[parents] * initial_struct_mass[parents]) / for_thickening)

        # The costs, in the legacy's order: per element visited (children first), its elongation's suppliers (tip
        # to base), then its own thickening, then a nodule's parent
        visit = ds.order("post", convention="openalea")
        rank = np.empty(visit.size, dtype=np.int64)
        rank[visit] = np.arange(visit.size)
        budget = growing_zone_factor * radius * radius ** 2 * pi
        owner, supplier, contribution = ds.path_contributions(budget, volume, C_hexose_root * struct_mass,
                                                              where=supplied, include=length > 0.)
        with np.errstate(divide="ignore", invalid="ignore"):
            elongation_events = elongation_cost[owner] * contribution / hexose_possibly_required_for_elongation[owner]
        own = np.flatnonzero(m)
        nod = np.flatnonzero(nodule)
        event_owner = np.concatenate([owner, own, nod])
        target = np.concatenate([supplier, own, parents[nod]])
        value = np.concatenate([elongation_events, own_share[own], parent_share[nod]])
        kind = np.concatenate([np.zeros(owner.size), np.ones(own.size), np.full(nod.size, 2.)])
        order = np.lexsort((np.arange(event_owner.size), kind, rank[event_owner]))
        target, value = target[order], value[order]
        amount, rate, respiration = (np.array(hexose_consumption_by_growth_amount),
                                     np.array(hexose_consumption_by_growth), np.array(resp_growth))
        np.add.at(amount, target, value)
        np.add.at(rate, target, value / ts)
        np.add.at(respiration, target, value * (1 - y[target]) * 6.)

        # The new structure
        volume_, mass, produced = np.array(volume), np.array(struct_mass), np.array(struct_mass_produced)
        initial_mass = np.array(initial_struct_mass)
        g = grows
        volume_[g] = volume_of(type, radius_, length_)[g]
        mass[g] = volume_[g] * rtd[g]
        produced[g] = (volume_[g] - initial_volume[g]) * rtd[g]
        shrinks = g & (mass < initial_struct_mass) & (produced > 0.)
        initial_mass[shrinks] = mass[shrinks]

        # Emergence, elongation
        type_, tte, ate = np.array(type), np.array(thermal_time_since_emergence), np.array(actual_time_since_emergence)
        elongation, elongation_rate, ramif = np.array(actual_elongation), np.array(actual_elongation_rate), np.array(
            dist_to_ramif)
        longer = g & (length_ > initial_length)
        emerges = longer & np.isin(type, (TYPE_SEMINAL_BEFORE_EMERGENCE, TYPE_ADVENTITIOUS_BEFORE_EMERGENCE,
                                          TYPE_BEFORE_EMERGENCE))
        type_[emerges] = TYPE_EMERGED
        tte[emerges] = thermal_potential_time_since_emergence[emerges]
        with np.errstate(divide="ignore", invalid="ignore"):
            ate[emerges] = tte[emerges] / temperature_modification[emerges]
            elongation[emerges] = length_[emerges] - initial_length[emerges]
            elongation_rate[emerges] = elongation[emerges] / ate[emerges]
        emerged = longer & (type == TYPE_EMERGED)
        elongation[emerged] = length_[emerged] - initial_length[emerged]
        elongation_rate[emerged] = elongation[emerged] / ts
        ramif[longer] += elongation[longer]

        return (length_, "radius", radius_, "volume", volume_, "struct_mass", mass, "struct_mass_produced", produced,
                "initial_struct_mass", initial_mass, "hexose_growth_demand", demand,
                "hexose_consumption_by_growth_amount", amount, "hexose_consumption_by_growth", rate,
                "resp_growth", respiration, "type", type_, "thermal_time_since_emergence", tte,
                "actual_time_since_emergence", ate, "actual_elongation", elongation,
                "actual_elongation_rate", elongation_rate, "dist_to_ramif", ramif,
                "step_elongating", elongating.astype(int))

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
    INPUTS_OF_NEW = ("soil_temperature", "root_tissue_density", "C_hexose_root")
    SEGMENTATION_PARAMETERS = ("segment_length", "random_choice", "IPD", "Dmin", "RMD", "CVDD", "root_order_treshold",
                               "GDs", "LDs", "main_roots_growth_extender", "new_root_tissue_density", "GD_highest",
                               "GD_high", "GD_medium", "GD_low", "GD_prob_low", "GD_prob_medium",
                               "nodule_formation_probability", "struct_mass_C_content")

    @segmentation
    def _segmentation_and_primordia_formation(self):
        ds = self.data_structure
        g = self.mtg
        P = g.properties()
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
            context = dict(p={name: values[name][apex] for name in values}, plant=plant, next_index=next_index,
                           stream=streams[plant])
            new = self._segment_apex(g, P, apex, context)
            for v in new:
                if v not in self.__dict__["_step_new_apices"]:
                    self.__dict__["_step_new_apices"].append(v)

    def _new_element(self, g, P, mother, edge, apex, values, context):
        """A new element under *mother* ('<' in its Organ; '+' a new Organ of its Axis, a lateral or a nodule), its
        legacy id the plant's next one, and *values* on the MPG (its inputs kept for _inputs_of_new_elements)."""
        s, labels = g.scales, g.labels
        label = labels.SubOrgan.RootApex if apex else labels.SubOrgan.RootSegment
        if edge == "<":
            vid = g.add_child(mother, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=label))
        else:
            organ = g.add_component(g.complex_at_scale(mother, s.Phytomer),
                                    **PropsConfig(scale=s.Organ, edge_type='/', label=labels.Organ.RootInternode))
            vid = g.add_component_with_topo(organ, mother, **PropsConfig(scale=s.SubOrgan, edge_type='+',
                                                                         label=label))
        plant = context["plant"]
        P["vertex_index"][vid] = context["next_index"][plant]
        P["parent_id"][vid] = P["vertex_index"][mother]
        context["next_index"][plant] += 1
        inputs = {}
        for name, value in values.items():
            if name in self.INPUTS_OF_NEW:
                inputs[name] = value
            elif name not in self.ELEMENT_PARAMETERS:
                P[name][vid] = value
        self.__dict__["_new_inputs"][vid] = inputs
        return vid

    def _value(self, P, name, vid):
        """An element's value, its pending one for a new element's input."""
        pending = self.__dict__["_new_inputs"].get(vid)
        if pending is not None and name in pending:
            return pending[name]
        return P[name][vid]

    def _copy_child(self, g, P, mother, type_, root_order, angle_down, angle_roll, length, context):
        """RootGrowthModel.ADDING_A_CHILD('<', label Apex, identical_properties=True)."""
        radius = P["radius"][mother]
        values = {name: self._value(P, name, mother) for name in self.COPIED}
        values.update(type=type_, root_order=root_order, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE,
                      emergence_cost=0., angle_down=angle_down, angle_roll=angle_roll, length=length,
                      radius=radius, original_radius=radius, potential_length=length, initial_length=length,
                      initial_radius=radius, volume=0.)
        return self._new_element(g, P, mother, "<", True, values, context)

    def _segment_apex(self, g, P, apex, context) -> list:
        """RootGrowthModel.segmentation_and_primordium_formation: the elements it creates or returns."""
        p, rng = context["p"], context["stream"]
        ts = float(self.time_step)
        T = P["temperature_modification"][apex]
        legacy_id = int(P["vertex_index"][apex])
        if self.random:
            rng.seed(int(p["random_choice"]) * legacy_id)
            segment_angle_down = rng.normal(0, 5)
            segment_angle_roll = rng.normal(0, 5)
            rng.normal(0, 5)                    # the legacy's apex_angle_down and apex_angle_roll, unused
            rng.normal(0, 5)
        else:
            segment_angle_down = segment_angle_roll = 0
        new = []
        root_order = P["root_order"][apex]
        initial_length = P["length"][apex]
        initial_dist_to_ramif = P["dist_to_ramif"][apex]
        initial_elongation = P["actual_elongation"][apex]
        initial_elongation_rate = P["actual_elongation_rate"][apex]
        shared = {name: P[name][apex] for name in self.SHARED}
        # (g1) the legacy overwrites its initial_struct_mass and initial_living_root_hairs_struct_mass with the apex's
        # values before growth: the mass fraction is relative to the mass before growth, and the living root hairs'
        # mass is shared from their mass before growth
        initial_struct_mass = P["initial_struct_mass"][apex]
        shared["living_root_hairs_struct_mass"] = P["initial_living_root_hairs_struct_mass"][apex]
        segment_length = p["segment_length"]

        if initial_length <= segment_length:
            new += self._primordium_formation(g, P, apex, initial_elongation_rate, context)
            rate = P["actual_elongation_rate"][apex]
            if rate > 0.:
                age_non_elongated = P["actual_time_since_cells_formation"][apex] + ts
                age_up = P["actual_elongation"][apex] / rate
                age_elongated = 0.5 * (0 + age_up)
                length, length0 = P["length"][apex], P["initial_length"][apex]
                P["actual_time_since_cells_formation"][apex] = (length0 * age_non_elongated
                                                                + (length - length0) * age_elongated) / length
                thermal_non_elongated = P["thermal_time_since_cells_formation"][apex] + ts * T
                thermal_elongated = age_elongated * T
                P["thermal_time_since_cells_formation"][apex] = (length0 * thermal_non_elongated
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
            P["length"][element] = segment_length
            P["dist_to_ramif"][element] = initial_dist_to_ramif - (initial_length - segment_length * i)
            self._share_mass(P, element, segment_length, initial_struct_mass, shared)
            rate = P["actual_elongation_rate"][element]
            if i == 1:
                age_non_elongated = P["actual_time_since_cells_formation"][element] + ts
                age_up = P["actual_elongation"][element] / rate
                age_down = (final_apex_length + (n_segments - 1) * segment_length) / rate
                age_elongated = 0.5 * (age_down + age_up)
                length, length0 = P["length"][element], P["initial_length"][element]
                P["actual_time_since_cells_formation"][element] = (length0 * age_non_elongated + (
                    length - length0) * age_elongated) / length
                thermal_non_elongated = P["thermal_time_since_cells_formation"][element] + ts * T
                thermal_elongated = age_elongated * T
                P["thermal_time_since_cells_formation"][element] = (length0 * thermal_non_elongated + (
                    length - length0) * thermal_elongated) / length
            else:
                age_up = (final_apex_length + (n_segments - i) * segment_length) / rate
                age_down = (final_apex_length + (n_segments - 1 - i) * segment_length) / rate
                P["actual_time_since_cells_formation"][element] = 0.5 * (age_down + age_up)
                P["thermal_time_since_cells_formation"][element] = P["actual_time_since_cells_formation"][element] * T
            new += self._primordium_formation(g, P, element, initial_elongation_rate, context)
            P["label"][element] = g.labels.SubOrgan.RootSegment
            if i < n_segments:
                element = self._copy_child(g, P, element, P["type"][element], root_order, segment_angle_down,
                                           segment_angle_roll, 0., context)
                P["actual_elongation"][element] = segment_length * i
            elif self.nodules and len(g.children(element)) < 2 and rng.random_sample() < p["nodule_formation_probability"]:
                self._nodule_formation(g, P, element, context)

        new_length = initial_length - n_segments * segment_length
        element = self._copy_child(g, P, element, P["type"][element], root_order, segment_angle_down,
                                   segment_angle_roll, new_length, context)
        P["potential_length"][element] = new_length
        P["initial_length"][element] = new_length
        P["dist_to_ramif"][element] = initial_dist_to_ramif
        P["actual_elongation"][element] = initial_elongation
        self._share_mass(P, element, new_length, initial_struct_mass, shared)
        age_up = final_apex_length / P["actual_elongation_rate"][element]
        P["actual_time_since_cells_formation"][element] = 0.5 * (0 + age_up)
        P["thermal_time_since_cells_formation"][element] = P["actual_time_since_cells_formation"][element] * T
        new += self._primordium_formation(g, P, element, initial_elongation_rate, context)
        new.append(element)
        return new

    def _share_mass(self, P, element, length, initial_struct_mass, shared):
        """A segment's (or the new apex's) volume and mass, and its share of the apex's amounts by its mass."""
        radius = P["radius"][element]
        volume = pi * radius ** 2 * length if P["type"][element] != TYPE_NODULE else 4 / 3. * pi * radius ** 3
        P["volume"][element] = volume
        P["struct_mass"][element] = volume * self._value(P, "root_tissue_density", element)
        fraction = P["struct_mass"][element] / initial_struct_mass
        for name, value in shared.items():
            P[name][element] = value * fraction

    def _primordium_formation(self, g, P, apex, elongation_rate, context) -> list:
        """RootGrowthModel.primordium_formation: a lateral primordium on *apex*, when its distance to the last one,
        its radius and its order allow."""
        p, rng = context["p"], context["stream"]
        if len(g.children(apex)) >= 1:
            return [apex]
        root_order = P["root_order"][apex]
        if self.root_order_limitation and root_order + 1 > p["root_order_treshold"]:
            return [apex]
        radius = P["radius"][apex]
        Dmin = p["Dmin"]
        mean = (radius - Dmin / 2.) * p["RMD"] + Dmin / 2.
        if self.random:
            rng.seed(int(p["random_choice"]) * int(P["vertex_index"][apex]))
            potential_radius = rng.normal((radius - Dmin / 2.) * p["RMD"] + Dmin / 2.,
                                          ((radius - Dmin / 2.) * p["RMD"] + Dmin / 2.) * p["CVDD"])
            apex_angle_roll = abs(rng.normal(120, 10))
            primordium_angle_down = abs(rng.normal(45, 10)) if root_order == 1 else abs(rng.normal(70, 10))
            primordium_angle_roll = abs(rng.normal(5, 5))
        else:
            potential_radius = (radius - Dmin / 2) * p["RMD"] + Dmin / 2.
            apex_angle_roll = 120
            primordium_angle_down = 45 if root_order == 1 else 70
            primordium_angle_roll = 5
        if potential_radius > radius:
            potential_radius = radius
        dist_to_ramif = P["dist_to_ramif"][apex]
        if not (dist_to_ramif > p["IPD"] and potential_radius >= Dmin and potential_radius <= radius):
            return []
        elongation_since_last_ramif = dist_to_ramif - p["IPD"]
        P["angle_roll"][apex] = apex_angle_roll
        if P["actual_elongation"][apex] > 0 and elongation_rate > 0.:
            time_since_formation = elongation_since_last_ramif / elongation_rate
        else:
            time_since_formation = 0.
        values = self._nil_values(p, TYPE_BEFORE_EMERGENCE, root_order + 1, primordium_angle_down,
                                  primordium_angle_roll, 0., potential_radius)
        ramif = self._new_element(g, P, apex, "+", True, values, context)
        if self.simple_growth_duration:
            P["growth_duration"][ramif] = p["GDs"] * (2. * potential_radius) ** 2 * p["main_roots_growth_extender"]
        else:
            P["growth_duration"][ramif] = self._growth_duration(potential_radius, int(P["vertex_index"][ramif]),
                                                                root_order + 1, p, rng)
        P["actual_time_since_primordium_formation"][ramif] = time_since_formation
        P["thermal_time_since_primordium_formation"][ramif] = (time_since_formation
                                                               * P["temperature_modification"][apex])
        P["dist_to_ramif"][apex] = elongation_since_last_ramif
        return [apex, ramif]

    def _growth_duration(self, radius, legacy_id, root_order, p, rng):
        """RootGrowthModel.calculate_growth_duration (not simple_growth_duration), with its if/if (g4)."""
        rng.seed(int(p["random_choice"]) * legacy_id)
        draw = rng.random_sample()
        if root_order == 1:
            return p["GD_highest"]
        if self.GD_by_frequency:
            if draw < p["GD_prob_low"]:
                duration = rng.uniform(0., p["GD_low"])
            if draw < p["GD_prob_medium"]:
                duration = rng.uniform(p["GD_low"], p["GD_medium"])
            else:
                duration = rng.uniform(p["GD_medium"], p["GD_high"])
            return duration
        if draw < p["GD_prob_low"]:
            return p["GD_low"]
        if draw < p["GD_prob_medium"]:
            return p["GD_medium"]
        if radius > 2 * p["Dmin"] / 2.:
            return p["GD_highest"]
        raise ValueError(f"no growth duration for a radius of {radius} m (the legacy's calculate_growth_duration "
                         "leaves it unassigned, g4)")

    def _nodule_formation(self, g, P, mother, context):
        """RootGrowthModel.nodule_formation: a nodule, a lateral segment without apex."""
        p = context["p"]
        values = self._nil_values(p, TYPE_NODULE, P["root_order"][mother] + 1, 90, 0, 0, 0)
        radius = P["radius"][mother]
        values.update(length=radius, radius=radius, original_radius=radius)
        values["volume"] = 4 / 3. * pi * radius ** 3
        values["struct_mass"] = values["volume"] * values["root_tissue_density"] * p["struct_mass_C_content"]
        return self._new_element(g, P, mother, "+", False, values, context)

    @segmentation
    def _soil_temperature(self, soil_temperature, root_tissue_density, C_hexose_root) -> outputs(2):
        """The inputs of the elements segmentation created: copied from their mother, or the legacy's values for a
        primordium (7.8 °C, g3; new_root_tissue_density; no hexose)."""
        pending = self.__dict__.get("_new_inputs") or {}
        soil, density, C = np.array(soil_temperature), np.array(root_tissue_density), np.array(C_hexose_root)
        if pending:
            ds = self.data_structure
            index = dict(zip(np.asarray(ds.entity_ids("node")).tolist(), range(soil.size)))
            for vid, inputs in pending.items():
                i = index[vid]
                soil[i] = inputs.get("soil_temperature", soil[i])
                density[i] = inputs.get("root_tissue_density", density[i])
                C[i] = inputs.get("C_hexose_root", C[i])
        self.__dict__["_new_inputs"] = {}
        return soil, "root_tissue_density", density, "C_hexose_root", C

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

    @postsegmentation
    def _root_hair_length(self, type, length, radius, struct_mass, distance_from_tip,
                          former_distance_from_tip, temperature_modification, C_hexose_root, root_tissue_density,
                          root_hair_length, actual_length_with_hairs, total_root_hairs_number, dead_root_hairs_number,
                          living_root_hairs_number, all_root_hairs_formed, root_hairs_struct_mass,
                          living_root_hairs_struct_mass, root_hairs_struct_mass_produced, root_hairs_volume,
                          actual_time_since_root_hairs_emergence_started,
                          thermal_time_since_root_hairs_emergence_started,
                          actual_time_since_root_hairs_emergence_stopped,
                          thermal_time_since_root_hairs_emergence_stopped, hexose_consumption_by_growth_amount,
                          hexose_consumption_by_growth, resp_growth, growing_zone_factor, root_hairs_density,
                          root_hair_max_length, root_hairs_elongation_rate, root_hair_radius, root_hairs_lifespan,
                          Km_elongation, struct_mass_C_content, yield_growth) -> outputs(16):
        """RootGrowthModel.root_hairs_dynamics_opt on the elements of the legacy's focus (a structural mass, a
        Segment or Apex label, an emerged, stopped, dead or nodule type; not a collar's zero-length support):
        emergence along the element beyond the growing zone, death after their lifespan, their elongation limited
        by hexose, their mass and its cost."""
        ds = self.data_structure
        ts = float(self.time_step)
        gz = growing_zone_factor * radius
        focus = (struct_mass > 0.) & np.isin(type, (TYPE_BASE, TYPE_EMERGED, TYPE_STOPPED, TYPE_JUST_STOPPED,
                                                    TYPE_DEAD, TYPE_JUST_DEAD, TYPE_NODULE))
        indptr, _ = ds.children()
        collar = ~self._apex() & (length == 0.) & (np.diff(np.asarray(indptr)) > 0)   # the legacy's collar_skip
        m = focus & ~collar & ~np.isin(type, (TYPE_JUST_DEAD, TYPE_DEAD, TYPE_NODULE)) & (distance_from_tip > gz)
        adjustment = np.maximum(1e-3, temperature_modification)
        elapsed = ts * adjustment
        m &= elapsed > 0.
        out = {name: np.array(value, dtype=float) for name, value in dict(
            root_hair_length=root_hair_length, actual_length_with_hairs=actual_length_with_hairs,
            total_root_hairs_number=total_root_hairs_number, dead_root_hairs_number=dead_root_hairs_number,
            living_root_hairs_number=living_root_hairs_number, all_root_hairs_formed=all_root_hairs_formed,
            root_hairs_struct_mass=root_hairs_struct_mass, living_root_hairs_struct_mass=living_root_hairs_struct_mass,
            root_hairs_struct_mass_produced=root_hairs_struct_mass_produced, root_hairs_volume=root_hairs_volume,
            started=actual_time_since_root_hairs_emergence_started,
            thermal_started=thermal_time_since_root_hairs_emergence_started,
            stopped=actual_time_since_root_hairs_emergence_stopped,
            thermal_stopped=thermal_time_since_root_hairs_emergence_stopped,
            amount=hexose_consumption_by_growth_amount, rate=hexose_consumption_by_growth,
            resp=resp_growth).items()}
        initial_mass = np.array(root_hairs_struct_mass, dtype=float)
        total = np.array(total_root_hairs_number, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            # All formed: they age
            a = m & (all_root_hairs_formed != 0)
            out["started"][a] += ts
            out["thermal_started"][a] += elapsed[a]
            out["stopped"][a] += ts
            out["thermal_stopped"][a] += elapsed[a]
            # Forming: along the part of the element beyond the growing zone
            b = m & (all_root_hairs_formed == 0) & (distance_from_tip - length < gz)
            initial_with_hairs = np.array(actual_length_with_hairs, dtype=float)
            out["actual_length_with_hairs"][b] = (distance_from_tip - gz)[b]
            increase = out["actual_length_with_hairs"] - initial_with_hairs
            total[b] = (root_hairs_density * radius * out["actual_length_with_hairs"])[b]
            rate_actual = (distance_from_tip - former_distance_from_tip) / ts
            rate_thermal = (distance_from_tip - former_distance_from_tip) / elapsed
            first = b & (rate_actual > 0.) & (initial_with_hairs <= 0.)
            out["started"][first] += (ts - increase / rate_actual)[first]
            out["thermal_started"][first] += (elapsed - increase / rate_thermal)[first]
            later = b & ~first
            out["started"][later] += ts
            out["thermal_started"][later] += elapsed[later]
            # All formed now: along the whole element
            c = m & (all_root_hairs_formed == 0) & ~(distance_from_tip - length < gz)
            out["started"][c] += ts
            out["thermal_started"][c] += elapsed[c]
            out["actual_length_with_hairs"][c] = length[c]
            increase = out["actual_length_with_hairs"] - initial_with_hairs
            total[c] = (root_hairs_density * radius * length)[c]
            moving = c & (rate_actual > 0.)
            out["stopped"][moving] += (ts - increase / rate_actual)[moving]
            out["thermal_stopped"][moving] += (elapsed - increase / rate_thermal)[moving]
            still = c & ~moving
            out["stopped"][still] += ts
            out["thermal_stopped"][still] += elapsed[still]
            out["all_root_hairs_formed"][c] = 1.
            out["total_root_hairs_number"][b | c] = total[b | c]
            # Death after their lifespan
            dead = np.where(out["thermal_started"] <= root_hairs_lifespan, 0.,
                            np.where(out["thermal_stopped"] > root_hairs_lifespan, total,
                                     total * ((out["thermal_started"] - root_hairs_lifespan)
                                              / (out["thermal_started"] - out["thermal_stopped"]))))
            out["dead_root_hairs_number"][m] = dead[m]
            living = total - out["dead_root_hairs_number"]
            out["living_root_hairs_number"][m] = living[m]
            # Elongation, limited by hexose
            short = m & (root_hair_length < root_hair_max_length)
            new_length = (root_hair_length + root_hairs_elongation_rate * root_hair_radius * C_hexose_root
                          * (out["actual_length_with_hairs"] / length) / (Km_elongation + C_hexose_root) * elapsed)
            out["root_hair_length"][short] = np.where(new_length > root_hair_max_length, root_hair_max_length,
                                                      new_length)[short]
            volume = (root_hair_radius ** 2 * pi) * out["root_hair_length"] * total
            out["root_hairs_volume"][m] = volume[m]
            mass = volume * root_tissue_density
            out["root_hairs_struct_mass"][m] = mass[m]
            out["living_root_hairs_struct_mass"][m] = np.where(total > 0., mass * living / total, 0.)[m]
            produced = mass - initial_mass
            out["root_hairs_struct_mass_produced"][m] = produced[m]
            cost = produced * struct_mass_C_content / yield_growth / 6.
            out["amount"][m] += cost[m]
            out["rate"][m] += (cost / ts)[m]
            out["resp"][m] += (cost * 6. * (1 - yield_growth))[m]
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
                "hexose_consumption_by_growth_amount", out["amount"], "hexose_consumption_by_growth", out["rate"],
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
