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
    def seedling(cls, p) -> list:
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
            """RootGrowthModel.ADDING_A_CHILD(nil_properties=True)."""
            return add(parent["vertex_index"], edge, apex, dict(
                type=type_, root_order=1, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE, emergence_cost=0.,
                angle_down=angle_down, angle_roll=angle_roll, length=length, radius=radius, original_radius=radius,
                potential_length=0., theoretical_radius=radius, potential_radius=radius, initial_length=0.,
                initial_radius=radius, volume=0., struct_mass=0., initial_struct_mass=0.,
                growth_duration=p["GDs"] * (2 * radius) ** 2, life_duration=p["LDs"] * 2. * radius * nrtd,
                temperature_modification=0.))

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
                    # the legacy's np.random.seed(random_choice + i j (· 3)), as a local stream
                    rng = np.random.RandomState(int(p["random_choice"]) + i * j * (1 if kind == "seminal" else 3))
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
        elements = cls.seedling(p)
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
