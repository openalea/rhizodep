"""
rhizodep's root architecture as a metafspm StructuralComponent, C-free (ArchiSimple): the development of the root
system (its apices' potential growth, durations and emergences, the pipe model, segmentation and lateral primordia,
deaths), its growth the potential one times a satisfaction coefficient. RootGrowth (openalea.rhizodep.growth) adds the
limitation of growth by resources; both are the port of the legacy RootGrowthModel (openalea.rhizodep.legacy, frozen;
devplan_metafspm.md §23).

    table = pd.DataFrame([dict(plant="p0", model=MyModel, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": {"D_ini": 7e-4, "n_adventitious_roots": 50}})])

with MyModel.initiators = (RootArchitecture,) (or RootGrowth): each plant's seedling is built by initiate_plant, its
parameters the class defaults overridden by the plant's scenario.

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
from math import floor, pi

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
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
class RootArchitecture(StructuralComponent):
    """
    The root architecture of rhizodep, C-free (ArchiSimple; Pagès et al., 2014): the apices' development (their
    elongation from their diameter, growth durations, emergences, stops and deaths), the pipe model, segmentation
    and lateral primordia; growth is the potential growth times the plant's satisfaction coefficient. RootGrowth
    adds the limitation by resources (devplan §23).
    """
    # --- Inputs ---
    soil_temperature: float = input_variable(**doc('°C', 'soil temperature in contact with roots'),
                                             by='model_soil', initialize=15, scale=SubOrgan)
    root_tissue_density: float = input_variable(**doc('g.m-3', 'root_tissue_density',
                                                      unit_comment='of structural mass'),
                                                by='model_anatomy', initialize=100000.0, scale=SubOrgan)

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
                                             state_variable_type=NII, initialize=0.001, scale=SubOrgan)   # g8
    total_root_hairs_number: float = state_variable(**doc('adim',
                                                          'Example root hairs number on segment external surface',
                                                          value_comment='30 * (1.6e-4 / radius) * length * 1e3',
                                                          references=' According to the work of Gahoonia et al. '
                                                                     '(1997), the root hair density is about 30 hairs'
                                                                     ' per mm for winter wheat, for a root radius of '
                                                                     'about 0.16 mm.'),
                                                    state_variable_type=NIE, initialize=41.142857142857146,
                                                    scale=SubOrgan)
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
    root_hairs_struct_mass: float = state_variable(**doc('g', "Structural mass of the element's root hairs"),
                                                   state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    root_hairs_volume: float = state_variable(**doc('m3', "Volume of the element's root hairs"),
                                              state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
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

    struct_mass_input: float = input_variable(
        **doc('g', 'Structural mass offered to the root system\'s growth in the step (ArchiSimple)',
              value_comment='infinite by default: no limitation'),
        by='model_shoot', initialize=float("inf"), scale=ScalesConfig.Plant)

    # --- The legacy's behaviours (devplan §6, §25): the reviewed fixes are the defaults; True reproduces the legacy
    # (LEGACY, the comparisons' preset)
    legacy_pipe_model: bool = parameter(**doc('adim', "The legacy's pipe model, blind to its children (g16): it "
                                                      "compares child.edge_type, a method, with '<' and '+', so no "
                                                      "child's section counts and roots never thicken"),
                                        by=BY, default=False)
    legacy_primordium_soil_temperature: bool = parameter(
        **doc('adim', "A new primordium's soil temperature: the legacy's 7.8 °C (g3), else its mother's"),
        by=BY, default=False)
    legacy_segmentation_masses: bool = parameter(
        **doc('adim', "Segmentation shares an apex's amounts by the segments' mass relative to the apex's mass before "
                      "growth, its living root hairs' too (g1), which loses amounts; else after growth"),
        by=BY, default=False)
    legacy_growth_duration_draws: bool = parameter(
        **doc('adim', "The probabilistic growth durations' if/if (g4): a low duration drawn, then overwritten; and no "
                      "duration for a thin root without GD_by_frequency (it raises); else exclusive classes, the "
                      "thinnest roots taking GD_high"),
        by=BY, default=False)
    legacy_nodule_radius: bool = parameter(
        **doc('adim', "A nodule's thickening to the constant radius (3 / 4π)^(1/3) (g5), else to its volume's"),
        by=BY, default=False)
    legacy_zero_temperature: bool = parameter(
        **doc('adim', "Actual times divided by a zero temperature factor (g6, infinite); else 0 then"),
        by=BY, default=False)
    legacy_support_development: bool = parameter(
        **doc('adim', "The zero-length supports take their axis apex's stop and die with their children (g11); else "
                      "they keep their type, as rhizodep's main branch guards them"),
        by=BY, default=False)
    legacy_seedling_seeds: bool = parameter(
        **doc('adim', "The seedling's draws seeded by random_choice + i·j (g15): with one root per emergence event, "
                      "j = 0, every seminal (nodal) root the same draws; else seeded by its support's id"),
        by=BY, default=False)

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
    emergence_delay: float = parameter(**doc('s', 'Delay of emergence of the primordium',
                                             unit_comment='time equivalent at temperature of T_ref',
                                             references='emergence_delay = 3 days (??)'),
                                       by=BY, default=282528.0)
    EL: float = parameter(**doc('s-1', 'Slope of the elongation rate = f(tip diameter) ',
                                unit_comment='meters of root per meter of radius per second equivalent to '
                                             'T_ref_growth',
                                references='EL = 5 mm mm-1 day-1 (??)'),
                          by=BY, default=0.00065)
    SGC: float = parameter(**doc('adim',
                                 'Proportionality coefficient between the section area of the segment and the sum of '
                                 'distal section areas'),
                           by=BY, default=0.0)
    struct_mass_C_content: float = parameter(**doc('mol.g-1', 'C content of structural mass',
                                                   unit_comment='of carbon',
                                                   references='We assume that the structural mass contains 44% of C. '
                                                              '(??)'),
                                             by=BY, default=0.03663613655287261)
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
    initial_segment_length: float = parameter(**doc('m', 'Initial segment length'), by=BY, default=0.001)
    initial_apex_length: float = parameter(**doc('m', 'Initial apex length'), by=BY, default=0.0001)
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
           "struct_mass_produced", "actual_time_since_primordium_formation", "actual_time_since_emergence",
           "actual_time_since_cells_formation", "actual_potential_time_since_emergence",
           "actual_time_since_growth_stopped", "actual_time_since_death", "thermal_time_since_primordium_formation",
           "thermal_time_since_emergence", "thermal_time_since_cells_formation",
           "thermal_potential_time_since_emergence", "thermal_time_since_growth_stopped", "thermal_time_since_death")


    # The options' values reproducing the legacy (the comparisons' preset; devplan §25)
    LEGACY = dict(legacy_pipe_model=True, legacy_primordium_soil_temperature=True, legacy_segmentation_masses=True,
                  legacy_growth_duration_draws=True, legacy_nodule_radius=True, legacy_zero_temperature=True,
                  legacy_support_development=True, legacy_seedling_seeds=True)

    # The resources of growth (RootGrowth's, devplan §21): none, the architecture is C-free
    resources = ()
    # ArchiSimple's development: C-free elongation, durations from the square diameter, emergence at once (the legacy's
    # simple_growth_duration branch; RootGrowth makes it a parameter)
    simple_growth_duration = True
    growth_duration_rule = "simple"
    # Variables zeroed at each step's start besides the resources' (RootGrowth: the respiration, the fungus' share)
    ZEROED_AT_START = ()

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
                    if parameters["legacy_seedling_seeds"]:
                        stream.seed(int(parameters["random_choice"]) + i * j * (1 if kind == "seminal" else 3))
                    else:                                   # by root: its support's id, the next element's
                        stream.seed(int(parameters["random_choice"]) * (len(elements) + 1))
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
        temperature factor, legacy parent, distance from tip and axis apex, the resources (_initial_resources)."""
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
                          root_tissue_density=parameters["root_tissue_density"],
                          **{resource.concentration: 0. for resource in cls.resources})
            values["vertex_index"] = element["vertex_index"]
            values["parent_id"] = (1 if element["vertex_index"] in collar_children else (element["parent"] or -1))
            volume = (4 / 3. * pi * values["radius"] ** 3 if values["type"] == TYPE_NODULE else
                      pi * values["radius"] ** 2 * values["length"])
            values["volume"] = volume
            values["struct_mass"] = volume * parameters["new_root_tissue_density"]
            values["living_struct_mass"] = values["struct_mass"] + values["living_root_hairs_struct_mass"]
            values["temperature_modification"] = factor
        cls._initial_resources(parameters, elements)
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
    def _initial_resources(cls, parameters, elements) -> None:
        """The seedling's resources (the architecture has none)."""

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

    def _actual_time(self, thermal_time, temperature_factor):
        """The actual time of a thermal time: divided by the temperature factor; 0 when it is 0 (else the legacy's
        division, g6)."""
        if self.legacy_zero_temperature:
            return thermal_time / temperature_factor
        return np.where(temperature_factor > 0., thermal_time / np.where(temperature_factor > 0., temperature_factor,
                                                                         1.), 0.)

    def supply_zones(self, supplied, radius, length, volume, struct_mass) -> dict:
        """The supply zones of the elongating apices *supplied*: {resource concentration: its concentration in each
        apex's zone} (the architecture: none)."""
        return {}

    def elongated_length(self, selected, start_length, radius, time, zone):
        """The potential length of the *selected* apices (RootGrowthModel.elongated_length): ArchiSimple's, C-free,
        EL · 2r · t. *zone*: the supply zones' concentrations (supply_zones)."""
        return start_length + self._parameter("EL", selected) * 2. * radius * time

    def thicken_nodules(self, nodule, struct_mass, radius, temperature_factor, time_step, theoretical_radius,
                        potential_radius) -> None:
        """The nodules' potential thickening, written in *theoretical_radius* and *potential_radius* (the
        architecture: none, a nodule thickens with its resources)."""

    def limited_potential_radius(self, normal, theoretical_radius, potential_radius, radius, initial_radius, length,
                                 temperature_factor, time_step) -> None:
        """The segments' (*normal*) potential radius from their pipe-model radius, written in *potential_radius*:
        ArchiSimple's, the pipe model's itself."""
        potential_radius[normal] = theoretical_radius[normal]

    def synchronise_emergence(self, type, thermal_time_since_primordium_formation, emergence_delay_in_thermal_time):
        """The primordia's times and emergence delays before their emergence is tested (the legacy: unchanged; a
        shoot model may synchronise the nodal roots' emergence with its phytomers)."""
        return thermal_time_since_primordium_formation, emergence_delay_in_thermal_time

    # ── Potential growth (RootGrowthModel.potential_growth; devplan §14) ───────────────────────────────────────

    @potential
    def _initial_length(self, length, radius, struct_mass, living_root_hairs_struct_mass, soil_temperature,
                        parent_id, vertex_index, process_at_T_ref, T_ref, A, B, C) -> outputs(12):
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
        for name in self.ZEROED_AT_START:
            ds.set(name, zero)
        return (np.array(length), "initial_radius", np.array(radius), "potential_radius", np.array(radius),
                "theoretical_radius", np.array(radius), "potential_length", np.array(length),
                "initial_struct_mass", np.array(struct_mass),
                "initial_living_root_hairs_struct_mass", np.array(living_root_hairs_struct_mass),
                "soil_temperature", soil, "temperature_modification", factor,
                "struct_mass_produced", zero, "root_hairs_struct_mass_produced", zero,
                "actual_elongation", zero, "actual_elongation_rate", zero)

    @potential
    def _potential_length(self, type, length, radius, initial_length, initial_radius, potential_length, volume,
                          struct_mass, temperature_modification, emergence_delay_in_thermal_time,
                          growth_duration, life_duration, thermal_time_since_primordium_formation,
                          actual_time_since_primordium_formation, thermal_potential_time_since_emergence,
                          thermal_time_since_emergence, actual_time_since_emergence,
                          thermal_time_since_growth_stopped, actual_time_since_growth_stopped,
                          thermal_time_since_cells_formation, actual_time_since_cells_formation,
                          thermal_time_since_death, actual_time_since_death,
                          lateral_root_emergence_possibility, lateral_primordium_index, vertex_index,
                          emergence_delay) -> outputs(15):
        """
        The apices (RootGrowthModel.potential_apex_development): the state machine as masked branches, the supply
        zone of those elongating (calculating_supply_for_elongation, by path windows), their potential length; a
        lateral primordium emerging marks its bearer (the last one visited when several emerge on one bearer).
        The branches are chosen on the state at the step's start (the arguments); each element is in one branch.
        The supply zones are the resources' (supply_zones).
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
        zone = self.supply_zones(supplied, radius, length, volume, struct_mass)

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
            clocks.actual["growth_stopped"][selected] = self._actual_time(stopped_since, temperature_factor[selected])
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
            clocks.actual["death"][selected] = self._actual_time(dead_since, temperature_factor[selected])
        clocks.advance(selected, time_step, thermal_time_step, "primordium_formation", "emergence", "cells_formation",
                       "growth_stopped")

        return (new_potential_length, "type", new_type,
                "thermal_potential_time_since_emergence", potential_time_since_emergence, *clocks.outputs(),
                "emergence_delay_in_thermal_time", emergence_delay_in_thermal_time,
                "lateral_root_emergence_possibility", possibility, "lateral_primordium_index", primordium_index)

    @potential
    def _theoretical_radius(self, type, length, radius, initial_radius, theoretical_radius, potential_radius,
                            struct_mass, temperature_modification, axis_apex_id, vertex_index,
                            thermal_time_since_primordium_formation, actual_time_since_primordium_formation,
                            thermal_time_since_emergence, actual_time_since_emergence,
                            thermal_time_since_cells_formation, actual_time_since_cells_formation,
                            thermal_time_since_growth_stopped, actual_time_since_growth_stopped,
                            thermal_time_since_death, actual_time_since_death, SGC) -> outputs(13):
        """
        The segments (RootGrowthModel.potential_segment_development): the type of their axis apex (stopped), death
        when all their children are dead, the pipe model, children first (a fold up the tree), then the potential
        radius (limited_potential_radius), the supports' radius, the times; the nodules' thickening (thicken_nodules).
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
        self.thicken_nodules(nodule, struct_mass, radius, temperature_factor, time_step, new_theoretical_radius,
                             new_potential_radius)

        # The axis apex's stop
        new_type = np.array(type)
        node_of = self._node_of(vertex_index)
        apex_node = node_of[np.clip(np.asarray(axis_apex_id, dtype=np.int64), 0, node_of.size - 1)]
        apex_type = np.where(apex_node >= 0, type[np.maximum(apex_node, 0)], -1)
        developing = normal
        if not self.legacy_support_development:         # the supports keep their type (rhizodep's main, g11)
            developing = normal & ~np.isin(type, (TYPE_SEMINAL_SUPPORT, TYPE_ADVENTITIOUS_SUPPORT))
        new_type[developing & (apex_type == TYPE_JUST_STOPPED)] = TYPE_JUST_STOPPED
        new_type[developing & (apex_type == TYPE_STOPPED)] = TYPE_STOPPED

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
            all_dead = todo & developing[nodes] & (n_dead == n_children)
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
            self.limited_potential_radius(normal, new_theoretical_radius, new_potential_radius, radius, initial_radius,
                                          length, temperature_factor, time_step)
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

    @actual
    def _length(self, type, length, radius, initial_length, initial_radius, potential_length, potential_radius,
                volume, struct_mass, struct_mass_produced, root_tissue_density, temperature_modification,
                thermal_potential_time_since_emergence, thermal_time_since_emergence, actual_time_since_emergence,
                actual_elongation, actual_elongation_rate, dist_to_ramif) -> outputs(11):
        """
        ArchiSimple's growth (the legacy's ArchiSimple_growth): each element grows by its potential increments times
        the plant's satisfaction coefficient SC (struct_mass_input over the potential growth's demand in structural
        mass, between 0 and 1; satisfaction_coefficient). Completed as the legacy's actual growth (devplan §23, D-P7a):
        the apices that could elongate are segmented, the primordia that elongate emerge.
        """
        ds = self.data_structure
        time_step = float(self.time_step)
        volume_of = self._volume
        satisfaction = self.satisfaction_coefficient(type, initial_radius, initial_length, potential_radius,
                                                     potential_length, root_tissue_density)
        grows = ~np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD, TYPE_STOPPED, TYPE_SEMINAL_SUPPORT,
                                TYPE_ADVENTITIOUS_SUPPORT))
        new_length, new_radius = np.array(length), np.array(radius)
        new_length[grows] += ((potential_length - initial_length) * satisfaction)[grows]
        new_radius[grows] += ((potential_radius - initial_radius) * satisfaction)[grows]
        elongation, elongation_rate = np.array(actual_elongation), np.array(actual_elongation_rate)
        elongation[grows] = (new_length - initial_length)[grows]
        temperature_factor = temperature_modification
        with np.errstate(divide="ignore", invalid="ignore"):
            just_emerged = ((thermal_potential_time_since_emergence > 0)
                            & (thermal_potential_time_since_emergence < time_step))
            elongation_rate[grows] = np.where(just_emerged, elongation / (thermal_potential_time_since_emergence
                                                                          / temperature_factor),
                                              elongation / time_step)[grows]
        new_volume, new_mass, produced = np.array(volume), np.array(struct_mass), np.array(struct_mass_produced)
        new_volume[grows] = volume_of(type, new_radius, new_length)[grows]
        new_mass[grows] = (new_volume * root_tissue_density)[grows]
        produced[grows] = ((new_volume - volume) * root_tissue_density)[grows]
        new_dist_to_ramif = np.array(dist_to_ramif)
        apices = grows & self._apex()
        new_dist_to_ramif[apices] += elongation[apices]

        # Completed (D-P7a): the apices that could elongate segment, the primordia that elongate emerge
        elongating = grows & (potential_length > initial_length)
        new_type = np.array(type)
        time_since_emergence = np.array(thermal_time_since_emergence)
        actual_since_emergence = np.array(actual_time_since_emergence)
        emerges = grows & (new_length > initial_length) & np.isin(type, (
            TYPE_SEMINAL_BEFORE_EMERGENCE, TYPE_ADVENTITIOUS_BEFORE_EMERGENCE, TYPE_BEFORE_EMERGENCE))
        new_type[emerges] = TYPE_EMERGED
        time_since_emergence[emerges] = thermal_potential_time_since_emergence[emerges]
        with np.errstate(divide="ignore", invalid="ignore"):
            actual_since_emergence[emerges] = self._actual_time(time_since_emergence[emerges],
                                                                temperature_factor[emerges])

        return (new_length, "radius", new_radius, "volume", new_volume, "struct_mass", new_mass,
                "struct_mass_produced", produced, "type", new_type, "thermal_time_since_emergence",
                time_since_emergence, "actual_time_since_emergence", actual_since_emergence,
                "actual_elongation", elongation, "actual_elongation_rate", elongation_rate,
                "dist_to_ramif", new_dist_to_ramif, "step_elongating", elongating.astype(int))

    def satisfaction_coefficient(self, type, initial_radius, initial_length, potential_radius, potential_length,
                                 root_tissue_density) -> np.ndarray:
        """ArchiSimple's satisfaction coefficient of each element's plant (the legacy's satisfaction_coefficient):
        the structural mass offered to growth (struct_mass_input) over the potential growth's demand, summed in the
        legacy's order (children first); 1 without demand; between 0 and 1."""
        ds = self.data_structure
        demand = (self._volume(type, potential_radius, potential_length)
                  - self._volume(type, initial_radius, initial_length)) * root_tissue_density
        owner = np.asarray(ds.owner("Plant"))
        visit = ds.order("post", convention="openalea")
        total = np.zeros(np.asarray(ds.entity_ids("Plant")).size)
        np.add.at(total, owner[visit], demand[visit])
        offered = np.maximum(np.asarray(ds.get("struct_mass_input"), dtype=float), 0.)
        with np.errstate(divide="ignore", invalid="ignore"):
            coefficient = np.where(total <= 0., 1., offered / total)
        return np.clip(coefficient, 0., 1.)[owner]

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
              "initial_struct_mass", "root_hair_length", "actual_length_with_hairs",
              "living_root_hairs_number", "dead_root_hairs_number", "total_root_hairs_number",
              "actual_time_since_root_hairs_emergence_started", "thermal_time_since_root_hairs_emergence_started",
              "actual_time_since_root_hairs_emergence_stopped", "thermal_time_since_root_hairs_emergence_stopped",
              "all_root_hairs_formed", "root_hairs_struct_mass", "root_hairs_struct_mass_produced",
              "living_root_hairs_struct_mass", "initial_living_root_hairs_struct_mass", "struct_mass_produced",
              "soil_temperature", "growth_duration", "life_duration", "actual_time_since_primordium_formation",
              "actual_time_since_emergence", "actual_time_since_cells_formation", "actual_time_since_growth_stopped",
              "actual_time_since_death", "thermal_time_since_primordium_formation", "thermal_time_since_emergence",
              "thermal_time_since_cells_formation", "thermal_potential_time_since_emergence",
              "thermal_time_since_growth_stopped", "thermal_time_since_death", "temperature_modification")
    # What segmentation shares between the segments by their mass
    SHARED = ("initial_struct_mass", "initial_living_root_hairs_struct_mass", "struct_mass_produced",
              "root_hairs_struct_mass", "root_hairs_struct_mass_produced", "living_root_hairs_struct_mass",
              "living_root_hairs_number", "dead_root_hairs_number", "total_root_hairs_number")

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
            for role in ("required", "demand", "consumption_amount", "consumption"):
                name = getattr(resource, role)
                values[name] = self._value(properties, name, mother) if resource.copied_by_new_elements else 0.
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
        if self.legacy_segmentation_masses:
            initial_struct_mass = properties["initial_struct_mass"][apex]
            shared["living_root_hairs_struct_mass"] = properties["initial_living_root_hairs_struct_mass"][apex]
        else:
            initial_struct_mass = properties["struct_mass"][apex]
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
        if self.simple_growth_duration or self.growth_duration_rule == "simple":
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
        """A primordium's soil temperature: its mother's (as Root_BRIDGES), or the legacy's 7.8 °C (g3)."""
        if self.legacy_primordium_soil_temperature:
            values["soil_temperature"] = 7.8
        else:
            values["soil_temperature"] = self._value(properties, "soil_temperature", mother)

    def lateral_growth_duration_factor(self, properties, apex):
        """Hook: the factor of a lateral primordium's growth duration with simple_growth_duration, from its bearing
        apex (*properties*: the MPG's; the legacy: 1)."""
        return 1.

    def _growth_duration(self, radius, legacy_id, root_order, parameters, stream):
        """RootGrowthModel.calculate_growth_duration (not simple_growth_duration): a duration drawn among classes
        (low, medium, high) by their probabilities, uniformly within the class (GD_by_frequency), or the class's
        value. The legacy's if/if and its unassigned branch with legacy_growth_duration_draws (g4)."""
        stream.seed(int(parameters["random_choice"]) * legacy_id)
        draw = stream.random_sample()
        if root_order == 1:
            return parameters["GD_highest"]
        if self.GD_by_frequency:
            if self.legacy_growth_duration_draws:
                if draw < parameters["GD_prob_low"]:
                    duration = stream.uniform(0., parameters["GD_low"])
                if draw < parameters["GD_prob_medium"]:
                    duration = stream.uniform(parameters["GD_low"], parameters["GD_medium"])
                else:
                    duration = stream.uniform(parameters["GD_medium"], parameters["GD_high"])
                return duration
            if draw < parameters["GD_prob_low"]:
                return stream.uniform(0., parameters["GD_low"])
            if draw < parameters["GD_prob_medium"]:
                return stream.uniform(parameters["GD_low"], parameters["GD_medium"])
            return stream.uniform(parameters["GD_medium"], parameters["GD_high"])
        if draw < parameters["GD_prob_low"]:
            return parameters["GD_low"]
        if draw < parameters["GD_prob_medium"]:
            return parameters["GD_medium"]
        if radius > 2 * parameters["Dmin"] / 2.:
            return parameters["GD_highest"]
        if self.legacy_growth_duration_draws:
            raise ValueError(f"no growth duration for a radius of {radius} m (the legacy's calculate_growth_duration "
                             "leaves it unassigned, g4)")
        return parameters["GD_high"]

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

    @postsegmentation
    def _root_hair_length(self, root_hair_length):
        """The root hairs' dynamics: none in the architecture (RootGrowth's; this placeholder keeps the legacy's order
        of the post-growth steps: distances, root hairs, living mass)."""
        return None

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
