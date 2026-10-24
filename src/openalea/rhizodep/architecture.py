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

The plant's structure, in the MPG (devplan §10, Q-R4, §32.5):

    Plant
     ├─ Axis (RootAxis)  the primary root        GrowthUnit ─ Phytomer ─ Organ (RootInternode), its `<` chain of
     │                                           SubOrgans: the base segment (the collar, the seed's place), the apex
     ├─ Axis             each seminal root       its Organ's first SubOrgan, its apex, branched (+) on the base
     └─ Axis             each nodal root         the same (on a shoot phytomer once a shoot model bears them)

A lateral root is an Organ of its bearer's Axis, its first SubOrgan branched (+) on its bearer's segment. Every
element is a SubOrgan (a RootSegment, or a RootApex for an axis' tip), carrying the variables below; elements are
identified by their MPG ids. Their random draws are keyed by their lineage `key` (devplan §32.3): a pure function of
their place in their plant, so independent of the visiting order and of the other plants of a population.

Codes: `type` (TYPE_*), `lateral_root_emergence_possibility` (EMERGENCE_*), `axis_type` (AXIS_*), integers.
"""
import os
from dataclasses import dataclass, fields
from math import floor, pi

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import (NewNodes, StructuralComponent, input_variable, parameter,
                                                   state_variable)
from openalea.metafspm.data_structure import random_streams
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
from openalea.metafspm.solve.decorator import actual, postsegmentation, potential, segmentation

SubOrgan = ScalesConfig.SubOrgan
NII, NIE, DESCRIPTOR, MASSIC = "NonInertialIntensive", "NonInertialExtensive", "descriptor", "massic_concentration"
BY = "model_growth"

# The legacy's element types (its type_* parameters)
TYPE_BASE = 1
TYPE_SEMINAL_BEFORE_EMERGENCE = 3
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


EDGES = {"<": 0, "+": 1}


def seed_keys(random_choice) -> np.ndarray:
    """The lineage key of a plant's first element (its base), from the plant's random_choice (devplan §32.3)."""
    return (random_streams.bits(np.asarray(random_choice, dtype=np.int64).reshape(-1), stream="seed")
            >> np.uint64(12)).astype(np.int64)


def child_keys(parent_keys, edge, rank=0) -> np.ndarray:
    """The lineage keys of new elements, from their parent's key, their edge to it and their rank among the parent's
    children of that edge created at once: a pure function of the element's place in its plant (devplan §32.3).
    52 bits, so that a key is exact in a float array."""
    return (random_streams.bits(np.asarray(parent_keys, dtype=np.int64).reshape(-1), stream="lineage",
                                step=EDGES[edge], draw=rank) >> np.uint64(12)).astype(np.int64)


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
    modified = process_at_T_ref * x ** (1 - C) * x ** (C * (soil_temperature - T_ref) / 10.)
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
    key: int = state_variable(**doc('dimensionless', "The element's lineage key, its random draws' identity: from its "
                                                     "parent's, its edge and its rank (devplan §32.3)",
                                    value_comment='-1 as default when not assigned'),
                              state_variable_type=DESCRIPTOR, initialize=-1, scale=SubOrgan, dtype='int')
    axis_apex_id: int = state_variable(**doc('dimensionless',
                                             'Node id (the MPG vertex) of the terminal apex of the root axis bearing '
                                             'the considered segment',
                                             value_comment='-1 as default when not assigned'),
                                       state_variable_type=NII, initialize=-1, scale=SubOrgan, dtype='int')
    lateral_primordium_index: int = state_variable(**doc('dimensionless',
                                                         'Node id of the lateral primordium emerging on the element'),
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
    azimuth: float = state_variable(**doc('°', "Roll of the parent's frame around its direction before the element's "
                                               "pitch (angle_down): the direction a root leaves the base in (the "
                                               "legacy's support's roll, devplan §32.5)"),
                                    state_variable_type=DESCRIPTOR, initialize=0.0, scale=SubOrgan)
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

    life_duration_diameter_exponent: float = parameter(
        **doc('adim', "The exponent of the diameter in the life duration LDs·d^e·density: 1, the legacy's and main's; "
                      "2, ArchiSimple's (Pagès et al. 2014, Eq. 3, LDs then recalibrated) (g21)"),
        by=BY, default=1.)
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


    # The resources of growth (RootGrowth's, devplan §21): none, the architecture is C-free
    resources = ()
    # ArchiSimple's development: elongation and thickening not limited by resources, emergence at once (RootGrowth's
    # is limited by its resources), durations from the square diameter (RootGrowth's: its growth_duration_rule)
    resource_limited = False
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
    def seedling(cls, parameters) -> list:
        """
        The initial elements (the legacy's RootGrowthModel.initiate_mtg, then its initiate_heterogeneous_variables and
        the post-growth they end with), for the plant's *parameters*: a list of dicts with "parent" (its index in the
        list, or None), "edge" ('<' / '+'), "apex" (a RootApex), "values" ({state variable: value}) and "axis_apex"
        (the index of its axis' apex). The base segment, the seminal roots then the nodal roots branched on it, the
        primary root's apex.
        """
        defaults = {f.name: f.default for f in fields(cls) if f.metadata.get("variable_type") == "state_variable"
                    and f.metadata.get("scale") == SubOrgan}
        elements = []
        tissue_density = parameters["new_root_tissue_density"]

        def add(parent, edge, apex, values):
            elements.append(dict(parent=parent, edge=edge, apex=apex,
                                 values={**defaults, **dict.fromkeys(cls.NIL, 0.), **values}))
            return elements[-1]

        def main_root_durations(radius):
            """The growth and life durations of a first-order root (with the main roots' extender)."""
            return dict(growth_duration=(parameters["GDs"] * (2. * radius) ** 2
                                         * parameters["main_roots_growth_extender"]),
                        life_duration=parameters["LDs"] * cls._diameter_term(radius, parameters) * tissue_density)

        # The base segment
        base_radius = parameters["D_ini"] / 2.
        length = parameters["initial_segment_length"]
        initial_mass = parameters["volume"] * tissue_density
        base_key = int(seed_keys(parameters["random_choice"])[0])
        add(None, "/", False, dict(
            type=TYPE_BASE, root_order=1, lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE, emergence_cost=0.,
            angle_down=0., angle_roll=0., length=length, radius=base_radius, original_radius=base_radius,
            initial_length=length, initial_radius=base_radius, distance_from_tip=length,
            former_distance_from_tip=length, dist_to_ramif=0., actual_elongation=length, actual_elongation_rate=0.,
            volume=parameters["volume"], struct_mass=initial_mass, initial_struct_mass=initial_mass, key=base_key,
            **main_root_durations(base_radius)))

        # The seminal, then the nodal (adventitious) roots, branched on the base: each its own draws (g15)
        rank = 0
        for kind in ("seminal", "adventitious"):
            number_of_roots = parameters[f"n_{kind}_roots"]
            forcing = parameters[f"forcing_{kind}_roots_events"]
            if not (number_of_roots > (1 if kind == "seminal" else 0) or not forcing):
                continue
            events = cls._emergence_events(parameters, kind)
            ratio = parameters["D_sem_to_D_ini_ratio" if kind == "seminal" else "D_adv_to_D_ini_ratio"]
            apex_type = TYPE_SEMINAL_BEFORE_EMERGENCE if kind == "seminal" else TYPE_ADVENTITIOUS_BEFORE_EMERGENCE
            for delay, roots_per_event in events:
                for _ in range(int(roots_per_event)):
                    key = child_keys([base_key], "+", rank=rank)
                    rank += 1
                    azimuth = float(np.abs(random_streams.normal(key, 180. if kind == "seminal" else 0., 180.,
                                                                 stream="azimuth"))[0])
                    angle_down = float(np.abs(random_streams.normal(key, 60., 10., stream="lateral_angle_down"))[0])
                    mean = parameters["D_ini"] / 2. * ratio
                    radius = (float(np.abs(random_streams.normal(key, mean, mean * parameters["CVDD"],
                                                                 stream="lateral_radius"))[0])
                              if parameters["random"] else mean)
                    apex = add(0, "+", True, cls._nil_values(parameters, apex_type, 1, angle_down, 5, 0., radius))
                    apex["values"].update(emergence_delay_in_thermal_time=delay, key=int(key[0]), azimuth=azimuth,
                                          **main_root_durations(radius))

        # The primary root's apex
        apex = add(0, "<", True, cls._nil_values(parameters, TYPE_EMERGED, 1, 0, 0, parameters["initial_apex_length"],
                                                 base_radius))
        apex["values"].update(initial_struct_mass=initial_mass, key=int(child_keys([base_key], "<")[0]),
                              **main_root_durations(base_radius))

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
            life_duration=parameters["LDs"] * cls._diameter_term(radius, parameters) * tissue_density,
            temperature_modification=0.)}

    @staticmethod
    def _diameter_term(radius, parameters):
        """d^e of the life duration (e: life_duration_diameter_exponent, g21); d itself for e = 1."""
        exponent = parameters["life_duration_diameter_exponent"]
        term = np.where(exponent == 1, 2. * radius, (2. * radius) ** exponent)
        return term if np.ndim(term) else float(term)

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
        temperature factor, distance from tip and axis apex, the resources (_initial_resources)."""
        factor = float(temperature_modification(parameters["soil_temperature"], parameters["process_at_T_ref"],
                                                parameters["T_ref"], parameters["A"], parameters["B"],
                                                parameters["C"]))
        for element in elements:
            values = element["values"]
            values.update(soil_temperature=parameters["soil_temperature"],
                          root_tissue_density=parameters["root_tissue_density"],
                          **{resource.concentration: 0. for resource in cls.resources})
            volume = (4 / 3. * pi * values["radius"] ** 3 if values["type"] == TYPE_NODULE else
                      pi * values["radius"] ** 2 * values["length"])
            values["volume"] = volume
            values["struct_mass"] = volume * parameters["new_root_tissue_density"]
            values["living_struct_mass"] = values["struct_mass"] + values["living_root_hairs_struct_mass"]
            values["temperature_modification"] = factor
        cls._initial_resources(parameters, elements)
        # Distance from tip and axis apex: children before parents, along the successors ('<')
        successor_of = {element["parent"]: index for index, element in enumerate(elements) if element["edge"] == "<"}
        for index in reversed(range(len(elements))):
            element = elements[index]
            values, successor = element["values"], successor_of.get(index)
            if element["apex"] or values["type"] == TYPE_NODULE or successor is None:
                values["distance_from_tip"], element["axis_apex"] = values["length"], index
            else:
                values["distance_from_tip"] = elements[successor]["values"]["distance_from_tip"] + values["length"]
                element["axis_apex"] = elements[successor]["axis_apex"]
            values["former_distance_from_tip"] = values["distance_from_tip"]

    @classmethod
    def _initial_resources(cls, parameters, elements) -> None:
        """The seedling's resources (the architecture has none)."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        """The legacy's seedling under *plant* (the class docstring's structure), its values on the MPG."""
        parameters = cls.plant_parameters(parameters)
        elements = cls.seedling(parameters)
        scales, labels = g.scales, g.labels
        vids = {}

        def new_organ():
            complex_ = plant
            for scale, label in ((scales.Axis, labels.Axis.Root), (scales.GrowthUnit, labels.GrowthUnit.Root),
                                 (scales.Phytomer, labels.Phytomer.Root), (scales.Organ, labels.Organ.RootInternode)):
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            return complex_

        for index, e in enumerate(elements):
            label = labels.SubOrgan.RootApex if e["apex"] else labels.SubOrgan.RootSegment
            if e["parent"] is None:
                vid = g.add_component(new_organ(), **PropsConfig(scale=scales.SubOrgan, edge_type='/', label=label))
            elif e["edge"] == "<":
                vid = g.add_child(vids[e["parent"]], **PropsConfig(scale=scales.SubOrgan, edge_type='<', label=label))
            else:
                vid = g.add_component_with_topo(new_organ(), vids[e["parent"]],
                                                **PropsConfig(scale=scales.SubOrgan, edge_type='+', label=label))
            vids[index] = vid
            for name, value in e["values"].items():
                g.property(name)[vid] = value
        for index, e in enumerate(elements):
            g.property("axis_apex_id")[vids[index]] = vids[e["axis_apex"]]
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

    def _index_of(self, node_ids) -> np.ndarray:
        """The node index of each node id (-1 where the id is -1)."""
        node_ids = np.asarray(node_ids, dtype=np.int64)
        known = node_ids >= 0
        out = np.full(node_ids.shape, -1, dtype=np.int64)
        out[known] = self.data_structure.index_of(node_ids[known])
        return out

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
        """The actual time of a thermal time: divided by the temperature factor; 0 when it is 0 (g6)."""
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
                        process_at_T_ref, T_ref, A, B, C) -> outputs(12):
        """The step's start: the fluxes zeroed, the initial and potential values copied, the temperature factor; a
        zero-length apex takes its parent's soil temperature."""
        ds = self.data_structure
        parent = np.asarray(ds.parents())
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
                          lateral_root_emergence_possibility, lateral_primordium_index,
                          emergence_delay) -> outputs(15):
        """
        The apices (RootGrowthModel.potential_apex_development): the state machine as masked branches, the supply
        zone of those elongating (calculating_supply_for_elongation, by path windows), their potential length; a
        lateral primordium emerging marks its bearer (the latest formed when several emerge on one bearer).
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
        if not self.resource_limited:
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
        if not self.resource_limited:
            new_type[selected] = TYPE_EMERGED
        else:
            emerging = np.flatnonzero(selected)                     # ascending ids: the latest formed last
            bearers = np.asarray(ds.parents())[emerging]
            last = bearers.size - 1 - np.unique(bearers[::-1], return_index=True)[1]   # the latest per bearer
            possibility[bearers[last]] = EMERGENCE_POSSIBLE
            primordium_index[bearers[last]] = np.asarray(ds.entity_ids("node"))[emerging[last]]

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
                            struct_mass, temperature_modification, axis_apex_id,
                            thermal_time_since_primordium_formation, actual_time_since_primordium_formation,
                            thermal_time_since_emergence, actual_time_since_emergence,
                            thermal_time_since_cells_formation, actual_time_since_cells_formation,
                            thermal_time_since_growth_stopped, actual_time_since_growth_stopped,
                            thermal_time_since_death, actual_time_since_death, SGC) -> outputs(13):
        """
        The segments (RootGrowthModel.potential_segment_development): the type of their axis apex (stopped), death
        when all their children are dead, the pipe model, children first (a fold up the tree), then the potential
        radius (limited_potential_radius), the times; the nodules' thickening (thicken_nodules).
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
        apex_node = self._index_of(axis_apex_id)
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
                section = out[:, 0] ** 2 * pi
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
        return np.where(type == TYPE_NODULE, 4 / 3. * pi * radius ** 3, pi * radius ** 2 * length)

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
        grows = ~np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD, TYPE_STOPPED))
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
        the structural mass offered to growth (struct_mass_input) over the potential growth's demand; 1 without demand;
        between 0 and 1."""
        ds = self.data_structure
        demand = (self._volume(type, potential_radius, potential_length)
                  - self._volume(type, initial_radius, initial_length)) * root_tissue_density
        owner = np.asarray(ds.owner("Plant"))
        total = np.bincount(owner, weights=demand, minlength=np.asarray(ds.entity_ids("Plant")).size)
        offered = np.maximum(np.asarray(ds.get("struct_mass_input"), dtype=float), 0.)
        with np.errstate(divide="ignore", invalid="ignore"):
            coefficient = np.where(total <= 0., 1., offered / total)
        return np.clip(coefficient, 0., 1.)[owner]

    # ── Segmentation and primordia (RootGrowthModel.segmentation_and_primordia_formation) ─────────────────────
    #
    # One array-style step (devplan §32.4): the apices that elongated in actual growth are cut into segments of
    # segment_length and a new apex; lateral primordia form on the apices and the new segments, nodules on the last
    # segments. All of a step's new elements are one NewNodes batch, created by metafspm in bulk. The draws are keyed
    # by the elements' lineage keys (devplan §32.3): they depend neither on the visiting order nor on the other plants.

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
    # The per-plant parameters segmentation reads
    SEGMENTATION_PARAMETERS = ("life_duration_diameter_exponent", "segment_length", "IPD", "Dmin", "RMD", "CVDD",
                               "root_order_treshold", "GDs", "LDs", "new_root_tissue_density", "GD_highest", "GD_high",
                               "GD_medium", "GD_low", "GD_prob_low", "GD_prob_medium", "nodule_formation_probability",
                               "struct_mass_C_content")

    def lateral_growth_duration_factor(self, bearers):
        """Hook: the factor of the lateral primordia's growth durations with the "simple" rule, from their bearers
        (node indices; -1 for a bearer created in the same step). The legacy: 1."""
        return np.ones(np.asarray(bearers).size)

    def _growth_durations(self, radius, keys, root_order, parameters):
        """The "probabilistic" rule (RootGrowthModel.calculate_growth_duration): a duration drawn among classes (low,
        medium, high) by their probabilities, uniformly within the class (GD_by_frequency), or the class's value; the
        classes exclusive, the thinnest roots taking GD_high (g4). *parameters*: {name: a value per primordium}."""
        p = parameters
        draw = random_streams.uniform(keys, stream="growth_duration_class")
        within = random_streams.uniform(keys, stream="growth_duration")
        low, medium = draw < p["GD_prob_low"], draw < p["GD_prob_medium"]
        if self.GD_by_frequency:
            duration = np.where(low, within * p["GD_low"],
                                np.where(medium, p["GD_low"] + within * (p["GD_medium"] - p["GD_low"]),
                                         p["GD_medium"] + within * (p["GD_high"] - p["GD_medium"])))
        else:
            duration = np.where(low, p["GD_low"], np.where(medium, p["GD_medium"], np.where(
                radius > p["Dmin"], p["GD_highest"], p["GD_high"])))
        return np.where(root_order == 1, p["GD_highest"], duration)

    def _default(self, name):
        """A variable's value for new elements (its declared default)."""
        return self.data_structure._variable_meta().get(name, {}).get("default", 0.)

    @segmentation
    def _dist_to_ramif(self, dist_to_ramif, type, root_order, length, radius, initial_length, volume, struct_mass,
                       root_tissue_density, soil_temperature, temperature_modification, actual_elongation,
                       actual_elongation_rate, actual_time_since_cells_formation, thermal_time_since_cells_formation,
                       angle_roll, key, step_elongating) -> outputs(7):
        """
        RootGrowthModel.segmentation_and_primordium_formation on the apices that elongated (step_elongating):

        - an apex longer than segment_length becomes the first of n segments of segment_length (n the largest with
          some length left), followed by n - 1 new segments and a new apex of the remaining length; each takes its
          share of the apex's amounts by its mass, its distance to the last branching, and the mean age of its cells
          (the elongated part's, the elongation spread at a constant rate);
        - a lateral primordium forms on the apex (or on its first segment) and on each new segment and apex without a
          child, when its distance to the last branching exceeds IPD and its drawn diameter reaches Dmin (g19); the
          position is used either way (g20); its radius RMD · (r - Dmin/2) + Dmin/2, drawn (CVDD), at most its
          bearer's; its growth duration by the rule;
        - a nodule on the last segment, by its probability (nodules on).

        The segments' and the new apex's other values are their apex's (ADDING_A_CHILD(identical_properties)); the
        primordia's and nodules' are nil (ADDING_A_CHILD(nil_properties)).
        """
        ds = self.data_structure
        labels, scales = ds._mtg.labels, ds._mtg.scales
        ids = np.asarray(ds.entity_ids("node"))
        time_step = float(self.time_step)
        indptr, _ = ds.children()
        n_children = np.diff(np.asarray(indptr))
        p = {name: np.asarray(self._parameter(name), dtype=float) for name in self.SEGMENTATION_PARAMETERS}
        new_length, new_volume, new_mass = np.array(length), np.array(volume), np.array(struct_mass)
        new_dist_to_ramif, new_angle_roll = np.array(dist_to_ramif), np.array(angle_roll)
        actual_cells, thermal_cells = (np.array(actual_time_since_cells_formation),
                                       np.array(thermal_time_since_cells_formation))
        shared = {name: np.array(ds.get(name), dtype=float) for name in self.SHARED}
        apices = np.flatnonzero(self._apex() & (type == TYPE_EMERGED) & (step_elongating == 1))
        empty = NewNodes(parent=np.zeros(0, dtype=np.int64))
        if apices.size == 0:
            return (new_dist_to_ramif, "length", new_length, "volume", new_volume, "struct_mass", new_mass,
                    "actual_time_since_cells_formation", actual_cells, "thermal_time_since_cells_formation",
                    thermal_cells, "angle_roll", new_angle_roll, "new_elements", empty)

        # Per apex
        L, S = length[apices], p["segment_length"][apices]
        ratio = L / S
        whole = np.floor(ratio)
        n_segments = np.where(L > S, np.where(ratio - whole > 0., whole, whole - 1), 0).astype(np.int64)
        final = L - n_segments * S
        rate, elongation = actual_elongation_rate[apices], actual_elongation[apices]
        start_length, factor = initial_length[apices], temperature_modification[apices]
        cells, thermal = actual_time_since_cells_formation[apices], thermal_time_since_cells_formation[apices]
        d0, r = dist_to_ramif[apices], radius[apices]
        split = n_segments > 0
        mass_before = struct_mass[apices]

        # The new chain elements: per split apex, its segments 2..n and the new apex (rank n + 1)
        owner = np.repeat(np.arange(apices.size), n_segments)       # the apex (its position in apices) of each
        first = np.cumsum(n_segments) - n_segments
        rank = np.arange(owner.size) - first[owner] + 2               # 2..n + 1
        is_new_apex = rank == n_segments[owner] + 1
        chain_parent = np.where(rank == 2, ids[apices][owner], -1 - (np.arange(owner.size) - 1))
        chain_keys = np.zeros(owner.size, dtype=np.int64)
        apex_keys = key[apices].astype(np.int64)
        for depth in range(2, int(rank.max(initial=1)) + 1):         # each element's key from its predecessor's
            at = np.flatnonzero(rank == depth)
            chain_keys[at] = child_keys(apex_keys[owner[at]] if depth == 2 else chain_keys[at - 1], "<")
        chain_length = np.where(is_new_apex, final[owner], S[owner])

        # Their segment angles, one draw per apex
        if self.random:
            segment_angle_down = random_streams.normal(key[apices], 0., 5., stream="segment_angle_down")
            segment_angle_roll = random_streams.normal(key[apices], 0., 5., stream="segment_angle_roll")
        else:
            segment_angle_down = segment_angle_roll = np.zeros(apices.size)

        # The split apices become their first segment
        cut = apices[split]
        new_length[cut] = S[split]
        new_dist_to_ramif[cut] = d0[split] - (L[split] - S[split])
        new_volume[cut] = pi * r[split] ** 2 * S[split]
        new_mass[cut] = new_volume[cut] * root_tissue_density[cut]
        fraction = new_mass[cut] / mass_before[split]
        for name in self.SHARED:
            shared[name][cut] = shared[name][cut] * fraction
        with np.errstate(divide="ignore", invalid="ignore"):
            age_up = elongation / rate
            age_down = (final + (n_segments - 1) * S) / rate
            age_elongated = 0.5 * (age_down + age_up)
            actual_cells[cut] = ((start_length * (cells + time_step) + (S - start_length) * age_elongated) / S)[split]
            thermal_cells[cut] = ((start_length * (thermal + time_step * factor)
                                   + (S - start_length) * age_elongated * factor) / S)[split]
            # The apices too short to split: the mean age of their cells
            kept = apices[~split & (rate > 0.)]
            short = ~split & (rate > 0.)
            age_elongated = 0.5 * (elongation / rate)
            actual_cells[kept] = ((start_length * (cells + time_step) + (L - start_length) * age_elongated) / L)[short]
            thermal_cells[kept] = ((start_length * (thermal + time_step * factor)
                                    + (L - start_length) * age_elongated * factor) / L)[short]

        # The new chain's values: their apex's, but for their place in the chain
        chain = {name: np.array(ds.get(name), dtype=float)[apices][owner] for name in self.COPIED}
        n_after = n_segments[owner] - rank                           # the segments after each (-1 for the apex)
        with np.errstate(divide="ignore", invalid="ignore"):
            chain_rate = rate[owner]
            cells_age = np.where(is_new_apex, 0.5 * final[owner] / chain_rate,
                                 0.5 * ((final[owner] + n_after * S[owner]) / chain_rate
                                        + (final[owner] + (n_after - 1) * S[owner]) / chain_rate))
        chain_volume = pi * r[owner] ** 2 * chain_length
        chain_mass = chain_volume * root_tissue_density[apices][owner]
        chain_fraction = chain_mass / mass_before[owner]
        chain.update(
            type=type[apices][owner], root_order=root_order[apices][owner],
            lateral_root_emergence_possibility=EMERGENCE_IMPOSSIBLE, emergence_cost=0.,
            angle_down=segment_angle_down[owner], angle_roll=segment_angle_roll[owner], length=chain_length,
            radius=r[owner], original_radius=r[owner], initial_radius=r[owner],
            potential_length=np.where(is_new_apex, final[owner], 0.),
            initial_length=np.where(is_new_apex, final[owner], 0.),
            dist_to_ramif=np.where(is_new_apex, d0[owner], d0[owner] - (L[owner] - S[owner] * rank)),
            actual_elongation=np.where(is_new_apex, elongation[owner], S[owner] * (rank - 1)),
            volume=chain_volume, struct_mass=chain_mass, actual_time_since_cells_formation=cells_age,
            thermal_time_since_cells_formation=cells_age * factor[owner], key=chain_keys)
        for resource in self.resources:
            chain[resource.concentration] = np.asarray(ds.get(resource.concentration), dtype=float)[apices][owner]
            for role in ("required", "demand", "consumption_amount", "consumption"):
                name = getattr(resource, role)
                chain[name] = (np.asarray(ds.get(name), dtype=float)[apices][owner] if resource.copied_by_new_elements
                               else np.zeros(owner.size))
        for name in self.SHARED:
            chain[name] = np.asarray(ds.get(name), dtype=float)[apices][owner] * chain_fraction

        # The sites of lateral primordia: the apices (or first segments), then the new chain's elements
        site_apex = np.concatenate([np.arange(apices.size), owner])  # each site's apex (position in apices)
        site_new = np.concatenate([np.zeros(apices.size, bool), np.ones(owner.size, bool)])
        site_parent = np.concatenate([ids[apices], -1 - np.arange(owner.size)])
        site_keys = np.concatenate([key[apices].astype(np.int64), chain_keys])
        site_distance = np.concatenate([new_dist_to_ramif[apices], chain["dist_to_ramif"]])
        site_elongation = np.concatenate([elongation, chain["actual_elongation"]])
        bearer = np.concatenate([apices, np.full(owner.size, -1)])
        has_children = np.concatenate([n_children[apices] > 0, np.zeros(owner.size, bool)])
        at = apices[site_apex]                                      # the parameters and inputs of each site's plant
        order, radius_site, factor_site = root_order[at], radius[at], temperature_modification[at]
        Dmin, IPD = p["Dmin"][at], p["IPD"][at]
        possible = ~has_children
        if self.root_order_limitation:
            possible &= ~(order + 1 > p["root_order_treshold"][at])
        mean = (radius_site - Dmin / 2.) * p["RMD"][at] + Dmin / 2.
        if self.random:
            primordium_radius = random_streams.normal(site_keys, mean, mean * p["CVDD"][at], stream="lateral_radius")
            bearer_angle_roll = np.abs(random_streams.normal(site_keys, 120., 10., stream="bearer_angle_roll"))
            primordium_angle_down = np.abs(random_streams.normal(site_keys, np.where(order == 1, 45., 70.), 10.,
                                                                 stream="lateral_angle_down"))
            primordium_angle_roll = np.abs(random_streams.normal(site_keys, 5., 5., stream="lateral_angle_roll"))
        else:
            primordium_radius = mean
            bearer_angle_roll = np.full(site_keys.size, 120.)
            primordium_angle_down = np.where(order == 1, 45., 70.)
            primordium_angle_roll = np.full(site_keys.size, 5.)
        primordium_radius = np.minimum(primordium_radius, radius_site)
        beyond = possible & (site_distance > IPD)
        forms = beyond & (2. * primordium_radius >= Dmin)                                         # g19
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_site = rate[site_apex]
            time_since_formation = np.where((site_elongation > 0.) & (rate_site > 0.),
                                            (site_distance - IPD) / rate_site, 0.)
        site_distance = np.where(beyond, site_distance - IPD, site_distance)                      # used (g20)
        new_dist_to_ramif[apices] = site_distance[:apices.size]
        chain["dist_to_ramif"] = site_distance[apices.size:]
        old = forms[:apices.size]
        new_angle_roll[apices[old]] = bearer_angle_roll[:apices.size][old]
        chain["angle_roll"] = np.where(forms[apices.size:], bearer_angle_roll[apices.size:], chain["angle_roll"])

        # The primordia
        f = np.flatnonzero(forms)
        primordium_keys = child_keys(site_keys[f], "+")
        plant = {name: values[at[f]] for name, values in p.items()}
        primordia = self._nil_values(plant, TYPE_BEFORE_EMERGENCE, order[f] + 1, primordium_angle_down[f],
                                     primordium_angle_roll[f], 0., primordium_radius[f])
        primordia["soil_temperature"] = soil_temperature[at[f]]                                   # its mother's (g3)
        if self.growth_duration_rule == "simple":                   # determinate laterals, no extender (g18)
            primordia["growth_duration"] = (plant["GDs"] * (2. * primordium_radius[f]) ** 2
                                            * self.lateral_growth_duration_factor(bearer[f]))
        else:
            primordia["growth_duration"] = self._growth_durations(primordium_radius[f], primordium_keys, order[f] + 1,
                                                                  plant)
        primordia["actual_time_since_primordium_formation"] = time_since_formation[f]
        primordia["thermal_time_since_primordium_formation"] = time_since_formation[f] * factor_site[f]
        primordia["key"] = primordium_keys

        # The nodules, on the last segments
        nodules = {}
        nodule_parent = np.zeros(0, dtype=np.int64)
        if self.nodules:
            last = np.flatnonzero(split)                           # each split apex's last segment: its site
            last_site = np.where(n_segments[last] == 1, last, apices.size + first[last] + n_segments[last] - 2)
            children_then = has_children[last_site].astype(int) + forms[last_site]
            drawn = random_streams.uniform(site_keys[last_site], stream="nodule")
            nodule_site = last_site[(children_then < 2)
                                    & (drawn < p["nodule_formation_probability"][apices[last]])]
            nodule_parent = site_parent[nodule_site]
            at_nodule = apices[site_apex[nodule_site]]
            plant = {name: values[at_nodule] for name, values in p.items()}
            bearer_radius = radius[at_nodule]
            nodules = self._nil_values(plant, TYPE_NODULE, root_order[at_nodule] + 1, 90., 0., 0., 0.)
            nodules.update(soil_temperature=soil_temperature[at_nodule], length=bearer_radius, radius=bearer_radius,
                           original_radius=bearer_radius, volume=4 / 3. * pi * bearer_radius ** 3,
                           key=child_keys(site_keys[nodule_site], "+", rank=1))
            nodules["struct_mass"] = (nodules["volume"] * nodules["root_tissue_density"]
                                      * plant["struct_mass_C_content"])

        # The batch: the chains (parents first), the primordia, the nodules
        groups = ((chain, owner.size), (primordia, f.size), (nodules, nodule_parent.size))
        names = set().union(*(group for group, _ in groups))
        values = {name: np.concatenate([np.broadcast_to(np.asarray(group.get(name, self._default(name)), dtype=float),
                                                        (size,)) for group, size in groups]) for name in names}
        segment, apex_label = labels.SubOrgan.RootSegment, labels.SubOrgan.RootApex
        new = NewNodes(
            parent=np.concatenate([chain_parent, site_parent[f], nodule_parent]),
            edge=["<"] * owner.size + ["+"] * (f.size + nodule_parent.size),
            label=np.concatenate([np.where(is_new_apex, apex_label, segment), np.full(f.size, apex_label),
                                  np.full(nodule_parent.size, segment)]).tolist(),
            values=values, new_complex=np.concatenate([np.zeros(owner.size, bool), np.ones(f.size + nodule_parent.size,
                                                                                            bool)]),
            complex_levels=((scales.Organ, labels.Organ.RootInternode),),
            relabel=(ids[cut], segment))
        for name in self.SHARED:
            ds.set(name, shared[name])
        return (new_dist_to_ramif, "length", new_length, "volume", new_volume, "struct_mass", new_mass,
                "actual_time_since_cells_formation", actual_cells, "thermal_time_since_cells_formation",
                thermal_cells, "angle_roll", new_angle_roll, "new_elements", new)

    # ── After growth (RootGrowthModel.post_growth_updating, standalone: the repartition is metafspm's) ─────────

    @postsegmentation
    def _distance_from_tip(self, length, type, distance_from_tip) -> outputs(2):
        """Distance from the tip and the axis' apex (its node id), along the successors from the tips (a fold up the
        tree); the former distance kept."""
        apex = self._apex()
        stop = apex | (type == TYPE_NODULE)
        node_ids = np.asarray(self.data_structure.entity_ids("node"), dtype=float)
        columns = np.column_stack([np.array(length, dtype=float), node_ids])

        def update(level, out):
            nodes = level.nodes
            successors = level.children(np.ones(out.shape[0]), op="sum", edge="<")
            son_distance = level.children(out[:, 0], op="sum", edge="<")
            son_apex = level.children(out[:, 1], op="sum", edge="<")
            tip = stop[nodes] | (successors == 0)
            rows = np.empty((nodes.size, 2))
            rows[:, 0] = np.where(tip, length[nodes], son_distance + length[nodes])
            rows[:, 1] = np.where(tip, node_ids[nodes], son_apex)
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
    def _living_struct_mass(self, struct_mass, living_root_hairs_struct_mass, type) -> outputs(1):
        """The living structural mass (none for a dead element, g22), and the plant's total."""
        ds = self.data_structure
        living = struct_mass + living_root_hairs_struct_mass
        living = np.where(np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD)), 0., living)
        owner = np.asarray(ds.owner("Plant"))
        totals = np.bincount(owner, weights=living, minlength=np.asarray(ds.entity_ids("Plant")).size)
        return living, "total_living_struct_mass", totals
