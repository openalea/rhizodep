"""
rhizodep's root growth as a metafspm StructuralComponent: ArchiSimple's root architecture (openalea.archisimple) with
its growth limited by resources, hexose by default, and the root hairs, as the legacy RootGrowthModel computes it
(openalea.rhizodep.legacy, frozen).

The resources (RootGrowth.resources, Resource descriptions) are drawn from each elongating apex's supply zone and from
each thickening element; their costs are scattered on the elements that supply them. The hooks let a
subclass change the resources and their rules, e.g. nitrogen (test/nitrogen: Root_BRIDGES' regulation).
"""
from dataclasses import dataclass
from math import pi

import numpy as np

from openalea.metafspm.coupling.component import input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.data_structure import random_streams
from openalea.metafspm.solve.decorator import actual, postsegmentation, potential
# The architecture's names (codes, helpers) are this module's too
from openalea.archisimple.architecture import *  # noqa: F401,F403
from openalea.archisimple.architecture import (BY, DESCRIPTOR, MASSIC, NIE, NII, SubOrgan, RootArchitecture, doc,
                                               outputs)


@dataclass(frozen=True)
class Resource:
    """
    A resource of growth: the declared variables that hold it, by role. The growth component limits
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
    supply_rate              the input of its rate of supply to the element during the step (mol g-1 s-1, e.g. a
                             carbon model's import; None: none): a supply zone then holds the elements' amounts plus
                             what they receive over the step, so that the cap on a step's elongation does not depend
                             on the time step
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
    supply_rate: str = None


HEXOSE = Resource(concentration="C_hexose_root", zone="growing_zone_C_hexose_root",
                  required="hexose_possibly_required_for_elongation",
                  available_for_thickening="hexose_available_for_thickening", demand="hexose_growth_demand",
                  consumption_amount="hexose_consumption_by_growth_amount", consumption="hexose_consumption_by_growth",
                  minimum_for_elongation="C_hexose_min_for_elongation",
                  minimum_for_thickening="C_hexose_min_for_thickening", supply_rate="hexose_supply_rate")



@dataclass
class RootGrowth(RootArchitecture):
    """
    Root growth of rhizodep (Rees et al.; Gérault et al.): the root architecture, its growth limited by its resources
    (hexose by default): the apices' elongation by the hexose of their supply zone, the segments' thickening by theirs,
    the costs and the growth respiration taken from the suppliers; the root hairs. The port of the legacy
    RootGrowthModel, bit for bit.
    """
    # --- Inputs ---
    C_hexose_root: float = input_variable(**doc('mol.g-1',
                                                'Hexose concentration of the element (given by a carbon model)',
                                                unit_comment='of hexose per gram of structural mass'),
                                          by='model_carbon', initialize=0., scale=SubOrgan, state_variable_type=MASSIC)
    hexose_supply_rate: float = input_variable(**doc('mol.g-1.s-1',
                                                     "Hexose supplied to the element during the step (given by a "
                                                     "carbon model), available to the supply zones it belongs to",
                                                     value_comment="0: the zones hold the elements' hexose only, "
                                                                   "the legacy's"),
                                               by='model_carbon', initialize=0., scale=SubOrgan)

    # --- State variables, declared by the legacy ---
    growing_zone_C_hexose_root: float = state_variable(**doc('mol.g-1', 'Hexose availability in the growing zone'),
                                                       state_variable_type=NII, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_growth: float = state_variable(**doc('mol.s-1',
                                                               'Hexose consumption rate by growth is coupled to a '
                                                               'root growth model'),
                                                         state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_growth_amount: float = state_variable(**doc('mol',
                                                                      'Hexose consumed by growth in the step'),
                                                                state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    resp_growth: float = state_variable(**doc('mol', 'Growth respiration in the step', unit_comment='of C'),
                                        state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_fungus: float = state_variable(**doc('mol.s-1', 'Hexose consumption rate by fungus'),
                                                         state_variable_type=NIE, initialize=0.0, scale=SubOrgan)

    # --- State variables the legacy writes without declaring them ---
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
    resp_growth_rate: float = state_variable(**doc('mol of C.s-1', 'Growth respiration rate'),
                                             state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    resp_maintenance: float = state_variable(**doc('mol of C', 'Maintenance respiration'),
                                             state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    struct_mass_contributing_to_elongation: float = state_variable(**doc('g',
                                                                         'Structural mass of the supply zone '
                                                                         "contributing to the apex's elongation"),
                                                                   state_variable_type=NIE, initialize=0.0,
                                                                   scale=SubOrgan)

    # --- Plant-scale state ---

    # --- Parameters ---
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
    growing_zone_factor: float = parameter(**doc('adim',
                                                 'Proportionality factor between the radius and the length of the '
                                                 'root apical zone in which C can sustain root elongation',
                                                 references='According to illustrations by Kozlova et al. (2020), the'
                                                            ' length of the growing zone corresponding to the root '
                                                            'cap, meristem and elongation zones is about 8 times the '
                                                            'diameter of the tip.'),
                                           by=BY, default=21.0)
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
    yield_growth: float = parameter(**doc('adim', 'Growth yield',
                                          unit_comment='mol of CO2 per mol of C used for structural mass',
                                          references='We use the range value (0.75-0.85) proposed by Thornley and '
                                                     'Cannell (2000)'),
                                    by=BY, default=0.8)
    growth_duration_rule: str = parameter(
        **doc('adim', "The laterals' growth durations: 'probabilistic' (classes GD_low / GD_medium / GD_high by their "
                      "probabilities) or 'simple' (GDs·d², ArchiSimple's), the growth staying limited by resources"),
        by=BY, default="probabilistic")
    initial_C_hexose_root: float = parameter(**doc('mol.g-1', 'Initial hexose concentration of root segments'),
                                             by=BY, default=0.0001)


    # --- The root hairs and the growth duration classes (RootGrowth's; ArchiSimple has neither) ---
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
    root_hair_length: float = state_variable(**doc('m', 'Example root hair length',
                                                   references='According to the work of Gahoonia et al. (1997), the '
                                                              'root hair maximal length for wheat and barley evolves '
                                                              'between 0.5 and 1.3 mm.'),
                                             state_variable_type=NII, initialize=0.001, scale=SubOrgan)
    total_root_hairs_number: float = state_variable(**doc('adim',
                                                          'Example root hairs number on segment external surface',
                                                          value_comment='30 * (1.6e-4 / radius) * length * 1e3',
                                                          references=' According to the work of Gahoonia et al. '
                                                                     '(1997), the root hair density is about 30 hairs'
                                                                     ' per mm for winter wheat, for a root radius of '
                                                                     'about 0.16 mm.'),
                                                    state_variable_type=NIE, initialize=41.142857142857146,
                                                    scale=SubOrgan)
    root_hairs_struct_mass_produced: float = state_variable(**doc('g', '', unit_comment='of dry weight'),
                                                            state_variable_type=NIE, initialize=0, scale=SubOrgan)
    actual_length_with_hairs: float = state_variable(**doc('m', 'Length of the element bearing root hairs'),
                                                     state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
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
    dead_root_hairs_number: float = state_variable(**doc('adim', 'Number of dead root hairs on the element'),
                                                   state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    living_root_hairs_number: float = state_variable(**doc('adim', 'Number of living root hairs on the element'),
                                                     state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    root_hairs_struct_mass: float = state_variable(**doc('g', "Structural mass of the element's root hairs"),
                                                   state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    root_hairs_volume: float = state_variable(**doc('m3', "Volume of the element's root hairs"),
                                              state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
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

    # The collar supply: an apex's supply zone reaching the base continues into the collar's other children by
    # volume ("collar", their own collar child excluded); "none": the legacy's, the zone stops at the base
    collar_supply: str = parameter(**doc('adim', "The supply zone's continuation into the collar's children"),
                                   by=BY, default="collar")

    resources = (HEXOSE,)
    ZEROED_AT_START = ("resp_growth", "hexose_consumption_by_fungus", "root_hairs_struct_mass_produced")
    # The root hairs' variables: nil in a primordium, copied by a new segment, shared by segmentation (their amounts)
    ROOT_HAIRS = ("root_hair_length", "actual_length_with_hairs", "living_root_hairs_number", "dead_root_hairs_number",
                  "total_root_hairs_number", "actual_time_since_root_hairs_emergence_started",
                  "thermal_time_since_root_hairs_emergence_started", "actual_time_since_root_hairs_emergence_stopped",
                  "thermal_time_since_root_hairs_emergence_stopped", "all_root_hairs_formed", "root_hairs_struct_mass",
                  "root_hairs_struct_mass_produced", "initial_living_root_hairs_struct_mass",
                  "living_root_hairs_struct_mass")
    ROOT_HAIRS_SHARED = ("initial_living_root_hairs_struct_mass", "root_hairs_struct_mass",
                         "root_hairs_struct_mass_produced", "living_root_hairs_struct_mass", "living_root_hairs_number",
                         "dead_root_hairs_number", "total_root_hairs_number")
    NIL = RootArchitecture.NIL + ROOT_HAIRS + ("resp_growth", "hexose_growth_demand",
                                               "hexose_consumption_by_growth_amount", "hexose_consumption_by_growth",
                                               "hexose_consumption_by_fungus", "hexose_possibly_required_for_elongation")
    COPIED = RootArchitecture.COPIED + ROOT_HAIRS + ("resp_growth", "hexose_consumption_by_fungus")
    SHARED = RootArchitecture.SHARED + ROOT_HAIRS_SHARED + ("resp_growth", "hexose_growth_demand",
                                                           "hexose_consumption_by_growth_amount",
                                                           "hexose_consumption_by_growth", "hexose_consumption_by_fungus")
    SEGMENTATION_PARAMETERS = RootArchitecture.SEGMENTATION_PARAMETERS + (
        "GD_highest", "GD_high", "GD_medium", "GD_low", "GD_prob_low", "GD_prob_medium")
    ELEMENT_PARAMETERS = ("root_hair_radius", "root_hairs_lifespan")
    resource_limited = True
    # Whether the last element of a supply zone (the fraction of it the zone takes) is subject to supply_included
    supply_gates_the_last_element = False

    def _growth_durations(self, radius, keys, root_order, parameters):
        """The "probabilistic" rule (RootGrowthModel.calculate_growth_duration): a duration drawn among classes (low,
        medium, high) by their probabilities, uniformly within the class (GD_by_frequency), or the class's value; the
        classes exclusive, the thinnest roots taking GD_high. *parameters*: {name: a value per primordium}."""
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

    @potential
    def _initial_living_root_hairs_struct_mass(self, living_root_hairs_struct_mass):
        """The root hairs' living mass at the step's start."""
        return np.array(living_root_hairs_struct_mass)

    @postsegmentation
    def _living_struct_mass(self, struct_mass, living_root_hairs_struct_mass, type) -> outputs(1):
        """The living structural mass, the root hairs' included (none for a dead element), and the plant's
        total."""
        ds = self.data_structure
        living = struct_mass + living_root_hairs_struct_mass
        living = np.where(np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD)), 0., living)
        owner = np.asarray(ds.owner("Plant"))
        totals = np.bincount(owner, weights=living, minlength=np.asarray(ds.entity_ids("Plant")).size)
        return living, "total_living_struct_mass", totals

    @classmethod
    def _initial_resources(cls, parameters, elements) -> None:
        """The seedling's hexose: the base's, the apex's when it has a length."""
        elements[0]["values"]["C_hexose_root"] = parameters["initial_C_hexose_root"]
        if parameters["initial_apex_length"] > 0.:
            elements[-1]["values"]["C_hexose_root"] = parameters["initial_C_hexose_root"]

    # ── Hooks: the legacy's by default; a subclass overrides them, e.g. for nitrogen ──────────────

    def supply_included(self, length, contributions):
        """The elements a supply zone takes from when it walks through them (the legacy: those with a length).
        *contributions*: {resource concentration: its amount in each element}."""
        return length > 0.

    def elongated_length(self, selected, start_length, radius, time, zone):
        """RootGrowthModel.elongated_length on the *selected* apices: Michaelis-Menten on the supply zone's hexose above its threshold (no elongation below). *zone*: {resource
        concentration: its concentration in each apex's supply zone}."""
        EL = self._parameter("EL", selected)
        C = zone["C_hexose_root"][selected]
        Km = self._parameter("Km_elongation", selected)
        with np.errstate(divide="ignore", invalid="ignore"):
            elongation = np.where(C > self._parameter("C_hexose_min_for_elongation", selected),
                                  EL * 2. * radius * C / (Km + C) * time, 0.)
        return start_length + elongation

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

    def supply_zones(self, supplied, radius, length, volume, struct_mass) -> dict:
        """The supply zones of the apices *supplied*: each resource's amount (required) and concentration (zone) in
        each, and the structural mass contributing; {resource concentration: its zone concentration}."""
        ds = self.data_structure
        growing_zone_factor = self._parameter("growing_zone_factor")
        owner, _, contributions, mass_contributions = self._supply(supplied, radius, length, volume, struct_mass,
                                                                   growing_zone_factor)
        mass = np.zeros(length.size)
        np.add.at(mass, owner, mass_contributions)
        contributing = np.array(ds.get("struct_mass_contributing_to_elongation"), dtype=float)
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
        ds.set("struct_mass_contributing_to_elongation", contributing)
        return zone

    def _supply(self, supplied, radius, length, volume, struct_mass, growing_zone_factor):
        """
        The supply zones of the apices *supplied* (RootGrowthModel.calculating_supply_for_elongation): from each apex
        towards the base, the elements' volumes until the zone's (growing_zone_factor · r³ · π), the last one in part;
        with the collar supply, a zone reaching the base continues into the collar's other children, by their
        volume. (owner, supplier, {resource concentration: contribution}, mass contribution).
        """
        ds = self.data_structure
        amounts = {}
        for resource in self.resources:                 # each element's amount, and what it receives in the step
            amount = np.asarray(ds.get(resource.concentration), dtype=float) * struct_mass
            if resource.supply_rate is not None:
                amount = amount + np.asarray(ds.get(resource.supply_rate), dtype=float) * struct_mass * float(self.time_step)
            amounts[resource.concentration] = amount
        include = np.asarray(self.supply_included(length, amounts), dtype=bool)
        budget = growing_zone_factor * radius * radius ** 2 * pi
        owner, supplier, taken = ds.path_contributions(budget, volume, volume, where=supplied, include=include,
                                                       order=None)
        share = np.where(volume[supplier] > 0., taken / np.where(volume[supplier] > 0., volume[supplier], 1.), 0.)
        contributions = {name: amount[supplier] * share for name, amount in amounts.items()}
        mass = struct_mass[supplier] * share
        if self.supply_gates_the_last_element:
            kept = include[supplier]
            mass = np.where(kept, mass, 0.)
            contributions = {name: np.where(kept, values, 0.) for name, values in contributions.items()}
        if self.collar_supply != "none":
            collar = self._collar_supply(supplied, budget, owner, taken, length, volume, struct_mass, amounts)
            owner, supplier = np.concatenate([owner, collar[0]]), np.concatenate([supplier, collar[1]])
            contributions = {name: np.concatenate([values, collar[2][name]]) for name, values in contributions.items()}
            mass = np.concatenate([mass, collar[3]])
        return owner, supplier, contributions, mass

    def _collar_supply(self, supplied, budget, owner, taken, length, volume, struct_mass, amounts):
        """The zones reaching the base with some of their volume left continue into the collar's other children (the
        base's children but the zone's own), each by its share of the volume left, at most its own (Root_BRIDGES'
        calculating_supply_for_elongation_opt). (owner, supplier, {resource concentration: contribution}, mass)."""
        ds = self.data_structure
        parents = np.asarray(ds.parents())
        n = parents.size
        base = parents < 0
        grandparent = np.where(parents >= 0, parents[np.maximum(parents, 0)], 0)
        collar = (parents >= 0) & (grandparent < 0)                 # the base's children
        # Each element's collar child (its ancestor among them; the base itself for the base), by pointer doubling
        ancestor = np.where(base | collar, np.arange(n), parents)
        while True:
            jumped = ancestor[ancestor]
            if np.array_equal(jumped, ancestor):
                break
            ancestor = jumped
        left = np.asarray(budget, dtype=float)[supplied] - np.bincount(owner, weights=taken, minlength=n)[supplied]
        apices = np.flatnonzero(supplied)
        root_of = np.where(base, np.arange(n), parents[ancestor])     # each element's base
        reaching = (left > 0.) & (length[root_of[apices]] > 0.)     # zones that took the whole base
        apices, left = apices[reaching], left[reaching]
        children = np.flatnonzero(collar & (length > 0.))
        empty = (np.zeros(0, dtype=np.int64),) * 2 + ({name: np.zeros(0) for name in amounts}, np.zeros(0))
        if apices.size == 0 or children.size == 0:
            return empty
        # The pairs (apex, collar child of its plant but its own)
        apex_of, child_of = np.meshgrid(np.arange(apices.size), np.arange(children.size), indexing="ij")
        apex_of, child_of = apex_of.ravel(), child_of.ravel()
        pairs = ((root_of[children[child_of]] == root_of[apices[apex_of]])
                 & (children[child_of] != ancestor[apices[apex_of]]))
        apex_of, child = apex_of[pairs], children[child_of[pairs]]
        total = np.bincount(apex_of, weights=volume[child], minlength=apices.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            source = np.minimum(left[apex_of] * volume[child] / total[apex_of], volume[child])
            share = np.where(volume[child] > 0., source / volume[child], 0.)
        return (apices[apex_of], child, {name: amount[child] * share for name, amount in amounts.items()},
                struct_mass[child] * share)

    # ── The thickening's resources ─────────────────────────────────────────────────────────────────────────────

    def thicken_nodules(self, nodule, struct_mass, radius, temperature_factor, time_step, theoretical_radius,
                        potential_radius) -> None:
        """Each resource's amount available to thickening (the element's; a nodule's and its bearer's), and the
        nodules' potential thickening from it."""
        ds = self.data_structure
        parents = np.asarray(ds.parents())
        nodule_max_radius = self._parameter("nodule_max_radius")
        new_theoretical_radius, new_potential_radius = theoretical_radius, potential_radius
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

    def limited_potential_radius(self, normal, theoretical_radius, potential_radius, radius, initial_radius, length,
                                 temperature_factor, time_step) -> None:
        """The segments' potential radius: their pipe-model radius, at most what their hexose allows in the step."""
        ds = self.data_structure
        new_theoretical_radius, new_potential_radius = theoretical_radius, potential_radius
        C_hexose_root = np.asarray(ds.get("C_hexose_root"), dtype=float)
        C_hexose_min_for_thickening = self._parameter("C_hexose_min_for_thickening")
        selected = (normal & (length > 0.) & (new_theoretical_radius > radius)
                    & (C_hexose_root > C_hexose_min_for_thickening))
        rate = self.thickening_rate(selected)
        rate = rate * temperature_factor[selected]
        largest = (1 + rate * time_step) * initial_radius[selected]
        new_potential_radius[selected] = np.where(new_theoretical_radius[selected] > largest, largest,
                                                  new_theoretical_radius[selected])

    # ── Actual growth (RootGrowthModel.actual_growth_and_corresponding_respiration) ───────────────────────────

    # Hooks of actual growth: the legacy's, by hexose alone

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
        nodule_radius = (3. * volume_max / (4. * pi)) ** (1. / 3.)      # the sphere of the volume reached
        possible_radius = np.where(type == TYPE_NODULE, nodule_radius, np.sqrt(volume_max / (length * pi)))
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
        that elongate emerge. Each resource's demand and consumptions are written.
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
            grows = ~np.isin(type, (TYPE_DEAD, TYPE_JUST_DEAD))
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
                          / (pi * initial_radius ** 2))
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

        # The costs: each elongation's on its suppliers, each thickening's on the element, a nodule's also on its parent
        owner, supplier, contributions, _ = self._supply(supplied, radius, length, volume, struct_mass,
                                                         growing_zone_factor)
        thickened, nodules = np.flatnonzero(thickening), np.flatnonzero(nodule)
        target = np.concatenate([supplier, thickened, parents[nodules]])
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
            value = np.concatenate([elongation_events, own_share[thickened], parent_share[nodules]])
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
            actual_since_emergence[emerges] = self._actual_time(time_since_emergence[emerges],
                                                                temperature_modification[emerges])
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

    # ── The root hairs (RootGrowthModel.root_hairs_dynamics_opt) ──────────────────────────────────────────────

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
            volume = (root_hair_radius ** 2 * pi) * out["root_hair_length"] * total
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
