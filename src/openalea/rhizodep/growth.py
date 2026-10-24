"""
rhizodep's root growth as a metafspm StructuralComponent: the root architecture (openalea.rhizodep.architecture) with
its growth limited by resources, hexose by default, as the legacy RootGrowthModel computes it (openalea.rhizodep.legacy,
frozen): bit for bit the same results (devplan_metafspm.md).

The resources (RootGrowth.resources, Resource descriptions) are drawn from each elongating apex's supply zone and from
each thickening element; their costs are scattered on the elements that supply them. The hooks (devplan §21) let a
subclass change the resources and their rules, e.g. nitrogen (test/nitrogen: Root_BRIDGES' regulation).
"""
from dataclasses import dataclass
from math import pi

import numpy as np

from openalea.metafspm.coupling.component import input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import actual, postsegmentation
# The architecture's names (codes, helpers) are this module's too, as before the split
from openalea.rhizodep.architecture import *  # noqa: F401,F403
from openalea.rhizodep.architecture import (BY, DESCRIPTOR, MASSIC, NIE, NII, SubOrgan, RootArchitecture, doc,
                                            outputs, power)


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



@dataclass
class RootGrowth(RootArchitecture):
    """
    Root growth of rhizodep (Rees et al.; Gérault et al.): the root architecture, its growth limited by its resources
    (hexose by default): the apices' elongation by the hexose of their supply zone, the segments' thickening by theirs,
    the costs and the growth respiration taken from the suppliers; the root hairs. The port of the legacy
    RootGrowthModel, bit for bit (devplan §8).
    """
    # --- Inputs ---
    C_hexose_root: float = input_variable(**doc('mol.g-1',
                                                'Hexose concentration of the element (given by a carbon model)',
                                                unit_comment='of hexose per gram of structural mass'),
                                          by='model_carbon', initialize=0., scale=SubOrgan, state_variable_type=MASSIC)

    # --- State variables, declared by the legacy ---
    growing_zone_C_hexose_root: float = state_variable(**doc('mol.g-1', 'Hexose availability in the growing zone'),
                                                       state_variable_type=NII, initialize=0.0, scale=SubOrgan)  # g8
    hexose_consumption_by_growth: float = state_variable(**doc('mol.s-1',
                                                               'Hexose consumption rate by growth is coupled to a '
                                                               'root growth model'),
                                                         state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    hexose_consumption_by_growth_amount: float = state_variable(**doc('mol',                         # g8
                                                                      'Hexose consumed by growth in the step'),
                                                                state_variable_type=NIE, initialize=0.0, scale=SubOrgan)
    resp_growth: float = state_variable(**doc('mol', 'Growth respiration in the step', unit_comment='of C'),  # g8
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
    simple_growth_duration: bool = parameter(
        **doc('adim', "The legacy's merged switch (g10): C-free elongation and thickening, emergence at once, and "
                      "growth durations GDs·d². The growth duration rule alone is growth_duration_rule; the C-free "
                      "development is RootArchitecture's",
              references='(Pagès et al., 2014)'),
        by=BY, default=False)
    growth_duration_rule: str = parameter(
        **doc('adim', "The laterals' growth durations: 'probabilistic' (classes GD_low / GD_medium / GD_high by their "
                      "probabilities) or 'simple' (GDs·d², ArchiSimple's), the growth staying limited by resources "
                      "(g10)"),
        by=BY, default="probabilistic")
    initial_C_hexose_root: float = parameter(**doc('mol.g-1', 'Initial hexose concentration of root segments'),
                                             by=BY, default=0.0001)


    # The collar supply (g9): an apex's supply zone reaching the base continues into the collar's other children by
    # volume ("collar", their own collar child excluded); "rootbridges": Root_BRIDGES', counting an axis' own collar
    # child again; "none": the legacy's, the zone stops at the base
    collar_supply: str = parameter(**doc('adim', "The supply zone's continuation into the collar's children (g9)"),
                                   by=BY, default="collar")

    resources = (HEXOSE,)
    ZEROED_AT_START = ("resp_growth", "hexose_consumption_by_fungus")
    NIL = RootArchitecture.NIL + ("resp_growth", "hexose_growth_demand", "hexose_consumption_by_growth_amount",
                                  "hexose_consumption_by_growth", "hexose_consumption_by_fungus",
                                  "hexose_possibly_required_for_elongation")
    COPIED = RootArchitecture.COPIED + ("resp_growth", "hexose_consumption_by_fungus")
    SHARED = RootArchitecture.SHARED + ("resp_growth", "hexose_growth_demand", "hexose_consumption_by_growth_amount",
                                        "hexose_consumption_by_growth", "hexose_consumption_by_fungus")
    SEGMENTATION_PARAMETERS = RootArchitecture.SEGMENTATION_PARAMETERS
    LEGACY = dict(RootArchitecture.LEGACY, collar_supply="none")
    # Whether the last element of a supply zone (the fraction of it the zone takes) is subject to supply_included
    supply_gates_the_last_element = False

    @classmethod
    def _initial_resources(cls, parameters, elements) -> None:
        """The seedling's hexose: the base's, the apex's when it has a length."""
        elements[0]["values"]["C_hexose_root"] = parameters["initial_C_hexose_root"]
        if parameters["initial_apex_length"] > 0.:
            elements[-1]["values"]["C_hexose_root"] = parameters["initial_C_hexose_root"]

    # ── Hooks (devplan §21): the legacy's by default; a subclass overrides them, e.g. for nitrogen ──────────────

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
        The supply zones of the apices *supplied* (RootGrowthModel.calculating_supply_for_elongation), element by
        element in the legacy's order (the apices visited children first, each zone from the tip to the base):
        (owner, supplier, {resource concentration: contribution}, mass contribution).
        """
        ds = self.data_structure
        concentrations = {r.concentration: np.asarray(ds.get(r.concentration), dtype=float) for r in self.resources}
        amounts = {name: concentration * struct_mass for name, concentration in concentrations.items()}
        include = np.asarray(self.supply_included(length, amounts), dtype=bool)
        budget = growing_zone_factor * radius * power(radius, 2) * pi
        if self.collar_supply != "none":
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
            remaining, current, previous, own = budget[apex], apex, apex, -1
            collar_set = set(collar_children.tolist())
            while remaining > 0:
                if remaining > volume[current]:
                    if length[current] > 0.:
                        if include[current]:
                            events.append((apex, current, None))
                            remaining = remaining - volume[current]
                        if parents[current] < 0:            # the base: the collar's other children
                            excluded = {previous} if self.collar_supply == "rootbridges" else {previous, own}
                            children = [k for k in collar_children.tolist()
                                        if owner_plant[k] == owner_plant[apex] and k not in excluded and length[k] > 0.]
                            # Python's sum of floats (compensated since 3.12), as Root_BRIDGES sums them
                            total = sum([float(volume[k]) for k in children])
                            for child in children:
                                source = min(remaining * volume[child] / total, volume[child])
                                events.append((apex, child, source))
                            if remaining <= total:
                                remaining = 0.
                    if parents[current] < 0:
                        break
                    if current in collar_set:
                        own = current                       # the zone's own collar child
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
        """The segments' potential radius: their pipe-model radius, at most what their hexose allows in the step
        (none with simple_growth_duration)."""
        ds = self.data_structure
        new_theoretical_radius, new_potential_radius = theoretical_radius, potential_radius
        C_hexose_root = np.asarray(ds.get("C_hexose_root"), dtype=float)
        C_hexose_min_for_thickening = self._parameter("C_hexose_min_for_thickening")
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

    # ── Actual growth (RootGrowthModel.actual_growth_and_corresponding_respiration) ───────────────────────────

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
        if self.legacy_nodule_radius:                   # g5: a constant
            nodule_radius = (3. / (4. * pi)) ** (1. / 3.)
        else:                                           # the sphere of the volume reached
            nodule_radius = power(3. * volume_max / (4. * pi), 1. / 3.)
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
