"""
Root_BRIDGES' nitrogen regulation of growth as a subclass of RootGrowth's hooks (devplan_metafspm.md, §21): the test
that the hooks are enough. Compared bit for bit with Root_BRIDGES' RootGrowthModelCoupled itself (test_rootbridges.py),
so its quirks are reproduced too:
- its root thickening rate uses the nodules' Km values, and halves the whole rate;
- its C-limited elongation volume divides the C content net of the amino acids' before the yield, its cost after;
- its root hairs are regulated by the plain mean of the C and N factors, its elongation by their C share.

Besides nitrogen, Root_BRIDGES' differences from rhizodep are its options: the collar supply, a primordium's soil
temperature from its mother, the pipe model that sees its children (g16 fixed), and a supply zone taking only
elements holding both hexose and amino acids.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import input_variable, parameter, state_variable
from openalea.rhizodep.growth import (DESCRIPTOR, HEXOSE, NIE, NII, BY, TYPE_ADVENTITIOUS_BEFORE_EMERGENCE, Resource,
                                      RootGrowth, SubOrgan, doc)

AMINO_ACIDS = Resource(concentration="AA", zone="growing_zone_amino_acids",
                       required="amino_acids_possibly_required_for_elongation",
                       available_for_thickening="amino_acids_available_for_thickening",
                       demand="amino_acids_growth_demand",
                       consumption_amount="amino_acids_consumption_by_growth_amount",
                       consumption="amino_acids_consumption_by_growth",
                       minimum_for_elongation="C_AA_min_for_elongation", minimum_for_thickening=None,
                       copied_by_new_elements=False, costs_by_available=True)


@dataclass
class RootGrowthN(RootGrowth):
    """RootGrowth co-limited by amino acids, as Root_BRIDGES' RootGrowthModelCoupled."""

    # --- Inputs ---
    AA: float = input_variable(**doc('mol.g-1', 'Amino acids concentration of the element'), by='model_nitrogen',
                               initialize=0., scale=SubOrgan, state_variable_type='massic_concentration')
    nitrate_transporters_affinity_factor: float = input_variable(
        **doc('adim', "Factor of the nitrate transporters' affinity (a signal of the laterals' growth duration)"),
        by='model_nitrogen', initialize=1., scale=SubOrgan)

    # --- The amino acids' variables of growth ---
    growing_zone_amino_acids: float = state_variable(**doc('mol.g-1', "Amino acids in the apex's supply zone"),
                                                     state_variable_type=NIE, initialize=0., scale=SubOrgan)
    amino_acids_possibly_required_for_elongation: float = state_variable(
        **doc('mol', "Amino acids of the apex's supply zone"), state_variable_type=NIE, initialize=0., scale=SubOrgan)
    amino_acids_available_for_thickening: float = state_variable(
        **doc('mol', "Amino acids available to the element's thickening"), state_variable_type=NIE, initialize=0.,
        scale=SubOrgan)
    amino_acids_growth_demand: float = state_variable(**doc('mol', "Amino acids the potential growth needs"),
                                                      state_variable_type=NIE, initialize=0., scale=SubOrgan)
    amino_acids_consumption_by_growth_amount: float = state_variable(
        **doc('mol', 'Amino acids consumed by growth in the step'), state_variable_type=NIE, initialize=0.,
        scale=SubOrgan)
    amino_acids_consumption_by_growth: float = state_variable(
        **doc('mol.s-1', 'Rate of amino acids consumption by growth'), state_variable_type=NIE, initialize=0.,
        scale=SubOrgan)
    hexose_growth_regulation: float = state_variable(**doc('adim', "Hexose's share of the elongation's regulation"),
                                                     state_variable_type=NII, initialize=0., scale=SubOrgan)
    AA_growth_regulation: float = state_variable(**doc('adim', "Amino acids' share of the elongation's regulation"),
                                                 state_variable_type=NII, initialize=0., scale=SubOrgan)

    # --- The nodal roots' emergence synchronised with the shoot (synchronize_adventitious_emergence) ---
    nodal_awaiting_synchronisation: int = state_variable(
        **doc('adim', 'A nodal primordium not synchronised with the shoot yet (Root_BRIDGES\' '
                      'adventitous_primordia_to_emerge)'),
        state_variable_type=DESCRIPTOR, initialize=0, scale=SubOrgan, dtype='int')
    synchronize_adventitious_emergence: bool = parameter(
        **doc('adim', "Whether the nodal roots' emergence follows the shoot's phytomers"), by=BY, default=False)

    # The shoot's delays of the next nodal emergences ({plant vid: [thermal time, ...]}), consumed one per
    # synchronised primordium, as Root_BRIDGES consumes its "adventitious_to_emerge": a list, which metafspm does not
    # declare yet
    nodal_emergence_delays: dict = None

    # --- Parameters (Root_BRIDGES') ---
    Km_elongation_amino_acids: float = parameter(**doc('mol.g-1', 'Affinity of elongation for amino acids'), by=BY,
                                                 default=1250 * 1e-6 / 5)
    Km_nodule_thickening_amino_acids: float = parameter(**doc('mol.g-1', 'Affinity of thickening for amino acids'),
                                                        by=BY, default=1250 * 1e-6 / 5. * 100)
    struct_mass_N_content: float = parameter(**doc('mol.g-1', 'Nitrogen content of the structural mass'), by=BY,
                                             default=0.005 / 14)
    yield_growth_N: float = parameter(**doc('adim', 'Yield of nitrogen in growth'), by=BY, default=1.)
    r_Nm_AA: float = parameter(**doc('adim', 'Nitrogen atoms per amino acid'), by=BY, default=1.4)
    r_C_AA: float = parameter(**doc('adim', 'Carbon atoms per amino acid'), by=BY, default=5)
    C_AA_min_for_elongation: float = parameter(**doc('mol.g-1', 'Amino acids kept in the supply zone'), by=BY,
                                               default=0 * 1e-5 * (6 * 12 / 0.44) * 0.005 / 14 / 1.4)

    resources = (HEXOSE, AMINO_ACIDS)
    collar_supply = True
    new_element_soil_temperature = "mother"
    legacy_pipe_model = False
    supply_gates_the_last_element = True

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        super().initiate_plant(g, plant, parameters)
        awaiting = g.property("nodal_awaiting_synchronisation")
        for vid in g.components_at_scale(plant, g.scales.SubOrgan):
            if g.property("type").get(vid) == TYPE_ADVENTITIOUS_BEFORE_EMERGENCE:
                awaiting[vid] = 1

    @classmethod
    def transplanted(cls, g, plant, record, vids):
        """On a transplanted Root_BRIDGES state: the primordia not synchronised yet (its adventitous_primordia_to_emerge)."""
        awaiting = g.property("nodal_awaiting_synchronisation")
        for vid in vids.values():
            awaiting[vid] = 0
        for legacy_id in record.meta.get("adventitious_primordia_to_emerge", {}):
            awaiting[vids[legacy_id]] = 1

    def synchronise_emergence(self, type, thermal_time_since_primordium_formation, emergence_delay_in_thermal_time):
        """Root_BRIDGES' synchronisation: when visited, the next nodal primordium (the earliest delay among those
        not synchronised yet) takes the shoot's next delay, its time since formation restarting; several in one step
        if the next one is visited later."""
        if not self.synchronize_adventitious_emergence:
            return thermal_time_since_primordium_formation, emergence_delay_in_thermal_time
        ds = self.data_structure
        awaiting = np.array(ds.get("nodal_awaiting_synchronisation"))
        vertex_index = np.asarray(ds.get("vertex_index"))
        plants = np.asarray(ds.entity_ids("Plant"))[np.asarray(ds.owner("Plant"))]
        delays = self.nodal_emergence_delays or {}

        def next_one(plant):
            candidates = np.flatnonzero((awaiting == 1) & (plants == plant))
            if candidates.size == 0:
                return None
            candidates = candidates[np.argsort(vertex_index[candidates], kind="stable")]
            return candidates[np.argmin(emergence_delay_in_thermal_time[candidates])]

        following = {plant: next_one(plant) for plant in np.unique(plants).tolist()}
        for element in ds.order("post", convention="openalea").tolist():
            plant = int(plants[element])
            if (type[element] == TYPE_ADVENTITIOUS_BEFORE_EMERGENCE and element == following[plant]
                    and len(delays.get(plant, [])) > 0):
                thermal_time_since_primordium_formation[element] = 0
                emergence_delay_in_thermal_time[element] = delays[plant][0]
                awaiting[element] = 0
                following[plant] = next_one(plant)
                delays[plant].pop(0)
        ds.set("nodal_awaiting_synchronisation", awaiting)
        return thermal_time_since_primordium_formation, emergence_delay_in_thermal_time

    def _C_share_of_amino_acids(self, selected=None):
        """Root_BRIDGES' prop_C_from_amino_acids: the share of the structure's carbon the amino acids bring."""
        p = self._parameter
        return (p("r_C_AA", selected) * p("struct_mass_N_content", selected) / p("r_Nm_AA", selected)
                / (p("struct_mass_C_content", selected)))

    def supply_included(self, length, contributions):
        return (length > 0.) & (contributions["C_hexose_root"] > 0.) & (contributions["AA"] > 0.)

    def elongated_length(self, selected, start_length, radius, time, zone):
        p = self._parameter
        EL = p("EL", selected)
        if self.simple_growth_duration:
            return start_length + EL * 2. * radius * time
        ds = self.data_structure
        C, AA = zone["C_hexose_root"][selected], zone["AA"][selected]
        share = self._C_share_of_amino_acids(selected)
        hexose_regulation = (1 - share) * C / (C + p("Km_elongation", selected))
        AA_regulation = share * AA / (AA + p("Km_elongation_amino_acids", selected))
        limitation = hexose_regulation + AA_regulation
        potential_elongation = EL * 2. * radius * time
        elongates = (C > p("C_hexose_min_for_elongation", selected)) & (AA > p("C_AA_min_for_elongation", selected))
        for name, values in (("hexose_growth_regulation", hexose_regulation), ("AA_growth_regulation", AA_regulation)):
            written = np.array(ds.get(name), dtype=float)
            written[selected] = values
            ds.set(name, written)
        return start_length + np.where(elongates, potential_elongation * limitation, 0.)

    def thickening_rate(self, selected):
        p, ds = self._parameter, self.data_structure
        C, AA = ds.get("C_hexose_root")[selected], ds.get("AA")[selected]
        with np.errstate(divide="ignore", invalid="ignore"):
            return p("relative_root_thickening_rate_max", selected) / (
                ((1 + p("Km_nodule_thickening", selected)) / C)
                + ((1 + p("Km_nodule_thickening_amino_acids", selected)) / AA)) / 2

    def nodule_thickening_rate(self, selected, regulating):
        p = self._parameter
        with np.errstate(divide="ignore", invalid="ignore"):
            return p("relative_nodule_thickening_rate_max", selected) / (
                ((1 + p("Km_nodule_thickening", selected)) / regulating["C_hexose_root"])
                * ((1 + p("Km_nodule_thickening_amino_acids", selected)) / regulating["AA"]))

    def growth_demand(self, delta_volume, root_tissue_density):
        p = self._parameter
        amino_acids = (delta_volume * root_tissue_density * p("struct_mass_N_content") / p("yield_growth_N")
                       / p("r_Nm_AA"))
        carbon = np.maximum(0., (delta_volume * root_tissue_density * p("struct_mass_C_content") / p("yield_growth"))
                            - amino_acids * p("r_C_AA"))
        hexose = carbon / 6.
        return {"C_hexose_root": hexose, "AA": np.where(hexose > 0., np.maximum(0., amino_acids), 0.)}

    def growth_continues(self, demand):
        return (demand["C_hexose_root"] > 0.) & (demand["AA"] > 0.)

    def can_grow(self, for_elongation, for_thickening):
        return (((for_elongation["C_hexose_root"] + for_thickening["C_hexose_root"]) > 0.)
                & ((for_elongation["AA"] + for_thickening["AA"]) > 0.))

    def elongation_volume_max(self, initial_volume, for_elongation, root_tissue_density):
        p = self._parameter
        C_content_of_amino_acids = p("struct_mass_N_content") * p("r_C_AA") / p("r_Nm_AA")
        C_content_of_hexose = p("struct_mass_C_content") - C_content_of_amino_acids
        volume_max_C = (initial_volume + for_elongation["C_hexose_root"] * 6.
                        / (root_tissue_density * C_content_of_hexose) * p("yield_growth"))
        volume_max_N = (initial_volume + for_elongation["AA"] * p("r_Nm_AA")
                        / (root_tissue_density * p("struct_mass_N_content")) * p("yield_growth_N"))
        return np.minimum(volume_max_C, volume_max_N)

    def elongation_costs(self, delta_volume, root_tissue_density):
        p = self._parameter
        amino_acids = (delta_volume * root_tissue_density * p("struct_mass_N_content") / p("yield_growth_N")
                       / p("r_Nm_AA"))
        hexose = (((delta_volume * root_tissue_density * p("struct_mass_C_content") / p("yield_growth"))
                   - amino_acids * p("r_C_AA")) / 6)
        return {"C_hexose_root": hexose, "AA": amino_acids}

    def thickening(self, type, initial_radius, length, potential_radius, root_tissue_density, for_thickening):
        p, volume_of = self._parameter, self._volume
        C_content, N_content = p("struct_mass_C_content"), p("struct_mass_N_content")
        yield_growth, yield_growth_N, r_C_AA, r_Nm_AA = p("yield_growth"), p("yield_growth_N"), p("r_C_AA"), p(
            "r_Nm_AA")
        hexose, amino_acids = for_thickening["C_hexose_root"], for_thickening["AA"]
        with np.errstate(divide="ignore", invalid="ignore"):
            increase_C = (hexose * 6. * yield_growth / (root_tissue_density * C_content)
                          + amino_acids * r_C_AA * yield_growth / (root_tissue_density * C_content))
            increase_N = amino_acids * r_Nm_AA * yield_growth_N / (root_tissue_density * N_content)
            volume_max = volume_of(type, initial_radius, length) + np.minimum(increase_C, increase_N)
            possible_radius = np.where(type == 12, (3. / (4. * np.pi)) ** (1. / 3.),
                                       np.sqrt(volume_max / (length * np.pi)))
            fits = possible_radius <= potential_radius
            radius = np.where(fits, possible_radius, potential_radius)
            net_increase = volume_of(type, radius, length) - volume_of(type, initial_radius, length)
            hexose_ratio = 6 * hexose / (r_C_AA * amino_acids + 6 * hexose)
            C_consumption = net_increase * root_tissue_density * C_content / yield_growth
            N_consumption = net_increase * root_tissue_density * N_content / yield_growth_N
            C_limited = C_consumption * (1 - hexose_ratio) > N_consumption * (r_C_AA / r_Nm_AA)
            amino_acids_cost = np.where(C_limited, C_consumption * (1 - hexose_ratio) / r_C_AA,
                                        N_consumption / r_Nm_AA)
            hexose_cost = np.where(C_limited, C_consumption * hexose_ratio / 6,
                                   (C_consumption - amino_acids_cost * r_C_AA) / 6)
        return radius, {"C_hexose_root": np.where(fits, hexose, hexose_cost),
                        "AA": np.where(fits, amino_acids, amino_acids_cost)}

    def root_hair_elongation(self, selected, root_hair_length, share_with_hairs, elapsed_thermal_time):
        p, ds = self._parameter, self.data_structure
        C, AA = ds.get("C_hexose_root")[selected], ds.get("AA")[selected]
        maximum = p("root_hair_max_length", selected)
        limitation = ((C / (C + p("Km_elongation", selected)))
                      + (AA / (AA + p("Km_elongation_amino_acids", selected)))) / 2
        new_length = (root_hair_length + p("root_hairs_elongation_rate", selected) * p("root_hair_radius", selected)
                      * share_with_hairs * limitation * elapsed_thermal_time)
        return np.where((C > 0.) & (AA > 0.), np.where(new_length > maximum, maximum, new_length), root_hair_length)

    def root_hair_costs(self, produced):
        p = self._parameter
        amino_acids = produced * p("struct_mass_N_content") / p("yield_growth_N") / p("r_Nm_AA")
        hexose = (produced * p("struct_mass_C_content") / p("yield_growth") - amino_acids * p("r_C_AA")) / 6.
        return {"C_hexose_root": hexose, "AA": amino_acids}

    def lateral_growth_duration_factor(self, properties, apex):
        # An apex segmentation created in the same step has no value yet: Root_BRIDGES' "is not None" gives 1
        factor = properties["nitrate_transporters_affinity_factor"].get(apex)
        return max(factor, 1) if factor is not None else 1
