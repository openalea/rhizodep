import numpy as np
from dataclasses import dataclass

from genericmodel.component import Model, declare
from genericmodel.component_factory import *


@dataclass
class SoilModel(Model):
    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM CARBON MODEL
    hexose_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    phloem_hexose_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    hexose_uptake_from_soil: float = declare(default=0., unit="mol.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    phloem_hexose_uptake_from_soil: float = declare(default=0., unit="mol.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    mucilage_secretion: float = declare(default=0., unit="mol.s-1", unit_comment="of equivalent hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    cells_release: float = declare(default=0., unit="mol.s-1", unit_comment="of equivalent hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    struct_mass: float = declare(default=1.35e-4, unit="g", unit_comment="", description="Example root segment structural mass", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---
    # Temperature
    soil_temperature_in_Celsius: float = declare(default=15., unit="°C", unit_comment="", description="soil temperature in contact with roots", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # Carbon and nitrogen concentrations
    C_hexose_soil: float = declare(default=30, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_mucilage_soil: float = declare(default=15, unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_cells_soil: float = declare(default=15, unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_mineralN_soil: float = declare(default=0.5, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_AA_soil: float = declare(default=0.5, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # Water
    water_potential_soil: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="Mean soil water potential", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    volume_soil: float = declare(default=1e-6, unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    # Degradation processes
    hexose_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of hexose consumption  at the soil-root interface", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mucilage_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of mucilage degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    cells_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of root cells degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    # --- INITIALIZES MODEL PARAMETERS ---

    # Temperature
    process_at_T_ref: float = declare(default=1., unit="adim", unit_comment="", description="Proportion of maximal process intensity occuring at T_ref", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    hexose_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    hexose_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    hexose_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="The value for B (Q10) has been fitted from the evolution of Vmax measured by Coody et al. (1986, SBB), who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    hexose_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    mucilage_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    cells_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Kinetic soil degradation parameters
    hexose_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, unit="mol.m-2.s-1", unit_comment="of hexose", description="Maximum degradation rate of hexose in soil", 
                                        value_comment="", references="According to what Jones and Darrah (1996) suggested, we assume that this Km is 2 times lower than the Km corresponding to root uptake of hexose (350 uM against 800 uM in the original article).", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_hexose_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for soil hexose degradation", 
                                        value_comment="", references="We assume that the maximum degradation rate is 10 times higher than the maximum hexose uptake rate by roots", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of mucilage in soil", 
                                        value_comment="", references="We assume that the maximum degradation rate for mucilage is equivalent to the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_mucilage_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil mucilage degradation ", 
                                        value_comment="", references="We assume that Km for mucilage degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10 / 2, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of root cells at the soil/root interface", 
                                        value_comment="", references="We assume that the maximum degradation rate for cells is equivalent to the half of the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_cells_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil cells degradation", 
                                        value_comment="", references="We assume that Km for cells degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    def __init__(self, g, time_step_in_seconds, **scenario: dict):
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
        self.choregrapher.add_data(instance=self, data_name="props", filter={"label": ["Segment", "Apex"], "type":["Base_of_the_root_system", "Normal_root_after_emergence", "Stopped", "Just_Stopped", "Root_nodule"]})
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        self.time_steps_in_seconds = time_step_in_seconds
        #self.voxels = self.initiate_voxel_soil() # TODO Not tested for now

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

    # Just ressource for now
    def initiate_voxel_soil(self):
        """
        Note : not tested for now, just computed to support discussions.
        """
        voxels = {}

        voxel_width = 1e-2
        voxel_height = 1e-3
        voxel_xy_range = 2e-1
        voxel_number_xy = int(voxel_xy_range / voxel_width)
        voxel_z_range = 1
        voxel_number_z = int(voxel_z_range / voxel_height)

        vid = 0
        for x in range(-voxel_number_xy, voxel_number_xy + 1):
            for y in range(-voxel_number_xy, voxel_number_xy + 1):
                for z in range(voxel_number_z):
                    vid += 1
                    voxels[vid] = {}
                    v = voxels[vid]
                    v["x1"] = (x - 0.5) * voxel_width
                    v["x2"] = (x + 0.5) * voxel_width
                    v["y1"] = (y - 0.5) * voxel_width
                    v["y2"] = (y + 0.5) * voxel_width
                    v["z1"] = z * voxel_height
                    v["z2"] = (z + 1) * voxel_height
                    v["volume_soil_solution"] = 2 * voxel_width * voxel_height * 0.1  # TODO quick estimation necessary

        return voxels

    def post_growth_updating(self):
        """
        Description :
            Extend property dictionary upon new element partitioning.
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.C_hexose_soil.keys()):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                for prop in self.state_variables:
                    # All concentrations, temperature and pressure are intensive, so we need structural mass wise partitioning to initialize
                    getattr(self, prop).update({vid: getattr(self, prop)[parent]})
                


    def apply_to_voxel(self, element, root_flows: list = []):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """
        neighbor_voxels = []
        v = self.voxels
        for vid in v.keys():
            if v[vid]["x1"] <= element.x1 <= v[vid]["x2"] and v[vid]["y1"] <= element.y1 <= v[vid]["y2"] and v[vid][
                "z1"] >= element.z1 >= v[vid]["z2"]:
                neighbor_voxels += [vid]
            if v[vid]["x1"] <= element.x2 <= v[vid]["x2"] and v[vid]["y1"] <= element.y2 <= v[vid]["y2"] and v[vid][
                "z1"] >= element.z2 >= v[vid]["z2"]:
                neighbor_voxels += [vid]

        neighbor_voxels = list(set(neighbor_voxels))

        if len(neighbor_voxels) == 1:
            for name in root_flows:
                setattr(v[neighbor_voxels[0]], name, getattr(element, name))
        elif len(neighbor_voxels) > 1:
            for vid in neighbor_voxels:
                for name in root_flows:
                    setattr(v[vid], name, getattr(element, name) / len(neighbor_voxels))

    def get_from_voxel(self, element, soil_states: list):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        neighbor_voxels = []
        v = self.voxels
        for vid in v.keys():
            if v[vid]["x1"] <= element.x1 <= v[vid]["x2"] and v[vid]["y1"] <= element.y1 <= v[vid]["y2"] and v[vid][
                "z1"] >= element.z1 >= v[vid]["z2"]:
                neighbor_voxels += [vid]
            if v[vid]["x1"] <= element.x2 <= v[vid]["x2"] and v[vid]["y1"] <= element.y2 <= v[vid]["y2"] and v[vid][
                "z1"] >= element.z2 >= v[vid]["z2"]:
                neighbor_voxels += [vid]

        neighbor_voxels = list(set(neighbor_voxels))

        if len(neighbor_voxels) == 1:
            for name in soil_states:
                setattr(element, name, getattr(v[neighbor_voxels[0]], name))
        elif len(neighbor_voxels) > 1:
            for name in soil_states:
                setattr(element, name, np.mean([getattr(v[vid], name) for vid in neighbor_voxels]))

    @rate
    def _hexose_degradation(self, C_hexose_soil, root_exchange_surface, soil_temperature_in_Celsius):
        """
        This function computes the rate of hexose "consumption" (in mol of hexose per seconds) at the soil-root interface
        for a given root element. It mimics the uptake of hexose by rhizosphere microorganisms, and is therefore described
        using a substrate-limited function (Michaelis-Menten).
        :param C_hexose_soil: hexose concentration in soil solution (mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_hexose_degradation_rate_max = self.hexose_degradation_rate_max * self.temperature_modification(
                                                                        soil_temperature=soil_temperature_in_Celsius,
                                                                        T_ref=self.hexose_degradation_rate_max_T_ref,
                                                                        A=self.hexose_degradation_rate_max_A,
                                                                        B=self.hexose_degradation_rate_max_B,
                                                                        C=self.hexose_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of hexose in the soil:
        return max(corrected_hexose_degradation_rate_max * root_exchange_surface * C_hexose_soil / (
                                                                            self.Km_hexose_degradation + C_hexose_soil), 0.)

    @rate
    def _mucilage_degradation(self, Cs_mucilage_soil, root_exchange_surface, soil_temperature_in_Celsius):
        """
        This function computes the rate of mucilage degradation outside the root (in mol of equivalent-hexose per second)
        for a given root element. Only the external surface of the root element is taken into account here, similarly to
        what is done for mucilage secretion.
        :param Cs_mucilage_soil: mucilage concentration in soil solution (equivalent hexose, mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_mucilage_degradation_rate_max = self.mucilage_degradation_rate_max * self.temperature_modification(
                                                                    soil_temperature=soil_temperature_in_Celsius,
                                                                    T_ref=self.mucilage_degradation_rate_max_T_ref,
                                                                    A=self.mucilage_degradation_rate_max_A,
                                                                    B=self.mucilage_degradation_rate_max_B,
                                                                    C=self.mucilage_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of mucilage
        # in the soil:
        return max(corrected_mucilage_degradation_rate_max * root_exchange_surface * Cs_mucilage_soil / (
                self.Km_mucilage_degradation + Cs_mucilage_soil), 0.)

    @rate
    def _cells_degradation(self, Cs_cells_soil, root_exchange_surface, soil_temperature_in_Celsius):
        """
        This function computes the rate of root cells degradation outside the root (in mol of equivalent-hexose per second)
        for a given root element. Only the external surface of the root element is taken into account as the exchange
        surface, similarly to what is done for root cells release.
        :param Cs_cells_soil: released cells concentration in soil solution (equivalent hexose, mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_cells_degradation_rate_max = self.cells_degradation_rate_max * self.temperature_modification(
                                                                        soil_temperature=soil_temperature_in_Celsius,
                                                                        T_ref=self.cells_degradation_rate_max_T_ref,
                                                                        A=self.cells_degradation_rate_max_A,
                                                                        B=self.cells_degradation_rate_max_B,
                                                                        C=self.cells_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of root cells
        # in the soil:
        return max(corrected_cells_degradation_rate_max * root_exchange_surface * Cs_cells_soil / (
                self.Km_cells_degradation + Cs_cells_soil), 0.)

    # TODO FOR TRISTAN: Consider adding similar functions for describing N mineralization/organization in the soil?

    @state
    def _C_hexose_soil(self, C_hexose_soil, volume_soil, hexose_degradation, hexose_exudation,
                             phloem_hexose_exudation, hexose_uptake_from_soil, phloem_hexose_uptake_from_soil):
        balance = C_hexose_soil + (self.time_steps_in_seconds / volume_soil) * (
            hexose_exudation
            + phloem_hexose_exudation
            - hexose_uptake_from_soil
            - phloem_hexose_uptake_from_soil
            - hexose_degradation
        )
        return max(balance, 0)

    @state
    def _Cs_mucilage_soil(self, Cs_mucilage_soil, volume_soil, mucilage_secretion, mucilage_degradation):
        balance = Cs_mucilage_soil + (self.time_steps_in_seconds / volume_soil) * (
            mucilage_secretion
            - mucilage_degradation
        )
        return max(balance, 0)
    
    @state
    def _Cs_cells_soil(self, Cs_cells_soil, volume_soil, cells_release, cells_degradation):
        balance = Cs_cells_soil + (self.time_steps_in_seconds / volume_soil) * (
                cells_release
                - cells_degradation
        )
        return max(balance, 0)
