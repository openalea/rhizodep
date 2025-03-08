# Public packages
import numpy as np
from dataclasses import dataclass

from metafspm.component import Model, declare
from metafspm.component_factory import *
from log.visualize import plot_mtg


family = "soil"


@dataclass
class RhizoInputsSoilModel(Model):
    
    # We need the module AND the class to be named the same way
    family = family

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
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")


    # --- STATE VARIABLES ---
    voxel_neighbor: int = declare(default=None, unit="adim", unit_comment="", description="",
                                                 value_comment="", references="", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="descriptor", edit_by="user")

    # Temperature
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                                 value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_temperature", state_variable_type="intensive", edit_by="user")

    # Carbon and nitrogen concentrations
    C_hexose_soil: float = declare(default=2.4e-3, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    content_hexose_soil: float = declare(default=2.4e-3, unit="mol.g-1", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_mucilage_soil: float = declare(default=15, unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_cells_soil: float = declare(default=15, unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
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

    # --- PARAMETERS ---

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
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.initiate_voxel_soil()
        self.time_step_in_seconds = time_step_in_seconds
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step_in_seconds, data=self.voxels, compartment="soil")
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        self.link_self_to_mtg()    

    # SERVICE FUNCTIONS

    # Just ressource for now
    def initiate_voxel_soil(self):
        """
        Note : not tested for now, just computed to support discussions.
        """
        self.voxels = {}
        # self.props.setdefault("voxel_neighbor", {})
        # self.props["voxel_neighbor"].update({key: None for key in self.vertices})
        # setattr(self, "voxel_neighbor", self.props["voxel_neighbor"])
        
        self.planting_depth = 5e-2

        cubic_length = 3e-2
        voxel_width = cubic_length
        voxel_height = cubic_length
        voxel_volume = voxel_height * voxel_width * voxel_width
        scene_xy_range = 5e-1

        self.delta_z = voxel_height
        self.voxels_Z_section_area = voxel_width * voxel_width

        self.voxel_number_xy = int(scene_xy_range / voxel_width)
        # We want the plant to be centered
        if self.voxel_number_xy%2 == 0:
            self.voxel_number_xy += 1
        scene_z_range = 1.
        self.voxel_number_z = int(scene_z_range / voxel_height)

        y, z, x = np.indices((self.voxel_number_xy, self.voxel_number_z, self.voxel_number_xy))
        self.voxels["x1"] = x * voxel_width - ((self.voxel_number_xy*voxel_width)/2)
        self.voxels["x2"] = self.voxels["x1"] + voxel_width
        self.voxels["y1"] = y * voxel_width - ((self.voxel_number_xy*voxel_width)/2)
        self.voxels["y2"] = self.voxels["y1"] + voxel_width
        self.voxels["z1"] = z * voxel_height
        self.voxels["z2"] = self.voxels["z1"] + voxel_height

        self.voxel_grid_to_self("voxel_volume", voxel_volume)

        for name in self.state_variables + self.inputs:
            if name != "voxel_volume":
                self.voxel_grid_to_self(name, init_value=getattr(self, name))


    def voxel_grid_to_self(self, name, init_value):
        self.voxels[name] = np.zeros((self.voxel_number_xy, self.voxel_number_z, self.voxel_number_xy))
        self.voxels[name].fill(init_value)
        #setattr(self, name, self.voxels[name])

    def compute_mtg_voxel_neighbors(self):
        # necessary to get updated coordinates.
        if "angle_down" in self.g.properties().keys():
            plot_mtg(self.g)
        for vid in self.vertices:
            if (not self.voxel_neighbor[vid]) or (self.length[vid] > self.initial_length[vid]):
                baricenter = (np.mean((self.props["x1"][vid], self.props["x2"][vid])), 
                            np.mean((self.props["y1"][vid], self.props["y2"][vid])),
                            -np.mean((self.props["z1"][vid], self.props["z2"][vid])))
                testx1 = self.voxels["x1"] <= baricenter[0]
                testx2 = baricenter[0] <= self.voxels["x2"]
                testy1 = self.voxels["y1"] <= baricenter[1]
                testy2 = baricenter[1] <= self.voxels["y2"]
                testz1 = self.voxels["z1"] <= baricenter[2] + self.planting_depth
                testz2 = baricenter[2] + self.planting_depth <= self.voxels["z2"]
                test = testx1 * testx2 * testy1 * testy2 * testz1 * testz2
                try:
                    self.voxel_neighbor[vid] = [int(v) for v in np.where(test)]
                except:
                    print(" WARNING, issue in computing the voxel neighbor for vid ", vid)
                    self.voxel_neighbor[vid] = None



    def post_growth_updating(self):
        """
        Description :
            Extend property dictionary upon new element partitioning.
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.C_hexose_soil.keys()):
                parent = self.g.parent(vid)
                for prop in self.state_variables:
                    # All concentrations, temperature and pressure are intensive, so we need structural mass wise partitioning to initialize
                    getattr(self, prop).update({vid: getattr(self, prop)[parent]})
                # Specific as it can be a mix of None and lists
                getattr(self, "voxel_neighbor").update({vid: None})

    def apply_to_voxel(self):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """
        for name in self.inputs:
            self.voxels[name].fill(0)
        
        for vid in self.vertices:
            if self.length[vid] > 0:
                vy, vz, vx = self.voxel_neighbor[vid]
                for name in self.inputs:
                    self.voxels[name][vy][vz][vx] += getattr(self, name)[vid]

    def get_from_voxel(self):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        for vid in self.vertices:
            if self.length[vid] > 0: # Equivalent to : if self.voxel_neighbor[vid] != None:
                vy, vz, vx = self.voxel_neighbor[vid]
                for name in self.state_variables:
                    if name != "voxel_neighbor":
                        getattr(self, name)[vid] = self.voxels[name][vy][vz][vx]


    def __call__(self, *args):
        self.pull_available_inputs()
        # self.compute_mtg_voxel_neighbors()
        self.apply_to_voxel()
        self.choregrapher(module_family=self.family, *args)
        # self.get_from_voxel()
    
    # MODEL EAQUATIONS

    #TP@rate
    def _hexose_degradation(self, C_hexose_soil, root_exchange_surface, soil_temperature):
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
                                                                        soil_temperature=soil_temperature,
                                                                        T_ref=self.hexose_degradation_rate_max_T_ref,
                                                                        A=self.hexose_degradation_rate_max_A,
                                                                        B=self.hexose_degradation_rate_max_B,
                                                                        C=self.hexose_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of hexose in the soil:
        result = corrected_hexose_degradation_rate_max * root_exchange_surface * C_hexose_soil / (
                                                                            self.Km_hexose_degradation + C_hexose_soil)
        result[result < 0.] = 0.
        return result

    #TP@rate
    def _mucilage_degradation(self, Cs_mucilage_soil, root_exchange_surface, soil_temperature):
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
                                                                    soil_temperature=soil_temperature,
                                                                    T_ref=self.mucilage_degradation_rate_max_T_ref,
                                                                    A=self.mucilage_degradation_rate_max_A,
                                                                    B=self.mucilage_degradation_rate_max_B,
                                                                    C=self.mucilage_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of mucilage
        # in the soil:
        result = corrected_mucilage_degradation_rate_max * root_exchange_surface * Cs_mucilage_soil / (
                self.Km_mucilage_degradation + Cs_mucilage_soil)
        result[result < 0.] = 0.

        return result

    #TP@rate
    def _cells_degradation(self, Cs_cells_soil, root_exchange_surface, soil_temperature):
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
                                                                        soil_temperature=soil_temperature,
                                                                        T_ref=self.cells_degradation_rate_max_T_ref,
                                                                        A=self.cells_degradation_rate_max_A,
                                                                        B=self.cells_degradation_rate_max_B,
                                                                        C=self.cells_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of root cells
        # in the soil:
        result = corrected_cells_degradation_rate_max * root_exchange_surface * Cs_cells_soil / (
                self.Km_cells_degradation + Cs_cells_soil)
        result[result < 0] = 0.
        return result

    # TODO FOR TRISTAN: Consider adding similar functions for describing N mineralization/organization in the soil?

    #TP@state
    def _C_hexose_soil(self, C_hexose_soil, soil_moisture, voxel_volume, hexose_degradation, hexose_exudation,
                             phloem_hexose_exudation, hexose_uptake_from_soil, phloem_hexose_uptake_from_soil):
        balance = C_hexose_soil + (self.time_step_in_seconds / (soil_moisture * voxel_volume)) * (
            hexose_exudation
            + phloem_hexose_exudation
            - hexose_uptake_from_soil
            - phloem_hexose_uptake_from_soil
            - hexose_degradation
        )
        balance[balance < 0.] = 0.
        return balance

    #TP@state
    def _Cs_mucilage_soil(self, Cs_mucilage_soil, soil_moisture, voxel_volume, mucilage_secretion, mucilage_degradation):
        balance = Cs_mucilage_soil + (self.time_step_in_seconds / (soil_moisture * voxel_volume)) * (
            mucilage_secretion
            - mucilage_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    #TP@state
    def _Cs_cells_soil(self, Cs_cells_soil, soil_moisture, voxel_volume, cells_release, cells_degradation):
        balance = Cs_cells_soil + (self.time_step_in_seconds / (soil_moisture * voxel_volume)) * (
                cells_release
                - cells_degradation
        )
        balance[balance < 0.] = 0.
        return balance

    def temperature_modification(self, soil_temperature=15, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
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
        # We compute a temperature-modified process, correspond to a Q10-modified relationship,
        # based on the work of Tjoelker et al. (2001):
        if C != 0 and C != 1:
            print("The modification of the process at T =", soil_temperature,
                  "only works for C=0 or C=1!")
            print("The modified process has been set to 0.")
            return np.zeros_like(soil_temperature)

        modified_process = process_at_T_ref * (A * (soil_temperature - T_ref) + B) ** (1 - C) \
                           * (A * (soil_temperature - T_ref) + B) ** (
                                   C * (soil_temperature - T_ref) / 10.)
        
        if C == 1:
            modified_process[(A * (soil_temperature - T_ref) + B) < 0.] = 0.

        modified_process[modified_process < 0.] = 0.

        return modified_process

