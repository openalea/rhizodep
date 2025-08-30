# Public packages
import numpy as np
from dataclasses import dataclass
import cmf
import time
from multiprocessing.shared_memory import SharedMemory
import numpy as np

# Utility packages
from openalea.metafspm.component_factory import *
from openalea.metafspm.component import Model, declare


debug = False

@dataclass
class SoilModel(Model):
    """
    Empty doc
    """

    # --- @note INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

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

    # --- @note STATE VARIABLES INITIALIZATION ---
    # Temperature
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                                 value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_temperature", state_variable_type="intensive", edit_by="user")

    
    C_hexose_soil: float = declare(default=2.4e-3, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cm_hexose_soil: float = declare(default=2.4e-3, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_mucilage_soil: float = declare(default=15, unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_cells_soil: float = declare(default=15, unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
    # Structure related
    voxel_volume: float = declare(default=1e-6, unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    soil_moisture: float = declare(default=0.3, unit="adim", unit_comment="g.g-1", description="Volumetric proportion of water per volume of soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    bulk_density: float = declare(default=1.42, unit="g.mL", unit_comment="", description="Volumic density of the dry soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    dry_soil_mass: float = declare(default=1.42, unit="g", unit_comment="", description="dry weight of the considered voxel element", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    
    # --- @note RATES INITIALIZATION ---
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

    # --- @note PARAMETERS ---
   
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
    

    def __init__(self, time_step, scene_xrange=1., scene_yrange=1., soil_depth=1., **scenario):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.initiate_voxel_soil(scene_xrange, scene_yrange, soil_depth)
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.voxels, compartment="soil") 
        self.voxel_neighbor = {}


    # SERVICE FUNCTIONS

    # Just ressource for now
    def initiate_voxel_soil(self, scene_xrange=1., scene_yrange=1., soil_depth=1., 
                            voxel_length=3e-2, voxel_height=3e-2):
        """
        Note : not tested for now, just computed to support discussions.
        """
        self.voxels = {}

        
        self.planting_depth = 5e-2

        voxel_width = voxel_length
        voxel_volume = voxel_height * voxel_width * voxel_width

        self.delta_z = voxel_height
        self.voxels_Z_section_area = voxel_width * voxel_width
        
        self.voxel_number_x = int(scene_xrange / voxel_width) + 1
        actual_voxel_width = scene_xrange / self.voxel_number_x
        self.scene_xrange = scene_xrange

        self.voxel_number_y = int(scene_yrange / voxel_width) + 1
        actual_voxel_length = scene_yrange / self.voxel_number_y
        self.scene_yrange = scene_yrange

        voxel_volume = voxel_height * actual_voxel_width * actual_voxel_length

        scene_zrange = soil_depth
        self.voxel_number_z = int(scene_zrange / voxel_height) + 1
        self.scene_zrange = scene_zrange

        # Uncentered, positive grid
        y, z, x = np.indices((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels["x1"] = x * actual_voxel_width
        self.voxels["x2"] = self.voxels["x1"] + actual_voxel_width
        self.voxels["y1"] = y * actual_voxel_length
        self.voxels["y2"] = self.voxels["y1"] + actual_voxel_length
        self.voxels["z1"] = z * voxel_height
        self.voxels["z2"] = self.voxels["z1"] + voxel_height

        self.voxel_dx = actual_voxel_width
        self.voxel_dy = actual_voxel_length
        self.voxel_dz = voxel_height

        self.voxel_grid_to_self("voxel_volume", voxel_volume)

        for name in self.state_variables + self.inputs:
            if name != "voxel_volume":
                self.voxel_grid_to_self(name, init_value=getattr(self, name))

        # Initialize volumic concentrations
        self.voxels["water_volume"] = self.voxels["voxel_volume"] * self.voxels["soil_moisture"]
        self.voxels["dry_soil_mass"] = 1e6 * self.voxels["voxel_volume"] * self.voxels["bulk_density"]
        self.voxels["C_hexose_soil"] = self.voxels["Cm_hexose_soil"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 6 / 12


    def voxel_grid_to_self(self, name, init_value):
        self.voxels[name] = np.zeros((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels[name].fill(init_value)
        #setattr(self, name, self.voxels[name])
    

    def compute_mtg_voxel_neighbors(self, props):

        # necessary to get updated coordinates.
        # if "angle_down" in g.properties().keys():
        #     plot_mtg(g)

        for vid in props["vertex_index"].keys():
            if (vid not in props["voxel_neighbor"].keys()) or (props["voxel_neighbor"][vid] is None) or (props["length"][vid] > props["initial_length"][vid]):
                baricenter = (np.mean((props["x1"][vid], props["x2"][vid])) % self.scene_xrange, # min value is 0
                            np.mean((props["y1"][vid], props["y2"][vid])) % self.scene_yrange, # min value is 0
                            -np.mean((props["z1"][vid], props["z2"][vid])))
                testx1 = self.voxels["x1"] <= baricenter[0]
                testx2 = baricenter[0] <= self.voxels["x2"]
                testy1 = self.voxels["y1"] <= baricenter[1]
                testy2 = baricenter[1] <= self.voxels["y2"]
                testz1 = self.voxels["z1"] <= baricenter[2]
                testz2 = baricenter[2] <= self.voxels["z2"]
                test = testx1 * testx2 * testy1 * testy2 * testz1 * testz2
                try:
                    props["voxel_neighbor"][vid] = [int(v) for v in np.where(test)]
                except:
                    print(" WARNING, issue in computing the voxel neighbor for vid ", vid)
                    props["voxel_neighbor"][vid] = None
        
        return props
    
    def compute_mtg_voxel_neighbors_fast(self, data, hs, mask,
                                         xmin=0, ymin=0, zmin=0,
                                        periodic_xy=True, flip_z=False):
        
        # barycenters (vectorized)
        bx = 0.5 * (data[hs["x1"]] + data[hs["x2"]])
        by = 0.5 * (data[hs["y1"]] + data[hs["y2"]])
        bz = 0.5 * (data[hs["z1"]] + data[hs["z2"]])

        # The integer indices of the vertices where the boolean mask need is True
        # idx = np.nonzero(mask)[0]
        # xs, ys, zs = bx[idx], by[idx], bz[idx]
        xs, ys, zs = bx[mask], by[mask], bz[mask]

        # grid params
        Ny, Nz, Nx = self.voxel_number_y, self.voxel_number_z, self.voxel_number_x
        dx, dy, dz = self.voxel_dx, self.voxel_dy, self.voxel_dz

        if flip_z:
            zs = -zs
        
        if periodic_xy:
            Lx, Ly = Nx * dx, Ny * dy
            xs = (xs - xmin) % Lx + xmin
            ys = (ys - ymin) % Ly + ymin

        ix = np.floor((xs - xmin) / dx).astype(np.int32)
        iy = np.floor((ys - ymin) / dy).astype(np.int32)
        iz = np.floor((zs - zmin) / dz).astype(np.int32)

        # clamp to valid range (if not periodic or due to tiny FP drift)
        np.clip(ix, 0, Nx - 1, out=ix)
        np.clip(iy, 0, Ny - 1, out=iy)
        np.clip(iz, 0, Nz - 1, out=iz)

        return iy, iz, ix
    
    
    def apply_to_voxel(self, props):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """

        for name in self.inputs:
            self.voxels[name].fill(0)
        
        for vid in props["vertex_index"].keys():
            if props["length"][vid] > 0:
                if props["voxel_neighbor"][vid] is not None:
                    vy, vz, vx = props["voxel_neighbor"][vid]
                    for name in self.inputs:
                        # print(name,  props[name])
                        self.voxels[name][vy][vz][vx] += props[name][vid]
                else:
                    print(f"WARNING! segment {vid} did not send its status to the soil")


    def apply_to_voxel_fast(self, iy, iz, ix, data, hs, model_name, mask):
        for name in self.inputs:
            self.voxels[name].fill(0.)
            
            if name in self.pullable_inputs[model_name]:
                source_variables = self.pullable_inputs[model_name][name]
                to_apply = np.zeros(mask.sum(), dtype=np.float64)
                for variable, unit_conversion in source_variables.items():
                    to_apply += unit_conversion * data[hs[variable]][mask]
            else:
                to_apply = data[hs[name]][mask]

            np.add.at(self.voxels[name], (iy, iz, ix), to_apply) 

            # print(name, self.voxels[name].sum())


    def get_from_voxel(self, props, soil_outputs):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        for vid, (vy, vz, vx) in props["voxel_neighbor"].items():
            for name in soil_outputs:
                if name != "voxel_neighbor":
                    props[name][vid] = self.voxels[name][vy][vz][vx]
        
        return props

    def get_from_voxel_fast(self, iy, iz, ix, data, hs, soil_outputs, mask):
        # cols = np.flatnonzero(mask)
        # Nx, Nz = self.voxel_number_x, self.voxel_number_z
        # linear_index = ((iy * Nz + iz) * Nx + ix)
        # print("idx shape", ix.shape)
        # print(linear_index.shape)
        for name in soil_outputs:
            # print('vs assigned', self.voxels[name].ravel())
            # print(name, ix[cols].shape, data[hs[name], mask].shape, self.voxels[name][iy, iz, ix].shape)
            data[hs[name], mask] = self.voxels[name][iy, iz, ix]
            # print(name, data[hs[name], :])
            


    def pull_available_inputs(self, props, model_name):
        # vertices = props["vertex_index"].keys()
        vertices = [vid for vid in props["vertex_index"].keys() if props["living_struct_mass"][vid] > 0]
        
        for input, source_variables in self.pullable_inputs[model_name].items():
            if input not in props:
                props[input] = {}
            # print(input, source_variables)
            props[input].update({vid: sum([props[variable][vid]*unit_conversion 
                                           for variable, unit_conversion in source_variables.items()]) 
                                 for vid in vertices})
        return props

    
    def __call__(self, queue_plants_to_soil, queues_soil_to_plants, soil_outputs: list=[], *args):

        # We get fluxes and voxel interception from the plant mtgs (If none passed, soil model can be autonomous)
        # Waiting for all plants to put their outputs
        t1 = time.time()

        batch = []
        for _ in range(len(queues_soil_to_plants)):
            plant_data = queue_plants_to_soil.get()
            batch.append(plant_data)

            # Deplaced here because if one plant hangs, their is not reason not to retreive others in the meantime
            self.get_from_plant(plant_data)
        
        t2 = time.time()
        if debug: print("soil waits plants: ", t2 - t1)

        # Run the soil model
        self.choregrapher(module_family=self.__class__.__name__, *args)

        t3 = time.time()
        if debug: print("soil solve: ", t3 - t2)

        for plant_data in batch:
            plant_id = plant_data["plant_id"]

            self.send_to_plant(plant_data, soil_outputs)
            
            # Send a message to plants so that they can resume with last soil state
            queues_soil_to_plants[plant_id].put("finished")
        
        t4 = time.time()
        if debug: print("soil sends plants: ", t4 - t3)

    def get_from_plant(self, plant_data):
        """
        TODO : probably transfer to composite
        """
        # Unpacking message
        plant_id = plant_data["plant_id"]
        model_name = plant_data["model_name"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        hs = plant_data["handshake"]
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention

        iy, iz, ix = self.compute_mtg_voxel_neighbors_fast(buf, hs, mask=vertices_mask, flip_z=True)
        self.apply_to_voxel_fast(iy, iz, ix, buf, hs, model_name, vertices_mask)
        # Stored for sending to plants later
        self.voxel_neighbor[plant_id] = (iy, iz, ix)

        shm.close()

        # Legacy code commented
        # props = plant_data["data"]
        # props = self.pull_available_inputs(props, model_name)
        # props = self.compute_mtg_voxel_neighbors(props)
        # self.apply_to_voxel(props)
        # voxel_neighbors[id] = props["voxel_neighbor"]


    def send_to_plant(self, plant_data, soil_outputs):
        """
        TODO : probably transfer to composite
        """
        # Then apply the states to the plants
        plant_id = plant_data["plant_id"]
        hs = plant_data["handshake"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention
        self.get_from_voxel_fast(*self.voxel_neighbor[plant_id], buf, hs, soil_outputs, vertices_mask)

        shm.close()

        # legacy_code_commented
        # # EDIT : removed modification of the input props, there should be no variable link between inputs and outputs, excepted voxel neighbors for interception 
        # outputs = {name: {} for name in soil_outputs}
        # outputs["voxel_neighbor"] = vn
        # outputs = self.get_from_voxel(outputs, soil_outputs=soil_outputs)
    

    # MODEL EQUATIONS

    @rate
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

    @rate
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

    @rate
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

    @state
    def _C_hexose_soil(self, C_hexose_soil, soil_moisture, voxel_volume, hexose_degradation, hexose_exudation,
                             phloem_hexose_exudation, hexose_uptake_from_soil, phloem_hexose_uptake_from_soil):
        balance = C_hexose_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
            hexose_exudation
            + phloem_hexose_exudation
            - hexose_uptake_from_soil
            - phloem_hexose_uptake_from_soil
            - hexose_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    @state
    def _C_hexose_soil(self, C_hexose_soil, dry_soil_mass, hexose_degradation, hexose_exudation,
                             phloem_hexose_exudation, hexose_uptake_from_soil, phloem_hexose_uptake_from_soil):
        balance = C_hexose_soil + (self.time_step / dry_soil_mass) * (
            hexose_exudation
            + phloem_hexose_exudation
            - hexose_uptake_from_soil
            - phloem_hexose_uptake_from_soil
            - hexose_degradation
        )
        balance[balance < 0.] = 0.
        return balance

    @state
    def _Cs_mucilage_soil(self, Cs_mucilage_soil, soil_moisture, voxel_volume, mucilage_secretion, mucilage_degradation):
        balance = Cs_mucilage_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
            mucilage_secretion
            - mucilage_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    
    @state
    def _Cs_cells_soil(self, Cs_cells_soil, soil_moisture, voxel_volume, cells_release, cells_degradation):
        balance = Cs_cells_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
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