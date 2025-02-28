import numpy as np
from math import pi
from dataclasses import dataclass

from metafspm.component import Model, declare
from metafspm.component_factory import *


family = "anatomical"


@dataclass
class RootAnatomy(Model):
    """
    Root anatomy model originating from both Rhizodep shoot.py and Root_CyNAPS model_topology.py

    Rhizodep forked :
        https://forgemia.inra.fr/tristan.gerault/rhizodep/-/commits/rhizodep_2022?ref_type=heads
    base_commit :
        92a6f7ad927ffa0acf01aef645f9297a4531878c
    """

    family = "anatomical"

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM GROWTH MODEL
    radius: float = declare(default=3.5e-4, unit="m", unit_comment="", description="Example root segment radius", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    length: float = declare(default=1.e-3, unit="m", unit_comment="", description="Example root segment length", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    distance_from_tip: float = declare(default=1.e-3, unit="m", unit_comment="", description="Example root segment distance from tip", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    struct_mass: float = declare(default=1.35e-4, unit="g", unit_comment="", description="Example root segment structural mass", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    root_hair_length: float = declare(default=1.e-3, unit="m", unit_comment="", description="Example root hair length", 
                            min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between 0.5 and 1.3 mm.", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    total_root_hairs_number: float = declare(default=30 * (1.6e-4 / 3.5e-4) * 3.e-3 * 1e3, unit="adim", unit_comment="", description="Example root hairs number on segment external surface", 
                            min_value="", max_value="", value_comment="30 * (1.6e-4 / radius) * length * 1e3", references=" According to the work of Gahoonia et al. (1997), the root hair density is about 30 hairs per mm for winter wheat, for a root radius of about 0.16 mm.", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    thermal_time_since_primordium_formation: float = declare(default=200, unit="°C.d", unit_comment="", description="Input thermal age of the organ. It is a declared input for consistence, but homogeneity makes no sense.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")


    # --- INITIALIZE MODEL STATE VARIABLES ---

    # Surfaces
    root_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    cortex_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic cortex. It excludes stele parenchyma surface. This is computed as the exchange surface for water absorption from soil to stele apoplasm, which is supposed at equilibrium with xylem vessels (so we neglect stele surface between symplasm and apoplasm, supposing quick equilibrium inside the root.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    xylem_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between root parenchyma and apoplasmic xylem vessels.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    phloem_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between root parenchyma and apoplasmic xylem vessels.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")

    # Volumes
    symplasmic_volume: float = declare(default=1e-9, unit="m3", unit_comment="", description="symplasmic volume for water content of root elements", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    xylem_volume: float = declare(default=0, unit="m3", unit_comment="", description="xylem volume for water transport between elements", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    total_phloem_volume: float = declare(default=0, unit="m3", unit_comment="", description="total phloem volume throughout the root system", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="plant_scale_state", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    xylem_vessel_radii: float = declare(default=0., unit="m", unit_comment="", description="list of individual xylem radius, also providing their numbering", 
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variables", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")

    # Differentiation factors
    endodermis_conductance_factor: float = declare(default=1., unit="adim", unit_comment="", description="The endodermis barrier differentiation factor", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    exodermis_conductance_factor: float = declare(default=0.5, unit="adim", unit_comment="", description="The exodermis barrier differentiation factor", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    xylem_differentiation_factor: float = declare(default=1., unit="adim", unit_comment="", description="Xylem differentiation, i.e. apoplasmic opening, from 0 to 1", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    
    # Whole segment conductance
    kr_symplasmic_water: float = declare(default=1., unit="mol.s-1.Pa-1", unit_comment="", description="Symplasmic water conductance of all cell layer contribution, including transmembrane and plasmodesmata resistance", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    kr_apoplastic_water: float = declare(default=1., unit="mol.s-1.Pa-1", unit_comment="", description="Apolastic water conductance including the endoderm differentiation blocking this pathway. Considering xylem volume to be equivalent to whole stele apoplasm, we only account for the cumulated resistance of cortex and epidermis cell wals.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")

    # Tissue density
    root_tissue_density: float = declare(default=0.10 * 1e6, unit="g.m3", unit_comment="of structural mass", description="root_tissue_density", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="state_variable", by="model_anatomy", state_variable_type="self_rate_state", edit_by="user")
    
    # --- INITIALIZES MODEL PARAMETERS ---

    # Differentiation parameters
    meristem_limit_zone_factor: float = declare(default=1., unit="adim", unit_comment="", description="Ratio between the length of the meristem zone and root radius", 
                            min_value="", max_value="", value_comment="Overwrite 1. where we assume that the length of the meristem zone is equal to the radius of the root", references="(??) see transition zone reference", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    growing_zone_factor: float = declare(default=16., unit="adim", unit_comment="", description="Proportionality factor between the radius and the length of the root apical zone in which C can sustain root elongation", 
                            min_value="", max_value="", value_comment="", references="According to illustrations by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap, meristem and elongation zones is about 8 times the diameter of the tip.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    start_distance_for_endodermis_factor : float = declare(default=10/0.1, unit="adim", unit_comment="", description="Ratio between the distance from tip where barriers formation starts/ends, and root radius", 
                            min_value="", max_value="", value_comment="", references="(Clarkson et al., 1971; Wu et al., 2019)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    end_distance_for_endodermis_factor : float = declare(default=150/0.1, unit="adim", unit_comment="", description="Ratio between the distance from tip where barriers formation starts/ends, and root radius", 
                            min_value="", max_value="", value_comment="", references="(Clarkson et al., 1971)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    start_distance_for_exodermis_factor : float = declare(default=20/0.1, unit="adim", unit_comment="", description="Ratio between the distance from tip where barriers formation starts/ends, and root radius", 
                            min_value="", max_value="", value_comment="", references="(Schreiber et al., 2005)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    end_distance_for_exodermis_factor : float = declare(default=400/0.1, unit="adim", unit_comment="", description="Ratio between the distance from tip where barriers formation starts/ends, and root radius", 
                            min_value="", max_value="", value_comment="", references=" (Ouyang et al., 2020)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    endodermis_a: float = declare(default=100., unit="adim", unit_comment="", description="This parameter corresponds to the asymptote of the process in Gompertz law describing the evolution of apoplastic barriers with cell age.", 
                            min_value="", max_value="", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    endodermis_b: float = declare(default=3.25 * (60.*60.*24.), unit="s", unit_comment="time equivalent at T_ref", description="This parameter corresponds to the time lag before the large increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", 
                            min_value="", max_value="", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    endodermis_c: float = declare(default=1.11 / (60.*60.*24.), unit="s-1", unit_comment="time equivalent at T_ref", description="This parameter reflects the slope of the increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", 
                            min_value="", max_value="", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    exodermis_a: float = declare(default=100., unit="adim", unit_comment="", description="This parameter corresponds to the asymptote of the process in Gompertz law describing the evolution of apoplastic barriers with cell age.", 
                            min_value="", max_value="", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    exodermis_b: float = declare(default=5.32 * (60.*60.*24.), unit="s", unit_comment="time equivalent at T_ref", description="This parameter corresponds to the time lag before the large increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", 
                            min_value="", max_value="", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    exodermis_c: float = declare(default=1.11 / (60.*60.*24.), unit="s-1", unit_comment="time equivalent at T_ref", description="This parameter reflects the slope of the increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", 
                            min_value="", max_value="", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")

    max_thermal_time_since_endodermis_disruption: float = declare(default=6 * 60. * 60., unit="s", unit_comment="time equivalent at T_ref", description="Maximal thermal time above which no endodermis disruption is considered anymore after a lateral root has emerged", 
                            min_value="", max_value="", value_comment="", references="We assume that after 6h, no disruption is observed anymore! (??)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    max_thermal_time_since_exodermis_disruption: float = declare(default=48 * 60. * 60., unit="s", unit_comment="time equivalent at T_ref", description="Maximal thermal time above which no exodermis disruption is considered anymore after a lateral root has emerged", 
                            min_value="", max_value="", value_comment="", references="We assume that after 48h, no disruption is observed anymore! (??)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    begin_xylem_differentiation: float = declare(default=0., unit="m", unit_comment="distance from apex", description="Parameter indicating at which apex distance xylem differentiation starts for a logistic function", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    span_xylem_differentiation: float = declare(default=0.002, unit="m", unit_comment="distance from apex", description="Parameter indicating what length span is necessary for the transition to fully opened xylem / stele apoplasm", 
                            min_value="", max_value="", value_comment="", references="We assume that after few mm, xylem apoplasm is open (??)", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")

    # Surfacic fractions
    # cortical_surfacic_fraction: float = declare(default=29., unit="adim", unit_comment="", description="Cortex (+exodermis) parenchyma surface ratio over root segment's cylinder surface", 
    #                         min_value="", max_value="", value_comment="", references="report", DOI="",
    #                         variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    # stellar_surfacic_fraction: float = declare(default=11., unit="adim", unit_comment="", description="Stele (+endodermis) surface ratio over root segment's cylinder surface", 
    #                         min_value="", max_value="", value_comment="", references="report", DOI="",
    #                         variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_surfacic_fraction: float = declare(default=2.5, unit="adim", unit_comment="", description="phloem surface ratio over root's cylinder surface", 
                            min_value="", max_value="", value_comment="", references="report", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    # apoplasm_cross_area_surfacic_fraction: float = declare(default=0.5, unit="adim", unit_comment="", description="symplasmic cross-section ratio over root segment's sectional surface", 
    #                         min_value="", max_value="", value_comment="", references="report", DOI="",
    #                         variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    # xylem_cross_area_surfacic_fraction: float = declare(default=0.84 * (0.36 ** 2), unit="adim", unit_comment="apoplasmic cross-section area ratio * stele radius ratio^2", description="apoplasmic cross-section ratio of xylem over root segment's sectional surface", 
    #                         min_value="", max_value="", value_comment="", references="report", DOI="",
    #                         variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    root_hair_radius: float = declare(default=12 * 1e-6 / 2., unit="m", unit_comment="", description="Average radius of root hair", 
                            min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair diameter is relatively constant for different genotypes of wheat and barley, i.e. 12 microns.", DOI="",
                            variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user")
    
    # Concentric layering parameters
    tissue_name: tuple = ("epidermis", "cortex", "stele", "phloem")
    num_layers: dict = {"epidermis": 1, "cortex": 5, "stele": 5, "phloem": 1}
    layer_cell_perimeter_toR: dict = {"epidermis": (0.05,), "cortex": (0.15 for _ in range(5)), "stele": (0.04 for _ in range(5)), "phloem": (0.03,)}
    layer_cross_sectional_surface_toRR: dict = {"epidermis": (0.05), "cortex": (0.15 for _ in range(5)), "stele": (0.04 for _ in range(5)), "phloem": (0.03,)}
    mean_cell_length_toR: dict = {"epidermis": (0.05), "cortex": (0.15 for _ in range(5)), "stele": (0.04 for _ in range(5)), "phloem": (0.03,)}
    mean_cell_width_toR: dict = {"epidermis": (0.05), "cortex": (0.15 for _ in range(5)), "stele": (0.04 for _ in range(5)), "phloem": (0.03,)}
    cell_wall_thickness: dict = {"epidermis": (1e-6), "cortex": (1e-6 for _ in range(5)), "stele": (1e-6 for _ in range(5)), "phloem": (1e-6,)}
    wall_connectivity_in_layer: dict = {"epidermis": (0.1), "cortex": (0.1 for _ in range(5)), "stele": (0.5 for _ in range(5)), "phloem": (0.5,)}
    wall_connectivity_with_inner_neighbor: dict = {"epidermis": (0.8,), "cortex": (0.5 for _ in range(5)), "stele": (0.8 for _ in range(5)), "phloem": (0.5)}
    cell_line_frequency: dict = {"epidermis": (1e4,), "cortex": (1e4 for _ in range(5)), "stele": (1e4 for _ in range(5)), "phloem": (1e4)}


    def __init__(self, g, time_step_in_seconds: int, **scenario: dict):
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
        self.time_step_in_seconds = time_step_in_seconds
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step_in_seconds, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

        self.init_cell_layers()

    def post_growth_updating(self):
        """
        Description :
            Extend property dictionary upon new element partitioning
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.root_exchange_surface.keys()):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                # All surfaces are extensive, so we need structural mass wise partitioning to initialize
                for prop in self.extensive_variables:
                    getattr(self, prop).update({vid: getattr(self, prop)[parent] * mass_fraction,
                                                    parent: getattr(self, prop)[parent] * (1-mass_fraction)})
                for prop in self.intensive_variables:
                    getattr(self, prop).update({vid: getattr(self, prop)[parent]})


    def init_cell_layers(self):
        self.cell_layers = []

        layer_number = 0
        current_bounding_radius = 1
        for tissue in self.tissue_name:
            for l in range(self.num_layers[tissue]):

                self.cell_layers.append(RootCellLayer(tissue_name=tissue, layer_surface_toS=self.layer_surface_toS[tissue][l], layer_cross_sectionnal_surface_toS=self.layer_cross_sectionnal_surface_toS[tissue][l],
                                                 layer_numbering=layer_number, mean_cell_length_toR=self.mean_cell_length_toR[tissue][l], mean_cell_width_toR=self.mean_cell_width_toR[tissue][l],
                                                 cell_wall_thickness=self.cell_wall_thickness[tissue][l], wall_connectivity_in_layer=self.wall_connectivity_in_layer[tissue][l],
                                                 wall_connectivity_with_inner_neighbor=self.wall_connectivity_with_inner_neighbor[tissue][l], cell_line_frequency=self.cell_line_frequency[tissue][l]))
                
                self.cell_layers[-1].layer_max_radius_toR = current_bounding_radius
                current_bounding_radius -= self.mean_cell_width_toR[tissue][l] + self.cell_wall_thickness[tissue][l]
                layer_number += 1
                    


    # Computation of transport limitations by xylem, endodermis and exodermis differentiations, sequentially.
    @potential
    @state
    def transport_barriers(self):
        """
        This function computes the actual relative conductances of cell walls, endodermis and exodermis for a given root
        element, based on either the distance to root tip or the age of the root segment. It is classified as potential rate as the result
        of this computation limits the actual surface
        :param vid: the vertex ID to compute conductance for (adim).
        :return: the updated element n with the new conductance factors
        """
        for vid in self.vertices:
            n = self.g.node(vid)

            age = n.thermal_time_since_primordium_formation
            
            # TIME WISE BARRIERS OF ENDODERMIS & exodermis:
            # ------------------------------------
            # WITH GOMPERTZ CONTINUOUS EVOLUTION:
            # Note: As the transition between 100% conductance and 0% for both endodermis and exodermis is described by a
            # Gompertz function involving a double exponential, we avoid unnecessary long calculations when the content of
            # the exponential is too high/low:
            # #if self.endodermis_b - self.endodermis_c * age > 1000:
            # #    endodermis_conductance_factor = 1.
            # #else:
            # #    endodermis_conductance_factor = (100 - self.endodermis_a * np.exp(
            # #        -np.exp(self.endodermis_b - self.endodermis_c * age))) / 100.
            # endodermis_conductance_factor = (100 - self.endodermis_a * np.exp(-np.exp(self.endodermis_b / (60.*60.*24.) - self.endodermis_c * age))) / 100.


            # #if self.exodermis_b - self.exodermis_c * age > 1000:
            # #    exodermis_conductance_factor = 1.
            # #else:
            # #    exodermis_conductance_factor = (100 - self.exodermis_a * np.exp(
            # #        -np.exp(self.exodermis_b - self.exodermis_c * age))) / 100.
            # exodermis_conductance_factor = (100 - self.exodermis_a * np.exp(-np.exp(self.exodermis_b / (60. * 60. * 24.) - self.exodermis_c * age))) / 100.

            # DISTANCE WISE APPARITION OF ENDODERMIS AND exodermis DIFFERENTIATION BOUNDARIES
            # We define the distances from apex where barriers start/end:
            start_distance_endodermis = self.start_distance_for_endodermis_factor * n.radius
            end_distance_endodermis = self.end_distance_for_endodermis_factor * n.radius
            start_distance_exodermis = self.start_distance_for_exodermis_factor * n.radius
            end_distance_exodermis = self.end_distance_for_exodermis_factor * n.radius
            
            barycenter_distance = (2 * n.distance_from_tip - n.length) / 2
            
            # ENDODERMIS:
            # Above the starting distance, we consider that the conductance rapidly decreases as the endodermis is formed:
            if barycenter_distance > start_distance_endodermis:
                # # OPTION 1: Conductance decreases as y = x0/x
                # conductance_endodermis = starting_distance_endodermis / distance_from_tip
                # OPTION 2: Conductance linearly decreases with x, up to reaching 0:
                endodermis_conductance_factor = 1 - (barycenter_distance - start_distance_endodermis) \
                                        / (end_distance_endodermis - start_distance_endodermis)
                if endodermis_conductance_factor < 0.:
                    endodermis_conductance_factor = 0.
            # Below the starting distance, the conductance is necessarily maximal:
            else:
                endodermis_conductance_factor = 1

            # EXODERMIS:
            # Above the starting distance, we consider that the conductance rapidly decreases as the exodermis is formed:
            if barycenter_distance > start_distance_exodermis:
                # # OPTION 1: Conductance decreases as y = x0/x
                # conductance_exodermis = starting_distance_exodermis / distance_from_tip
                # OPTION 2: Conductance linearly decreases with x, up to reaching 0:
                exodermis_conductance_factor = 1 - (barycenter_distance - start_distance_exodermis) \
                                        / (end_distance_exodermis - start_distance_exodermis)
                if exodermis_conductance_factor < 0.:
                    exodermis_conductance_factor = 0.
                # Below the starting distance, the conductance is necessarily maximal:
            else:
                exodermis_conductance_factor = 1


            # SPECIAL CASE: # We now consider a special case where the endodermis and/or exodermis barriers are temporarily
            # opened because of the emergence of a lateral root.

            # If there are more than one child, then it means there are lateral roots:
            lateral_children = self.g.Sons(vid, EdgeType='+')
            if len(lateral_children) > 1:
                # We define two maximal thermal durations, above which the barriers are not considered to be affected anymore:
                t_max_endo = self.max_thermal_time_since_endodermis_disruption
                t_max_exo = self.max_thermal_time_since_exodermis_disruption
                # We initialize empty lists of new conductances:
                possible_conductances_endo = []
                possible_conductances_exo = []
                # We cover all possible lateral roots emerging from the current element:
                for child_vid in lateral_children:
                    # We get the lateral root as "son":
                    son = self.g.node(child_vid)
                    # We register the time since this lateral root emerged, and its current length:
                    t = son.thermal_time_since_emergence
                    lateral_length = son.length
                    # If there is a lateral root and this has not emerged yet:
                    if lateral_length <= 0.:
                        # Then we move to the next possible lateral root (otherwise, the loop stops and conductance remains unaltered):
                        continue
                    # ENDODERMIS: If the lateral root has emerged recently, its endodermis barrier has been diminished as soon
                    # as the lateral started to elongate:
                    t_since_endodermis_was_disrupted = t
                    if t_since_endodermis_was_disrupted < t_max_endo:
                        # We increase the relative conductance of endodermis according to the age of the lateral root,
                        # considering that the conductance starts at 1 and linearily decreases with time until reaching 0.
                        # However, if the barrier was not completely formed initially, we should not set it to zero, and therefore
                        # define the new conductance as the maximal value between the original conductance and the new one:
                        new_conductance = max(endodermis_conductance_factor,
                                            (t_max_endo - t_since_endodermis_was_disrupted) / t_max_endo)
                        possible_conductances_endo.append(new_conductance)

                    # exodermis: If the lateral root has emerged recently, its exodermis barrier may have been diminished,
                    # provided that the length of the lateral root is actually higher than the radius of the mother root
                    # (i.e. that the lateral root tip has actually crossed the exodermis of the mother root):
                    if lateral_length >= n.radius:
                        # We approximate the time since the exodermis was disrupted, considering the the lateral root has
                        # elongated at a constant speed:
                        t_since_exodermis_was_disrupted = t * (lateral_length - n.radius) / lateral_length
                        # If this time is small enough, the exodermis barrier may have been compromised:
                        if t_since_exodermis_was_disrupted < t_max_exo:
                            # We increase the relative conductance of exodermis according to the time elapsed since the lateral
                            # root crossed the exodermis, considering that the conductance starts at 1 and linearily decreases
                            # with time until reaching 0. However, if the barrier was not completely formed initially, we should
                            # not set it to zero, and we therefore define the new conductance as the maximal value between the
                            # original conductance and the new one:
                            new_conductance = max(exodermis_conductance_factor,
                                                (t_max_exo - t_since_exodermis_was_disrupted) / t_max_exo)
                            possible_conductances_exo.append(new_conductance)

                # Now that we have covered all lateral roots, we limit the conductance of the barriers of the mother root
                # element by choosing the least limiting lateral root (only active if the lists did not remain empty):
                if possible_conductances_endo:
                    endodermis_conductance_factor = max(possible_conductances_endo)
                if possible_conductances_exo:
                    exodermis_conductance_factor = max(possible_conductances_exo)

            # We record the new conductances of cell walls, endodermis and exodermis:

            self.endodermis_conductance_factor[vid] = endodermis_conductance_factor
            self.exodermis_conductance_factor[vid] = exodermis_conductance_factor
        
            # Logistic xylem differentiation
            logistic_precision = 0.99
            self.xylem_differentiation_factor[vid] = 1 / (1 + (logistic_precision / ((1 - logistic_precision) * np.exp(
                                                            -self.begin_xylem_differentiation)) * np.exp(-barycenter_distance / self.span_xylem_differentiation)))

    # Utility, no decorator needed
    def root_hairs_external_surface(self, root_hair_length, total_root_hairs_number):
        """
        Compute root hairs surface for the considered segment

        :param root_hair_length: the root hait length from exodermis surface to hair tip (m)
        :param total_root_hairs_number: number of root hairs on considered segment (adim)
        :return: the surface (m2)
        """
        return ((self.root_hair_radius * 2 * pi) * root_hair_length) * total_root_hairs_number
    

    @actual
    @state
    def _root_exchange_surface(self, radius, length, exodermis_conductance_factor, endodermis_conductance_factor, root_hair_length, total_root_hairs_number):
        """
        Exchange surface between soil and symplasmic parenchyma.
        Note : here max() is used to prevent going bellow cylinder surface upon exodermis closing.

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param exodermis_conductance_factor: the endodermis barrier differentiation factor (adim)
        :param endodermis_conductance_factor: the endodermis barrier differentiation factor (adim)
        :param root_hair_length: the root hait length from exodermis surface to hair tip (m)
        :param total_root_hairs_number: number of root hairs on considered segment (adim)
        :return: the surface (m2)
        """

        return (exodermis_conductance_factor * sum([layer.cell_surface(radius, length) for layer in self.cell_layers if layer.tissue_name in ("epidermis", "cortex")])
                + endodermis_conductance_factor * sum([layer.cell_surface(radius, length) for layer in self.cell_layers if layer.tissue_name == "stele"])
                + self.root_hairs_external_surface(root_hair_length, total_root_hairs_number))

    @actual
    @state
    def _cortex_exchange_surface(self, radius, length, exodermis_conductance_factor, root_hair_length, total_root_hairs_number):
        """
        Exchange surface between soil and symplasmic cortex. It excludes stele parenchyma surface.
        This is computed as the exchange surface for water absorption from soil to stele apoplasm, which is supposed
        at equilibrium with xylem vessels (so we neglect stele surface between symplasm and apoplasm,
        supposing quick equilibrium inside the root.
        Note: stelar parencyma surface = root_exchange_surface - cortex_exchange_surface

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param exodermis_conductance_factor: the exodermis barrier differentiation factor (adim)
        :param root_hair_length: the root hait length from exodermis surface to hair tip (m)
        :param total_root_hairs_number: number of root hairs on considered segment (adim)
        :return: the surface (m2)
        """
        return (exodermis_conductance_factor * sum([l.cell_surface(radius, length) for l in self.cell_layers if l.tissue_name in ("epidermis", "cortex")]) 
                + self.root_hairs_external_surface(root_hair_length, total_root_hairs_number))
    

    @actual
    @state
    def _xylem_exchange_surface(self, radius, length, xylem_differentiation_factor):
        """
        Exchange surface between root parenchyma and apoplasmic xylem vessels.

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param xylem_differentiation_factor: xylem differentiation, i.e. apoplasmic opening, from 0 to 1 (adim)
        :return: the surface (m2)
        """
        return sum([layer.cell_surface(radius, length) for layer in self.cell_layers if layer.tissue_name == "stele"]) * xylem_differentiation_factor


    @actual
    @state
    def _phloem_exchange_surface(self, radius, length):
        """
        Exchange surface between root parenchyma and apoplasmic xylem vessels.

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :return: the surface (m2)
        """
        return self.cell_layers[-1].cell_surface(radius, length)


    @actual
    @state
    def _symplasmic_volume(self, radius, length):
        """
        Computes symplasmic volume for water content of elements

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :return: the volume (m3)
        """
        return sum([layer.cell_volume(radius, length) for layer in self.cell_layers])


    @actual
    @state
    def _xylem_volume(self, radius, length):
        """
        Computes xylem volume for water transport between elements

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :return: the volume (m3)
        """
        stele_symplasm_volume = sum([layer.cell_volume(radius, length) for layer in self.cell_layers if layer.tissue_name == "stele"])
        stele_radius = radius - sum([layer.mean_cell_width_toR * radius + layer.cell_wall_thickness for layer in ("epidermis", "cortex")])

        return 2 * pi * stele_radius - stele_symplasm_volume
    

    @actual
    @state
    def _xylem_vessel_radii(self, radius):
        vessels_radii = []
        num_central_metaxylem = 1
        num_peripheric_metaxylem = 6
        num_protoxylem_per_metaxylem = 3

        for k in range(num_central_metaxylem):
            vessels_radii.append(0.1*radius)

        for k in range(num_peripheric_metaxylem):
            vessels_radii.append(0.1*0.25*radius)

            for i in range(num_protoxylem_per_metaxylem):
                vessels_radii.append(0.1*0.25*0.25*radius)

        return vessels_radii
    

    @actual
    @state
    def _kr_symplasmic_water(self, radius, length):
        kr_eq = 0

        for layer in self.cell_layers:
            kr_eq = layer.kr_symplasmic_water(kr_eq, radius, length)

        return kr_eq
    

    @actual
    @state
    def _kr_apoplastic_water(self, radius, length, endodermis_conductance_factor):
        return endodermis_conductance_factor * 1 / sum([layer.R_apoplasmic_water(radius, length) for layer in self.cell_layers])
    

    @totalstate
    def _total_phloem_volume(self, radius, length):
        return sum([self.cell_layers[-1].cell_volume(r, l) for r, l in zip(radius.values(), length.values())])


class RootCellLayer:
    
    # Water related parameters
    transmembrane_conductance: float = 1
    plasmodesmata_conductance: float = 1
    cell_wall_conductivity: float = 1

    def __init__(self, tissue_name, layer_surface_toS, layer_cell_perimeter_toR, layer_cross_sectional_surface_toRR, layer_numbering, 
                 mean_cell_length_toR, mean_cell_width_toR, cell_wall_thickness, wall_connectivity_in_layer, wall_connectivity_with_inner_neighbor,
                 # Axial aspects
                 cell_line_frequency, 
                 # Optionnal
                 number_of_cell_lines=None):
        
        vars(self).update({k: v for k, v in locals().items() if k != "self"})


    def cell_surface(self, radius, length):
        return self.layer_cell_perimeter_toR * radius * length + 2 * self.layer_cross_sectional_surface_toRR * (radius ** 2) * self.cell_line_frequency * length


    def cell_volume(self, radius, length):
        return self.layer_cross_sectional_surface_toRR * (radius ** 2) * length * (1 - self.cell_line_frequency * self.cell_wall_thickness)
    

    def kr_symplasmic_water(self, k_symplasmic_eq, radius, length):
        layer_surface = self.cell_surface(radius, length)
        r_transmembrane = 1 / (self.transmembrane_conductance * layer_surface + k_symplasmic_eq)
        if self.layer_numbering == 0:
            R_plasmodesmata = 0

        else:
            bounding_surface = 2 * np.pi * self.layer_max_radius_toR * radius * length
            R_plasmodesmata = 1 / (self.plasmodesmata_conductance * bounding_surface * self.wall_connectivity_with_inner_neighbor)

        return 1 / (r_transmembrane + R_plasmodesmata)
    

    def R_apoplasmic_water(self, radius, length):
        
        crossed_wall_length_between_cell_lines = (1 - self.wall_connectivity_in_layer) * (1 - self.wall_connectivity_with_inner_neighbor) * self.layer_cell_perimeter_toR * radius
        crossed_wall_area_between_cell_lines = self.cell_wall_thickness * length

        kr_between_cell_lines = self.cell_wall_conductivity * crossed_wall_area_between_cell_lines / crossed_wall_length_between_cell_lines
        # Note that we substract shared walls only once because they still contribute once to the flux
        # the whole perimeter is also retreive to account f the pathway on the 2 edges of the cylinfer

        crossed_wall_area_within_cell_line = 2 * np.pi * self.layer_max_radius_toR * radius * self.cell_wall_thickness

        kr_walls_within_cell_line = self.cell_wall_conductivity * crossed_wall_area_within_cell_line / (self.mean_cell_width_toR * radius)

        return 1 / (kr_between_cell_lines + kr_walls_within_cell_line)