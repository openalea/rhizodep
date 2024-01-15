import numpy as np
from math import pi
from dataclasses import dataclass, field, fields
import inspect as ins
from functools import partial


@dataclass
class RootAnatomy:
    """
    Root anatomy model originating from both Rhizodep model.py and Root_CyNAPS model_topology.py

    Rhizodep forked :
        https://forgemia.inra.fr/tristan.gerault/rhizodep/-/commits/rhizodep_2022?ref_type=heads
    base_commit :
        92a6f7ad927ffa0acf01aef645f9297a4531878c
    """
    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM GROWTH MODEL
    radius: float = field(default=3.5e-4, metadata=dict(unit="m", unit_comment="", description="Example root segment radius", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    length: float = field(default=1.e-3, metadata=dict(unit="m", unit_comment="", description="Example root segment length", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    struct_mass: float = field(default=1.35e-4, metadata=dict(unit="g", unit_comment="", description="Example root segment structural mass", value_comment="", references="", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    root_hair_length: float = field(default=1.e-3, metadata=dict(unit="m", unit_comment="", description="Example root hair length", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between 0.5 and 1.3 mm.", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))
    total_root_hairs_number: float = field(default=30 * (1.6e-4 / 3.5e-4) * 3.e-3 * 1e3, metadata=dict(unit="adim", unit_comment="", description="Example root hairs number on segment external surface", value_comment="30 * (1.6e-4 / radius) * length * 1e3", references=" According to the work of Gahoonia et al. (1997), the root hair density is about 30 hairs per mm for winter wheat, for a root radius of about 0.16 mm.", variable_type="input", by="model_growth", state_variable_type="", edit_by="user"))

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # Surfaces
    root_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="extensive", edit_by="user"))
    cortex_echange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic cortex. It excludes stele parenchyma surface. This is computed as the exchange surface for water absorption from soil to stele apoplasm, which is supposed at equilibrium with xylem vessels (so we neglect stele surface between symplasm and apoplasm, supposing quick equilibrium inside the root.", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="extensive", edit_by="user"))
    apoplasmic_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface to account for exchanges between xylem + stele apoplasm and soil. We account for it through cylindrical surface, a pathway closing as soon as endodermis differentiates", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="extensive", edit_by="user"))
    xylem_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between root parenchyma and apoplasmic xylem vessels.", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="extensive", edit_by="user"))
    phloem_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between root parenchyma and apoplasmic xylem vessels.", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="extensive", edit_by="user"))

    # Volumes
    xylem_volume: float = field(default=0., metadata=dict(unit="m3", unit_comment="", description="xylem volume for water transport between elements", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="extensive", edit_by="user"))

    # Differentiation factors
    endodermis_conductance_factor: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="The endodermis barrier differentiation factor", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="intensive", edit_by="user"))
    epidermis_conductance_factor: float = field(default=0.5, metadata=dict(unit="adim", unit_comment="", description="The epidermis barrier differentiation factor", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="intensive", edit_by="user"))
    xylem_differentiation_factor: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="Xylem differentiation, i.e. apoplasmic opening, from 0 to 1", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="instensive", edit_by="user"))

    # Tissue density
    root_tissue_density: float = field(default=0.10 * 1e6, metadata=dict(unit="g.m3", unit_comment="of structural mass", description="root_tissue_density", value_comment="", references="", variable_type="state_variable", by="model_anatomy", state_variable_type="", edit_by="user"))

    # --- INITIALIZES MODEL PARAMETERS ---

    # Differentiation parameters
    meristem_limit_zone_factor: float = field(default=3., metadata=dict(unit="adim", unit_comment="", description="Ratio between the length of the meristem zone and root radius", value_comment="Overwrite 1. where we assume that the length of the meristem zone is equal to the radius of the root", references="(??) see transition zone reference", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    endodermis_a: float = field(default=100., metadata=dict(unit="adim", unit_comment="", description="This parameter corresponds to the asymptote of the process in Gompertz law describing the evolution of apoplastic barriers with cell age.", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    endodermis_b: float = field(default=3.25 * (60.*60.*24.), metadata=dict(unit="s", unit_comment="time equivalent at T_ref", description="This parameter corresponds to the time lag before the large increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    endodermis_c: float = field(default=1.11 / (60.*60.*24.), metadata=dict(unit="s-1", unit_comment="time equivalent at T_ref", description="This parameter reflects the slope of the increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    epidermis_a: float = field(default=100., metadata=dict(unit="adim", unit_comment="", description="This parameter corresponds to the asymptote of the process in Gompertz law describing the evolution of apoplastic barriers with cell age.", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    epidermis_b: float = field(default=5.32 * (60.*60.*24.), metadata=dict(unit="s", unit_comment="time equivalent at T_ref", description="This parameter corresponds to the time lag before the large increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    epidermis_c: float = field(default=1.11 / (60.*60.*24.), metadata=dict(unit="s-1", unit_comment="time equivalent at T_ref", description="This parameter reflects the slope of the increase in Gompertz law describing the evolution of apoplastic barriers with cell age.", value_comment="", references="estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))

    max_thermal_time_since_endodermis_disruption: float = field(default=6 * 60. * 60., metadata=dict(unit="s", unit_comment="time equivalent at T_ref", description="Maximal thermal time above which no endodermis disruption is considered anymore after a lateral root has emerged", value_comment="", references="We assume that after 6h, no disruption is observed anymore! (??)", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    max_thermal_time_since_epidermis_disruption: float = field(default=48 * 60. * 60., metadata=dict(unit="s", unit_comment="time equivalent at T_ref", description="Maximal thermal time above which no epidermis disruption is considered anymore after a lateral root has emerged", value_comment="", references="We assume that after 48h, no disruption is observed anymore! (??)", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    begin_xylem_differentiation: float = field(default=0., metadata=dict(unit="s", unit_comment="time equivalent at T_ref", description="Parameter indicating at which age xylem differentiation starts for a logistic function", value_comment="", references="", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    span_xylem_differentiation: float = field(default=6. * 60. * 60., metadata=dict(unit="s", unit_comment="time equivalent at T_ref", description="Parameter indicating what time span is necessary for the transition to fully opened xylem / stele apoplasm", value_comment="", references="We assume that after 6h, xylem apoplasm is open (??)", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))

    # Surfacic fractions
    cortical_surfacic_fraction: float = field(default=29., metadata=dict(unit="adim", unit_comment="", description="Cortex (+epidermis) parenchyma surface ratio over root segment's cylinder surface", value_comment="", references="report", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    stellar_surfacic_fraction: float = field(default=11., metadata=dict(unit="adim", unit_comment="", description="Stele (+endodermis) surface ratio over root segment's cylinder surface", value_comment="", references="report", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    phloem_surfacic_fraction: float = field(default=2.5, metadata=dict(unit="adim", unit_comment="", description="phloem surface ratio over root's cylinder surface", value_comment="", references="report", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    xylem_cross_area_surfacic_fraction: float = field(default=0.84 * (0.36 ** 2), metadata=dict(unit="adim", unit_comment="apoplasmic cross-section area ratio * stele radius ratio^2", description="apoplasmic cross-section ratio over root segment's sectional surface", value_comment="", references="report", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))
    root_hair_radius: float = field(default=12 * 1e-6 / 2., metadata=dict(unit="m", unit_comment="", description="Average radius of root hair", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair diameter is relatively constant for different genotypes of wheat and barley, i.e. 12 microns.", variable_type="parameter", by="model_anatomy", state_variable_type="", edit_by="user"))

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
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        self.time_step_in_seconds = time_step_in_seconds

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)

        self.state_variables = [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]

        for name in self.state_variables:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
            # set default in mtg
            self.props[name].update({key: getattr(self, name) for key in self.vertices})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

    def apply_scenario(self, **kwargs):
        """
        Method to superimpose default parameters in order to create a scenario.
        Use Model.documentation to discover model parameters and state variables.
        :param kwargs: mapping of existing variable to superimpose.
        """
        for changed_parameter, value in kwargs.items():
            if changed_parameter in dir(self):
                setattr(self, changed_parameter, value)

    def post_coupling_init(self):
        self.store_functions_call()
        self.check_if_coupled()

    def store_functions_call(self):
        """
        Storing function calls in selffor later mapping of processes
        """

        # segment scale surfaces update...
        self.update_methods = [getattr(self, func) for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                             for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_names = [func[7:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'update' in func)]

    def check_if_coupled(self):
        # For all expected input...
        input_variables = [name for name, value in self.__dataclass_fields__.items() if value.metadata["variable_type"] == "input"]
        for inpt in input_variables:
            # If variable type has not gone to dictionary as it is part of the coupling process
            # we use provided default value to create the dictionnary used in the rest of the model
            if type(getattr(self, inpt)) != dict:
                if inpt not in self.props.keys():
                    self.props.setdefault(inpt, {})
                # set default in mtg
                self.props[inpt].update({key: getattr(self, inpt) for key in self.vertices})
                # link mtg dict to self dict
                setattr(self, inpt, self.props[inpt])

    def get_available_inputs(self):
        for inputs in self.available_inputs:
            source_model = inputs["applier"]
            linker = inputs["linker"]
            for name, source_variables in linker.items():
                # if variables have to be summed
                if len(source_variables.keys()) > 1:
                    return setattr(self, name, dict(zip(getattr(source_model, "vertices"), [sum([getattr(source_model, source_name)[vid] * unit_conversion for source_name, unit_conversion in source_variables.items()]) for vid in getattr(source_model, "vertices")])))
                else:
                    return setattr(self, name, getattr(source_model, list(source_variables.keys())[0]))

    def run_actualize_anatomy(self):
        # We have to renew this call at each time step to ensure model inputs are well updated
        self.get_available_inputs()

        # 1- Update data structure according to structure change
        self.add_new_segments()

        # 2- We update the conductance of tissues
        map(self.transport_barriers, self.vertices)

        # 3- Finally update surfaces, volumes and pathway indicators
        self.props.update(self.upd_resolution())

    def upd_resolution(self):
        return dict(zip([name for name in self.update_names], map(self.dict_mapper, *(self.update_methods, self.update_args))))

    def dict_mapper(self, fcn, args):
        return dict(zip(args[0](), map(fcn, *(d().values() for d in args))))

    def get_up_to_date(self, prop):
        return getattr(self, prop)

    def add_new_segments(self):
        """
        Description :
            Extend property dictionary upon new element partitioning
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.root_exchange_surface.keys()):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                for prop in self.state_variables:
                    # All surfaces are extensive, so we need structural mass wise partitioning to initialize
                    getattr(self, prop).update({vid: getattr(self, prop)[parent] * mass_fraction,
                                                    parent: getattr(self, prop)[parent] * (1-mass_fraction)})

    # Computation of transport limitations by xylem, endodermis and epidermis differentiations, sequentially.
    def transport_barriers(self, vid):
        """
        This function computes the actual relative conductances of cell walls, endodermis and epidermis for a given root
        element, based on either the distance to root tip or the age of the root segment.
        :param vid: the vertex ID to compute conductance for (adim).
        :return: the updated element n with the new conductance factors
        """
        n = self.g.node(vid)

        # CELL WALLS AGING SLOWED AT MERISTEMATIC ZONE:
        if n.distance_from_tip - n.length < self.meristem_limit_zone_factor * n.radius:
            age = n.thermal_time_since_primordium_formation * ((n.distance_from_tip - n.length) / self.meristem_limit_zone_factor)
        else:
            age = n.thermal_time_since_primordium_formation
        
        # BARRIERS OF ENDODERMIS & epidermis:
        # ------------------------------------
        # WITH GOMPERTZ CONTINUOUS EVOLUTION:
        # Note: As the transition between 100% conductance and 0% for both endodermis and epidermis is described by a
        # Gompertz function involving a double exponential, we avoid unnecessary long calculations when the content of
        # the exponential is too high/low:
        if self.endodermis_b - self.endodermis_c * age > 1000:
            endodermis_conductance_factor = 1.
        else:
            endodermis_conductance_factor = (100 - self.endodermis_a * np.exp(
                -np.exp(self.endodermis_b - self.endodermis_c * age))) / 100.
        if self.epidermis_b - self.epidermis_c * age > 1000:
            epidermis_conductance_factor = 1.
        else:
            epidermis_conductance_factor = (100 - self.epidermis_a * np.exp(
                -np.exp(self.epidermis_b - self.epidermis_c * age))) / 100.

        # SPECIAL CASE: # We now consider a special case where the endodermis and/or epidermis barriers are temporarily
        # opened because of the emergence of a lateral root.

        # If there are more than one child, then it means there are lateral roots:
        lateral_children = self.g.Sons(vid, EdgeType='+')
        if len(lateral_children) > 1:
            # We define two maximal thermal durations, above which the barriers are not considered to be affected anymore:
            t_max_endo = self.max_thermal_time_since_endodermis_disruption
            t_max_exo = self.max_thermal_time_since_epidermis_disruption
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
                    # considering that the barrier starts at 1 and linearily decreases with time until reaching 0. However,
                    # if the barrier was not completely formed initially, we should not set it to zero, and therefore
                    # define the new conductance as the maximal value between the original conductance and the new one:
                    new_conductance = max(endodermis_conductance_factor,
                                          (t_max_endo - t_since_endodermis_was_disrupted) / t_max_endo)
                    possible_conductances_endo.append(new_conductance)

                # EPIDERMIS: If the lateral root has emerged recently, its epidermis barrier may have been diminished,
                # provided that the length of the lateral root is actually higher than the radius of the mother root
                # (i.e. that the lateral root tip has actually crossed the epidermis of the mother root):
                if lateral_length >= n.radius:
                    # We approximate the time since the epidermis was disrupted, considering the the lateral root has
                    # elongated at a constant speed:
                    t_since_epidermis_was_disrupted = t * (lateral_length - n.radius) / lateral_length
                    # If this time is small enough, the epidermis barrier may have been compromised:
                    if t_since_epidermis_was_disrupted < t_max_exo:
                        # We increase the relative conductance of epidermis according to the time elapsed since the lateral
                        # root crossed the epidermis, considering that the barrier starts at 1 and linearily decreases with
                        # time until reaching 0. However, if the barrier was not completely formed initially, we should not
                        # set it to zero, and we therefore define the new conductance as the maximal value between the
                        # original conductance and the new one:
                        new_conductance = max(epidermis_conductance_factor,
                                              (t_max_exo - t_since_epidermis_was_disrupted) / t_max_exo)
                        possible_conductances_exo.append(new_conductance)

            # Now that we have covered all lateral roots, we limit the conductance of the barriers of the mother root
            # element by choosing the least limiting lateral root (only active if the lists did not remain empty):
            if possible_conductances_endo:
                endodermis_conductance_factor = max(possible_conductances_endo)
            if possible_conductances_exo:
                epidermis_conductance_factor = max(possible_conductances_exo)

        # We record the new conductances of cell walls, endodermis and epidermis:

        self.endodermis_conductance_factor[vid] = endodermis_conductance_factor
        self.epidermis_conductance_factor[vid] = epidermis_conductance_factor

        # Logistic xylem differentiation
        logistic_precision = 0.99
        self.xylem_differentiation_factor[vid] = 1 / (1 + (logistic_precision / ((1 - logistic_precision) * np.exp(
                                                        -self.begin_xylem_differentiation)) * np.exp(-age / self.span_xylem_differentiation)))

    def root_hairs_external_surface(self, root_hair_length, total_root_hairs_number):
        """
        Compute root hairs surface for the considered segment

        :param root_hair_length: the root hait length from epidermis surface to hair tip (m)
        :param total_root_hairs_number: number of root hairs on considered segment (adim)
        :return: the surface (m2)
        """
        return ((self.root_hair_radius * 2 * pi) * root_hair_length) * total_root_hairs_number

    def update_root_exchange_surface(self, radius, length, epidermis_conductance_factor, endodermis_conductance_factor, root_hair_length, total_root_hairs_number):
        """
        Exchange surface between soil and symplasmic parenchyma.
        Note : here max() is used to prevent going bellow cylinder surface upon epidermis closing.

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param epidermis_conductance_factor: the endodermis barrier differentiation factor (adim)
        :param endodermis_conductance_factor: the endodermis barrier differentiation factor (adim)
        :param root_hair_length: the root hait length from epidermis surface to hair tip (m)
        :param total_root_hairs_number: number of root hairs on considered segment (adim)
        :return: the surface (m2)
        """
        return (2 * pi * radius * length * max(self.cortical_surfacic_fraction * epidermis_conductance_factor +
                                               self.stellar_surfacic_fraction * endodermis_conductance_factor, 1.) +
                self.root_hairs_external_surface(root_hair_length, total_root_hairs_number))

    def update_cortex_exchange_surface(self, radius, length, epidermis_conductance_factor, root_hair_length, total_root_hairs_number):
        """
        Exchange surface between soil and symplasmic cortex. It excludes stele parenchyma surface.
        This is computed as the exchange surface for water absorption from soil to stele apoplasm, which is supposed
        at equilibrium with xylem vessels (so we neglect stele surface between symplasm and apoplasm,
        supposing quick equilibrium inside the root.
        Note: stelar parencyma surface = root_exchange_surface - cortex_exchange_surface

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param epidermis_conductance_factor: the epidermis barrier differentiation factor (adim)
        :param root_hair_length: the root hait length from epidermis surface to hair tip (m)
        :param total_root_hairs_number: number of root hairs on considered segment (adim)
        :return: the surface (m2)
        """
        return (2 * pi * radius * length * self.cortical_surfacic_fraction * epidermis_conductance_factor +
                self.root_hairs_external_surface(root_hair_length, total_root_hairs_number))

    def update_apoplasmic_exchange_surface(self, radius, length, endodermis_conductance_factor):
        """
        Exchange surface to account for exchanges between xylem + stele apoplasm and soil.
        We account for it through cylindrical surface, a pathway closing as soon as endodermis differentiates

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param endodermis_conductance_factor: the endodermis barrier differentiation factor (adim)
        :return: the surface (m2)
        """
        return 2 * pi * radius * length * endodermis_conductance_factor

    def udpate_xylem_exchange_surface(self, radius, length, xylem_differentiation_factor):
        """
        Exchange surface between root parenchyma and apoplasmic xylem vessels.

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :param xylem_differentiation_factor: xylem differentiation, i.e. apoplasmic opening, from 0 to 1 (adim)
        :return: the surface (m2)
        """
        return 2 * pi * radius * length * self.stellar_surfacic_fraction * xylem_differentiation_factor

    def udpate_phloem_exchange_surface(self, radius, length):
        """
        Exchange surface between root parenchyma and apoplasmic xylem vessels.

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :return: the surface (m2)
        """
        return 2 * pi * radius * length * self.phloem_surfacic_fraction

    def update_xylem_volume(self, radius, length):
        """
        Computes xylem volume for water transport between elements

        :param radius: the root segment radius (m)
        :param length: the root segment length (m)
        :return: the volume (m3)
        """
        return pi * (radius ** 2) * self.xylem_cross_area_surfacic_fraction * length
