from dataclasses import dataclass, field, fields
import inspect as ins
from functools import partial


@dataclass
class SoilModel:
    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM CARBON MODEL
    hexose_exudation: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user"))
    phloem_hexose_exudation: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user"))
    hexose_uptake_from_soil: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user"))
    phloem_hexose_uptake_from_soil: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of hexose", description="", value_comment="", references="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user"))
    mucilage_secretion: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of equivalent hexose", description="", value_comment="", references="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user"))
    cells_release: float = field(default=0., metadata=dict(unit="mol.s-1", unit_comment="of equivalent hexose", description="", value_comment="", references="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user"))

    # FROM ANATOMY MODEL
    root_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", value_comment="", references="", variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user"))

    # --- INITIALIZE MODEL STATE VARIABLES ---
    # Temperature
    soil_temperature_in_Celsius: float = field(default=15., metadata=dict(unit="°C", unit_comment="", description="soil temperature in contact with roots", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user"))

    # Carbon and nitrogen concentrations
    C_hexose_soil: float = field(default=1e-5, metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Hexose concentration in soil", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user"))
    Cs_mucilage_soil: float = field(default=1e-5, metadata=dict(unit="mol.g-1", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user"))
    Cs_cells_soil: float = field(default=1e-5, metadata=dict(unit="mol.g-1", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user"))
    C_mineralN_soil: float = field(default=5, metadata=dict(unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user"))
    C_AA_soil: float = field(default=1e-4, metadata=dict(unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user"))

    # Water
    water_potential_soil: float = field(default=-0.1e6, metadata=dict(unit="Pa", unit_comment="", description="Mean soil water potential", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="", edit_by="user"))
    volume_soil: float = field(default=1e-7, metadata=dict(unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment", value_comment="", references="", variable_type="state_variable", by="model_soil", state_variable_type="", edit_by="user"))

    # --- INITIALIZES MODEL PARAMETERS ---

    # Temperature
    process_at_T_ref: float = field(default=1., metadata=dict(unit="adim", unit_comment="", description="Proportion of maximal process intensity occuring at T_ref", value_comment="", references="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))

    hexose_degradation_rate_max_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    hexose_degradation_rate_max_A: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    hexose_degradation_rate_max_B: float = field(default=3.98, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="The value for B (Q10) has been fitted from the evolution of Vmax measured by Coody et al. (1986, SBB), who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user"))
    hexose_degradation_rate_max_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))

    mucilage_degradation_rate_max_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    mucilage_degradation_rate_max_A: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    mucilage_degradation_rate_max_B: float = field(default=3.98, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user"))
    mucilage_degradation_rate_max_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))

    cells_degradation_rate_max_T_ref: float = field(default=20, metadata=dict(unit="°C", unit_comment="", description="the reference temperature", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    cells_degradation_rate_max_A: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    cells_degradation_rate_max_B: float = field(default=3.98, metadata=dict(unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user"))
    cells_degradation_rate_max_C: float = field(default=1, metadata=dict(unit="adim", unit_comment="", description="parameter C (either 0 or 1)", value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))

    # Kinetic soil degradation parameters
    hexose_degradation_rate_max: float = field(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, metadata=dict(unit="mol.m-2.s-1", unit_comment="of hexose", description="Maximum degradation rate of hexose in soil", value_comment="", references="According to what Jones and Darrah (1996) suggested, we assume that this Km is 2 times lower than the Km corresponding to root uptake of hexose (350 uM against 800 uM in the original article).", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    Km_hexose_degradation: float = field(default=1000 * 1e-6 / 12., metadata=dict(unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for soil hexose degradation", value_comment="", references="We assume that the maximum degradation rate is 10 times higher than the maximum hexose uptake rate by roots", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    mucilage_degradation_rate_max: float = field(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, metadata=dict(unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of mucilage in soil", value_comment="", references="We assume that the maximum degradation rate for mucilage is equivalent to the one defined for hexose.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    Km_mucilage_degradation: float = field(default=1000 * 1e-6 / 12., metadata=dict(unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil mucilage degradation ", value_comment="", references="We assume that Km for mucilage degradation is identical to the one for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    cells_degradation_rate_max: float = field(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10 / 2, metadata=dict(unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of root cells at the soil/root interface", value_comment="", references="We assume that the maximum degradation rate for cells is equivalent to the half of the one defined for hexose.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))
    Km_cells_degradation: float = field(default=1000 * 1e-6 / 12., metadata=dict(unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil cells degradation", value_comment="", references="We assume that Km for cells degradation is identical to the one for hexose degradation.", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user"))

    def __init__(self, g, time_step_in_seconds):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step_in_seconds: time step of the simulation (s)
        :return:
        """
        self.g = g
        self.props = self.g.properties()
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        self.time_steps_in_seconds = time_step_in_seconds

        self.state_variables = [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]

        for name in self.state_variables:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
            # set default in mtg
            self.props[name].update({key: getattr(self, name) for key in self.vertices})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

    def post_coupling_init(self):
        self.store_functions_call()
        self.check_if_coupled()

    def store_functions_call(self):
        # Storing function calls

        # Local scale processes...
        self.process_methods = [getattr(self, func) for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                                for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'process' in func)]
        self.process_names = [func[8:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'process' in func)]

        # Local scale update...
        self.update_methods = [getattr(self, func) for func in dir(self) if
                                (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_args = [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(getattr(self, func))[0] if arg != "self"]
                             for func in dir(self) if
                             (callable(getattr(self, func)) and '__' not in func and 'update' in func)]
        self.update_names = [func[7:] for func in dir(self) if
                              (callable(getattr(self, func)) and '__' not in func and 'update' in func)]

    def check_if_coupled(self):
        # For all expected input...
        input_variables = [f.name for f in fields(self) if f.metadata["variable_type"] == "input"]
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

    def run_exchanges_and_balance(self):
        """
        Groups and order carbon exchange processes and balance for the provided time step
        """
        # TODO print(self.struct_mass, self.initial_struct_mass to check is they are indeed different here)
        self.add_new_segments()

        self.props.update(self.prc_resolution())
        self.props.update(self.upd_resolution())


    def prc_resolution(self):
        return dict(zip([name for name in self.process_names], map(self.dict_mapper, *(self.process_methods, self.process_args))))

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
            if vid not in list(self.C_hexose_soil.keys()):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                for prop in self.state_variables:
                    # All concentrations, temperature and pressure are intensive, so we need structural mass wise partitioning to initialize
                    getattr(self, prop).update({vid: getattr(self, prop)[parent]})

    # Modification of a process according to soil temperature:
    # --------------------------------------------------------
    def temperature_modification(self, T_ref=0., A=-0.05, B=3., C=1.):
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

        # We initialize the value of the temperature-modified process:
        modified_process = 0.

        # We avoid unwanted cases:
        if C != 0 and C != 1:
            print("The modification of the process at T =", self.soil_temperature_in_Celsius,
                  "only works for C=0 or C=1!")
            print("The modified process has been set to 0.")
            modified_process = 0.
            return 0.
        elif C == 1:
            if (A * (self.soil_temperature_in_Celsius - T_ref) + B) < 0.:
                print("The modification of the process at T =", self.soil_temperature_in_Celsius,
                      "is unstable with this set of parameters!")
                print("The modified process has been set to 0.")
                modified_process = 0.
                return modified_process

        # We compute a temperature-modified process, correspond to a Q10-modified relationship,
        # based on the work of Tjoelker et al. (2001):
        modified_process = self.process_at_T_ref * (A * (self.soil_temperature_in_Celsius - T_ref) + B) ** (1 - C) \
                           * (A * (self.soil_temperature_in_Celsius - T_ref) + B) ** (
                                       C * (self.soil_temperature_in_Celsius - T_ref) / 10.)

        if modified_process < 0.:
            modified_process = 0.

        return modified_process

    def process_hexose_degradation(self, C_hexose_soil, root_exchange_surface):
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
                                                                        T_ref=self.hexose_degradation_rate_max_T_ref,
                                                                        A=self.hexose_degradation_rate_max_A,
                                                                        B=self.hexose_degradation_rate_max_B,
                                                                        C=self.hexose_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of hexose in the soil:
        return max(corrected_hexose_degradation_rate_max * root_exchange_surface * C_hexose_soil / (
                                                                            self.Km_hexose_degradation + C_hexose_soil), 0.)

    def process_mucilage_degradation(self, Cs_mucilage_soil, root_exchange_surface):
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
                                                                    T_ref=self.mucilage_degradation_rate_max_T_ref,
                                                                    A=self.mucilage_degradation_rate_max_A,
                                                                    B=self.mucilage_degradation_rate_max_B,
                                                                    C=self.mucilage_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of mucilage
        # in the soil:
        return max(corrected_mucilage_degradation_rate_max * root_exchange_surface * Cs_mucilage_soil / (
                self.Km_mucilage_degradation + Cs_mucilage_soil), 0.)

    def process_cells_degradation(self, Cs_cells_soil, root_exchange_surface):
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
                                                                        T_ref=self.cells_degradation_rate_max_T_ref,
                                                                        A=self.cells_degradation_rate_max_A,
                                                                        B=self.cells_degradation_rate_max_B,
                                                                        C=self.cells_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of root cells
        # in the soil:
        return max(corrected_cells_degradation_rate_max * root_exchange_surface * Cs_cells_soil / (
                self.Km_cells_degradation + Cs_cells_soil), 0.)

    # TODO FOR TRISTAN: Consider adding similar functions for describing N mineralization/organization in the soil?

    def update_C_hexose_soil(self, C_hexose_soil, volume_soil, hexose_degradation, hexose_exudation,
                             phloem_hexose_exudation, hexose_uptake_from_soil, phloem_hexose_uptake_from_soil):
        return C_hexose_soil + (self.time_steps_in_seconds / volume_soil) * (
            hexose_exudation
            + phloem_hexose_exudation
            - hexose_uptake_from_soil
            - phloem_hexose_uptake_from_soil
            - hexose_degradation
        )

    def update_Cs_mucilage_soil(self, Cs_mucilage_soil, volume_soil, mucilage_secretion, mucilage_degradation):
        return Cs_mucilage_soil + (self.time_steps_in_seconds / volume_soil) * (
            mucilage_secretion
            - mucilage_degradation
        )

    def update_Cs_cells_soil(self, Cs_cells_soil, volume_soil, cells_release, cells_degradation):
        return Cs_cells_soil + (self.time_steps_in_seconds / volume_soil) * (
                cells_release
                - cells_degradation
        )
