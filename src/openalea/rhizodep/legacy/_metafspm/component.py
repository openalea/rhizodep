from dataclasses import dataclass, field, fields
from typing import Literal
import numpy as np

from .component_factory import *



def declare(unit: str, unit_comment: str, description: str,  min_value: float, max_value: float, value_comment: str, references: str, DOI: list,
              variable_type: Literal["state_variable", "plant_scale_state", "input", "parameter"], by: str,
              state_variable_type: Literal["massic_concentration", "intensive", "extensive", "NonInertialExtensive", "NonInertialIntensive", "descriptor"],
              edit_by: Literal["user", "dev"], default=None, default_factory=None):
    """
    Resulting from a consensus, this function is used to constrain component variables declaration in a dataclass in a commonly admitted way.

    :param default: Default value of the attribute if not superimposed by scenario of coupling.
    :param unit: International system unit
    :param unit_comment: More precision about unit (example : mol of ... per g of ...)
    :param description: Full description of the variable purpose, scale and related hypotheses.
    :param value_comment: Commentary if default value identified from literature has been overwritten, store original value and explain why it was changed.
    :param references: References for value and hypotheses (format Author et al., Year).
    :param DOI: DOIs of the referred papers, stored in a list of str.
    :param variable_type: variable type according to component standards; -input- will be read as expected input from another model. If not coupled, the model will keep this default. -state_variable- are the segment scale state variables. -plant_scale_state- are the summed totals variables or the plant scale variables. -parameter- are the model parameters.
    :param by: which model is provider of the considered variable?
    :param state_variable_type: intensive (size invariant property) or extensive (size dependant property).
    :param edit_by: Explicit whether the considered default value could be superimposed only by a developer or a user. This can be used to exposed more precisely certain parametrization sensitive variables for user.
    :raise BadDefaultError: If '(type(min_value)==float and default < min_value) or (type(max_value)==float and default > max_value)':
    :raise BadUnitError: If 'unit not in dir(UnitRegistry())'
    :raise BadVariableTypeError: If 'variable_type not in ("input", "state_variable", "plant_scale_state", "parameter")'
    :raise BadStateTypeError: If 'state_variable_type not in ("intensive", "extensive")'
    :raise BadEditError: If 'edit_by not in ("user", "developer")'
    """
    if default_factory:
        return field(default_factory=default_factory,
                        metadata=dict(unit=unit, unit_comment=unit_comment, description=description, min_value=min_value,
                                    max_value=max_value, value_comment=value_comment, references=references, DOI=DOI,
                                    variable_type=variable_type, by=by,
                                    state_variable_type=state_variable_type, edit_by=edit_by))
    else:
        return field(default=default,
                        metadata=dict(unit=unit, unit_comment=unit_comment, description=description, min_value=min_value,
                                    max_value=max_value, value_comment=value_comment, references=references, DOI=DOI,
                                    variable_type=variable_type, by=by,
                                    state_variable_type=state_variable_type, edit_by=edit_by))


@dataclass
class Model:
    """
    Base component for structuring base FSPM modules

    HYPOTHESES:
        self.g.properties() must have been stored self.props during child class __init__
    """

    choregrapher = Choregrapher()
    #available_inputs = []  # Will be incremented during the coupling
    # pullable_inputs = {}

    def __call__(self, *args):
        self.pull_available_inputs()
        self.choregrapher(module_family=self.__class__.__name__, *args)

    @property
    def inputs(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "input"]

    @property
    def state_variables(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]
    
    @property
    def extensive_variables(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] == "extensive")]
    
    @property
    def massic_concentration(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] == "massic_concentration")]
    
    @property
    def intensive_variables(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] == "intensive")]
    
    @property
    def non_inertial_extensive(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] == "NonInertialExtensive")]

    @property
    def non_inertial_intensive(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] == "NonInertialIntensive")]
    
    @property
    def non_inertial_variables(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] in ("NonInertialIntensive", "NonInertialExtensive"))]

    @property
    def descriptor(self):
        return [f.name for f in fields(self) if (f.metadata["variable_type"] == "state_variable" and f.metadata["state_variable_type"] == "descriptor")]

    @property
    def plant_scale_state(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "plant_scale_state"]

    @property
    def parameter(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "parameter"]

    def apply_scenario(self, **kwargs):
        """
        Method to superimpose default parameters in order to create a scenario.
        Use Model.documentation to discover model parameters and state variables.
        :param kwargs: mapping of existing variable to superimpose.
        """
        for changed_parameter, value in kwargs.items():
            if changed_parameter in dir(self):
                setattr(self, changed_parameter, value)

    def link_self_to_mtg(self, ignore=[]):
        # for input variables, initialize homogeneous values on each vertices. 
        # This behavior will be overwritten in case of module providing the input variable
        for name in self.inputs:
            if not (name in self.props.keys() and len(self.props[name]) == len(self.vertices)): 
                # if it is not provided by mtg file, Use by default value everywhere
                self.props.setdefault(name, {})
                self.props[name].update({key: getattr(self, name) for key in self.vertices})

        # for segment scale state variables
        for name in self.state_variables:
            if name not in ignore:
                self.props.setdefault(name, {})
                # set default in mtg, state_variable prevail on inputs
                self.props[name].update({key: getattr(self, name) for key in self.vertices})

        # for plant scale state variables
        for name in self.plant_scale_state:
            self.props.setdefault(name, {})
            # set default in mtg, state_variable prevail on inputs
            self.props[name].update({1: getattr(self, name)})
                

    def pull_available_inputs(self):
        props = self.props
        for input, source_variables in self.pullable_inputs.items():
            vertices = props[list(source_variables.keys())[0]].keys()
            props[input].update({vid: sum([props[variable][vid]*unit_conversion 
                                           for variable, unit_conversion in source_variables.items()]) 
                                 for vid in vertices})


    def temperature_modification_old(self, soil_temperature=15, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
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
        # We avoid unwanted cases:
        if C != 0 and C != 1:
            print("The modification of the process at T =", soil_temperature,
                  "only works for C=0 or C=1!")
            print("The modified process has been set to 0.")
            return 0.
        elif C == 1:
            if (A * (soil_temperature - T_ref) + B) < 0.:
                print("The modification of the process at T =", soil_temperature,
                      "is unstable with this set of parameters!")
                print("The modified process has been set to 0.")
                modified_process = 0.
                return modified_process

        # We compute a temperature-modified process, correspond to a Q10-modified relationship,
        # based on the work of Tjoelker et al. (2001):
        modified_process = process_at_T_ref * (A * (soil_temperature - T_ref) + B) ** (1 - C) \
                           * (A * (soil_temperature - T_ref) + B) ** (
                                   C * (soil_temperature - T_ref) / 10.)

        return max(modified_process, 0.)
    

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
        modified_process = process_at_T_ref * (A * (soil_temperature - T_ref) + B) ** (1 - C) \
                           * (A * (soil_temperature - T_ref) + B) ** (
                                   C * (soil_temperature - T_ref) / 10.)
        
        return np.where(((C != 0) & (C != 1)) | ((C == 1) & ((A * (soil_temperature - T_ref) + B) < 0.)), 0.,
                        np.maximum(modified_process, 0.))

