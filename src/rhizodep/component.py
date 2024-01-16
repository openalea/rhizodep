from dataclasses import dataclass, field, fields
import inspect as ins
from functools import partial
from pint import UnitRegistry


def model_var(default, min_value: float, max_value: float, unit: str, unit_comment: str, description: str, value_comment: str, references: str, DOI: list,
              variable_type: str, by: str, state_variable_type: str, edit_by: str):
    """
    This function is used to constrain component variables declaration in a dataclass in a commonly admitted way

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
    return field(default=default,
                 metadata=dict(unit=unit, unit_comment=unit_comment, description=description, min_value=min_value,
                               max_value=max_value, value_comment=value_comment, references=references, DOI=DOI,
                               variable_type=variable_type, by=by,
                               state_variable_type=state_variable_type, edit_by=edit_by))


@dataclass
class ComponentBase:
    """
    Base component for structuring base FSPM modules

    HYPOTHESES:
        self.g.properties() must have been stored self.props during child class __init__
    """
    available_inputs = []

    @property
    def inputs(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]

    @property
    def state_variables(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]

    @property
    def plant_scale_state(self):
        return [f.name for f in fields(self) if f.metadata["variable_type"] == "state_variable"]

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

    def link_self_to_mtg(self):
        # for segment scale state variables
        for name in self.state_variables:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
                # set default in mtg
                self.props[name].update({key: getattr(self, name) for key in self.vertices})
            else:
                self.props[name].update({1: getattr(self, name)})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

        # for plant scale state variables
        for name in self.plant_scale_state:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
                # set default in mtg
                self.props[name].update({key: getattr(self, name) for key in self.vertices})
            else:
                self.props[name].update({1: getattr(self, name)})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

    def check_if_coupled(self):
        # For all expected input...
        for inpt in self.inputs:
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
                    return setattr(self, name, dict(zip(getattr(source_model, "vertices"), [
                        sum([getattr(source_model, source_name)[vid] * unit_conversion for source_name, unit_conversion
                             in source_variables.items()]) for vid in getattr(source_model, "vertices")])))
                else:
                    return setattr(self, name, getattr(source_model, list(source_variables.keys())[0]))

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


class ComponentIndependentSegments(ComponentBase):
    """
    Base component for modules considering methods affecting all segments in parallel.

    HYPOTHESES :
        self.keywords_method_filters, a list containing methods to aggregate for parallel resolution, will have to be
        declared on child class __init__
        self.struct_mass must be a state variable.
    """

    def post_coupling_init(self):
        self.get_available_inputs()
        self.store_functions_call()
        self.check_if_coupled()

    def store_functions_call(self):
        """
        Storing function calls in self for later mapping of processes

        self.keywords_method_filters should be probided in child class!
        This indicates which methods to gather by keyword name for parallel resolution.
        """
        for keyword in self.keywords_method_filters:
            setattr(self, keyword + "_methods", [getattr(self, func) for func in dir(self) if
                                                 (callable(
                                                     getattr(self, func)) and '__' not in func and 'process' in func)])
            setattr(self, keyword + "_arguments", [[partial(self.get_up_to_date, arg) for arg in ins.getfullargspec(
                getattr(self, func))[0] if arg != "self"] for func in dir(self) if (callable(getattr(self, func))
                                                                                    and '__' not in func and 'process' in func)])
            setattr(self, keyword + "_names", [func[len(keyword) + 1:] for func in dir(self) if
                                               (callable(
                                                   getattr(self, func)) and '__' not in func and 'process' in func)])

    def parallel_resolution(self, keyword, fcn):
        names = getattr(self, keyword + "_names")
        methods = getattr(self, keyword + "_methods")
        arguments = getattr(self, keyword + "_arguments")
        return dict(zip([name for name in names], map(fcn, *(methods, arguments))))

    def dict_mapper(self, fcn, args):
        return dict(zip(args[0](), map(fcn, *(d().values() for d in args))))

    def dict_no_mapping(self, fcn, args):
        return {1: fcn(*(d() for d in args))}

    def get_up_to_date(self, prop):
        return getattr(self, prop)

    def resolution_over_vertices(self, chunk, fncs):
        for vid in chunk:
            if self.struct_mass[vid] > 0:
                for method in fncs:
                    method(v=vid)
