import rhizodep
from rhizodep.root_growth import RootGrowthModel
from rhizodep.root_carbon import RootCarbonModel
from rhizodep.root_anatomy import RootAnatomy
from rhizodep.rhizo_soil import SoilModel

from metafspm.composite_wrapper import CompositeModel


class Model(CompositeModel):
    """
    Rhizodep model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, time_step: int, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """
        
        # INIT INDIVIDUAL MODULES
        self.root_growth = self.load(RootGrowthModel, time_step, **scenario)
        self.g = self.root_growth.g
        self.root_anatomy = self.load(RootAnatomy, self.g, time_step, **scenario)
        self.root_carbon = self.load(RootCarbonModel, self.g, time_step, **scenario)
        self.soil = self.load(SoilModel, self.g, time_step, **scenario)

        self.models = (self.soil, self.root_growth, self.root_anatomy, self.root_carbon)

        # LINKING MODULES
        self.link_around_mtg(translator_path=rhizodep.__path__[0])

        # Some initialization must be performed AFTER linking modules
        [m.post_coupling_init() for m in self.models]

    def run(self):
        self.soil()
        self.root_growth()

        self.root_anatomy.post_growth_updating()
        self.root_carbon.post_growth_updating()
        self.soil.post_growth_updating()

        self.root_anatomy()
        self.root_carbon()
        #self.root_carbon.check_balance()