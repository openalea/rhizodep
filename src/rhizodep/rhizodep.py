import os
import pickle

from model_growth import RootGrowthModel
from model_carbon import RootCarbonModel
from model_anatomy import RootAnatomy
from model_soil import SoilModel

from root_cynaps.wrapper import ModelWrapper


class Model(ModelWrapper):
    """
    Rhizodep model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step an time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, time_step: int):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """


        # INIT INDIVIDUAL MODULES
        self.root_growth = RootGrowthModel(time_step)
        self.g = self.root_growth.g
        self.root_anatomy = RootAnatomy(self.g, time_step)
        self.root_carbon = RootCarbonModel(self.g, time_step)
        self.soil = SoilModel(self.g, time_step)

        # Voir initialiser dedans
        self.models = (self.root_growth, self.root_anatomy, self.root_carbon, self.soil)

        # LINKING MODULES
        if not os.path.isfile("translator.pckl"):
            self.translator_matrix_builder()
        with open("translator.pckl", "rb") as f:
            translator = pickle.load(f)
        self.link_around_mtg(translator)

        # Some initialization must be performed after linking modules
        self.root_carbon.post_coupling_init()
        self.root_growth.post_coupling_init()

    def run(self):
        # Update environment boundary conditions
        # Update soil state
        self.soil.run_update_patches()

        # Compute state variations for water and then nitrogen
        self.root_carbon.run_exchanges_and_balance()

        # Compute root growth from resulting states
        self.root_growth.run_time_step_growth()
        # Update topological surfaces and volumes based on other evolved structural properties
        self.root_anatomy.run_anatomy_update()
