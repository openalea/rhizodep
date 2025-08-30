from multiprocessing.shared_memory import SharedMemory
import numpy as np
from openalea.metafspm.utils import ArrayDict, mtg_to_arraydict

# Modules / components
from rhizodep.root_carbon import RootCarbonModel
from rhizodep.root_growth import RootGrowthModel
from openalea.rootcynaps import RootAnatomy

# Utilities
from openalea.metafspm.composite_wrapper import CompositeModel
from openalea.metafspm.component_factory import Choregrapher
from openalea.fspm.utility.writer.visualize import plot_mtg


debug = False

class RhizoDep(CompositeModel):
    """
    RhizoDep model class composed 
    """

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil,
                name: str="Plant", time_step: int=3600, coordinates: list=[0, 0, 0], rotation: float=0, translator_path: dict = {}, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """
        # DECLARE GLOBAL SIMULATION TIME STEP, FOR THE CHOREGRAPHER TO KNOW IF IT HAS TO SUBDIVIDE TIME-STEPS
        self.name = name
        self.coordinates = coordinates
        self.rotation = rotation

        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0

        parameters = scenario["parameters"]
        root_parameters = parameters["rhizodep"]["roots"]
        self.input_tables = scenario["input_tables"]

        # INIT INDIVIDUAL MODULES
        if len(scenario["input_mtg"]) > 0:
            self.root_growth = RootGrowthModel(g=scenario["input_mtg"]["root_mtg_file"], time_step=time_step, **root_parameters)
        else:
            self.root_growth = RootGrowthModel(g=None, time_step=time_step, **root_parameters)
        self.g_root = self.root_growth.g
        # We have to update the coordinates of the new / imported MTG for other model's initialization
        plot_mtg(self.g_root)
        self.root_anatomy = RootAnatomy(self.g_root, time_step, **root_parameters)
        self.root_carbon = RootCarbonModel(self.g_root, time_step, **root_parameters)

        components = (self.root_growth, self.root_anatomy, self.root_carbon)
        descriptors = []
        for c in components:
            descriptors += c.descriptor
        # descriptors.remove("vertex_index")

        # NOTE : Important that this type conversion occurs after initiation of the modules
        mtg_to_arraydict(self.g_root, ignore=descriptors)
        
        # LINKING MODULES
        self.declare_data_and_couple_components(root=self.g_root,
                                                translator_path=translator_path,
                                                components=components)
        
        self.soil_handshake = {v: k for k, v in enumerate(self.plant_side_soil_inputs + self.soil_outputs)}
        # print(self.soil_handshake)
        
        # Specific here TODO remove later
        self.root_carbon.collar_children = self.root_growth.collar_children
        self.root_carbon.collar_skip = self.root_growth.collar_skip

        # Provide signature for the MTG
        # Retreive the queues to communicate with environment models
        self.queues_soil_to_plants=queues_soil_to_plants
        self.queue_plants_to_soil=queue_plants_to_soil

        # Get properties from each MTG
        self.root_props = self.g_root.properties()
        
        # Performed in initialization and run to update coordinates
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

        self.name = name
        # ROOT PROPERTIES INITIAL PASSING IN MTG
        self.root_props["plant_id"] = name
        self.root_props["model_name"] = self.__class__.__name__
        self.model_name = self.__class__.__name__
        self.carried_components = [component.__class__.__name__ for component in self.components]

        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        # print(buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if isinstance(value, ArrayDict):
                buf[self.soil_handshake[name],:len(value)] = value.values_array()
            else:
                print(name, "should be passed")
        
        shm.close()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "carried_components": self.carried_components, "handshake": self.soil_handshake})

        # Retreive post environments init states
        self.get_environment_boundaries()

        # Send command to environments models to run first
        self.send_plant_status_to_environment()

        # TP 
        self.root_props["parent_id"] = ArrayDict(self.root_props["parent_id"])


    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)

        # Retrieve soil and light status for plant
        self.get_environment_boundaries()
        
        # Compute root growth from resulting states
        self.root_growth(modules_to_update=[c for c in self.components if c.__class__.__name__ != "RootGrowthModel"], soil_boundaries_to_infer=self.soil_outputs)
        
        # Update MTG coordinates accounting for position in the scene
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

        # Update topological surfaces and volumes based on other evolved structural properties
        self.root_anatomy()

        # Compute state variations for water and then carbon and nitrogen
        self.root_carbon()

        # Send plant status to soil and light models
        self.send_plant_status_to_environment()

        self.time += 1


    def get_environment_boundaries(self):
        # Wait for results from both soil and light model before begining
        soil_boundary_props = self.queues_soil_to_plants[self.name].get()

        # NOTE : here you have to perform a per-variable update otherwise dynamic links are broken
        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        vertices = buf[self.soil_handshake["vertex_index"]]
        vertices_mask = vertices >= 1
        for variable_name in self.soil_outputs: # TODO : soil_outputs come from declare_data_and_couple_components, not a good structure to keep
            # print(len(self.root_props[variable_name]))
            if variable_name not in self.root_props.keys(): # Actually used? I am not sure
                self.root_props[variable_name] = ArrayDict()
            
            # self.root_props[variable_name].assign_all(buf[self.soil_handshake[variable_name]][vertices_mask])
            self.root_props[variable_name].scatter(vertices[vertices_mask], buf[self.soil_handshake[variable_name]][vertices_mask])
            
        shm.close()


    def send_plant_status_to_environment(self):
        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        # print(buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if isinstance(value, ArrayDict):
                buf[self.soil_handshake[name],:len(value)] = value.values_array()
            else:
                print(name, "should be passed")
        
        shm.close()

        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "handshake": self.soil_handshake})
