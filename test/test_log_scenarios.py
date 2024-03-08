from rhizodep.rhizodep import Model
from data_utility.logging import Logger
from data_utility.preprocess_scenario import preprocessor

def test_single_run(scenario):
    rhizodep = Model(time_step=3600, random=False, **scenario)

    logger = Logger(model_instance=rhizodep, outputs_dirpath="test/outputs", 
                    time_step_in_hours=1,
                    logging_period_in_hours=24,
                    recording_images=False, 
                    recording_mtg=False,
                    recording_raw=True,
                    recording_sums=False,
                    recording_performance=True,
                    echo=True)
    
    for step in range(200):
        # Placed here also to capture mtg initialization
        logger()

        rhizodep.run()
    
    logger.terminate()

def test_multiple_scenarios():
    scenarios = preprocessor()

    for scenario in scenarios:
        test_single_run(scenario=scenario)


test_multiple_scenarios()
