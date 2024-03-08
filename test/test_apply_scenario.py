from rhizodep.rhizodep import Model
from data_utility.logging import Logger
from data_utility.data_analysis import analyze_data
from data_utility.preprocess_scenario import make_scenarios

def test_small_run():
    rhizodep = Model(time_step=3600)

    logger = Logger(model_instance=rhizodep, outputs_dirpath="test/outputs", 
                    time_step_in_hours=1,
                    logging_period_in_hours=1,
                    recording_images=True, 
                    recording_mtg=False,
                    recording_raw=False,
                    recording_sums=True,
                    recording_performance=False,
                    echo=True)
    
    for step in range(300):
        # Placed here also to capture mtg initialization
        logger()

        rhizodep.run()
    
    logger.stop()
    analyze_data(outputs_dirpath="test/outputs", 
                 on_sums=True,
                 on_performance=False,
                 target_properties=[]
                 )

test_small_run()