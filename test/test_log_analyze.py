from rhizodep.rhizodep import Model
from data_utility.logging import Logger
from data_utility.data_analysis import analyze_data

def test_small_run():
    rhizodep = Model(time_step=3600, random=False)

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
    analyze_data()

test_small_run()