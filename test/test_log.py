from rhizodep.rhizodep import Model
from data_utility.logging import Logger

def test_small_run():
    rhizodep = Model(time_step=3600)
    mtg = rhizodep.g
    logger = Logger(model_instance=rhizodep, outputs_dirpath="test/outputs", 
                    time_step_in_hours=1,
                    logging_period_in_hours=24,
                    recording_sums=True,
                    recording_raw=False,
                    recording_images=False, 
                    recording_mtg=False,
                    recording_performance=True,
                    echo=True)
    
    for step in range(50):
        # Placed here also to capture mtg initialization
        logger()

        rhizodep.run()
    
    logger.stop()

test_small_run()