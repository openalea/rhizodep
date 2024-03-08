from rhizodep.rhizodep import Model
from data_utility.logging import Logger


def test_small_run():
    rhizodep = Model(time_step=3600)
    logger = Logger(model_instance=rhizodep, outputs_dirpath="test/outputs", 
                    time_step_in_hours=1,
                    logging_period_in_hours=1,
                    recording_sums=False,
                    recording_raw=False,
                    recording_images=True, plotted_property="hexose_exudation",
                    recording_mtg=False,
                    recording_performance=True,
                    echo=True)
    
    for step in range(30):
        # Placed here also to capture mtg initialization
        logger()

        rhizodep.run()
    
    logger.stop()

test_small_run()
