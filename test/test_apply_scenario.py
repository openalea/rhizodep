import os
from rhizodep.rhizodep import Model
from data_utility.logging import Logger
from data_utility.data_analysis import analyze_data
from data_utility.preprocess_scenario import MakeScenarios as ms

def single_run(scenario, outputs_dirpath="test/outputs"):
    rhizodep = Model(time_step=3600, **scenario)

    logger = Logger(model_instance=rhizodep, outputs_dirpath=outputs_dirpath, 
                    time_step_in_hours=1,
                    logging_period_in_hours=1,
                    recording_images=False, 
                    recording_mtg=False,
                    recording_raw=False,
                    recording_sums=True,
                    recording_performance=True,
                    echo=True)
    
    for step in range(250):
        # Placed here also to capture mtg initialization
        logger()

        rhizodep.run()
    
    logger.stop()
    analyze_data(outputs_dirpath=outputs_dirpath, 
                 on_sums=True,
                 on_performance=True,
                 target_properties=[]
                 )
    
def test_apply_scenarios():
    scenarios = ms.from_excel(file_path="test/inputs/scenario_test.xlsx", which=["Reference"])
    for scenario_name, scenario in scenarios.items():
        single_run(scenario=scenario, outputs_dirpath=os.path.join("test/outputs", str(scenario_name)))

test_apply_scenarios()