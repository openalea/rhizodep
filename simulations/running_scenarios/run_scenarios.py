import os
from rhizodep.rhizodep import Model
from data_utility.logging import Logger
from data_utility.data_analysis import analyze_data
from data_utility.preprocess_scenario import MakeScenarios as ms

def single_run(scenario, outputs_dirpath="simulations/running_scenarios/outputs"):
    rhizodep = Model(time_step=3600, **scenario)

    logger = Logger(model_instance=rhizodep, outputs_dirpath=outputs_dirpath, 
                    time_step_in_hours=1,
                    logging_period_in_hours=12,
                    recording_images=False, plotted_property="C_hexose_root",
                    recording_mtg=False,
                    recording_raw=False,
                    recording_sums=True,
                    recording_performance=True,
                    echo=True)
    
    for step in range(400):
        # Placed here also to capture mtg initialization
        logger()

        rhizodep.run()
    
    logger.stop()
    analyze_data(outputs_dirpath=outputs_dirpath, 
                 on_sums=True,
                 on_performance=True,
                 target_properties=[]
                 )
    
def main_run_scenarios():
    scenarios = ms.from_excel(file_path="simulations/running_scenarios/inputs/scenario_test.xlsx", which=["Reference"])
    for scenario_name, scenario in scenarios.items():
        single_run(scenario=scenario, outputs_dirpath=os.path.join("simulations/running_scenarios/outputs", str(scenario_name)))

if __name__ == "__main__":
    main_run_scenarios()