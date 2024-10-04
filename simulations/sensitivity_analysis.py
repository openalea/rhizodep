# Public packages
import os
import numpy as np
from SALib.analyze import sobol
# Model packages
# Utility packages
from log.logging import Logger
from initialize.initialize import MakeScenarios as ms
from initialize.initialize import read_table
import matplotlib.pyplot as plt
from simulations.single_plant.simulation import simulate_scenarios


def sobol_analysis(problem, results_dirpath, scenarios_names, times=[], outputs=[]):
    result_tables = [read_table(os.path.join(results_dirpath, name, "MTG_properties/MTG_properties_summed/plant_scale_properties.csv")) 
                     for name in scenarios_names]
    analyses = {}
    for t in times:
        results = {output: np.array([table[output][t] for table in result_tables], dtype=float) for output in outputs}
        analyses[t] = {output: sobol.analyze(problem, results[output], calc_second_order=True) for output in outputs}

    return analyses


if __name__ == '__main__':
    problem, scenarios_filename, scenarios_names = ms.from_factorial_plan("inputs/Factorial_plan_SA.xlsx", N=20)
    scenarios = ms.from_table(scenarios_filename, which=scenarios_names)
    simulate_scenarios(scenarios, simulation_length=300, echo=False, log_settings=Logger.light_log)
    times=[142, 284]
    outputs=["total_struct_mass", "length", "C_hexose_root", "AA", "hexose_exudation", "import_Nm"]
    analyses = sobol_analysis(problem=problem, results_dirpath="outputs", scenarios_names=scenarios_names, times=times, outputs=outputs)
    
    fig, axes = plt.subplots(len(outputs), len(times), figsize=(16, 6))

    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(problem["names"])))
    already_a_legend = False
    
    for t in range(len(times)):
        for o in range(len(outputs)):
            if len(times) == 1 and len(outputs)==1:
                ax = axes
            else:
                ax = axes[o][t]
            indices = analyses[times[t]][outputs[o]]
            
            left_S1, left_ST, left_S2 = 0, 0, 0
            for k in range(len(problem["names"])):
                ax.barh("S1", indices["S1"][k], color=colors[k], label=problem['names'][k], left=left_S1)
                ax.barh("ST", indices["ST"][k], color=colors[k], left=left_ST)
                #ax.barh("S2", indices["S2"][k], xerr=indices["S2_conf"][k], label=problem['names'][k], left=left_S2)
                left_S1 += indices["S1"][k]
                left_ST += indices["ST"][k]
                #left_S2 += indices["S2"][k]
            
            if t == 0.:
                ax.set_ylabel(outputs[o])
            if o == 0:
                ax.set_title(f"t = {times[t]} h")
            if not already_a_legend:
                ax.legend(loc="center left", bbox_to_anchor=(-2, 0.5))
                already_a_legend = True

    fig.savefig("outputs/SA.png", bbox_inches='tight')