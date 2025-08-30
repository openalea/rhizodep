# Public packages
import numpy as np
# Model packages
# Utility packages
from initialize.initialize import MakeScenarios as ms
import matplotlib.pyplot as plt
from simulations.sensitivity_analysis import sobol_analysis


if __name__ == '__main__':
    problem, scenarios_filename, scenarios_names = ms.from_factorial_plan("inputs/Factorial_plan_SA.xlsx", save_scenarios=False, N=20)
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
                #ax.barh("S2", indices["S2"][k], color=colors[k], left=left_S2)
                ax.barh("S1", indices["S1"][k], color=colors[k], label=problem['names'][k], left=left_S1)
                ax.barh("ST", indices["ST"][k], color=colors[k], left=left_ST)

                left_S1 += indices["S1"][k]
                left_ST += indices["ST"][k]
                #left_S2 += indices["S2"][k]
            
            if t == 0.:
                ax.set_ylabel(outputs[o], rotation=0, labelpad=30)
            if o == 0:
                ax.set_title(f"t = {times[t]} h")
            if not already_a_legend:
                ax.legend(loc="center left", bbox_to_anchor=(-2, 0.5))
                already_a_legend = True

    fig.savefig("outputs/SA.png", bbox_inches='tight')

