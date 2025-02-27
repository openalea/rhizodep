# -*- coding: latin-1 -*-

"""
    This script allows to run RhizoDep over one scenario or a list of tutorial, using a specific set of parameters and
    options, which are specified in an input file.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

from openalea.rhizodep.tool.running_scenarios import *


########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == '__main__':
    # (Note: this condition avoids launching automatically the program when imported in another file)

    # # CASE 1 - CALLING ONE SCENARIO ONLY:
    # #####################################
    #
    # inputs = None
    # outputs = 'outputs'
    # scenario = 4
    #
    # try:
    #     opts, args = getopt.getopt(sys.argv[1:], "i:o:s:d", ["inputs=", "outputs=", "scenario="])
    # except getopt.GetoptError as err:
    #     print(str(err))
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt in ("-i", "--inputs"):
    #         inputs = arg
    #     elif opt in ("-o", "--outputs"):
    #         outputs = arg
    #     elif opt in ("-s", "--scenario"):
    #         scenario = int(arg)
    #
    # run_one_scenario(scenario_id=1,
    #                  inputs_dir_path="C:/Users/frees/rhizodep/simulations/running_scenarios/inputs",
    #                  outputs_dir_path='outputs',
    #                  scenarios_list="scenarios_list.xlsx")

    # CASE 2 - CALLING MULTIPLE SCENARIOS:
    ######################################
    # We can clear the folder containing previous outputs:
    previous_outputs_clearing(output_path="outputs")
    # We run the tutorial in parallel :
    # WATCH OUT: you will still need to manually modify the default arguments of 'run_one_scenarios',
    # e.g. the name of the file where to read scenario instructions, even if you have entered it below!
    run_multiple_scenarios(scenarios_list="scenarios_list.xlsx",
                           input_path="inputs",
                           output_path="outputs")