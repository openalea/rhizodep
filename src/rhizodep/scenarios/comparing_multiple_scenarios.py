# -*- coding: latin-1 -*-

"""
    This script allows to compare the outputs from different scenarios obtained with running_scenarios.py.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import os
import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

def creating_new_frames_from_scenarios(properties=['Root structural mass (g)'],
                                       scenario_numbers_by_group={'Group_1': [1,3], 'Group_2': [2,4,5]}):

    """
    This function is used to compare the outputs of different scenarios at the scale of the whole root system over time.
    The function uses the files 'simulation_results.csv' stored in the folders of each scenario within the main folder
    'outputs'. It generates two csv files for each property, one displaying all values from the different scenarios, and
    the other one showing some statistical results for comparing the outputs of the different scenarios among groups.
    :param properties: a list containing the different properties to be compared among scenarios
    :param scenario_numbers_by_group: a dictionnary associating a list of scenario numbers to a given group
    :return: different csv files in the 'outputs' folder
    """

    # We intialize an empty dictionnary and list:
    dict_of_dataframes = {}
    list_of_all_scenarios = []
    # We cover each group of scenarios:
    for group in list(scenario_numbers_by_group.keys()):
        # We load all dataframes corresponding to each scenario of this group in a central dictionary:
        for i in scenario_numbers_by_group[group]:
            scenario_name = 'Scenario_%.4d' % i
            results_path = os.path.join('outputs', scenario_name, 'simulation_results.csv')
            dict_of_dataframes[scenario_name] = pd.read_csv(results_path)
            # And we add the current scenario number to the list containing all scenario numbers:
            list_of_all_scenarios.append(i)

    # FOR EACH PROPERTY TO CONSIDER:
    for property in properties:

        # CREATING ONE TABLE WITH THE VALUES OF THE PROPERTY FOR EACH SCENARIO:
        # We initialize an empty dictionnary that will contain the property values for each scenario:
        dict_property = {}
        # We initialize the first column of the final table with the time in days from the last table:
        dict_property['Time_in_days'] = dict_of_dataframes[scenario_name]['Final time (days)']
        # For each scenario to consider:
        for i in list_of_all_scenarios:
            scenario_name = 'Scenario_%.4d' % i
            df = dict_of_dataframes[scenario_name]
            dict_property[scenario_name] = df[property]
        # We create a data frame from the column corresponding to the property of each scenario:
        final_table = pd.DataFrame.from_dict(dict_property)
        # We record the final table comparing the values of this specific property among scenarios:
        final_table_name = 'Comparing ' + property + ' for all scenarios.csv'
        final_table.to_csv(os.path.join('outputs', final_table_name), na_rep='NA', index=False, header=True)

        # CREATING ONE TABLE WITH THE MEAN +/- SD VALUES OF THE PROPERTY FOR EACH GROUP:
        # We initialize the stat table from the previous table by keeping only the first column with time:
        stat_table = final_table.filter(["Time_in_days"])
        # We now cover each group of scenarios:
        for group in list(scenario_numbers_by_group.keys()):
            # We create a list containing the names of scenarios to be kept for performing the statistics:
            list_of_group_scenario_names = []
            # For each scenario in the group:
            for i in scenario_numbers_by_group[group]:
                scenario_name = 'Scenario_%.4d' % i
                list_of_group_scenario_names.append(scenario_name)
            # We define the group table as a subset of the final table previously created:
            group_table = final_table.filter(list_of_group_scenario_names)
            # We add new columns to stat_table containing the mean and sd values of all columns of group_table:
            # (except the first one corresponding to time):
            mean_name = 'Mean_' + group
            stat_table[mean_name] = group_table.mean(axis=1)
            std_name = 'Standard_deviation_' + group
            stat_table[std_name] = group_table.std(axis=1)
        # We record the final table comparing the statistical results for this property among the different groups:
        stat_table_name = 'Comparing mean ' + property + '.csv'
        stat_table.to_csv(os.path.join('outputs', stat_table_name), na_rep='NA', index=False, header=True)

    return

def plotting_multiple_scenarios(table="Comparing Root structural mass (g).csv", scenario_numbers=[1,2,3,4,5],
                                y_label="Root structural mass (g)", xmin=0, xmax=20, ymin=0, ymax=0.02,
                                saving_plot=True, plot_name="root_structural_mass_values.png"):
    """
    This function creates a plot showing the evolution of a given property of the whole root system over time,
    for different scenarios. It uses a csv file generated by the function 'creating_new_frames_from_scenarios'.
    :param table: the name of the csv file to be read in the folder 'outputs'
    :param scenario_numbers: a list containing the numbers of each scenario to be displayed
    :param y_label: the title of the y-axis
    :param xmin: minimal value of time on the x-axis
    :param xmax: maximal value of time on the x-axis
    :param ymin: minimal value on the y-axis
    :param ymax: maximal value on the y-axis
    :param saving_plot: if True, the plot will be recorded
    :param plot_name: if needed, the name of the plot image to be saved
    :return:
    """

    # 1) We load one csv file corresponding to the values of one property over time among different scenarios:
    table_path = os.path.join('outputs', table)
    try:
        df = pd.read_csv(table_path)
    except:
        print("ERROR: the file", table, "cannot be accessed!")
        return

    # 2) We plot the evolution of the property over time for each scenario.

    # We specifiy the display parameters of Matplotlib:
    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(8,6))

    # For each scenario, we add the curve corresponding to its outputs:
    for i in scenario_numbers:
        column_name = 'Scenario_%.4d' % i
        plt.plot(df["Time_in_days"], df[column_name], linewidth=3) #color='darkorange'
    # We improve the plot:
    plt.xlabel("Time (days)", fontsize=20, fontweight='bold', family="calibri", labelpad=15)
    plt.ylabel(y_label, fontsize=20, fontweight='bold', family="calibri", labelpad=15)
    axes = plt.gca()
    axes.set_frame_on(False)
    axes.axhline(linewidth=4, color="black")
    axes.axvline(linewidth=4, color="black")
    axes.set_xlim(xmin, xmax*1.001)
    axes.set_ylim(ymin, ymax*1.005)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    axes.legend(labels=scenario_numbers, title='Scenario', loc='upper left')
    #axes.xaxis.get_major_ticks()[0].label1.set_visible(False) # To remove the first 0
    plt.tick_params(axis = 'both', direction='out',width=2, length=7, labelsize = 20, pad=10)
    plt.xticks(fontweight='bold', family="calibri")
    plt.yticks(fontweight='bold', family="calibri")

    # We create a nice legend:
    axes.legend(labels=scenario_numbers, prop={'size': 15, 'weight':'bold'},
                loc='upper left', frameon=False).set_title(title='Scenarios', prop={'size': 15, 'weight': 'bold'})

    # If needed, we record the plot:
    if saving_plot:
        if plot_name is None:
            plot_name = 'plot_values.png'
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(os.path.join('outputs', plot_name), dpi=300)

    return

def plotting_mean_values(table="Comparing Root structural mass (g) for all groups.csv",
                         groups=['Group_1', 'Group_2', 'Group_3'],
                         y_label="Root structural mass (g)",
                         xmin=0, xmax=100, ymin=0, ymax=0.2,
                         saving_plot=True, plot_name="root_structural_mass_mean.png"):
    """
    This function creates a plot showing the evolution of a given property of the whole root system over time,
    averaged among different scenarios. It uses a csv file generated by function 'creating_new_frames_from_scenarios'.
    :param table: the name of the csv file to be read in the folder 'outputs'
    :param groups: a list containing the names of the groups to be displayed on the plot
    :param y_label: the title of the y-axis
    :param xmin: minimal value of time on the x-axis
    :param xmax: maximal value of time on the x-axis
    :param ymin: minimal value on the y-axis
    :param ymax: maximal value on the y-axis
    :param saving_plot: if True, the plot will be recorded
    :param plot_name: if needed, the name of the plot image to be saved
    :return:
    """

    # 1) We load one csv file corresponding to the mean value of one property for different scenarios over time:
    table_path = os.path.join('outputs', table)
    try:
        df = pd.read_csv(table_path)
    except:
        print("ERROR: the file", table, "cannot be accessed!")
        return

    # 2) We plot the evolution of the property over time for each scenario.

    # We specifiy the display parameters of Matplotlib:
    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(8,6))

    # We cover each group of scenarios to be treated together:
    for group in groups:
        mean_name = 'Mean_' + group
        std_name = 'Standard_deviation_' + group

        # We add the curve showing the mean +/- sd values of the group over time:
        plt.plot(df["Time_in_days"], df[mean_name], linewidth=3) #color='darkorange'
        plt.fill_between(df["Time_in_days"], df[mean_name] - df[std_name], df[mean_name] + df[std_name],
                        alpha=0.2, label='_nolegend_') #facecolor='darkorange'

    # We improve the plot:
    plt.xlabel("Time (days)", fontsize=20, fontweight='bold', family="calibri", labelpad=15)
    plt.ylabel(y_label, fontsize=20, fontweight='bold', family="calibri", labelpad=15)
    axes = plt.gca()
    axes.set_frame_on(False)
    axes.axhline(linewidth=6, color="black")
    axes.axvline(linewidth=6, color="black")
    axes.set_xlim(xmin, xmax*1.001)
    axes.set_ylim(ymin, ymax*1.005)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    plt.tick_params(axis='both', direction='out', width=3, length=7, labelsize=20, pad=10)
    plt.xticks(fontweight='bold', family="calibri")
    plt.yticks(fontweight='bold', family="calibri")

    # We may modify the position and the presence of the ticks if we wish...
    xlocs, xlabels = plt.xticks()
    ylocs, ylabels = plt.yticks()
    # Here we remove the first and last ticks of each axis:
    new_xlocs = np.delete(xlocs,[0,len(xlocs)-1])
    new_ylocs = np.delete(ylocs, [0, len(ylocs) - 1])
    plt.xticks(ticks=new_xlocs)
    plt.yticks(ticks=new_ylocs)

    # We create a nice legend:
    legend_names = []
    for group in groups:
        legend_names.append(group.replace('Group_',''))
    axes.legend(labels=legend_names, prop={'size': 15, 'weight':'bold'},
                loc='upper left', frameon=False).set_title(title='Groups', prop={'size': 15, 'weight': 'bold'})

    # If needed, we record the plot:
    if saving_plot:
        if plot_name is None:
            plot_name = 'plot_mean.png'
        fig = plt.gcf()
        fig.set_size_inches(10,10)
        fig.savefig(os.path.join('outputs', plot_name), dpi=300)

    return

########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == '__main__':
    # (Note: this condition avoids launching automatically the program when imported in another file)

    # EXAMPLE: We want to plot the property "Root structural mass" from the scenario 1, 2 and 3, by reading
    # the corresponding csv files in the 'outputs' folder, and show the mean value between scenario 1 and 2.

    print(" Extracting the values for different scenarios...")
    creating_new_frames_from_scenarios(properties=['Root structural mass (g)'],
                                       scenario_numbers_by_group={'Group_1': [1,2],
                                                                  'Group_2': [3]})
    print("   > Outputs comparison is done, new csv files have been generated!")

    print("Plotting the values of each scenario...")
    plotting_multiple_scenarios(table="Comparing Root structural mass (g) for all scenarios.csv", scenario_numbers=[1,2,3],
                                y_label="Root structural mass (g)", xmin=0, xmax=1, ymin=0, ymax=0.1,
                                saving_plot=True, plot_name="root_structural_mass_values.png")

    print("Plotting the mean values of each group...")
    plotting_mean_values(table="Comparing mean Root structural mass (g).csv",
                         groups=['Group_1', 'Group_2'],
                         y_label="Root structural mass (g)",
                         xmin=0, xmax=1, ymin=0, ymax=0.1,
                         saving_plot=True, plot_name="root_structural_mass_mean.png")
    plt.show()
    plt.close()