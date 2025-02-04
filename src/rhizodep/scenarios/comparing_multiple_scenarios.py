# -*- coding: latin-1 -*-

"""
    This script compares outputs from different scenarios obtained with main_multiple_scenario.py.

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

    dict_of_dataframes = {}
    stat_table=None

    for group in list(scenario_numbers_by_group.keys()):

        # We load all dataframes corresponding to each scenario in a central dictionary:
        for i in scenario_numbers_by_group[group]:
            scenario_name = 'Scenario_%.4d' % i
            results_path = os.path.join('outputs', scenario_name, 'simulation_results.csv')
            dict_of_dataframes[scenario_name] = pd.read_csv(results_path)

        for property in properties:
            scenario_name = 'Scenario_%.4d' % scenario_numbers_by_group[group][0]
            dict_property = {}
            # We initialize the first column of the final table with the time in days in the first table:
            dict_property['Time_in_days'] = dict_of_dataframes[scenario_name]['Time (days)']
            for i in scenario_numbers_by_group[group]:
                scenario_name = 'Scenario_%.4d' % i
                df = dict_of_dataframes[scenario_name]
                dict_property[scenario_name] = df[property]
            # We create a data frame from the column corresponding to the property of each scenario:
            final_table = pd.DataFrame.from_dict(dict_property)
            # We initialize the first column containing Time_in_days for the "stat_table" if this hasn't been done yet:
            if stat_table is None:
                stat_table = final_table.filter(["Time_in_days"])
            # We add new columns to stat_table containing the mean and std values of all columns
            # (except the first one corresponding to time):
            mean_name = 'Mean_' + group
            stat_table[mean_name] = final_table.drop('Time_in_days', axis=1).mean(axis=1)
            std_name = 'Standard_deviation_' + group
            stat_table[std_name] = final_table.drop('Time_in_days', axis=1).std(axis=1)

            final_table_name = 'Comparing ' + property + ' for ' + group + '.csv'
            final_table.to_csv(os.path.join('outputs', final_table_name), na_rep='NA', index=False, header=True)
            stat_table_name = 'Comparing ' + property + ' for all groups.csv'
            stat_table.to_csv(os.path.join('outputs', stat_table_name), na_rep='NA', index=False, header=True)

    return

def plotting_multiple_scenarios(table="Comparing Root structural mass (g).csv", scenario_numbers=[1,2,3,4,5],
                                    y_label="Root structural mass (g)", xmin=0, xmax=20, ymin=0, ymax=0.02,
                                    saving_figure=True):

    table_path = os.path.join('outputs', table)
    df = pd.read_csv(table_path)

    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(8,6))

    for i in scenario_numbers:
        column_name = 'Scenario_%.4d' % i
        plt.plot(df["Time_in_days"], df[column_name], linewidth=3) #color='darkorange'
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

    if saving_figure:
        fig = plt.gcf()
        fig.set_size_inches(50,50)
        fig.savefig('figure.png', dpi=100)

    return

def plotting_mean_values(table="Comparing Root structural mass (g) for all groups.csv",
                         groups=['Group_1', 'Group_2', 'Group_3'],
                         y_label="Root structural mass (g)",
                         xmin=0, xmax=100, ymin=0, ymax=0.2, saving_figure=False):

    table_path = os.path.join('outputs', table)
    df = pd.read_csv(table_path)

    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(8,6))

    for group in groups:
        mean_name = 'Mean_' + group
        std_name = 'Standard_deviation_' + group

        plt.plot(df["Time_in_days"], df[mean_name], linewidth=3) #color='darkorange'
        plt.fill_between(df["Time_in_days"], df[mean_name] - df[std_name], df[mean_name] + df[std_name],
                        alpha=0.2, label='_nolegend_') #facecolor='darkorange'

    plt.xlabel("Time (days)", fontsize=20, fontweight='bold', family="calibri", labelpad=15)
    plt.ylabel(y_label, fontsize=20, fontweight='bold', family="calibri", labelpad=15)
    axes = plt.gca()
    axes.set_frame_on(False)
    axes.axhline(linewidth=6, color="black")
    axes.axvline(linewidth=6, color="black")
    axes.set_xlim(xmin, xmax*1.003)
    axes.set_ylim(ymin, ymax*1.003)
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

    if saving_figure:
        fig = plt.gcf()
        fig.set_size_inches(10,10)
        fig.savefig('figure.png', dpi=300)

    return

########################################################################################################################
########################################################################################################################

creating_new_frames_from_scenarios(properties=['Root surface (m2)'],
                                   scenario_numbers_by_group={'Group_1': [1,2,3,4,5],
                                                              'Group_2': [6,7,8,9,10],
                                                              'Group_3': [11,12,13,14,15]})
# plotting_multiple_scenarios(scenario_numbers=range(1,16), saving_figure=False)
plotting_mean_values(table="Comparing Root surface (m2) for all groups.csv",
                     groups=['Group_1', 'Group_2', 'Group_3'],
                     y_label="Root surface ($\mathregular{m^2}$)",
                     xmin=0, xmax=100, ymin=0, ymax=0.05, saving_figure=True)
plt.show()
plt.close()