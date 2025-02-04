import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp, trunc
from matplotlib.ticker import StrMethodFormatter

plt.rcdefaults()

# Transforming dataset:
# ---------------------
def treating_z_dataframe(directory='outputs',
                         z_min=0., z_max=1., z_interval=0.05,
                         plants_per_m2=35,
                         input_treatment=False):

    # Then we read the file and copy it in a dataframe "df":
    input_file_path=os.path.join(directory, 'z_classification.csv')
    df = pd.read_csv(input_file_path, sep=',', header=0)

    # We will use a function that gives a list containing the useful names of the variables with each z range:
    def grouping_z_columns_by_property(property):
        """
        This will return a list containing all headers related to the specified property.
        """
        list = []
        # For each interval of z values to be considered:
        for z_start in np.arange(z_min, z_max, z_interval):
            # We recreate the exact name of the category to display on the graph:
            # name_category_z = str(z_start) + "-" + str(z_start + z_interval) + " m"
            part_1 = '{:.2f}'.format(z_start)
            part_2 = '{:.2f}'.format(z_start + z_interval)
            # We recreate the exact name of the header where to read the value
            name_values_z = property + '_' + str(round(z_start, 3)) + "-" \
                            + str(round(z_start + z_interval, 3)) + "_m"
            list.append(name_values_z)
        return list

    # We create a new variable containing the number of the current day:
    df['day_number'] = np.trunc(df['time_in_days'])

    if input_treatment:
        # Calculating net inputs from cumulated values:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for name in grouping_z_columns_by_property("struct_mass"):
            new_name = "net_" + name
            # We create a new "net" column that will contain the net production of struct_mass for a given z-range:
            df[new_name] = df[name] - df.shift(1)[name]
        for name in grouping_z_columns_by_property("root_necromass"):
            new_name = "net_" + name
            # We create a new "net" column that will contain the net production of struct_mass for a given z-range:
            df[new_name] = df[name] - df.shift(1)[name]

    # We create a copy of df for calculating cumulative values:
    df_cum = df.copy(deep=True)
    # We calculate cumulative values for each variable in the columns:
    for col in df.columns:
        new_var = 'cum_' + col
        df_cum[new_var] = np.cumsum(df[col])
    # We remove unnecessary cumulative time columns:
    df_cum = df_cum.drop(columns=['cum_time_in_days', 'cum_day_number'])
    # We select only the lines where a new day is reached:
    df_cum = df_cum.loc[df_cum['time_in_days'] == df_cum['day_number']]
    # # We remove the column 'day_number':
    # df_cum = df_cum.drop(columns=['day_number'])
    # We move the column 'time_in_days' at the beginning of the file:
    first_column = df_cum.pop('time_in_days')
    df_cum.insert(0, 'time_in_days', first_column)
    # We record the table in a new csv file:
    cum_by_day_path=os.path.join(directory, 'z_classification_cum_by_day.csv')
    df_cum.to_csv(cum_by_day_path, na_rep='NA', index=False, header=True)

    # We group by day_number and sum each variable:
    df_sum = df.groupby(by=['day_number'], as_index=False).sum()
    # We adjust the time in days according to day_number:
    df_sum['time_in_days'] = df_sum['day_number'] + 1
    # We remove the column 'day_number':
    df_sum = df_sum.drop(columns=['day_number'])
    # We move the column 'time_in_days' at the beginning of the file:
    first_column = df_sum.pop('time_in_days')
    df_sum.insert(0, 'time_in_days', first_column)
    # We record the table in a new csv file:
    sum_by_day_path = os.path.join(directory, 'z_classification_sum_by_day.csv')
    df_sum.to_csv(sum_by_day_path, na_rep='NA', index=False, header=True)

    # We group by day_number and average each variable:
    df_mean = df.groupby(by=['day_number'], as_index=False).mean()
    # We adjust the time in days according to day_number:
    df_mean['time_in_days'] = df_mean['day_number'] + 1
    # We remove the column 'day_number':
    df_mean = df_mean.drop(columns=['day_number'])
    # We move the column 'time_in_days' at the beginning of the file:
    first_column = df_mean.pop('time_in_days')
    df_mean.insert(0, 'time_in_days', first_column)
    # We record the table in a new csv file:
    mean_by_day_path = os.path.join(directory, 'z_classification_mean_by_day.csv')
    df_mean.to_csv(mean_by_day_path, na_rep='NA', index=False, header=True)

    # We group by day_number and take the max of each variable:
    df_max = df.groupby(by=['day_number'], as_index=False).max()
    # We adjust the time in days according to day_number:
    df_max['time_in_days'] = df_max['day_number'] + 1
    # We remove the column 'day_number':
    df_max = df_max.drop(columns=['day_number'])
    # We move the column 'time_in_days' at the beginning of the file:
    first_column = df_max.pop('time_in_days')
    df_max.insert(0, 'time_in_days', first_column)
    # We record the table in a new csv file:
    max_by_day_path = os.path.join(directory, 'z_classification_max_by_day.csv')
    df_max.to_csv(max_by_day_path, na_rep='NA', index=False, header=True)

    if input_treatment:
        # We create a special table for root C inputs by day:
        df1 = pd.DataFrame(df_sum["time_in_days"])
        df2 = pd.DataFrame(df_sum[grouping_z_columns_by_property("net_struct_mass")])
        df3 = pd.DataFrame(df_sum[grouping_z_columns_by_property("net_root_necromass")])
        df4 = pd.DataFrame(df_sum[grouping_z_columns_by_property("net_hexose_exudation")])
        df5 = pd.DataFrame(df_sum[grouping_z_columns_by_property("total_rhizodeposition")])

        # We convert the inputs into the proper units (gC m-2):
        df2 = df2 * 0.44 * plants_per_m2
        df3 = df3 * 0.44 * plants_per_m2
        df4 = df4 * 6 * 12.01 * plants_per_m2
        df5 = df5 * 6 * 12.01 * plants_per_m2

        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df3.reset_index(drop=True, inplace=True)
        df4.reset_index(drop=True, inplace=True)
        df5.reset_index(drop=True, inplace=True)

        frames = [df1, df2, df3, df4, df5]
        df_input = pd.concat(frames, axis=1, sort=False, ignore_index=False)
        input_by_day_path = os.path.join(directory, 'z_classification_input_by_day.csv')
        df_input.to_csv(input_by_day_path, na_rep='NA', index=False, header=True)

        # We cumulate input values:
        df_input_cum = df_input.copy(deep=True)
        # We calculate cumulative values for each variable in the columns:
        for col in df_input.columns:
            new_var = 'cum_' + col
            df_input_cum[new_var] = np.cumsum(df_input[col])
        # We remove unnecessary cumulative time columns:
        df_input_cum = df_input_cum.drop(columns=['cum_time_in_days'])
        # We move the column 'time_in_days' at the beginning of the file:
        first_column = df_input_cum.pop('time_in_days')
        df_input_cum.insert(0, 'time_in_days', first_column)
        # We record the table in a new csv file:
        cum_input_path = os.path.join(directory, 'z_classification_cum_input_by_day.csv')
        df_input_cum.to_csv(cum_input_path, na_rep='NA', index=False, header=True)

    # Finally:
    print("")
    print("Additional tables have been created in the folder", directory, "(check the files!).")

    return


# Creating a bar plot:
# ---------------------
def making_a_bar_graph(categories, values, stacked_barplot=False, value_min=1e-12, value_max=1e0, log=True, title="",
                       format_x='%.2E', label=None, color=None):

    # Default values
    if label is None:
        label = [""]
    if color is None:
        color = ["blue", "green"]

    fig, ax = plt.subplots(figsize=(8, 6))  # Create the figure
    # fig.subplots_adjust(left=0.115, right=0.88)
    y_pos = np.arange(len(categories))
    values = np.array(values)
    values_cum = values.cumsum(axis=1)

    if stacked_barplot:

        y = 0
        # Stacked horizontal bar plot:
        # ---------------------
        for x in range(0, len(values[0, :])):
            # legend_name= label[x]
            # y += values[:, x]
            # ax.barh(y_pos, y, log=log, align='center', color=color[x], height=1, label = legend_name)
            legend_name = label[x]
            y = values[:, x]
            y_start = values_cum[:, x] - values[:, x]
            ax.barh(y_pos, y, left=y_start, log=log, align='center', color=color[x], height=1, label=legend_name)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(title)
        ax.set_xlim(value_min, value_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.patch.set_facecolor('white')
        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2E'))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(format_x))
        ax.legend(loc="lower right")
        plt.tight_layout()

    else:

        # Horizontal bar plot:
        # ---------------------
        # ax.barh(y_pos, values, 0.3, align='center', color='green')
        ax.barh(y_pos, values, log=log, align='center', color=color[0], height=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(title)
        ax.set_xlim(value_min, value_max)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.patch.set_facecolor('white')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter(format_x))
        # plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0,0), useOffset=False)
        plt.tight_layout()
        # ax.set_title(title)

    return


# Plotting several graphs:
# -------------------------
def plotting_on_z(input_file_path='z_classification.csv',
                  property='hexose_exudation',
                  title="",
                  color=None,
                  z_min=0., z_max=1., z_interval=0.1,
                  value_min=1e-12,
                  value_max=1e0,
                  log=True,
                  recording_images=True,
                  outputs_path='outputs'
                  ):

    # Default value
    if color is None:
        color = ['green']

    # Then we read the file and copy it in a dataframe "df":
    df = pd.read_csv(input_file_path, sep=',', header=0)

    if recording_images:
        # We define the directory "z_barplots":
        z_bar_dir = os.path.join('z_barplots')
        # If this directory doesn't exist:
        if not os.path.exists(z_bar_dir):
            # Then we create it:
            os.mkdir(z_bar_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(z_bar_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    # We initialize empty lists:
    categories = []
    values = []

    # For each line of the data frame that contains information on sucrose input:
    for i in range(0, len(df['time_in_days'])):

        print("Printing plot", i + 1, "out of", len(df['time_in_days']), "...")

        # For each interval of z values to be considered:
        for z_start in np.arange(z_min, z_max, z_interval):
            # We recreate the exact name of the category to display on the graph:
            # name_category_z = str(z_start) + "-" + str(z_start + z_interval) + " m"
            part_1 = '{:.2f}'.format(z_start)
            part_2 = '{:.2f}'.format(z_start + z_interval)
            name_category_z = part_1 + "-" + part_2 + " m"
            categories.append(name_category_z)
            # We recreate the exact name of the header where to read the value
            name_values_z = property + '_' + str(round(z_start, 3)) + "-" \
                            + str(round(z_start + z_interval, 3)) + "_m"
            values.append(df.loc[i, name_values_z])

        # Creating the chart for this time step:
        making_a_bar_graph(categories=categories, values=values, value_min=value_min, value_max=value_max,
                           log=log, title=title, label=[""], color=color)

        # figure_name = 'graph_' + str(i) + '.png'
        if recording_images:
            number = "z_plot_" + str(i).zfill(5)
            image_name = os.path.join(z_bar_dir, number)
            plt.savefig(image_name)
            plt.close()

        # Emptying the current list of values:
        categories = []
        values = []

    plt.show()

    return


# Plotting several graphs:
# -------------------------
def plotting_on_z_stacked(input_file_path='z_classification.csv',
                          property=None,
                          label=None,
                          title="",
                          color=None,
                          z_min=0., z_max=1., z_interval=0.1,
                          value_min=1e-12,
                          value_max=1e0,
                          format_x='%.2E',
                          log=True,
                          recording_images=True,
                          outputs_path='outputs'
                          ):

    # Default values
    if color is None:
        color = ['blue', 'green']
    if label is None:
        label = ['Cumulated hexose exudation', 'Cumulated hexose degradation']
    if property is None:
        property = ['hexose_exudation', 'struct_mass']

    # Then we read the file and copy it in a dataframe "df":
    df = pd.read_csv(input_file_path, sep=',', header=0)

    if recording_images:
        # We define the directory "z_barplots":
        z_bar_dir = os.path.join(outputs_path, 'z_barplots')
        # If this directory doesn't exist:
        if not os.path.exists(z_bar_dir):
            # Then we create it:
            os.mkdir(z_bar_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(z_bar_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    # We initialize empty lists:
    categories_z = []
    name_values_z = []
    # We initialize an empty Numpy array:
    n_z_categories = trunc((z_max - z_min) / z_interval)
    values = np.zeros([n_z_categories, len(property)])

    # For each line of the data frame that contains information on sucrose input:
    for i in range(0, len(df['time_in_days']) - 1):

        print("Printing plot", i + 1, "out of", len(df['time_in_days']) - 1, "...")
        index_z = 0
        # For each interval of z values to be considered:
        for z_start in np.arange(z_min, z_max, z_interval):
            # We recreate the exact name of the category to display on the graph:
            # name_category_z = str(z_start) + "-" + str(z_start + z_interval) + " m"
            part_1 = '{:.2f}'.format(z_start)
            part_2 = '{:.2f}'.format(z_start + z_interval)
            name_category_z = part_1 + "-" + part_2 + " m"
            categories_z.append(name_category_z)
            # We recreate the exact name of the header where to read the value
            for x in range(0, len(property)):
                name_x = property[x] + "_" + str(round(z_start, 3)) + "-" + str(round(z_start + z_interval, 3)) + "_m"
                name_values_z.append(name_x)
                # np.append(values, df.loc[i, name_values_z[x]], x)
                values[index_z][x] = df.loc[i, name_values_z[-1]]
            index_z += 1

        # Creating the chart for this time step:
        making_a_bar_graph(categories=categories_z, values=values, stacked_barplot=True, value_min=value_min, value_max=value_max,
                           format_x=format_x, log=log, title=title, label=label, color=color)

        # figure_name = 'graph_' + str(i) + '.png'
        if recording_images:
            number = "z_plot_" + str(i).zfill(5)
            image_name = os.path.join(z_bar_dir, number)
            plt.savefig(image_name)
            plt.close()

        # Emptying the current list of values:
        categories_z = []
        values = np.zeros([n_z_categories, len(property)])

    plt.show()

    return


########################################################################################################################
########################################################################################################################

# ACTUAL COMPUTING:
###################

# Plotting the evolution of one rate over time:
# ----------------------------------------------
# treating_z_dataframe(file_name='z_classification.csv')
# plotting_on_z(file_name='z_classification_sum_by_day.csv',
#               property='net_hexose_exudation',
#               title= "Net hexose exudation (mol of hexose per day)",
#               color='royalblue',
#               value_min=0, value_max=1e-5,
#               z_min=0.00, z_max=1., z_interval=0.05,
#               log=False)


# # For creating plots of net inputs over time:
# #---------------------------------------------
# treating_z_dataframe(file_name='z_classification.csv',
#                      z_min=0., z_max=1., z_interval=0.05,
#                      plants_per_m2=35,
#                      input_treatment=True)
# plotting_on_z_stacked(file_name='z_classification_input_by_day.csv',
#                       property=['net_hexose_exudation', 'net_root_necromass'],
#                       label = ['Exudates', 'Dead roots'],
#                       color=['royalblue', 'darkkhaki'],
#                       title="\n Net root-derived C inputs (gC m-2 day-1) per 5-cm layer",
#                       z_min=0., z_max=1., z_interval=0.05,
#                       value_min=0,
#                       value_max=0.02,
#                       log=False,
#                       recording_images=True
#                       )

# # For creating plots of root surface over time:
# #----------------------------------------------
# treating_z_dataframe(file_name='z_classification.csv',
#                      z_min=0., z_max=1., z_interval=0.05,
#                      plants_per_m2=350,
#                      input_treatment=True)
# plotting_on_z_stacked(file_name='z_classification_max_by_day.csv',
#                       property=['surface'],
#                       label = ['Root surface'],
#                       color=['brown'],
#                       title="\n Root surface (m-2) produced over time per 5-cm layer",
#                       z_min=0., z_max=1, z_interval=0.05,
#                       value_min=0,
#                       value_max=0.005,
#                       format_x='%.3f',
#                       log=False,
#                       recording_images=True
#                       )

# # For creating plots of cumulated inputs over time:
# # -------------------------------------------------
#
# targeted_path=os.path.join('outputs', 'Scenario_0001')
#
# treating_z_dataframe(directory=targeted_path,
#                      z_min=0., z_max=1., z_interval=0.1,
#                      plants_per_m2=35,
#                      input_treatment=True)
# plotting_on_z_stacked(input_file_path=os.path.join(targeted_path,'z_classification_cum_input_by_day.csv'),
#                       property=['cum_net_hexose_exudation', 'cum_net_root_necromass'],
#                       label=['Exudates', 'Dead roots'],
#                       color=['royalblue', 'darkkhaki'],
#                       title="\n Cumulated root-derived C inputs (gC m-2) per 5-cm layer",
#                       z_min=0., z_max=1., z_interval=0.1,
#                       value_min=0,
#                       value_max=0.6,
#                       log=False,
#                       recording_images=True,
#                       outputs_path=targeted_path
#                       )

# For creating plots of cumulated inputs over time for a list of scenarios:
# -------------------------------------------------------------------------
scenario_numbers=[1]
for i in scenario_numbers:

    scenario_name = 'Scenario_%.4d' % i
    targeted_path=os.path.join('outputs', scenario_name)

    treating_z_dataframe(directory=targeted_path,
                         z_min=0., z_max=0.5, z_interval=0.05,
                         plants_per_m2=240,
                         input_treatment=True)
    plotting_on_z_stacked(input_file_path=os.path.join(targeted_path,'z_classification_cum_input_by_day.csv'),
                          property=['cum_total_rhizodeposition', 'cum_net_root_necromass'],
                          label=['Rhizodeposits', 'Dead roots'],
                          color=['royalblue', 'darkkhaki'],
                          title="\n Cumulated root-derived C inputs (gC m-2) per 5-cm layer",
                          z_min=0., z_max=0.5, z_interval=0.05,
                          value_min=0,
                          value_max=1.,
                          log=False,
                          recording_images=True,
                          outputs_path=targeted_path
                          )

# # For plotting the SOM pools over time:
# # --------------------------------------
# treating_z_dataframe(file_name='SOM_dynamics_by_z_RAPE_50_without_priming.csv')
# plotting_on_z_stacked(file_name='z_classification_max_by_day.csv',
#                       property=['exudates','dead_roots','POM','MAOM'],
#                       # property=['POM','MAOM'],
#                       label = ['Exudates','Dead roots','POM','MAOM'],
#                       # label=['POM','MAOM'],
#                       color=['royalblue', 'darkkhaki', 'gold','saddlebrown'],
#                       # color=['royalblue', 'brown'],
#                       title="Soil organic C (gC m-2) per 5-cm layer",
#                       z_min=0., z_max=1., z_interval=0.05,
#                       value_min=0,
#                       value_max=10,
#                       log=False,
#                       recording_images=True
#                       )

# Plotting the accumulation over time:
# -------------------------------------

# # Exudation:
# plotting_on_z(file_name='z_classification_cum.csv',
#               property='cum_net_hexose_exudation',
#               title= "Cumulated net hexose exudation (mol of hexose)",
#               color='green',
#               value_min=0, value_max=1e-3,
#               z_min=0.00, z_max=1., z_interval=0.05,
#               log=False)
#
# # Root biomass:
# plotting_on_z(file_name='z_classification_cum.csv',
#               property='struct_mass',
#               title= "Root biomass (g)",
#               color='brown',
#               value_min=0, value_max=1e-1,
#               z_min=0.00, z_max=1., z_interval=0.05,
#               log=False)
