#  -*- coding: utf-8 -*-

"""
    rhizodep.tools
    ~~~~~~~~~~~~~

    The module :mod:`rhizodep.tools` defines useful functions for data preprocessing, graph making...

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

# TODO: add functions from making_graph.py and video.py

import os
from decimal import Decimal
from math import pi, cos, sin, floor, ceil, trunc, log10
from copy import deepcopy # Allows to make a copy of a dictionnary and change it without modifying the original, whatever it is

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogFormatter

from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl

from . import parameters as param


# FUNCTIONS FOR DATA PREPROCESSING :
####################################

def formatted_inputs(original_input_file="None", final_input_file='updated_input_file.csv',
                     original_time_step_in_days=1 / 24., final_time_step_in_days=1.,
                     simulation_period_in_days=60., do_not_execute_if_file_with_suitable_size_exists=False):
    """
    This function creates a new input file containing data on soil temperature and sucrose input rate (in mol of sucrose per second),
    based on an original file. The new file is adapted to the required time step (higher, equal or lower than the original time step).
    If the option 'do_not_execute_if_file_with_suitable_size_exists' is set to True, a new file will be created only if the already-
    existing 'input_file.csv' does not contain the correct number of lines.
    """

    # If there is a file where the inputs of sucrose in the root system have to be read:
    if original_input_file != "None":

        # We first define the path and the file to read as a .csv:
        PATH1 = os.path.join('.', original_input_file)
        # Then we read the file and copy it in a dataframe "df":
        df = pd.read_csv(PATH1, sep=',', header=0)

        # simulation_period_in_days = df['time_in_days'].max()-df['time_in_days'].min()
        n_steps = round(simulation_period_in_days / final_time_step_in_days)
        print("The total number of steps is", n_steps)

        if do_not_execute_if_file_with_suitable_size_exists:
            PATH2 = os.path.join('.', 'input_file.csv')
            previous_file = pd.read_csv(PATH2, sep=',', header=0)

            if len(previous_file['step_number']) == n_steps:
                print("There is already an 'input_file.csv' of proper length for this scenarios, "
                      "we therefore did not create a new input file here (if you wish to do so, please select 'do_not_execute_if_file_with_suitable_size_exists=False').")
                return previous_file

        # We create a new, final dataframe that will be used to read inputs of sucrose:
        input_frame = pd.DataFrame(
            columns=["step_number", "initial_time_in_days", "final_time_in_days", "soil_temperature_in_degree_Celsius",
                     "sucrose_input_rate"])
        initial_step_number = df["step_number"].iloc[0]
        input_frame["step_number"] = range(initial_step_number * round(original_time_step_in_days/final_time_step_in_days),
                                           initial_step_number * round(original_time_step_in_days/final_time_step_in_days) + n_steps)
        input_frame["initial_time_in_days"] = input_frame["step_number"] * final_time_step_in_days
        input_frame["final_time_in_days"] = input_frame["initial_time_in_days"] + final_time_step_in_days

        print("Creating a new input file adapted to the required time step (time step =",
              "{:.2f}".format(Decimal(final_time_step_in_days)), "days)...")

        # CASE 1: the final time step is higher than the original one, so we have to calculate an average value:
        # -------------------------------------------------------------------------------------------------------
        if final_time_step_in_days >= original_time_step_in_days:

            # We initialize the row ID "j" in which exact sucrose inputs will be recorded:
            j = 0
            # We initialize the time over which sucrose input will be integrated:
            cumulated_time_in_days = 0.
            time_in_days = 0.
            sucrose_input = 0.
            sum_of_temperature_in_degree_days = 0.

            # For each line of the data frame that contains information on sucrose input:
            for i in range(0, len(df['time_in_days'])):

                # If the current cumulated time is below the time step of the main scenarios:
                if (cumulated_time_in_days + original_time_step_in_days) < 0.9999 * final_time_step_in_days:
                    # Then the amount of time elapsed here is the initial time step:
                    net_elapsed_time_in_days = original_time_step_in_days
                    remaining_time = 0.
                else:
                    # Otherwise, we limit the elapsed time to the one necessary for reaching a round number of final_time_step_in_days:
                    net_elapsed_time_in_days = original_time_step_in_days - (
                            cumulated_time_in_days + original_time_step_in_days - final_time_step_in_days)
                    remaining_time = (cumulated_time_in_days + original_time_step_in_days - final_time_step_in_days)

                # In any case, the sucrose input and the sum of temperature are increased according to the elapsed time considered here:
                sucrose_input += df.loc[i, 'sucrose_input_rate'] * net_elapsed_time_in_days * 60 * 60 * 24.
                sum_of_temperature_in_degree_days += df.loc[i, 'soil_temperature_in_Celsius'] * net_elapsed_time_in_days

                # If at the end of the original time step corresponding to the line i in df, one final time step has been reached
                # or if this is the last line of the original table df to be read:
                if cumulated_time_in_days + original_time_step_in_days >= 0.9999 * final_time_step_in_days \
                        or i == len(df['time_in_days']) - 1:

                    # We move to the next row in the table to create:
                    j += 1
                    # print("   Creating line", j, "on", n_steps, "lines in the new input frame...")
                    # We record the final temperature and sucrose input rate:
                    input_frame.loc[j - 1, 'sucrose_input_rate'] = sucrose_input / (
                            (cumulated_time_in_days + net_elapsed_time_in_days) * 60. * 60. * 24.)
                    input_frame.loc[j - 1, 'soil_temperature_in_degree_Celsius'] = sum_of_temperature_in_degree_days \
                                                                                   / ((
                            cumulated_time_in_days + net_elapsed_time_in_days))

                    # We reinitialize the counter with the exact time that has not been included yet:
                    cumulated_time_in_days = remaining_time
                    # print "After reinitializing cumulated time, it is equal to", cumulated_time_in_days
                    # We reinitialize the sucrose input with the remaining sucrose that has not been taken into account:
                    sucrose_input = df.loc[i, 'sucrose_input_rate'] * (cumulated_time_in_days) * 60 * 60 * 24.
                    sum_of_temperature_in_degree_days = df.loc[i, 'soil_temperature_in_Celsius'] * (
                        cumulated_time_in_days)

                else:
                    cumulated_time_in_days += original_time_step_in_days

                # If the number of total lines of the final input frame has been reached:
                if j >= n_steps:
                    # Then we stop the loop here:
                    break

        # CASE 2: the final time step is lower than the original one, so we have to interpolate values between two points:
        # ----------------------------------------------------------------------------------------------------------------
        if final_time_step_in_days < original_time_step_in_days:

            # We initialize the row ID "i" in which the original values are read:
            i = 1
            # print "      Considering now line", i, "in the original table for time = ", df.loc[i, 'time_in_days']

            # We initialize the time over which sucrose input will be integrated:
            cumulated_time_in_days = 0.
            time_in_days = 0.
            sucrose_input = 0.
            sum_of_temperature_in_degree_days = 0.

            sucrose_input_rate_initial = df.loc[0, 'sucrose_input_rate']
            temperature_initial = df.loc[0, 'soil_temperature_in_Celsius']

            # For each line of the final table:
            for j in range(0, n_steps):

                # If the current cumulated time is below the time step of the main scenarios:
                if cumulated_time_in_days + final_time_step_in_days < 0.9999 * original_time_step_in_days:
                    # Then the amount of time elapsed here is the initial time step:
                    net_elapsed_time_in_days = final_time_step_in_days
                    remaining_time = 0.
                else:
                    # Otherwise, we limit the elapsed time to the one necessary for reaching a round number of original_time_step_in_days:
                    net_elapsed_time_in_days = final_time_step_in_days - (
                            cumulated_time_in_days + final_time_step_in_days - original_time_step_in_days)
                    remaining_time = (cumulated_time_in_days + final_time_step_in_days - original_time_step_in_days)

                # Then we calculate the input rate and the temperature for the current line:
                sucrose_input_rate = sucrose_input_rate_initial \
                                     + (df.loc[i, 'sucrose_input_rate'] - sucrose_input_rate_initial) \
                                     / original_time_step_in_days * cumulated_time_in_days
                temperature = temperature_initial \
                              + (df.loc[i, 'soil_temperature_in_Celsius'] - temperature_initial) \
                              / original_time_step_in_days * cumulated_time_in_days

                print("   Creating line", j + 1, "on", n_steps, "lines in the new input frame...")
                # We record the final temperature and sucrose input rate:
                input_frame.loc[j, 'sucrose_input_rate'] = sucrose_input_rate
                input_frame.loc[j, 'soil_temperature_in_degree_Celsius'] = temperature

                # If at the end of the final time step corresponding to the line j in input_frame, one original time step has been reached:
                if cumulated_time_in_days + final_time_step_in_days >= 0.9999 * original_time_step_in_days:

                    # We record the current sucrose input rate and temperature in the original table:
                    sucrose_input_rate_initial = df.loc[i, 'sucrose_input_rate']
                    temperature_initial = df.loc[i, 'soil_temperature_in_Celsius']

                    # We move to the next row in the original table, unless we are already at the last line:
                    if i < len(df['time_in_days']) - 1:
                        i += 1
                        # print "      Considering now line", i, "in the original table for time = ", df.loc[i, 'time_in_days']

                    # We reinitialize the counter with the exact time that has not been included yet:
                    cumulated_time_in_days = remaining_time

                else:

                    cumulated_time_in_days += final_time_step_in_days

                # If the number of total lines of the final input frame has been reached:
                if j >= n_steps:
                    # Then we stop the loop here:
                    break

    input_frame.to_csv(final_input_file, na_rep='NA', index=False, header=True)
    print("The new input file adapted to the required time step has been created and saved as 'updated_input_file.csv'.")
    print("")

    return input_frame


def is_int(dt):
    """
    Returns True if dt is an integer, False in other cases.
    """
    try:
        num = int(dt)
    except ValueError:
        return False
    return True


def _buildDic(keyList, val, dic):
    if len(keyList) == 1:
        dic[keyList[0]] = val
        return

    newDic = dic.get(keyList[0], {})
    dic[keyList[0]] = newDic
    _buildDic(keyList[1:], val, newDic)


def buildDic(dict_scenario, dic=None):
    """
    Function that build a nested dictionary (dict of dict), which is used in simulations/scenario_parameters/main_one_scenario.py
    e.g. buildDic({'a:b:c': 1, 'a:b:d': 2}) returns {'a': {'b': {'c': 1, 'd': 2}}}
    """
    if not dic:
        dic = {}

    for k, v in dict_scenario.items():
        if not pd.isnull(v):
            keyList = k.split(':')
            keyList_converted = []
            for kk in keyList:
                if is_int(kk):
                    keyList_converted.append(int(kk))
                else:
                    keyList_converted.append(kk)
            _buildDic(keyList_converted, v, dic)

    return dic


# FUNCTIONS TO DISPLAY THE RESULTS IN A GRAPH:
##############################################

# Defining functions for diplaying the root system in a 3D graph in PlantGL:
# --------------------------------------------------------------------------

def get_root_visitor():
    """
    This function describes the movement of the 'turtle' along the MTG for creating a graph on PlantGL.
    :return: root_visitor
    """

    def root_visitor(g, v, turtle):
        n = g.node(v)

        # For displaying the radius or length X times larger than in reality, we can define a zoom factor:
        zoom_factor = 1.
        # We look at the geometrical properties already defined within the root element:
        radius = n.radius * zoom_factor
        length = n.length * zoom_factor
        angle_down = n.angle_down
        angle_roll = n.angle_roll

        # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
        position1 = turtle.getPosition()
        n.x1 = position1[0] / zoom_factor
        n.y1 = position1[1] / zoom_factor
        n.z1 = position1[2] / zoom_factor

        # The direction of the turtle is changed:
        turtle.down(angle_down)
        turtle.rollL(angle_roll)

        # Tropism is then taken into account:
        # diameter = 2 * n.radius * zoom_factor
        # elong = n.length * zoom_factor
        # alpha = tropism_intensity * diameter * elong
        # turtle.rollToVert(alpha, tropism_direction)
        # if g.edge_type(v)=='+':
        # diameter = 2 * n.radius * zoom_factor
        # elong = n.length * zoom_factor
        # alpha = tropism_intensity * diameter * elong
        turtle.elasticity = param.gravitropism_coefficient * (n.original_radius / g.node(1).original_radius)
        turtle.tropism = (0, 0, -1)

        # The turtle is moved:
        turtle.setId(v)
        if n.type != "Root_nodule":
            # We define the radius of the cylinder to be displayed:
            turtle.setWidth(radius)
            # We move the turtle by the length of the root segment:
            turtle.F(length)
        else: # SPECIAL CASE FOR NODULES
            # We define the radius of the sphere to be displayed:
            turtle.setWidth(radius)
            # We "move" the turtle, but not according to the length (?):
            turtle.F()

        # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
        position2 = turtle.getPosition()
        n.x2 = position2[0] / zoom_factor
        n.y2 = position2[1] / zoom_factor
        n.z2 = position2[2] / zoom_factor

    return root_visitor


def my_colormap(g, property_name, cmap='jet', vmin=None, vmax=None, lognorm=True):
    """
    This function computes a property 'color' on a MTG based on a given MTG's property.
    :param g: the investigated MTG
    :param property_name: the name of the property of the MTG that will be displayed
    :param cmap: the type of color map
    :param vmin: the min value to be displayed
    :param vmax: the max value to be displayed
    :param lognorm: a Boolean describing whether the scale is logarithmic or not
    :return: the MTG with the corresponding color
    """

    # We make sure that the user did not accidently switch between vmin and vmax:
    if vmin >= vmax:
        raise Exception("Sorry, the vmin and vmax values of the color scale of the graph are wrong!")
    if lognorm and (vmin <=0 or vmax <=0):
        raise Exception("Sorry, it is not possible to represent negative values in a log scale - check vmin and vmax!")

    # We access the property of the MTG and calculated normalized values between 0 and 255 from it, for each element:
    prop = g.property(property_name)
    keys = prop.keys()
    values = list(prop.values())
    _cmap = color.get_cmap(cmap)
    norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
    values = norm(values)
    colors = (_cmap(values)[:, 0:3]) * 255
    colors = np.array(colors, dtype=np.int).tolist()

    # In case no color values could be calculated from the given information:
    if len(colors) == 0:
        print("WATCH OUT: we could not attribute colors to the values of the property", property_name,"; the colors were set to a default value.")
        prop = g.property("label")
        keys = prop.keys()
        colors = [[250,150,100]]*len(keys)
        colors = np.array(colors, dtype=np.int).tolist()

    # Finally, the property "color" is created/updated with the new computed values:
    g.properties()['color'] = dict(zip(keys, colors))

    # We also check whether dead roots should be displayed in a specific way:
    for vid in g.vertices_iter(scale=1):
        n=g.node(vid)
        if n.length <= 0.:
            continue
        if n.type=="Dead" or n.type=="Just_dead":
            n.color = [0,0,0]

    return g


def prepareScene(scene, width=1200, height=1200, scale=1, x_center=0., y_center=0., z_center=0.,
                 x_cam=0., y_cam=0., z_cam=-1.5, grid=True, background_color=[255,255,255]):
    """
    This function returns the scene that will be used in PlantGL to display the MTG.
    :param scene: the scene to start with
    :param width: the width of the graph (in pixels)
    :param height: the height of the graph (in pixels)
    :param scale: a dimensionless factor for zooming in or out
    :param x_center: the x-coordinate of the center of the graph
    :param y_center: the y-coordinate of the center of the graph
    :param z_center: the z-coordinate of the center of the graph
    :param x_cam: the x-coordinate of the camera looking at the center of the graph
    :param y_cam: the y-coordinate of the camera looking at the center of the graph
    :param z_cam: the z-coordinate of the camera looking at the center of the graph
    :param grid: a Boolean describing whether grids should be displayed on the graph
    :return: scene
    """

    # We define the coordinates of the point cam_target that will be the center of the graph:
    cam_target = pgl.Vector3(x_center * scale,
                             y_center * scale,
                             z_center * scale)
    # We define the coordinates of the point cam_pos that represents the position of the camera:
    cam_pos = pgl.Vector3(x_cam * scale,
                          y_cam * scale,
                          z_cam * scale)

    # We position the camera in the scene relatively to the center of the scene:
    pgl.Viewer.camera.lookAt(cam_pos, cam_target)

    # # We define the absolute values of the (x,y,z) coordinates of the camera:
    # pgl.Viewer.camera.position=cam_pos
    # pgl.Viewer.camera.angle=(0,180,180)
    # print(" >>> The camera position for this plot is:", pgl.Viewer.camera.position)
    # We define the dimensions of the graph:
    pgl.Viewer.frameGL.setSize(width, height)
    # We define whether grids are displayed or not:
    pgl.Viewer.grids.set(grid, grid, grid, grid)
    # We define the background color of the scene:
    pgl.Viewer.frameGL.setBgColor(background_color[0],background_color[1],background_color[2])

    return scene


def circle_coordinates(x_center=0., y_center=0., z_center=0., radius=1., n_points=50):
    """
    This function calculates the coordinates of n points evenly distributed on a circle of a specified radius
    within the x-y plane for a given height (z).
    :param x_center: the x-coordinate of the center of the circle
    :param y_center: the y-coordinate of the center of the circle
    :param z_center: the z-coordinate of the center of the circle
    :param radius: the radius of the circle
    :param n_points: the number of points distributed on the circle
    :return: three lists containing all x-coordinates, y-coordinates and z-coordinates, respectively
    """

    # We initialize empty lists of coordinates for each of the three dimensions:
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []

    # We initalize the angle at 0 rad:
    angle = 0
    # We calculate the increment theta that will be added to the angle for each new point:
    theta = 2 * pi / float(n_points)

    # For each point of the circle which coordinates should be calculated:
    for step in range(0, n_points):
        # We calculate the coordinates of the point corresponding to the new angle:
        x_coordinates.append(x_center + radius * cos(angle))
        y_coordinates.append(y_center + radius * sin(angle))
        z_coordinates.append(z_center)
        # And we increase the angle by theta:
        angle += theta

    return x_coordinates, y_coordinates, z_coordinates


def plot_mtg(g,
             # scene, scene_for_hairs, scene_for_fungus,
             prop_cmap='hexose_exudation', cmap='jet', lognorm=True, vmin=1e-12, vmax=3e-7,
             root_hairs_display=True,
             mycorrhizal_fungus_display=True,
             width=1200, height=1200,
             x_center=0., y_center=0., z_center=0.,
             x_cam=1., y_cam=0., z_cam=0.,
             # # For a black background:
             # background_color=[0,0,0]
             # For a "soil" background:
             background_color=[94,76,64],
             displaying_PlantGL_Viewer=True,
             grid_display=False,
             ):
    """
    This function creates a graph on PlantGL that displays a MTG and color it according to a specified property.
    :param g: the investigated MTG
    :param prop_cmap: the name of the property of the MTG that will be displayed in color
    :param cmap: the type of color map
    :param lognorm: a Boolean describing whether the scale is logarithmic or not
    :param vmin: the min value to be displayed
    :param vmax: the max value to be displayed
    :param x_center: the x-coordinate of the center of the graph
    :param y_center: the y-coordinate of the center of the graph
    :param z_center: the z-coordinate of the center of the graph
    :param x_cam: the x-coordinate of the camera looking at the center of the graph
    :param y_cam: the y-coordinate of the camera looking at the center of the graph
    :param z_cam: the z-coordinate of the camera looking at the center of the graph
    :return: the updated scene
    """

    # Consider: https://learnopengl.com/In-Practice/Text-Rendering

    # MOVING THE TURTLE AND PREPARING THE ROOTS' SCENE:
    #--------------------------------------------------
    visitor = get_root_visitor()
    # We initialize a turtle in PlantGL:
    turtle = turt.PglTurtle()

    # # We reset the turtle:
    # turtle.reset()

    MTG_starting_position = pgl.Vector3(x_center,y_center,z_center)
    angle_down = 180
    angle_roll = 0

    # And we define its starting position:
    turtle.move(MTG_starting_position)
    turtle.down(angle_down)
    turtle.rollL(angle_roll)

    # We initialize the scene with the MTG g:
    scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)
    new_scene = scene

    if displaying_PlantGL_Viewer:
        # # TODO: WATCH OUT the following
        # # We update the scene with the specified position of the center of the graph and the camera:
        # scene = prepareScene(scene, width=width, height=height,
        #                      x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
        #                      background_color=background_color)

        # We compute the colors of the graph:
        my_colormap(g, prop_cmap, cmap=cmap, vmin=vmin, vmax=vmax, lognorm=lognorm)
        # We get a list of all shapes in the scene:
        shapes = dict((sh.id, sh) for sh in scene)
        # We use the property 'color' of the MTG calculated by the function 'my_colormap':
        colors = g.property('color')

        # We cover each node of the MTG:
        for vid in colors:
            if vid in shapes:
                n = g.node(vid)
                # If the element is not dead:
                if n.type != "Dead":
                    # We color it according to the property cmap defined by the user:
                    shapes[vid].appearance = pgl.Material(colors[vid], transparency=0.0)
                else:
                    # Otherwise, we print it in black in a semi-transparent way:
                    shapes[vid].appearance = pgl.Material([0, 0, 0], transparency=0.8)
                # # SPECIAL CASE:
                # if mycorrhizal_fungus_display and n.fungal_infection_severity > 0.:
                #     # Otherwise, we print it in black in a semi-transparent way:
                #     shapes[vid].appearance = pgl.Material([255, 255, 255], transparency=0.0)

                # SPECIAL CASE: If the element is a nodule, we transform the cylinder into a sphere:
                if n.type == "Root_nodule":
                    # We create a sphere corresponding to the radius of the element:
                    s = pgl.Sphere(n.radius * 1.)
                    # We transform the cylinder into the sphere:
                    shapes[vid].geometry.geometry = pgl.Shape(s).geometry
                    # We select the parent element supporting the nodule:
                    index_parent = g.Father(vid, EdgeType='+')
                    parent = g.node(index_parent)
                    # We move the center of the sphere on the circle corresponding to the external envelop of the
                    # parent:
                    angle = parent.angle_roll
                    circle_x = parent.radius * cos(angle)
                    circle_y = parent.radius * sin(angle)
                    circle_z = 0
                    shapes[vid].geometry.translation += (circle_x, circle_y, circle_z)

        # DISPLAYING ROOT HAIRS:
        #-----------------------
        if root_hairs_display:
            visitor_for_hair = get_root_visitor()
            # We initialize a turtle in PlantGL:
            turtle_for_hair = turt.PglTurtle()
            # And we define its starting position:
            turtle_for_hair.move(MTG_starting_position)
            turtle_for_hair.down(angle_down)
            turtle_for_hair.rollL(angle_roll)

            # We initialize the scene with the MTG g:
            scene_for_hairs = turt.TurtleFrame(g, visitor=visitor, turtle=turtle_for_hair, gc=False)
            # # We update the scene with the specified position of the center of the graph and the camera:
            # prepareScene(scene_for_hairs, width=width, height=height,
            #              x_center=x_center, y_center=y_center, z_center=z_center, x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
            #              background_color=background_color)
            # We get a list of all shapes in the scene:
            shapes_for_hairs = dict((sh.id, sh) for sh in scene_for_hairs)

            # We cover each node of the MTG:
            for vid in colors:
                if vid in shapes_for_hairs:
                    n = g.node(vid)
                    # If the element has no detectable root hairs:
                    if n.root_hair_length<=0.:
                        # Then the element is set to be transparent:
                        shapes_for_hairs[vid].appearance = pgl.Material(colors[vid], transparency=1)
                    else:
                        # We color the root hairs according to the proportion of living and dead root hairs:
                        dead_transparency = 0.9
                        dead_color_vector=[0,0,0]
                        dead_color_vector_Red=dead_color_vector[0]
                        dead_color_vector_Green=dead_color_vector[1]
                        dead_color_vector_Blue=dead_color_vector[2]

                        living_transparency = 0.8
                        living_color_vector=colors[vid]
                        living_color_vector_Red = colors[vid][0]
                        living_color_vector_Green = colors[vid][1]
                        living_color_vector_Blue = colors[vid][2]

                        living_fraction = n.living_root_hairs_number/n.total_root_hairs_number

                        transparency = dead_transparency + (living_transparency - dead_transparency) * living_fraction
                        color_vector_Red = floor(dead_color_vector_Red
                                                 + (living_color_vector_Red - dead_color_vector_Red) * living_fraction)
                        color_vector_Green = floor(dead_color_vector_Green
                                                   + (living_color_vector_Green - dead_color_vector_Green) * living_fraction)
                        color_vector_Blue = floor(dead_color_vector_Blue
                                                  + (living_color_vector_Blue - dead_color_vector_Blue) * living_fraction)
                        color_vector = [color_vector_Red,color_vector_Green,color_vector_Blue]

                        shapes_for_hairs[vid].appearance = pgl.Material(color_vector, transparency=transparency)

                        # We finally transform the radius of the cylinder:
                        if vid > 1:
                            # For normal cases:
                            shapes_for_hairs[vid].geometry.geometry.geometry.radius = n.radius + n.root_hair_length
                        else:
                            # For the base of the root system [don't ask why this has not the same formalism..!]:
                            shapes_for_hairs[vid].geometry.geometry.radius = n.radius + n.root_hair_length

        # DISPLAYING MYCORRHIZAL FUNGI:
        #------------------------------
        if mycorrhizal_fungus_display:
            # We recreate a new list of shapes corresponding to the MTG:
            visitor_for_fungus = get_root_visitor()
            # We initialize a new turtle in PlantGL:
            turtle_for_fungus = turt.PglTurtle()
            # # We make the graph upside down:
            # turtle_for_fungus.down(180)
            # And we define its starting position:
            turtle_for_fungus.move(MTG_starting_position)
            turtle_for_fungus.down(angle_down)
            turtle_for_fungus.rollL(angle_roll)
            # We initialize the scene with the MTG g:
            scene_for_fungus = turt.TurtleFrame(g, visitor=visitor_for_fungus, turtle=turtle_for_fungus, gc=False)
            # We get a list of all shapes in the scene:
            shapes_for_fungus = dict((sh.id, sh) for sh in scene_for_fungus)

            for vid in colors:
                if vid in shapes_for_fungus:
                    n = g.node(vid)
                    try:
                        # If the current root element has been infected by the fungus:
                        if n.fungal_infection_severity > 0.:
                            # We set the reference color:
                            color_vector = [150, 50, 100]
                            # We set the transparency based on the level of infection:
                            transparency = 0.3 + (1 - n.fungal_infection_severity) * 0.7
                            shapes_for_fungus[vid].appearance = pgl.Material(color_vector, transparency=transparency)

                            # We finally transform the radius of the cylinder:
                            if vid > 1:
                                # For normal cases:
                                shapes_for_fungus[vid].geometry.geometry.geometry.radius = (n.radius + n.root_hair_length) * 1.5
                            else:
                                # For the base of the root system [don't ask why this has not the same formalism ...!]:
                                shapes_for_fungus[vid].geometry.geometry.radius = (n.radius + n.root_hair_length) * 1.5

                            # We also recolor the underlying root segment, by mixing the colors with pure white.
                            # We get the normal color of the root element:
                            # initial_red = shapes[vid].appearance.ambient.red
                            # initial_green = shapes[vid].appearance.ambient.green
                            # initial_blue = shapes[vid].appearance.ambient.blue
                            # # And we set the final color to have the mean value between the normal color and pure white:
                            # final_red = round((initial_red + 255)/2.)
                            # final_green = round((initial_green + 255) / 2)
                            # final_blue = round((initial_blue + 255) / 2)
                            # shapes[vid].appearance = pgl.Material([final_red, final_green, final_blue], transparency=0.0)

                    except:
                        continue

        # CREATING THE ACTUAL SCENE:
        #---------------------------
        # We add the shapes from the root hairs:
        if root_hairs_display:
            for vid in shapes_for_hairs:
                new_scene += shapes_for_hairs[vid]
        # And we eventually add the shapes of the mycorrhizal fungi, if any:
        if mycorrhizal_fungus_display:
            for vid in shapes_for_fungus:
                new_scene += shapes_for_fungus[vid]

        # TODO: WATCH OUT the following
        # We update the scene with the specified position of the center of the graph and the camera:
        new_scene = prepareScene(new_scene, width=width, height=height,
                                 x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
                                 grid=grid_display,
                                 background_color=background_color)

        # # For preventing PlantGL to update the display of the graph at each new time step:
        # pgl.Viewer.redrawPolicy=False

    return new_scene

# FUNCTIONS FOR FITTING DATA:
#############################

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy
# import scipy.stats
# import time
#
# # We get the data showing the distribution of observations.
# # Here, the example is the distribution of growth durations of lateral root elongation along a mother root:
# data_path="D:/Documents/PHOTOS/2022 Essais Pots et Rhizotrons - Projet MezAgri/Scan racines - Projet CarboStockTMM/GDs_data_brut.csv"
# data = pd.read_csv(data_path)
# # We create a proper histogram from this data file:
# bins=121
# y, x = np.histogram(data, bins=bins, density=True)
# # We define x as the middle of each class:
# x = (x + np.roll(x, -1))[:-1] / 2.0
#
# # We test different possibilities of known probability distributions:
# dist_names = ['norm', 'beta','gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2']
#
# # For each possible distribution suggeted in "dist_names":
# for name in dist_names:
#     print("")
#     print("Considering the method", name, "...")
#     try:
#         # We try to fit the corresponding distribution curve on the data:
#         dist = getattr(scipy.stats, name)
#         # We record the parameters from this fitted distribution:
#         param = dist.fit(data)
#         loc = param[-2]
#         scale = param[-1]
#         arg = param[:-2]
#         pdf = dist.pdf(x, *arg, loc=loc, scale=scale)
#
#         # We check whether the sum of square differences is low enough:
#         model_sse = np.sum((y - pdf) ** 2)
#         print("This method corresponded to a fit with square difference of", model_sse, "with following parameters:")
#         print("> Param:", param)
#         print("> Arg:", arg)
#         print("> Loc:", loc)
#         print("> Scale:", scale)
#
#         # We create a plot:
#         plt.figure(figsize=(12,8))
#         plt.plot(x,pdf,label=name, linewidth=3)
#         plt.plot(x,y,alpha=0.6)
#         plt.legend()
#         plt.show()
#     except:
#         print("Method failed!")

# # Checking if random data can reproduce the original distribution:
# for i in range(1, 50):
#     new_GD = np.random.standard_t(0.6829556179151338, size=10)
#     print(new_GD)

########################################################################################################################

# Define function for string formatting of scientific notation
def sci_notation(num, just_print_ten_power=True, decimal_digits=1, precision=None, exponent=None):
    """
    This function returns a string representation of the scientific notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified explicitly.
    """
    if exponent is None:
        if num != 0.:
            if num >= 1:
                exponent = int(ceil(log10(abs(num))))
            else:
                exponent = int(floor(log10(abs(num))))
        else:
            exponent = 0

    coeff = round(num / float(10 ** exponent), 1)

    if precision is None:
        precision = decimal_digits

    if num == 0:
        return r"${}$".format(0)

    if just_print_ten_power:
        return r"$10^{{{0:d}}}$".format(exponent)
    elif decimal_digits==0:
        # return r"${0:.{2}f}/cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
        return str(int(round(coeff))) + " " + r"$10^{{{0:d}}}$".format(exponent)
    else:
        return str(coeff) + " " + r"$10^{{{0:d}}}$".format(exponent)

# Function that draws a colorbar:
def colorbar(title="Radius (m)", cmap='jet', lognorm=True, vmin=1e-12, vmax=1e3, ticks=[]):
    """
    This function creates a colorbar for showing the legend of a plot. The scale can be linear or normal. If no "ticks"
    number are provided, the function will automatically add ticks and numbers on the graph. Note that numbers that are
    too close to the left or right border of the bar will not be displayed.
    :param title: the name of the property to be displayed on the bar
    :param cmap: the name of the specific colormap in Python
    :param lognorm: if True, the scale will be a log scale, otherwise, it will be a linear scale
    :param ticks: a list of float number to be displayed on the colorbar.
    :param vmin: the min value of the color scale
    :param vmax: the max value of the color scale
    :return: the new colorbar object
    """

    # CREATING THE COLORBAR WITH THICKS
    ####################################

    # Creating the box that will contain the colorbar:
    fig, ax = plt.subplots(figsize=(36, 6))
    fig.subplots_adjust(bottom=0.5)

    _cmap = color.get_cmap(cmap)

    # If the bar is to be displayed with log scale:
    if lognorm:
        # We check that the min value of the colorbar is positive:
        if vmin <=0.:
            print("WATCH OUT: when making the colorbar, vmin can't be equal or below zero when lognorm is True. "
                  "Therefore vmin has been turned to 1e-10 by default.")
            vmin=1e-10
            if len(ticks) > 0 and ticks[0] == 0:
                ticks[0] = vmin
        # We create the logscaled color values:
        norm = color.LogNorm(vmin=vmin, vmax=vmax)
        # We create the log-scale color bar:
        cbar = mpl.colorbar.ColorbarBase(ax,
                                         cmap=cmap,
                                         norm=norm,
                                         orientation='horizontal')

        # If a list of numbers where ticks are supposed to be displayed has not been provided:
        if ticks==[]:
            # Then we create our own set of major ticks:
            min10 = ceil(np.log10(vmin))
            max10 = floor(np.log10(vmax))
            # We calculate the interval to cover:
            n_intervals = int(abs(max10 - min10)) + 1
            # We start with the first number
            min_number = 10 ** min10
            ticks.append(min_number)
            # Then for subsequent numbers, we just add a new number that is 10 time higher than the previous number:
            for i in range(1,n_intervals):
                ticks.append(ticks[i-1]*10)
        # Now we can define the positions of each label above major ticks as:
        label_positions = [(log10(i)-log10(vmin))/(log10(vmax)-log10(vmin)) for i in ticks]
        # Eventually, we add the ticks to the colorbar:
        cbar.set_ticks(ticks)

    # Otherwise the colorbar is in linear scale:
    else:
        # If a list of numbers where ticks are supposed to be displayed has not been provided:
        if ticks == []:
            # We set the number of intervals between two ticks:
            n_intervals = 4
            # We calculate the x-difference between two consecutive ticks:
            delta = (vmax-vmin)/float(n_intervals)
            # We start with the first number
            min_number = vmin
            ticks.append(min_number)
            # Then for subsequent numbers, we just add a new number that is 10 time higher than the previous number:
            for i in range(1, n_intervals+1):
                ticks.append(ticks[i-1] + delta)
        # Now we can define the positions of each label above major ticks as:
        label_positions = [(i-vmin) / (vmax - vmin) for i in ticks]
        # We create the normal-scale color bar:
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax,
                                         cmap=cmap,
                                         norm=norm,
                                         ticks=ticks, # We specify a number of ticks to display
                                         orientation='horizontal')

    # In any case, we remove stupid automatic tick labels:
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    # We also specify the characteristics of the ticks and mines:
    cbar.outline.set_linewidth(3)  # Thickness of the box lines
    cbar.set_label(title, fontsize=40, weight='bold', labelpad=-130)  # Adjust the caption under the bar
    cbar.ax.tick_params(which="major",
                        direction="in",  # Position of the ticks in or out the bar
                        labelsize=0,  # Size of the text
                        length=20,  # Length of the ticks
                        width=5,  # Thickness of the ticks
                        pad=-60  # Distance between ticks and label
                        )
    cbar.ax.tick_params(which="minor",
                        direction="in",  # Position of the ticks in or out the bar
                        labelsize=0,  # Size of the text
                        length=10,  # Length of the ticks
                        width=3,  # Thickness of the ticks
                        pad=-60  # Distance between ticks and label
                        )

    # For adding minor ticks:
    ax.minorticks_on()
    # minorticks = [0.1, 0.2, 0.3]
    # ax.xaxis.set_ticks(minorticks, minor=True)
    # ax.yaxis.set_ticks(minorticks, minor=True)

    # CREATING SCIENTIFIC NUMBERS TO DISPLAY:
    # We initialize empty lists:
    numbers_to_display = []
    # If we use a logs-cale:
    if lognorm:
        # We initialize a Boolean specifying whether coefficient of scientific notations should be dispayed or not:
        just_print_ten_power = True
        # We cover each number to display:
        for number in ticks:
            # If one number does not correspond to one power of tenth:
            if floor(np.log10(number)) != np.log10(number):
                # Then we will print the normal scientific notation for all numbers:
                just_print_ten_power = False
                break
        for number in ticks:
            coeff = number / 10**(floor(log10(number)))
            if coeff > floor(coeff)*(1+1e-3):
                number_of_digits = 1
                break
            else:
                number_of_digits = 0

        # We create the list of strings in a scientific format for displaying the numbers on the colorbar:
        for number in ticks:
            numbers_to_display.append(sci_notation(number, just_print_ten_power=just_print_ten_power,
                                                   decimal_digits=number_of_digits))
    # If we use a linear scale:
    else:
        # We will print numbers with coefficient and tenth-power:
        just_print_ten_power = False
        # We calculate the number of digits to display:
        for number in ticks:
            if number == 0:
                coeff = 0
            else:
                coeff = round(number / 10**(floor(log10(number))),2)

            if coeff > floor(coeff) * (1 + 1e-3) and coeff != 1:
                number_of_digits = 1
                break
            else:
                number_of_digits = 0

        # We create the list of strings in a scientific format for displaying the numbers on the colorbar:
        for number in ticks:
            numbers_to_display.append(sci_notation(number, just_print_ten_power=just_print_ten_power,
                                                   decimal_digits=number_of_digits))

    # ADJUSTMENT OF LABEL POSITIONS AND CORRECTIONS FOR BORDERS:
    # We define limits between 0 and 1 within which the labels are authorized:
    xmin=0.00
    xmax=0.90
    # We also defined the translation of labels' position so that labels are correctly centered above the ticks:
    if just_print_ten_power:
        x_translation = -0.012
    else:
        x_translation = - 0.018
    # We now modify the label positions, and possibly mask some labels when too close to the borders:
    for i in range(0,len(ticks)):
        label_positions[i] += x_translation
        # If the first and last label position are too close to the limit of the colorbar, we don't display the numbers:
        if label_positions[i] < xmin or label_positions[i] > xmax:
            numbers_to_display[i] = ""

    # ADDING THE LABELS:
    # Finally, we cover each number to add its label on the colorbar:
    for i in range(0, len(numbers_to_display)):
        position = 'left'
        # We add the corresponding number on the colorbar:
        cbar.ax.text(x=label_positions[i],
                     y=0.4,
                     s=numbers_to_display[i],
                     va='top',
                     ha=position,
                     fontsize=40,
                     # fontweight='bold', # This doesn't change much the output, unfortunately...
                     transform=ax.transAxes)

    print("The colorbar has been made!")
    return fig

########################################################################################################################

# Function for defining each element according to its axis ID:
#-------------------------------------------------------------
def indexing_root_MTG(g):
    """
    This function assigns a new property called 'axis_ID' to each root element, corresponding to a chain of character
    related to its position in the topology of the root system. Each axis is defined by the segment from which it emerges,
    and by a number, which is usually 1, or a higher number in case several axis emerges from the same segment).
    Example 1: '[...]-Se00004-Ax00002' describes the second axis emerging from the fourth segment on the mother axis.
    The final code for a given root element can be read from right to left, indicating the element position as 'Se' + i,
    where i stands for the position of the segment along the axis (starting at 00001 at the base of the axis),
    or as 'Ap0000' in case it corresponds to the terminal axis of the axis.
    Example 2: for the the third segment of the only lateral axis emerging from the segment N°2 of the main seminal axis,
    the axis_ID will be written 'Ax00001-Se00002-Ax00001-Se00003'.
    Example 3: for the apex of the third seminal axis, the axis_ID will be written 'Ax00001-Se00001-Ax00002-Ap00000',
    as currently the third seminal axis is the second axis emerging from the first segment of the main axis.
    :param g: the root MTG to process
    :return: [none]
    """

    #-------------------------------------------------------------------------------------------------------------------

    # We create an internal function that computes the axis ID of each element on a given axis:
    def indexing_segments(starting_vid=1, axis_string="Ax00001"):
        """
        This internal function starts from a specific root element on a specific axis, then covers all subsequent
        children of the root element on this axis and generates for each of them a unique axis-element identifier.
        :param starting_vid:  the vertex ID to start with
        :param axis_string: the string code for the axis
        :return: two lists, containing the vertices IDs from all first axis element of lateral roots emerging from the
        current axis, and the axis-element identifier of each of them, respectively.
        """

        # We initialize temporary variables:
        segment_number = 1
        subsegment_number = 1
        vid = starting_vid
        list_of_lateral_vid = []
        list_of_lateral_axis_strings = []
        # We will also use a special notation for all displayed number in the string, always with 5 digits:
        number_string = '%.5d'

        # We start with the first element and assign to it the proper axis ID:
        n = g.node(vid)
        # If the current element is the terminal apex of the root axis:
        if n.label == "Apex":
            # We set the segment_number to 0, which will stop the loop:
            segment_number = 0
            # We then define the correct axis ID for this apex:
            n.axis_ID = axis_string + "-Ap" + number_string % segment_number
        else:
            n.axis_ID = axis_string + "-Se" + number_string % segment_number

        # We check whether there is one lateral root emerging from the current root element:
        if len(g.Sons(vid, EdgeType="+")) > 0:
            # If so, we add this to the list of all emerging lateral elements from the current root axis:
            list_of_lateral_vid.extend(g.Sons(vid, EdgeType="+"))
            # SPECIAL CASE: if the current element is a support element at the base of the root system:
            # if n.type == "Support_for_seminal_root" or n.type == "Support_for_adventitious_root":
            #     # We give a special name to the lateral axis, made with the current number of the segment AND a specific subnumber:
            #     lateral_axis_string = n.axis_ID + "-" + str(subsegment_number)
            #     # We also increment the subnumber for the next lateral root on the same segment:
            #     subsegment_number += 1
            # else:
                # Otherwise, we give a "classical" name for the lateral axis:

            lateral_axis_string = n.axis_ID
            # We add this axis name to the list of all lateral axes' names for the current root axis:
            list_of_lateral_axis_strings.append(lateral_axis_string)

        # For a given axis, we keep looping until the segment number is set to 0:
        while segment_number > 0:

            # We move to the next element of the axis:
            vid = g.Successor(vid)
            # And we define the current element as this element:
            n = g.node(vid)

            # DEFINING THE AXIS ID OF THE CURRENT ELEMENT:
            # If the current element is the terminal apex of the root axis:
            if n.label == "Apex":
                # We set the segment_number to 0, which will stop the loop:
                segment_number = 0
                # We then define the correct axis ID for this apex:
                n.axis_ID = axis_string + "-Ap" + number_string % segment_number
            # Otherwise, the root element is a segment:
            elif n.length > 0.:
                # We increase the segment number by 1:
                segment_number += 1
                # We now assign the correct axis ID to the current element:
                n.axis_ID = axis_string + "-Se" + number_string % segment_number
            elif n.type == "Support_for_seminal_root" or n.type == "Support_for_adventitious_root":
                # We keep the same segment number and assign the same axis ID as before to the current element:
                n.axis_ID = axis_string + "-Se" + number_string % segment_number

            # DEFINING THE AXIS ID NAME OF LATERAL EMERGING SEGMENTS:
            # We check whether there is one lateral root emerging from the current root element:
            if len(g.Sons(vid, EdgeType="+")) > 0:
                # If so, we add its vid to the list of all emerging lateral elements from the current root axis:
                list_of_lateral_vid.extend(g.Sons(vid, EdgeType="+"))
                # SPECIAL CASE: if the current element is a support element at the base of the root system:
                if n.type == "Support_for_seminal_root" or n.type == "Support_for_adventitious_root":
                    # We give a special name to the lateral axis, made with the current number of the segment AND a specific subnumber:
                    lateral_axis_string = n.axis_ID + "-Ax" + number_string % subsegment_number
                    # We also increment the subnumber for the next lateral root on the same segment:
                    subsegment_number += 1
                else:
                    # Otherwise, we give a "classical" name for the lateral axis:
                    lateral_axis_string = n.axis_ID + "-Ax00001"
                # We add this axis name to the list of all lateral axes' names for the current root axis:
                list_of_lateral_axis_strings.append(lateral_axis_string)

        # Finally, all the elements directly belonging to the axis have received an axis ID, and their lateral emerging
        # elements have been listed in a first list containing the vertex ID, and a second list corresponding to the
        # name of each corresponding lateral axis. We return the two lists.

        return list_of_lateral_vid, list_of_lateral_axis_strings

    #-------------------------------------------------------------------------------------------------------------------

    # # We define "root" as the starting point of the loop below:
    # root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    # root = next(root_gen)

    # We first call the function indexing_segments on the main initial axis of the root system:
    list_of_lateral_vid, list_of_lateral_axis_strings = indexing_segments(starting_vid=1, axis_string="Ax00001")
    # We now have a list of all lateral elements emerging from this first axis, and a second list corresponding to the
    # name of each corresponding lateral axis.

    # We repeat the same operation over successive "root orders", until there is no new lateral axis to consider:
    while len(list_of_lateral_vid) > 0.:
        # We reinitialize two empty lists:
        new_list_of_starting_vid = []
        new_list_of_lateral_axis_strings = []
        # We cover each lateral axis defined by their starting element in the current list:
        for i in range(0,len(list_of_lateral_vid)):
            new_vids, new_strings \
                = indexing_segments(starting_vid=list_of_lateral_vid[i], axis_string=list_of_lateral_axis_strings[i])
            new_list_of_starting_vid.extend(new_vids)
            new_list_of_lateral_axis_strings.extend(new_strings)
        # At this point, all the lateral axes originating from the "list_of_starting_vis" have been processed.
        # We can now move to an upper root order, as we provide a new list of starting vids to consider:
        list_of_lateral_vid = new_list_of_starting_vid
        list_of_lateral_axis_strings = new_list_of_lateral_axis_strings

    # print("The property 'axis_ID' has been computed on the whole root MTG!")

    return

