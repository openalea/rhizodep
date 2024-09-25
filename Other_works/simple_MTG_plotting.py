#  -*- coding: utf-8 -*-

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.traversal import pre_order, post_order
from openalea.mtg.plantframe import color
import openalea.plantgl.all as pgl

import numpy as np
import pickle
import time

########################################################################################################################

# Function for moving the turtle and create cylinders:
def get_root_visitor():
    """
    This function describes the movement of the 'turtle' along the MTG for creating a root graph on PlantGL.
    :return: root_visitor
    """

    def root_visitor(g, v, turtle):
        n = g.node(v)

        # We look at the geometrical properties already defined within the root element:
        radius = n.radius
        length = n.length
        angle_down = n.angle_down
        angle_roll = n.angle_roll

        # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
        position1 = turtle.getPosition()
        n.x1 = position1[0]
        n.y1 = position1[1]
        n.z1 = position1[2]

        # The direction of the turtle is changed:
        turtle.down(angle_down)
        turtle.rollL(angle_roll)

        # The turtle is moved:
        turtle.setId(v)
        # We define the radius of the cylinder to be displayed:
        turtle.setWidth(radius)
        # We move the turtle by the length of the root segment:
        turtle.F(length)

        # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
        position2 = turtle.getPosition()
        n.x2 = position2[0]
        n.y2 = position2[1]
        n.z2 = position2[2]

    return root_visitor

# Function for creating a color map for a MTG:
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

    prop = g.property(property_name)
    keys = prop.keys()
    values = list(prop.values())
    # m, M = int(values.min()), int(values.max())

    _cmap = color.get_cmap(cmap)
    norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
    values = norm(values)

    colors = (_cmap(values)[:, 0:3]) * 255
    colors = np.array(colors, dtype=int).tolist()

    g.properties()['color'] = dict(zip(keys, colors))

    return g

# Function for setting the center anc camera position in a scene:
def setting_PGLViewer(width=1200, height=1200,
                      x_center=0., y_center=0., z_center=0.,
                      x_cam=0., y_cam=0., z_cam=-0.2,
                      grid=True,
                      background_color = [255,255,255]):
    """
    This function sets the center of the graph and the relative position of the camera for displaying a scene in PGL.
    :param width: width of the window (pixels)
    :param height: height of the window (pixels)
    :param x_center: x-coordinate of the center at which the camera looks
    :param y_center: y-coordinate of the center at which the camera looks
    :param z_center: z-coordinate of the center at which the camera looks
    :param x_cam: x-coordinate of the camera position
    :param y_cam: y-coordinate of the camera position
    :param z_cam: z-coordinate of the camera position
    :param grid: if True, all grids will be displayed
    :param background_color: a RGB vector corresponding to the background color of the graph
    :return:
    """

    # We define the coordinates of the point cam_target that will be the center of the graph:
    cam_target = pgl.Vector3(x_center, y_center, z_center)
    # We define the coordinates of the point cam_pos that represents the position of the camera:
    cam_pos = pgl.Vector3(x_cam, y_cam, z_cam)
    # We position the camera in the scene relatively to the center of the scene:
    pgl.Viewer.camera.lookAt(cam_pos, cam_target)
    # We define the dimensions of the graph:
    pgl.Viewer.frameGL.setSize(width, height)
    # We define the background of the scene:
    pgl.Viewer.frameGL.setBgColor(background_color[0], background_color[1], background_color[2])
    # We define whether grids are displayed or not:
    pgl.Viewer.grids.set(grid, grid, grid, grid)

    return

# Main function for plotting multiple MTG in a single graph:
def plotting_multiple_MTGs(list_of_MTGs = [MTG()],
                           list_of_starting_coordinates=[(0,0,0)],
                           list_of_angles_from_vertical=[180],
                           list_of_rolling_angles=[0],
                           width=1200, height=900,
                           x_center=0, y_center=-0, z_center=0.,
                           x_cam=5., y_cam=5., z_cam=-1,
                           grid=True,
                           background_color=[94, 76, 64],
                           single_color=None,
                           property_name='radius', cmap='jet', vmin=1e-4, vmax=1e-2, lognorm=True,
                           recording_plot=True,
                           plot_name='plot.png'):

    """
    This function generates a new PGL scene displaying a bunch of MTG, and can be saved as an image. Instructions about
    MTG and their positions are provided as lists, one item in each list corresponding to 1 MTG.
    :param list_of_MTGs: list of all MTG to be included in the scene
    :param list_of_starting_coordinates: list of the (x,y,z) vectors corresponding to the starting positions of the MTGs
    :param list_of_angles_from_vertical: list of vertical angles for the initial position of the turtle
    :param list_of_rolling_angles: list of rolling angles for the initial position of the turtle
    :param width: width of the window (pixels)
    :param height: height of the window (pixels)
    :param x_center: x-coordinate of the center at which the camera looks
    :param y_center: y-coordinate of the center at which the camera looks
    :param z_center: z-coordinate of the center at which the camera looks
    :param x_cam: x-coordinate of the camera position
    :param y_cam: y-coordinate of the camera position
    :param z_cam: z-coordinate of the camera position
    :param grid: if True, all grids will be displayed
    :param background_color: a RGB vector corresponding to the background color of the graph
    :param single_color: a RGB vector used to apply the same color everywhere on the MTG (None by default)
    :param property_name: if single_color is None, the name of the MTG property on which the color map is based
    :param cmap: type of the color map
    :param vmin: the min value to be displayed
    :param vmax: the max value to be displayed
    :param lognorm: a Boolean describing whether the scale is logarithmic or not
    :param recording_plot: if True, the image of the graph will be saved
    :param plot_name: the name of the image (with extension) on which the graph has been saved
    :return: the new scene with all the shapes from each MTG
    """

    # We initialize a turtle in PlantGL:
    turtle = turt.PglTurtle()

    # We create a list of indices to match each MTG in the list of MTGs:
    indices = []
    i=-1
    for g in list_of_MTGs:
        i +=1
        indices.append(i)

    # We initialize a scene:
    new_scene = pgl.Scene()

    # We cover the items in the lists:
    for i in indices:

        # MOVING THE TURTLE TO A NEW POSITION:
        # We get the initial position and angles of the turtle:
        new_position = list_of_starting_coordinates[i]
        angle_down = list_of_angles_from_vertical[i]
        angle_roll = list_of_rolling_angles[i]

        # We reset the turtle:
        turtle.reset()
        # And we reposition the turtle from there:
        turtle.move(new_position)
        turtle.down(angle_down)
        turtle.rollL(angle_roll)
        # We get the new MTG to consider:
        g = list_of_MTGs[i]

        # SETTING THE NEW SCENE WITH THE TURTLE:
        # We create the scene by moving the turtle along the current MTG:
        scene = turt.TurtleFrame(g, visitor=get_root_visitor(), turtle=turtle, gc=False)

        # SETTING THE COLOR OF THE SHAPES:
        # We compute the colors of the graph:
        my_colormap(g, property_name=property_name, cmap=cmap, vmin=vmin, vmax=vmax, lognorm=lognorm)
        # We use the property 'color' of the MTG calculated by the function 'my_colormap':
        colors = g.property('color')
        # We get a dictionnary of all shapes in the scene:
        shapes = dict((sh.id, sh) for sh in scene)
        # We cover each node of the MTG:
        for vid in colors:
            if vid in shapes:
                n = g.node(vid)
                if single_color is None:
                    # We color it according to the property cmap defined by the user:
                    shapes[vid].appearance = pgl.Material(colors[vid], transparency=0.0)
                else:
                    # Otherwise, we can print it with a unique color:
                    shapes[vid].appearance = pgl.Material(single_color, transparency=0.0)

        # We add each shape of the current MTG to the final scene:
        for shape in scene:
            new_scene += shape

    # SETTING THE VIEWER OPTIONS:
    setting_PGLViewer(width=width, height=height,
                      x_center=x_center, y_center=y_center, z_center=z_center,
                      x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
                      grid=grid,
                      background_color=background_color)

    # DISPLAYING:
    pgl.Viewer.display(new_scene)
    # RECORDING:
    if recording_plot:
        pgl.Viewer.saveSnapshot(plot_name)

    return new_scene

########################################################################################################################

# MAIN SCRIPT:
#-------------

# # EXAMPLE 1: Combinging two simplified root MTG
# ################################################
# # Initializing a first MTG g1, corresponding to 1 vertical cylinder:
# g1 = MTG()
# vid = g1.add_component(g1.root, label='Segment')
# n = g1.node(vid)
# n.length = 1
# n.radius = 0.2
# n.angle_down = -5
# n.angle_roll = -5
#
# # Initializing a second MTG g2, corresponding to 1 oblique cylinder:
# g2 = MTG()
# vid = g2.add_component(g2.root, label='Segment')
# n = g2.node(vid)
# n.length = 0.5
# n.radius = 0.1
# n.angle_down = 45
# n.angle_roll = 45

# # EXAMPLE 2: Combinging 4 root MTG
# ##################################

# Loading an actual root MTG that was previously recorded:
# filename = 'C:/Users/frees/rhizodep/simulations/saving_outputs/outputs_2023-06/Scenario_0017/MTG_files/root02016.pckl'
filename = 'D:/Documents/EVENEMENTS/2021-08 EUROSOIL/Data/g_file_2021-08-24-S68.pckl'
f = open(filename, 'rb')
g = pickle.load(f)
f.close()

# We define the list of the different MTG to be incorporated:
list_of_MTGs=[g,g,g,g]
# We define the list of corresponding vectors (x,y,z) of starting positions, as well as the list of initial angle of the turtle:
list_of_starting_coordinates=[(0,-0.4,0),(0,-0.2,0),(0,0,0),(0,0.2,0)]
list_of_angles_from_vertical=[180, 178, 185, 182]
list_of_rolling_angles=[0, 90, 180, 270]

# We finally plot all the MTG together in the desired way:
plotting_multiple_MTGs(list_of_MTGs = list_of_MTGs,
                       list_of_starting_coordinates=list_of_starting_coordinates,
                       list_of_angles_from_vertical=list_of_angles_from_vertical,
                       list_of_rolling_angles=list_of_rolling_angles,
                       width=1200, height=900,
                       x_center=0., y_center=0., z_center=0.,
                       x_cam=5., y_cam=5., z_cam=-1,
                       grid=True,
                       background_color=[94, 76, 64],
                       single_color=[200, 100, 100],
                       # property_name='C_hexose_root', cmap='jet', vmin=1e-4, vmax=1e-2, lognorm=True,
                       recording_plot=True,
                       plot_name='plot.png')

# new_scene = pgl.Scene()
# radius=0.05
# length=0.05
# shape1 = pgl.Cone(radius=radius,height=length)
# new_scene += shape1
# pgl.Viewer.display(new_scene)
# # pgl.Viewer.show()
# print("Here is the camera position:", pgl.Viewer.camera.position)
# pgl.Viewer.camera.position = (0.2, 0, -0.2)
# pgl.Viewer.display(new_scene)
# # pgl.Viewer.show()
# print("Here is the camera position:", pgl.Viewer.camera.position)