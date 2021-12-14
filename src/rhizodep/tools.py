
from math import pi, cos, sin
import numpy as np

from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
import openalea.plantgl.all as pgl
import rhizodep.parameters as param

########################################################################################################################
########################################################################################################################
# COMMON GENERAL FUNCTIONS POSSIBLY USED IN EACH MODULE
########################################################################################################################
########################################################################################################################

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
        # For displaying the radius or length 10 times larger than in reality:
        zoom_factor = 10.
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
        if n.type == "Root_nodule":
            # s=turt.Sphere(radius)
            # turtle.draw(s)
            turtle.setWidth(radius)
            index_parent = g.Father(v, EdgeType='+')
            parent = g.node(index_parent)
            turtle.F()
        else:
            turtle.setWidth(radius)
            turtle.F(length)

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

    prop = g.property(property_name)
    keys = prop.keys()
    values = list(prop.values())
    # m, M = int(values.min()), int(values.max())

    _cmap = color.get_cmap(cmap)
    norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
    values = norm(values)

    colors = (_cmap(values)[:, 0:3]) * 255
    colors = np.array(colors, dtype=np.int).tolist()

    g.properties()['color'] = dict(zip(keys, colors))
    return g


def prepareScene(scene, width=1200, height=1200, scale=0.8, x_center=0., y_center=0., z_center=0.,
                 x_cam=0., y_cam=0., z_cam=-1.5, grid=False):
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
    # We define the dimensions of the graph:
    pgl.Viewer.frameGL.setSize(width, height)
    # We define whether grids are displayed or not:
    pgl.Viewer.grids.set(grid, grid, grid, grid)

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


def plot_mtg(g, prop_cmap='hexose_exudation', cmap='jet', lognorm=True, vmin=1e-12, vmax=3e-7,
             x_center=0., y_center=0., z_center=0.,
             x_cam=1., y_cam=0., z_cam=0.):
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

    visitor = get_root_visitor()
    # We initialize a turtle in PlantGL:
    turtle = turt.PglTurtle()
    # We make the graph upside down:
    turtle.down(180)
    # We initialize the scene with the MTG g:
    scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)
    # We update the scene with the specified position of the center of the graph and the camera:
    prepareScene(scene, x_center=x_center, y_center=y_center, z_center=z_center, x_cam=x_cam, y_cam=y_cam, z_cam=z_cam)
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
                shapes[vid].appearance = pgl.Material(colors[vid])
            else:
                # Otherwise, we print it in black:
                shapes[vid].appearance = pgl.Material([0, 0, 0])

    # Changing some shapes geometry according to the element:
    for vid in shapes:
        n = g.node(vid)
        # If the element is a nodule, we transform the cylinder into a sphere:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n.type == "Root_nodule":
            # We create a sphere corresponding to the radius of the element:
            s = pgl.Sphere(n.radius * 1.)
            # We transform the cylinder into the sphere:
            shapes[vid].geometry.geometry = pgl.Shape(s).geometry
            # We select the parent element supporting the nodule:
            index_parent = g.Father(vid, EdgeType='+')
            parent = g.node(index_parent)
            # We move the center of the sphere on the circle corresponding to the external envelop of the parent:
            angle = parent.angle_roll
            circle_x = parent.radius * 10 * cos(angle)
            circle_y = parent.radius * 10 * sin(angle)
            circle_z = 0
            shapes[vid].geometry.translation += (circle_x, circle_y, circle_z)

    # We return the new updated scene:
    new_scene = pgl.Scene()
    for vid in shapes:
        new_scene += shapes[vid]

    # Consider: https://learnopengl.com/In-Practice/Text-Rendering

    return new_scene

