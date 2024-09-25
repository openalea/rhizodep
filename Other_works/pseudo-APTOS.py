import openalea.mtg
from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.traversal import pre_order, post_order
from openalea.mtg.plantframe import color
import openalea.plantgl.all as pgl

import numpy as np
import pandas as pd
import os

########################################################################################################################

# DEFINING GENERAL FUNCTIONS:
#############################

#-----------------------------------------------------------------------------------------------------------------------
# This function enables to generate an initial MTG with leaves and stems:
def create_shoot_MTG(number_of_leaves=3):

    # We initialize an empty MTG:
    g = MTG()

    # For each new leaf, we first create an internode, and then a connected leaf:
    for i in range(0, number_of_leaves):

        # We first create a new internode.
        # If it corresponds to the first element of the MTG, we create it directly from the 'root' of the MTG:
        if i == 0:
            id_internode = g.add_component(g.root,
                                           label='Internode',
                                           radius=0.1,
                                           length=0.2,
                                           angle_down = 0.,
                                           angle_roll = 0.)
            internode = g.node(id_internode)
        # If an internode has already been created before, we simply add a new internode from it:
        else:
            internode = internode.add_child(edge_type='<',
                                            label='Internode',
                                            radius=0.1,
                                            length=0.1,
                                            angle_down=0.,
                                            angle_roll=138.5)

        # Then we create a "branch" on this internode, which will correspond to a leaf:
        leaf = internode.add_child(edge_type='+',
                                   label='Leaf',
                                   radius=0.1,
                                   length=0.5,
                                   angle_down=60,
                                   angle_roll=0)
    return g
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# This function enables to plot a MTG with PlantGL:
def plot_shoot_MTG(g,
                   width=1200, height=900,
                   x_center=0, y_center=-0, z_center=0.,
                   x_cam=5., y_cam=5., z_cam=-1,
                   grid=True,
                   background_color=[94, 76, 64],
                   single_color=None,
                   property_name='radius', cmap='jet', vmin=1e-4, vmax=1e-2, lognorm=True,
                   recording_plot=True,
                   plot_name='plot.png'):

    # Sub-function for moving the turtle and create specific shapes for stems and leaves:
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

            # # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
            # position1 = turtle.getPosition()
            # n.x1 = position1[0]
            # n.y1 = position1[1]
            # n.z1 = position1[2]

            # The direction of the turtle is changed:
            turtle.down(angle_down)
            turtle.rollL(angle_roll)

            # The turtle is moved:
            turtle.setId(v)

            if n.label == 'Internode':
                # We define the radius of the cylinder to be displayed:
                turtle.setWidth(radius)
                # We move the turtle by the length of the root segment:
                turtle.F(length)
            elif n.label == 'Leaf':
                # We define the radius of the cylinder to be displayed:
                turtle.setWidth(radius)
                # We move the turtle by the length of the root segment:
                turtle.F(length)


            # # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
            # position2 = turtle.getPosition()
            # n.x2 = position2[0]
            # n.y2 = position2[1]
            # n.z2 = position2[2]

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
        if lognorm and (vmin <= 0 or vmax <= 0):
            raise Exception(
                "Sorry, it is not possible to represent negative values in a log scale - check vmin and vmax!")

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
                          background_color=[255, 255, 255]):
        """
        This function sets the center of the graph and the relative position of the camera for displaying a scene in
        PGL.
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

    # CREATING THE PLOT:
    #-------------------
    # We initialize a turtle in PlantGL:
    turtle = turt.PglTurtle()

    # We initialize a scene:
    new_scene = pgl.Scene()

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
                # If the element has not fallen:
                if n.fallen == False:
                    # Then we color it according to the property cmap defined by the user, with full opacity:
                    shapes[vid].appearance = pgl.Material(colors[vid], transparency=0.0)
                else:
                    # Otherwise, we color it in a very transparent way:
                    shapes[vid].appearance = pgl.Material(colors[vid], transparency=0.9)
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

    # DISPLAYING THE MTG:
    pgl.Viewer.display(new_scene)

    # RECORDING THE MTG:
    if recording_plot:
        pgl.Viewer.saveSnapshot(plot_name)

    return
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# This function enables to record the properties of each element of the MTG:
def recording_MTG_properties(g, file_name='g_properties.csv'):
    """
    This function records the properties of each node of the MTG "g" inside a csv file.
    :param g: the MTG where properties are recorded
    :param file_name: the name of the csv file where properties of each node will be recorded
    :return: [no return]
    """

    # We define the list of all properties of the MTG:
    initial_list_of_properties = list(g.properties().keys())
    # We verify that the list does not contain any "None" property:
    list_of_properties = []
    for property in initial_list_of_properties:
        if property is None:
            print("WARNING: a None property was detected in the MTG!")
            continue
        else:
            list_of_properties.append(property)
    # We sort the list by alphabetical order
    list_of_properties.sort()

    # We create an empty list of node indices:
    node_index = []
    # We create an empty list that will contain the properties of each node:
    g_properties = []

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # Initializing an empty list of properties for the current node:
        node_properties = []
        # Adding the index at the beginning of the list:
        node_properties.append(vid)
        # n represents the vertex:
        n = g.node(vid)
        # For each possible property:
        for property in list_of_properties:
            # We add the value of this property to the list:
            node_properties.append(getattr(n, property, "NA"))
        # Finally, we add the new node's properties list as a new item in g_properties:
        g_properties.append(node_properties)
    # We create a list containing the headers of the dataframe:
    column_names = ['node_index']
    column_names.extend(list_of_properties)
    # We create the final dataframe:
    data_frame = pd.DataFrame(g_properties, columns=column_names)
    # We record the dataframe as a csv file:
    data_frame.to_csv(file_name, na_rep='NA', index=False, header=True)

    return

def create_MTG_from_csv_file(csv_filename='MTG_00003.csv'):

    # We first read the csv file where all the properties of each vertex has been previously recorded:
    dataframe = pd.read_csv(csv_filename, sep=',', header=0)
    # We initialize an empty MTG:
    g = MTG()

    # We define the list of vertices and properties to be added to the MTG:
    list_of_vid = dataframe['node_index']
    list_of_properties = list(dataframe.columns.values)
    list_of_properties.remove('node_index')

    # We cover each new vertex to be added to the MTG:
    for vid in list_of_vid:
        # We get the previous element on which to add the new element:
        n = g.node(vid-1)
        # We now define the new element as the child of the previous element:
        n = n.add_child()
        # For this new element, we cover all the properties defined in the csv file:
        for property in list_of_properties:
            # For the specific property, we get the single value corresponding to the current vertex ID:
            selected_series = dataframe[property].loc[dataframe['node_index'] == vid].iloc[0]
            # NB: we need to add 'iloc[0]' as the previous expression otherwise returns a series, not a single value.
    return g
#-----------------------------------------------------------------------------------------------------------------------

# DEFINING METABOLIC FUNCTIONS:
###############################

# This function considers whether a leaf should start senescing or not:
def starting_senescence(g):
    print("Checking whether leaves are in senescence state or not...")
    return

# This function calculates the redistribution of N among leaves:
def N_exchange_within_shoots(g):
    print("Exchanging N between leaves...")
    return

# This function considers whether a leaf should fall or not:
def leaves_falling(g, N_treshold = 0.):
    print("Checking whether leaves should fall or not...")

    # We cover each element of the MTG (including internodes!):
    for vid in g.vertices_iter(scale=1):
        # We access the element with the current vid:
        n = g.node(vid)
        # If this corresponds to a leaf and if the N content is lower than the treshold:
        if n.label == "Leaf" and n.N_content < N_treshold:
            # Then the leaf has fallen!
            n.fallen = True

    return

########################################################################################################################
########################################################################################################################

# MAIN SIMULATION:
##################

# We read the environemental/initial conditions:
input_dataframe = pd.read_excel('APTOS_input_1.xlsx', header=0, sheet_name="scenario")

# # OPTION 1: We create a MTG from its known properties (NOT COMPLETELY FINISHED!):
# g = create_MTG_from_csv_file()

# OPTION 2: We create an initial MTG corresponding to the shoot (stem and leaves):
g = create_shoot_MTG()

# We initialize the content of N in each leaf/internode:
for vid in g.vertices_iter(scale=1):
    n = g.node(vid)
    # n.N_content = input_dataframe['Initial_N_content'].loc[vid-1]
    n.N_content = float(input_dataframe['Initial_N_content'].loc[input_dataframe['element_ID'] == vid])

# We initialize the PAR in each leaf/internode:
for vid in g.vertices_iter(scale=1):
    n = g.node(vid)
    n.PAR = float(input_dataframe['PAR'].loc[input_dataframe['element_ID'] == vid])

# We initialize a new property of the MTG g that corresponds to the status "Fallen / Not fallen":
for vid in g.vertices_iter(scale=1):
    n = g.node(vid)
    n.fallen = False

# We display the MTG in its initial state:
empty_scene = pgl.Scene()
pgl.Viewer.display(empty_scene)
plot_shoot_MTG(g, width=1200, height=900,
               x_center=0, y_center=0, z_center=0.,
               x_cam=5., y_cam=5., z_cam=1.,
               grid=False,
               background_color=[0, 0, 0],
               single_color=None,
               property_name='N_content', cmap='jet', vmin=0, vmax=0.2, lognorm=False,
               recording_plot=True,
               plot_name='plot_0000.png')

# We record the properties in its initial state:
recording_MTG_properties(g, 'MTG_0000.csv')

# We define an initial and final step for covering a certain period:
initial_step = 1
final_step = 3

print("Simulation has started...")

# For each time step covering the scenarios period:
for step in range(initial_step, final_step+1):

    print("")
    print("For step", step,":")

    # We successively call the different modules:
    starting_senescence(g)
    N_exchange_within_shoots(g)
    leaves_falling(g, N_treshold=0.05)

    # We display the MTG in its new state:
    plot_shoot_MTG(g, width=1200, height=900,
                   x_center=0, y_center=0, z_center=0.,
                   x_cam=5., y_cam=5., z_cam=1.,
                   grid=False,
                   background_color=[0, 0, 0],
                   # single_color=[100,150,50],
                   property_name='N_content', cmap='jet', vmin=0, vmax=0.2, lognorm=False,
                   recording_plot=True,
                   plot_name = 'plot_%.5d.png' % step)

    # We record the main properties of the plant:
    properties_filename = 'MTG_%.5d.csv' % step
    recording_MTG_properties(g, properties_filename)

print("")
print("Simulation is done!")