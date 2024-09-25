# import pkg_resources, os, time
#
# for package in pkg_resources.working_set:
#     print("%s: %s" % (package, time.ctime(os.path.getctime(package.location))))

from openalea.plantgl.all import *
import time

# # PLAYING WITH PLANT GL
# radius=0.1
# length=0.1
# new_scene = Scene()
# s1 = Cone(radius=radius,height=length)
# s2 = Cone(radius=radius)
# m = Material("red", Color3(150,0,0))
# Shape1 = Shape(s1,m)
# Shape2 = Shape(s2,m)
# new_scene += Shape2
# Viewer.display(new_scene)
# time.sleep(10)

########################################################################################################################
# PLAYING WITH PYVISTA:

import pyvista as pv
import numpy as np

########################################################################################################################
# DEFINING A FUNCTION FOR PLOTTING NON-CONNECTED SEGMENTS:

def plotting_segments(list_of_segments, recording_image=True, image_title="plot.png"):
    """
    This function will display/record the plots of non-connected tubes acccording to:
    i) the (x,y,z) coordinates of the starting and ending point of each segment,
    ii) the radius of each segment,
    iii) the color and opacity of each segment.
    It is expected that the list of segments contains segment objects where x1, x2, y1, y2, z1, z2, radius and color
    properties can be called.
    :param list_of_segments: the list containing the segment objects to plot
    :param recording_image: if True, the plot will be saved
    :param image_title: the name of the image file when saving the plot
    """

    # CREATING CORRECT LISTS OF PROPERTIES FROM THE LIST OF SEGMENT OBJECTS:
    # We initialize empty lists:
    x = []
    y = []
    z = []
    radius = []
    colors = []

    # We cover each element in the list provided for creating lists of specific properties that have the correct size:
    for n in list_of_segments:
        # We add the radius two times for each node, as the number of cells will be twice the number of segments
        # when using line_segments_from_points:
        radius.append(n.radius), radius.append(n.radius)
        # To each coordinate list, we add the corresponding coordinate from the starting point of the segment,
        # and then the coordinate from its ending point, for the future creation of line_segments_from_points:
        x.append(n.x1), x.append(n.x2)
        y.append(n.y1), y.append(n.y2)
        z.append(n.z1), z.append(n.z2)
        # We add the color only once for each tube created from segments:
        color = pv.Color(n.color)
        colors.append(color.int_rgba)

    # CREATING THE SET OF TUBES WITH CORRECT RADIUS:
    # We now create the points dataset used for created non-connected segments:
    points = np.column_stack((x, y, z))
    # We create a geometry of non-connected segments:
    lines = pv.line_segments_from_points(points)
    # We register for each cell the value of the corresponding radius:
    lines["radius"] = np.array(radius)
    # And we create tubes from the segments, scaling the radius from each tube on the prescribed radius values:
    mesh = lines.tube(scalars="radius", absolute=True)
    mesh = mesh.clean()  # Not sure whether clean() is really needed...
    # => "mesh" now contains the set of tubes with the correct prescribed coordinates and radius, but does not have color information.

    # ADDING COLOR INFORMATION:
    # We create another list of segments, that will be only used for creating a list of colors & opacities of the right size:
    points_for_colors = np.column_stack((x, y, z))
    lines_for_opacities = pv.line_segments_from_points(points_for_colors)
    # We register the RGBA colors we want in this second list:
    lines_for_opacities["color_rgba"] = np.array(colors)
    # We create again tubes from the segments and assign the RGBA colors in the scalar values:
    mesh_with_colors = lines_for_opacities.tube(scalars="color_rgba") #WATCH OUT: this is the line that generate unexpected errors...
    mesh_with_colors = mesh_with_colors.clean()  # Not sure whether clean() is really needed...
    # And now we can finally register the correct list of colors & opacities with the proper dimension to the original mesh of tubes:
    mesh["color_rgba"] = mesh_with_colors.cell_data.active_scalars

    # CREATING THE FINAL PLOT:
    # We initialize a plot:
    p = pv.Plotter()
    # Finally, we add the final mesh of tubes with proper radius, to which we add the right colors.
    p.add_mesh(mesh, scalars="color_rgba", rgba=True)

    # We display and/or record the plot:
    if recording_image:
        p.off_screen = True
        p.show(screenshot=image_title)
    else:
        p.show()
    # time.sleep(0.1)
    pv.close_all()

########################################################################################################################

# DEFINING ROOT SEGMENTS' PROPERTIES:
# We define a class for creating root elements with specific properties:
class root_element(object):
    # We initiate properties of the root element:
    def __init__(self):
        self.x1 = 0 + np.random.random_sample()*10 # x-coordinate of the starting point
        self.x2 = 2 + np.random.random_sample()*10 # x-coordinate of the ending point
        self.y1 = 0 + np.random.random_sample()*10 # y-coordinate of the starting point
        self.y2 = 2 + np.random.random_sample()*10 # y-coordinate of the ending point
        self.z1 = 0 + np.random.random_sample()*10 # z-coordinate of the starting point
        self.z2 = 2 + np.random.random_sample()*10 # z-coordinate of the ending point
        self.radius = np.random.random_sample()*0.1 # radius of the segment
        self.color = [np.random.randint(0,255),
                      np.random.randint(0,255),
                      np.random.randint(0,255),
                      np.random.random_sample()] # Color in RGBA format - the last value is opacity between 0 and 1

# We create a series of "root systems":
list_of_root_systems = []
for root_system_number in range(0,11):
    # We initialize a root system:
    list_of_root_elements = []
    # We create a series of different root elements within the current root system:
    for segment_number in range(0,1001):
        n = root_element()
        list_of_root_elements.append(n)
    list_of_root_systems.append(list_of_root_elements)
# Now we have 10 root systems, each containing 1000 root elements, each of them defined by their (x1, y1, z1)
# and (x2,y2,z2) coordinates, their radius and their color.

# PLOTTING ALL THE ROOT SYSTEMS IN A SEQUENCE:
for root_system_ID in range(0,1):
# for root_system_ID in range(0, len(list_of_root_systems)):
    print("Plotting the root system", root_system_ID, "...")
    plot_name = 'plot_%.5d.png' % root_system_ID
    root_system = list_of_root_systems[root_system_ID]
    plotting_segments(root_system, recording_image=True, image_title=plot_name)

# For printing details about versions:
print(pv.Report())