from math import sqrt, pi, trunc, floor, cos, sin
from decimal import Decimal
import time
import numpy as np
import pandas as pd
from statistics import mean
import os, os.path
from path import Path
import timeit

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl
from openalea.plantgl.all import *
from PIL import Image, ImageDraw, ImageFont
from rhizodep.tools import my_colormap, get_root_visitor, prepareScene, circle_coordinates, plot_mtg

import pickle

########################################################################################################################
# DEFINING ADDTIONAL FUNCTIONS FOR DISPLAYING THE MTG IN A 3D GRAPH WITH PLANTGL
########################################################################################################################

# Function creating an image containing a text:
#----------------------------------------------
def drawing_text(text="TEXT !", image_name="text.png", length=220, height=110, font_size=100):
    """
    This function enables to create and save an image containing showing a text on a transparent background.
    :param text: the text to add
    :param image_name: the name of the image file
    :param length: length of the image
    :param height: height of the image
    :param font_size: the size of the text
    :return:
    """
    # We create a new image:
    im = Image.new("RGB", (length, height), (255, 255, 255, 0))
    # For making it transparent:
    im.putalpha(0)
    # We draw on this image:
    draw = ImageDraw.Draw(im)
    # Relative coordinates of the text:
    (x1, y1) = (0, 0)
    # Defining font type and font size:
    font_time = ImageFont.truetype("./timesbd.ttf", font_size)  # See a list of available fonts on:
    # https:/docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
    # We draw the text on the created image:
    # draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 50), fill=(255, 255, 255, 200))
    draw.text((x1, y1), text, fill=(0, 0, 0), font=font_time)
    # We save the image as a png file:
    im.save(image_name, 'PNG')
    return

# Function positionning a certain image:
#---------------------------------------
def showing_image(image_name="text.png",
                  x1=0, y1=0, z1=0,
                  x2=0, y2=0, z2=1,
                  x3=0, y3=1, z3=1,
                  x4=0, y4=1, z4=0):
    """
    This function returns the shape corresponding to the image positionned at a certain position.
    :param image_name: the name of the image file
    :param x,y,z for 1-4: x,y,z coordinates of each 4 points of the rectangle containig the image
    :return: the corresponding shape
    """
    # We define the list of points that will correspond to the coordinates of the image to display:
    # points =  [(0,0,0),
    #            (0,0,1),
    #            (0,1,1),
    #            (0,1,0)]
    points = [(x1, y1, z1),
              (x2, y2, z2),
              (x3, y3, z3),
              (x4, y4, z4), ]
    # We define a list of indices:
    indices = [(0, 1, 2, 3)]
    # We define a zone that will correspond to these coordinates:
    carre = QuadSet(points, indices)
    # We load an image as a texture material:
    my_path = os.path.join("../../simulations/running_scenarios/", image_name)
    tex = ImageTexture(my_path)
    # We define the texture coordinates that we will use:
    # texCoord = [(0,0),
    #             (0,1),
    #             (1,1),
    #             (1,0)]
    texCoord = [(y1, z1),
                (y2, z2),
                (y3, z3),
                (y4, z4)]
    # And how texture coordinates are associated to vertices:
    texCoordIndices = [(0, 1, 2, 3)]
    # We finally display the new image:
    carre.texCoordList = texCoord
    carre.texCoordIndexList = texCoordIndices
    shape = Shape(carre, tex)
    # Viewer.display(shape)
    return shape

# Function positionning some margins:
#------------------------------------
def add_margin(image, top, right, bottom, left, color):
    """
    This function adds some margins around a specific image.
    :param image: the image object to which margins are added
    :param top: width of the top margin
    :param right: width of the right margin
    :param bottom: width of the bottom margin
    :param left: width of the left margin
    :param color: color of the margins
    :return:
    """
    width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height), color)
    result.paste(image, (left, top))
    return result

########################################################################################################################
########################################################################################################################

# Useful functions for calculating on MTG files:
################################################

# Calculation of the length of a root element intercepted between two z coordinates:
# ----------------------------------------------------------------------------------
def sub_length_z(x1, y1, z1, x2, y2, z2, z_first_layer, z_second_layer):
    """
    This function returns the length of a segment postionned between (x1,y1,z1) and (x2,y2,z2) that is located between
    two horizontal planes at z=z_first_layer and z=z_second_layer.
    :return: the computed length between the two planes
    """
    # We make sure that the z coordinates are ordered in the right way:
    min_z = min(z1, z2)
    max_z = max(z1, z2)
    z_start = min(z_first_layer, z_second_layer)
    z_end = max(z_first_layer, z_second_layer)

    # For each row, if at least a part of the root segment formed between point 1 and 2 is included between z_start and z_end:
    if min_z < z_end and max_z >= z_start:
        # The z value to start from is defined as:
        z_low = max(z_start, min_z)
        # The z value to stop at is defined as:
        z_high = min(z_end, max_z)

        # If Z2 is different from Z1:
        if max_z > min_z:
            # Then we calculate the (x,y) coordinates of both low and high points between which length will be computed:
            x_low = (x2 - x1) * (z_low - z1) / (z2 - z1) + x1
            y_low = (y2 - y1) * (z_low - z1) / (z2 - z1) + y1
            x_high = (x2 - x1) * (z_high - z1) / (z2 - z1) + x1
            y_high = (y2 - y1) * (z_high - z1) / (z2 - z1) + y1
            # Geometrical explanation:
            # *************************
            # # The root segment between Start-point (X1, Y1, Z1) and End-Point (X2, Y2, Z2)
            # draws a line in the 3D space which is characterized by the following parametric equation:
            # { x = (X2-X1)*t + X1, y = (Y2-Y1)*t + Y1, z = (Z2-Z1)*t + Z1}
            # To find the coordinates x and y of a new point of coordinate z on this line, we have to solve this system of equations,
            # knowing that: z = (Z2-Z1)*t + Z1, which gives t = (z - Z1)/(Z2-Z1), and therefore:
            # x = (X2-X1)*(z - Z1)/(Z2-Z1) + X1
            # y = (Y2-Y1)*(z - Z1)/(Z2-Z1) + Y1
        # Otherwise, the calculation is much easier, since the whole segment is included in the plan x-y:
        else:
            x_low = x1
            y_low = y1
            x_high = x2
            y_high = y2

        # In every case, the length between the low and high points is computed as:
        inter_length = ((x_high - x_low) ** 2 + (y_high - y_low) ** 2 + (z_high - z_low) ** 2) ** 0.5
    # Otherwise, the root element is not included between z_first_layer and z_second_layer, and intercepted length is 0:
    else:
        inter_length = 0

    # We return the computed length:
    return inter_length

# Integration of root variables within different z_intervals:
# -----------------------------------------------------------
def classifying_on_z(g, z_min=0, z_max=1, z_interval=0.1):
    """
    This function calculates the distribution of certain characteristics of a MTG g according to the depth z.
    For each z-layer between z_min and z_max and each root segment, specific variables are computed, depending on the
    length within the segment that is intercepted between the upper and lower horizontal plane [use of 'sub_length' functino].
    :param g: the MTG on which calculations are made
    :param z_min: the depth to which we start computing
    :param z_max: the maximal depth to which we stop computing
    :param z_interval: the thickness of each layer to consider between z_min and z_max
    :return: a dictionnary containing the results
    """
    # We initialize empty dictionnaries:
    included_length = {}
    dictionnary_length = {}
    dictionnary_struct_mass = {}
    dictionnary_root_necromass = {}
    dictionnary_surface = {}
    dictionnary_net_hexose_exudation = {}
    dictionnary_hexose_degradation = {}
    final_dictionnary = {}

    # For each interval of z values to be considered:
    for z_start in np.arange(z_min, z_max, z_interval):

        # We create the names of the new properties of the MTG to be computed, based on the current z interval:
        name_length_z = "length_" + str(z_start) + "-" + str(z_start + z_interval) + "_m"
        name_struct_mass_z = "struct_mass_" + str(z_start) + "-" + str(z_start + z_interval) + "_m"
        name_root_necromass_z = "root_necromass_" + str(z_start) + "-" + str(z_start + z_interval) + "_m"
        name_surface_z = "surface_" + str(z_start) + "-" + str(z_start + z_interval) + "_m"
        name_net_hexose_exudation_z = "net_hexose_exudation_" + str(z_start) + "-" + str(z_start + z_interval) + "_m"
        name_hexose_degradation_z = "hexose_degradation_" + str(z_start) + "-" + str(z_start + z_interval) + "_m"

        # We (re)initialize total values:
        total_included_length = 0
        total_included_struct_mass = 0
        total_included_root_necromass = 0
        total_included_surface = 0
        total_included_net_hexose_exudation = 0
        total_included_hexose_degradation = 0

        # We cover all the vertices in the MTG:
        for vid in g.vertices_iter(scale=1):
            # n represents the vertex:
            n = g.node(vid)

            # We make sure that the vertex has a positive length:
            if n.length > 0.:
                # We calculate the fraction of the length of this vertex that is included in the current range of z value:
                fraction_length = sub_length_z(x1=n.x1, y1=n.y1, z1=-n.z1, x2=n.x2, y2=n.y2, z2=-n.z2,
                                               z_first_layer=z_start,
                                               z_second_layer=z_start + z_interval) / n.length
                included_length[vid] = fraction_length * n.length
            else:
                # Otherwise, the fraction length and the length included in the range are set to 0:
                fraction_length = 0.
                included_length[vid] = 0.

            # We summed different variables based on the fraction of the length included in the z interval:
            total_included_length += n.length * fraction_length
            total_included_struct_mass += n.struct_mass * fraction_length
            if n.type == "Dead" or n.type == "Just_dead":
                total_included_root_necromass += n.struct_mass * fraction_length
            total_included_surface += n.external_surface * fraction_length
            total_included_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake) * fraction_length
            total_included_hexose_degradation += n.hexose_degradation * fraction_length

        # We record the summed values for this interval of z in several dictionnaries:
        dictionnary_length[name_length_z] = total_included_length
        dictionnary_struct_mass[name_struct_mass_z] = total_included_struct_mass
        dictionnary_root_necromass[name_root_necromass_z] = total_included_root_necromass
        dictionnary_surface[name_surface_z] = total_included_surface
        dictionnary_net_hexose_exudation[name_net_hexose_exudation_z] = total_included_net_hexose_exudation
        dictionnary_hexose_degradation[name_hexose_degradation_z] = total_included_hexose_degradation

        # We also create a new property of the MTG that corresponds to the fraction of length of each node in the z interval:
        g.properties()[name_length_z] = included_length

    # Finally, we merge all dictionnaries into a single one that will be returned by the function:
    final_dictionnary = dict(list(dictionnary_length.items())
                             + list(dictionnary_struct_mass.items())
                             + list(dictionnary_root_necromass.items())
                             + list(dictionnary_surface.items())
                             + list(dictionnary_net_hexose_exudation.items())
                             + list(dictionnary_hexose_degradation.items())
                             )

    return final_dictionnary


# Integration of root variables within different z_intervals:
# -----------------------------------------------------------
def recording_MTG_properties(g, file_name='g_properties.csv'):
    """
    This function records all the properties of each node of the MTG "g" in a specific csv file.
    """

    # We define and reorder the list of all properties of the MTG:
    list_of_properties = list(g.properties().keys())
    list_of_properties.sort()

    # We create an empty list of node indices:
    node_index = []
    # We create an empty list that will contain the properties of each node:
    g_properties = []

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # Inititalizing an empty list of properties for the current node:
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

########################################################################################################################

########################################################################################################################
# MAIN FUNCTION FOR LOADING AND DISPLAYING/EXTRACTING PROPERTIES FROM MTG FILES
########################################################################################################################

def loading_MTG_files(my_path='',
                      opening_list=False,
                      MTG_directory='MTG_files',
                      single_MTG_filename='root00001.pckl',
                      list_of_MTG_ID=None,
                      property="C_hexose_root", vmin=1e-5, vmax=1e-2, log_scale=True, cmap='jet',
                      width=1200, height=1200,
                      x_center=0, y_center=0, z_center=0,
                      x_cam=0, y_cam=0, z_cam=0,
                      step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                      background_color=[0,0,0],
                      root_hairs_display=True,
                      mycorrhizal_fungus_display=False,
                      adding_images_on_plot=False,
                      recording_images=False,
                      images_directory='root_new_images',
                      z_classification=False, z_min=0.0, z_max=1., z_interval=0.1, time_step_in_days=1 / 24.,
                      printing_sum=True,
                      recording_sum=True,
                      printing_warnings=True,
                      recording_g_properties=True):
    """
    This function opens one MTG file or a list of MTG files, displays them and record some of their properties if needed.
    :param my_path: the general file path, in which the directory 'MTG_directory' will be located
    :param opening_list: if True, the function opens all (or some) MTG files located in the 'MTG_directory'
    :param MTG_directory: the name of the directory when MTG files are located
    :param single_MTG_filename: the name of the single MTG to open (if opening_list=False)
    :param list_of_MTG_ID: a list containing the ID number of each MTG to be opened (each MTG name is assumed to be in the format 'rootXXXXX.pckl')
    :param property: the property of the MTG to be displayed
    :param vmin, vmax, log_scale, cmap, width, height, x_center, y_center, z_center, z_cam, camera_distance,
    step_back_coefficient, camera_rotation, n_rotation_points: [cf parameters of the function plot_MTG]
    :param adding_images_on_plot: for adding some additional features on the plots' images
    :param recording_images: if True, images opened in PlantGL will be recorded
    :param images_directory: the name of the directory where images will be recorded
    :param z_classification: if True, specific properties will be integrated or averaged for different z-layers
    :param z_min: the depth to which we start computing
    :param z_max: the maximal depth to which we stop computing
    :param z_interval: the thickness of each layer to consider between z_min and z_max
    :param time_step_in_days: the time step at which is MTG file was generated (used to display time on the images)
    :param printing_sum: if True, properties summed over the whole MTG(s) will be shown in the console
    :param recording_sum: if True, properties summed over the whole MTG(s) will be recorded
    :param printing_warnings: if True, warnings will be displayed
    :param recording_g_properties: if True, all the properties of each MTG's node will be recorded in a file
    :return: The MTG file "g" that was loaded at last.
    """

    # Preparing the folders:
    # -----------------------

    if opening_list:
        # We define the directory "MTG_files":
        g_dir = os.path.join(my_path, MTG_directory)
        print("Loading MTG files located in", g_dir, "...")
    else:
        g_dir = my_path
        print("Loading the MTG file located in", g_dir,"...")

    if recording_images:
        # We define the directory "video"
        video_dir = os.path.join(my_path, images_directory)
        # If this directory doesn't exist:
        if not os.path.exists(video_dir):
            # Then we create it:
            os.mkdir(video_dir)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g_properties:
        # We define the directory "MTG_properties"
        prop_dir = os.path.join(my_path, 'MTG_properties')
        # If this directory doesn't exist:
        if not os.path.exists(prop_dir):
            # Then we create it:
            os.mkdir(prop_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(prop_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    if z_classification:
        # We create an empty dataframe that will contain the results of z classification:
        z_dictionnary_series = []

    # We get the list of the names of all MTG files:
    filenames = Path(g_dir).glob('*pckl')
    filenames = sorted(filenames)
    # We initialize a list containing the numbers of the MTG to be opened:
    list_of_MTG_numbers = []

    # If the instructions are to open the whole list of MTGs in the directory and not a subset of it:
    if opening_list and not list_of_MTG_ID:
        # We cover each name of MTG ending as 'rootXXXX.pckl' (where X is a digit), extract the corresponding number,
        # and add it to the list:
        for filename in filenames:
            MTG_ID = int(filename[-10:-5])
            list_of_MTG_numbers.append(MTG_ID)
    # If the instructions are to open a specific list:
    elif opening_list and list_of_MTG_ID:
        list_of_MTG_numbers =  list_of_MTG_ID
    # Otherwise, we open a single file:
    else:
        list_of_MTG_numbers = [int(single_MTG_filename[-10:-5])]

    # # Preparing the PGL viewer:
    # #--------------------------
    # # Defining the ID and full name of the MTG file:
    # ID = list_of_MTG_numbers[-1]
    # filename = 'root%.5d.pckl' % ID
    #
    # # Loading the MTG file:
    # MTG_path = os.path.join(g_dir, filename)
    # f = open(MTG_path, 'rb')
    # g = pickle.load(f)
    # f.close()
    #
    # MTG_scene = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
    #                      width=width,
    #                      height=height,
    #                      x_center=x_center,
    #                      y_center=y_center,
    #                      z_center=z_center,
    #                      x_cam=x_cam,
    #                      y_cam=y_cam,
    #                      z_cam=z_cam,
    #                      background_color=background_color,
    #                      root_hairs_display=root_hairs_display,
    #                      mycorrhizal_fungus_display=mycorrhizal_fungus_display)
    # MTG_scene = prepareScene(MTG_scene, width=width, height=height,
    #                          x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
    #                          background_color=background_color)
    # pgl.Viewer.display(MTG_scene)

    # # TODO: WATCH OUT!
    # pgl.Viewer.redrawPolicy=False

    # We cover each of the MTG files in the list (or only the specified file when requested):
    # ---------------------------------------------------------------------------------------
    # for step in range(step_initial, step_final):
    for MTG_position in range(0,len(list_of_MTG_numbers)):
        # Defining the ID and full name of the MTG file:
        ID = list_of_MTG_numbers[MTG_position]
        filename = 'root%.5d.pckl' % ID

        print("Dealing with MTG", ID, "-", MTG_position+1,"out of", len(list_of_MTG_numbers), "MTGs to consider...")

        # Loading the MTG file:
        MTG_path = os.path.join(g_dir,filename)
        f = open(MTG_path, 'rb')
        g = pickle.load(f)
        f.close()

        # Plotting the MTG:
        # ------------------

        # If the rotation of the camera around the root system is required:
        if camera_rotation:
            # We calculate the coordinates of the camera on the circle around the center:
            x_coordinates, y_coordinates, z_coordinates = circle_coordinates(z_center=z_cam,
                                                                             radius=camera_distance,
                                                                             n_points=n_rotation_points)
            # We initialize the index for reading each coordinates:
            index_camera = 0
            x_cam = x_coordinates[index_camera]
            y_cam = y_coordinates[index_camera]
            z_cam = z_coordinates[index_camera]
            # We plot the current file:
            sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                          width=width,
                          height=height,
                          x_center=x_center,
                          y_center=y_center,
                          z_center=z_center,
                          x_cam=x_cam,
                          y_cam=y_cam,
                          z_cam=z_cam)
        else:
            # We plot the current file:
            MTG_scene = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                                 width=width,
                                 height=height,
                                 x_center=x_center,
                                 y_center=y_center,
                                 z_center=z_center,
                                 x_cam=x_cam,
                                 y_cam=y_cam,
                                 z_cam=z_cam,
                                 background_color=background_color,
                                 root_hairs_display=root_hairs_display,
                                 mycorrhizal_fungus_display=mycorrhizal_fungus_display)
            # # We get a list of all shapes in the scene:
            # shapes = dict((MTG_scene.id, sh) for sh in MTG_scene)
            # # We add the shapes of the MTG in the initial general scene:
            # for vid in shapes:
            #     general_scene += shapes[vid]

            # general_scene += MTG_scene

            # for shape in MTG_scene:
            #     general_scene += shape

            # # And we move the camera further from the root system:
            # x_camera = x_cam + x_cam * step_back_coefficient * MTG_position
            # z_camera = z_cam + z_cam * step_back_coefficient * MTG_position

            # general_scene = prepareScene(MTG_scene, width=width, height=height,
            #                              x_center=x_center, y_center=y_center, z_center=z_center,
            #                              x_cam=x_cam, y_cam=y_cam, z_cam=z_cam, background_color=[0, 0, 0])

        if adding_images_on_plot:
            text = "t = 100 days"
            # length_text=len(text)*50
            # height_text = 120
            length_text = 600
            height_text = 600
            font_size = 100
            drawing_text(text=text, length=length_text, height=height_text, font_size=font_size, image_name="text.png")
            # Adding text to the plot:
            lower_left_x = 3
            lower_left_y = -2
            lower_left_z = -2
            length = 1
            height = 1
            shape1 = showing_image(image_name="text.png",
                                   x1=lower_left_x, y1=lower_left_y, z1=lower_left_z,
                                   x2=lower_left_x, y2=lower_left_y, z2=lower_left_z + height,
                                   x3=lower_left_x, y3=lower_left_y + length, z3=lower_left_z + height,
                                   x4=lower_left_x, y4=lower_left_y + length, z4=lower_left_z
                                   )
            # Viewer.display(shape)
            sc += shape1

            # Adding colorbar to the plot:
            # Adding text to the plot:
            lower_left_x = 6.5
            lower_left_y = -1
            lower_left_z = -3.8
            length = 1
            height = 1
            shape2 = showing_image(image_name="colorbar_new.png",
                                   x1=lower_left_x, y1=lower_left_y, z1=lower_left_z,
                                   x2=lower_left_x, y2=lower_left_y, z2=lower_left_z + height,
                                   x3=lower_left_x, y3=lower_left_y + length, z3=lower_left_z + height,
                                   x4=lower_left_x, y4=lower_left_y + length, z4=lower_left_z
                                   )
            # Viewer.display(shape)
            sc += shape2

        # DISPLAYING:
        #############
        # # We update the scene with the specified position of the center of the graph and the camera:
        # MTG_scene = prepareScene(MTG_scene, width=width, height=height,
        #                           x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
        #                           background_color=background_color)

        # We finally display the MTG on PlantGL:
        pgl.Viewer.display(MTG_scene)
        # pgl.Viewer.update()

        # # We wait for 1 second, if needed:
        # time.sleep(1)

        # For recording the image of the graph for making a video later:
        # -------------------------------------------------------------
        if recording_images:
            image_name = os.path.join(video_dir, 'root%.5d.png')
            pgl.Viewer.saveSnapshot(image_name % ID)

        # For recording the properties of g in a csv file:
        # ------------------------------------------------
        if recording_g_properties:
            prop_file_name = os.path.join(prop_dir, 'root%.5d.csv')
            recording_MTG_properties(g, file_name=prop_file_name % ID)

        # For integrating root variables on the z axis:
        # ----------------------------------------------
        if z_classification:
            z_dictionnary = classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
            z_dictionnary["time_in_days"] = time_step_in_days * ID
            z_dictionnary_series.append(z_dictionnary)

    # At the end of the loop, we can record the classification according to z:
    if z_classification:
        # We create a data_frame from the vectors generated in the main program up to this point:
        data_frame_z = pd.DataFrame.from_dict(z_dictionnary_series)
        # We save the data_frame in a CSV file:
        data_frame_z.to_csv('z_classification.csv', na_rep='NA', index=False, header=True)
        print("A new file 'z_classification.csv' has been saved.")

    return g

########################################################################################################################
########################################################################################################################
# MAIN FUNCTIONS FOR AVERAGING SEVERAL MTG INTO ONE
########################################################################################################################

# Function for creating one average MTG from a list of MTG:
#----------------------------------------------------------
def averaging_a_list_of_MTGs(list_of_MTG_numbers=[143, 167, 191], list_of_properties=[],
                             directory_path = 'MTG_files', recording_averaged_MTG=True,
                             averaged_MTG_name=None,
                             recording_directory = 'averaged_MTG_files'):
    """
    This function calculate an "averaged" MTG that attributes for the property of a given node the mean value of that
    property value for the same node found in a list of MTG. This is especially useful for calculating mean values over
    a time range.
    :param list_of_MTG_numbers: a list containing the ID number of each MTG to be opened (each MTG name is assumed to be in the format 'rootXXXXX.pckl')
    :param list_of_properties: a list containing the names of the properties to average
    :param directory_path: the path of the directory where MTG files are stored
    :param recording_averaged_MTG: if True, the calculated averaged MTG will be recorded
    :param averaged_MTG_name: the name of the new averaged MTG file
    :param recording_directory: the directory in which the new averaged MTG will be recorded
    :return: the new averaged MTG
    """

    # If the new MTG file is to be recorded:
    if recording_averaged_MTG:
        # We check the directory where the average MTG will be recorded:
        if not os.path.exists(recording_directory):
            # Then we create it:
            os.mkdir(recording_directory)

    # We initialize the list of MTG:
    list_of_MTGs = []
    # We open all the MTG, as specified from the list of MTG numbers:
    for number in list_of_MTG_numbers:
        filename = os.path.join(directory_path, 'root%.5d.pckl' % number).replace("//", "/")
        f = open(filename, 'rb')
        g = pickle.load(f)
        f.close()
        list_of_MTGs.append(g)

    # We initialize the number of vertices in the final MTG:
    n_vertices = 0
    # We cover all the MTG in the list and compare their number of vertices, so that we select the MTG with the highest
    # number of vertices, which will be the frame of the new, average MTG:
    for g in list_of_MTGs:
        if g.nb_vertices() > n_vertices:
            n_vertices = g.nb_vertices()
            averaged_MTG = g

    # We cover each possible vertex in the MTGs:
    for vid in range(1, n_vertices):
        # We cover each property to be averaged:
        for property in list_of_properties:
            # We initialize an empty list that will contain the values to be averaged:
            temporary_list = []
            # We cover each MTG in the list of MTGs:
            for g in list_of_MTGs:
                # We try to access the value corresponding to the specific property, MTG and vertex:
                try:
                    value = g.property(property)[vid]
                    # If it exists, we add this value to the list:
                    temporary_list.append(value)
                except:
                    # Otherwise, we add a nul value instead:
                    temporary_list.append(0)
            # Finally, we calculate the mean value from the list:
            mean_value = mean(temporary_list)
            # And we assign the mean value to the vertex in the averaged MTG:
            averaged_MTG.property(property)[vid] = mean_value

    if recording_averaged_MTG:
        # If no specific name for the averaged MTG has been specified:
        if not averaged_MTG_name:
            first = list_of_MTG_numbers[0]
            last = list_of_MTG_numbers[-1]
            print("We record the averaged MTG calculated from the list", first,"-",last," in the directory", os.getcwd())
            new_name = 'averaged_root_' + str(first) + '_' + str(last) + '.pckl'
        else:
            new_name=averaged_MTG_name
        # We record the averaged MTG:
        g_file_name = os.path.join(recording_directory, new_name)
        with open(g_file_name, 'wb') as output:
            pickle.dump(g, output, protocol=2)

    return  averaged_MTG

# # Example:
# averaged_MTG = averaging_a_list_of_MTGs(list_of_MTG_numbers=range(0,23),
#                                         list_of_properties=['length', 'C_hexose_root'],
#                                         directory_path = 'C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0097/MTG_files/',
#                                         recording_directory='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0097/averaged_MTG_files/')

# Function for creating many averaged MTG when covering a large list of MTG files:
#---------------------------------------------------------------------------------
def averaging_through_a_series_of_MTGs(MTG_directory='MTG_files', list_of_properties=[],
                                       odd_number_of_MTGs_for_averaging = 3, recording_averaged_MTG=True,
                                       recording_directory='averaged_MTG_files'):
    """
    This function covers each MTG in a directory, and for each one calculates an averaged MTG based on a specific number
    of MTG files (located right before and right after the current MTG in the list), for a certain list of properties.
    Ex: for the MTG 'root00005.pckl', the new averaged MTG 'root00004.pckl' will be calculated from 'root00004.pckl',
    'root00005.pckl' and 'root00006.pckl' if the specified number of MTGs to average is 3. This may not be exactly true
    for the very first or very last MTG files (they are averaged on either only the subsequent MTG files or only the
    preceding files, respectively).
    :param MTG_directory: the path of the directory in which MTG files are stored
    :param list_of_properties: a list containing the names of the properties to average
    :param odd_number_of_MTGs_for_averaging: an odd number (i.e. 1, 3, 5, 7, etc) that corresponds to the total number of MTG for averaging
    :param recording_averaged_MTG: if True, the new averaged MTG files will be recorded
    :param recording_directory: the name of the directory in which the new averaged MTG files will be saved
    :return:
    """

    print("We now calculate a mean MTG averaged over", odd_number_of_MTGs_for_averaging,
          "successive MTGs, for each MTG present in the directory", MTG_directory, "...")

    if recording_averaged_MTG:
        # If the directory where the average MTG will be recorded does not exist:
        if not os.path.exists(recording_directory):
            # Then we create it:
            os.mkdir(recording_directory)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(recording_directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    # Defining the list of MTG files:
    # -------------------------------
    filenames = Path(MTG_directory).glob('*pckl')
    # We reorder the files if needed, from the lowest to the highest number:
    filenames = sorted(filenames)

    # Calculating the number of MTG at a higher or lower position compared to the targeted MTG to include when averaging:
    half_number = floor(odd_number_of_MTGs_for_averaging/2.)
    # For example, if the odd number is 7, then we will average the MTG together with 3 MTG located before it and 3 MTG located after it.

    # We cover each MTG in the directory:
    for target_MTG_position in range(0,len(filenames)):
        # If the current position of the MTG in the list is stricly lower than half of the prescribed number of MTGs used
        # for averaging (X):
        if target_MTG_position < half_number:
            # Then the averaged MTG will be calculated starting at the first MTG of the list and ending X MTG afer the current position:
            MTG_positions_to_cover = range(0,
                                           target_MTG_position + half_number + 1)
        # If the current position of the MTG in the list is in an intermediate, "normal" position:
        elif target_MTG_position >= half_number and target_MTG_position < len(filenames) - half_number:
            # Then the averaged MTG will be calculated starting X MTG before and ending X MTG afer the current position:
            MTG_positions_to_cover =range(target_MTG_position - half_number,
                                          target_MTG_position + half_number + 1)
        # If the current position of the MTG in the list is close to the last MTG of the list:
        elif target_MTG_position >= len(filenames) - half_number:
            # Then the averaged MTG will be calculated starting X MTG before and ending at the last MTG of the list:
            MTG_positions_to_cover = range(target_MTG_position - half_number,
                                           len(filenames))

        # We initialize an empty list of MTG ID number:
        list_of_MTG_ID = []
        # And we fill this list with all the MTG numbers to average:
        for position in MTG_positions_to_cover:
            # We get the full name of the MTG file, ending by 'rootXXXXX.pckl', where X is a digit:
            MTG_path=filenames[position]
            # We extract the characters corresponding to the ID number of the MTG, and convert it to an integer:
            MTG_ID = int(MTG_path[-10:-5])
            # We add the number to the list:
            list_of_MTG_ID.append(MTG_ID)
        # Eventually, we proceed to the averaging using the function 'averaging_a_list_of_MTGs':
        target_MTG_ID = int(filenames[target_MTG_position][-10:-5])
        print("For the MTG", str(target_MTG_ID), "we average over the MTGs' list", list_of_MTG_ID, "...")
        averaging_a_list_of_MTGs(list_of_MTG_numbers=list_of_MTG_ID,
                                 list_of_properties=list_of_properties,
                                 directory_path =MTG_directory,
                                 recording_averaged_MTG=recording_averaged_MTG,
                                 averaged_MTG_name='root%.5d.pckl' % target_MTG_ID,
                                 recording_directory=recording_directory)

    return

# # Example:
# averaging_through_a_series_of_MTGs(MTG_directory='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0097/MTG_files',
#                                    list_of_properties=['length', 'C_hexose_root'],
#                                    odd_number_of_MTGs_for_averaging=7,
#                                    recording_averaged_MTG=True,
#                                    recording_directory='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0097/averaged_MTG_files')

########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == "__main__":

    print("Considering opening and treating recorded MTG files...")

    # # If we want to add time and colorbar:
    # # image_path = os.path.join("./", 'colobar.png')
    # im = Image.open('colorbar.png')
    # im_new = add_margin(image=im, top=3000, right=0, bottom=0, left=0, color=(0, 0, 0, 0))
    # newsize = (8000, 8000)
    # im_resized = im_new.resize(newsize)
    # im_resized.save('colorbar_new.png', quality=95)
    # # im_new.save('colorbar_new.png', quality=95)

    # To open a list:
    loading_MTG_files(my_path='C:/Users/frees/rhizodep/src/rhizodep/scenarios/outputs/Scenario_0142',
                      opening_list=True,
                      MTG_directory='MTG_files',
                      list_of_MTG_ID=range(0, 900),
                      # property="C_hexose_reserve", vmin=1e-8, vmax=1e-5, log_scale=True,
                      # property="net_sucrose_unloading", vmin=1e-12, vmax=1e-8, log_scale=True,
                      # property="net_hexose_exudation_rate_per_day_per_gram", vmin=1e-5, vmax=1e-2, log_scale=True,
                      property="net_rhizodeposition_rate_per_day_per_cm", vmin=1e-8, vmax=1e-5, log_scale=True,
                      # property="C_hexose_root", vmin=1e-6, vmax=1e-3, log_scale=True,
                      cmap='jet',
                      width=1200, height=1200,
                      x_center=0.0, y_center=0.0, z_center=0.0,
                      x_cam=0.2, y_cam=0.0, z_cam=-0.2,
                      background_color=[94,76,64],
                      root_hairs_display=True,
                      mycorrhizal_fungus_display=False,
                      step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                      adding_images_on_plot=False,
                      recording_images=True,
                      images_directory="new_root_image_test",
                      printing_sum=False,
                      recording_sum=False,
                      recording_g_properties=False,
                      z_classification=False, z_min=0.00, z_max=1., z_interval=0.05, time_step_in_days=1)
    print("Done!")

    # To avoid closing PlantGL as soon as the run is done:
    input()



    # # (Re-)plotting the original MTG files:
    # loading_MTG_files(my_path='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0097/',
    #                   opening_list=True,
    #                   MTG_directory="MTG_files",
    #                   list_of_MTG_ID=range(3,68),
    #                   # property="C_hexose_reserve", vmin=1e-8, vmax=1e-5, log_scale=True,
    #                   # property="net_sucrose_unloading", vmin=1e-12, vmax=1e-8, log_scale=True,
    #                   # property="net_hexose_exudation_rate_per_day_per_gram", vmin=1e-5, vmax=1e-2, log_scale=True,
    #                   # property="net_rhizodeposition_rate_per_day_per_cm", vmin=1e-8, vmax=1e-5, log_scale=True, cmap='jet',
    #                   property="C_hexose_root", vmin=1e-6, vmax=1e-3, log_scale=True, cmap='jet',
    #                   width=1200, height=1200,
    #                   x_center=0, y_center=0, z_center=-0.1, z_cam=-0.2,
    #                   camera_distance=0.4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
    #                   # x_center=0, y_center=0, z_center=-0, z_cam=-0,
    #                   # camera_distance=1, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
    #                   adding_images_on_plot=False,
    #                   recording_images=True,
    #                   images_directory="original_root_images",
    #                   printing_sum=False,
    #                   recording_sum=False,
    #                   recording_g_properties=False,
    #                   z_classification=False, z_min=0.00, z_max=1., z_interval=0.05, time_step_in_days=1)
    #
    # # Calculating average MTG:
    # averaging_through_a_series_of_MTGs(MTG_directory='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0100/MTG_files',
    #                                    list_of_properties=['length', 'C_hexose_root', 'net_rhizodeposition_rate_per_day_per_cm'],
    #                                    odd_number_of_MTGs_for_averaging=7,
    #                                    recording_averaged_MTG=True,
    #                                    recording_directory='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0100/averaged_MTG_files')
    # # Plotting the averaged MTG files:
    # loading_MTG_files(my_path='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0100/',
    #                   opening_list=True,
    #                   MTG_directory="averaged_MTG_files",
    #                   # property="C_hexose_reserve", vmin=1e-8, vmax=1e-5, log_scale=True,
    #                   # property="net_sucrose_unloading", vmin=1e-12, vmax=1e-8, log_scale=True,
    #                   # property="net_hexose_exudation_rate_per_day_per_gram", vmin=1e-5, vmax=1e-2, log_scale=True,
    #                   # property="net_rhizodeposition_rate_per_day_per_cm", vmin=1e-8, vmax=1e-5, log_scale=True, cmap='jet',
    #                   property="C_hexose_root", vmin=1e-6, vmax=1e-3, log_scale=True, cmap='jet',
    #                   width=1200, height=1200,
    #                   x_center=0, y_center=0, z_center=-0.1, z_cam=-0.2,
    #                   camera_distance=0.4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
    #                   # x_center=0, y_center=0, z_center=-0, z_cam=-0,
    #                   # camera_distance=1, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
    #                   adding_images_on_plot=False,
    #                   recording_images=True,
    #                   images_directory="averaged_root_images",
    #                   printing_sum=False,
    #                   recording_sum=False,
    #                   recording_g_properties=False,
    #                   z_classification=False, z_min=0.00, z_max=1., z_interval=0.05, time_step_in_days=1)
# print("Done!")