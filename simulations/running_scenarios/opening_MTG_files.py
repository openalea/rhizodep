from math import sqrt, pi, trunc, floor, cos, sin
from decimal import Decimal
import time
import numpy as np
import pandas as pd
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
# DEFINING FUNCTIONS FOR DISPLAYING THE MTG IN A 3D GRAPH WITH PLANTGL
########################################################################################################################

def drawing_text(text="TEXT !", image_name="text.png", length=220, height=110, font_size=100):
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
    # https://docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
    # We draw the text on the created image:
    # draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 50), fill=(255, 255, 255, 200))
    draw.text((x1, y1), text, fill=(0, 0, 0), font=font_time)
    # We save the image as a png file:
    im.save(image_name, 'PNG')


def showing_image(image_name="text.png",
                  x1=0, y1=0, z1=0,
                  x2=0, y2=0, z2=1,
                  x3=0, y3=1, z3=1,
                  x4=0, y4=1, z4=0):
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
    my_path = os.path.join("./", image_name)
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


def add_margin(image, top, right, bottom, left, color):
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
    This function records the properties of each node of the MTG "g" in a csv file.
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


########################################################################################################################

########################################################################################################################
# MAIN FUNCTION FOR LOADING AND DISPLAYING/EXTRACTING PROPERTIES FROM MTG FILES
########################################################################################################################

def loading_MTG_files(my_path='',
                      opening_list=False,
                      single_MTG_file_number=3600,
                      property="C_hexose_root", vmin=1e-5, vmax=1e-2, log_scale=True, cmap='jet',
                      x_center=0, y_center=0, z_center=-1, z_cam=-2,
                      camera_distance=4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                      adding_images_on_plot=False,
                      recording_images=False,
                      z_classification=False, z_min=0.0, z_max=1., z_interval=0.1, time_step_in_days=1 / 24.,
                      printing_sum=True,
                      recording_sum=True,
                      printing_warnings=True,
                      recording_g=False,
                      recording_g_properties=True):
    """
    This function opens one MTG file or a list of MTG files, displays them and record some of their properties if needed.
    :return: The MTG file "g" that was loaded at last.
    """

    # Preparing the folders:
    # -----------------------

    # We define the directory "MTG_files"
    g_dir = os.path.join(my_path, 'MTG_files')
    print("MTG files are located in", g_dir)

    if recording_images:
        # We define the directory "video"
        video_dir = os.path.join(my_path, 'root_new_images')
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

    # Defining the list of files:
    # ----------------------------
    filenames = Path(g_dir).glob('*pckl')
    filenames = sorted(filenames)
    if opening_list:
        n_steps = len(filenames)
        step_initial = 0
        step_final = n_steps+1
        filename = filenames[0]
    else:
        step_initial = single_MTG_file_number - 1
        step_final = single_MTG_file_number + 1
        filename = filenames[single_MTG_file_number]

        # We cover each of the MTG files in the list (or only the specified file when requested):
    # ----------------------------------------------------------------------------------------
    for step in range(step_initial, step_final):

        # Defining the name of the MTG file:
        if opening_list:
            filename = filenames[step]
        print("Dealing with file number", step, "out of", step_final-1, "...")

        # Loading the MTG file:
        f = open(filename, 'rb')
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
                          x_center=x_center,
                          y_center=y_center,
                          z_center=z_center,
                          x_cam=x_cam,
                          y_cam=y_cam,
                          z_cam=z_cam)
        else:
            x_camera = camera_distance
            x_cam = camera_distance
            z_camera = z_cam
            # We plot the current file:
            sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                          x_center=x_center,
                          y_center=y_center,
                          z_center=z_center,
                          x_cam=x_camera,
                          y_cam=0,
                          z_cam=z_camera)

            # And we move the camera further from the root system:
            x_camera = x_cam + x_cam * step_back_coefficient * step
            z_camera = z_cam + z_cam * step_back_coefficient * step

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
        # We finally display the MTG on PlantGL:
        pgl.Viewer.display(sc)

        # For recording the image of the graph for making a video later:
        # -------------------------------------------------------------
        if recording_images:
            image_name = os.path.join(video_dir, 'root%.5d.png')
            pgl.Viewer.saveSnapshot(image_name % step)

        # For recording the properties of g in a csv file:
        # ------------------------------------------------
        if recording_g_properties:
            prop_file_name = os.path.join(prop_dir, 'root%.5d.csv')
            recording_MTG_properties(g, file_name=prop_file_name % step)

        # For integrating root variables on the z axis:
        # ----------------------------------------------
        if z_classification:
            z_dictionnary = classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
            z_dictionnary["time_in_days"] = time_step_in_days * step
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

# ACTUAL COMMAND:
#################

# # If we want to add time and colorbar:
# # image_path = os.path.join("./", 'colobar.png')
# im = Image.open('colorbar.png')
# im_new = add_margin(image=im, top=3000, right=0, bottom=0, left=0, color=(0, 0, 0, 0))
# newsize = (8000, 8000)
# im_resized = im_new.resize(newsize)
# im_resized.save('colorbar_new.png', quality=95)
# # im_new.save('colorbar_new.png', quality=95)

loading_MTG_files(my_path='C:\\Users\\frees\\rhizodep\\simulations\\running_scenarios\\outputs\\Scenario_0008',
                  # single_MTG_file_number=3598,
                  opening_list=True,
                  # property="C_hexose_reserve", vmin=1e-3, vmax=5e-3, log_scale=False,
                  # property="net_sucrose_unloading", vmin=1e-12, vmax=1e-8, log_scale=True,
                  # property="net_hexose_exudation_rate_per_day_per_gram", vmin=1e-5, vmax=1e-2, log_scale=True,
                  property="hexose_exudation_rate", vmin=1e-15, vmax=1e-10, log_scale=True, cmap='jet',
                  x_center=0, y_center=0, z_center=-0.1, z_cam=-0.2,
                  camera_distance=0.4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                  adding_images_on_plot=False,
                  recording_images=True,
                  printing_sum=False,
                  recording_sum=False,
                  recording_g=False,
                  recording_g_properties=False,
                  z_classification=False, z_min=0.00, z_max=1., z_interval=0.05, time_step_in_days=1 / 24.)

print("Done!")

# # To avoid closing PlantGL as soon as the run is done:
# input()