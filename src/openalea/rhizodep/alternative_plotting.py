import pyvista as pv
import pyvistaqt as pvqt
import pickle
from math import floor
import time
import os, os.path
import numpy as np
from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order
from openalea.mtg import turtle as turt
from rhizodep.tools import get_root_visitor, my_colormap

import pandas as pd


########################################################################################################################
# DEFINING PLOTTING FUNCTIONS:
##############################

# Function for plotting a root MTG with pyvista (written by Tristan Gerault):
#----------------------------------------------------------------------------

def plot_mtg_alt(g, cmap_property):
    props = g.properties()
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    plotted_vids = []
    tubes = []
    for vid in pre_order(g, root):

        if vid not in plotted_vids:
            root = g.Axis(vid)
            plotted_vids += root
            if vid != 1:
                parent = g.Father(vid)
                grandparent = g.Father(parent)
                # We need a minimum of two anchors for the new axis
                root = [grandparent, parent] + root

            points = np.array([[props["x2"][v], props["y2"][v], props["z2"][v]] for v in root])
            spline = pv.Spline(points)
            spline[cmap_property] = [props[cmap_property][v] for v in root]
            # Adjust radius of each element
            spline["radius"] = [props["radius"][v] for v in root]
            tubes += [spline.tube(scalars="radius", absolute=True)]

    root_system = pv.MultiBlock(tubes)
    root_system.plot()
    return root_system

# Function for plotting a root MTG with pyvista (written by Tristan Gerault):
#----------------------------------------------------------------------------

def plot_mtg_new(g):

    plotted_vids = []
    tubes = []
    colors = []
    opacities = []

    # We get a dictionnary of all properties for all nodes:
    props = g.properties()
    # We start from the origin of the root system:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We cover all the nodes from the base up to the tips:
    for vid in pre_order(g, root):

        # If the node has not been added to the list of registered shapes for plotting:
        if vid not in plotted_vids:
            # We get the indices of all the nodes belonging to the axis:
            current_axis = g.Axis(vid)
            # We add the indices to the list of already-plotted vid:
            plotted_vids += current_axis
            # If the node does not correspond to the base of the root system:
            if vid != 1:
                parent = g.Father(vid)
                grandparent = g.Father(parent)
                # We need a minimum of two anchors for the new axis
                current_axis = [grandparent, parent] + current_axis

            points = np.array([[props["x2"][v], props["y2"][v], props["z2"][v]] for v in current_axis])
            lines = pv.MultipleLines(points)
            # We assign the property "radius" to the lines:
            lines["radius"] = [props["radius"][v] for v in current_axis]
            # And we create tubes from the lines with adapted radius, which we add to the preexisting tubes:
            tubes += [lines.tube(scalars="radius", absolute=True)]
            colors += [props["color"][v] for v in current_axis]

    root_system = pv.MultiBlock(tubes)
    # root_system.plot()

    # Adding the multi-blocks to a plot:
    p = pv.Plotter()
    actor, mapper = p.add_composite(root_system)
    print("Adding the colors and transparencies of the plot...")
    # Setting the color of each block in the multiblocks:
    for i in range(1, root_system.n_blocks + 1):
        mapper.block_attr[i].color = colors[i - 1]
        # mapper.block_attr[i].opacity = opacity[i - 1]
    p.show()

    return root_system

# Function that methodically creates each shape (takes a long time for large root systems):
#------------------------------------------------------------------------------------------

def plotting_roots_with_pyvista(g, displaying_root_hairs = True,
                                showing=True, recording_image=True, image_file='plot_.png',
                                factor_of_higher_resolution = 1,
                                background_color = [94, 76, 64],
                                plot_width=1000, plot_height=750,
                                camera_x=0.3, camera_y=0., camera_z=-0.07,
                                focal_x=0., focal_y=0., focal_z=-0.07,
                                closing_window=False):
    """
    This functions aims to plot a root system with Pyvista, creating step by step every shape.
    :param g: the root MTG to be displayed
    :param displaying_root_hairs: if True, root hairs will be displayed around the roots
    :param showing: if True, a Pyvista window will be opened
    :param recording_image: if True, an image file of the plot will be recorded
    :param image_file: name of the image file
    :param closing_window: if True, the Pyvista window will be closed after having shown the plot
    :param background_color: a list containing three integers corresponding to Red, Green and Blue values (0-255)
    :param plot_width: width of the plot (in pixels)
    :param plot_height: height of the plot (in pixels)
    :param camera_x: x-coordinate of the camera position
    :param camera_y: y-coordinate of the camera position
    :param camera_z: z-coordinate of the camera position
    :param focal_x: x-coordinate of the focal point (from which plot is centered and allowed to rotate from)
    :param focal_y: y-coordinate of the focal point (from which plot is centered and allowed to rotate from)
    :param focal_z: z-coordinate of the focal point (from which plot is centered and allowed to rotate from)
    :return:
    """

    # We initialize a plot:
    p = pv.Plotter()
    # p = pvqt.BackgroundPlotter() # If we want to update the graph when opened
    if recording_image and not showing:
        p.off_screen=True

    # We initialize a list of blocks and empty lists:
    blocks = pv.MultiBlock()
    colors=[]
    opacity=[]

    # CREATING THE SHAPES:
    ######################

    # The creation of the cylinders with no directions raises a Warning in numpy, but we can ignore it.
    np.seterr(invalid='ignore')

    print("   > Creating the shapes of the plot...")
    # We cover all the vertices in the MTG g:
    for vid in g.vertices_iter(scale=1):

        # We define the current root element as n:
        n = g.node(vid)
        try:
            # We skip this element if it does not have a positive length:
            if n.length <= 0.:
                continue
        except:
            print("PROBLEM!")

        # We calculate the geometric features required by pyvista to display a cylinder at the right spot:
        x1 = n.x1
        x2 = n.x2
        y1 = n.y1
        y2 = n.y2
        z1 = n.z1
        z2 = n.z2
        center_x = (x1 + x2) / 2.
        center_y = (y1 + y2) / 2.
        center_z = (z1 + z2) / 2.
        direction_x = x2 - x1
        direction_y = y2 - y1
        direction_z = z2 - z1
        height = np.sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1))
        radius = n.radius
        # We add the color and opacity to a list:
        color = n.color
        colors.append(color)
        opacity.append(1) # Here, all the elements are supposed to be fully opaques

        # We create a new cylinder with the proper dimensions and position:
        cyl = pv.Cylinder(center=(center_x,center_y,center_z), direction=(direction_x, direction_y, direction_z),
                          radius=radius, height=height)
        # And we add this cylinder to the list of blocks:
        blocks.append(cyl)

        # If root hairs have to be displayed and if this element should show some root hairs:
        if displaying_root_hairs and n.root_hair_length > 0.:
            # We color the root hairs according to the proportion of living and dead root hairs.
            # For dead hairs:
            dead_transparency = 0.90
            dead_color_vector = [0, 0, 0]
            dead_color_vector_Red = dead_color_vector[0]
            dead_color_vector_Green = dead_color_vector[1]
            dead_color_vector_Blue = dead_color_vector[2]
            # For living hairs:
            living_transparency = 0.70
            living_color_vector = n.color
            living_color_vector_Red = living_color_vector[0]
            living_color_vector_Green = living_color_vector[1]
            living_color_vector_Blue = living_color_vector[2]
            # For the final mix between living hairs and dead hairs:
            living_fraction = n.living_root_hairs_number / n.total_root_hairs_number
            transparency = dead_transparency + (living_transparency - dead_transparency) * living_fraction
            color_vector_Red = floor(dead_color_vector_Red
                                     + (living_color_vector_Red - dead_color_vector_Red) * living_fraction)
            color_vector_Green = floor(dead_color_vector_Green
                                       + (living_color_vector_Green - dead_color_vector_Green) * living_fraction)
            color_vector_Blue = floor(dead_color_vector_Blue
                                      + (living_color_vector_Blue - dead_color_vector_Blue) * living_fraction)
            color_for_hairs = [color_vector_Red, color_vector_Green, color_vector_Blue]

            # We add the color and opacity to a list:
            colors.append(color_for_hairs)
            opacity.append(1-transparency)

            # We create a new cylinder with the proper dimensions and position:
            cyl_for_hairs = pv.Cylinder(center = (center_x, center_y, center_z),
                                        direction = (direction_x, direction_y, direction_z),
                                        radius = radius + n.root_hair_length,
                                        height=height, capping=False)
            # And we add this cylinder to the list of blocks:
            blocks.append(cyl_for_hairs)

    # We reset the warning option in Numpy:
    np.seterr(invalid=None)

    # CREATING THE SCENE:
    # Adding the multi-blocks to a plot:
    actor, mapper = p.add_composite(blocks)
    print("   > Adding the colors and transparencies of the plot...")
    # We cover each block and set its color and opacity:
    for i in range(1,blocks.n_blocks+1):
        mapper.block_attr[i].color = colors[i-1]
        mapper.block_attr[i].opacity = opacity[i-1]

    # We set the color of the background:
    p.background_color = background_color

    # ADJUSTING THE CAMERA:
    # We can define the target position and the camera position:
    p.camera.focal_point = (focal_x, focal_y, focal_z)
    p.camera.position = (camera_x, camera_y, camera_z)
    # Or we can define the relative distance and elevation of the camera compared to the middle of the plot:
    # p.camera.distance = 0.2
    # p.camera_elevation = -0.2
    # p.reset_camera()
    # p.view_yz()
    # pv.Axes.origin=(0,0,0.0)
    p.camera_set = True

    # SETTING AXES:
    # p.add_axes()
    # p.add_axes_at_origin()

    # DIVERSE SETTINGS:
    # We set the background color of the plot:
    p.background_color = background_color
    # We set the final size of the window:
    p.window_size = [plot_width, plot_height]
    # We can improve the resolution of the image:
    p.image_scale = factor_of_higher_resolution

    # p.line_smoothing = True
    # p.add_bounding_box(line_width=3, color='black')
    # p.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)

    print("The plot has been created!")

    # SHOWING/RECORDING THE PLOT:
    if recording_image:
        # p.screenshot(filename='test_MTG.png')
        p.show(screenshot=image_file, auto_close=closing_window)
        # # p.close()
    elif showing:
        p.show(auto_close=closing_window)

    if closing_window:
        pv.close_all()

    return p

# Function that generates very quickly a 3D plot of the root MTG (memory issue with large MTG?):
#-----------------------------------------------------------------------------------------------

def fast_plotting_roots_with_pyvista(g, displaying_root_hairs = False,
                                     showing=True, recording_image=True, image_file='plot.png',
                                     factor_of_higher_resolution=1,
                                     background_color = [94, 76, 64],
                                     plot_width=1000, plot_height=750,
                                     camera_x=0.3, camera_y=0., camera_z=-0.07,
                                     focal_x=0., focal_y=0., focal_z=-0.07,
                                     closing_window=False):
    """
    This functions aims to plot root system with Pyvista.
    :param g: the root MTG to be displayed
    :param displaying_root_hairs: if True, root hairs will be displayed around the roots
    :param showing: if True, a pyvista window will be opened
    :param recording_image: if True, an image file of the plot will be recorded
    :param image_file: name of the image file
    :param closing_window: if True, the Pyvista window will be closed after having shown the plot
    :param background_color: a list containing three integers corresponding to Red, Green and Blue values (0-255)
    :param plot_width: width of the plot (in pixels)
    :param plot_height: height of the plot (in pixels)
    :param camera_x: x-coordinate of the camera position
    :param camera_y: y-coordinate of the camera position
    :param camera_z: z-coordinate of the camera position
    :param focal_x: x-coordinate of the focal point (from which plot is centered and allowed to rotate from)
    :param focal_y: y-coordinate of the focal point (from which plot is centered and allowed to rotate from)
    :param focal_z: z-coordinate of the focal point (from which plot is centered and allowed to rotate from)
    :return:
    """

    # CREATING LISTS OF PROPERTIES FROM THE MTG:
    # We initialize empty lists:
    x = []
    y = []
    z = []
    radius = []
    radius_for_hairs = []
    colors = []
    colors_with_opacity = []
    colors_for_hairs = []

    # We cover each node of the MTG:
    for vid in g.vertices_iter(scale=1):
        # We get access to the current node "n":
        n = g.node(vid)
        if n.length <= 0.:
            continue
        # We add the radius two times for each node, as the number of cells will be twice the number of segments:
        radius.append(n.radius), radius.append(n.radius)
        # To each coordinate list, we add the corresponding coordinate from the starting point of the segment,
        # and then the coordinate from its ending point, for the future creation of line_segments_from_points:
        x.append(n.x1), x.append(n.x2)
        y.append(n.y1), y.append(n.y2)
        z.append(n.z1), z.append(n.z2)
        # If RGB color without opacity is enough to plot, we add the color of the node one time for each future tube:
        colors.append(n.color)
        # Otherwise, we extract the color Red, Green and Blue in 8-bit:
        color_r = n.color[0]
        color_g = n.color[1]
        color_b = n.color[2]
        # We set the transparency:
        transparency = 0.
        # We create a RGBA color containing the opacity in the fourth position:
        color=pv.Color([color_r, color_g, color_b,1 - transparency])
        # We add this RGBA color to the list:
        colors_with_opacity.append(color.int_rgba)

        # If root hairs are displayed, we color them according to the proportion of living and dead root hairs:
        if displaying_root_hairs:
            # If the root hairs are visible:
            if n.root_hair_length > 0:
                # Then we set the radius according to the length of the root hair and add it two times to the corresponding list:
                radius_for_hairs.append(n.root_hair_length), radius_for_hairs.append(n.root_hair_length)
            else:
                # Otherwise, we still add two times a very small radius of cylinder that will be masked by the actual
                # radius of the cylinder (this is necessary for getting the right size of mesh_for_hairs when adding
                # the root hairs colors below):
                radius_for_hairs.append(n.radius/10.), radius_for_hairs.append(n.radius/10.)
            # For dead hairs:
            dead_transparency = 0.90
            dead_color_hairs_Red = 0
            dead_color_hairs_Green = 0
            dead_color_hairs_Blue = 0
            # For living hairs:
            living_transparency = 0.70
            living_color_hairs_Red = color_r
            living_color_hairs_Green = color_g
            living_color_hairs_Blue = color_b
            # For the final mix between living hairs and dead hairs:
            if n.total_root_hairs_number <= 0.:
                living_fraction = 0.
            else:
                living_fraction = n.living_root_hairs_number / n.total_root_hairs_number

            transparency = dead_transparency + (living_transparency - dead_transparency) * living_fraction
            color_hairs_Red = floor(dead_color_hairs_Red
                                     + (living_color_hairs_Red - dead_color_hairs_Red) * living_fraction)
            color_hairs_Green = floor(dead_color_hairs_Green
                                       + (living_color_hairs_Green - dead_color_hairs_Green) * living_fraction)
            color_hairs_Blue = floor(dead_color_hairs_Blue
                                      + (living_color_hairs_Blue - dead_color_hairs_Blue) * living_fraction)
            # We create a RGBA color containing the opacity in the fourth position:
            color_hairs = pv.Color([color_hairs_Red, color_hairs_Green, color_hairs_Blue, 1 - transparency])
            # We add this RGBA color to the list:
            colors_for_hairs.append(color_hairs.int_rgba)
    # Now we have created lists that have the size of the number of nodes with positive length.

    # data_frame = pd.DataFrame({"Original_colors": colors_with_opacity,
    #                            "Root_hairs_colors": colors_for_hairs})
    # print(data_frame)

    # CREATING THE SEGMENTS AND TUBES WITH CORRECT RADIUS:
    # We now create the points dataset used for creating non-connected segments, using the x,y,z coordinates of all vertices:
    points = np.column_stack((x, y, z))
    # We create a geometry of non-connected segments:
    lines = pv.line_segments_from_points(points)
    # We register for each cell the value of the corresponding radius:
    lines["radius"] = np.array(radius)
    # If root hairs are to be displayed, then we also register the value of the radius for hairs:
    if displaying_root_hairs:
        lines["radius_for_hairs"] = np.array(radius_for_hairs)
    # And we create tubes from the segments, scaling the radius from each tube on the prescribed radius values:
    mesh = lines.tube(scalars="radius", absolute=True)
    mesh = mesh.clean()
    # => "mesh" now contains the set of tubes with the correct prescribed coordinates and radius, but does not have colors.
    # If root hairs are to be displayed, we create another set of tubes with larger radius:
    if displaying_root_hairs:
        mesh_for_hairs = lines.tube(scalars="radius_for_hairs", absolute=True)
        mesh_for_hairs = mesh_for_hairs.clean()

    # # ADDING COLOR INFORMATION - IF ONLY RGB COLORS ARE USED AS SCALARS:
    # # We create a second list of segments, that will be only used for creating a list of colors of the right size:
    # lines_for_colors = pv.line_segments_from_points(points)
    # # We register the colors we want in this second list:
    # lines_for_colors["color"] = np.array(colors)
    # # We create again tubes from the segments and assign the colors in the scalar values:
    # mesh_for_colors = lines_for_colors.tube(scalars="color")
    # # And now we can finally register the correct list of colors with the proper dimension to the original mesh of tubes:
    # mesh["color"] = mesh_for_colors.cell_data.active_scalars

    # ADDING COLOR INFORMATION - IF RGBA COLORS WITH OPACITY ARE USED AS SCALARS:
    # We create another list of segments, that will be only used for creating a list of colors & opacities of the right size:
    lines_for_opacities = pv.line_segments_from_points(points)
    # We register the RGBA colors we want in this second list:
    lines_for_opacities["color_rgba"] = np.array(colors_with_opacity)
    # We create again tubes from the segments and assign the RGBA colors in the scalar values:
    mesh_for_opacities = lines_for_opacities.tube(scalars="color_rgba")
    mesh_for_opacities = mesh_for_opacities.clean()
    # And now we can finally register the correct list of colors & opacities with the proper dimension to the original mesh of tubes:
    mesh["color_rgba"] = mesh_for_opacities.cell_data.active_scalars

    # If root hairs are to be displayed, we do a similar operation for getting the right colors list:
    if displaying_root_hairs:
        # We create another list of segments, that will be only used for creating a list of colors & opacities of the
        # right size for root hairs:
        lines_for_hair_colors = pv.line_segments_from_points(points)
        # We register the RGBA colors we want in this second list:
        lines_for_hair_colors["color_rgba_hairs"] = np.array(colors_for_hairs)
        # We create again tubes from the segments and assign the RGBA colors in the scalar values:
        mesh_for_hair_colors = lines_for_hair_colors.tube(scalars="color_rgba_hairs")
        mesh_for_hair_colors = mesh_for_hair_colors.clean()
        # And now we can finally register the correct list of colors & hairs with the proper dimension to the
        # original mesh of tubes:
        mesh_for_hairs["color_rgba_hairs"] = mesh_for_hair_colors.cell_data.active_scalars

    # CREATING THE FINAL PLOT:
    # We initialize a plot:
    p = pv.Plotter()
    # p = pvqt.BackgroundPlotter() # If we want to update the graph when opened
    if recording_image:
        p.off_screen = True
    # # Finally, we add the final mesh of tubes with proper radius, to which we add the right colors.
    # Either in simple RGB:
    # p.add_mesh(mesh, scalars="color", rgb=True)
    # Or in RGBA format:
    p.add_mesh(mesh, scalars="color_rgba", rgba=True)
    # If root hairs are to be displayed, we also add the additional set of tubes with the specific colors and opacities:
    if displaying_root_hairs:
        p.add_mesh(mesh_for_hairs, scalars="color_rgba_hairs", rgba=True)

    # ADJUSTING THE CAMERA:
    # We can define the target position and the camera position:
    p.camera.focal_point = (focal_x, focal_y, focal_z)
    p.camera.position = (camera_x, camera_y, camera_z)
    # Or we can define the relative distance and elevation of the camera compared to the middle of the plot:
    # p.camera.distance = 0.2
    # p.camera_elevation = -0.2
    # p.reset_camera()
    # p.view_yz()
    # pv.Axes.origin=(0,0,0.0)
    p.camera_set = True

    # SETTING AXES:
    # p.add_axes()
    # p.add_axes_at_origin()

    # DIVERSE SETTINGS:
    # We set the background color of the plot:
    p.background_color = background_color
    # We set the final size of the window:
    p.window_size = [plot_width, plot_height]
    # We can improve the resolution of the image:
    p.image_scale = factor_of_higher_resolution

    # p.line_smoothing = True
    # p.add_bounding_box(line_width=3, color='black')
    # p.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)

    print("The plot has been created!")

    # SHOWING/RECORDING THE PLOT:
    if recording_image:
        # p.screenshot(filename='test_MTG.png')
        p.show(screenshot=image_file, auto_close=closing_window)
        # # p.close()
    elif showing:
        p.show(auto_close=closing_window)

    # if closing_window:
    #     pv.close_all()

    return p

########################################################################################################################

# USE CASE:
###########

if __name__ == '__main__':

    # Loading an actual root MTG that was previously recorded:
    filepath = 'C:/Users/frees/rhizodep/saved_outputs/selected_MTG_files/root01847.pckl'
    f = open(filepath, 'rb')
    g = pickle.load(f)
    f.close()
    #
    # In case it has not been done so far - we define the colors in g according to a specific property:
    # my_colormap(g, property_name="C_hexose_root", cmap='jet', vmin=1e-6, vmax=1e-3, lognorm=True)
    # my_colormap(g, property_name="total_exchange_surface_with_soil_solution", cmap='jet', vmin=1e-6, vmax=1e-3, lognorm=True)
    my_colormap(g, property_name = "net_rhizodeposition_rate_per_day_per_cm", cmap='jet', vmin=1e-8, vmax=1e-5, lognorm=True)

    # We can use the function based on plotting each axis (from Tristan):
    # plot_mtg_alt(g, cmap_property='length')
    # plot_mtg_new(g)

    # # We can create the "long-way" plot (TAKES TIME, BUT ALWAYS WORKS!):
    # p = plotting_roots_with_pyvista(g, displaying_root_hairs = False,
    #                                 showing=True, recording_image=True,
    #                                 image_file='C:/Users/frees/rhizodep/saved_outputs/plot.png',
    #                                 background_color = [94, 76, 64],
    #                                 plot_width=800, plot_height=800,
    #                                 camera_x=0.3, camera_y=0., camera_z=-0.07,
    #                                 focal_x=0., focal_y=0., focal_z=-0.07)

    # We can use the "fast" function that uses some tricks to plot all at once (FAST, BUT SOMETIMES BUGS!):
    fast_plotting_roots_with_pyvista(g, displaying_root_hairs = False,
                                     showing=True, recording_image=False,
                                     image_file='C:/Users/frees/rhizodep/saved_outputs/plot.png',
                                     background_color = [94, 76, 64],
                                     plot_width=1000, plot_height=800,
                                     camera_x=0.3, camera_y=0., camera_z=-0.07,
                                     focal_x=0., focal_y=0., focal_z=-0.07)