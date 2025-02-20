import os
from math import floor, ceil, trunc, log10

import imageio
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
# from pygifsicle import optimize

from openalea.mtg.plantframe import color

from openalea.rhizodep.tools import colorbar, sci_notation

# Use LaTeX as text renderer to get text in true LaTeX
# If the two following lines are left out, Mathtext will be used
# import matplotlib as mpl
# mpl.rc('text', usetex=True)

########################################################################################################################

# Definition of a function that can resize a list of images and make a movie from it:
#------------------------------------------------------------------------------------
def resizing_and_film_making(outputs_path='outputs',
                             images_folder='root_images',
                             resized_images_folder='root_images_resized',
                             film_making=True,
                             film_name="root_movie.gif",
                             image_transforming=True,
                             resizing=False, dividing_size_by=1.,
                             colorbar_option=True, colorbar_position=1,
                             colorbar_title="Radius (m)",
                             colorbar_cmap='jet', colorbar_lognorm=True,
                             ticks=[],
                             vmin=1e-6, vmax=1e0,
                             time_printing=True, time_position=1,
                             time_step_in_days=1., sampling_frequency=1, fps=24,
                             title=""):

    """
    This function enables to resize some images, add a time indication and a colorbar on them, and create a movie from it.
    :param outputs_path: the general path in which the folders containing images are located
    :param images_folder: the name of the folder in which images have been stored
    :param resized_images_folder: the name of the folder to create, in which transformed images will be saved
    :param film_making: if True, a movie will be created from the original or transformed images
    :param film_name: the name of the movie file to be created
    :param image_transforming: if True, images will first be transformed
    :param resizing: if True, images can be resized
    :param dividing_size_by: the number by which the original dimensions will be divided to create the resized image
    :param colorbar_option: if True, a colorbar will be added
    :param colorbar_position: the position of the colorbar (1 = bottom right, 2 = bottom middle),
    :param colorbar_title: the name of the property to be displayed on the bar
    :param colorbar_cmap: the name of the specific colormap in Python
    :param colorbar_lognorm: if True, the scale will be a log scale, otherwise, it will be a linear scale
    :param ticks: a list of values at which to put a major thick, and, if possible, the corresponding number label (if the list is empty, the ticks are positionned automatically),
    :param vmin: the min value of the color scale
    :param vmax: the max value of the color scale
    :param time_printing: if True, a time indication will be calculated and displayed on the image
    :param time_position: the position of the time indication (1 = top left for root graphs, 2 = bottom right for z-barplots)
    :param time_step_in_days: the original time step at which MTG images were generated
    :param sampling_frequency: the frequency at which images should be picked up and included in the transformation/movie (i.e. 1 image every X images)
    :param fps: frames per second for the .gif movie to create
    :param title: the name of the movie file
    :return:
    """

    images_directory=os.path.join(outputs_path, images_folder)
    resized_images_directory = os.path.join(outputs_path, resized_images_folder)

    # Getting a list of the names of the images found in the directory "video":
    filenames = Path(images_directory).glob('*.png')
    filenames = sorted(filenames)

    # We define the final number of images that will be considered, based on the "sampling_frequency" variable:
    number_of_images = floor(len(filenames) / float(sampling_frequency))

    if colorbar_option:
        path_colorbar = os.path.join(outputs_path, 'colorbar.png')
        # We create the colorbar:
        bar = colorbar(title=colorbar_title,
                       cmap=colorbar_cmap,
                       lognorm=colorbar_lognorm,
                       ticks=ticks,
                       vmin=vmin, vmax=vmax)
        # We save it in the output directory:
        bar.savefig(path_colorbar, facecolor="None", edgecolor="None")
        # We reload the bar with Image package:
        bar = Image.open(path_colorbar)
        if colorbar_position==1:
            new_size = (1200, 200)
            bar = bar.resize(new_size)
            box_colorbar = (-120,1070)
        elif colorbar_position==2:
            new_size = (1200, 200)
            bar = bar.resize(new_size)
            box_colorbar = (-120, 870)
        else:
            new_size = (1500, 250)
            bar = bar.resize(new_size)
            box_colorbar = (0, 1400)

    # 1. COMPRESSING THE IMAGES:
    if image_transforming:
        # If this directory doesn't exist:
        if not os.path.exists(resized_images_directory):
            # Then we create it:
            print(resized_images_directory)
            os.mkdir(resized_images_directory)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(resized_images_directory):
                for file in files:
                    os.remove(os.path.join(root, file))

        # We modify each image:
        print("Transforming the images and copying them into the directory 'root_images_resized'...")
        # We initialize the counts:
        number = 0
        count = 0
        remaining_images = number_of_images

        # We calculate the dimensions of the new images according to the variable size_division:
        dimensions = (int(1600 / dividing_size_by), int(1055 / dividing_size_by))

        # We cover each image in the directory:
        for filename in filenames:

            # We get the ID of the image in order to calculate the proper time step to be displayed:
            MTG_ID = int(filename[-9:-4])

            # The time is calculated:
            # time_in_days = time_step_in_days * (number_of_images - remaining_images) * sampling_frequency
            time_in_days = time_step_in_days * MTG_ID
            # The count is increased:
            count += 1
            # If the count corresponds to the target number, the image is added to the gif:
            if count == sampling_frequency:
                print("Transforming the images - please wait:", str(int(remaining_images)), "image(s) left")

                # Opening the image to modify:
                im = Image.open(filename)

                # Adding colorbar:
                if colorbar_option:
                    im.paste(bar, box_colorbar, bar.convert('RGBA'))

                # Adding text:
                if time_printing:

                    # OPTION 1 FOR ROOT SYSTEMS:
                    # ---------------------------
                    if time_position == 1:
                        draw = ImageDraw.Draw(im)
                        # For time display:
                        # ------------------
                        # draw.text((x, y),"Sample Text",(r,g,b))
                        font_time = ImageFont.truetype("./timesbd.ttf", 35)
                        # See a list of available fonts on: https://docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
                        (x1, y1) = (40, 40)
                        time_text = "t = " + str(int(floor(time_in_days))) + " days"
                        draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 50), fill=(255, 255, 255, 200))
                        draw.text((x1, y1), time_text, fill=(0, 0, 0), font=font_time)
                        # draw.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

                    # OPTION 2 FOR Z BARPLOTS:
                    # -----------------------
                    if time_position == 2:
                        draw = ImageDraw.Draw(im)
                        # For time display:
                        # ------------------
                        # draw.text((x, y),"Sample Text",(r,g,b))
                        font_time = ImageFont.truetype("./timesbd.ttf", 40)
                        # See a list of available fonts on: https://docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
                        (x1, y1) = (880, 800)
                        time_text = "t = " + str(int(floor(time_in_days))) + " days"
                        draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 30), fill=(255, 255, 255, 0))
                        draw.text((x1, y1), time_text, fill=(0, 0, 0), font=font_time)

                    # # For caption of colorbar:
                    # # ------------------------
                    # title_text_position=(100,1020)
                    # font_title = ImageFont.truetype("./timesbd.ttf", 23)
                    # # text_color =(200,200,200,255) #RGBA, the last digit corresponds to alpha canal (transparency)
                    # # draw.text(title_text_position, title, (0, 0, 0), font=font_title, fill=text_color)
                    # draw.text(title_text_position, title, (0, 0, 0), font=font_title)

                # Transforming the image:
                if resizing:
                    im_to_print = im.resize(dimensions, resample=0)
                else:
                    im_to_print = im

                # We get the last characters of the path of the file, which correspond to the actual name 'rootXXXXX':
                name = filename[-13:-4] + '.png'
                # Saving the new image:
                image_name = os.path.join(resized_images_directory, name)
                im_to_print.save(image_name, quality=20, optimize=True)

                # We update the local counts:
                number = number + 1
                remaining_images = remaining_images - 1
                count = 0
        print("The new images have been transformed!")

    # 2. CREATING THE VIDEO FILE:
    if film_making:

        print("Making the video...")

        with imageio.get_writer(os.path.join(outputs_path, film_name), mode='I', fps=fps) as writer:
            if image_transforming:
                filenames = Path(resized_images_directory).glob('*.png')
                filenames = sorted(filenames)
                sampling_frequency = 1
            else:
                filenames = Path(images_directory).glob('*.png')
                filenames = sorted(filenames)
                sampling_frequency = sampling_frequency
            remaining_images = floor(len(filenames) / float(sampling_frequency)) + 1
            print(remaining_images, "images are considered at this stage.")
            # We add the first image:
            filename = filenames[0]
            image = imageio.imread(str(filename))
            writer.append_data(image)
            # We reduce the number of images left:
            remaining_images = remaining_images - 1
            # We start the count at 0:
            count = 0
            # We cover each image in the directory:
            for filename in filenames:
                # The count is increased:
                count += 1
                # If it corresponds to the target number, the image is added to the gif:
                if count == sampling_frequency:
                    print("Creating the video - please wait:", str(int(remaining_images)), "image(s) left")
                    image = imageio.imread(str(filename))
                    writer.append_data(image)
                    remaining_images = remaining_images - 1
                    # We reset the count to 0:
                    count = 0
        print("The video has been made!")

    return

# Definition of a function that can create a similar movie for different scenarios' outputs
#-------------------------------------------------------------------------------------------
def resizing_and_film_making_for_scenarios(general_outputs_folder='outputs',
                                           images_folder="root_images",
                                           resized_images_folder="root_images_resided",
                                           scenario_numbers=[1, 2, 3, 4],
                                           film_making=True,
                                           film_name = "root_movie.gif",
                                           image_transforming=True, resizing=False, dividing_size_by=1.,
                                           colorbar_option=True, colorbar_position=1,
                                           colorbar_title="Radius (m)",
                                           colorbar_cmap='jet', colorbar_lognorm=True,
                                           ticks=[],
                                           vmin=1e-6, vmax=1e0,
                                           time_printing=True, time_position=1,
                                           time_step_in_days=1., sampling_frequency=1, frames_per_second=24,
                                           title=""
                                           ):

    """
    This function creates the same type of movie in symetric outputs generated from different scenarios.
    :param general_outputs_folder: the path of the general foleder, in which respective output folders from different scenarios have been recorded
    :param images_folder: the name of the images folder in each scenario
    :param resized_images_folder: the image of the transformed images folder in each scenario
    :param scenario_numbers: a list of numbers corresponding to the different scenarios to consider
    :[other parameters]: [cf the parameters from the function 'resizing_and_film_making']
    :return:
    """

    for i in scenario_numbers:

        scenario_name = 'Scenario_%.4d' % i
        scenario_path = os.path.join(general_outputs_folder, scenario_name)

        print("")
        print("Creating a movie for", scenario_name,"..." )

        resizing_and_film_making(outputs_path=scenario_path,
                                 images_folder=images_folder,
                                 resized_images_folder=resized_images_folder,
                                 film_making=film_making,
                                 film_name=film_name,
                                 sampling_frequency=sampling_frequency, fps=frames_per_second,
                                 time_step_in_days=time_step_in_days,
                                 image_transforming=image_transforming,
                                 time_printing=time_printing, time_position=time_position,
                                 colorbar_option=colorbar_option, colorbar_position=colorbar_position,
                                 colorbar_title=colorbar_title,
                                 colorbar_cmap=colorbar_cmap, colorbar_lognorm=colorbar_lognorm,
                                 ticks=ticks,
                                 vmin=vmin, vmax=vmax,
                                 resizing=resizing, dividing_size_by=dividing_size_by,
                                 title=title)

    return

########################################################################################################################
########################################################################################################################

# MAIN PROGRAM:
###############

if __name__ == "__main__":

    print("Considering creating a video...")

    # Creating a colorbar:
    ######################

    # path_colorbar = os.path.join('outputs', os.path.join('Scenario_0008','colorbar.png'))
    # # We create the colorbar:
    # vmin=1e-15
    # vmax=1e-10
    # bar = colorbar(title="Exudation rate from the phloem vessels (gC per day per cm)",
    #                cmap='jet',
    #                lognorm=True,
    #                n_ticks_for_linear_scale=6,
    #                vmin=vmin, vmax=vmax)
    # # We save it in the output directory:
    # bar.savefig(path_colorbar, facecolor="None", edgecolor="None")

    # main_folder_path = 'C:/Users/frees/rhizodep/saved_outputs/outputs_2024-11/Scenario_0202/'

    # bar_title = "Net rhizodeposition rate (gC per day per cm)"
    # bar_filename = "colorbar_net_rhizodeposition.png"
    # vmin = 1e-8
    # vmax = 1e-4
    # lognorm = True

    # bar_title = "Actual root exchange surface (m2 per cm)"
    # bar_filename = os.path.join(main_folder_path, "colorbar_exchange_surface_per_cm.png")
    # vmin = 1e-4
    # vmax = 8e-4
    # lognorm = False

    # bar_title = "Net sucrose unloading rate (gC per day per cm)"
    # bar_filename = "colorbar_unloading.png"
    # vmin = 1e-8
    # vmax = 1e-4
    # lognorm = True

    # bar = colorbar(title="Root mobile hexose concentration (mol of hexose per gDW of structural mass)", cmap='jet',
    # bar = colorbar(title="Root exchange surface with soil solution (square meter per cm of root)", cmap='jet',
    # bar = colorbar(title="Net unloading rate from phloem (moles of sucrose per cm of root per day)", cmap='jet',

    # bar = colorbar(title=bar_title, cmap='jet', lognorm=lognorm, ticks=[], vmin=vmin, vmax=vmax)

    # bar = colorbar(title=bar_title, cmap='jet', lognorm=lognorm, ticks=[2e-4,4e-4, 6e-4], vmin=vmin, vmax=vmax)

    # We save it in the output directory:
    # bar_name = os.path.join(main_folder_path, "colorbar_C_hexose_root.png")
    # bar_name = os.path.join(main_folder_path, "colorbar_surface.png")
    # bar_name = os.path.join(main_folder_path, "colorbar_unloading.png")
    # bar_name = os.path.join(main_folder_path, "colorbar_rhizodeposition.png")
    # bar_name = os.path.join(main_folder_path, bar_filename)
    # bar.savefig(bar_filename, facecolor="None", edgecolor="None")

    # # Creating a new movie from root systems for one given scenario:
    # ################################################################

    # FROM ORIGINAL GRAPHS - classic:
    resizing_and_film_making(outputs_path='C:/Users/frees/SIMBAL/simulation/outputs',
                             # outputs_path='C:/Users/frees/rhizodep/saved_outputs/outputs_2024-11/Scenario_0185',
                             # outputs_path='C:/Users/frees/SIMBAL/simulation/outputs',
                             # images_folder='root_images_net_rhizodeposition',
                             # images_folder='root_images',
                             images_folder='plots',
                             # images_folder='new_axis_images',
                             # resized_images_folder='root_images_resized',
                             resized_images_folder='plots_resized',
                             # resized_images_folder='new_root_images_resized',
                             film_making=True,
                             film_name="root_movie.gif",
                             image_transforming=True,
                             resizing=False, dividing_size_by=1.,
                             colorbar_option=False, colorbar_position=1,
                             colorbar_title="Net rhizodeposition rate (gC per day per cm)",
                             # colorbar_title=bar_title,
                             # colorbar_cmap='jet', colorbar_lognorm=lognorm,
                             # ticks=[],
                             # vmin=vmin, vmax=vmax,
                             # time_printing=True, time_position=1,
                             time_printing=True,
                             # time_step_in_days=6/24., sampling_frequency=1, fps=24,
                             time_step_in_days=1 / 24., sampling_frequency=1, fps=24,
                             title="")

    # # FROM AXIS GRAPHS:
    # resizing_and_film_making(outputs_path='C:/Users/frees/rhizodep/saved_outputs/outputs_2024-11/Scenario_0202',
    #                          # outputs_path='C:/Users/frees/SIMBAL/simulation/outputs',
    #                          # images_folder='root_images_net_rhizodeposition',
    #                          images_folder='axis_55-65_days_images_surface',
    #                          # images_folder='new_axis_images',
    #                          resized_images_folder='axis_55-65_days_images_resized',
    #                          # resized_images_folder='new_root_images_resized',
    #                          film_making=True,
    #                          film_name="axis_55-65_days_movie.gif",
    #                          image_transforming=True,
    #                          resizing=False, dividing_size_by=1.,
    #                          colorbar_option=True, colorbar_position=3,
    #                          colorbar_title=bar_title,
    #                          colorbar_cmap='jet', colorbar_lognorm=lognorm,
    #                          # ticks=[],
    #                          vmin=vmin, vmax=vmax,
    #                          # time_printing=True, time_position=1,
    #                          time_printing=True,
    #                          # time_step_in_days=6/24., sampling_frequency=1, fps=24,
    #                          time_step_in_days=1 / 24., sampling_frequency=1, fps=24,
    #                          title="")

    # # FROM REDRAWN GRAPHS:
    # resizing_and_film_making(outputs_path=os.path.join('outputs', 'Scenario_0088'),
    #                          # images_folder='root_images',
    #                          images_folder='root_new_images',
    #                          resized_images_folder='root_images_resized',
    #                          film_making=True,
    #                          film_name="root_movie_Rhizodeposition.gif",
    #                          image_transforming=True,
    #                          resizing=False, dividing_size_by=1.,
    #                          colorbar_option=True, colorbar_position=1,
    #                          colorbar_title="Net rhizodeposition rate (mol of hexose per day per cm)",
    #                          # colorbar_title="Hexose concentration (mol of hexose per gram)",
    #                          colorbar_cmap='jet', colorbar_lognorm=True,
    #                          n_ticks_for_linear_scale=6,
    #                          vmin=1e-8, vmax=1e-5,
    #                          time_printing=True, time_position=1,
    #                          time_step_in_days=1., sampling_frequency=1, fps=6,
    #                          title="")

    # # FROM AVERAGED GRAPHS:
    # resizing_and_film_making(outputs_path='C:/Users/frees/rhizodep/simulations/running_scenarios/outputs/Scenario_0100/',
    #                          images_folder='averaged_root_images',
    #                          # images_folder='root_new_images',
    #                          resized_images_folder='root_images_resized',
    #                          film_making=True,
    #                          film_name="root_movie_C_hexose_averaged.gif",
    #                          image_transforming=True,
    #                          resizing=False, dividing_size_by=1.,
    #                          colorbar_option=True, colorbar_position=1,
    #                          # colorbar_title="Net rhizodeposition rate (mol of hexose per day per cm)",
    #                          colorbar_title="Hexose concentration (mol of hexose per gram)",
    #                          colorbar_cmap='jet', colorbar_lognorm=True,
    #                          n_ticks_for_linear_scale=6,
    #                          vmin=1e-6, vmax=1e-3,
    #                          time_printing=True, time_position=1,
    #                          time_step_in_days=1. / 24., sampling_frequency=1, fps=6,
    #                          title="")

    # # Creating a new movie from root systems for a set of scenarios:
    # ################################################################
    # resizing_and_film_making_for_scenarios(general_outputs_folder='outputs',
    #                                        images_folder="root_images",
    #                                        resized_images_folder="root_images_resized",
    #                                        scenario_numbers=[1, 2, 3, 4],
    #                                        film_making=True,
    #                                        film_name = "root_movie.gif",
    #                                        image_transforming=True, resizing=False, dividing_size_by=1.,
    #                                        colorbar_option=True, colorbar_position=1,
    #                                        colorbar_title="Concentration of root mobile hexose (mol of hexose per gram of root)",
    #                                        colorbar_cmap='jet', colorbar_lognorm=True,
    #                                        n_ticks_for_linear_scale=6,
    #                                        vmin=1e-6, vmax=1e0,
    #                                        time_printing=True, time_position=1,
    #                                        time_step_in_days=1., sampling_frequency=24, frames_per_second=24,
    #                                        title="")

    # # Creating a new movie from z-barplots for one scenario:
    # ########################################################
    #
    # outputs_path='C:/Users/frees/rhizodep/saved_outputs/outputs_2024-11/Scenario_0185'
    # resizing_and_film_making(outputs_path=outputs_path,
    #                          images_folder='z_barplots',
    #                          resized_images_folder='z_barplots_resized',
    #                          film_making=True,
    #                          film_name="z_movie.gif",
    #                          image_transforming=True,
    #                          resizing=False, dividing_size_by=1.,
    #                          colorbar_option=False,
    #                          time_printing=True, time_position=2,
    #                          time_step_in_days=1., sampling_frequency=1, fps=12,
    #                          title="")

    # # Creating a new movie from z-barplots for a set of scenarios:
    # ##############################################################
    # resizing_and_film_making_for_scenarios(general_outputs_folder='outputs',
    #                                        images_folder="z_barplots",
    #                                        resized_images_folder="z_barplots_resized",
    #                                        scenario_numbers=[1, 2, 3, 4],
    #                                        film_making=True,
    #                                        film_name = "z_movie.gif",
    #                                        image_transforming=True, resizing=False, dividing_size_by=1.,
    #                                        colorbar_option=False, colorbar_position=1,
    #                                        colorbar_title="Concentration of root mobile hexose (mol of hexose per gram of root)",
    #                                        colorbar_cmap='jet', colorbar_lognorm=True,
    #                                        n_ticks_for_linear_scale=6,
    #                                        vmin=1e-6, vmax=1e0,
    #                                        time_printing=True, time_position=2,
    #                                        time_step_in_days=1., sampling_frequency=1, frames_per_second=24,
    #                                        title="")

    # # For optimizing the GIF (doesn't seem to work as the package can't be installed):
    # from pygifsicle import *
    # gif_path=os.path.join('outputs','Scenario_0001','root_movie.gif')
    # optimize(gif_path) # For creating a new one
