import imageio
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
# from pygifsicle import optimize
from openalea.mtg.plantframe import color
from path import Path
from math import floor, ceil, trunc, log10
import matplotlib.pyplot as plt
import matplotlib as mpl


# Use LaTeX as text renderer to get text in true LaTeX
# If the two following lines are left out, Mathtext will be used
# import matplotlib as mpl
# mpl.rc('text', usetex=True)

########################################################################################################################

# Define function for string formatting of scientific notation
def sci_notation(num, just_print_ten_power=True, decimal_digits=0, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        if num != 0.:
            if num >= 1:
                exponent = int(floor(log10(abs(num))))
            else:
                exponent = int(ceil(log10(abs(num))))
        else:
            exponent = 0
    coeff = round(num / float(10 ** exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    if num == 0:
        return r"${}$".format(0)

    if just_print_ten_power:
        return r"$10^{{{0:d}}}$".format(exponent)
    else:
        return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


# Function that draws a colorbar:
def colorbar(title="Radius (m)", cmap='jet', lognorm=True, n_thicks=5, fmt='%.1e', vmin=1e-12, vmax=1e3):
    """
    """

    # Creating the box:
    fig, ax = plt.subplots(figsize=(36, 6))
    fig.subplots_adjust(bottom=0.5)

    _cmap = color.get_cmap(cmap)

    if lognorm:
        # Adding the color bar:
        norm = color.LogNorm(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax,
                                         cmap=cmap,
                                         norm=norm,
                                         orientation='horizontal')
    else:
        # Adding the color bar:
        norm = color.Normalize(vmin=vmin, vmax=vmax)
        ticks = np.linspace(vmin, vmax, n_thicks)
        cbar = mpl.colorbar.ColorbarBase(ax,
                                         cmap=cmap,
                                         norm=norm,
                                         ticks=ticks,
                                         orientation='horizontal')

    cbar.ax.tick_params(direction="in",  # Position of the ticks in or out the bar
                        labelsize=0,  # Size of the text
                        length=15,  # Length of the ticks
                        width=3,  # Thickness of the ticks
                        pad=-60  # Distance between ticks and label
                        )
    cbar.outline.set_linewidth(3)  # Thickness of the box lines
    cbar.set_label(title, fontsize=40, weight='bold', labelpad=-130)  # Adjust the caption under the bar

    if lognorm:
        # We get the exponents of the powers of 10th closets from vmin and vmax:
        min10 = ceil(np.log10(vmin))
        max10 = floor(np.log10(vmax))
        # We calculate the interval to cover:
        n_intervals = int(abs(max10 - min10))

        # We initialize empty lists:
        list_number = []
        numbers_to_display = []
        x_positions = []

        # We start from the closest power of tenth equal or higher than vmin:
        number = 10 ** min10
        # And we start from a specific position:
        position = 0.0
        # We cover the range from vmin to vmax:
        for i in range(0, n_intervals):
            list_number.append(number)
            x_positions.append(position)
            number = number * 10
            position = position + 1 / float(n_intervals)
        # We correct the first position, if needed:
        x_positions[0] = 0.005

        # We create the list of strings in a scientific format for displaying the numbers on the colorbar:
        for number in list_number:
            # numbers_to_display.append("{:.0e}".format(number))
            numbers_to_display.append(sci_notation(number, just_print_ten_power=True))

        # We cover each number to add on the colorbar:
        for i in range(0, len(numbers_to_display)):
            # # We specify that the coordinates of the last number on the right should corresponding to the right edge:
            # if i == len(numbers_to_display) - 1:
            #     position="right"
            # else:
            #     position="left"
            position = 'left'

            # We add the corresponding number on the colorbar:
            cbar.ax.text(x=x_positions[i],
                         y=0.4,
                         s=numbers_to_display[i],
                         va='top',
                         ha=position,
                         fontsize=50, fontweight='heavy')
    else:
        # We display the vmin and vmax values on left and right sides of the bar:
        cbar.ax.text(x=0.005,
                     y=0.4,
                     s=sci_notation(vmin, just_print_ten_power=False),
                     va='top',
                     ha='left',
                     fontsize=50, fontweight='heavy')
        cbar.ax.text(x=0.995,
                     y=0.4,
                     s=sci_notation(vmax, just_print_ten_power=False),
                     va='top',
                     ha='right',
                     fontsize=50, fontweight='heavy')

    print("The colorbar has been made!")
    return fig


# Definition of a function that can resize a list of images and make a movie from it:
def resizing_and_film_making(image_transforming=True, resizing=True, dividing_size_by=1.,
                             colorbar=True, time_printing=True, time_position=1,
                             film_making=True, film_directory='video',
                             time_step_in_days=1., sampling_frequency=1, fps=24,
                             title=""):
    # Getting a list of the names of the images found in the directory "video":
    filenames = Path(film_directory).glob('*.png')
    filenames = sorted(filenames)

    # We define the final number of images that will be considered, based on the "sampling_frequency" variable:
    number_of_images = floor(len(filenames) / float(sampling_frequency)) + 1

    # 1. COMPRESSING THE IMAGES:
    if image_transforming:
        # We create a new directory "video_resized" that will contain the resized images:
        video_dir = 'video_resized'
        # If this directory doesn't exist:
        if not os.path.exists(video_dir):
            # Then we create it:
            os.mkdir(video_dir)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

        # We modify each image:
        print("Transforming the images and copying them into the directory 'video_resized'...")
        # We initialize the counts:
        number = 0
        count = 0
        remaining_images = number_of_images

        # We calculate the dimensions of the new images according to the variable size_division:
        dimensions = (int(1600 / dividing_size_by), int(1055 / dividing_size_by))

        # We cover each image in the directory:
        for filename in filenames:

            # The time is calculated:
            time_in_days = time_step_in_days * (number_of_images - remaining_images) * sampling_frequency
            # The count is increased:
            count += 1
            # If the count corresponds to the target number, the image is added to the gif:
            if count == sampling_frequency:
                print("Please wait:", str(int(remaining_images)), "image(s) left")

                # Opening the image to modify:
                im = Image.open(filename)

                # Adding colorbar:
                if colorbar:
                    path_colorbar = os.path.join('C:/', 'Users', 'frees', 'rhizodep', 'test', 'colorbar.png')
                    colorbar = Image.open(path_colorbar)
                    new_size = (1200, 200)
                    colorbar = colorbar.resize(new_size)
                    # box_colorbar = (-120,1070)
                    box_colorbar = (-120, 870)
                    im.paste(colorbar, box_colorbar, colorbar.convert('RGBA'))

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
                        font_time = ImageFont.truetype("./timesbd.ttf", 20)
                        # See a list of available fonts on: https://docs.microsoft.com/en-us/typography/fonts/windows_10_font_list
                        (x1, y1) = (650, 420)
                        time_text = "t = " + str(int(floor(time_in_days))) + " days"
                        draw.rectangle((x1 - 10, y1 - 10, x1 + 200, y1 + 30), fill=(255, 255, 255, 0))
                        draw.text((x1, y1), time_text, fill=(0, 0, 0), font=font_time)
                        # draw.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

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
                # Saving the new image:
                image_name = os.path.join(video_dir, 'root%.4d.png')
                im_to_print.save(image_name % number, quality=20, optimize=True)

                # We update the local counts:
                number = number + 1
                remaining_images = remaining_images - 1
                count = 0
        print("The new images have been transformed!")

    # 2. CREATING THE VIDEO FILE:
    if film_making:

        print("Making the video...")

        with imageio.get_writer('root_movie.gif', mode='I', fps=fps) as writer:
            if image_transforming:
                filenames = Path('video_resized').glob('*.png')
                filenames = sorted(filenames)
                sampling_frequency = 1
            else:
                filenames = Path(film_directory).glob('*.png')
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
                    print("Please wait:", str(int(remaining_images)), "image(s) left")
                    image = imageio.imread(str(filename))
                    writer.append_data(image)
                    remaining_images = remaining_images - 1
                    # We reset the count to 0:
                    count = 0
        print("The video has been made!")


########################################################################################################################

# MAIN PROGRAM:
################

if __name__ == "__main__":

    # We set the working directory:
    my_path = r'C:\\Users\\frees\\rhizodep\\test'
    if not os.path.exists(my_path):
        my_path = os.path.abspath('.')
    os.chdir(my_path)
    print("The current directory is:", os.getcwd())

    # # Creating a new colobar:
    # fig = colorbar(
    #     # title=" Net exudation rate (mol of hexose per day per cm)",
    #     title="Net exudation rate (mol of hexose per day per cm of root)",
    #     vmin=1e-9, vmax=1e-6,lognorm=True, cmap='jet') #, n_thicks=3 # n_thicks is only used when lognorm=False!
    #     # cmap='gist_rainbow')
    #     # cmap='jet')
    # fig.savefig('colorbar.png', facecolor="None", edgecolor="None")
    # print("The colorbar has been saved in the current directory.")

    # Creating a new colobar:
    fig = colorbar(
        # title=" Net exudation rate (mol of hexose per day per cm)",
        title="Concentration of root mobile hexose (mol of hexose per gram of root)",
        vmin=1e-4, vmax=1e-1, lognorm=True, cmap='jet')  # , n_thicks=3 # n_thicks is only used when lognorm=False!
    # cmap='gist_rainbow')
    # cmap='jet')
    fig.savefig('colorbar.png', facecolor="None", edgecolor="None")
    print("The colorbar has been saved in the current directory.")

    # Creating a new movie from root systems:
    resizing_and_film_making(film_making=True, film_directory='video', sampling_frequency=1, fps=24,
                             time_step_in_days=1 / 24.,
                             image_transforming=True, time_printing=True, time_position=1, colorbar=True,
                             resizing=False, dividing_size_by=1,
                             title="")

    # # Creating a new movie from z-barplots:
    # resizing_and_film_making(film_making=True, film_directory='z_barplots', sampling_frequency=1, fps=10,
    #                          time_step_in_days=1.,
    #                          image_transforming=True, time_printing=True, time_position=2, colorbar=False,
    #                          resizing=False, dividing_size_by=1,
    #                          title="")

    # # We set the working directory:
    # my_path = r'C:\\Users\\frees\\rhizodep\\test'
    # if not os.path.exists(my_path):
    #     my_path = os.path.abspath('.')
    # os.chdir(my_path)
    # print("The current directory is:", os.getcwd())
    #
    # # For optimizing the GIF (doesn't seem to work as the package can't be installed):
    # from pygifsicle import optimize
    #
    # gif_path=os.path_join(my_path,'root_movie.gif')
    # optimize(gif_path, "optimized.gif") # For creating a new one
