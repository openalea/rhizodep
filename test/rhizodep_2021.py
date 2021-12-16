# Todo: Introduce explicit threshold concentration for limiting processes, rather than "0"
# Todo: Consider giving priority to root maintenance, and provoque senescence when root maintenance is not ensured
# Todo: Watch the calculation of surface and volume for the apices - if they correspond to cones, the mass balance for segmentation may not be correct!
# Todo: Include cell sloughing and mucilage production
# Todo: Make a distinction between seminal roots and adventitious roots
# Todo: Check actual_elongation_rate with thermal time...

# Importation of functions from the system:
###########################################

from math import sqrt, pi, trunc, floor, cos, sin
from decimal import Decimal
import numpy as np
import pandas as pd
import os
import os.path
import timeit

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl

import pickle

# Setting the randomness in the whole code to reproduce the same root system over different runs:
# random_choice = int(round(np.random.normal(100,50)))
random_choice = 8
print("The random seed used for this run is", random_choice)
np.random.seed(random_choice)

########################################################################################################################
########################################################################################################################
# LIST OF PARAMETERS
########################################################################################################################
########################################################################################################################

# ArchiSimple parameters for root growth:
# ---------------------------------------

# Initial tip diameter of the primary root (in m):
D_ini = 0.80 / 1000.
# => Reference: Di=0.5 mm

# Proportionality coefficient between the tip diameter of an adventitious root and D_ini (dimensionless):
D_ini_to_D_adv_ratio = 0.8

# Threshold tip diameter, below which there is no elongation (i.e. the diameter of the finest observable roots)(in m):
Dmin = 0.35 / 1000.
# => Reference: Dmin=0.05 mm

# Slope of the potential elongation rate versus tip diameter (in m m-1 s-1):
EL = 1.39 * 20 / (60. * 60. * 24.)
# => Reference: EL = 5 mm mm-1 day-1

# Proportionality coefficient between the section area of the segment and the sum of distal section areas
# (dimensionless, 0 = no radial growth):
SGC = 0.2

# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
RMD = 0.638

# Relative variation of the daughter root diameter (dimensionless):
CVDD = 0.2

# Delay of emergence of the primordium (in s):
emergence_delay = 158 / 20. * (60. * 60. * 24.)
# => Reference: emergence_delay = 3 days

# Inter-primordium distance (in m):
IPD = 5.11 / 1000.
# => Reference: IPD = 7.6 mm

# Maximal number of roots emerging from the base (including primary and seminal roots)(dimensionless):
n_adventitious_roots = 0

# Emission rate of adventitious roots (in s-1):
ER = 1.0 / (60. * 60. * 24.)
# => Reference: ER = 0.5 day-1

# Coefficient of growth duration (in s m-2):
GDs = 800 * (60. * 60. * 24.) * 1000. ** 2.
# => Reference: GDs=400. day mm-2

# Coefficient of the life duration (in s m g-1 m3):
LDs = 4000. * (60. * 60. * 24.) * 1000 * 1e-6
# => Reference: LDs = 5000 day mm-1 g-1 cm3

# Root tissue density (in g m-3):
root_tissue_density = 0.10 * 1e6
# => Reference: RTD=0.1 g cm-3

# C content of struct_mass (mol of C per g of struct_mass):
struct_mass_C_content = 0.44 / 12.01
# => We assume that the structural mass contains 44% of C.

# Gravitropism (dimensionless):
gravitropism_coefficient = 0.06
# tropism_intensity = 2e6 # Value between 0 and 1.
# tropism_direction = (0,0,-1) # Force of the tropism

# Length of a segment (in m):
segment_length = 3. / 1000.

# Parameters for temperature adjustments:
# ---------------------------------------
# Reference temperature for growth, i.e. at which the relative effect of temperature is 1 (in degree Celsius):
T_ref_growth = 20.
# => We assume that the parameters for root growth were obtained for T = 20 degree Celsius

# Proportionality coefficient between growth and temperature (in unit of growth per degree Celsius)
# - note that, when this parameter is equal to 0, temperature will have no influence on growth:
growth_increase_with_temperature = 1 / T_ref_growth
# => We suggest to calculate this parameter so that relative growth is 0 at 0 degree Celsius and 1 at reference temperature.

# Parameters for growth respiration:
# -----------------------------------
# Growth yield (in mol of CO2 per mol of C used for struct_mass):
yield_growth = 0.8
# => We use the range value (0.75-0.85) proposed by Thornley and Cannell (2000)

# Parameters for maintenance respiration:
# ----------------------------------------
# Maximal maintenance respiration rate (in mol of CO2 per g of struct_mass per s):
resp_maintenance_max = 4.1e-6 / 6. * (0.44 / 60. / 14.01) * 5
# => According to Barillot et al. (2016): max. residual maintenance respiration rate is 4.1e-6 umol_C umol_N-1 s-1,
# i.e. 4.1e-6/60*0.44 mol_C g-1 s-1 assuming an average struct_C:N_tot molar ratio of root of 60 [cf simulations by CN-Wheat 47 in 2020]
# and a C content of struct_mass of 44%. And according to the same simulations, total maintenance respiration is about
# 5 times more than residual maintenance respiration.

# Affinity constant for maintenance respiration (in mol of hexose per g of struct_mass):
Km_maintenance = 1.67e3 * 1e-6 / 6.
# => According to Barillot et al. (2016): Km=1670 umol of C per gram for residual maintenance respiration (based on sucrose concentration!)

# Expected concentrations in a typical root segment (used for estimating a few parameters):
# ------------------------------------------------------------------------------------------
# Expected sucrose concentration in root (in mol of sucrose per g of root):
expected_C_sucrose_root = 0.01 / 12.01 / 12
# => 0.0025 is a plausible value according to the results of Gauthier (2019, pers. communication), but here, we use
# a plausible sucrose concentration (10 mgC g-1) in roots according to various experimental results.

# Expected hexose concentration in root (in mol of hexose per g of root):
expected_C_hexose_root = 0.01 / 12.01 / 6
# => Here, we use a plausible hexose concentration in roots (10 mgC g-1) according to various experimental results.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
expected_C_hexose_root = 1e-3

# Expected hexose concentration in soil (in mol of hexose per g of root):
expected_C_hexose_soil = expected_C_hexose_root / 100.
# => We expect the soil concentration to be 2 orders of magnitude lower than the root concentration.

# Expected hexose concentration in the reserve pool (in mol of hexose per g of reserve pool):
expected_C_hexose_reserve = expected_C_hexose_root * 2.
# => We expect the reserve pool to be two times higher than the mobile one.

# Parameters for estimation the surfaces of exchange between root symplasm, apoplasm and  phloem:
# ------------------------------------------------------------------------------------------------

# Phloem surface as a fraction of the root external surface (in m2 of phloem surface per m2 of root external surface):
phloem_surfacic_fraction = 1.

# Symplasm surface as a fraction of the root external surface (in m2 of symplasm surface per m2 of root external surface):
symplasm_surfacic_fraction = 10.

# Parameters for phloem unloading/reloading:
# ----------------------------------
# Maximum unloading rate of sucrose from the phloem through the surface of the phloem vessels (in mol of sucrose per m2 per second):
surfacic_unloading_rate_reference = 0.03 / 12 * 1e-6 / (0.5 * phloem_surfacic_fraction)
# => According to Barillot et al. (2016b), this value is 0.03 umol C g-1 s-1, and we assume that 1 gram of dry root mass
# is equivalent to 0.5 m2 of surface. However, the unloading of sucrose is not done exactly the same in CN-Wheat as in RhizoDep (
# for example, exudation is a fraction of the sucrose unloading, unlike in RhizoDep where it is a fraction of hexose).
# CHEATING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
surfacic_unloading_rate_reference = surfacic_unloading_rate_reference * 200.

# Maximum unloading rate of sucrose from the phloem through the section of a primordium (in mol of sucrose per m2 per second):
surfacic_unloading_rate_primordium = surfacic_unloading_rate_reference * 3.
# => We expect the maximum unloading rate through an emerging primordium to be xxx times higher than the usual unloading rate

# ALTERNATIVE: We use a permeability coefficient and unloading occurs through diffusion only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Coefficient of permeability of unloading phloem (in gram per m2 per second):
phloem_permeability = 2e-4
# => Quite arbitrary value...

# gamma_unloading = 0.3
# # Arbitrary

# Maximum reloading rate of hexose inside the phloem vessels (in mol of hexose per m2 per second):
surfacic_loading_rate_reference = surfacic_unloading_rate_reference * 2.
# => We expect the maximum loading rate of hexose to be equivalent to the maximum unloading rate.

# Affinity constant for sucrose unloading (in mol of sucrose per g of struct_mass):
Km_unloading = 1000 * 1e-6 / 12.
# => According to Barillot et al. (2016b), this value is 1000 umol C g-1

# Affinity constant for sucrose loading (in mol of hexose per g of struct_mass):
Km_loading = Km_unloading * 2.
# => We expect the Km of loading to be equivalent as the Km of unloading, as it may correspond to the same transporter.

gamma_loading = 0.0

# Parameters for mobilization/immobilization in the reserve pool:
# ----------------------------------------------------------------

# Maximum concentration of hexose in the reserve pool (in mol of hexose per gram of structural mass):
C_hexose_reserve_max = 5e-3
# Minimum concentration of hexose in the reserve pool (in mol of hexose per gram of structural mass):
C_hexose_reserve_min = 0.
# Minimum concentration of hexose in the mobile pool for immobilization (in mol of hexose per gram of structural mass):
C_hexose_root_min_for_reserve = 5e-3

# Maximum immobilization rate of hexose in the reserve pool (in mol of hexose per gram of structural mass per second):
max_immobilization_rate = 8 * 1e-6 / 6.
# => According to Gauthier et al. (2020): the new maximum rate of starch synthesis for vegetative growth in the shoot is 8 umolC g-1 s-1

# Affinity constant for hexose immobilization in the reserve pool (in mol of hexose per gram of structural mass):
Km_immobilization = expected_C_hexose_root * 1.

# Maximum mobilization rate of hexose from the reserve pool (in mol of hexose per gram of structural mass per second):
max_mobilization_rate = max_immobilization_rate

# Affinity constant for hexose remobilization from the reserve (in mol of hexose per gram of structural mass):
Km_mobilization = expected_C_hexose_reserve * 5.

# Parameters for root hexose exudation:
# --------------------------------------
# Expected exudation rate (in mol of hexose per m2 per s):
expected_exudation_efflux = 608 * 0.000001 / 12.01 / 6 / 3600 * 1 / (0.5 * 10)

# => According to Jones and Darrah (1992): the net efflux of C for a root of maize is 608 ug C g-1 root DW h-1,
# and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface.
# OR:
# expected_exudation_efflux = 5.2 / 12.01 / 6. * 1e-6 * 100. ** 2. / 3600.
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour

# Permeability coefficient (in g of struct_mass per m2 per s):
Pmax_apex = expected_exudation_efflux / (expected_C_hexose_root - expected_C_hexose_soil)
# => We calculate the permeability according to the expected exudation flux and expected concentration gradient between cytosol and soil.

# Coefficient affecting the decrease of permeability with distance from the apex (dimensionless):
gamma_exudation = 0.4
# => According to Personeni et al (2007), this gamma coefficient showing the decrease in permeability along the root is 0.41-0.44

# Parameters for root hexose uptake from soil:
# ---------------------------------------------
# Maximum rate of influx of hexose from soil to roots (in mol of hexose per m2 per s):
# uptake_rate_max = 0.3/3600.*3*0.000001/12.01/6.*10000
# => According to Personeni et al (2007), the uptake coefficient beta used in the relationship Uptake = beta*S*Ce should be
# 0.20-0.38 cm h-1 with concentrations in the external solution of about 3 ugC cm-3.
# OR:
uptake_rate_max = 277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * symplasm_surfacic_fraction)
# => According to Jones and Darrah (1996), the uptake rate measured for all sugars tested with an individual external concentration
# of 100 uM is equivalent to 277 nmol hexose mg-1 day-1, and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface.

# Affinity constant for hexose uptake (in mol of hexose per g of struct_mass):
Km_uptake = expected_C_hexose_soil * 10
# => We assume that half of the max influx is reached when the soil concentration equals 10 times the expected root concentration
# CHEATING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Km_uptake = Km_loading

# Parameters for soil hexose degradation:
# ----------------------------------------
# Maximum degradation rate of hexose in soil (in mol of hexose per m2 per s):
degradation_rate_max = uptake_rate_max * 10.
# => We assume that the maximum degradation rate is 10 times higher than the maximum uptake rate

# Affinity constant for soil hexose degradation (in mol of hexose per g of struct_mass):
Km_degradation = Km_uptake / 2.
# => According to what Jones and Darrah (1996) suggested, we assume that this Km is 2 times lower than the Km corresponding
# to root uptake of hexose (350 uM against 800 uM in the original article).

# Parameters for C-controlled growth rates:
# -----------------------------------------

# Proportionality factor between the radius and the length of the root apical zone in which C can be used to sustain root elongation:
growing_zone_factor = 8  # m m-1
# => According to illustrations by by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap,
# meristem and elongation zones is about 8 times the diameter of the tip.

# Affinity constant for root elongation (in mol of hexose per g of struct_mass):
Km_elongation = 1250 * 1e-6 / 6.
# => According to Barillot et al. (2016b): Km for root growth is 1250 umol C g-1 for sucrose
# => According to Gauthier et al (2020): Km for regulation of the RER by sucrose concentration in hz = 100-150 umol C g-1
# CHEATING:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Km_elongation = Km_elongation*10
# Km_elongation = Km_loading

# Affinity constant for root thickening (in mol of hexose per g of struct_mass):
Km_thickening = Km_elongation
# => We assume that the Michaelis-Menten constant for thickening is the same as for root elongation.
# Maximal rate of relative increase in root radius (in m per m per s):
relative_root_thickening_rate_max = 5. / 100. / (24. * 60. * 60.)
# => We consider that the radius can't increase by more than 5% every day

# We define a probability (between 0 and 1) of nodule formation for each apex that elongates:
nodule_formation_probability = 0.5
# Maximal radius of nodule (in m):
nodule_max_radius = D_ini * 20.
# Affinity constant for nodule thickening (in mol of hexose per g of struct_mass):
Km_nodule_thickening = Km_elongation * 100
# Maximal rate of relative increase in nodule radius (in m per m per s):
relative_nodule_thickening_rate_max = 20. / 100. / (24. * 60. * 60.)


# => We consider that the radius can't increase by more than 20% every day

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
        turtle.elasticity = gravitropism_coefficient * (n.original_radius / g.node(1).original_radius)
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


# FUNCTIONS FOR CALCULATING PROPERTIES ON THE MTG
#################################################

# Defining the surface of each root element in contact with the soil:
# -------------------------------------------------------------------
def surfaces_and_volumes(element, radius, length):
    """
    The function "surfaces_and_volumes" computes different surfaces (m2) and volumes (m3) of a root element,
    based on the properties radius (m) and length (m).
    :param element: the investigated node of the MTG
    :param radius: the radius of the root element (m)
    :param length: the length of the root element (m)
    :return: a dictionary containing the calculated surfaces and volumes of the given element
    """

    n = element
    vid = n.index()
    number_of_children = n.nb_children()

    # CALCULATIONS OF EXTERNAL SURFACE AND VOLUME:
    # If the root element corresponds to an apex or a segment without lateral roots:
    if number_of_children == 0 or number_of_children == 1:
        external_surface = 2 * pi * radius * length
        volume = pi * radius ** 2 * length
    # Otherwise there is one or more lateral roots branched on the root segment:
    else:
        # So we sum all the sections of the lateral roots branched on the root segment:
        sum_ramif_sections = 0
        for child_vid in g.Sons(vid, EdgeType='+'):
            son = g.node(child_vid)
            # We avoid to remove the section of the sphere of a nodule:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if son.type != "Root_nodule":
                sum_ramif_sections += pi * son.radius ** 2
        # And we subtract this sum of sections from the external area of the main cylinder:
        external_surface = 2 * pi * radius * length - sum_ramif_sections
        volume = pi * radius ** 2 * length

    # SPECIAL CASE FOR NODULE:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if n.type == "Root_nodule":
        # We consider the surface and volume of a sphere:
        external_surface = 4 * pi * radius ** 2
        volume = 4 / 3. * pi * radius ** 3

    # CALCULATIONS OF THE TOTAL EXCHANGE SURFACE OF PHLOEM VESSELS:
    phloem_surface = phloem_surfacic_fraction * external_surface

    # CALCULATIONS OF THE TOTAL EXCHANGE SURFACE BETWEEN ROOT SYMPLASM AND APOPLASM:
    symplasm_surface = symplasm_surfacic_fraction * external_surface

    # CREATION OF A DICTIONARY THAT WILL BE USED TO RECORD THE OUTPUTS:
    dictionary = {"external_surface": external_surface,
                  "volume": volume,
                  "phloem_surface": phloem_surface,
                  "symplasm_surface": symplasm_surface
                  }

    return dictionary


# Defining the distance of a vertex from the tip:
# -----------------------------------------------
def dist_to_tip(g):
    """
    The function "dist_to_tip" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
    :param g: the investigated MTG
    :return: the MTG with an updated property 'dist_to_tip'
    """

    # We initialize an empty dictionary for to_tips:
    to_tips = {}
    # We use the property "length" of each vertex based on the function "length":
    length = g.property('length')

    # We define "root" as the starting point of the loop below (i.e. the first apex in the MTG)(?):
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # If the vertex corresponds to a root apex:
        vertex = g.node(vid)
        try:
            # We get the vertex ID of the successor of the root segment in the same root axis:
            son_id = g.Successor(vid)
            # And we calculate the new distance from the tip by adding the distance of the successor and its length:
            to_tips[vid] = to_tips[son_id] + length[son_id]
        except:
            # If there is no successor because the element is an apex or root nodule
            to_tips[vid] = 0.
        # if vertex.label=="Apex" or vertex.type=="Root_nodule":
        #     # Then the distance is 0 meter by definition:
        #     to_tips[vid] = 0.
        # else:
        #     try:
        #         # Else we get the vertex ID of the successor of the root segment in the same root axis:
        #         son_id = g.Successor(vid)
        #         # And we calculate the new distance from the tip by adding the distance of the successor and its length:
        #         to_tips[vid] = to_tips[son_id] + length[son_id]
        #     except:
        #         print "When calculating dist_to_tip for element", vid, "of type", vertex.type, "the ID of the son is", son_id
        #         to_tips[vid] = 0.

        # We assign the result "dist_to_tip" as a new property of each vertex in the MTG "g":
    g.properties()['dist_to_tip'] = to_tips

    # We return a modified version of the MTG "g" with a new property "dist_to_tip":
    return g


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
def classifying_on_z(g, z_min=0., z_max=1., z_interval=0.1):
    # We initialize empty dictionaries:
    included_length = {}
    dictionary_length = {}
    dictionary_struct_mass = {}
    dictionary_root_necromass = {}
    dictionary_surface = {}
    dictionary_net_hexose_exudation = {}
    dictionary_hexose_degradation = {}
    final_dictionary = {}

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

        # We record the summed values for this interval of z in several dictionaries:
        dictionary_length[name_length_z] = total_included_length
        dictionary_struct_mass[name_struct_mass_z] = total_included_struct_mass
        dictionary_root_necromass[name_root_necromass_z] = total_included_root_necromass
        dictionary_surface[name_surface_z] = total_included_surface
        dictionary_net_hexose_exudation[name_net_hexose_exudation_z] = total_included_net_hexose_exudation
        dictionary_hexose_degradation[name_hexose_degradation_z] = total_included_hexose_degradation

        # We also create a new property of the MTG that corresponds to the fraction of length of each node in the z interval:
        g.properties()[name_length_z] = included_length

    # Finally, we merge all dictionaries into a single one that will be returned by the function:
    final_dictionary = {}
    for d in [dictionary_length, dictionary_struct_mass, dictionary_root_necromass, dictionary_surface, dictionary_net_hexose_exudation,
              dictionary_hexose_degradation]:
        final_dictionary.update(d)

    return final_dictionary


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


# Modification of a process according to soil temperature:
# --------------------------------------------------------
def temperature_modification(temperature_in_Celsius, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
    """
    This function calculates how the value of a process should be modified according to soil temperature (in degrees Celsius).
    Parameters correspond the the value of the process at reference temperature T_ref (process_at_T_ref),
    to two empirical coefficients A and B, and to a coefficient C used to switch between different formalisms.
    If C=0 and B=1, then the relationship corresponds to a classical linear increase with temperature (thermal time).
    If C=1, A=0 and B>1, then the relationship corresponds to a classical exponential increase with temperature (Q10).
    If C=1, A<0 and B>0, then the relationship corresponds to bell-shaped curve, close to the one from Parent et al. (2010).
    """

    # We initialize the value of the temperature-modified process:
    modified_process = 0.

    # We avoid unwanted cases:
    if C != 0 and C != 1:
        print("The modification of the process at T =", temperature_in_Celsius, "only works for C=0 or C=1!")
        print("The modified process has been set to 0.")
        modified_process = 0.
        return 0.
    elif C == 1:
        if (A * (temperature_in_Celsius - T_ref) + B) < 0.:
            print("The modification of the process at T =", temperature_in_Celsius, "is unstable with this set of parameters!")
            print("The modified process has been set to 0.")
            modified_process = process_at_T_ref
            return modified_process

    # We compute a temperature-modified process, correspond to a Q10-modified relationship,
    # based on the work of Tjoelker et al. (2001):
    modified_process = process_at_T_ref * (A * (temperature_in_Celsius - T_ref) + B) ** (1 - C) \
                       * (A * (temperature_in_Celsius - T_ref) + B) ** (C * (temperature_in_Celsius - T_ref) / 10.)

    if modified_process < 0.:
        modified_process = 0.

    return modified_process


# Adding a new root element with pre-defined properties:
# ------------------------------------------------------
def ADDING_A_CHILD(mother_element, edge_type='+', label='Apex', type='Normal_root_before_emergence',
                   angle_down=45., angle_roll=0., length=0., radius=0.,
                   identical_properties=True, nil_properties=False):
    """
    This function creates a new child element on the mother element, based on the function add_child.
    When called, this allows to automatically define standard properties without defining them in the main code.
    :param edge_type:
    :return:
    """

    # If nil_properties = True, then we set most of the properties of the new element to 0:
    if nil_properties:

        new_child = mother_element.add_child(edge_type=edge_type,
                                             # Characteristics:
                                             # -----------------
                                             label=label,
                                             type=type,
                                             # Authorizations and C requirements:
                                             # -----------------------------------
                                             lateral_emergence_possibility='Impossible',
                                             emergence_cost=0.,
                                             # Geometry and topology:
                                             # -----------------------
                                             angle_down=angle_down,
                                             angle_roll=angle_roll,
                                             # The length of the primordium is set to 0:
                                             length=length,
                                             radius=radius,
                                             original_radius=radius,
                                             potential_length=0.,
                                             theoretical_radius=radius,
                                             potential_radius=radius,
                                             initial_length=0.,
                                             initial_radius=radius,
                                             external_surface=0.,
                                             volume=0.,

                                             dist_to_ramif=0.,
                                             dist_to_tip=0.,
                                             actual_elongation=0.,
                                             adventitious_emerging_primordium_index=0,
                                             # Quantities and concentrations:
                                             # -------------------------------
                                             struct_mass=0.,
                                             initial_struct_mass=0.,
                                             C_hexose_root=0.,
                                             C_hexose_reserve=0.,
                                             C_hexose_soil=0.,
                                             C_sucrose_root=0.,
                                             Deficit_hexose_root=0.,
                                             Deficit_hexose_soil=0.,
                                             Deficit_sucrose_root=0.,
                                             # Fluxes:
                                             # --------
                                             resp_maintenance=0.,
                                             resp_growth=0.,
                                             struct_mass_produced=0.,
                                             hexose_growth_demand=0.,
                                             hexose_consumption_by_growth=0.,
                                             hexose_production_from_phloem=0.,
                                             sucrose_loading_in_phloem=0.,
                                             hexose_mobilization_from_reserve=0.,
                                             hexose_immobilization_as_reserve=0.,
                                             hexose_exudation=0.,
                                             hexose_uptake=0.,
                                             hexose_degradation=0.,
                                             specific_net_exudation=0.,
                                             # Time indications:
                                             # ------------------
                                             growth_duration=GDs * radius ** 2 * 4,
                                             life_duration=LDs * 2. * radius * root_tissue_density,
                                             actual_time_since_primordium_formation=0.,
                                             actual_time_since_emergence=0.,
                                             actual_potential_time_since_emergence=0.,
                                             actual_time_since_growth_stopped=0.,
                                             actual_time_since_death=0.,
                                             thermal_time_since_primordium_formation=0.,
                                             thermal_time_since_emergence=0.,
                                             thermal_potential_time_since_emergence=0.,
                                             thermal_time_since_growth_stopped=0.,
                                             thermal_time_since_death=0.
                                             )

    # Otherwise, if identical_properties=True, then we copy most of the properties of the mother element in the new element:
    elif identical_properties:

        new_child = mother_element.add_child(edge_type=edge_type,
                                             # Characteristics:
                                             # -----------------
                                             label=label,
                                             type=type,
                                             # Authorizations and C requirements:
                                             # -----------------------------------
                                             lateral_emergence_possibility='Impossible',
                                             emergence_cost=0.,
                                             # Geometry and topology:
                                             # -----------------------
                                             angle_down=angle_down,
                                             angle_roll=angle_roll,
                                             # The length of the primordium is set to 0:
                                             length=length,
                                             radius=mother_element.radius,
                                             original_radius=mother_element.radius,
                                             potential_length=length,
                                             theoretical_radius=mother_element.theoretical_radius,
                                             potential_radius=mother_element.potential_radius,
                                             initial_length=length,
                                             initial_radius=mother_element.radius,
                                             external_surface=0.,
                                             volume=0.,

                                             dist_to_ramif=mother_element.dist_to_ramif,
                                             dist_to_tip=mother_element.dist_to_tip,
                                             actual_elongation=mother_element.actual_elongation,
                                             adventitious_emerging_primordium_index=mother_element.adventitious_emerging_primordium_index,
                                             # Quantities and concentrations:
                                             # -------------------------------
                                             struct_mass=mother_element.struct_mass,
                                             initial_struct_mass=mother_element.initial_struct_mass,
                                             C_hexose_root=mother_element.C_hexose_root,
                                             C_hexose_reserve=mother_element.C_hexose_reserve,
                                             C_hexose_soil=mother_element.C_hexose_soil,
                                             C_sucrose_root=mother_element.C_sucrose_root,
                                             Deficit_hexose_root=mother_element.Deficit_hexose_root,
                                             Deficit_hexose_soil=mother_element.Deficit_hexose_soil,
                                             Deficit_sucrose_root=mother_element.Deficit_sucrose_root,
                                             # Fluxes:
                                             # -------
                                             resp_maintenance=mother_element.resp_maintenance,
                                             resp_growth=mother_element.resp_growth,
                                             struct_mass_produced=mother_element.struct_mass_produced,
                                             hexose_growth_demand=mother_element.hexose_growth_demand,
                                             hexose_production_from_phloem=mother_element.hexose_production_from_phloem,
                                             sucrose_loading_in_phloem=mother_element.sucrose_loading_in_phloem,
                                             hexose_mobilization_from_reserve=mother_element.hexose_mobilization_from_reserve,
                                             hexose_immobilization_as_reserve=mother_element.hexose_immobilization_as_reserve,
                                             hexose_exudation=mother_element.hexose_exudation,
                                             hexose_uptake=mother_element.hexose_uptake,
                                             hexose_degradation=mother_element.hexose_degradation,
                                             hexose_consumption_by_growth=mother_element.hexose_consumption_by_growth,
                                             specific_net_exudation=mother_element.specific_net_exudation,
                                             # Time indications:
                                             # ------------------
                                             growth_duration=mother_element.growth_duration,
                                             life_duration=mother_element.life_duration,
                                             actual_time_since_primordium_formation=mother_element.actual_time_since_primordium_formation,
                                             actual_time_since_emergence=mother_element.actual_time_since_emergence,
                                             actual_time_since_growth_stopped=mother_element.actual_time_since_growth_stopped,
                                             actual_time_since_death=mother_element.actual_time_since_death,

                                             thermal_time_since_primordium_formation=mother_element.thermal_time_since_primordium_formation,
                                             thermal_time_since_emergence=mother_element.thermal_time_since_emergence,
                                             thermal_potential_time_since_emergence=mother_element.thermal_potential_time_since_emergence,
                                             thermal_time_since_growth_stopped=mother_element.thermal_time_since_growth_stopped,
                                             thermal_time_since_death=mother_element.thermal_time_since_death
                                             )

    return new_child


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "POTENTIAL GROWTH"
########################################################################################################################
########################################################################################################################

# A FUNCTION FOR CALCULATING THE NEW LENGTH AFTER ELONGATION:
##########################################################

def elongated_length(initial_length=0., radius=0., C_hexose_root=expected_C_hexose_root, elongation_time_in_seconds=0.,
                     ArchiSimple=True, printing_warnings=False, soil_temperature_in_Celsius=20):
    """

    :param initial_length:
    :param radius:
    :param C_hexose_root:
    :param elongation_time_in_seconds:
    :param ArchiSimple:
    :param printing_warnings:
    :return:
    """

    # If we keep the classical ArchiSimple rule:
    if ArchiSimple:
        # Then the elongation is calculated following the rules of Pages et al. (2014):
        elongation = EL * 2. * radius * elongation_time_in_seconds
    else:
        # Otherwise, we additionnaly consider a limitation of the elongation according to the local concentration of hexose,
        # based on a Michaelis-Menten formalism:
        if C_hexose_root > 0.:
            elongation = EL * 2. * radius * C_hexose_root / (Km_elongation + C_hexose_root) * elongation_time_in_seconds
        else:
            elongation = 0.

    # # We add a correction factor for temperature - NOT TO DO IF THE TIME IS ALREADY ADJUSTED TO TEMPERATURE:
    # elongation = elongation * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
    #                                                    process_at_T_ref=1.,
    #                                                    T_ref=20,
    #                                                    A=0.05,
    #                                                    B=1,
    #                                                    C=0)

    # We calculate the new potential length corresponding to this elongation:
    new_length = initial_length + elongation
    if new_length < initial_length:
        print("!!! ERROR: There is a problem of elongation, with the initial length", initial_length,
              " and the radius", radius, "and the elongation time", elongation_time_in_seconds)
    return new_length


# FUNCTION FOR FORMING A PRIMORDIUM ON AN APEX
###############################################

def primordium_formation(apex, elongation_rate=0., time_step_in_seconds=1. * 60. * 60. * 24.,
                         soil_temperature_in_Celsius=20, random=False):
    """

    :param apex:
    :param elongation_rate:
    :param time_step_in_seconds:
    :param random:
    :return:
    """

    # NOTE: This function has to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=T_ref_growth,
                                                           A=growth_increase_with_temperature,
                                                           B=1,
                                                           C=0)

    # OPERATING PRIMORDIUM FORMATION:
    # --------------------------------
    # We initialize the new_apex that will be returned by the function:
    new_apex = []

    # VERIFICATION: We make sure that no lateral root has already been form on the present apex.
    # We calculate the number of children of the apex (it should be 0!):
    n_children = len(apex.children())
    # If there is at least one children, it means that there is already one lateral primordium or root on the apex:
    if n_children >= 1:
        # Then we don't add any primordium and simply return the unaltered apex:
        new_apex.append(apex)
        return new_apex

    # We get the order of the current root axis:
    vid = apex.index()
    order = g.order(vid)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # If the order of the current apex is too high, we forbid the formation of a new primordium of higher order:
    if order >= 2:
        # Then we don't add any primordium and simply return the unaltered apex:
        new_apex.append(apex)
        return new_apex

    # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
    # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
    # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014).
    # We also set the root angles depending on random:
    if random:
        potential_radius = np.random.normal((apex.radius - Dmin / 2.) * RMD + Dmin / 2.,
                                            ((apex.radius - Dmin) * RMD + Dmin / 2.) * CVDD)
        apex_angle_roll = abs(np.random.normal(120, 10))
        if order == 1:
            primordium_angle_down = abs(np.random.normal(45, 10))
        else:
            primordium_angle_down = abs(np.random.normal(70, 10))
        primordium_angle_roll = abs(np.random.normal(5, 5))
    else:
        potential_radius = (apex.radius - Dmin / 2) * RMD + Dmin / 2.
        apex_angle_roll = 120
        if order == 1:
            primordium_angle_down = 45
        else:
            primordium_angle_down = 70
        primordium_angle_roll = 5

    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:
    if apex.dist_to_ramif > IPD and potential_radius >= Dmin and potential_radius <= apex.radius:  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # The distance that the tip of the apex has covered since the actual primordium formation is calculated:
        elongation_since_last_ramif = apex.dist_to_ramif - IPD

        # A specific rolling angle is attributed to the parent apex:
        apex.angle_roll = apex_angle_roll

        # We verify that the apex has actually elongated:
        if apex.actual_elongation > 0:
            # Then the actual time since the primordium must have been formed is precisely calculated
            # according to the actual growth of the parent apex since primordium formation,
            # taking into account the actual growth rate of the parent defined as
            # apex.actual_elongation / time_step_in_seconds
            actual_time_since_formation = elongation_since_last_ramif / elongation_rate
        else:
            actual_time_since_formation = 0.

        # And we add the primordium of a possible new lateral root:
        ramif = ADDING_A_CHILD(mother_element=apex, edge_type='+', label='Apex', type='Normal_root_before_emergence',
                               angle_down=primordium_angle_down,
                               angle_roll=primordium_angle_roll,
                               length=0.,
                               radius=potential_radius,
                               identical_properties=False,
                               nil_properties=True)
        # We specify the exact time since formation:
        ramif.actual_time_since_primordium_formation = actual_time_since_formation
        ramif.thermal_time_since_primordium_formation = actual_time_since_formation * temperature_time_adjustment
        # And the new distance between the parent apex and the last ramification is redefined,
        # by taking into account the actual elongation of apex since the child formation:
        apex.dist_to_ramif = elongation_since_last_ramif
        # We also put in memory the index of the child:
        apex.lateral_primordium_index = ramif.index()
        # We add the apex and its ramif in the list of apices returned by the function:
        new_apex.append(apex)
        new_apex.append(ramif)

    return new_apex


def calculating_C_supply_for_elongation(element):
    """

    :param element:
    :return:
    """

    n = element

    # We initialize each amount of hexose available for growth:
    n.hexose_available_for_elongation = 0.
    n.struct_mass_contributing_to_elongation = 0.

    # We initialize empty lists:
    list_of_elongation_supporting_elements = []
    list_of_elongation_supporting_elements_hexose = []
    list_of_elongation_supporting_elements_mass = []

    # We then calculate the length of an apical zone of a fixed length which can provide the amount of hexose required for growth:
    growing_zone_length = growing_zone_factor * n.radius
    # We calculate the corresponding volume to which this length should correspond based on the diameter of this apex:
    supplying_volume = growing_zone_length * n.radius ** 2 * pi

    # We start counting the hexose at the apex:
    index = n.index()
    current_element = n

    # We initialize a temporary variable that will be used as a counter:
    remaining_volume = supplying_volume

    # As long the remaining volume is not zero:
    while remaining_volume > 0:
        # If the volume of the current element is lower than the remaining volume:
        if remaining_volume > current_element.volume:
            # We add to the amount of hexose available all the hexose in the current element:
            hexose_contribution = current_element.C_hexose_root * current_element.struct_mass
            n.hexose_available_for_elongation += hexose_contribution
            n.struct_mass_contributing_to_elongation += current_element.struct_mass
            # We record the index of the contributing element:
            list_of_elongation_supporting_elements.append(index)
            # We record the amount of hexose that the current element can provide:
            list_of_elongation_supporting_elements_hexose.append(hexose_contribution)
            # We record the structural mass from which the current element contributes:
            list_of_elongation_supporting_elements_mass.append(current_element.struct_mass)
            # We subtract the volume of the current element to the remaining volume:
            remaining_volume = remaining_volume - current_element.volume

            # And we try to move the index to the segment preceding the current element:
            index_attempt = g.Father(index, EdgeType='<')
            # If there is no father element on this axis:
            if index_attempt is None:
                # Then we try to move to the mother root, if any:
                index_attempt = g.Father(index, EdgeType='+')
                # If there is no such root:
                if index_attempt is None:
                    # print "!!! ERROR: For element", n.index(),"of type", n.type, "there is no possibility to move higher than the element", index
                    # Then we exit the loop here:
                    break
            # We set the new index:
            index = index_attempt
            # We define the new element to consider according to the new index:
            current_element = g.node(index)
        # Otherwise, this is the last preceding element to consider:
        else:
            # We finally add to the amount of hexose available for elongation a part of the hexose of the current element:
            hexose_contribution = current_element.C_hexose_root * current_element.struct_mass \
                                  * remaining_volume / current_element.volume
            n.hexose_available_for_elongation += hexose_contribution
            n.struct_mass_contributing_to_elongation += current_element.struct_mass \
                                                        * remaining_volume / current_element.volume
            # We record the index of the contributing element:
            list_of_elongation_supporting_elements.append(index)
            # We record the amount of hexose that the current element can provide:
            list_of_elongation_supporting_elements_hexose.append(hexose_contribution)
            # We record the structural mass from which the current element contributes:
            list_of_elongation_supporting_elements_mass.append(
                current_element.struct_mass * remaining_volume / current_element.volume)
            # And the remaining volume to consider is set to 0:
            remaining_volume = 0.
            # And we exit the loop here:
            break

    # We record the average concentration in hexose of the whole zone of hexose supply contributing to elongation:
    if n.struct_mass_contributing_to_elongation > 0.:
        n.growing_zone_C_hexose_root = n.hexose_available_for_elongation / n.struct_mass_contributing_to_elongation
    else:
        print("!!! ERROR: the mass contributing to elongation in element", n.index(), "of type", n.type, "is",
              n.struct_mass_contributing_to_elongation,
              "g, and its structural mass is", n.struct_mass, "g!")
        n.growing_zone_C_hexose_root = 0.

    n.list_of_elongation_supporting_elements = list_of_elongation_supporting_elements
    n.list_of_elongation_supporting_elements_hexose = list_of_elongation_supporting_elements_hexose
    n.list_of_elongation_supporting_elements_mass = list_of_elongation_supporting_elements_mass

    return list_of_elongation_supporting_elements, list_of_elongation_supporting_elements_hexose, list_of_elongation_supporting_elements_mass


# FUNCTION: POTENTIAL APEX DEVELOPMENT
#######################################

def potential_apex_development(apex, time_step_in_seconds=1. * 60. * 60. * 24., ArchiSimple=False,
                               soil_temperature_in_Celsius=20, printing_warnings=False):
    """

    :param apex:
    :param time_step_in_seconds:
    :param ArchiSimple:
    :param printing_warnings:
    :return:
    """

    # We initialize an empty list in which the modified apex will be added:
    new_apex = []
    # We record the current radius and length prior to growth as the initial radius and length:
    apex.initial_radius = apex.radius
    apex.initial_length = apex.length
    # We initialize the properties "potential_radius" and "potential_length" returned by the function:
    apex.potential_radius = apex.radius
    apex.potential_length = apex.length

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=T_ref_growth,
                                                           A=growth_increase_with_temperature,
                                                           B=1,
                                                           C=0)

    # CASE 1: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING ADVENTITIOUS ROOT
    # -----------------------------------------------------------------------------------------
    # If the adventitious root has not emerged yet:
    if apex.type == "adventitious_root_before_emergence":

        global thermal_time_since_last_adventitious_root_emergence
        global adventitious_root_emergence

        # If the time elapsed since the last emergence of adventitious root is higher than the prescribed frequency
        # of adventitious root emission, and if no other adventitious root has already been allowed to emerge:
        if thermal_time_since_last_adventitious_root_emergence + time_step_in_seconds * temperature_time_adjustment > 1. / ER \
                and adventitious_root_emergence == "Possible":
            # The time since primordium formation is incremented:
            apex.actual_time_since_primordium_formation += time_step_in_seconds
            apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
            # The adventitious root may have emerged, and the potential time elapsed
            # since its possible emergence over this time step is calculated:
            apex.thermal_potential_time_since_emergence = thermal_time_since_last_adventitious_root_emergence \
                                                          + time_step_in_seconds * temperature_time_adjustment - 1. / ER
            # If the apex could have emerged sooner:
            if apex.thermal_potential_time_since_emergence > time_step_in_seconds * temperature_time_adjustment:
                # The time since emergence is equal to the time elapsed during this time step (since it must have emerged at this time step):???????????????????????????
                apex.thermal_potential_time_since_emergence = time_step_in_seconds * temperature_time_adjustment

            # We record the different element that can contribute to the C supply necessary for growth,
            # and we calculate a mean concentration of hexose in this supplying zone:
            calculating_C_supply_for_elongation(apex)
            # The corresponding elongation of the apex is calculated:
            apex.potential_length = elongated_length(initial_length=apex.initial_length, radius=apex.initial_radius,
                                                     C_hexose_root=apex.growing_zone_C_hexose_root,
                                                     elongation_time_in_seconds=apex.thermal_potential_time_since_emergence,
                                                     ArchiSimple=ArchiSimple,
                                                     soil_temperature_in_Celsius=soil_temperature_in_Celsius)

            # If ArchiSimple has been chosen as the growth model:
            if ArchiSimple:
                apex.type = "Normal_root_after_emergence"
                # We reset the time since an adventitious root may have emerged (REMINDER: it is a "global" value):
                thermal_time_since_last_adventitious_root_emergence = apex.thermal_potential_time_since_emergence
                # We forbid the emergence of other adventitious root for the current time step (REMINDER: it is a "global" value):
                adventitious_root_emergence = "Impossible"
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # Otherwise, we control the actual emergence of this primordium through the management of the parent:
            else:
                # We select the parent on which the primordium is supposed to receive its C, i.e. the base of the root system:
                parent = g.node(1)
                # The possibility of emergence of a lateral root from the parent
                # and the associated struct_mass C cost are recorded inside the parent:
                parent.lateral_emergence_possibility = "Possible"
                # parent.emergence_cost = emergence_cost
                # We record the index of the primordium inside the parent:
                parent.lateral_primordium_index = apex.index()
                # WATCH OUT: THE CODE DOESN'T HANDLE THE SITUATION WHERE MORE THAN ONE adventitious ROOT SHOULD EMERGE IN THE SAME TIME STEP!!!!!!!!!!!!!!!!!!!!!!!!!
                # We forbid the emergence of other adventitious root for the current time step (REMINDER: it is a "global" value):
                adventitious_root_emergence = "Impossible"
                # The new element returned by the function corresponds to the potentially emerging apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
        else:
            # Otherwise, the adventitious root cannot emerge at this time step and is left unchanged:
            apex.actual_time_since_primordium_formation += time_step_in_seconds
            apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
            new_apex.append(apex)
            # And the function returns this apex and stops here:
            return new_apex

    # CASE 2: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING NORMAL LATERAL ROOT
    # ---------------------------------------------------------------------------------------------
    if apex.type == "Normal_root_before_emergence":
        # If the time since primordium formation is higher than the delay of emergence:
        if apex.thermal_time_since_primordium_formation + time_step_in_seconds * temperature_time_adjustment > emergence_delay:
            # The time since primordium formation is incremented:
            apex.actual_time_since_primordium_formation += time_step_in_seconds
            apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
            # The potential time elapsed at the end of this time step since the emergence is calculated:
            apex.thermal_potential_time_since_emergence = apex.thermal_time_since_primordium_formation - emergence_delay
            # If the apex could have emerged sooner:
            if apex.thermal_potential_time_since_emergence > time_step_in_seconds * temperature_time_adjustment:
                # The time since emergence is equal to the time elapsed during this time step (since it must have emerged at this time step):
                apex.thermal_potential_time_since_emergence = time_step_in_seconds * temperature_time_adjustment
            # We record the different element that can contribute to the C supply necessary for growth,
            # and we calculate a mean concentration of hexose in this supplying zone:
            calculating_C_supply_for_elongation(apex)
            # The corresponding elongation of the apex is calculated:
            apex.potential_length = elongated_length(initial_length=apex.initial_length, radius=apex.initial_radius,
                                                     C_hexose_root=apex.growing_zone_C_hexose_root,
                                                     elongation_time_in_seconds=apex.thermal_potential_time_since_emergence,
                                                     ArchiSimple=ArchiSimple,
                                                     soil_temperature_in_Celsius=soil_temperature_in_Celsius)

            # If ArchiSimple has been chosen as the growth model:
            if ArchiSimple:
                apex.type = "Normal_root_after_emergence"
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # Otherwise, we control the actual emergence of this primordium through the management of the parent:
            else:
                # We select the parent on which the primordium has been formed:
                vid = apex.index()
                index_parent = g.Father(vid, EdgeType='+')
                parent = g.node(index_parent)
                # The possibility of emergence of a lateral root from the parent
                # and the associated struct_mass C cost are recorded inside the parent:
                parent.lateral_emergence_possibility = "Possible"
                # parent.emergence_cost = emergence_cost
                parent.lateral_primordium_index = apex.index()
                # And the new element returned by the function corresponds to the potentially emerging apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
        # Otherwise, the time since primordium formation is simply incremented:
        else:
            apex.actual_time_since_primordium_formation += time_step_in_seconds
            apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
            # And the new element returned by the function corresponds to the modified apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

    # CASE 3: THE APEX BELONGS TO AN AXIS THAT HAS ALREADY EMERGED:
    # --------------------------------------------------------------
    # IF THE APEX CAN CONTINUE GROWING:
    if apex.thermal_time_since_emergence + time_step_in_seconds * temperature_time_adjustment < apex.growth_duration:
        # The times are incremented:
        apex.actual_time_since_primordium_formation += time_step_in_seconds
        apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
        apex.actual_time_since_emergence += time_step_in_seconds
        apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
        # We record the different element that can contribute to the C supply necessary for growth,
        # and we calculate a mean concentration of hexose in this supplying zone:
        calculating_C_supply_for_elongation(apex)
        # The corresponding potential elongation of the apex is calculated:
        # apex.potential_length = elongated_length(apex.length, apex.radius, time_step_in_seconds)
        apex.potential_length = elongated_length(initial_length=apex.length, radius=apex.radius,
                                                 C_hexose_root=apex.growing_zone_C_hexose_root,
                                                 elongation_time_in_seconds=time_step_in_seconds * temperature_time_adjustment,
                                                 ArchiSimple=ArchiSimple,
                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius)
        # And the new element returned by the function corresponds to the modified apex:
        new_apex.append(apex)
        # And the function returns this new apex and stops here:
        return new_apex

    # OTHERWISE, THE APEX HAD TO STOP:
    else:
        # IF THE APEX HAS NOT REACHED ITS LIFE DURATION:
        if apex.thermal_time_since_growth_stopped + time_step_in_seconds * temperature_time_adjustment < apex.life_duration:
            # IF THE APEX HAS ALREADY BEEN STOPPED AT A PREVIOUS TIME STEP:
            if apex.type == "Stopped" or apex.type == "Just_stopped":
                # The time since growth stopped is simply increased by one time step:
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
                # The type is (re)declared "Stopped":
                apex.type = "Stopped"
                # The times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # OTHERWISE, THE APEX HAS TO STOP DURING THIS TIME STEP:
            else:
                # The type is declared "Just stopped":
                apex.type = "Just_stopped"
                # Then the exact time since growth stopped is calculated:
                apex.thermal_time_since_growth_stopped = apex.thermal_time_since_emergence + time_step_in_seconds * temperature_time_adjustment - apex.growth_duration
                apex.actual_time_since_growth_stopped = apex.thermal_time_since_growth_stopped / temperature_time_adjustment

                # We record the different element that can contribute to the C supply necessary for growth,
                # and we calculate a mean concentration of hexose in this supplying zone:
                calculating_C_supply_for_elongation(apex)
                # And the potential elongation of the apex before growth stopped is calculated:
                apex.potential_length = elongated_length(initial_length=apex.length, radius=apex.radius,
                                                         C_hexose_root=apex.growing_zone_C_hexose_root,
                                                         elongation_time_in_seconds=time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped,
                                                         ArchiSimple=ArchiSimple,
                                                         soil_temperature_in_Celsius=soil_temperature_in_Celsius)
                # VERIFICATION:
                if time_step_in_seconds * temperature_time_adjustment - apex.thermal_time_since_growth_stopped < 0.:
                    print("!!! ERROR: The apex", apex.index(), "has stopped since", apex.actual_time_since_growth_stopped,
                          "seconds; the time step is", time_step_in_seconds)
                    print("We set the potential length of this apex equal to its initial length.")
                    apex.potential_length = apex.initial_length

                # The times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

        # OTHERWISE, THE APEX MUST BE DEAD:
        else:
            # IF THE APEX HAS ALREADY DIED AT A PREVIOUS TIME STEP:
            if apex.type == "Dead" or apex.type == "Just_dead":
                # The type is (re)declared "Dead":
                apex.type = "Dead"
                # And the times are simply incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.actual_time_since_death += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_death += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # OTHERWISE, THE APEX HAS TO DIE DURING THIS TIME STEP:
            else:
                # Then the apex is declared "Just dead":
                apex.type = "Just_dead"
                # The exact time since the apex died is calculated:
                apex.thermal_time_since_death = apex.thermal_time_since_growth_stopped + time_step_in_seconds * temperature_time_adjustment - apex.life_duration
                apex.actual_time_since_death = apex.thermal_time_since_death / temperature_time_adjustment
                # And the other times are incremented:
                apex.actual_time_since_primordium_formation += time_step_in_seconds
                apex.actual_time_since_emergence += time_step_in_seconds
                apex.actual_time_since_growth_stopped += time_step_in_seconds
                apex.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment
                apex.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

    # VERIFICATION: If the apex does not match any of the cases listed above:
    print("!!! ERROR: No case found for defining growth of apex", apex.index(), "of type", apex.type)
    new_apex.append(apex)
    return new_apex


# FUNCTION: POTENTIAL SEGMENT DEVELOPMENT
#########################################

def potential_segment_development(segment, time_step_in_seconds=60. * 60. * 24., radial_growth="Possible",
                                  ArchiSimple=False, soil_temperature_in_Celsius=20):
    """

    """

    # We initialize an empty list that will contain the new segment to be returned:
    new_segment = []
    # We record the current radius and length prior to growth as the initial radius and length:
    segment.initial_radius = segment.radius
    segment.initial_length = segment.length
    # We initialize the properties "potential_radius" and "potential_length":
    segment.theoretical_radius = segment.radius
    segment.potential_radius = segment.radius
    segment.potential_length = segment.length

    # CASE 1: THE SEGMENT IS A NODULE:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #################################

    if segment.type == "Root_nodule":
        # We consider the amount of hexose available in the nodule AND in the parent segment:
        index_parent = g.Father(segment.index(), EdgeType='+')
        parent = g.node(index_parent)
        segment.hexose_available_for_thickening = parent.C_hexose_root * parent.struct_mass \
                                                  + segment.C_hexose_root * segment.struct_mass
        # We calculate an average concentration of hexose that will help to regulate nodule growth:
        C_hexose_regulating_nodule_growth = segment.hexose_available_for_thickening / (parent.struct_mass + segment.struct_mass)
        # We modulate the relative increase in radius by the amount of C available in the nodule:
        thickening_rate = relative_nodule_thickening_rate_max \
                          * C_hexose_regulating_nodule_growth / (Km_nodule_thickening + C_hexose_regulating_nodule_growth)
        # We modulate the relative increase in radius by the temperature:
        thickening_rate = thickening_rate * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                     process_at_T_ref=1,
                                                                     T_ref=T_ref_growth,
                                                                     A=growth_increase_with_temperature,
                                                                     C=0)
        segment.theoretical_radius = segment.radius * (1 + thickening_rate * time_step_in_seconds)
        if segment.theoretical_radius > nodule_max_radius:
            segment.potential_radius = nodule_max_radius
        else:
            segment.potential_radius = segment.theoretical_radius
        # We add the modified segment to the list of new segments, and we quit the function here:
        new_segment.append(segment)
        return new_segment

    # CASE 2: THE SEGMENT IS NOT A NODULE:
    ######################################

    # We initialize internal variables:
    son_section = 0.
    sum_of_lateral_sections = 0.
    number_of_actual_children = 0.
    death_count = 0.
    list_of_times_since_death = []

    # We define the amount of hexose available for thickening:
    segment.hexose_available_for_thickening = segment.C_hexose_root * segment.struct_mass

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # ------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=T_ref_growth,
                                                           A=growth_increase_with_temperature,
                                                           B=1,
                                                           C=0)

    # CALCULATING DEATH OR RADIAL GROWTH:
    # ------------------------------------

    # For each child of the segment:
    for child in segment.children():

        # Then we add one child to the actual number of children:
        number_of_actual_children += 1

        if child.radius < 0. or child.potential_radius < 0.:
            print("!!! ERROR: the radius of the element", child.index(), "is negative!")
        # If the child belongs to the same axis:
        if child.properties()['edge_type'] == '<':
            # Then we record the THEORETICAL section of this child:
            son_section = child.theoretical_radius ** 2 * pi
            # # Then we record the section of this child:
            # son_section = child.radius * child.radius * pi
        # Otherwise if the child is the element of a lateral root AND if this lateral root has already emerged
        # AND the lateral element is not a nodule:
        elif child.properties()['edge_type'] == '+' and child.length > 0. and child.type != "Root_nodule":
            # We add the POTENTIAL section of this child to a sum of lateral sections:
            sum_of_lateral_sections += child.theoretical_radius ** 2 * pi
            # # We add the section of this child to a sum of lateral sections:
            # sum_of_lateral_sections += child.radius ** 2 * pi

        # If this child has just died or was already dead:
        if child.type == "Just_dead" or child.type == "Dead":
            # Then we add one dead child to the death count:
            death_count += 1
            # And we record the exact time since death:
            list_of_times_since_death.append(child.actual_time_since_death)

    # If each child in the list of children has been recognized as dead or just dead:
    if death_count == number_of_actual_children:
        # If the investigated segment was already declared dead at the previous time step:
        if segment.type == "Just_dead" or segment.type == "Dead":
            # Then we transform its status into "Dead"
            segment.type = "Dead"
        else:
            # Then the segment has to die:
            segment.type = "Just_dead"
    # Otherwise, at least one of the children axis is not dead, so the father segment should not be dead

    # REGULATION OF RADIAL GROWTH BY AVAILABLE CARBON:
    # ------------------------------------------------
    # If the radial growth is possible:
    if radial_growth == "Possible":
        # The radius of the root segment is defined according to the pipe model.
        # In ArchiSimp9, the radius is increased by considering the sum of the sections of all the children,
        # by adding a fraction (SGC) of this sum of sections to the current section of the parent segment,
        # and by calculating the new radius that corresponds to this new section of the parent:
        segment.theoretical_radius = sqrt(son_section / pi + SGC * sum_of_lateral_sections / pi)
        # However, if the net difference is below 0.1% of the initial radius:
        if (segment.theoretical_radius - segment.initial_radius) <= 0.001 * segment.initial_radius:
            # Then the potential radius is set to the initial radius:
            segment.theoretical_radius = segment.initial_radius
        # If we consider simple ArchiSimple rules:
        if ArchiSimple:
            # Then the potential radius to form is equal to the theoretical one determined by geometry:
            segment.potential_radius = segment.theoretical_radius
        # Otherwise, if we don't strictly follow simple ArchiSimple rules and if there can be an increase in radius:
        elif segment.length > 0. and segment.theoretical_radius > segment.radius:
            # We calculate the maximal increase in radius that can be achieved over this time step,
            # based on a Michaelis-Menten formalism that regulates the maximal rate of increase
            # according to the amount of hexose available:
            thickening_rate = relative_root_thickening_rate_max \
                              * segment.C_hexose_root / (Km_thickening + segment.C_hexose_root)
            # We add a correction factor for temperature:
            thickening_rate = thickening_rate * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                         process_at_T_ref=1,
                                                                         T_ref=T_ref_growth,
                                                                         A=growth_increase_with_temperature,
                                                                         C=0)
            # The maximal possible new radius according to this regulation is therefore:
            new_radius_max = (1 + thickening_rate * time_step_in_seconds) * segment.initial_radius
            # If the potential new radius is higher than the maximal new radius:
            if segment.theoretical_radius > new_radius_max:
                # Then potential thickening is limited up to the maximal new radius:
                segment.potential_radius = new_radius_max
            # Otherwise, the potential radius to achieve is equal to the theoretical one:
            else:
                segment.potential_radius = segment.theoretical_radius
        # And if the segment corresponds to one of the elements of length 0 supporting one adventitious root:
        if segment.type == "Support_for_adventitious_root":
            # Then the radius is directly increased, as this element will not be considered in the function calculating actual growth:
            segment.radius = segment.potential_radius

    # We increase the various time variables:
    segment.actual_time_since_primordium_formation += time_step_in_seconds
    segment.actual_time_since_emergence += time_step_in_seconds
    segment.thermal_time_since_primordium_formation += time_step_in_seconds * temperature_time_adjustment
    segment.thermal_time_since_emergence += time_step_in_seconds * temperature_time_adjustment

    if segment.type == "Stopped" or segment.type == "Just_stopped":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
    if segment.type == "Just_dead":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
        # We check that the list of times_since_death is not empty:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if list_of_times_since_death:
            segment.actual_time_since_death = min(list_of_times_since_death)
        else:
            segment.actual_time_since_death = 0.
        segment.thermal_time_since_death = segment.actual_time_since_death * temperature_time_adjustment
    if segment.type == "Dead":
        segment.actual_time_since_growth_stopped += time_step_in_seconds
        segment.thermal_time_since_growth_stopped += time_step_in_seconds * temperature_time_adjustment
        segment.actual_time_since_death += time_step_in_seconds
        segment.thermal_time_since_death += time_step_in_seconds * temperature_time_adjustment

    new_segment.append(segment)
    return new_segment


# SIMULATION OF POTENTIAL ROOT GROWTH FOR ALL ROOT ELEMENTS
###########################################################

# We define a class "Simulate" which is used to simulate the development of apices and segments on the whole MTG "g":
class Simulate_potential_growth(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        """ Simulate on MTG. """
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex" or "Segment", from the tip to the base:
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = next(root_gen)
        self.apices_list = [g.node(v) for v in pre_order(g, root) if g.label(v) == 'Apex']
        self.segments_list = [g.node(v) for v in post_order(g, root) if g.label(v) == 'Segment']

    def step(self, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth="Possible", ArchiSimple=False,
             soil_temperature_in_Celsius=20):
        list_of_apices = list(self.apices_list)
        list_of_segments = list(self.segments_list)

        # For each apex in the list of apices:
        for apex in list_of_apices:
            # We define the new list of apices with the function apex_development:
            new_apices = [potential_apex_development(apex, time_step_in_seconds=time_step_in_seconds, ArchiSimple=ArchiSimple,
                                                     soil_temperature_in_Celsius=soil_temperature_in_Celsius)]

            # We add these new apices to apex:
            self.apices_list.extend(new_apices)
        # For each segment in the list of segments:
        for segment in list_of_segments:
            # We define the new list of apices with the function apex_development:
            new_segments = [potential_segment_development(segment, time_step_in_seconds=time_step_in_seconds,
                                                          radial_growth=radial_growth, ArchiSimple=ArchiSimple,
                                                          soil_temperature_in_Celsius=soil_temperature_in_Celsius)]

            # We add these new apices to apex:
            self.segments_list.extend(new_segments)


# We finally define the function that calculates the potential growth of the whole MTG at a given time step:
def potential_growth(g, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth="Possible", ArchiSimple=False,
                     soil_temperature_in_Celsius=20):
    # We simulate the development of all apices and segments in the MTG:
    simulator = Simulate_potential_growth(g)
    simulator.step(time_step_in_seconds=time_step_in_seconds, radial_growth=radial_growth, ArchiSimple=ArchiSimple,
                   soil_temperature_in_Celsius=soil_temperature_in_Celsius)


# FUNCTION: A GIVEN APEX CAN BE TRANSFORMED INTO SEGMENTS AND GENERATE NEW PRIMORDIA:
#####################################################################################

def segmentation_and_primordium_formation(apex, time_step_in_seconds=1. * 60. * 60. * 24.,
                                          soil_temperature_in_Celsius=20, ArchiSimple=False, random=True,
                                          nodules=True):
    # NOTE: This function is supposed to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    # Optional - We can add random geometry, or not:
    if random:
        np.random.seed(random_choice + apex.index())
        angle_mean = 0
        angle_var = 5
        segment_angle_down = np.random.normal(angle_mean, angle_var)
        segment_angle_roll = np.random.normal(angle_mean, angle_var)
        apex_angle_down = np.random.normal(angle_mean, angle_var)
        apex_angle_roll = np.random.normal(angle_mean, angle_var)
    else:
        segment_angle_down = 0
        segment_angle_roll = 0
        apex_angle_down = 0
        apex_angle_roll = 0

    # We initialize the new_apex that will be returned by the function:
    new_apex = []
    # We record the initial geometrical features and total amounts,
    # knowing that the concentrations will not be changed by the segmentation:
    initial_length = apex.length
    initial_dist_to_ramif = apex.dist_to_ramif
    initial_elongation = apex.actual_elongation
    initial_elongation_rate = apex.actual_elongation_rate
    initial_struct_mass = apex.struct_mass
    initial_resp_maintenance = apex.resp_maintenance
    initial_resp_growth = apex.resp_growth
    initial_struct_mass_produced = apex.struct_mass_produced
    initial_initial_struct_mass = apex.initial_struct_mass
    # Note: this is not an error, we also need to record the initial structural mass before growth!
    initial_hexose_exudation = apex.hexose_exudation
    initial_hexose_uptake = apex.hexose_uptake
    initial_hexose_degradation = apex.hexose_degradation
    initial_hexose_growth_demand = apex.hexose_growth_demand
    initial_hexose_consumption_by_growth = apex.hexose_consumption_by_growth
    initial_hexose_production_from_phloem = apex.hexose_production_from_phloem
    initial_sucrose_loading_in_phloem = apex.sucrose_loading_in_phloem
    initial_hexose_mobilization_from_reserve = apex.hexose_mobilization_from_reserve
    initial_hexose_immobilization_as_reserve = apex.hexose_immobilization_as_reserve
    initial_Deficit_sucrose_root = apex.Deficit_sucrose_root
    initial_Deficit_hexose_root = apex.Deficit_hexose_root
    initial_Deficit_hexose_soil = apex.Deficit_hexose_soil

    # We record the type of the apex, as it may correspond to an apex that has stopped (or even died):
    initial_type = apex.type
    initial_lateral_emergence_possibility = apex.lateral_emergence_possibility

    # If the length of the apex is smaller than the defined length of a root segment:
    if apex.length <= segment_length:

        # We assume that the growth functions that may have been called previously have only modified radius and length,
        # but not the struct_mass and the total amounts present in the root element.
        # We modify the geometrical features of the present element according to the new length and radius:
        apex.volume = surfaces_and_volumes(apex, apex.radius, apex.potential_length)["volume"]
        apex.struct_mass = apex.volume * root_tissue_density

        # # WATCH OUT: the mass fraction in this case should be 1 and quantities in that element should NOT be changed!
        # # We modify the variables representing total amounts according to the new struct_mass:
        # mass_fraction = apex.struct_mass / initial_struct_mass
        #
        # apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        # apex.resp_growth = initial_resp_growth * mass_fraction
        # apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction
        # apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        # apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        # apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        # apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        # apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
        #
        # apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
        # apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
        # apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
        # apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction
        #
        # apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        # apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        # apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction

        # We simply call the function primordium_formation to check whether a primordium should have been formed
        # (Note: we assume that the segment length is always smaller than the inter-branching distance IBD,
        # so that in this case, only 0 or 1 primordium may have been formed - the function is called only once):
        new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random))

    # Otherwise, we have to calculate the number of entire segments within the apex.
    else:

        # If the final length of the apex does not correspond to an entire number of segments:
        if apex.length / segment_length - floor(apex.length / segment_length) > 0.:
            # Then the total number of segments to be formed is:
            n_segments = floor(apex.length / segment_length)
        else:
            # Otherwise, the number of segments to be formed is decreased by 1,
            # so that the last element corresponds to an apex with a positive length:
            n_segments = floor(apex.length / segment_length) - 1
        n_segments = int(n_segments)

        # We develop each new segment, except the last one, by transforming the current apex into a segment
        # and by adding a new apex after it, in an iterative way for (n-1) segments:
        for i in range(1, n_segments):
            # We define the length of the present element as the constant length of a segment:
            apex.length = segment_length
            # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
            apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - segment_length * i)
            # We modify the geometrical features of the present element according to the new length:
            apex.volume = surfaces_and_volumes(apex, apex.radius, apex.length)["volume"]
            apex.struct_mass = apex.volume * root_tissue_density

            # We calculate the mass fraction that the segment represents compared to the whole element prior to segmentation:
            mass_fraction = apex.struct_mass / initial_struct_mass

            # We modify the variables representing total amounts according to this mass fraction:
            apex.resp_maintenance = initial_resp_maintenance * mass_fraction
            apex.resp_growth = initial_resp_growth * mass_fraction

            apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
            apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction

            apex.hexose_exudation = initial_hexose_exudation * mass_fraction
            apex.hexose_uptake = initial_hexose_uptake * mass_fraction
            apex.hexose_degradation = initial_hexose_degradation * mass_fraction
            apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
            apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

            apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
            apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
            apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
            apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

            apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
            apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
            apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
            # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
            new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                                 time_step_in_seconds=time_step_in_seconds,
                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                                                 random=random))
            # The current element that has been elongated up to segment_length is now considered as a segment:
            apex.label = 'Segment'

            # And we add a new apex after this segment, initially of length 0, that is now the new element called "apex":
            apex = ADDING_A_CHILD(mother_element=apex, edge_type='<', label='Apex',
                                  type=apex.type,
                                  angle_down=segment_angle_down,
                                  angle_roll=segment_angle_roll,
                                  length=0.,
                                  radius=apex.radius,
                                  identical_properties=True,
                                  nil_properties=False)
            apex.actual_elongation = segment_length * i

        # Finally, we do this operation one last time for the last segment:
        # We define the length of the present element as the constant length of a segment:
        apex.length = segment_length
        apex.potential_length = apex.length
        apex.initial_length = apex.length
        # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
        apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - segment_length * n_segments)
        # We modify the geometrical features of the present element according to the new length:
        apex.volume = surfaces_and_volumes(apex, apex.radius, apex.length)["volume"]
        apex.struct_mass = apex.volume * root_tissue_density
        # We modify the variables representing total amounts according to the mass fraction:
        mass_fraction = apex.struct_mass / initial_struct_mass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
        apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

        apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
        apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
        apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
        apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
        new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random))
        # The current element that has been elongated up to segment_length is now considered as a segment:
        apex.label = 'Segment'

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # We add the possibility of a nodule formation on the segment that is closest to the apex:
        if nodules and len(apex.children()) < 2 and np.random.random() < nodule_formation_probability:
            nodule_formation(mother_element=apex)

        # And we define the new, final apex after the new defined segment, with a new length defined as:
        new_length = initial_length - n_segments * segment_length
        apex = ADDING_A_CHILD(mother_element=apex, edge_type='<', label='Apex',
                              type=apex.type,
                              angle_down=segment_angle_down,
                              angle_roll=segment_angle_roll,
                              length=new_length,
                              radius=apex.radius,
                              identical_properties=True,
                              nil_properties=False)
        apex.potential_length = new_length
        apex.initial_length = new_length
        apex.dist_to_ramif = initial_dist_to_ramif
        apex.actual_elongation = initial_elongation

        # We modify the geometrical features of the new apex according to the defined length:
        apex.volume = surfaces_and_volumes(apex, apex.radius, apex.length)["volume"]
        apex.struct_mass = apex.volume * root_tissue_density
        # We modify the variables representing total amounts according to the new struct_mass:
        mass_fraction = apex.struct_mass / initial_struct_mass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.initial_struct_mass = initial_initial_struct_mass * mass_fraction
        apex.struct_mass_produced = initial_struct_mass_produced * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction

        apex.hexose_production_from_phloem = initial_hexose_production_from_phloem * mass_fraction
        apex.sucrose_loading_in_phloem = initial_sucrose_loading_in_phloem * mass_fraction
        apex.hexose_mobilization_from_reserve = initial_hexose_mobilization_from_reserve * mass_fraction
        apex.hexose_immobilization_as_reserve = initial_hexose_immobilization_as_reserve * mass_fraction

        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        # And we call the function primordium_formation to check whether a primordium should have been formed
        new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,
                                             soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random))
        # And we add the last apex present at the end of the elongated axis:
        new_apex.append(apex)

    return new_apex


# We define a class "Simulate_segmentation_and_primordia_formation" which is used to simulate the segmentation of apices
# and the apparition of primordium for a given MTG:
class Simulate_segmentation_and_primordia_formation(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        """ Simulate on MTG. """
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex":
        self._apices = [g.node(v) for v in g.vertices_iter(scale=1) if g.label(v) == 'Apex']

    def step(self, time_step_in_seconds, soil_temperature_in_Celsius=20, random=True, nodules=False):
        # We define "apices_list" as the list of all apices in g:
        apices_list = list(self._apices)
        # For each apex in the list of apices:
        for apex in apices_list:
            if apex.type == "Normal_root_after_emergence" and apex.length > 0.:  # Is it needed? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # We define the new list of apices with the function apex_development:
                new_apex = segmentation_and_primordium_formation(apex, time_step_in_seconds,
                                                                 soil_temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                 random=random,
                                                                 nodules=nodules)
                # We add these new apices to apex:
                self._apices.extend(new_apex)


# We finally define the function that creates new segments and priomordia in "g":
def segmentation_and_primordia_formation(g, time_step_in_seconds=1. * 60. * 60. * 24.,
                                         soil_temperature_in_Celsius=20,
                                         random=True, printing_warnings=False,
                                         nodules=False):
    # We simulate the segmentation of all apices:
    simulator = Simulate_segmentation_and_primordia_formation(g)
    simulator.step(time_step_in_seconds, soil_temperature_in_Celsius=soil_temperature_in_Celsius, random=random,
                   nodules=nodules)


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "ACTUAL GROWTH AND ASSOCIATED RESPIRATION"
########################################################################################################################
########################################################################################################################

# ACTUAL ELONGATION AND RADIAL GROWTH OF ROOT ELEMENTS:
#######################################################

# Function calculating the actual growth and the corresponding growth respiration:
def actual_growth_and_corresponding_respiration(g, time_step_in_seconds, soil_temperature_in_Celsius=20,
                                                printing_warnings=False):
    """
    This function defines how a segment, an apex and possibly an emerging root primordium will grow according to the amount
    of hexose present in the segment, taking into account growth respiration based on the model of Thornley and Cannell
    (2000). The calculation is based on the values of potential_radius, potential_length, lateral_emergence_possibility
    and emergence_cost defined in each element by the module "POTENTIAL GROWTH".
    The function returns the MTG "g" with modified values of radius and length of each element, the possibility of the
    emergence of lateral roots, and the cost of growth in terms of hexose demand.
    """

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=T_ref_growth,
                                                           A=growth_increase_with_temperature,
                                                           B=1,
                                                           C=0)

    # PROCEEDING TO ACTUAL GROWTH:
    # -----------------------------

    global adventitious_root_emergence

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)
    # We cover all the vertices in the MTG, from the tips to the base:
    for vid in post_order(g, root):

        # n represents the current root element:
        n = g.node(vid)

        # AVOIDANCE OF UNWANTED CASES:
        # -----------------------------
        # We make sure that the element is not dead:
        if n.type == "Dead" or n.type == "Just_dead" or n.type == "Support_for_adventitious_root":
            # In such case, we just pass to the next element in the iteration:
            continue

        # We make sure that there is a potential growth for this element:
        if n.potential_length <= n.initial_length and n.potential_radius <= n.initial_radius:
            # In such case, we just pass to the next element in the iteration:
            continue

        # INITIALIZATION AND CALCULATIONS OF POTENTIAL GROWTH DEMAND IN HEXOSE:
        # ----------------------------------------------------------------------

        # WARNING: All growth related variables should have been initialized by another module at the beginning of the time step!!!

        # We calculate the initial volume of the element:
        initial_volume = surfaces_and_volumes(n, n.initial_radius, n.initial_length)["volume"]
        # We calculate the potential volume of the element based on the potential radius and potential length:
        potential_volume = surfaces_and_volumes(n, n.potential_radius, n.potential_length)["volume"]
        # We calculate the number of moles of hexose required for growth, including the respiration cost according to
        # the yield growth included in the model of Thornley and Cannell (2000), where root_tissue_density is the dry structural
        # weight per volume (g m-3) and struct_mass_C_content is the amount of C per gram of dry structural mass (mol_C g-1):
        n.hexose_growth_demand = (potential_volume - initial_volume) \
                                 * root_tissue_density * struct_mass_C_content / yield_growth * 1 / 6.
        # We verify that this potential growth demand is positive:
        if n.hexose_growth_demand < 0.:
            print("!!! ERROR: a negative growth demand of", n.hexose_growth_demand,
                  "was calculated for the element", n.index(), "of class", n.label)
            print("The initial volume is", initial_volume, "the potential volume is", potential_volume)
            print("The initial length was", n.initial_length, "and the potential length was", n.potential_length)
            print("The initial radius was", n.initial_radius, "and the potential radius was", n.potential_radius)
            n.hexose_growth_demand = 0.
            # In such case, we just pass to the next element in the iteration:
            continue

        # CALCULATIONS OF THE AMOUNT OF HEXOSE AVAILABLE FOR GROWTH:
        # ---------------------------------------------------------

        # We initialize each amount of hexose available for growth:
        hexose_available_for_elongation = 0.
        hexose_available_for_thickening = 0.

        # If elongation is possible:
        if n.potential_length > n.length:
            hexose_available_for_elongation = n.hexose_available_for_elongation
            list_of_elongation_supporting_elements = n.list_of_elongation_supporting_elements
            list_of_elongation_supporting_elements_hexose = n.list_of_elongation_supporting_elements_hexose
            list_of_elongation_supporting_elements_mass = n.list_of_elongation_supporting_elements_mass

        # If radial growth is possible:
        if n.potential_radius > n.radius:
            # We only consider the amount of hexose immediately available in the element that can increase in radius:
            hexose_available_for_thickening = n.hexose_available_for_thickening

        # In case no hexose is available at all:
        if (hexose_available_for_elongation + hexose_available_for_thickening) <= 0.:
            # Then we move to the next element in the main loop:
            continue

        # We initialize the temporary variable "remaining_hexose" that computes the amount of hexose left for growth:
        remaining_hexose_for_elongation = hexose_available_for_elongation
        remaining_hexose_for_thickening = hexose_available_for_thickening

        # ACTUAL ELONGATION IS FIRST CONSIDERED:
        # ---------------------------------------

        # We calculate the maximal possible length of the root element according to all the hexose available for elongation:
        volume_max = initial_volume + hexose_available_for_elongation * 6. / (
                root_tissue_density * struct_mass_C_content) * yield_growth
        length_max = volume_max / (pi * n.initial_radius ** 2)

        # If the element can elongate:
        if n.potential_length > n.initial_length:

            # CALCULATING ACTUAL ELONGATION:
            # If elongation is possible but is limited by the amount of hexose available:
            if n.potential_length >= length_max:
                # Elongation is limited using all the amount of hexose available:
                n.length = length_max
            # Otherwise, elongation can be done up to the full potential:
            else:
                # Elongation is done up to the full potential:
                n.length = n.potential_length
            # The corresponding new volume is calculated:
            volume_after_elongation = (pi * n.initial_radius ** 2) * n.length
            # The overall cost of elongation is calculated as:
            hexose_consumption_by_elongation = 1. / 6. * (
                    volume_after_elongation - initial_volume) * root_tissue_density * struct_mass_C_content / yield_growth

            # If there has been an actual elongation:
            if n.length > n.initial_length:

                # REGISTERING THE COSTS FOR ELONGATION:
                # We cover each of the elements that have provided hexose for sustaining the elongation of element n:
                for i in range(0, len(list_of_elongation_supporting_elements)):
                    index = list_of_elongation_supporting_elements[i]
                    supplying_element = g.node(index)
                    # We define the actual contribution of the current element based on total hexose consumption by growth
                    # of element n and the relative contribution of the current element to the pool of the available hexose:
                    hexose_actual_contribution_to_elongation = hexose_consumption_by_elongation \
                                                               * list_of_elongation_supporting_elements_hexose[
                                                                   i] / hexose_available_for_elongation
                    # The amount of hexose used for growth in this element is increased:
                    supplying_element.hexose_consumption_by_growth += hexose_actual_contribution_to_elongation
                    # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                    supplying_element.resp_growth += hexose_actual_contribution_to_elongation * (1 - yield_growth) * 6.

                    # # POSSIBLE LIMITATION OF UPCOMING RADIAL GROWTH:
                    # # In the case of the first supplying element, i.e. the element that has elongated,
                    # # we subtract the contribution of this element to elongation to the amount of hexose available for thickening:
                    # if index == vid:
                    #     remaining_hexose_for_thickening = remaining_hexose_for_thickening - hexose_actual_contribution_to_elongation

        # ACTUAL RADIAL GROWTH IS THEN CONSIDERED:
        # -----------------------------------------

        # If the radius of the element can increase:
        if n.potential_radius > n.initial_radius:

            # CALCULATING ACTUAL THICKENING:
            # We calculate the increase in volume that can be achieved with the amount of hexose available:
            possible_radial_increase_in_volume = remaining_hexose_for_thickening * 6. * yield_growth / (
                    root_tissue_density * struct_mass_C_content)
            # We calculate the maximal possible volume based on the volume of the new cylinder after elongation
            # and the increase in volume that could be achieved by consuming all the remaining hexose:
            volume_max = surfaces_and_volumes(n, n.initial_radius, n.length)["volume"] \
                         + possible_radial_increase_in_volume
            # We then calculate the corresponding new possible radius corresponding to this maximum volume:
            if n.type == "Root_nodule":
                # If the element corresponds to a nodule, then it we calculate the radius of a theoretical sphere:
                possible_radius = (3. / (4. * pi)) ** (1. / 3.)
            else:
                # Otherwise, we calculate the radius of a cylinder:
                possible_radius = sqrt(volume_max / (n.length * pi))
            if possible_radius < 0.9999 * n.initial_radius:  # We authorize a difference of 0.01% due to calculation errors!
                print("!!! ERROR: the calculated new radius of element", n.index(), "is lower than the initial one!")
                print("The possible radius was", possible_radius, "and the initial radius was", n.initial_radius)

            # If the maximal radius that can be obtained is lower than the potential radius suggested by the potential growth module:
            if possible_radius <= n.potential_radius:
                # Then radial growth is limited and there is no remaining hexose after radial growth:
                n.radius = possible_radius
                hexose_actual_contribution_to_thickening = remaining_hexose_for_thickening
                remaining_hexose_for_thickening = 0.
            else:
                # Otherwise, radial growth is done up to the full potential and the remaining hexose is calculated:
                n.radius = n.potential_radius
                net_increase_in_volume = surfaces_and_volumes(n, n.radius, n.length)["volume"] \
                                         - surfaces_and_volumes(n, n.initial_radius, n.length)["volume"]
                # net_increase_in_volume = pi * (n.radius ** 2 - n.initial_radius ** 2) * n.length
                # We then calculate the remaining amount of hexose after thickening:
                hexose_actual_contribution_to_thickening = 1. / 6. * net_increase_in_volume * root_tissue_density * struct_mass_C_content / yield_growth

            # REGISTERING THE COSTS FOR THICKENING:
            fraction_of_available_hexose_in_the_element = (n.C_hexose_root * n.initial_struct_mass) / hexose_available_for_thickening
            # The amount of hexose used for growth in this element is increased:
            n.hexose_consumption_by_growth += (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)
            # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
            n.resp_growth += (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) * (1 - yield_growth) * 6.
            if n.type == "Root_nodule":
                index_parent = g.Father(n.index(), EdgeType='+')
                parent = g.node(index_parent)
                fraction_of_available_hexose_in_the_element = (parent.C_hexose_root * parent.initial_struct_mass) / hexose_available_for_thickening
                # The amount of hexose used for growth in this element is increased:
                parent.hexose_consumption_by_growth += (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element)
                # And the amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
                parent.resp_growth += (hexose_actual_contribution_to_thickening * fraction_of_available_hexose_in_the_element) * (
                        1 - yield_growth) * 6.

        # RECORDING THE ACTUAL STRUCTURAL MODIFICATIONS:
        # -----------------------------------------------

        # The new volume and surfaces of the element is automatically calculated:
        n.external_surface = surfaces_and_volumes(n, n.radius, n.length)["external_surface"]
        n.volume = surfaces_and_volumes(n, n.radius, n.length)["volume"]
        n.phloem_surface = surfaces_and_volumes(n, n.radius, n.length)["phloem_surface"]
        n.symplasm_surface = surfaces_and_volumes(n, n.radius, n.length)["symplasm_surface"]
        # The new dry structural struct_mass of the element is calculated from its new volume:
        n.struct_mass = n.volume * root_tissue_density
        n.struct_mass_produced = n.struct_mass - n.initial_struct_mass

        # Verification: we check that no negative length or struct_mass have been generated!
        if n.volume < 0:
            print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length, "and a mass of", n.struct_mass)
            # We then reset all the geometrical values to their initial values:
            n.length = n.initial_length
            n.radius = n.initial_radius
            n.struct_mass = n.initial_struct_mass
            n.struct_mass_produced = 0.
            n.external_surface = initial_surface  # TODO : unresolved reference
            n.volume = initial_volume

        # If there has been an actual elongation:
        if n.length > n.initial_length:

            # MODIFYING PROPERTIES:

            # If the elongated apex corresponded to an adventitious root:
            if n.type == "adventitious_root_before_emergence":
                # We reset the time since an adventitious root has emerged (REMINDER: it is a "global" value):
                global thermal_time_since_last_adventitious_root_emergence
                thermal_time_since_last_adventitious_root_emergence = n.thermal_potential_time_since_emergence

            # If the elongated apex corresponded to any primordium that has emerged:
            if n.type == "adventitious_root_before_emergence" or n.type == "Normal_root_before_emergence":

                # We select the parent from which the primordium has emerged:
                if n.type == "adventitious_root_before_emergence":
                    parent = g.node(1)
                    adventitious_root_emergence = "Impossible"
                else:
                    index_parent = g.Father(n.index(), EdgeType='+')
                    parent = g.node(index_parent)

                # We now consider the apex to have emerged:
                n.type = "Normal_root_after_emergence"

                # The possibility of emergence of a lateral root from the parent is forbidden again:
                parent.lateral_emergence_possibility = "Impossible"

                # The exact time since emergence is recorded:
                n.thermal_time_since_emergence = n.thermal_potential_time_since_emergence
                n.actual_time_since_emergence = n.thermal_time_since_emergence / temperature_time_adjustment
                # The actual elongation rate is calculated:
                n.actual_elongation = n.length - n.initial_length
                n.actual_elongation_rate = n.actual_elongation / n.actual_time_since_emergence
                # Note: at this stage, no sugar has been allocated to the emerging primordium itself!

            elif n.type == "Normal_root_after_emergence":
                # The actual elongation rate is calculated:
                n.actual_elongation = n.length - n.initial_length
                n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

            # The distance to the last ramification is increased:
            n.dist_to_ramif += n.actual_elongation

    return g


# Function calculating a satisfaction coefficient for the growth of the whole root system, based on the "available struct_mass":
def satisfaction_coefficient(g, struct_mass_input):
    # We initialize the sum of individual demands for struct_mass:
    sum_struct_mass_demand = 0.
    SC = 0.

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):
        # n represents the current root element:
        n = g.node(vid)

        # We calculate the initial volume of the element:
        initial_volume = surfaces_and_volumes(n, n.initial_radius, n.initial_length)["volume"]
        # We calculate the potential volume of the element based on the potential radius and potential length:
        potential_volume = surfaces_and_volumes(n, n.potential_radius, n.potential_length)["volume"]

        # The growth demand of the element in struct_mass is calculated:
        n.growth_demand_in_struct_mass = (potential_volume - initial_volume) * root_tissue_density
        sum_struct_mass_demand += n.growth_demand_in_struct_mass

    # We calculate the overall satisfaction coefficient SC described by Pages et al. (2014):
    if sum_struct_mass_demand <= 0:
        SC = 1.
    else:
        SC = struct_mass_input / sum_struct_mass_demand

    return SC


# Function performing the actual growth of each element based on the potential growth and the satisfaction coefficient SC:
def ArchiSimple_growth(g, SC, time_step_in_seconds, soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    SC is the satisfaction coefficient for growth calculated on the whole root system.
    """

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # CALCULATING AN EQUIVALENT OF THERMAL TIME:
    # -------------------------------------------

    # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
    temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                           process_at_T_ref=1,
                                                           T_ref=T_ref_growth,
                                                           A=growth_increase_with_temperature,
                                                           B=1,
                                                           C=0)

    # PERFORMING ARCHISIMPLE GROWTH:
    # -------------------------------

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):

        # n represents the current root element:
        n = g.node(vid)

        # We make sure that the element is not dead and has not already been stopped at the previous time step:
        if n.type == "Dead" or n.type == "Just_dead" or n.type == "Stopped":
            # Then we pass to the next element in the iteration:
            continue
        # We make sure that the root elements at the basis that support adventitious root are not considered:
        if n.type == "Support_for_adventitious_root":
            # Then we pass to the next element in the iteration:
            continue

        # We perform each type of growth according to the satisfaction coefficient SC:
        if SC > 1.:
            relative_reduction = 1.
        else:
            relative_reduction = SC

        # WARNING: This approach is not an exact C balance on the root system! The relative reduction of growth caused
        # by SC should not be the same between elongation and radial growth!
        n.length += (n.potential_length - n.initial_length) * relative_reduction
        n.actual_elongation = n.length - n.initial_length

        # We calculate the actual elongation rate of this element:
        if (n.thermal_potential_time_since_emergence > 0) and (n.thermal_potential_time_since_emergence < time_step_in_seconds):
            n.actual_elongation_rate = n.actual_elongation / (
                    n.thermal_potential_time_since_emergence / temperature_time_adjustment)
        else:
            n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

        n.radius += (n.potential_radius - n.initial_radius) * relative_reduction
        # The volume of the element is automatically calculated:
        n.volume = surfaces_and_volumes(n, n.radius, n.length)["volume"]
        # The new dry structural struct_mass of the element is calculated from its new volume:
        n.struct_mass = n.volume * root_tissue_density

        # In case where the root element corresponds to an apex, the distance to the last ramification is increased:
        if n.label == "Apex":
            n.dist_to_ramif += n.actual_elongation

        # VERIFICATION:
        if n.length < 0 or n.struct_mass < 0:
            print("!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length, "and a mass of", n.struct_mass)

    return g


def reinitializing_growth_variables(g):
    global adventitious_root_emergence
    adventitious_root_emergence = "Possible"

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We set to 0 the growth-related variables:
        n.hexose_consumption_by_growth = 0.
        n.resp_growth = 0.
        n.struct_mass_produced = 0.
        n.hexose_growth_demand = 0.
        n.actual_elongation = 0.
        n.actual_elongation_rate = 0.

        # We make sure that the initial values of length, radius and struct_mass are correctly initialized:
        n.initial_length = n.length
        n.initial_radius = n.radius
        n.potential_radius = n.radius
        n.theoretical_radius = n.radius
        n.initial_struct_mass = n.struct_mass


# FUNCTION: FORMATION OF NODULES
################################

def nodule_formation(mother_element,
                     time_step_in_seconds=1. * 60. * 60. * 24.,
                     soil_temperature_in_Celsius=20,
                     random=True):
    """
    This function simulates the formation of one nodule on a root mother element. The nodule is considered as a special
    lateral root segment that has no apex connected to it.
    """

    # We add a lateral root element called "nodule" on the mother element:
    nodule = ADDING_A_CHILD(mother_element, edge_type='+', label='Segment', type='Root_nodule',
                            angle_down=90, angle_roll=0, length=0, radius=0,
                            identical_properties=False, nil_properties=True)
    nodule.type = "Root_nodule"
    # nodule.length=mother_element.radius
    # nodule.radius=mother_element.radius/10.
    nodule.length = mother_element.radius
    nodule.radius = mother_element.radius
    nodule.original_radius = nodule.radius
    dict = surfaces_and_volumes(element=nodule, radius=nodule.radius, length=nodule.length)
    nodule.external_surface = dict['external_surface']
    nodule.volume = dict['volume']
    nodule.phloem_surface = dict['phloem_surface']
    nodule.symplasm_surface = dict['symplasm_surface']
    nodule.struct_mass = nodule.volume * root_tissue_density * struct_mass_C_content

    print("Nodule", nodule.index(), "has been formed!")

    return nodule


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "SUCROSE SUPPLY FROM THE SHOOTS"
########################################################################################################################
########################################################################################################################

# Calculation of the total amount of sucrose and structural struct_mass in the root system:
# -----------------------------------------------------------------------------------------
def total_root_sucrose_and_living_struct_mass(g):
    """
    This function computes the total amount of sucrose of the root system (in mol of sucrose),
    and the total dry structural mass of the root system (in g of dry structural mass).
    :param g: the investigated MTG
    :return: total_sucrose_root(mol of sucrose), total_struct_mass (g of dry structural mass)
    """

    # We initialize the values to 0:
    total_sucrose_root = 0.
    total_living_struct_mass = 0.

    # We cover all the vertices in the MTG, whether they are dead or not:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # We increment the total amount of sucrose in the root system:
        # total_sucrose_root += (n.C_sucrose_root * n.struct_mass)
        total_sucrose_root += (n.C_sucrose_root * n.struct_mass - n.Deficit_sucrose_root)
        # We only select the elements that have a positive struct_mass and are not dead:
        if n.struct_mass > 0. and n.type != "Dead":  # and n.type != "Just dead":#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # We calculate the total living struct_mass by summing all the local struct_masses:
            total_living_struct_mass += n.struct_mass

        # if n.type=="Dead" or n.type=="Just_dead":
        #     print "When calculating total sucrose in the root, the element", n.index(), "of type", n.type, \
        #         "has contributed by",(n.C_sucrose_root * n.struct_mass - n.Deficit_sucrose_root),\
        #         "because C_sucrose_root was", n.C_sucrose_root, "and the deficit was", n.Deficit_sucrose_root

    # We return a list of two numeric values:
    return total_sucrose_root, total_living_struct_mass


# Calculating the net input of sucrose by the aerial parts into the root system:
# ------------------------------------------------------------------------------
def shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=1e-9, time_step_in_seconds=1. * 60. * 60. * 24.,
                                       printing_warnings=False):
    """
    This function calculates the new root sucrose concentration (mol of sucrose per gram of dry root structural mass)
    AFTER the supply of sucrose from the shoot.
    """

    # The input of sucrose over this time step is calculated
    # from the sucrose transport rate provided as input of the function:
    sucrose_input = sucrose_input_rate * time_step_in_seconds

    # We calculate the remaining amount of sucrose in the root system,
    # based on the current sucrose concentration and struct_mass of each root element:
    total_sucrose_root, total_living_struct_mass = total_root_sucrose_and_living_struct_mass(g)
    # Note: The total sucrose is the total amount of sucrose present in the root system, including local deficits
    # BUT NOT INCLUDING A POSSIBLE GLOBAL DEFICIT!
    # The total struct_mass only corresponds to the structural mass of the living roots.

    # We use a global variable recorded outside this function, that corresponds to the possible deficit of sucrose
    # (in moles of sucrose) of the whole root system calculated at the previous time_step:
    global global_sucrose_deficit
    if global_sucrose_deficit > 0.:
        print("!!! Before homogenizing sucrose concentration, the deficit in sucrose is", global_sucrose_deficit)
        C_sucrose_root_after_supply = (
                                              total_sucrose_root + sucrose_input) / total_living_struct_mass  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    else:
        # The new average sucrose concentration in the root system is calculated as:
        C_sucrose_root_after_supply = (total_sucrose_root + sucrose_input - global_sucrose_deficit) / total_living_struct_mass

    if C_sucrose_root_after_supply >= 0.:
        new_C_sucrose_root = C_sucrose_root_after_supply
        # We reset the global variable global_sucrose_deficit:
        global_sucrose_deficit = 0.
    else:
        # We record the general deficit in sucrose:
        global_sucrose_deficit = - C_sucrose_root_after_supply * total_living_struct_mass
        print("!!! After homogenizing sucrose concentration, the deficit in sucrose is", global_sucrose_deficit)
        # We defined the new concentration of sucrose as 0:
        new_C_sucrose_root = 0.

    # We go through the MTG to modify the sugars concentrations:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)
        # If the element has not emerged yet, it doesn't contain any sucrose yet;
        # if is dead, it should not contain any sucrose anymore:
        if n.length <= 0. or n.type == "Dead":  # or n.type == "Just_dead":!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            n.C_sucrose_root = 0.
        else:
            # The local sucrose concentration in the root is calculated from the new sucrose concentration calculated above:
            n.C_sucrose_root = new_C_sucrose_root

        # AND BECAUSE THE LOCAL DEFICITS OF SUCROSE HAVE BEEN ALREADY INCLUDED IN THE TOTAL SUCROSE CALCULATION,
        # WE RESET ALL LOCAL DEFICITS TO 0:
        n.Deficit_sucrose_root = 0.

    return g


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "EXCHANGE BETWEEN SUCROSE AND HEXOSE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def exchange_with_phloem(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                         soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "exchange_with_phloem" simulates the process of sucrose unloading from phloem over time
    (in seconds) and its immediate conversion into hexose, for a given root element with an external surface (m2).
    It also simulates the process of sucrose loading (if any) that works in the other way.
    It returns the variable hexose_production_from_phloem (in mol of hexose) and sucrose_loading_in_phloem (in mol of sucrose),
    considering that 2 mol of hexose are produced for 1 mol of sucrose.
    The unloading of sucrose is represented as an active process with a substrate-limited relationship
    (Michaelis-Menten function), where unloading_coeff (in mol m-2 s-1) is the maximal amount of sucrose unloading
    and Km_unloading (in mol per gram of root structural struct_mass) represents the sucrose concentration
    for which the rate of hexose production is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the unloading coefficient and the production of hexose:
        n.unloading_coeff = 0.
        n.hexose_production_from_phloem = 0.
        n.sucrose_loading_in_phloem = 0.
        n.net_sucrose_unloading = 0.

        # We verify that the element does not correspond to a primordium that has not emerged:
        if n.length <= 0.:
            continue
        # If the element has died or is already dead, we consider that there is possible exchange:
        if n.type == "Dead" or n.type == "Just_dead":
            continue

        # We verify that the concentration of sucrose and hexose in root are not negative:
        if n.C_sucrose_root < 0. or n.C_hexose_root < 0.:
            if printing_warnings:
                print("WARNING: No exchange with phloem occurred for node", n.index(),
                      "because root sucrose concentration was", n.C_sucrose_root,
                      "mol/g and root hexose concentration was", n.C_hexose_root, "mol/g.")
            continue

        # We verify that the concentration of sucrose and hexose in root are not negative:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n.Deficit_sucrose_root > 0. or n.Deficit_hexose_root > 0.:
            if printing_warnings:
                print("WARNING: No exchange with phloem occurred for node", n.index(),
                      "because there was deficit in sucrose of", n.Deficit_sucrose_root,
                      "mol/g and in hexose of", n.Deficit_hexose_root, "mol/g.")
            continue

        # We calculate the current external surface of the element:
        n.phloem_surface = surfaces_and_volumes(n, n.radius, n.length)["phloem_surface"]
        # We calculate the surface (m2) corresponding to the section of an emerging primordium (if any):
        if n.lateral_emergence_possibility == "Possible":
            primordium = g.node(n.lateral_primordium_index)
            primordium_section = pi * primordium.radius ** 2
        else:
            primordium_section = 0.

        # OPTION 1: UNLOADING THROUGH TRANSPORTER (Michaelis-Menten function)
        # --------------------------------------------------------------------

        # We calculate the maximal unloading rate according to the net surface of exchange of the phloem
        # and the possibility of an extra unloading through the section of the possibly emerging primordium:
        n.max_unloading_rate = surfacic_unloading_rate_reference * n.phloem_surface \
                               + surfacic_unloading_rate_primordium * primordium_section
        # We calculate the maximal loading rate according to the net surface of exchange of the phloem:
        n.max_loading_rate = surfacic_loading_rate_reference * n.phloem_surface

        # We deal with special cases:
        if n.type == "Stopped" or n.type == "Just_stopped":
            # If the element has stopped its growth, we decrease its unloading coefficient:
            n.max_unloading_rate = n.max_unloading_rate / 50.
            n.max_loading_rate = n.max_loading_rate / 50

        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if n.type == "Root_nodule":
        #     n.max_unloading_rate = n.max_unloading_rate/10.
        #     n.max_loading_rate = n.max_loading_rate/100.

        # We correct both loading and unloading rates according to soil temperature:
        n.max_unloading_rate = n.max_unloading_rate * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=10,
            A=-0.02,
            B=2,
            C=1)
        n.max_loading_rate = n.max_loading_rate * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=10,
            A=-0.02,
            B=2,
            C=1)
        #
        # # # In case n corresponds to an apex, we increase the unloading:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # # if n.label=="Apex":
        # #     n.max_unloading_rate = n.max_unloading_rate * 2.

        # We calculate the potential production of hexose from sucrose (in mol) according to the Michaelis-Menten function:
        # n.hexose_production_from_phloem = 2. * n.max_unloading_rate * n.C_sucrose_root \
        #                         / (Km_unloading + n.C_sucrose_root) * time_step_in_seconds
        # The factor 2 originates from the conversion of 1 molecule of sucrose into 2 molecules of hexose.

        # OPTION 2: UNLOADING THROUGH DIFFUSION ONLY (non-saturable function)
        # --------------------------------------------------------------------

        # # We have the permeability of the phloem vessels evolve with distance from apex:
        # if n.label == "Apex":
        #     n.phloem_permeability = phloem_permeability
        # elif n.lateral_emergence_possibility == "Possible":
        #     n.phloem_permeability = phloem_permeability
        # else:
        #     n.phloem_permeability = phloem_permeability / (1 + n.dist_to_tip / n.original_radius) ** gamma_unloading

        # # We deal with special cases:
        # if n.type == "Stopped" or n.type == "Just_stopped":
        #     # If the element has stopped its growth, we decrease its unloading coefficient:
        #     n.phloem_permeability = n.phloem_permeability / 50.

        n.phloem_permeability = phloem_permeability

        # We deal with special cases:
        if n.type == "Stopped" or n.type == "Just_stopped":
            # If the element has stopped its growth, we decrease its unloading coefficient:
            n.phloem_permeability = n.phloem_permeability / 50.

        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if n.type == "Root_nodule":
        #     n.phloem_permeability = n.phloem_permeability*1.

        n.hexose_production_from_phloem = 2. * n.phloem_permeability * (n.C_sucrose_root - n.C_hexose_root / 2.) \
                                          * n.phloem_surface * time_step_in_seconds

        # We make sure that hexose production can't become negative:
        if n.hexose_production_from_phloem < 0.:
            n.hexose_production_from_phloem = 0.

        # Loading:
        # ---------

        # We correct the max loading rate according to the distance from the tip
        n.max_loading_rate = n.max_loading_rate * (1. - 1. / (1. + n.dist_to_tip / n.original_radius) ** gamma_loading)

        # We calculate the potential production of sucrose from hexose (in mol) according to the Michaelis-Menten function:!!!!!!!!!!!!!!!!!!!WHERE IS S ????
        n.sucrose_loading_in_phloem = 0.5 * n.max_loading_rate * n.C_hexose_root \
                                      / (Km_loading + n.C_hexose_root) * time_step_in_seconds
        n.net_sucrose_unloading = n.hexose_production_from_phloem / 2. - n.sucrose_loading_in_phloem

    return g


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "EXCHANGE BETWEEN MOBILE HEXOSE AND RESERVE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def exchange_with_reserve(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                          soil_temperature_in_Celsius=20, printing_warnings=False):
    """

    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the unloading coefficient and the production of hexose:
        n.hexose_mobilization_from_reserve = 0.
        n.hexose_immobilization_as_reserve = 0.
        n.net_hexose_immobilization = 0.

        # We verify that the element does not correspond to a primordium that has not emerged:
        if n.length <= 0.:
            continue
        # We verify that the concentration of sucrose and hexose in root are not negative:
        if n.C_hexose_root < 0. or n.C_hexose_reserve < 0.:
            if printing_warnings:
                print("WARNING: No exchange with phloem occurred for node", n.index(),
                      "because root sucrose concentration was", n.C_sucrose_root,
                      "mol/g, root hexose concentration was", n.C_hexose_root,
                      "mol/g, and hexose reserve concentration was", n.C_hexose_reserve)
            continue

        # If the element was already dead at the beginning of the time step, we don't consider it
        # (and if it has just died, we set its max reserve concentration to 0, see below):
        if n.type == "Dead":
            continue

        # If the element was already dead at the beginning of the time step, we don't consider it
        # (and if it has just died, we set its max reserve concentration to 0, see below):
        if n.type == "Root_nodule":
            continue

        # CALCULATION OF THE MAXIMAL CONCENTRATION OF HEXOSE IN THE RESERVE POOL:
        if n.type == "Just_dead":
            # If the element has just died, all reserve is emptied over this time step:
            n.hexose_mobilization_from_reserve = n.C_hexose_reserve * n.struct_mass
            n.C_hexose_reserve = 0.
            # And the immobilization rate remains 0.
            # And we move to the next element:
            continue

        # The maximal concentration in the reserve is defined:
        n.C_hexose_reserve_max = C_hexose_reserve_max

        # We correct max loading and unloading rates according to soil temperature:
        corrected_max_mobilization_rate = max_mobilization_rate \
                                          * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                     process_at_T_ref=1,
                                                                     T_ref=10,
                                                                     A=-0.02,
                                                                     B=2,
                                                                     C=1)
        corrected_max_immobilization_rate = max_immobilization_rate \
                                            * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=10,
            A=-0.02,
            B=2,
            C=1)

        # CALCULATIONS OF THEORETICAL MOBILIZATION / IMMOBILIZATION RATES:
        # We calculate the potential mobilization of hexose from reserve (in mol) according to the Michaelis-Menten function:
        n.hexose_mobilization_from_reserve = corrected_max_mobilization_rate * n.C_hexose_reserve \
                                             / (
                                                     Km_mobilization + n.C_hexose_reserve) * time_step_in_seconds * n.struct_mass
        # We calculate the potential immobilization of hexose as reserve (in mol) according to the Michaelis-Menten function:
        if n.C_hexose_root < C_hexose_root_min_for_reserve:
            # If the concentration of mobile hexose is already too low, there is no immobilization:
            n.hexose_immobilization_as_reserve = 0.
        else:
            n.hexose_immobilization_as_reserve = corrected_max_immobilization_rate * n.C_hexose_root \
                                                 / (
                                                         Km_immobilization + n.C_hexose_root) * time_step_in_seconds * n.struct_mass

        # CARBON BALANCE AND ADJUSTMENTS:
        # We control the balance on the reserve by calculating the new theoretical concentration in the reserve pool:
        C_hexose_reserve_new = (n.C_hexose_reserve * n.initial_struct_mass
                                + n.hexose_immobilization_as_reserve - n.hexose_mobilization_from_reserve) / n.struct_mass

        # If the new concentration is lower than the minimal concentration:
        if C_hexose_reserve_new < C_hexose_reserve_min:
            # The amount of hexose that can be mobilized is lowered, while the amount of hexose immobilized is not modified:
            n.hexose_mobilization_from_reserve = n.hexose_mobilization_from_reserve - (
                    C_hexose_reserve_min - C_hexose_reserve_new) * n.struct_mass
            # And we set the concentration to the minimal concentration:
            n.C_hexose_reserve = C_hexose_reserve_min
        # Otherwise, if the concentration in the reserve is higher than the maximal one:
        elif C_hexose_reserve_new > n.C_hexose_reserve_max:
            # The mobilized amount is not modified and the maximal amount of hexose that can be immobilized is reduced:
            n.hexose_immobilization_as_reserve = n.hexose_immobilization_as_reserve - (
                    C_hexose_reserve_new - n.C_hexose_reserve_max) * n.struct_mass
            # And we limit the concentration to the maximal concentration:
            n.C_hexose_reserve = n.C_hexose_reserve_max
        # Else, the mobilization and immobilization processes are unchanged, and we record the final concentration as the expected one:
        else:
            n.C_hexose_reserve = C_hexose_reserve_new

        # In any case, the net balance is recorded:
        n.net_hexose_immobilization = n.hexose_immobilization_as_reserve - n.hexose_mobilization_from_reserve

        if n.hexose_immobilization_as_reserve < 0. or n.hexose_mobilization_from_reserve < 0.:
            print("!!! ERROR: hexose immobilization and mobilization rate were",
                  n.hexose_immobilization_as_reserve, "and", n.hexose_mobilization_from_reserve)

    return


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "ROOT MAINTENANCE"
########################################################################################################################
########################################################################################################################

# Function calculating maintenance respiration:
def maintenance_respiration(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                            soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "maintenance" calculates the amount resp_maintenance (mol of CO2) corresponding to the consumption
    of a part of the local hexose pool to cover the costs of maintenance processes, i.e. any biological process in the
    root that is NOT linked to the actual growth of the root. The calculation is derived from the model of Thornley and
    Cannell (2000), who initially used this formalism to describe the residual maintenance costs that could not be
    accounted for by known processes. The local amount of CO2 respired for maintenance is calculated as a
    Michaelis-Menten function of the local concentration of hexose "C_hexose_root" (in mol of hexose per gram of root
    structural struct_mass. "g" represents the MTG describing the root system, "resp_maintenance__max" (mol of CO2 per gram
    of root structural struct_mass per second) is the maximal rate of maintenance respiration, and "Km_maintenance" (mol of
    hexose per gram of root structural struct_mass) represents the hexose concentration for which the rate of respiration is
    equal to half of its maximum. "struct_mass" is the root structural struct_mass (g) and time is expressed in seconds.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the maintenance respiration:
        n.resp_maintenance = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We consider that dead elements cannot respire (unless over the first time step following death,
        # i.e. when the type is "Just_dead"):
        if n.type == "Dead":
            continue
        # We also check whether the concentration of hexose in root is positive or not:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No maintenance occurred for node", n.index(),
                      "because root hexose concentration was", n.C_hexose_root, "mol/g.")
            continue

        # We correct the maximal respiration according to soil temperature:
        corrected_resp_maintenance_max = resp_maintenance_max \
                                         * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                    process_at_T_ref=1,
                                                                    T_ref=10,
                                                                    A=0,
                                                                    B=2,
                                                                    C=1)

        # We calculate the number of moles of CO2 generated by maintenance respiration over the time_step:
        n.resp_maintenance = corrected_resp_maintenance_max * n.C_hexose_root / (Km_maintenance + n.C_hexose_root) \
                             * n.struct_mass * time_step_in_seconds

        if n.resp_maintenance < 0.:
            print("!!! ERROR: a negative maintenance respiration was calculated for the element", n.index())
            n.resp_maintenance = 0.


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "RHIZODEPOSITION"
########################################################################################################################
########################################################################################################################

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def root_hexose_exudation(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                          soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_hexose_exudation" computes the net amount (in mol of hexose) of hexose accumulated
    outside the root over time (in seconds), without considering any degradation process of hexose
    outside the root or hexose uptake by the root.
    Exudation corresponds to the difference between the efflux of hexose from the root
    to the soil by a passive diffusion. The efflux by diffusion is calculated from the product of the root external
    surface (m2), the permeability coefficient (g m-2) and the gradient of hexose concentration (mol of hexose per gram of dry
    root structural struct_mass).
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the exudation of hexose in the element:
        n.hexose_exudation = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We also check whether the concentration of hexose in root is positive or not:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print("WARNING: No hexose exudation occurred for node", n.index(),
                      "because root hexose concentration was", n.C_hexose_root, "mol/g.")
            continue

        # We calculate the permeability coefficient P according to the distance of the element from the apex:
        # OPTION 1 (Personeni et al. 2007):
        # n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip*100) ** gamma_exudation
        # OPTION 2:
        if n.label == "Apex":
            n.permeability_coeff = Pmax_apex
        elif n.lateral_emergence_possibility == "Possible":
            n.permeability_coeff = Pmax_apex
        else:
            n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip / n.original_radius) ** gamma_exudation
        # OPTION 3: Pmax is the same everywhere

        # We calculate the total surface of exchange between symplasm and apoplasm in the root cortex + epidermis:
        n.symplasm_surface = surfaces_and_volumes(n, n.radius, n.length)["symplasm_surface"]

        # We correct the permeability coefficient according to soil temperature:
        corrected_permeability_coeff = n.permeability_coeff \
                                       * temperature_modification(temperature_in_Celsius=soil_temperature_in_Celsius,
                                                                  process_at_T_ref=1,
                                                                  T_ref=10,
                                                                  A=0.01,
                                                                  B=1,
                                                                  C=1)

        # Then exudation is calculated as an efflux by diffusion, even for dead root elements:
        n.hexose_exudation \
            = n.symplasm_surface * corrected_permeability_coeff * (
                n.C_hexose_root - n.C_hexose_soil) * time_step_in_seconds
        if n.hexose_exudation < 0.:
            if printing_warnings:
                print("WARNING: a negative exudation flux was calculated for the element", n.index(),
                      "; exudation flux has therefore been set up to zero!")
            n.hexose_exudation = 0.

        # NOTE : We consider that dead elements can also liberate hexose in the soil, until they are empty.


# Uptake of hexose from the soil by the root:
# -------------------------------------------
def root_hexose_uptake(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                       soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "root_hexose_uptake" computes the amount (in mol of hexose) of hexose taken up by roots from the soil.
    This influx of hexose is represented as an active process with a substrate-limited
    relationship (Michaelis-Menten function), where uptake_rate_max (in mol) is the maximal influx, and Km_uptake
    (in mol per gram of root structural struct_mass) represents the hexose concentration for which
    hexose_degradation is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the uptake of hexose in the element:
        n.hexose_uptake = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_hexose_soil <= 0.:
            if printing_warnings:
                print("WARNING: No uptake of hexose from the soil occurred for node", n.index(),
                      "because soil hexose concentration was", n.C_hexose_soil, "mol/g.")
            continue
        # We consider that dead elements cannot take up any hexose from the soil:
        if n.type == "Just_dead" or n.type == "Dead":
            continue

        # We correct the maximal uptake rate according to soil temperature:
        corrected_uptake_rate_max = uptake_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=10,
            A=-0.02,
            B=2,
            C=1)

        # We calculate the total surface of exchange between symplasm and apoplasm in the root cortex + epidermis:
        n.symplasm_surface = surfaces_and_volumes(n, n.radius, n.length)["symplasm_surface"]
        # The uptake of hexose by the root from the soil is calculated:
        n.hexose_uptake \
            = n.symplasm_surface * corrected_uptake_rate_max * n.C_hexose_soil / (
                Km_uptake + n.C_hexose_soil) * time_step_in_seconds


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MODULE "SOIL TRANSFORMATION"
########################################################################################################################
########################################################################################################################

# Degradation of hexose in the soil (microbial consumption):
# ---------------------------------------------------------
def soil_hexose_degradation(g, time_step_in_seconds=1. * (60. * 60. * 24.),
                            soil_temperature_in_Celsius=20, printing_warnings=False):
    """
    The function "hexose_degradation" computes the decrease of the concentration of hexose outside the root (in mol of
    hexose per gram of root structural mass) over time (in seconds). It mimics the uptake of hexose by rhizosphere
    microorganisms, and is therefore described using a substrate-limited function (Michaelis-Menten). g represents the
    MTG describing the root system, degradation_rate_max is the maximal degradation of hexose (mol m-2), and Km_degradation
    (mol per gram of root structural mass) represents the hexose concentration for which the rate of hexose_degradation
    is equal to half of its maximum. The surface of the symplasm rather than the external surface of the root element
    is taken into account here, similarly to what is done for exudation or re-uptake of hexose by the root.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the degradation of hexose:
        n.hexose_degradation = 0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0.:
            continue

        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_hexose_soil <= 0.:
            if printing_warnings:
                print("WARNING: No degradation in the soil occurred for node", n.index(),
                      "because soil hexose concentration was", n.C_hexose_soil, "mol/g.")
            continue

        # We correct the maximal degradation rate according to soil temperature:
        corrected_degradation_rate_max = degradation_rate_max * temperature_modification(
            temperature_in_Celsius=soil_temperature_in_Celsius,
            process_at_T_ref=1,
            T_ref=20,
            A=0,
            B=3.82,
            C=1)
        # => The value for B (Q10) is adapted from the work of Coody et al. (1986, SBB),
        # who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.

        # We calculate the total surface of exchange between symplasm and apoplasm in the root cortex + epidermis:
        n.symplasm_surface = surfaces_and_volumes(n, n.radius, n.length)["symplasm_surface"]
        # hexose_degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.hexose_degradation = n.symplasm_surface * corrected_degradation_rate_max * n.C_hexose_soil \
                               / (Km_degradation + n.C_hexose_soil) * time_step_in_seconds

    return


########################################################################################################################

########################################################################################################################
########################################################################################################################
# MAIN PROGRAM:
########################################################################################################################
########################################################################################################################

# CARBON BALANCE:
#################

def balance(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We exclude root elements that have not emerged yet:
        if n.length <= 0.:
            continue

        # CREATING NEW PROPERTIES:
        # We calculate the net exudation of hexose (in mol of hexose):
        n.net_hexose_exudation = n.hexose_exudation - n.hexose_uptake
        # We calculate the total biomass of each element, including the structural mass and all sugars:
        n.biomass = n.struct_mass + (
                n.C_hexose_root * 6 * 12.01 + n.C_hexose_reserve * 6 * 12.01 + n.C_sucrose_root * 12 * 12.01) * n.struct_mass
        # We calculate a net rate of exudation, in gram of C per gram of dry structural mass per day:
        n.net_hexose_exudation_rate_per_day_per_gram = (
                                                               n.net_hexose_exudation / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.struct_mass

        # We calculate a net rate of exudation, in gram of C per cm of root per day:
        n.net_hexose_exudation_rate_per_day_per_cm = (
                                                             n.net_hexose_exudation / time_step_in_seconds) * 24. * 60. * 60. * 6. * 12.01 / n.length / 100

        # BALANCE ON HEXOSE AT THE SOIL/ROOT INTERFACE:
        # We calculate the new concentration of hexose in the soil according to hexose degradation, exudation and uptake:
        n.C_hexose_soil = (n.C_hexose_soil * n.initial_struct_mass - n.Deficit_hexose_soil
                           - n.hexose_degradation + n.hexose_exudation - n.hexose_uptake) / n.struct_mass
        # We reset the deficit to 0:
        n.Deficit_hexose_soil = 0.
        if n.C_hexose_soil < 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit of soil hexose for element", n.index(),
                      "that corresponds to", n.Deficit_hexose_soil,
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_hexose_soil = -n.C_hexose_soil * n.struct_mass
            # And we set the concentration to 0:
            n.C_hexose_soil = 0.

        # BALANCE ON HEXOSE IN THE ROOT CYTOPLASM:
        # We calculate the new concentration of hexose in the root cytoplasm:
        n.C_hexose_root = (n.C_hexose_root * n.initial_struct_mass - n.Deficit_hexose_root
                           - n.hexose_exudation + n.hexose_uptake
                           - n.resp_maintenance / 6. - n.hexose_consumption_by_growth
                           + n.hexose_production_from_phloem - 2. * n.sucrose_loading_in_phloem
                           + n.hexose_mobilization_from_reserve - n.hexose_immobilization_as_reserve) / n.struct_mass
        # We reset the deficit to 0:
        n.Deficit_hexose_root = 0.
        if n.C_hexose_root < 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit of root hexose for element", n.index(),
                      "that corresponds to", n.Deficit_hexose_root,
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_hexose_root = - n.C_hexose_root * n.struct_mass
            # And we set the concentration to 0:
            n.C_hexose_root = 0.

        # # BALANCE ON HEXOSE IN THE RESERVE:
        # # We calculate the new concentration of hexose in the reserve pool in the root:
        # n.C_hexose_reserve = (n.C_hexose_reserve*n.initial_struct_mass - n.Deficit_hexose_reserve
        #                    + n.hexose_immobilization_as_reserve - n.hexose_mobilization_from_reserve) / n.struct_mass
        # # We reset the deficit to 0:
        # n.Deficit_hexose_reserve=0.
        # if n.C_hexose_reserve < 0:
        #     if printing_warnings:
        #         print "WARNING: After balance, there is a deficit of reserve hexose for element", n.index() , \
        #             "; the concentration has been set to 0 and the deficit will be included in the next balance."
        #     # We define a positive deficit (mol of hexose) based on the negative concentration:
        #     n.Deficit_hexose_reserve = - n.C_hexose_reserve*n.struct_mass
        #     # And we set the concentration to 0:
        #     n.C_hexose_reserve = 0.

        # if n.type == "Just_dead" or n.type == "Dead":
        #     print "BEFORE BALANCE: The element", n.index(), "is dead, here are its properties:"
        #     print "label:", n.label, "; edge:", n.edge_type(), "; type:", n.type, "; mass:", n.struct_mass, "; C_sucrose_root:", n.C_sucrose_root, \
        #         "; Deficit_sucrose_root:", n.Deficit_sucrose_root, "; loading:", n.sucrose_loading_in_phloem, "; unloading:", n.hexose_production_from_phloem

        # BALANCE ON SUCROSE IN THE ROOT:
        # We calculate the new concentration of sucrose in the root according to sucrose conversion into hexose:
        # (NOTE: The deficit in sucrose is not included in this balance, since it has been considered in the function shoot_supply before)
        n.C_sucrose_root = (n.C_sucrose_root * n.initial_struct_mass
                            + n.sucrose_loading_in_phloem - n.hexose_production_from_phloem / 2.) / n.struct_mass
        # We reset the local deficit to 0:
        n.Deficit_sucrose_root = 0.  # NOTE: it should already have been reset to 0 by the function shoot_supply!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if n.C_sucrose_root < 0:
            # We define a positive deficit (mol of sucrose) based on the negative concentration:
            n.Deficit_sucrose_root = -n.C_sucrose_root * n.struct_mass
            # And we set the concentration to 0:
            if printing_warnings:
                print("WARNING: After balance, there is a deficit in root sucrose for element", n.index(),
                      "that corresponds to", n.Deficit_sucrose_root,
                      "; the concentration has been set to 0 and the deficit will be included in the next balance.")
            n.C_sucrose_root = 0.

            # CHEATING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # We define a positive deficit (mol of sucrose) based on the negative concentration:
            n.Deficit_sucrose_root = 0.
        # PLEASE NOTE: The global (if any) deficit in sucrose is only used by the function "shoot_sucrose_and_spreading"
        # when defining the new homogeneous concentration of sucrose within the root system,
        # and when performing a true carbon balance of the root system in "summing".

        # if n.type == "Just_dead" or n.type == "Dead":
        #     print "AFTER BALANCE: The element", n.index(), "is dead, here are its properties:"
        #     print "label:", n.label, "; edge:", n.edge_type(), "; type:", n.type, "; mass:", n.struct_mass, "; C_sucrose_root:", n.C_sucrose_root, \
        #         "; Deficit_sucrose_root:", n.Deficit_sucrose_root, "; loading:", n.sucrose_loading_in_phloem, "; unloading:", n.hexose_production_from_phloem

        # If the element corresponds to the apex of the primary root:
        if n.radius == D_ini / 2. and n.label == "Apex":
            # Then the function will give its specific concentration of mobile hexose:
            tip_C_hexose_root = n.C_hexose_root

    return tip_C_hexose_root


# Calculation of total amounts and dimensions of the root system:
# ---------------------------------------------------------------
def summing(g, printing_total_length=True, printing_total_struct_mass=True, printing_all=False):
    """
    This function computes a number of general properties summed over the whole MTG.
    :param g: the investigated MTG
    :param printing_total_length: a Boolean defining whether total_length should be printed on the screen or not
    :param printing_total_struct_mass: a Boolean defining whether total_struct_mass should be printed on the screen or not
    :param printing_all: a Boolean defining whether all properties should be printed on the screen or not
    :return: a dictionary containing the numerical value of each property integrated over the whole MTG
    """

    # We initialize the values to 0:
    total_length = 0.
    total_dead_length = 0.
    total_struct_mass = 0.
    total_dead_struct_mass = 0.
    total_surface = 0.
    total_dead_surface = 0.
    total_sucrose_root = 0.
    total_hexose_root = 0.
    total_hexose_reserve = 0.
    total_hexose_soil = 0.
    total_sucrose_root_deficit = 0.
    total_hexose_root_deficit = 0.
    total_hexose_soil_deficit = 0.
    total_respiration = 0.
    total_respiration_root_growth = 0.
    total_respiration_root_maintenance = 0.
    total_struct_mass_produced = 0.
    total_hexose_production_from_phloem = 0.
    total_sucrose_loading_in_phloem = 0.
    total_hexose_immobilization_as_reserve = 0.
    total_hexose_mobilization_from_reserve = 0.
    total_hexose_exudation = 0.
    total_hexose_uptake = 0.
    total_net_hexose_exudation = 0.
    total_hexose_degradation = 0.

    C_in_the_root_soil_system = 0.
    C_degraded = 0.
    C_respired_by_roots = 0.

    global global_sucrose_deficit

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # If the current element has no length, there is no need to include it in the balance:
        if n.length <= 0.:
            continue

        if n.type == "Dead" or n.type == "Just_dead":
            total_dead_struct_mass += n.struct_mass
            total_dead_length += n.length
            total_dead_surface += surfaces_and_volumes(n, n.radius, n.length)["external_surface"]
        elif n.type != "Dead" and n.type != "Just_dead" and n.length > 0.:
            total_length += n.length
            total_struct_mass += n.struct_mass
            total_surface += surfaces_and_volumes(n, n.radius, n.length)["external_surface"]

        total_sucrose_root += n.C_sucrose_root * n.struct_mass
        total_hexose_root += n.C_hexose_root * n.struct_mass
        total_hexose_reserve += n.C_hexose_reserve * n.struct_mass
        total_hexose_soil += n.C_hexose_soil * n.struct_mass

        # if n.type == "Just_dead" or n.type == "Dead":
        #     print "IN SUMMING: The element", n.index(), "has contributed to the total amount of sucrose by", n.C_sucrose_root * n.struct_mass

        total_sucrose_root_deficit += n.Deficit_sucrose_root
        total_hexose_root_deficit += n.Deficit_hexose_root
        total_hexose_soil_deficit += n.Deficit_hexose_soil

        total_respiration += n.resp_maintenance + n.resp_growth
        total_respiration_root_growth += n.resp_growth
        total_respiration_root_maintenance += n.resp_maintenance
        total_struct_mass_produced += n.struct_mass_produced
        total_hexose_production_from_phloem += n.hexose_production_from_phloem
        total_sucrose_loading_in_phloem += n.sucrose_loading_in_phloem
        total_hexose_immobilization_as_reserve += n.hexose_immobilization_as_reserve
        total_hexose_mobilization_from_reserve += n.hexose_mobilization_from_reserve
        total_hexose_exudation += n.hexose_exudation
        total_hexose_uptake += n.hexose_uptake
        total_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake)
        total_hexose_degradation += n.hexose_degradation

        # if n.type == "Just_dead" or n.type == "Dead":
        #     print "Global sucrose deficit is", global_sucrose_deficit
        #     print "The element", n.index(), "is dead, here are its properties:"
        #     print "label:", n.label, "; edge:", n.edge_type(), "; type:", n.type, "; length:", n.length, "; C_sucrose_root:", n.C_sucrose_root, \
        #         "; Deficit_sucrose_root:", n.Deficit_sucrose_root, "; loading:", n.sucrose_loading_in_phloem, "; unloading:", n.hexose_production_from_phloem

    # We add to the sum of local deficits in sucrose the possible global deficit in sucrose used in shoot_supply function:
    total_sucrose_root_deficit += global_sucrose_deficit

    # CARBON BALANCE:
    # --------------
    # We check that the carbon balance is correct (in moles of C):
    C_in_the_root_soil_system = (total_struct_mass + total_dead_struct_mass) * struct_mass_C_content \
                                + (total_sucrose_root - total_sucrose_root_deficit) * 12. \
                                + (total_hexose_root - total_hexose_root_deficit) * 6. \
                                + (total_hexose_reserve) * 6. \
                                + (total_hexose_soil - total_hexose_soil_deficit) * 6.
    C_degraded = total_hexose_degradation * 6.
    C_respired_by_roots = total_respiration

    if printing_total_length:
        print("   New state of the root system:")
        print("      The current total root length is",
              "{:.1f}".format(Decimal(total_length * 100)), "cm.")
    if printing_total_struct_mass:
        print("      The current total root structural mass is",
              "{:.2E}".format(Decimal(total_struct_mass)), "g, i.e.",
              "{:.2E}".format(Decimal(total_struct_mass * struct_mass_C_content)), "mol of C.")
    if printing_all:
        print("      The current total dead root struct_mass is",
              "{:.2E}".format(Decimal(total_dead_struct_mass)), "g, i.e.",
              "{:.2E}".format(Decimal(total_dead_struct_mass * struct_mass_C_content)), "mol of C.")
        print("      The current amount of sucrose in the roots (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_sucrose_root - total_sucrose_root_deficit)), "mol of sucrose, i.e.",
              "{:.2E}".format(Decimal((total_sucrose_root - total_sucrose_root_deficit) * 12)), "mol of C.")
        print("      The current amount of mobile hexose in the roots (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_hexose_root - total_hexose_root_deficit)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal((total_hexose_root - total_hexose_root_deficit) * 6)), "mol of C.")
        print("      The current amount of hexose stored as reserve in the roots is",
              "{:.2E}".format(Decimal(total_hexose_reserve)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_hexose_reserve * 6)), "mol of C.")
        print("      The current amount of hexose in the soil (including possible deficit and dead roots) is",
              "{:.2E}".format(Decimal(total_hexose_soil - total_hexose_soil_deficit)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal((total_hexose_soil - total_hexose_soil_deficit) * 6)), "mol of C.")
        print("      The total amount of CO2 respired by roots over this time step was",
              "{:.2E}".format(Decimal(total_respiration)), "mol of C, including",
              "{:.2E}".format(Decimal(total_respiration_root_growth)), "mol of C for growth and",
              "{:.2E}".format(Decimal(total_respiration_root_maintenance)), "mol of C for maintenance.")
        print("      The total net amount of hexose exuded by roots over this time step was",
              "{:.2E}".format(Decimal(total_net_hexose_exudation)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_net_hexose_exudation * 6)), "mol of C.")
        print("      The total amount of hexose degraded in the soil over this time step was",
              "{:.2E}".format(Decimal(total_hexose_degradation)), "mol of hexose, i.e.",
              "{:.2E}".format(Decimal(total_hexose_degradation * 6)), "mol of C.")

    dictionary = {"total_living_root_length": total_length,
                  "total_dead_root_length": total_dead_length,
                  "total_living_root_struct_mass": total_struct_mass,
                  "total_dead_root_struct_mass": total_dead_struct_mass,
                  "total_living_root_surface": total_surface,
                  "total_dead_root_surface": total_dead_surface,
                  "total_sucrose_root": total_sucrose_root,
                  "total_hexose_root": total_hexose_root,
                  "total_hexose_reserve": total_hexose_reserve,
                  "total_hexose_soil": total_hexose_soil,
                  "total_sucrose_root_deficit": total_sucrose_root_deficit,
                  "total_hexose_root_deficit": total_hexose_root_deficit,
                  "total_hexose_soil_deficit": total_hexose_soil_deficit,
                  "total_respiration": total_respiration,
                  "total_respiration_root_growth": total_respiration_root_growth,
                  "total_respiration_root_maintenance": total_respiration_root_maintenance,
                  "total_structural_mass_production": total_struct_mass_produced,
                  "total_hexose_production_from_phloem": total_hexose_production_from_phloem,
                  "total_sucrose_loading_in_phloem": total_sucrose_loading_in_phloem,
                  "total_hexose_immobilization_as_reserve": total_hexose_immobilization_as_reserve,
                  "total_hexose_mobilization_from_reserve": total_hexose_mobilization_from_reserve,
                  "total_hexose_exudation": total_hexose_exudation,
                  "total_hexose_uptake": total_hexose_uptake,
                  "total_hexose_degradation": total_hexose_degradation,
                  "total_net_hexose_exudation": total_net_hexose_exudation,
                  "C_in_the_root_soil_system": C_in_the_root_soil_system,
                  "C_degraded_in_the_soil": C_degraded,
                  "C_respired_by_roots": C_respired_by_roots
                  }

    return dictionary


# CHECKING FOR ANOMALIES IN THE MTG:
# ----------------------------------
def control_of_anomalies(g):
    """

    """

    # CHECKING THAT UNEMERGED ROOT ELEMENTS DO NOT CONTAIN CARBON:
    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        if n.length <= 0.:
            if n.C_sucrose_root != 0.:
                print("")
                print("??? ERROR: for element", n.index(), " of length", n.length,
                      "m, the concentration of root sucrose is", n.C_sucrose_root)
            if n.C_hexose_root != 0.:
                print("")
                print("??? ERROR: for element", n.index(), " of length", n.length,
                      "m, the concentration of root hexose is", n.C_hexose_root)
            if n.C_hexose_soil != 0.:
                print("")
                print("??? ERROR: for element", n.index(), " of length", n.length,
                      "m, the concentration of soil hexose is", n.C_hexose_soil)

    return


# INITIALIZATION OF THE MTG
###########################

def initiate_mtg(random=True):
    g = MTG()

    base_radius = D_ini / 2.

    # We first add one initial element:
    root = g.add_component(g.root, label='Segment')
    segment = g.node(root)
    # Characteristics:
    # -----------------
    segment.type = "Base_of_the_root_system"
    # Authorizations and C requirements:
    # -----------------------------------
    segment.lateral_emergence_possibility = 'Impossible'
    segment.emergence_cost = 0.
    # Geometry and topology:
    # -----------------------
    segment.angle_down = 0
    segment.angle_roll = 0
    segment.length = segment_length / 3.
    segment.radius = base_radius
    segment.original_radius = base_radius
    segment.initial_length = segment_length
    segment.initial_radius = base_radius

    surface_dictionary = surfaces_and_volumes(segment, segment.radius, segment.length)
    segment.external_surface = surface_dictionary["external_surface"]
    segment.volume = surface_dictionary["volume"]
    segment.phloem_surface = surface_dictionary["phloem_surface"]
    segment.symplasm_surface = surface_dictionary["symplasm_surface"]

    segment.dist_to_tip = 0.
    segment.dist_to_ramif = 0.
    segment.actual_elongation = segment.length
    segment.actual_elongation_rate = 0
    segment.lateral_primordium_index = 0
    segment.adventitious_emerging_primordium_index = 0
    # Quantities and concentrations:
    # -------------------------------
    segment.struct_mass = segment.volume * root_tissue_density
    segment.initial_struct_mass = segment.struct_mass
    # We define the initial sugar concentrations:
    segment.C_sucrose_root = 1e-3
    segment.C_hexose_root = 1e-3
    segment.C_hexose_reserve = 0.
    segment.C_hexose_soil = 0.
    segment.Deficit_sucrose_root = 0.
    segment.Deficit_hexose_root = 0.
    segment.Deficit_hexose_soil = 0.
    # Fluxes:
    # --------
    segment.resp_maintenance = 0.
    segment.resp_growth = 0.
    segment.hexose_growth_demand = 0.
    segment.hexose_consumption_by_growth = 0.
    segment.hexose_production_from_phloem = 0.
    segment.sucrose_loading_in_phloem = 0.
    segment.hexose_mobilization_from_reserve = 0.
    segment.hexose_immobilization_as_reserve = 0.
    segment.hexose_exudation = 0.
    segment.hexose_uptake = 0.
    segment.hexose_degradation = 0.
    segment.specific_net_exudation = 0.
    # Time indications:
    # ------------------
    segment.growth_duration = GDs * 100 * (
            2. * base_radius) ** 2  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    segment.life_duration = LDs * 2. * base_radius * root_tissue_density
    segment.actual_time_since_primordium_formation = 0.
    segment.actual_time_since_emergence = 0.
    segment.actual_time_since_growth_stopped = 0.
    segment.actual_time_since_death = 0.
    segment.thermal_time_since_primordium_formation = 0.
    segment.thermal_time_since_emergence = 0.
    segment.thermal_potential_time_since_emergence = 0.
    segment.thermal_time_since_growth_stopped = 0.
    segment.thermal_time_since_death = 0.

    # If there should be more than one main root (i.e. adventitious roots formed at the basis):
    if n_adventitious_roots > 1:
        # Then we form one supporting segment of length 0 + one primordium of adventitious root:
        for i in range(1, n_adventitious_roots):
            # We make sure that the adventitious roots will have different random insertion angles:
            np.random.seed(random_choice + i)

            # We add one new segment without any length on the same axis as the base:
            segment = ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Segment',
                                     type='Support_for_adventitious_root',
                                     angle_down=0,
                                     angle_roll=abs(np.random.normal(90, 180)),
                                     length=0.,
                                     radius=base_radius,
                                     identical_properties=False,
                                     nil_properties=True)

            # We define the radius of an adventitious root according to the parameter Di:
            if random:
                radius_adventitious = abs(
                    np.random.normal(D_ini / 2. * D_ini_to_D_adv_ratio, D_ini / 2. * D_ini_to_D_adv_ratio * CVDD))
                if radius_adventitious > D_ini:
                    radius_adventitious = D_ini
            else:
                radius_adventitious = D_ini / 2. * D_ini_to_D_adv_ratio
            # And we add one new primordium of adventitious root on the previously defined segment:
            apex_adventitious = ADDING_A_CHILD(mother_element=segment, edge_type='+', label='Apex',
                                             type='adventitious_root_before_emergence',
                                             angle_down=abs(np.random.normal(60, 10)),
                                             angle_roll=5,
                                             length=0.,
                                             radius=radius_adventitious,
                                             identical_properties=False,
                                             nil_properties=True)
            apex_adventitious.original_radius = radius_adventitious
            apex_adventitious.initial_radius = radius_adventitious
            apex_adventitious.growth_duration = GDs * 100 * (
                    2. * radius_adventitious) ** 2  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Finally, we add the apex that is going to develop the main axis:
    apex = ADDING_A_CHILD(mother_element=segment, edge_type='<', label='Apex',
                          type='Normal_root_after_emergence',
                          angle_down=0,
                          angle_roll=0,
                          length=0.,
                          radius=base_radius,
                          identical_properties=False,
                          nil_properties=True)
    apex.original_radius = apex.radius
    apex.initial_radius = apex.radius
    apex.growth_duration = GDs * 100 * (
            2. * base_radius) ** 2  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    apex.volume = surfaces_and_volumes(apex, apex.radius, apex.length)["volume"]
    apex.struct_mass = apex.volume * root_tissue_density
    apex.initial_struct_mass = apex.struct_mass
    apex.C_sucrose_root = 0.  # 1e-3
    apex.C_hexose_root = 0.  # 1e-3
    apex.C_hexose_reserve = 0.
    apex.C_hexose_soil = 0.
    return g


# SIMULATION OVER TIME:
#######################

def formatted_inputs(original_input_file="None", original_time_step_in_days=1 / 24., final_time_step_in_days=1.,
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
        n_steps = int(floor(simulation_period_in_days / final_time_step_in_days)) + 1

        if do_not_execute_if_file_with_suitable_size_exists:
            PATH2 = os.path.join('.', 'input_file.csv')
            previous_file = pd.read_csv(PATH2, sep=',', header=0)

            if len(previous_file['step_number']) == n_steps:
                print("There is already an 'input_file.csv' of proper length for this simulation, "
                      "we therefore did not create a new input file here (if you wish to do so, please select 'do_not_execute_if_file_with_suitable_size_exists=False').")
                return previous_file

        # We create a new, final dataframe that will be used to read inputs of sucrose:
        input_frame = pd.DataFrame(
            columns=["step_number", "initial_time_in_days", "final_time_in_days", "soil_temperature_in_degree_Celsius",
                     "sucrose_input_rate"])
        input_frame["step_number"] = range(1, n_steps + 1)
        input_frame["initial_time_in_days"] = (input_frame["step_number"] - 1) * final_time_step_in_days * 1.
        input_frame["final_time_in_days"] = input_frame["initial_time_in_days"] + final_time_step_in_days

        print("Creating a new input file adapted to the required time step (time step =", final_time_step_in_days, "days):")

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

                # If the current cumulated time is below the time step of the main simulation:
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
                    print("   Creating line", j, "on", n_steps, "lines in the new input frame...")
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

        # CASE 2: the final time step is higher than the original one, so we have to calculate an average value:
        # -------------------------------------------------------------------------------------------------------
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

                # If the current cumulated time is below the time step of the main simulation:
                if cumulated_time_in_days + final_time_step_in_days < 0.9999 * original_time_step_in_days:
                    # Then the amount of time elapsed here is the initial time step:
                    net_elapsed_time_in_days = final_time_step_in_days
                    remaining_time = 0.
                else:
                    # Otherwise, we limit the elapsed time to the one necessary for reaching a round number of original_time_step_in_days:
                    net_elapsed_time_in_days = final_time_step_in_days - (
                            cumulated_time_in_days + final_time_step_in_days - original_time_step_in_days)
                    remaining_time = (cumulated_time_in_days + final_time_step_in_days - original_time_step_in_days)

                sucrose_input_rate = sucrose_input_rate_initial \
                                     + (df.loc[
                                            i, 'sucrose_input_rate'] - sucrose_input_rate_initial) / original_time_step_in_days * cumulated_time_in_days
                temperature = temperature_initial \
                              + (df.loc[
                                     i, 'soil_temperature_in_Celsius'] - temperature_initial) / original_time_step_in_days * cumulated_time_in_days

                print("   Creating line", j + 1, "on", n_steps, "lines in the new input frame...")
                # We record the final temperature and sucrose input rate:
                input_frame.loc[j, 'sucrose_input_rate'] = sucrose_input_rate
                input_frame.loc[j, 'soil_temperature_in_degree_Celsius'] = temperature

                # If at the end of the final time step corresponding to the line j in input_frame, one original time step has been reached:
                if cumulated_time_in_days + final_time_step_in_days >= 0.9999 * original_time_step_in_days:

                    # We record the current sucrose input rate and temperature in the original table:
                    sucrose_input_rate_initial = df.loc[i, 'sucrose_input_rate']
                    temperature = df.loc[i, 'soil_temperature_in_Celsius']

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

    input_frame.to_csv('input_file.csv', na_rep='NA', index=False, header=True)
    print("The new input file adapted to the required time step has been created and saved as 'input_file.csv'.")

    return input_frame


# We define the main simulation program:
def main_simulation(g, simulation_period_in_days=20., time_step_in_days=1.,
                    radial_growth="Impossible", ArchiSimple=False,
                    property="C_hexose_root", vmin=1e-6, vmax=1e-0, log_scale=True, cmap='brg',
                    input_file="None",
                    constant_sucrose_input_rate=1.e-6,
                    constant_soil_temperature_in_Celsius=20,
                    nodules=False,
                    x_center=0, y_center=0, z_center=-1, z_cam=-1,
                    camera_distance=10., step_back_coefficient=0., camera_rotation=False, n_rotation_points=24 * 5,
                    recording_images=False,
                    z_classification=False, z_min=0., z_max=1., z_interval=0.5,
                    printing_sum=False,
                    recording_sum=False,
                    printing_warnings=False,
                    recording_g=False,
                    recording_g_properties=True,
                    random=False):
    # We convert the time step in seconds:
    time_step_in_seconds = time_step_in_days * 60. * 60. * 24.
    # We calculate the number of steps necessary to reach the end of the simulation period:
    if simulation_period_in_days == 0. or time_step_in_days == 0.:
        print("WATCH OUT: No simulation was done, as time input was 0.")
        n_steps = 0
    else:
        n_steps = trunc(simulation_period_in_days / time_step_in_days) + 1

    print("n_steps is ", n_steps)  # TODO: delete

    # We call global variables:
    global thermal_time_since_last_adventitious_root_emergence
    global adventitious_root_emergence

    # We initialize empty variables at t=0:
    step = 0
    time = 0.
    total_struct_mass = 0.
    cumulated_hexose_exudation = 0.
    cumulated_respired_CO2 = 0.
    cumulated_struct_mass_production = 0.
    sucrose_input_rate = 0.
    C_cumulated_in_the_degraded_pool = 0.
    C_cumulated_in_the_gaz_phase = 0.

    # We initialize empty lists for recording the macro-results:
    time_in_days_series = []
    sucrose_input_series = []
    total_living_root_length_series = []
    total_dead_root_length_series = []
    total_living_root_surface_series = []
    total_dead_root_surface_series = []
    total_living_root_struct_mass_series = []
    total_dead_root_struct_mass_series = []
    total_sucrose_root_series = []
    total_hexose_root_series = []
    total_hexose_reserve_series = []
    total_hexose_soil_series = []

    total_sucrose_root_deficit_series = []
    total_hexose_root_deficit_series = []
    total_hexose_soil_deficit_series = []

    total_respiration_series = []
    total_respiration_root_growth_series = []
    total_respiration_root_maintenance_series = []
    total_structural_mass_production_series = []
    total_hexose_production_from_phloem_series = []
    total_sucrose_loading_in_phloem_series = []
    total_hexose_mobilization_from_reserve_series = []
    total_hexose_immobilization_as_reserve_series = []
    total_hexose_exudation_series = []
    total_hexose_uptake_series = []
    total_hexose_degradation_series = []
    total_net_hexose_exudation_series = []
    C_in_the_root_soil_system_series = []
    C_cumulated_in_the_degraded_pool_series = []
    C_cumulated_in_the_gaz_phase_series = []
    global_sucrose_deficit_series = []
    tip_C_hexose_root_series = []

    # We create an empty dictionary that will contain the results of z classification:
    z_dictionary_series = {}

    if recording_images:
        # We define the directory "video"
        video_dir = 'video'
        # If this directory doesn't exist:
        if not os.path.exists(video_dir):
            # Then we create it:
            os.mkdir(video_dir)
        else:
            # Otherwise, we delete all the images that are already present inside:
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g:
        # We define the directory "MTG_files"
        g_dir = 'MTG_files'
        # If this directory doesn't exist:
        if not os.path.exists(g_dir):
            # Then we create it:
            os.mkdir(g_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(g_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    if recording_g_properties:
        # We define the directory "MTG_properties"
        prop_dir = 'MTG_properties'
        # If this directory doesn't exist:
        if not os.path.exists(prop_dir):
            # Then we create it:
            os.mkdir(prop_dir)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(prop_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

    # READING THE INPUT FILE:
    # -----------------------
    if input_file != "None" and (constant_sucrose_input_rate <= 0 or constant_soil_temperature_in_Celsius <= 0):
        # # We first define the path and the file to read as a .csv:
        # PATH = os.path.join('.', input_file)
        # # Then we read the file and copy it in a dataframe "df":
        # input_frame = pd.read_csv(PATH, sep=',')
        # We use the function 'formatted inputs' to create a table containing the input data (soil temperature and sucrose input)
        # for each required step, depending on the chosen time step:
        input_frame = formatted_inputs(original_input_file=input_file,
                                       original_time_step_in_days=1 / 24.,
                                       final_time_step_in_days=time_step_in_days,
                                       simulation_period_in_days=simulation_period_in_days,
                                       do_not_execute_if_file_with_suitable_size_exists=False)

    # RECORDING THE INITIAL STATE OF THE MTG:
    # ---------------------------------------
    step = 0

    # If the rotation of the camera around the root system is required:
    if camera_rotation:
        # We calculate the coordinates of the camera on the circle around the center:
        x_coordinates, y_coordinates, z_coordinates = circle_coordinates(z_center=z_cam, radius=camera_distance,
                                                                         n_points=n_rotation_points)
        # We initialize the index for reading each coordinates:
        index_camera = 0
        x_cam = x_coordinates[index_camera]
        y_cam = y_coordinates[index_camera]
        z_cam = z_coordinates[index_camera]
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
        sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                      x_center=x_center,
                      y_center=y_center,
                      z_center=z_center,
                      x_cam=x_camera,
                      y_cam=0,
                      z_cam=z_camera)
        # We move the camera further from the root system:
        x_camera = x_cam + x_cam * step_back_coefficient * step
        z_camera = z_cam + z_cam * step_back_coefficient * step
    # We finally display the MTG on PlantGL:
    print("OK")
    pgl.Viewer.display(sc)

    # For recording the graph at each time step to make a video later:
    # -----------------------------------------------------------------
    if recording_images:
        image_name = os.path.join(video_dir, 'root%.5d.png')
        pgl.Viewer.saveSnapshot(image_name % step)

    # For integrating root variables on the z axis:
    # ----------------------------------------------
    if z_classification:
        z_dictionary = classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
        z_dictionary["time_in_days"] = 0
        z_dictionary_series.update(z_dictionary)
        print(z_dictionary_series)

    # For recording the MTG at each time step to load it later on:
    # ------------------------------------------------------------
    if recording_g:
        g_file_name = os.path.join(g_dir, 'root%.5d.pckl')
        with open(g_file_name % step, 'wb') as output:
            pickle.dump(g, output, protocol=2)

    # For recording the properties of g in a csv file:
    # ------------------------------------------------
    if recording_g_properties:
        prop_file_name = os.path.join(prop_dir, 'root%.5d.csv')
        recording_MTG_properties(g, file_name=prop_file_name % step)

    # SUMMING AND PRINTING VARIABLES ON THE ROOT SYSTEM:
    # --------------------------------------------------

    # We reset to 0 all growth-associated C costs:
    reinitializing_growth_variables(g)

    if printing_sum:
        dictionary = summing(g,
                             printing_total_length=True,
                             printing_total_struct_mass=True,
                             printing_all=True)
    elif not printing_sum and recording_sum:
        dictionary = summing(g,
                             printing_total_length=True,
                             printing_total_struct_mass=True,
                             printing_all=False)
    if recording_sum:
        time_in_days_series.append(time_step_in_days * step)
        sucrose_input_series.append(sucrose_input_rate * time_step_in_seconds)
        total_living_root_length_series.append(dictionary["total_living_root_length"])
        total_dead_root_length_series.append(dictionary["total_dead_root_length"])
        total_living_root_struct_mass_series.append(dictionary["total_living_root_struct_mass"])
        total_dead_root_struct_mass_series.append(dictionary["total_dead_root_struct_mass"])
        total_living_root_surface_series.append(dictionary["total_living_root_surface"])
        total_dead_root_surface_series.append(dictionary["total_dead_root_surface"])
        total_sucrose_root_series.append(dictionary["total_sucrose_root"])
        total_hexose_root_series.append(dictionary["total_hexose_root"])
        total_hexose_reserve_series.append(dictionary["total_hexose_reserve"])
        total_hexose_soil_series.append(dictionary["total_hexose_soil"])

        total_sucrose_root_deficit_series.append(dictionary["total_sucrose_root_deficit"])
        total_hexose_root_deficit_series.append(dictionary["total_hexose_root_deficit"])
        total_hexose_soil_deficit_series.append(dictionary["total_hexose_soil_deficit"])

        total_respiration_series.append(dictionary["total_respiration"])
        total_respiration_root_growth_series.append(dictionary["total_respiration_root_growth"])
        total_respiration_root_maintenance_series.append(dictionary["total_respiration_root_maintenance"])
        total_structural_mass_production_series.append(dictionary["total_structural_mass_production"])
        total_hexose_production_from_phloem_series.append(dictionary["total_hexose_production_from_phloem"])
        total_sucrose_loading_in_phloem_series.append(dictionary["total_sucrose_loading_in_phloem"])
        total_hexose_mobilization_from_reserve_series.append(dictionary["total_hexose_mobilization_from_reserve"])
        total_hexose_immobilization_as_reserve_series.append(dictionary["total_hexose_immobilization_as_reserve"])
        total_hexose_exudation_series.append(dictionary["total_hexose_exudation"])
        total_hexose_uptake_series.append(dictionary["total_hexose_uptake"])
        total_hexose_degradation_series.append(dictionary["total_hexose_degradation"])
        total_net_hexose_exudation_series.append(dictionary["total_net_hexose_exudation"])

        C_in_the_root_soil_system_series.append(dictionary["C_in_the_root_soil_system"])
        C_cumulated_in_the_degraded_pool += dictionary["C_degraded_in_the_soil"]
        C_cumulated_in_the_degraded_pool_series.append(C_cumulated_in_the_degraded_pool)
        C_cumulated_in_the_gaz_phase += dictionary["C_respired_by_roots"]
        C_cumulated_in_the_gaz_phase_series.append(C_cumulated_in_the_gaz_phase)
        global_sucrose_deficit_series.append(global_sucrose_deficit)

        tip_C_hexose_root_series.append(g.node(0).C_hexose_root)

        # Initializing the amount of C in the root_soil_CO2 system:
        previous_C_in_the_system = dictionary["C_in_the_root_soil_system"] + C_cumulated_in_the_gaz_phase
        theoretical_cumulated_C_in_the_system = previous_C_in_the_system

    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # The code will try to run the following code until it is finished or an error has been raised:
    try:
        # An iteration is done for each time step:
        for step in range(1, n_steps):

            # At the beginning of the time step, we reset the global variable allowing the emergence of adventitious roots:
            adventitious_root_emergence = "Possible"
            # We keep in memory the value of the global variable time_since_adventitious_root_emergence at the beginning of the time steo:
            initial_time_since_adventitious_root_emergence = thermal_time_since_last_adventitious_root_emergence

            # We calculate the current time in hours:
            current_time_in_hours = step * time_step_in_days * 24.

            # DEFINING THE INPUT OF CARBON TO THE ROOTS FOR THIS TIME STEP:
            # --------------------------------------------------------------
            if constant_sucrose_input_rate > 0 or input_file == "None":
                sucrose_input_rate = constant_sucrose_input_rate
            else:
                sucrose_input_rate = input_frame.loc[step, 'sucrose_input_rate']

            # DEFINING THE TEMPERATURE OF THE SOIL FOR THIS TIME STEP:
            # --------------------------------------------------------
            if constant_soil_temperature_in_Celsius > 0 or input_file == "None":
                soil_temperature = constant_soil_temperature_in_Celsius
            else:
                soil_temperature = input_frame.loc[step, 'soil_temperature_in_degree_Celsius']

            # CALCULATING AN EQUIVALENT OF THERMAL TIME:
            # -------------------------------------------

            # We calculate a coefficient that will modify the different "ages" experienced by roots according to soil temperature:
            temperature_time_adjustment = temperature_modification(temperature_in_Celsius=soil_temperature,
                                                                   process_at_T_ref=1,
                                                                   T_ref=T_ref_growth,
                                                                   A=growth_increase_with_temperature,
                                                                   B=1,
                                                                   C=0)

            # STARTING THE ACTUAL SIMULATION:
            # --------------------------------
            print("")
            print("From t =", "{:.2f}".format(Decimal((step - 1) * time_step_in_days)), "days to t =",
                  "{:.2f}".format(Decimal(step * time_step_in_days)), "days:")
            print("------------------------------------")
            print("   Soil temperature is", soil_temperature, "degree Celsius.")
            print("   The input rate of sucrose to the root for time=", current_time_in_hours, "h is",
                  "{:.2E}".format(Decimal(sucrose_input_rate)), "mol of sucrose per second, i.e.",
                  "{:.2E}".format(Decimal(sucrose_input_rate * 60. * 60. * 24.)), "mol of sucrose per day.")

            print("   The root system initially includes", len(g) - 1, "root elements.")

            # CASE 1: WE REPRODUCE THE GROWTH WITHOUT CONSIDERATIONS OF LOCAL CONCENTRATIONS
            # -------------------------------------------------------------------------------

            if ArchiSimple:

                # The input of C (gram of C) from shoots is calculated from the input of sucrose:
                C_input = sucrose_input_rate * 12 * 12.01 * time_step_in_seconds
                # We assume that only a fraction of this C_input will be used for producing struct_mass:
                fraction = 0.20
                struct_mass_input = C_input / struct_mass_C_content * fraction

                # We calculate the potential growth, already based on ArchiSimple rules:
                potential_growth(g, time_step_in_seconds=time_step_in_seconds,
                                 radial_growth=radial_growth,
                                 ArchiSimple=True,
                                 soil_temperature_in_Celsius=soil_temperature)

                # We use the function ArchiSimple_growth to adapt the potential growth to the available struct_mass:
                SC = satisfaction_coefficient(g, struct_mass_input=struct_mass_input)
                ArchiSimple_growth(g, SC, time_step_in_seconds,
                                   soil_temperature_in_Celsius=soil_temperature,
                                   printing_warnings=printing_warnings)

                # We proceed to the segmentation of the whole root system (NOTE: segmentation should always occur AFTER actual growth):
                segmentation_and_primordia_formation(g, time_step_in_seconds, printing_warnings=printing_warnings,
                                                     soil_temperature_in_Celsius=soil_temperature, random=random,
                                                     nodules=nodules)

            else:

                # CASE 2: WE PERFORM THE COMPLETE MODEL WITH C BALANCE IN EACH ROOT ELEMENT
                # --------------------------------------------------------------------------

                # We reset to 0 all growth-associated C costs:
                reinitializing_growth_variables(g)

                # Calculation of potential growth without consideration of available hexose:
                potential_growth(g, time_step_in_seconds=time_step_in_seconds,
                                 radial_growth=radial_growth,
                                 soil_temperature_in_Celsius=soil_temperature,
                                 ArchiSimple=False)

                # Calculation of actual growth based on the hexose remaining in the roots,
                # and corresponding consumption of hexose in the root:
                actual_growth_and_corresponding_respiration(g, time_step_in_seconds=time_step_in_seconds,
                                                            soil_temperature_in_Celsius=soil_temperature,
                                                            printing_warnings=printing_warnings)
                # # We proceed to the segmentation of the whole root system (NOTE: segmentation should always occur AFTER actual growth):
                segmentation_and_primordia_formation(g, time_step_in_seconds,
                                                     soil_temperature_in_Celsius=soil_temperature,
                                                     random=random,
                                                     nodules=nodules)
                dist_to_tip(g)

                # Consumption of hexose in the soil:
                soil_hexose_degradation(g, time_step_in_seconds=time_step_in_seconds,
                                        soil_temperature_in_Celsius=soil_temperature,
                                        printing_warnings=printing_warnings)

                # Transfer of hexose from the root to the soil, consumption of hexose inside the roots:
                root_hexose_exudation(g, time_step_in_seconds=time_step_in_seconds,
                                      soil_temperature_in_Celsius=soil_temperature,
                                      printing_warnings=printing_warnings)
                # Transfer of hexose from the soil to the root, consumption of hexose in the soil:
                root_hexose_uptake(g, time_step_in_seconds=time_step_in_seconds,
                                   soil_temperature_in_Celsius=soil_temperature,
                                   printing_warnings=printing_warnings)

                # Consumption of hexose in the root by maintenance respiration:
                maintenance_respiration(g, time_step_in_seconds=time_step_in_seconds,
                                        soil_temperature_in_Celsius=soil_temperature,
                                        printing_warnings=printing_warnings)

                # Unloading of sucrose from phloem and conversion of sucrose into hexose:
                exchange_with_phloem(g, time_step_in_seconds=time_step_in_seconds,
                                     soil_temperature_in_Celsius=soil_temperature,
                                     printing_warnings=printing_warnings)

                # Net immobilization of hexose within a reserve pool:
                exchange_with_reserve(g, time_step_in_seconds=time_step_in_seconds,
                                      soil_temperature_in_Celsius=soil_temperature,
                                      printing_warnings=printing_warnings)

                # Calculation of the new concentrations in hexose and sucrose once all the processes have been done:
                tip_C_hexose_root = balance(g, printing_warnings=printing_warnings)

                # Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
                shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=sucrose_input_rate,
                                                   time_step_in_seconds=time_step_in_seconds,
                                                   printing_warnings=printing_warnings)
                # WARNING: The function "shoot_sucrose_supply_and_spreading" must be called AFTER the function "balance",!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # otherwise the deficit in sucrose may be counted twice!!!

                # # OPTIONAL: checking of possible anomalies in the root system:
                control_of_anomalies(g)

            # A the end of the time step, if the global variable "time_since_adventitious_root_emergence" has been unchanged:
            if thermal_time_since_last_adventitious_root_emergence == initial_time_since_adventitious_root_emergence:
                # Then we increment it by the time step:
                thermal_time_since_last_adventitious_root_emergence += time_step_in_seconds * temperature_time_adjustment
            # Otherwise, the variable has already been reset when the emergence of one adventitious root has been allowed.

            # PLOTTING THE MTG:
            # ------------------

            # If the rotation of the camera around the root system is required:
            if camera_rotation:
                x_cam = x_coordinates[index_camera]
                y_cam = y_coordinates[index_camera]
                z_cam = z_coordinates[index_camera]
                sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                              x_center=x_center,
                              y_center=y_center,
                              z_center=z_center,
                              x_cam=x_cam,
                              y_cam=y_cam,
                              z_cam=z_cam)
                # We define the index of the coordinates to read at the next step:
                index_camera = index_camera + 1
                # If this index is higher than the number of coordinates in each vector:
                if index_camera >= n_rotation_points:
                    # Then we reset the index to 0:
                    index_camera = 0
            # Otherwise, the camera will stay on a fixed position:
            else:

                sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax, cmap=cmap,
                              x_center=x_center,
                              y_center=y_center,
                              z_center=z_center,
                              x_cam=x_camera,
                              y_cam=0,
                              z_cam=z_camera)
                # We move the camera further from the root system:
                x_camera = x_cam + x_cam * step_back_coefficient * step
                z_camera = z_cam + z_cam * step_back_coefficient * step
            # We finally display the MTG on PlantGL:
            pgl.Viewer.display(sc)

            # For recording the graph at each time step to make a video later:
            # -----------------------------------------------------------------
            if recording_images:
                image_name = os.path.join(video_dir, 'root%.5d.png')
                pgl.Viewer.saveSnapshot(image_name % step)


            # For integrating root variables on the z axis:
            # ----------------------------------------------
            if z_classification:
                z_dictionary = classifying_on_z(g, z_min=z_min, z_max=z_max, z_interval=z_interval)
                z_dictionary["time_in_days"] = time_step_in_days * step
                z_dictionary_series.update(z_dictionary)

            # For recording the MTG at each time step to load it later on:
            # ------------------------------------------------------------
            if recording_g:
                g_file_name = os.path.join(g_dir, 'root%.5d.pckl')
                with open(g_file_name % step, 'wb') as output:
                    pickle.dump(g, output, protocol=2)

            # For recording the properties of g in a csv file:
            # --------------------------------------------------
            if recording_g_properties:
                prop_file_name = os.path.join(prop_dir, 'root%.5d.csv')
                recording_MTG_properties(g, file_name=prop_file_name % step)

            # SUMMING AND PRINTING VARIABLES ON THE ROOT SYSTEM:
            # --------------------------------------------------
            if printing_sum:
                dictionary = summing(g,
                                     printing_total_length=True,
                                     printing_total_struct_mass=True,
                                     printing_all=True)
            elif not printing_sum and recording_sum:
                dictionary = summing(g,
                                     printing_total_length=True,
                                     printing_total_struct_mass=True,
                                     printing_all=False)
            if recording_sum:
                time_in_days_series.append(time_step_in_days * step)
                sucrose_input_series.append(sucrose_input_rate * time_step_in_seconds)
                total_living_root_length_series.append(dictionary["total_living_root_length"])
                total_dead_root_length_series.append(dictionary["total_dead_root_length"])
                total_living_root_struct_mass_series.append(dictionary["total_living_root_struct_mass"])
                total_dead_root_struct_mass_series.append(dictionary["total_dead_root_struct_mass"])
                total_living_root_surface_series.append(dictionary["total_living_root_surface"])
                total_dead_root_surface_series.append(dictionary["total_dead_root_surface"])
                total_sucrose_root_series.append(dictionary["total_sucrose_root"])
                total_hexose_root_series.append(dictionary["total_hexose_root"])
                total_hexose_reserve_series.append(dictionary["total_hexose_reserve"])
                total_hexose_soil_series.append(dictionary["total_hexose_soil"])

                total_sucrose_root_deficit_series.append(dictionary["total_sucrose_root_deficit"])
                total_hexose_root_deficit_series.append(dictionary["total_hexose_root_deficit"])
                total_hexose_soil_deficit_series.append(dictionary["total_hexose_soil_deficit"])

                total_respiration_series.append(dictionary["total_respiration"])
                total_respiration_root_growth_series.append(dictionary["total_respiration_root_growth"])
                total_respiration_root_maintenance_series.append(dictionary["total_respiration_root_maintenance"])
                total_structural_mass_production_series.append(dictionary["total_structural_mass_production"])
                total_hexose_production_from_phloem_series.append(dictionary["total_hexose_production_from_phloem"])
                total_sucrose_loading_in_phloem_series.append(dictionary["total_sucrose_loading_in_phloem"])
                total_hexose_mobilization_from_reserve_series.append(
                    dictionary["total_hexose_mobilization_from_reserve"])
                total_hexose_immobilization_as_reserve_series.append(
                    dictionary["total_hexose_immobilization_as_reserve"])
                total_hexose_exudation_series.append(dictionary["total_hexose_exudation"])
                total_hexose_uptake_series.append(dictionary["total_hexose_uptake"])
                total_hexose_degradation_series.append(dictionary["total_hexose_degradation"])
                total_net_hexose_exudation_series.append(dictionary["total_net_hexose_exudation"])

                C_in_the_root_soil_system_series.append(dictionary["C_in_the_root_soil_system"])
                C_cumulated_in_the_degraded_pool += dictionary["C_degraded_in_the_soil"]
                C_cumulated_in_the_degraded_pool_series.append(C_cumulated_in_the_degraded_pool)
                C_cumulated_in_the_gaz_phase += dictionary["C_respired_by_roots"]
                C_cumulated_in_the_gaz_phase_series.append(C_cumulated_in_the_gaz_phase)
                global_sucrose_deficit_series.append(global_sucrose_deficit)

                tip_C_hexose_root_series.append(tip_C_hexose_root)

                # CHECKING CARBON BALANCE:
                current_C_in_the_system = dictionary[
                                              "C_in_the_root_soil_system"] + C_cumulated_in_the_gaz_phase + C_cumulated_in_the_degraded_pool
                theoretical_current_C_in_the_system = (
                        previous_C_in_the_system + sucrose_input_rate * time_step_in_seconds * 12.)
                theoretical_cumulated_C_in_the_system += sucrose_input_rate * time_step_in_seconds * 12.

                if abs(
                        current_C_in_the_system - theoretical_cumulated_C_in_the_system) / current_C_in_the_system > 1e-10:
                    print("!!! ERROR ON CARBON BALANCE: the current amount of C in the system is",
                          "{:.2E}".format(Decimal(current_C_in_the_system)), "but it should be",
                          "{:.2E}".format(Decimal(theoretical_current_C_in_the_system)), "mol of C")
                    print("This corresponds to a net disappearance of C of",
                          "{:.2E}".format(Decimal(theoretical_current_C_in_the_system - current_C_in_the_system)),
                          "mol of C, and the cumulated difference since the start of the simulation and the current one is",
                          "{:.2E}".format(
                              Decimal(theoretical_cumulated_C_in_the_system - current_C_in_the_system)), "mol of C.")

                    # We reinitialize the "previous" amount of C in the system with the current one for the next time step:
                previous_C_in_the_system = current_C_in_the_system

            print("      The root system finally includes", len(g) - 1, "root elements.")


    # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # At the end of the simulation (or just before an error is about to interrupt the program!):
    # -------------------------------------------------------------------------------------------
    finally:
        print("")
        print("The program has stopped at time t = {:.2f}".format(Decimal(step * time_step_in_days)), "days.")
        # We can record all the results in a CSV file:
        if recording_sum:
            # We create a data_frame from the vectors generated in the main program up to this point:
            data_frame = pd.DataFrame({"Time (days)": time_in_days_series,
                                       "Sucrose input (mol of sucrose)": sucrose_input_series,
                                       "Root structural mass (g)": total_living_root_struct_mass_series,
                                       "Root necromass (g)": total_dead_root_struct_mass_series,
                                       "Root length (m)": total_living_root_length_series,
                                       "Root surface (m2)": total_living_root_surface_series,
                                       "Sucrose in the root (mol of sucrose)": total_sucrose_root_series,
                                       "Hexose in the mobile pool of the roots (mol of hexose)": total_hexose_root_series,
                                       "Hexose in the reserve pool of the roots (mol of hexose)": total_hexose_reserve_series,
                                       "Hexose in the soil (mol of hexose)": total_hexose_soil_series,

                                       "Deficit of sucrose in the root (mol of sucrose)": total_sucrose_root_deficit_series,
                                       "Deficit of hexose in the mobile pool of the roots (mol of hexose)": total_hexose_root_deficit_series,
                                       "Deficit of hexose in the soil (mol of hexose)": total_hexose_soil_deficit_series,

                                       "CO2 originating from root growth (mol of C)": total_respiration_root_growth_series,
                                       "CO2 originating from root maintenance (mol of C)": total_respiration_root_maintenance_series,
                                       "Structural mass produced (g)": total_structural_mass_production_series,
                                       "Hexose unloaded from phloem (mol of hexose)": total_hexose_production_from_phloem_series,
                                       "Sucrose reloaded in the phloem (mol of hexose)": total_sucrose_loading_in_phloem_series,
                                       "Hexose mobilized from reserve (mol of hexose)": total_hexose_mobilization_from_reserve_series,
                                       "Hexose stored as reserve (mol of hexose)": total_hexose_immobilization_as_reserve_series,
                                       "Hexose emitted in the soil (mol of hexose)": total_hexose_exudation_series,
                                       "Hexose taken up from the soil (mol of hexose)": total_hexose_uptake_series,
                                       "Hexose degraded in the soil (mol of hexose)": total_hexose_degradation_series,

                                       "Cumulated amount of C present in the root-soil system (mol of C)": C_in_the_root_soil_system_series,
                                       "Cumulated amount of C that has been degraded in the soil (mol of C)": C_cumulated_in_the_degraded_pool_series,
                                       "Cumulated amount of C that has been respired by roots (mol of C)": C_cumulated_in_the_gaz_phase_series,
                                       "Final deficit in sucrose of the whole root system (mol of sucrose)": global_sucrose_deficit_series,

                                       "Concentration of hexose in the main root tip (mol of hexose per g)": tip_C_hexose_root_series
                                       },
                                      # We re-order the columns:
                                      columns=["Time (days)",
                                               "Sucrose input (mol of sucrose)",
                                               "Final deficit in sucrose of the whole root system (mol of sucrose)",
                                               "Cumulated amount of C present in the root-soil system (mol of C)",
                                               "Cumulated amount of C that has been respired by roots (mol of C)",
                                               "Cumulated amount of C that has been degraded in the soil (mol of C)",
                                               "Root structural mass (g)",
                                               "Root necromass (g)",
                                               "Root length (m)",
                                               "Root surface (m2)",
                                               "Sucrose in the root (mol of sucrose)",
                                               "Hexose in the mobile pool of the roots (mol of hexose)",
                                               "Hexose in the reserve pool of the roots (mol of hexose)",
                                               "Hexose in the soil (mol of hexose)",
                                               "Deficit of sucrose in the root (mol of sucrose)",
                                               "Deficit of hexose in the mobile pool of the roots (mol of hexose)",
                                               "Deficit of hexose in the soil (mol of hexose)",
                                               "CO2 originating from root growth (mol of C)",
                                               "CO2 originating from root maintenance (mol of C)",
                                               "Structural mass produced (g)",
                                               "Hexose unloaded from phloem (mol of hexose)",
                                               "Sucrose reloaded in the phloem (mol of hexose)",
                                               "Hexose mobilized from reserve (mol of hexose)",
                                               "Hexose stored as reserve (mol of hexose)",
                                               "Hexose emitted in the soil (mol of hexose)",
                                               "Hexose taken up from the soil (mol of hexose)",
                                               "Hexose degraded in the soil (mol of hexose)",
                                               "Concentration of hexose in the main root tip (mol of hexose per g)"
                                               ])
            # We save the data_frame in a CSV file:
            try:
                # In case the results file is not opened, we simply re-write it:
                data_frame.to_csv('simulation_results.csv', na_rep='NA', index=False, header=True)
                print("The main results have been written in the file 'simulation_results.csv'.")
            except:
                # Otherwise we write the data in a new result file as back-up option:
                data_frame.to_csv('simulation_results_BACKUP.csv', na_rep='NA', index=False, header=True)
                print("")
                print("WATCH OUT: The main results have been written in the alternative file 'simulation_results_BACKUP.csv'.")

        # We create another data frame that contains the results classified by z intervals:
        if z_classification:
            # We create a data_frame from the vectors generated in the main program up to this point:
            data_frame_z = pd.DataFrame.from_dict(z_dictionary_series)
            # We save the data_frame in a CSV file:
            data_frame_z.to_csv('z_classification.csv', na_rep='NA', index=False, header=True)


# RUNNING THE SIMULATION:
#########################

if __name__ == "__main__":

    # We set the working directory:
    my_path = r'C:\\Users\\Marion\\Documents\\Marion\\rhizodep\\test'
    if not os.path.exists(my_path):
        my_path = os.path.abspath('.')
    os.chdir(my_path)
    print("The current directory is:", os.getcwd())

    # We record the time when the run starts:
    start_time = timeit.default_timer()

    # We initiate the properties of the MTG "g":
    g = initiate_mtg(random=True)
    # We initiate the time variable that will be used to determine the emergence of adventitious roots:
    thermal_time_since_last_adventitious_root_emergence = 0.
    # We initiate the global variable that corresponds to a possible general deficit in sucrose of the whole root system:
    global_sucrose_deficit = 0.

    # We launch the main simulation program:
    print("Simulation starts ...")
    main_simulation(g, simulation_period_in_days=1., time_step_in_days=1. / 24., radial_growth="Possible",
                    ArchiSimple=False,
                    # property="net_hexose_exudation_rate_per_day_per_cm", vmin=1e-9, vmax=1e-6, log_scale=True, cmap='jet',
                    property="C_hexose_root", vmin=1e-4, vmax=1e-1, log_scale=True, cmap='jet',
                    # property="C_sucrose_root", vmin=1e-4, vmax=1e2, log_scale=True, cmap='brg',
                    # property="C_hexose_reserve", vmin=1e-4, vmax=1e4, log_scale=True, cmap='brg',
                    input_file="sucrose_input_0047.csv",
                    constant_sucrose_input_rate=0,
                    constant_soil_temperature_in_Celsius=0,
                    nodules=False,
                    x_center=0, y_center=0, z_center=-1, z_cam=-2,
                    camera_distance=4, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12 * 10,
                    z_classification=False, z_min=0.00, z_max=1., z_interval=0.05,
                    recording_images=True,
                    printing_sum=False,
                    recording_sum=True,
                    printing_warnings=False,
                    recording_g=True,
                    recording_g_properties=False,
                    random=True)

    print("")
    print("***************************************************************")
    end_time = timeit.default_timer()
    print("Run is done! The system took", round(end_time - start_time, 1), "seconds to complete the run.")

    # We save the final MTG:
    with open('g_file.pckl', 'wb') as output:
        pickle.dump(g, output, protocol=2)

    print("The whole root system has been saved in the file 'g_file.pckl'.")

    # To avoid closing PlantGL as soon as the run is done:
    pgl.Viewer.exit()
    # input()
