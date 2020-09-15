# Todo: GET A CORRECT CARBON BALANCE!
# Todo: Impose a delay describing how long the parent segment must sustain the growth of the child - OR define a minimal volume of the new emerged apex?
# Todo: Consider giving priority to root maintenance, and provoque senescence when root maintenance is not ensured
# Todo: Watch the calculation of surface and volume for the apices - if they correspond to cones, the mass balance for segmentation may not be correct!
# Todo: Include cell sloughing and mucilage production

# Importation of functions from the system:
###########################################

from math import sqrt, pi, trunc, floor, cos, sin
from decimal import Decimal
import time
import numpy as np
import pandas as pd
import os, os.path
import timeit

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl

import pickle

# Setting the randomness in the whole code to reproduce the same root system over different runs:
# random_choice = int(round(np.random.normal(100,50)))
random_choice = 122
print "The random seed used for this run is", random_choice
np.random.seed(random_choice)

########################################################################################################################
########################################################################################################################
# LIST OF PARAMETERS
########################################################################################################################
########################################################################################################################

# Parameters for root growth:
# ----------------------------
# Maximal number of adventitious roots (including primary)(dimensionless):
MNP = 5
# Emission rate of adventious roots (in s-1):
# ER = 0.5 day-1
ER = 1. / (60. * 60. * 24.)
# Tip diameter of the emitted root(in s) ():
# Di=0.5 mm
Di = 1. / 1000.
# Slope of the potential elongation rate versus tip diameter (in m m-1 s-1):
# EL = 5 mm mm-1 day-1 = 5 m m-1 day-1
EL = 3. / (60. * 60. * 24.)
# Threshold tip diameter below which there is no possible elongation (diameter of the finest roots)(in m):
# Dmin=0.05 mm
Dmin = 0.15 / 1000.
# Coefficient of growth duration (in s m-2):
# GDs=400. day mm-2
GDs = 120. * (60. * 60. * 24.) * 1000. ** 2.
# Gravitropism (dimensionless):
G = 1.
# Delay of emergence of the primordium (in s):
# emergence_delay = 3. days
emergence_delay = 5.0 * (60. * 60. * 24.)
# Inter-primordium distance (in m):
# IPD = 7.6 mm
IPD = 8 /1000.
# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
RMD = 0.3
# Relative variation of the daughter root diameter (dimensionless):
CVDD = 0.20
# Proportionality coefficient between section area of the segment and the sum of distal section areas (dimensionless):
SGC = 0.6
# Root tissue density (in g m-3):
# RTD=0.1 g cm-3
RTD = 0.10 * 1e6
density = RTD
# Coefficient of the life duration (in s m g-1 m3):
# LDs = 5000 day mm-1 g-1 cm3
LDs = 5000. * (60. * 60. * 24.) * 1000 * 1e-6
# Length of a segment (in m):
segment_length = 3. / 1000.
# C content of biomass (mol of C per g of biomass):
biomass_C_content = 0.44 / 12.01  # We assume that the biomass contains 44% of C.

# Parameters for growth respiration:
# -----------------------------------
# Growth yield (in mol of CO2 per mol of C used for biomass):
yield_growth = 0.8
# => Explanation: We use the value proposed by Thornley and Cannell (2000)

# Parameters for maintenance respiration:
# ----------------------------------------
# Maximal maintenance respiration (in mol of CO2 per g of biomass per s):
resp_maintenance_max = 4.1e-6 / 20 * biomass_C_content
# => Explanation: According to Barillot et al. (2016): km_max = 4.1e-6 umol_C umol_N-1 s-1,
# i.e. 4.1e-6/20*0.44 mol_C g-1 s-1 assuming a C:N molar ratio of root biomass of 20 and a C content of biomass of 44%
# Affinity constant for maintenance respiration (in mol of hexose per g of biomass):
Km_maintenance = 1.67e-3 / 6.
# => Explanation: We use the value of 1.67e-3 mol_C per g proposed by Barillot et al. (2016)

# Parameters for root hexose exudation:
# --------------------------------------
# Expected exudation (in mol of hexose per m2 per s):
expected_exudation_efflux = 5.2 / 12.01 / 6. * 1e-6 * 100. ** 2. / 3600.
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour
# Expected sucrose concentration in root (in mol of sucrose per g of root):
expected_C_sucrose_root = 0.0025
# => Explanation: This is a plausible value according to the results of Gauthier (2019, pers. communication)
# Expected hexose concentration in root (in mol of hexose per g of root):
expected_C_hexose_root = 20. / 12.01 / 6. * 1e-3
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour
# Expected hexose concentration in soil (in mol of hexose per g of root):
expected_C_hexose_soil = expected_C_hexose_root / 100.
# => Explanation: We expect the soil concentration to be 2 orders of magnitude lower than the root concentration
# Permeability coefficient (in g of biomass per m2 per s):
Pmax_apex = expected_exudation_efflux / (expected_C_hexose_root - expected_C_hexose_soil) / 100.
# => Explanation: We calculate the permeability coefficient according to the expected flux and hexose concentrations.
# Coefficient affecting the decrease of permeability with distance from the apex (adimensional):
gamma_exudation = 0.4

# Parameters for root hexose uptake from soil:
# ---------------------------------------------
# Maximum influx of hexose from soil to roots (in mol of hexose per m2 per s):
Imax = expected_exudation_efflux * 0.1
# Affinity constant for hexose uptake (in mol of hexose per g of biomass):
Km_influx = expected_C_hexose_root * 10.
# => Explanation: We assume that half of the max influx is reached when the soil concentration equals 10 times the expected root concentration

# Parameters for soil hexose degradation:
# ----------------------------------------
# Maximum degradation rate of hexose in soil (in mol of hexose per m2 per s):
degradation_max = Imax / 2.
# => Explanation: We assume that the maximum degradation rate is 2 times lower than the maximum uptake rate
# Affinity constant for soil hexose degradation (in mol of hexose per g of biomass):
Km_degradation = Km_influx / 50.
# => Explanation: We assume that half of the maximum degradation rate can be reached 50 times sooner than half of the maximum uptake rate

# Parameters for sucrose unloading:
# ----------------------------------
# Maximum unloading rate of sucrose from the phloem (in mol of sucrose per m2 per s):
unloading_apex = 5e-6
# => Explanation: According to Barillot et al. (2016b), this value is 0.03 umol C g-1 s-1
# Coefficient affecting the decrease of unloading rate with distance from the apex (adimensional):
gamma_unloading = 5.
# Affinity constant for sucrose unloading (in mol of sucrose per g of biomass):
Km_unloading = expected_C_sucrose_root
# => Explanation: According to Barillot et al. (2016b), this value is 1000 umol C g-1

########################################################################################################################
########################################################################################################################
# COMMON FUNCTIONS POSSIBLY USED IN EACH MODULE
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
        # For displaying the radius x times larger than in reality:
        radius = n.radius * 10
        length = n.length * 10
        angle_down = n.angle_down
        angle_roll = n.angle_roll

        # Moving the turtle:
        turtle.down(angle_down)
        turtle.rollL(angle_roll)
        turtle.setId(v)
        turtle.setWidth(radius)
        turtle.F(length)

    return root_visitor

def my_colormap(g, property_name, cmap='jet', vmin=None, vmax=None, lognorm=True):
    """
    This function computes a property 'color' on a MTG based on a given MTG's property.
    :param g: the investigated MTG
    :param property_name: the name of the property of the MTG that will be displayed
    :param cmap: the type of color map
    :param vmin: the min value to be displayed
    :param vmax: the max value to be displayed
    :param lognorm: a Boolean decribing whether the scale is logarythmic or not
    :return: the MTG with the corresponding color
    """

    prop = g.property(property_name)
    keys = prop.keys()
    values = np.array(prop.values())
    # m, M = int(values.min()), int(values.max())

    _cmap = color.get_cmap(cmap)
    norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
    values = norm(values)

    colors = (_cmap(values)[:, 0:3]) * 255
    colors = np.array(colors, dtype=np.int).tolist()

    g.properties()['color'] = dict(zip(keys, colors))
    return g

def prepareScene(scene, width=1600, height=1200, scale=0.8, x_center=0, y_center=0, z_center=0,
                 x_cam=0, y_cam=0, z_cam=-1.5, grid=False):
    """
    This function returns the scene that will be used in PlantGL to display the MTG.
    :param scene: the scene to start with
    :param width: the width of the graph (in pixels)
    :param height: the height of the graph (in pixels)
    :param scale: an adimentionnal factor for zooming in or out
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
    cam_target=pgl.Vector3(x_center*scale,
                           y_center*scale,
                           z_center*scale)
    # We define the coordinates of the point cam_pos that represents the position of the camera:
    cam_pos = pgl.Vector3(x_cam*scale,
                          y_cam*scale,
                          z_cam*scale)
    # We position the camera in the scene relatively to the center of the scene:
    pgl.Viewer.camera.lookAt(cam_pos, cam_target)
    # We define the dimensions of the graph:
    pgl.Viewer.frameGL.setSize(width, height)
    # We define whether grids are displayed or not:
    pgl.Viewer.grids.set(grid,grid,grid,grid)

    return scene

def circle_coordinates(x_center=0, y_center=0, z_center=0, radius=1, n_points=50):
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
    x_coordinates=[]
    y_coordinates=[]
    z_coordinates=[]

    # We initalize the angle at 0 rad:
    angle = 0
    # We calculate the increment theta that will be added to the angle for each new point:
    theta = 2*pi / float(n_points)

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
             x_center=0, y_center=0, z_center=0,
             x_cam=1, y_cam=0, z_cam=0):
    """
    This function creates a graph on PlantGL that displays a MTG and color it according to a specified property.
    :param g: the investigated MTG
    :param property_name: the name of the property of the MTG that will be displayed in color
    :param cmap: the type of color map
    :param lognorm: a Boolean decribing whether the scale is logarythmic or not
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
    # We intialize a turtle in PlantGL:
    turtle = turt.PglTurtle()
    # We make the graph upside down:
    turtle.down(180)
    # We initalize the scene with the MTG g:
    scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)
    # We update the scene with the specified position of the center of the graph and the camera:
    prepareScene(scene, x_center=x_center, y_center=y_center, z_center=z_center, x_cam=x_cam, y_cam=y_cam, z_cam=z_cam)
    # We compute the colors of the graph:
    my_colormap(g, prop_cmap, cmap=cmap, vmin=vmin, vmax=vmax, lognorm=lognorm)
    # We get a list of all shapes in the scene:
    shapes = dict((sh.getId(), sh) for sh in scene)
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
    # We return the new updated scene:
    scene = pgl.Scene(shapes.values())
    return scene

# FUNCTIONS FOR CALCULATING PROPERTIES ON THE MTG
#################################################

# Defining the surface of each root element in contact with the soil:
# -------------------------------------------------------------------
def surface_and_volume(element, radius, length):
   """
   The function "surface_and_volume" computes the "external_surface" (m2) and "volume" (m3) of a root element,
   based on the properties radius (m) and length (m). If the root element is an apex, the external surface is defined
   as the surface of a cone of height = "length" and radius = "radius", and the volume is the volume of this cone.
   If the root element is a root segment, the external surface is defined as the lateral surface of a cylinder
   of height = "length" and radius = "radius", and the volume corresponds to the volume of this cylinder.
   For each branching on the root segment, the section of the daughter root is subtracted from the cylinder surface
   of the root segment of the mother root.
   :param element: the investigated node of the MTG
   :param radius: the radius of the root element (m)
   :param length: the length of the root element (m)
   :return: external_surface (m2), volume (m3)
   """

   n = element
   vid = n.index()
   number_of_children = n.nb_children()

   # WATCH OUT! THE VOLUME OF AN APEX IS THE ONE OF A CYLINDER FOR NOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   # If there is no children to the root element:
   if number_of_children == 0:
       # Then the root element corresponds to an apex considered as a cone
       # of height = "length" and radius = "radius":
       # external_surface = pi * radius*sqrt(radius**2 + length**2)
       # volume = pi * radius**2 * length / 3
       external_surface = 2 * pi * radius * length
       volume = pi * radius ** 2 * length

   # If there is only one child to the root element:
   elif number_of_children == 1:
       # Then the root element is a root segment without branching, considered as a cylinder
       # of height = "length" and radius = "radius":
       external_surface = 2 * pi * radius * length
       volume = pi * radius ** 2 * length
   # Otherwise there is one or more lateral roots branched on the root segment:
   else:
       # Then we sum all the sections of the lateral roots branched on the root segment:
       sum_ramif_sections = 0
       for child_vid in g.Sons(vid, EdgeType='+'):
           son = g.node(child_vid)
           sum_ramif_sections += pi * son.radius ** 2
       # And we subtract this sum of sections from the external area of the main cylinder:
       external_surface = 2 * pi * radius * length - sum_ramif_sections
       volume = pi * radius ** 2 * length

   return external_surface, volume


# Defining the distance of a vertex from the tip:
# -----------------------------------------------
def dist_to_tip(g):
    """
    The function "dist_to_tip" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
    :param g: the investigated MTG
    :return: the MTG with an updated property 'dist_to_tip'
    """

    # We initialize an empty dictionnary for to_tips:
    to_tips = {}
    # We use the property "length" of each vertex based on the function "length":
    length = g.property('length')

    # We define "root" as the starting point of the loop below (i.e. the first apex in the MTG)(?):
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # If the vertex corresponds to a root apex:
        vertex = g.node(vid)
        if vertex.label == "Apex":
            # Then the distance is 0 meter by definition:
            to_tips[vid] = 0.
        else:
            # Else we get the vertex ID of the successor of the root segment in the same root axis:
            son_id = g.Successor(vid)
            # And we calculate the new distance from the tip by adding the distance of the successor and its length:
            to_tips[vid] = to_tips[son_id] + length[son_id]

    # We assign the result "dist_to_tip" as a new property of each vertex in the MTG "g":
    g.properties()['dist_to_tip'] = to_tips
    # We return a modified version of the MTG "g" with a new property "dist_to_tip":

    # for vid in g.vertices_iter(scale=1):
    #     # n represents the vertex:
    #     n = g.node(vid)
    #     n.dist_to_tip=0.

    return g


# Calculation of the total amount of sucrose and structural biomass in the root system:
# ------------------------------------------------------------------------------------
def total_root_sucrose_and_biomass(g):
    """
    This function computes the total amount of sucrose of the root system (in mol of sucrose),
    and the total dry biomass of the root system (in g of dry structural biomass).
    :param g: the investigated MTG
    :return: total_sucrose_root(mol of sucrose), total_biomass (g of dry structural biomass)
    """

    # We initialize the values to 0:
    total_sucrose_root = 0.
    total_biomass = 0.

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # We only select the elements that have a positive biomass and are not dead:
        if n.type != "Dead" and n.type != "Just dead" and n.biomass >0.:
            # We calculate the total living biomass by summing all the local biomasses:
            total_biomass += n.biomass
            # We calculate the total amount of sucrose by summing all the local products of concentrations with biomass:
            total_sucrose_root += n.C_sucrose_root * n.biomass

    # We return a list of two numeric values:
    return total_sucrose_root, total_biomass


# Calculation of total amounts and dimensions of the root system:
# ---------------------------------------------------------------
def summing(g, printing_total_length=True, printing_total_biomass=True, printing_all=False):
    """
    This function computes a number of general properties summed over the whole MTG.
    :param g: the investigated MTG
    :param printing_total_length: a Boolean defining whether total_length should be printed on the screen or not
    :param printing_total_biomass: a Boolean defining whether total_biomass should be printed on the screen or not
    :param printing_all: a Boolean defining whether all properties should be printed on the screen or not
    :return: a dictionnary containing the numerical value of each property integrated over the whole MTG
    """

    # We initialize the values to 0:
    total_length = 0.
    total_dead_length=0.
    total_biomass = 0.
    total_dead_biomass = 0.
    total_surface=0.
    total_dead_surface=0.
    total_sucrose_root = 0.
    total_hexose_root = 0.
    total_hexose_soil = 0.
    total_CO2 = 0.
    total_CO2_root_growth = 0.
    total_CO2_root_maintenance = 0.
    total_prod_hexose = 0.
    total_hexose_exudation = 0.
    total_hexose_uptake = 0.
    total_net_hexose_exudation = 0.
    total_hexose_degradation = 0.

    C_in_the_root_soil_system = 0.
    C_emitted_towards_the_gazeous_phase = 0.

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        if n.type=="Dead" or n.type=="Just_dead":
            total_dead_biomass += n.biomass
            total_dead_length += n.length
            total_dead_surface += n.external_surface
        else:
            total_length += n.length
            total_biomass += n.biomass
            total_surface += n.external_surface
            # We substract to the current concentration of sugar the possible deficit if the concentration is zero:
            total_sucrose_root += (n.C_sucrose_root * n.biomass - n.Deficit_sucrose_root)
            total_hexose_root += (n.C_hexose_root * n.biomass - n.Deficit_hexose_root)
            total_hexose_soil +=  (n.C_hexose_soil * n.biomass - n.Deficit_hexose_soil)
            total_CO2 += n.resp_maintenance + n.resp_growth
            total_CO2_root_growth += n.resp_growth
            total_CO2_root_maintenance += n.resp_maintenance
            total_prod_hexose += n.prod_hexose
            total_hexose_exudation += n.hexose_exudation
            total_hexose_uptake += n.hexose_uptake
            total_net_hexose_exudation += (n.hexose_exudation - n.hexose_uptake)
            total_hexose_degradation += n.hexose_degradation

    # We subtract the possible global deficit in sucrose from the variable total_sucrose_root:
    global global_sucrose_deficit
    total_sucrose_root = total_sucrose_root - global_sucrose_deficit

    #CARBON BALANCE:
    #--------------
    # We check that the carbon balance is correct (in moles of C):
    C_in_the_root_soil_system = (total_biomass+total_dead_biomass)*biomass_C_content \
                           + total_sucrose_root*12 + total_hexose_root*6. + total_hexose_soil*6.
    C_emitted_towards_the_gazeous_phase = total_CO2

    if printing_total_length:
        print "New state of the root system:"
        print "   The total root length is", \
            "{:.1f}".format(Decimal(total_length*100)), "cm."
    if printing_total_biomass:
        print "   The total root biomass is", \
            "{:.2E}".format(Decimal(total_biomass)), "g, i.e.", \
            "{:.2E}".format(Decimal(total_biomass*biomass_C_content)), "mol of C."
    if printing_all:
        print "   The total amount of sucrose in the roots (including possible deficit) is", \
            "{:.2E}".format(Decimal(total_sucrose_root)), "mol of sucrose, i.e.",\
            "{:.2E}".format(Decimal(total_sucrose_root*12)), "mol of C."
        print "   The total amount of hexose in the roots (including possible deficit) is", \
            "{:.2E}".format(Decimal(total_hexose_root)), "mol of hexose, i.e.",\
            "{:.2E}".format(Decimal(total_hexose_root*6)), "mol of C."
        print "   The total amount of hexose in the soil (including possible deficit) is", \
            "{:.2E}".format(Decimal(total_hexose_soil)), "mol of hexose, i.e.",\
            "{:.2E}".format(Decimal(total_hexose_soil*6)), "mol of C."
        print "   The total amount of CO2 respired by the roots is", \
            "{:.2E}".format(Decimal(total_CO2)), "mol of C, including", \
            "{:.2E}".format(Decimal(total_CO2_root_growth)),"mol of C for growth and", \
            "{:.2E}".format(Decimal(total_CO2_root_maintenance)),"mol of C for maintenance."
        print "   The total net amount of hexose exuded by roots is", \
            "{:.2E}".format(Decimal(total_hexose_exudation-total_hexose_uptake)), "mol of hexose, i.e.", \
            "{:.2E}".format(Decimal(total_net_hexose_exudation*6)), "mol of C."
        print "   The total net amount of hexose degraded in the soil is", \
            "{:.2E}".format(Decimal(total_hexose_degradation)), "mol of hexose, i.e.", \
            "{:.2E}".format(Decimal(total_hexose_degradation*6)), "mol of C."
        print "   The total dead root biomass is", \
            "{:.2E}".format(Decimal(total_dead_biomass)), "g, i.e.", \
            "{:.2E}".format(Decimal(total_dead_biomass*biomass_C_content)), "mol of C."

    dictionnary = {"total_living_root_length": total_length,
                   "total_dead_root_length": total_dead_length,
                   "total_living_root_biomass": total_biomass,
                   "total_dead_root_biomass": total_dead_biomass,
                   "total_living_root_surface": total_surface,
                   "total_dead_root_surface": total_dead_surface,
                   "total_sucrose_root": total_sucrose_root,
                   "total_hexose_root": total_hexose_root,
                   "total_hexose_soil": total_hexose_soil,
                   "total_CO2": total_CO2,
                   "total_CO2_root_growth": total_CO2_root_growth,
                   "total_CO2_root_maintenance": total_CO2_root_maintenance,
                   "total_prod_hexose": total_prod_hexose,
                   "total_hexose_exudation": total_hexose_exudation,
                   "total_hexose_uptake": total_hexose_uptake,
                   "total_hexose_degradation": total_hexose_degradation,
                   "total_net_hexose_exudation": total_hexose_exudation - total_hexose_uptake,
                   "C_in_the_root_soil_system": C_in_the_root_soil_system,
                   "C_emitted_towards_the_gazeous_phase": C_emitted_towards_the_gazeous_phase
                   }

    return dictionnary

def recording_MTG_properties(g,file_name='g_properties.csv'):
    """
    This function records the properties of each node of the MTG "g" in a csv file.
    """

    # We define and reorder the list of all properties of the MTG:
    list_of_properties = list(g.properties().keys())
    list_of_properties.sort()

    # We create an empty list of node indices:
    node_index=[]
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
            node_properties.append(getattr(n,property,"NA"))
        # Finally, we add the new node's properties list as a new item in g_properties:
        g_properties.append(node_properties)
    # We create a list containing the headers of the dataframe:
    column_names=['node_index']
    column_names.extend(list_of_properties)
    # We create the final dataframe:
    data_frame = pd.DataFrame(g_properties, columns=column_names)
    # We record the dataframe as a csv file:
    data_frame.to_csv(file_name, na_rep='NA', index=False, header=True)

########################################################################################################################
########################################################################################################################
# MODULE "SOIL TRANSFORMATION"
########################################################################################################################
########################################################################################################################

# Degradation of hexose in the soil (microbial consumption):
# ---------------------------------------------------------
def soil_hexose_degradation(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    """
    The function "hexose_degradation" computes the decrease of the concentration of hexose outside the root (in mol of
    hexose per gram of root structural biomass) over time (in seconds). It mimics the uptake of hexose by rhizosphere
    microorganisms, and is therefore described using a substrate-limited function (Michaelis-Menten). g represents the
    MTG describing the root system, degradation_max is the maximal degradation of hexose (mol m-2),
    and Km_degradation (mol per gram of root structural biomass) represents the hexose concentration for which the rate
    of hexose_degradation is equal to half of its maximum.
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
                print "WARNING: No degradation in the soil occured for node", n.index(), \
                    "because soil hexose concentration was", n.C_hexose_soil, "mol/g."
            continue

        # hexose_degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.hexose_degradation = n.external_surface * degradation_max * n.C_hexose_soil \
                               / (Km_degradation + n.C_hexose_soil) * time_step_in_seconds

########################################################################################################################
########################################################################################################################
# MODULE "RHIZODEPOSITION"
########################################################################################################################
########################################################################################################################

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def root_hexose_exudation(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    """
    The function "root_hexose_exudation" computes the net amount (in mol of hexose) of hexose accumulated
    outside the root over time (in seconds), without considering any degradation process of hexose
    outside the root or hexose uptake by the root.
    Exudation corresponds to the difference between the efflux of hexose from the root
    to the soil by a passive diffusion. The efflux by diffusion is calculated from the product of the root external
    surface (m2), the permeability coefficient (g m-2) and the gradient of hexose concentration (mol of hexose per gram of dry
    root structural biomass).
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
                print "WARNING: No hexose exudation occured for node", n.index(), \
                    "because root hexose concentration was", n.C_hexose_root, "mol/g."
            continue

        # We calculate the permeability coefficient P according to the distance of the element from the apex:
        # OPTION 1 (Personeni et al. 2007): n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip*100) ** gamma_exudation
        # OPTION 2:
        if n.label == "Apex":
            n.permeability_coeff = Pmax_apex
        elif n.lateral_emergence_possibility == "Possible":
            n.permeability_coeff = Pmax_apex / 5.
        else:
            n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip * 10) ** gamma_exudation

        # hexose_exudation is calculated as an efflux by diffusion, even for dead root elements:
        n.hexose_exudation \
            = n.external_surface * n.permeability_coeff * (n.C_hexose_root - n.C_hexose_soil) * time_step_in_seconds
        if n.hexose_exudation<0.:
            if printing_warnings:
                print "WARNING: a negative exudation flux was calculated for the element", n.index(),"; exudation flux has therefore been set up to zero!"
            n.hexose_exudation=0.


# Uptake of hexose from the soil by the root:
# -------------------------------------------
def root_hexose_uptake(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    """
    The function "root_hexose_uptake" computes the amount (in mol of hexose) of hexose taken up by roots from the soil.
    This influx of hexose is represented as an active process with a substrate-limited
    relationship (Michaelis-Menten function), where Imax (in mol) is the maximal influx, and Km_influx
    (in mol per gram of root structural biomass) represents the hexose concentration for which
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
                print "WARNING: No uptake of hexose from the soil occured for node", n.index(), \
                    "because soil hexose concentration was", n.C_hexose_soil, "mol/g."
            continue
        # We consider that dead elements cannot take up any hexose from the soil:
        if n.type == "Dead":
            n.hexose_uptake=0.

        # The uptake of hexose by the root from the soil is calculated:
        n.hexose_uptake \
            = n.external_surface * Imax * n.C_hexose_soil / (Km_influx + n.C_hexose_soil) * time_step_in_seconds

########################################################################################################################
########################################################################################################################
# MODULE "MAINTENANCE RESPIRATION"
########################################################################################################################
########################################################################################################################

# Function calculating maintenance respiration:
def maintenance_respiration(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    """
    The function "maintenance" calculates the amount resp_maintenance (mol of CO2) corresponding to the consumption
    of a part of the local hexose pool to cover the costs of maintenance processes, i.e. any biological process in the
    root that is NOT linked to the actual growth of the root. The calculation is derived from the model of Thornley and
    Cannell (2000), who initially used this formalism to describe the residual maintenance costs that could not be
    accounted for by known processes. The local amount of CO2 respired for maintenance is calculated as a
    Michaelis-Menten function of the local concentration of hexose "C_hexose_root" (in mol of hexose per gram of root
    structural biomass. "g" represents the MTG describing the root system, "resp_maintenance__max" (mol of CO2 per gram
    of root structural biomass per second) is the maximal rate of maintenance respiration, and "Km_maintenance" (mol of
    hexose per gram of root structural biomass) represents the hexose concentration for which the rate of respiration is
    equal to half of its maximum. "biomass" is the root structural biomass (g) and time is expressed in seconds.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the maintenance respiration:
        n.resp_maintenance=0.

        # First, we ensure that the element has a positive length:
        if n.length <= 0:
            continue
        # We consider that dead elements cannot respire:
        if n.type == "Dead":
            continue
        # We also check whether the concentration of hexose in root is positive or not:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print "WARNING: No maintenance occured for node", n.index(), \
                    "because root hexose concentration was", n.C_hexose_root, "mol/g."
            continue

        # We calculate the number of moles of CO2 generated by maintenance respiration over the time_step:
        n.resp_maintenance = resp_maintenance_max * n.C_hexose_root / (Km_maintenance + n.C_hexose_root) \
                             * n.biomass * time_step_in_seconds

        if n.resp_maintenance<0.:
            print "!!! ERROR: a negative maintenance respiration was calculated for the element", n.index()
            n.resp_maintenance=0.

########################################################################################################################
########################################################################################################################
# MODULE "POTENTIAL GROWTH"
########################################################################################################################
########################################################################################################################

# A FUNCTION FOR CALCULATING THE NEW LENGTH AFTER ELONGATION:
##########################################################

def elongated_length(initial_length, radius, elongation_time_in_seconds, printing_warnings=False):
    # The elongation is calculated following the rules of Pages et al. (2014):
    elongation = EL * 2. * radius * elongation_time_in_seconds
    new_length = initial_length + elongation
    if new_length < initial_length:
        print "!!! ERROR: There is a problem of elongation, with the initial length", initial_length,\
            " and the radius", radius, "and the elongation time", elongation_time_in_seconds
    return new_length


# FUNCTION: A GIVEN APEX CAN FORM A PRIMORDIUM ON ITS SURFACE
###############################################################

def primordium_formation(apex, elongation_rate=0., time_step_in_seconds=1. * 60. * 60. * 24., random=False):
    # NOTE: This function has to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    # We initialize the new_apex that will be returned by the function:
    new_apex = []

    # VERIFICATION: We make sure that no lateral root has already been form on the present apex.
    # We calculate the number of children of the apex (it should be 0):
    n_children = len(apex.children())
    # If there is at least one children:
    if n_children >= 1:
        # We don't add any primordium and simply return the unaltered apex:
        new_apex.append(apex)
        return new_apex

    # We get the order of the current root axis:
    vid = apex.index()
    order = g.order(vid)

    # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
    # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
    # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014).
    # We also set the root angles depending on random:
    if random:
        potential_radius = np.random.normal((apex.radius - Dmin) * RMD + Dmin, ((apex.radius - Dmin) * RMD + Dmin) * CVDD)
        apex_angle_roll = abs(np.random.normal(120, 10))
        if order == 1:
            primordium_angle_down = abs(np.random.normal(45, 10))
        else:
            primordium_angle_down = abs(np.random.normal(70, 10))
        primordium_angle_roll = abs(np.random.normal(3, 3))
    else:
        potential_radius = (apex.radius - Dmin) * RMD + Dmin
        apex_angle_roll=120
        if order == 1:
            primordium_angle_down = 45
        else:
            primordium_angle_down = 70
        primordium_angle_roll = 3

    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:
    if apex.dist_to_ramif > IPD and potential_radius >= Dmin:

        # The distance that the tip of the apex has covered since the actual primordium formation is calculated:
        elongation_since_last_ramif = apex.dist_to_ramif - IPD

        # A specific rolling angle is attributed to the parent apex:
        apex.angle_roll=apex_angle_roll

        # We verify that the apex has actually elongated:
        if apex.actual_elongation > 0:
            # Then the time since the primordium must have been formed is precisely calculated
            # according to the actual growth of the parent apex since primordium formation,
            # taking into account the actual growth rate of the parent defined as
            # apex.actual_elongation / time_step_in_seconds
            time_since_formation = elongation_since_last_ramif / elongation_rate
        else:
            time_since_formation = 0.

        ramif = apex.add_child(edge_type='+',
                               # Characteristics:
                               # -----------------
                               label='Apex',
                               type='Normal_root_before_emergence',
                               # Authorizations and C requirements:
                               # -----------------------------------
                               lateral_emergence_possibility='Impossible',
                               emergence_cost=0.,
                               # Geometry and topology:
                               # -----------------------
                               angle_down=primordium_angle_down,
                               angle_roll=primordium_angle_roll,
                               # The length of the primordium is set to 0:
                               length=0.,
                               radius=potential_radius,
                               potential_length=0.,
                               potential_radius=0.,
                               initial_length=0.,
                               initial_radius=potential_radius,
                               external_surface=0.,
                               volume=0.,
                               # The distance between this root apex and the last ramification on the axis is set to zero by definition:
                               dist_to_ramif=0.,
                               actual_elongation=0.,
                               adventious_emerging_primordium_index=0,
                               # Quantities and concentrations:
                               # -------------------------------
                               biomass=0.,
                               initial_biomass=0.,
                               C_hexose_root=0.,
                               C_hexose_soil=0.,
                               C_sucrose_root=0.,
                               Deficit_hexose_root=0.,
                               Deficit_hexose_soil=0.,
                               Deficit_sucrose_root=0.,
                               # Fluxes:
                               # --------
                               resp_maintenance=0.,
                               resp_growth=0.,
                               hexose_growth_demand=0.,
                               prod_hexose=0.,
                               hexose_exudation=0.,
                               hexose_uptake=0.,
                               hexose_degradation=0.,
                               specific_net_exudation=0.,
                               # Time indications:
                               # ------------------
                               growth_duration=GDs * potential_radius * potential_radius * 4,
                               life_duration=LDs * 2. * potential_radius * RTD,
                               time_since_primordium_formation=time_since_formation,
                               time_since_emergence=0.,
                               potential_time_since_emergence=0.,
                               time_since_growth_stopped=0.,
                               time_since_death=0.
                               )

        # And the new distance between the parent apex and the last ramification is redefined,
        # by taking into account the actual elongation of apex since the child formation:
        apex.dist_to_ramif = elongation_since_last_ramif
        # We also put in memory the index of the child:
        apex.lateral_primordium_index = ramif.index()
        # We add the apex and its ramif in the list of apices returned by the function:
        new_apex.append(apex)
        new_apex.append(ramif)

        # print "The primordium", ramif.index(), "has been formed on node", apex.index(), "."

    return new_apex


# FUNCTION: POTENTIAL APEX DEVELOPMENT
#######################################

def potential_apex_development(apex, time_step_in_seconds=1. * 60. * 60. * 24., Archisimple=False, printing_warnings=False):

    # We initialize an empty list in which the modified apex will be added:
    new_apex = []
    # We record the current radius and length prior to growth as the initial radius and length:
    apex.initial_radius = apex.radius
    apex.initial_length = apex.length
    # We initialize the properties "potential_radius" and "potential_length" returned by the function:
    apex.potential_radius = apex.radius
    apex.potential_length = apex.length

    # CASE 1: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING ADVENTIOUS ROOT
    #-----------------------------------------------------------------------------------------
    # If the adventious root has not emerged yet:
    if apex.type == "Adventious_root_before_emergence":
        global time_since_last_adventious_root_emergence
        global adventious_root_emergence
        # If the time elapsed since the last emergence of adventious root is higher than the prescribed frequency
        # of adventious root emission, and if no other adventious root has already been allowed to emerge:
        if time_since_last_adventious_root_emergence + time_step_in_seconds > 1. / ER \
                and adventious_root_emergence == "Possible":
            # The time since primordium formation is incremented:
            apex.time_since_primordium_formation += time_step_in_seconds
            # The adventious root may have emerged, and the potential time elapsed
            # since its possible emergence over this time step is calculated:
            apex.potential_time_since_emergence = time_since_last_adventious_root_emergence + time_step_in_seconds - 1. / ER
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # If the apex could have emerged sooner:
            if apex.potential_time_since_emergence > time_step_in_seconds:
                # The time since emergence is equal to the time elapsed during this time step (since it must have emerged at this time step):
                apex.potential_time_since_emergence = time_step_in_seconds
            # The corresponding elongation of the apex is calculated:
            apex.potential_length = elongated_length(apex.initial_length, apex.initial_radius,
                                                     elongation_time_in_seconds=apex.potential_time_since_emergence)

            # If ArchiSimple has been choosen as the growth model:
            if Archisimple:
                apex.type="Normal_root_after_emergence"
                # We reset the time since an adventious root may have emerged (REMINDER: it is a "global" value):
                time_since_last_adventious_root_emergence = apex.potential_time_since_emergence
                # We forbid the emergence of other adventious root for the current time step (REMINDER: it is a "global" value):
                adventious_root_emergence = "Impossible"
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # Otherwise, we control the actual emergence of this primordium through the management of the parent:
            else:
                # What this elongation may represent in terms of C inclusion in biomass is calculated:
                potential_surface, potential_volume = surface_and_volume(apex, apex.radius, apex.potential_length)
                emergence_cost = potential_volume * density * biomass_C_content
                # We select the parent on which the primordium is supposed to receive its C, i.e. the base of the root system:
                parent = g.node(1)
                # The possibility of emergence of a lateral root from the parent
                # and the associated biomass C cost are recorded inside the parent:
                parent.lateral_emergence_possibility = "Possible"
                parent.emergence_cost = emergence_cost
                # We record the index of the primordium inside the parent:
                parent.lateral_primordium_index = apex.index()
                # WATCH OUT: THE CODE DOESN'T HANDLE THE SITUATION WHERE MORE THAN ONE ADVENTIOUS ROOT SHOULD EMERGE IN THE SAME TIME STEP!!!!!!!!!!!!!!!!!!!!!!!!!
                # We forbid the emergence of other adventious root for the current time step (REMINDER: it is a "global" value):
                adventious_root_emergence = "Impossible"
                # The new element returned by the function corresponds to the potentially emerging apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
        else:
            # Otherwise, the adventious root cannot emerge at this time step and is left unchanged:
            apex.time_since_primordium_formation += time_step_in_seconds
            new_apex.append(apex)
            # And the function returns this apex and stops here:
            return new_apex

    # CASE 2: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A POTENTIALLY EMERGING NORMAL LATERAL ROOT
    #---------------------------------------------------------------------------------------------
    if apex.type == "Normal_root_before_emergence":
        # If the time since primordium formation is higher than the delay of emergence:
        if apex.time_since_primordium_formation + time_step_in_seconds > emergence_delay:
            # The time since primordium formation is incremented:
            apex.time_since_primordium_formation += time_step_in_seconds
            # The potential time elapsed at the end of this time step since the emergence is calculated:
            apex.potential_time_since_emergence = apex.time_since_primordium_formation - emergence_delay
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # If the apex could have emerged sooner:
            if apex.potential_time_since_emergence > time_step_in_seconds:
                # The time since emergence is equal to the time elapsed during this time step (since it must have emerged at this time step):
                apex.potential_time_since_emergence = time_step_in_seconds
            # The corresponding elongation of the apex is calculated:
            apex.potential_length = elongated_length(apex.initial_length, apex.initial_radius, apex.potential_time_since_emergence)

            # If ArchiSimple has been choosen as the growth model:
            if Archisimple:
                apex.type="Normal_root_after_emergence"
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # Otherwise, we control the actual emergence of this primordium through the management of the parent:
            else:
                # What this elongation may represent in terms of C inclusion in biomass is calculated:
                potential_surface, potential_volume = surface_and_volume(apex, apex.radius, apex.potential_length)
                emergence_cost = potential_volume * density * biomass_C_content
                # We select the parent on which the primordium has been formed:
                vid = apex.index()
                index_parent = g.Father(vid, EdgeType='+')
                parent = g.node(index_parent)
                # The possibility of emergence of a lateral root from the parent
                # and the associated biomass C cost are recorded inside the parent:
                parent.lateral_emergence_possibility = "Possible"
                parent.emergence_cost = emergence_cost
                parent.lateral_primordium_index = apex.index()
                # And the new element returned by the function corresponds to the potentially emerging apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
        # Otherwise, the time since primordium formation is simply incremented:
        else:
            apex.time_since_primordium_formation += time_step_in_seconds
            # And the new element returned by the function corresponds to the modified apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

    # CASE 3: THE APEX BELONGS TO AN AXIS THAT HAS ALREADY EMERGED:
    #--------------------------------------------------------------
    # IF THE APEX CAN CONTINUE GROWING:
    if apex.time_since_emergence + time_step_in_seconds < apex.growth_duration:
        # The times are incremented:
        apex.time_since_primordium_formation += time_step_in_seconds
        apex.time_since_emergence += time_step_in_seconds
        # The corresponding potential elongation of the apex is calculated:
        apex.potential_length = elongated_length(apex.length, apex.radius, time_step_in_seconds)
        # And the new element returned by the function corresponds to the modified apex:
        new_apex.append(apex)
        # And the function returns this new apex and stops here:
        return new_apex
    # OTHERWISE, THE APEX HAD TO STOP:
    else:
        # IF THE APEX HAS NOT REACHED ITS LIFE DURATION:
        if apex.time_since_growth_stopped + time_step_in_seconds < apex.life_duration:
            # IF THE APEX HAS ALREADY BEEN STOPPED AT A PREVIOUS TIME STEP:
            if apex.type == "Stopped" or apex.type == "Just_stopped":
                # The time since growth stopped is simply increased by one time step:
                apex.time_since_growth_stopped += time_step_in_seconds
                # The type is (re)declared "Stopped":
                apex.type == "Stopped"
                # The times are incremented:
                apex.time_since_primordium_formation += time_step_in_seconds
                apex.time_since_emergence += time_step_in_seconds
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # OTHERWISE, THE APEX HAS TO STOP DURING THIS TIME STEP:
            else:
                # The type is declared "Just stopped":
                apex.type = "Just_stopped"
                # Then the exact time since growth stopped is calculated:
                apex.time_since_growth_stopped = apex.time_since_emergence + time_step_in_seconds - apex.growth_duration
                # And the potential elongation of the apex before growth stopped is calculated:
                apex.potential_length = elongated_length(apex.length, apex.radius,
                                                         elongation_time_in_seconds=time_step_in_seconds - apex.time_since_growth_stopped)
                # VERIFICATION:
                if time_step_in_seconds - apex.time_since_growth_stopped < 0.:
                    print "!!! ERROR: The apex", apex.index(), "has stopped since", \
                        apex.time_since_growth_stopped, "seconds; the time step is", time_step_in_seconds
                    print "    We set the potential length of this apex equal to its initial length."
                    apex.potential_length = apex.initial_length

                # The times are incremented:
                apex.time_since_primordium_formation += time_step_in_seconds
                apex.time_since_emergence += time_step_in_seconds
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

        # OTHERWISE, THE APEX HAS TO BE DEAD:
        else:
            # IF THE APEX HAS ALREADY DIED AT A PREVIOUS TIME STEP:
            if apex.type == "Dead" or apex.type == "Just_dead":
                # The type is (re)declared "Dead":
                apex.type = "Dead"
                # And the times are simply incremented:
                apex.time_since_primordium_formation += time_step_in_seconds
                apex.time_since_emergence += time_step_in_seconds
                apex.time_since_growth_stopped += time_step_in_seconds
                apex.time_since_death += time_step_in_seconds
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex
            # OTHERWISE, THE APEX HAS TO DIE DURING THIS TIME STEP:
            else:
                # Then the apex is declared "Just dead":
                apex.type = "Just_dead"
                # The exact time since the apex died is calculated:
                apex.time_since_death = apex.time_since_growth_stopped + time_step_in_seconds - apex.life_duration
                # And the other times are incremented:
                apex.time_since_primordium_formation += time_step_in_seconds
                apex.time_since_emergence += time_step_in_seconds
                apex.time_since_growth_stopped += time_step_in_seconds
                # The new element returned by the function corresponds to this apex:
                new_apex.append(apex)
                # And the function returns this new apex and stops here:
                return new_apex

    # VERIFICATION: If the apex does not match any of the cases listed above:
    print "!!! ERROR: No case found for defining growth of apex", apex.index(), "of type", apex.type
    new_apex.append(apex)
    return new_apex

# FUNCTION: POTENTIAL SEGMENT DEVELOPMENT
#########################################

def potential_segment_development(segment, time_step_in_seconds=60.*60.*24., radial_growth="Possible"):

    # We initialize an empty list that will contain the new segment to be returned:
    new_segment = []
    # We record the current radius and length prior to growth as the initial radius and length:
    segment.initial_radius = segment.radius
    segment.initial_length = segment.length
    # We initialize the properties "potential_radius" and "potential_length":
    segment.potential_radius = segment.radius
    segment.potential_length = segment.length

    # We initialize internal variables:
    son_section = 0.
    sum_of_lateral_sections = 0.
    death_count = 0.

    # For each child of the segment:
    for child in segment.children():

        if child.radius < 0.:
            print "!!! ERROR: the radius of the element", child.index(), "is negative!"
        # If the child belongs to the same axis:
        if child.properties()['edge_type']=='<':
            # Then we record the section of this child:
            son_section = child.radius * child.radius * pi
        # Otherwise if the child is the segment of a lateral root AND if this lateral root has already emerged:
        elif child.properties()['edge_type']=='+'and child.length>0.:
            # We add the section of this child to a sum of lateral sections:
            sum_of_lateral_sections += child.radius * child.radius * pi

        # If this child is dead:
        if child.type == "Dead" or child.type == "Just_dead":
            # Then we add one dead child to the death count:
            death_count += 1

    # If each child in the list of children has been recognized as dead:
    if death_count == len(segment.children()):
        # Then the segment becomes dead:
        segment.type = "Dead"
        # Otherwise, at least one of the children axis is not dead, so the father segment should not be dead

    # If the radial growth is possible:
    if radial_growth == "Possible":
        # The radius of the root segment is defined according to the pipe model.
        # In ArchiSimp9, the radius is increased by considering the sum of the sections of all the children,
        # by adding a fraction (SGC) of this sum of sections to the current section of the parent segment,
        # and by calculating the new radius that corresponds to this new section of the parent:
        segment.potential_radius = sqrt(son_section / pi + SGC * sum_of_lateral_sections / pi)
        # However, if the net difference is below 0.1% of the initial radius:
        if (segment.potential_radius - segment.initial_radius) <= 0.001*segment.initial_radius:
            # Then the potential radius is set to the initial radius:
            segment.potential_radius = segment.initial_radius
        # And if the segment corresponds to one of the elements of length 0 supporting one adventious root:
        if segment.type == "Support_for_adventious_root":
            # Then the radius is directly increased, as this element will not be considered in the function calculating actual growth:
            segment.radius = segment.potential_radius

    # We increase the various time variables:
    segment.time_since_primordium_formation += time_step_in_seconds
    segment.time_since_emergence += time_step_in_seconds
    if segment.type=="Stopped" or segment.type=="Just_stopped":
        segment.time_since_growth_stopped += time_step_in_seconds
    if segment.type == "Dead" or segment.type=="Just_dead":
        segment.time_since_growth_stopped += time_step_in_seconds
        segment.time_since_death += time_step_in_seconds

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
        root = root_gen.next()
        self.apices_list = [g.node(v) for v in pre_order(g, root) if g.label(v) == 'Apex']
        self.segments_list = [g.node(v) for v in post_order(g, root) if g.label(v) == 'Segment']

    def step(self, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth="Possible",
             Archisimple=False):
        list_of_apices = list(self.apices_list)
        list_of_segments = list(self.segments_list)
        # print "The root system contains", len(list_of_apices) + len(list_of_segments),"different elements."

        # For each apex in the list of apices:
        for apex in list_of_apices:
            new_apices = []
            # We define the new list of apices with the function apex_development:
            new_apices.append(potential_apex_development(apex, time_step_in_seconds=time_step_in_seconds, Archisimple=Archisimple))
            # We add these new apices to apex:
            self.apices_list.extend(new_apices)
        # For each segment in the list of segments:
        for segment in list_of_segments:
            new_segments = []
            # We define the new list of apices with the function apex_development:
            new_segments.append(potential_segment_development(segment, time_step_in_seconds=time_step_in_seconds, radial_growth=radial_growth))
            # We add these new apices to apex:
            self.segments_list.extend(new_segments)


# We finally define the function that calculates the potential growth of the whole MTG at a given time step:
def potential_growth(g, time_step_in_seconds=1. * (60. * 60. * 24.), radial_growth="Possible", Archisimple=False):
    # We simulate the development of all apices and segments in the MTG:
    simulator = Simulate_potential_growth(g)
    simulator.step(time_step_in_seconds=time_step_in_seconds, radial_growth=radial_growth, Archisimple=Archisimple)


# FUNCTION: A GIVEN APEX CAN BE TRANSFORMED INTO SEGMENTS AND GENERATE NEW PRIMORDIA:
#####################################################################################

def segmentation_and_primordium_formation(apex, time_step_in_seconds=1. * 60. * 60. * 24., Archisimple=False, random=True):
    # NOTE: This function is supposed to be called AFTER the actual elongation of the apex has been done and the distance
    # between the tip of the apex and the last ramification (dist_to_ramif) has been increased!

    # Optional - We can add random geometry, or not:
    if random:
        np.random.seed(random_choice + apex.index())
        angle_mean=3
        angle_var=3
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
    initial_biomass = apex.biomass
    initial_resp_maintenance = apex.resp_maintenance
    initial_resp_growth = apex.resp_growth
    initial_hexose_exudation = apex.hexose_exudation
    initial_hexose_uptake = apex.hexose_uptake
    initial_hexose_degradation = apex.hexose_degradation
    initial_hexose_growth_demand = apex.hexose_growth_demand
    initial_hexose_consumption_by_growth = apex.hexose_consumption_by_growth
    initial_Deficit_sucrose_root = apex.Deficit_sucrose_root
    initial_Deficit_hexose_root = apex.Deficit_hexose_root
    initial_Deficit_hexose_soil = apex.Deficit_hexose_soil

    # We record the type of the apex, as it may correspond to an apex that has stopped (or even died):
    initial_type = apex.type
    initial_lateral_emergence_possibility = apex.lateral_emergence_possibility

    # If the length of the apex is smaller than the defined length of a root segment:
    if apex.length <= segment_length:
        # We assume that the growth functions that may have been called previously have only modified radius and length,
        # but not the biomass and the total amounts present in the root element.
        # We modify the geometrical features of the present element according to the new length and radius:
        apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
        apex.biomass = apex.volume * density
        # We modify the variables representing total amounts according to the new biomass:
        mass_fraction = apex.biomass / initial_biomass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        # We simply call the function primordium_formation to check whether a primordium should have been formed
        # (Note: we assume that the segment length is always smaller than the inter-branching distance IBD,
        # so that in this case, only 0 or 1 primordium may have been formed - the function is called only once):
        new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,random=random))

    else:
        # Otherwise, we have to calculate the number of entire segments within the apex.
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
            apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
            apex.biomass = apex.volume * density
            # We modify the variables representing total amounts according to the new biomass:
            mass_fraction = apex.biomass / initial_biomass
            apex.resp_maintenance = initial_resp_maintenance * mass_fraction
            apex.resp_growth = initial_resp_growth * mass_fraction
            apex.hexose_exudation = initial_hexose_exudation * mass_fraction
            apex.hexose_uptake = initial_hexose_uptake * mass_fraction
            apex.hexose_degradation = initial_hexose_degradation * mass_fraction
            apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
            apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
            apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
            apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
            apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
            # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
            new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,random=random))
            # The current element that has been elongated up to segment_length is now considered as a segment:
            apex.label = 'Segment'

            # And we add a new apex after this segment, initially of length 0, that is now the new element called "apex":
            apex = apex.add_child(edge_type='<',
                                  # Characteristics:
                                  # -----------------
                                  label='Apex',
                                  type=apex.type,
                                  # Authorizations and C requirements:
                                  # -----------------------------------
                                  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                  lateral_emergence_possibility="Impossible",
                                  emergence_cost=0.,
                                  # Geometry and topology:
                                  # -----------------------
                                  angle_down=segment_angle_down,
                                  angle_roll=segment_angle_roll,
                                  # The length of the primordium is set to 0:
                                  length=0.,
                                  radius=apex.radius,
                                  potential_length=0.,
                                  potential_radius=apex.radius,
                                  initial_length=0.,
                                  initial_radius=apex.radius,
                                  external_surface=0.,
                                  volume=0.,
                                  # The dist_to_ramif of the new apex is identical to the dist_to_ramif of the preceding segment:
                                  dist_to_ramif=apex.dist_to_ramif,
                                  # The elongation that this apex has virtually done over a fraction of the time step is:
                                  actual_elongation=segment_length * i,
                                  adventious_emerging_primordium_index=apex.adventious_emerging_primordium_index,
                                  # Quantities and concentrations:
                                  # -------------------------------
                                  biomass=0.,
                                  initial_biomass=0.,
                                  C_hexose_root=apex.C_hexose_root,
                                  C_hexose_soil=apex.C_hexose_soil,
                                  C_sucrose_root=apex.C_sucrose_root,
                                  Deficit_hexose_root=0.,
                                  Deficit_hexose_soil=0.,
                                  Deficit_sucrose_root=0.,
                                  # Fluxes:
                                  # -------
                                  resp_maintenance=0.,
                                  resp_growth=0.,
                                  hexose_growth_demand=0.,
                                  prod_hexose=0.,
                                  hexose_exudation=0.,
                                  hexose_uptake=0.,
                                  hexose_degradation=0.,
                                  hexose_consumption_by_growth=0.,
                                  specific_net_exudation=0.,
                                  # Time indications:
                                  # ------------------
                                  growth_duration=apex.growth_duration,
                                  life_duration=apex.life_duration,
                                  time_since_primordium_formation=apex.time_since_primordium_formation,
                                  time_since_emergence=apex.time_since_emergence,
                                  time_since_growth_stopped=apex.time_since_growth_stopped,
                                  time_since_death=apex.time_since_death
                                  )

        # Finally, we do this operation one last time for the last segment:
        # We define the length of the present element as the constant length of a segment:
        apex.length = segment_length
        apex.potential_length = apex.length
        apex.initial_length = apex.length
        # We define the new dist_to_ramif, which is smaller than the one of the initial apex:
        apex.dist_to_ramif = initial_dist_to_ramif - (initial_length - segment_length * n_segments)
        # We modify the geometrical features of the present element according to the new length:
        apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
        apex.biomass = apex.volume * density
        # We modify the variables representing total amounts according to the new biomass:
        mass_fraction = apex.biomass / initial_biomass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
        new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,random=random))
        # The current element that has been elongated up to segment_length is now considered as a segment:
        apex.label = 'Segment'

        # And we define the new, final apex after the new defined segment, with a new length defined as:
        new_length = initial_length - n_segments * segment_length
        apex = apex.add_child(edge_type='<',
                              # Characteristics:
                              # -----------------
                              label='Apex',
                              type=apex.type,
                              # Authorizations and C requirements:
                              # -----------------------------------
                              # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                              lateral_emergence_possibility="Impossible",
                              emergence_cost=0.,
                              # Geometry and topology:
                              # -----------------------
                              angle_down=apex_angle_down,
                              angle_roll=apex_angle_roll,
                              # The length of the primordium is set to 0:
                              length=new_length,
                              radius=apex.radius,
                              potential_length=new_length,
                              potential_radius=apex.radius,
                              initial_length=new_length,
                              initial_radius=apex.radius,
                              external_surface=0.,
                              volume=0.,
                              # The dist_to_ramif of the new apex is identical to the initial dist_to_ramif:
                              dist_to_ramif=initial_dist_to_ramif,
                              # The elongation that this apex has done is identical to the initial elongation:
                              actual_elongation=initial_elongation,
                              adventious_emerging_primordium_index=apex.adventious_emerging_primordium_index,
                              # Quantities and concentrations:
                              # -------------------------------
                              biomass=0.,
                              initial_biomass=0.,
                              C_hexose_root=apex.C_hexose_root,
                              C_hexose_soil=apex.C_hexose_soil,
                              C_sucrose_root=apex.C_sucrose_root,
                              Deficit_hexose_root=0.,
                              Deficit_hexose_soil=0.,
                              Deficit_sucrose_root=0.,
                              # Fluxes:
                              # -------
                              resp_maintenance=0.,
                              resp_growth=0.,
                              hexose_growth_demand=0.,
                              prod_hexose=0.,
                              hexose_exudation=0.,
                              hexose_uptake=0.,
                              hexose_degradation=0.,
                              hexose_consumption_by_growth=0.,
                              specific_net_exudation=0.,
                              # Time indications:
                              # ------------------
                              growth_duration=apex.growth_duration,
                              life_duration=apex.life_duration,
                              time_since_primordium_formation=apex.time_since_primordium_formation,
                              time_since_emergence=apex.time_since_emergence,
                              time_since_growth_stopped=apex.time_since_growth_stopped,
                              time_since_death=apex.time_since_death
                              )
        # We modify the geometrical features of the new apex according to the defined length:
        apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
        apex.biomass = apex.volume * density
        # We modify the variables representing total amounts according to the new biomass:
        mass_fraction = apex.biomass / initial_biomass
        apex.resp_maintenance = initial_resp_maintenance * mass_fraction
        apex.resp_growth = initial_resp_growth * mass_fraction
        apex.hexose_exudation = initial_hexose_exudation * mass_fraction
        apex.hexose_uptake = initial_hexose_uptake * mass_fraction
        apex.hexose_degradation = initial_hexose_degradation * mass_fraction
        apex.hexose_growth_demand = initial_hexose_growth_demand * mass_fraction
        apex.hexose_consumption_by_growth = initial_hexose_consumption_by_growth * mass_fraction
        apex.Deficit_sucrose_root = initial_Deficit_sucrose_root * mass_fraction
        apex.Deficit_hexose_root = initial_Deficit_hexose_root * mass_fraction
        apex.Deficit_hexose_soil = initial_Deficit_hexose_soil * mass_fraction
        # And we call the function primordium_formation to check whether a primordium should have been formed
        new_apex.append(primordium_formation(apex, elongation_rate=initial_elongation_rate,
                                             time_step_in_seconds=time_step_in_seconds,random=random))
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

    def step(self, time_step_in_seconds, random=True):
        # We define "apices_list" as the list of all apices in g:
        apices_list = list(self._apices)
        # For each apex in the list of apices:
        for apex in apices_list:
            if apex.type == "Normal_root_after_emergence" and apex.length > 0.:  # Is it needed? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # We define the new list of apices with the function apex_development:
                new_apex = segmentation_and_primordium_formation(apex, time_step_in_seconds, random=random)
                # We add these new apices to apex:
                self._apices.extend(new_apex)


# We finally define the function that creates new segments and priomordia in "g":
def segmentation_and_primordia_formation(g, time_step_in_seconds=1. * 60. * 60. * 24., random=True, printing_warnings=False):
    # We simulate the segmentation of all apices:
    simulator = Simulate_segmentation_and_primordia_formation(g)
    simulator.step(time_step_in_seconds, random=random)


########################################################################################################################
########################################################################################################################
# MODULE "ACTUAL GROWTH AND ASSOCIATED RESPIRATION"
########################################################################################################################
########################################################################################################################

# ACTUAL ELONGATION AND RADIAL GROWTH OF ROOT ELEMENTS:
#######################################################

# Function calculating the actual growth and the corresponding growth respiration:
def actual_growth_and_corresponding_respiration(g, time_step_in_seconds, printing_warnings=False):
    """
    This function defines how a segment, an apex and possibly an emerging root primordium will grow according to the amount
    of hexose present in the segment, taking into account growth respiration based on the model of Thornley and Cannell
    (2000). The calculation is based on the values of potential_radius, potential_length, lateral_emergence_possibility
    and emergence_cost defined in each element by the module "POTENTIAL GROWTH".
    The function returns the MTG "g" with modified values of radius and length of each element, the possibility of the
    emergence of lateral roots, and the cost of growth in terms of hexose demand.
    """

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):

        # n represents the current root element:
        n = g.node(vid)

        # We make sure that the initial values of length, radius and biomass are correctly calculated:
        n.initial_length = n.length
        n.initial_radius = n.radius
        # We calculate the initial surface and volume of the element:
        initial_surface, initial_volume = surface_and_volume(n, n.initial_radius, n.initial_length)
        # The initial biomass of the element is also recorded:
        n.initial_biomass = initial_volume * density

        # We reinitialize growth-related variables:
        n.hexose_growth_demand=0.
        n.hexose_consumption_by_growth=0.
        n.resp_growth = 0.
        n.actual_elongation = 0.
        n.actual_elongation_rate = 0.

        # AVOIDANCE OF UNWANTED CASES:
        #------------------------------
        # We make sure that the element is not dead:
        if n.type == "Dead" or n.type == "Just_dead":
            # In such case, we just pass to the next element in the iteration:
            continue
        # We make sure that the root elements of length 0 located after the base that support adventious roots are not considered:
        if n.type == "Support_for_adventious_root":
            # In such case, we just pass to the next element in the iteration:
            continue
        # We make sure that the element does not correspond to a primordium that has not emerged,
        # as they are managed through their parent. If the element is a primordium:
        if n.type == "Normal_root_before_emergence" or n.type == "Adventious_root_before_emergence":
            # In such case, we just pass to the next element in the iteration:
            continue
        # We verify that the initial biomass of the element is strictly positive:
        if n.biomass <= 0.:
            print "!!! ERROR: the biomass of the element", n.index(), "before actual growth consideration is", n.biomass
            print "    This element was therefore not considered for actual growth at this time step."
            # In such case, we just pass to the next element in the iteration:
            continue
        # We make sure that the concentration of hexose is not nil or negative, otherwise no growth is possible:
        if n.C_hexose_root <= 0.:
            if printing_warnings:
                print "WARNING: No actual growth or lateral emergence occured for node", n.index(), \
                    "because root hexose concentration was", n.C_hexose_root, "mol/g."
            # In such case, we just pass to the next element in the iteration:
            continue
        # We make sure that there is a potential growth (of the current element or its primordium):
        if n.potential_length <= n.initial_length \
                and n.potential_radius <= n.initial_radius \
                and n.lateral_emergence_possibility != "Possible":
            # In such case, we just pass to the next element in the iteration:
            continue

        # INITIALIZATION AND CALCULATIONS OF POTENTIAL GROWTH DEMAND IN HEXOSE:
        #----------------------------------------------------------------------
        # We calculate the potential surface and volume of the element based on the potential radius and potential length:
        potential_surface, potential_volume = surface_and_volume(n, n.potential_radius, n.potential_length)
        # PRECAUTION: If the emergence of a primordium is not possible:
        if n.lateral_emergence_possibility != "Possible":
            # Then the emergence cost is set to 0 (else, the emergence cost has already been calculated):
            n.emergence_cost = 0.
        # We calculate the number of moles of C included in the biomass potentially produced over the time_step,
        # where density is the dry structural weight per volume (g m-3), biomass_C_content is the amount of C
        # per gram of dry structural mass (mol_C g-1) and n.emergence_cost the biomass C requirement (mol of C) of a lateral
        # primordium to be emerged, if any:
        potential_growth_demand = (potential_volume - initial_volume) * density * biomass_C_content + n.emergence_cost

        # We verify that this potential growth demand is positive:
        if potential_growth_demand < 0.:
            print "!!! ERROR: a negative growth demand of", potential_growth_demand, \
                "was calculated for the element", n.index(), "of class", n.label
            print "    The initial volume is", initial_volume, "the potential volume is", potential_volume, \
                "and the emergence cost is", n.emergence_cost
            print "    The initial length was", n.initial_length, "and the potential length was", n.potential_length
            print "    The initial radius was", n.initial_radius, "and the potential radius was", n.potential_radius
            potential_growth_demand = 0.
        # The amount of hexose (mol of hexose) required for sustaining the potential growth is calculated by including
        # the respiration cost according to the model of Thornley and Cannell (2000):
        n.hexose_growth_demand = 1. / 6. * potential_growth_demand / yield_growth
        # The total amount of hexose available at this stage in the root element is calculated:
        n.hexose_available = n.C_hexose_root * n.initial_biomass
        # We initialize the temporary variable "remaining_hexose" that computes the amount of hexose left for growth:
        remaining_hexose = n.hexose_available
        # We calculate the maximal possible volume of the root element (without considering primordium emergence) according to all the hexose available:
        volume_max = initial_volume + n.hexose_available * 6. / (density * biomass_C_content) * yield_growth
        # We calculate the maximal possible length of the root element based on a constant radius and no lateral root emergence:
        length_max = volume_max / (pi * n.initial_radius ** 2)

        # ACTUAL GROWTH: PRIORITY IS GIVEN TO ROOT ELONGATION, THEN LATERAL ROOT EMERGENCE, THEN RADIAL GROWTH
        #-----------------------------------------------------------------------------------------------------
        # If elongation should occur but is limited by the amount of hexose available:
        if length_max <= n.potential_length and n.potential_length > n.initial_length:
            # ELONGATION IS LIMITED using all the amount of hexose available;
            # no radial growth and no primordium emergence are therefore possible:
            n.length = length_max
            remaining_hexose=0.
            # The radius or the emergence cost and possibility of a primordium are left unchanged.
        # Otherwise, at least one other type of growth can be performed (primordium emergence or radial growth):
        else:
            # ELONGATION IS DONE UP TO THE FULL POTENTIAL (including the case where potential elongation is 0):
            n.length = n.potential_length
            # The corresponding new volume is calculated:
            volume_after_elongation = (pi * n.initial_radius ** 2) * n.length
            # We then calculate the remaining amount of hexose after elongation:
            remaining_hexose = n.hexose_available - 1. / 6. * (volume_after_elongation - initial_volume) * density * biomass_C_content / yield_growth
            # EMERGENCE OF A LATERAL ROOT IS THEN CONSIDERED:
            # If the emergence of the primordium has been declared possible by the potential growth module:
            if n.lateral_emergence_possibility == "Possible" and remaining_hexose >0.:
                # We get the node corresponding to the primordium:
                index_primordium = n.lateral_primordium_index
                primordium = g.node(index_primordium)
                # If the primordium corresponds to an adventious root that can now emerge:
                if primordium.type == "Adventious_root_before_emergence":
                    # We reset the time since an adventious root has emerged (REMINDER: it is a "global" value):
                    global time_since_last_adventious_root_emergence
                    time_since_last_adventious_root_emergence = primordium.potential_time_since_emergence

                # If there is enough hexose for the complete primordium emergence:
                if remaining_hexose - n.emergence_cost * 1. / 6. / yield_growth >= 0.:
                    # The emergence of the primordium is done up to the full length:
                    primordium.length = primordium.potential_length
                    # And the remaining hexose is calculated after the emergence:
                    remaining_hexose = remaining_hexose - n.emergence_cost * 1. / 6. / yield_growth
                # Otherwise, the emergence is done proportionally to the available hexose, and no radial growth occurs:
                else:
                    # We calculate the net increase in volume of the emerging primordium:
                    corresponding_volume = remaining_hexose * 6. * yield_growth / (density * biomass_C_content)
                    # The emergence of the primordium is done up to a certain length:
                    primordium.length = corresponding_volume / (primordium.radius ** 2 * pi)
                    # No hexose remains anymore:
                    remaining_hexose = 0.
                # In both cases, following properties of the new lateral root element are modified:
                primordium.dist_to_ramif = primordium.length
                primordium.label = "Apex"
                primordium.type = "Normal_root_after_emergence"
                n.lateral_emergence_possibility = "Impossible"
                # The exact time since emergence is recorded:
                primordium.time_since_emergence = primordium.potential_time_since_emergence
                # The actual elongation rate is calculated:
                primordium.actual_elongation = primordium.length
                primordium.actual_elongation_rate = primordium.length / primordium.time_since_emergence
                # The corresponding external surface and volume of the emerged apex are calculated:
                primordium.external_surface, primordium.volume = surface_and_volume(primordium, primordium.radius, primordium.length)
                primordium.biomass = primordium.volume * density
                primordium.initial_biomass = primordium.biomass
                # Note: at this stage, no sugar has been allocated to the primordium itself!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # EVENTUALLY, RADIAL GROWTH IS CONSIDERED according to the remaining amount of hexose, if any:
            if n.potential_radius > n.initial_radius:
                possible_radial_increase_in_volume = remaining_hexose * 6. * yield_growth / (density * biomass_C_content)
                # We calculate the maximal possible volume based on the volume of the new cylinder after elongation
                # and the increase in volume that could be achieved by consuming all the remaining hexose:
                volume_max = (pi*n.initial_radius**2) * n.length + possible_radial_increase_in_volume
                # We then calculate the corresponding new possible radius corresponding to this maximum volume:
                possible_radius = sqrt((volume_max) / (n.length * pi))
                # VERIFICATION:
                if possible_radius < 0.9999 * n.initial_radius: # We authorize a difference of 0.01% due to calculation errors!
                    print "!!! ERROR: the calculated new radius of element",n.index(),"is lower than the initial one!"
                    print "    The possible radius was", possible_radius, "and the initial radius was", n.initial_radius
                # If the maximal radius that can be obtained is lower than the potential radius asked by the potential growth module:
                if possible_radius <= n.potential_radius:
                    n.radius = possible_radius
                    remaining_hexose = 0.
                else:
                    n.radius = n.potential_radius
                    remaining_hexose = n.hexose_available - n.hexose_growth_demand

        # IN ALL CASES:
        # The overall cost of growth is calculated as:
        n.hexose_consumption_by_growth = n.hexose_available - remaining_hexose
        # The amount of hexose that has been used for growth respiration is calculated and transformed into moles of CO2:
        n.resp_growth = (1./yield_growth - 1.) * n.hexose_consumption_by_growth * 6.
        # The new surface and volume of the element are automatically calculated:
        n.external_surface, n.volume = surface_and_volume(n, n.radius, n.length)
        # The new dry structural biomass of the element is calculated from its new volume:
        n.biomass = n.volume * density
        # The actual elongation rate is calculated:
        n.actual_elongation = n.length - n.initial_length
        n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

        # Verification: we check that no negative length or biomass have been generated!
        if n.length < 0 or n.biomass < 0:
            print "!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length, "and a mass of", n.biomass
            # We then reset all the geometrical values to their initial values:
            n.length = n.initial_length
            n.radius = n.initial_radius
            n.biomass = n.initial_biomass
            n.external_surface = initial_surface
            n.volume = initial_volume

        # In case the root element corresponds to an apex, the distance to the last ramification is increased:
        if n.label == "Apex":
            n.dist_to_ramif += n.length - n.initial_length

    return g

# Function calculating a satisfaction coefficient for the growth of the whole root system, based on the "available biomass":
def satisfaction_coefficient(g, biomass_input):

    #We initialize the sum of individual demands for biomass:
    sum_biomass_demand=0.
    SC=0.

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):
        # n represents the current root element:
        n = g.node(vid)

        # We calculate the initial surface and volume of the element:
        initial_surface, initial_volume = surface_and_volume(n, n.initial_radius, n.initial_length)
        # We calculate the potential surface and volume of the element based on the potential radius and potential length:
        potential_surface, potential_volume = surface_and_volume(n, n.potential_radius, n.potential_length)

        # The growth demand of the element in biomass is calculated:
        n.growth_demand_in_biomass = (potential_volume - initial_volume) * density
        sum_biomass_demand += n.growth_demand_in_biomass

    # We calculate the overall satisfaction coefficient SC described by Pages et al. (2014):
    if sum_biomass_demand <=0:
        SC=1.
    else:
        SC = biomass_input / sum_biomass_demand

    return SC

# Function performing the actual growth of each element based on the potential growth and the satisfaction coefficient SC:
def archisimple_growth(g, SC, time_step_in_seconds,printing_warnings=False):
    """
    SC is the satisfaction coefficient for growth calculated on the whole root system.
    """

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):

        # n represents the current root element:
        n = g.node(vid)

        # We make sure that the element is not dead and has not already been stopped at the previous time step:
        if n.type == "Dead" or n.type == "Just_dead" or n.type == "Stopped":
            # Then we pass to the next element in the iteration:
            continue
        # We make sure that the root elements at the basis that support adventious root are not considered:
        if n.type == "Support_for_adventious_root":
            # Then we pass to the next element in the iteration:
            continue

        # We perform each type of growth according to the satisfaction coefficient SC:
        if SC > 1.:
            relative_reduction=1.
        else:
            relative_reduction=SC

        #WARNING: This approach is not an exact C balance on the root system! The relative reduction of growth caused
        # by SC should not be the same between elongation and radial growth!
        n.length += (n.potential_length - n.initial_length)*relative_reduction
        n.actual_elongation = n.length - n.initial_length

        # We calculate the actual elongation rate of this element:
        if n.potential_time_since_emergence >0 and n.potential_time_since_emergence < time_step_in_seconds:
            n.actual_elongation_rate = n.actual_elongation/n.potential_time_since_emergence
        else:
            n.actual_elongation_rate = n.actual_elongation / time_step_in_seconds

        n.radius += (n.potential_radius - n.initial_radius) * relative_reduction
        # The surface and volume of the element are automatically calculated:
        n.external_surface, n.volume = surface_and_volume(n, n.radius, n.length)
        # The new dry structural biomass of the element is calculated from its new volume:
        n.biomass = n.volume * density

        # In case where the root element corresponds to an apex, the distance to the last ramification is increased:
        if n.label == "Apex":
            n.dist_to_ramif += n.actual_elongation

        # VERIFICATION:
        if n.length < 0 or n.biomass < 0:
            print "!!! ERROR: the element", n.index(), "of class", n.label, "has a length of", n.length, "and a mass of", n.biomass

    return g

########################################################################################################################
########################################################################################################################
# MODULE "CONVERSION OF SUCROSE INTO HEXOSE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def sucrose_to_hexose(g, time_step_in_seconds=1. * (60. * 60. * 24.), printing_warnings=False):
    """
    The function "sucrose_to_hexose" simulates the process of sucrose unloading from phloem over time
    (in seconds) and its immediate conversion into hexose, for a given root element (cylinder or cone)
    with an external surface (m2). It returns the variable prod_hexose (in mol), considering that 2 mol of hexose
    are produced for 1 mol of sucrose.
    The unloading of sucrose is represented as an active process with a substrate-limited relationship
    (Michaelis-Menten function), where unloading_coeff (in mol m-2 s-1) is the maximal amount of sucrose unloading
    and Km_unloading (in mol per gram of root structural biomass) represents the sucrose concentration
    for which sucrose_to_hexose is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        # We re-initialize the unloading coefficient and the production of hexose:
        n.unloading_coeff = 0.
        n.prod_hexose=0.

        # First, we ensure that the element does not correspond to a primordium that has not emerged:
        if n.length <= 0.:
            continue
        # We also check whether the concentration of sucrose in root is positive or not:
        if n.C_sucrose_root <= 0.:
            if printing_warnings:
                print "WARNING: No unloading occured for node", n.index(), \
                    "because root sucrose concentration was", n.C_sucrose_root, "mol/g."
            continue

        # We calculate the current external surface of the element:
        n.external_surface, n.volume = surface_and_volume(n, n.radius, n.length) # Necessary????????????????????????????????????????????????????

        # We calculate the unloading coefficient according to the distance from the apex:
        # # OPTION 1 By analogy with what is done for hexose exudation according to Personeni et al. (2007):
        # n.unloading_coeff = unloading_apex / (1 + n.dist_to_tip*100) ** gamma_unloading
        # OPTION 2:
        # We consider that unloading coefficient is maximal at apices, medium when a lateral must emerge, and low everywhere else:
        if n.label == "Apex" and n.type == "Normal_root_after_emergence":
            n.unloading_coeff = unloading_apex
        elif n.lateral_emergence_possibility == "Possible":
            if n.type == "Base_of_the_root_system":
                n.unloading_coeff = unloading_apex*10.
            else:
                n.unloading_coeff = unloading_apex / 5.
        else:
            if n.type == "Dead":
                n.unloading_coeff = 0.
            elif n.type == "Stopped":
                n.unloading_coeff = unloading_apex / 100.
            else:
                n.unloading_coeff = unloading_apex / 50.

        # We calculate the potential production of hexose (in mol) according to the Michaelis-Menten function:
        n.prod_hexose = 2. * n.external_surface * n.unloading_coeff * n.C_sucrose_root \
                                / (Km_unloading + n.C_sucrose_root) * time_step_in_seconds
        # The factor 2 originates from the conversion of 1 molecule of sucrose into 2 molecules of hexose.


########################################################################################################################
########################################################################################################################
# MODULE "SUCROSE SUPPLY FROM THE SHOOTS"
########################################################################################################################
########################################################################################################################

# Calculating the net input of sucrose by the aerial parts into the root system:
# ------------------------------------------------------------------------------
def shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=1e-9, time_step_in_seconds=1. * 60. * 60. * 24., printing_warnings=False):
    """
    This function calculates the new root sucrose concentration (mol of sucrose per gram of dry root structural mass)
    AFTER the supply of sucrose from the shoot.
    """

    # The input of sucrose over this time step is calculated
    # from the sucrose transport rate provided as input of the function:
    sucrose_input = sucrose_input_rate * time_step_in_seconds

    # We calculate the remaining amount of sucrose in the root system,
    # based on the current sucrose concentration and biomass of each root element:
    total_sucrose_root, total_biomass = total_root_sucrose_and_biomass(g)
    # The total sucrose is the total amount of sucrose present in the root system.
    # The total biomass is only the living biomass.

    # We use a global variable recorded outside this function, that corresponds to the possible deficit of sucrose
    # (in moles of sucrose) of the whole root system calculated at the previous time_step:
    global global_sucrose_deficit
    # The new average sucrose concentration in the root system is calculated as:
    C_sucrose_root_after_supply = (total_sucrose_root + sucrose_input - global_sucrose_deficit) / total_biomass

    if C_sucrose_root_after_supply >=0.:
        new_C_sucrose_root = C_sucrose_root_after_supply
        # We reset the global variable global_sucrose_deficit:
        global_sucrose_deficit = 0.
    else:
        if printing_warnings:
            print "WARNING: A negative sucrose concentration was calculated. We set it to zero throughout the root system," \
                  " and recorded the deficit in a global variable for the balance at the next time step."
        # We record the general deficit in sucrose:
        global_sucrose_deficit = - C_sucrose_root_after_supply * total_biomass
        # We defined the new concentration of sucrose as 0:
        new_C_sucrose_root = 0.

    # We go through the MTG to modify the sugars concentrations:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)
        # If the element has not emerged yet, it doesn't contain any sucrose yet;
        # if is dead, it doesn't contain any sucrose anymore:
        if n.length <=0. or n.type == "Dead" or n.type == "Just_dead":
            n.C_sucrose_root = 0.
        else:
            # The local sucrose concentration in the root is calculated from the new sucrose concentration calculated above:
            n.C_sucrose_root = new_C_sucrose_root

    return g


########################################################################################################################
########################################################################################################################
# MAIN PROGRAM:
########################################################################################################################
########################################################################################################################

# BALANCE:
##########

def balance(g, printing_warnings=False):

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # n represents the vertex:
        n = g.node(vid)

        # We exclude root elements that have not emerged yet:
        if n.length <=0.:
            continue

        # We calculate the new concentration of hexose in the soil according to hexose degradation, exudation and uptake:
        n.C_hexose_soil = (n.C_hexose_soil*n.initial_biomass - n.Deficit_hexose_soil - n.hexose_degradation + n.hexose_exudation - n.hexose_uptake) / n.biomass
        n.net_hexose_exudation = n.hexose_exudation - n.hexose_uptake
        # We reset the deficit to 0:
        n.Deficit_hexose_soil=0.
        if n.C_hexose_soil <0:
            if printing_warnings:
                print "WARNING: After balance, there is a deficit of soil hexose for element", n.index(), \
                    "; the concentration has been set to 0 and the deficit will be included in the next balance."
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_hexose_soil = -n.C_hexose_soil*n.biomass
            # And we set the concentration to 0:
            n.C_hexose_soil = 0.

        # We calculate the new concentration of hexose in the root according to hexose exudation and uptake, maintenance, growth and sucrose conversion:
        n.C_hexose_root = (n.C_hexose_root*n.initial_biomass - n.Deficit_hexose_root - n.hexose_exudation + n.hexose_uptake
                           - n.resp_maintenance / 6. - n.hexose_consumption_by_growth + n.prod_hexose) / n.biomass
        # We reset the deficit to 0:
        n.Deficit_hexose_root=0.
        if n.C_hexose_root <0:
            if printing_warnings:
                print "WARNING: After balance, there is a deficit of root hexose for element", n.index() , \
                    "; the concentration has been set to 0 and the deficit will be included in the next balance."
            # We define a positive deficit (mol of hexose) based on the negative concentration:
            n.Deficit_hexose_root = - n.C_hexose_root*n.biomass
            # And we set the concentration to 0:
            n.C_hexose_root = 0.

        # We calculate the new concentration of sucrose in the root according to sucrose conversion into hexose:
        n.C_sucrose_root = (n.C_sucrose_root*n.initial_biomass - n.Deficit_sucrose_root - n.prod_hexose / 2.) / n.biomass
        # We reset the deficit to 0:
        n.Deficit_sucrose_root=0.
        if n.C_sucrose_root <0:
            # We define a positive deficit (mol of sucrose) based on the negative concentration:
            n.Deficit_sucrose_root = -n.C_sucrose_root*n.biomass
            # And we set the concentration to 0:
            if printing_warnings:
                print "WARNING: After balance, there is a deficit in root sucrose for element", n.index(), \
                    "; the concentration has been set to 0 and the deficit will be included in the next balance."
            n.C_sucrose_root = 0.


# INITIALIZATION OF THE MTG
###########################

def initiate_mtg(random=True):
    g = MTG()

    base_radius=Di/2.

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
    segment.length = segment_length
    segment.radius = base_radius
    segment.initial_length = segment_length
    segment.initial_radius = base_radius
    segment.external_surface, segment.volume = surface_and_volume(segment, segment.radius, segment.length)
    segment.dist_to_tip = 0.
    segment.dist_to_ramif = 0.
    segment.actual_elongation = segment.length
    segment.actual_elongation_rate = 0
    segment.lateral_primordium_index = 0
    segment.adventious_emerging_primordium_index = 0
    # Quantities and concentrations:
    # -------------------------------
    segment.biomass = segment.volume * density
    segment.initial_biomass = segment.biomass
    # We define the initial sugar concentrations:
    segment.C_sucrose_root = 1e-3
    segment.C_hexose_root = 1e-3
    segment.C_hexose_soil = 0.
    segment.Deficit_sucrose_root = 0.
    segment.Deficit_hexose_root = 0.
    segment.Deficit_hexose_soil = 0.
    # Fluxes:
    # --------
    segment.resp_maintenance = 0.
    segment.resp_growth = 0.
    segment.hexose_growth_demand = 0.
    segment.prod_hexose = 0.
    segment.hexose_exudation = 0.
    segment.hexose_uptake = 0.
    segment.hexose_degradation = 0.
    segment.specific_net_exudation = 0.
    # Time indications:
    # ------------------
    segment.growth_duration = GDs * (2.*base_radius)**2
    segment.life_duration = LDs * 2. * base_radius * RTD
    segment.time_since_primordium_formation = 0.
    segment.time_since_emergence = 0.
    segment.time_since_growth_stopped = 0.
    segment.time_since_death = 0.

    # If there should be more than one main root (i.e. adventious roots formed at the basis):
    if MNP > 1:
        # Then we form one supporting segment of length 0 + one primordium of adventious root:
        for i in range(1, MNP):
            # We make sure that the adventious roots will have different random insertion angles:
            np.random.seed(random_choice + i)
            # We add one new segment without any length on the same axis as the base:
            segment = segment.add_child(edge_type='<',
                                        # Characteristics:
                                        # ----------------
                                        label='Segment',
                                        type="Support_for_adventious_root",
                                        # Authorizations and C requirements:
                                        # ----------------------------------
                                        lateral_emergence_possibility='Impossible',
                                        emergence_cost=0.,
                                        # Geometry and topology:
                                        # ----------------------
                                        angle_down=0,
                                        angle_roll = abs(np.random.normal(90, 180)),
                                        # The length of the primordium is set to 0:
                                        length=0.,
                                        radius=Di/2.,
                                        potential_length=0.,
                                        potential_radius=0.,
                                        initial_length=0.,
                                        initial_radius=Di/2.,
                                        external_surface=0.,
                                        volume=0.,
                                        dist_to_ramif=0.,
                                        actual_elongation=0.,
                                        adventious_emerging_primordium_index=0,
                                        # Quantities and concentrations:
                                        # ------------------------------
                                        biomass=0.,
                                        initial_biomass=0.,
                                        C_hexose_root=0.,
                                        C_hexose_soil=0.,
                                        C_sucrose_root=0.,
                                        Deficit_hexose_root=0.,
                                        Deficit_hexose_soil=0.,
                                        Deficit_sucrose_root=0.,
                                        # Fluxes:
                                        # -------
                                        resp_maintenance=0.,
                                        resp_growth=0.,
                                        hexose_growth_demand=0.,
                                        prod_hexose=0.,
                                        hexose_exudation=0.,
                                        hexose_uptake=0.,
                                        hexose_degradation=0.,
                                        specific_net_exudation=0.,
                                        # Time indications:
                                        # -----------------
                                        growth_duration=GDs * (2.*base_radius)**2,
                                        life_duration=LDs * 2. * base_radius * RTD,
                                        time_since_primordium_formation=0.,
                                        time_since_emergence=0.,
                                        potential_time_since_emergence=0.,
                                        time_since_growth_stopped=0.,
                                        time_since_death=0.
                                        )
            # We define the radius of an adventious root according to the parameter Di:
            if random:
                radius_adventious = abs(np.random.normal(Di/2. * 0.8, Di/2. * 0.8 * CVDD))
                # if radius_adventious > Di:
                #     radius_adventious=Di
            else:
                radius_adventious = Di/2.*0.8
            # And we add one new primordium of adventious root on the previously defined segment:
            apex_adventious = segment.add_child(edge_type='+',
                                                # Characteristics:
                                                # ----------------
                                                label='Apex',
                                                type="Adventious_root_before_emergence",
                                                # Authorizations and C requirements:
                                                # ----------------------------------
                                                lateral_emergence_possibility='Impossible',
                                                emergence_cost=0.,
                                                # Geometry and topology:
                                                # ----------------------
                                                angle_down=abs(np.random.normal(50,10)),
                                                angle_roll=5,
                                                # The length of the primordium is set to 0:
                                                length=0.,
                                                radius=radius_adventious,
                                                potential_length=0.,
                                                potential_radius=0.,
                                                initial_length=0.,
                                                initial_radius=radius_adventious,
                                                external_surface=0.,
                                                volume=0.,
                                                dist_to_ramif=0.,
                                                actual_elongation=0.,
                                                adventious_emerging_primordium_index=0,
                                                # Quantities and concentrations:
                                                # ------------------------------
                                                biomass=0.,
                                                initial_biomass=0.,
                                                C_hexose_root=0.,
                                                C_hexose_soil=0.,
                                                C_sucrose_root=0.,
                                                Deficit_hexose_root=0.,
                                                Deficit_hexose_soil=0.,
                                                Deficit_sucrose_root=0.,
                                                # Fluxes:
                                                # -------
                                                resp_maintenance=0.,
                                                resp_growth=0.,
                                                hexose_growth_demand=0.,
                                                prod_hexose=0.,
                                                hexose_exudation=0.,
                                                hexose_uptake=0.,
                                                hexose_degradation=0.,
                                                specific_net_exudation=0.,
                                                # Time indications:
                                                # -----------------
                                                growth_duration=GDs * radius_adventious * radius_adventious * 4,
                                                life_duration=LDs * 2. * radius_adventious * RTD,
                                                time_since_primordium_formation=0.,
                                                time_since_emergence=0.,
                                                potential_time_since_emergence=0.,
                                                time_since_growth_stopped=0.,
                                                time_since_death=0.
                                                )

    # Finally, we add the apex that is going to develop the main axis:
    apex = segment.add_child(edge_type='<',
                             # Characteristics:
                             # ----------------
                             label='Apex',
                             type="Normal_root_after_emergence",
                             # Authorizations and C requirements:
                             # ----------------------------------
                             lateral_emergence_possibility='Impossible',
                             emergence_cost=0.,
                             # Geometry and topology:
                             # ----------------------
                             angle_down=0,
                             angle_roll=85,
                             # The length of the primordium is set to half of the segment length:
                             length=segment_length/2.,
                             radius=Di/2.,
                             potential_length=0.,
                             potential_radius=0.,
                             initial_length=0.,
                             initial_radius=Di/2.,
                             external_surface=0.,
                             volume=0.,
                             dist_to_ramif=0.,
                             actual_elongation=0.,
                             lateral_primordium_index=0,
                             # Quantities and concentrations:
                             # ------------------------------
                             biomass=0.,
                             initial_biomass=0.,
                             C_hexose_root=0.,
                             C_hexose_soil=0.,
                             C_sucrose_root=0.,
                             Deficit_hexose_root=0.,
                             Deficit_hexose_soil=0.,
                             Deficit_sucrose_root=0.,
                             # Fluxes:
                             # -------
                             resp_maintenance=0.,
                             resp_growth=0.,
                             hexose_growth_demand=0.,
                             prod_hexose=0.,
                             hexose_exudation=0.,
                             hexose_uptake=0.,
                             hexose_degradation=0.,
                             specific_net_exudation=0.,
                             # Time indications:
                             # -----------------
                             growth_duration = GDs * (2.*base_radius)**2,
                             life_duration = LDs * 2. * base_radius * RTD,
                             time_since_primordium_formation=0.,
                             time_since_emergence=0.,
                             potential_time_since_emergence=0.,
                             time_since_growth_stopped=0.,
                             time_since_death=0.
                             )
    apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
    apex.biomass = apex.volume * density
    apex.initial_biomass = apex.biomass
    apex.C_sucrose_root = 1e-3
    apex.C_hexose_root = 1e-3
    apex.C_hexose_soil = 0.
    return g


# SIMULATION OVER TIME:
#######################

# We read the data showing the unloading rate of sucrose as a function of time from a file:
# ------------------------------------------------------------------------------------------
# We first define the path and the file to read as a .csv:
PATH = os.path.join('C:/', 'Users', 'frees', 'rhizodep', 'test', 'organs_states.csv')
# Then we read the file and copy it in a dataframe "df":
df = pd.read_csv(PATH, sep=',')
# We only keep the two columns of interest:
df = df[['t', 'Unloading_Sucrose']]
# We remove any "NA" in the column of interest:
df = df[pd.notnull(df['Unloading_Sucrose'])]
# We remove any values below 0:
sucrose_input_frame = df[(df.Unloading_Sucrose >= 0.)]
# We reset the indices after having filtered the data:
sucrose_input_frame = sucrose_input_frame.reset_index(drop=True)

# We define the main simulation program:
def main_simulation(g, simulation_period_in_days=120., time_step_in_days=1.0/100., radial_growth="Impossible", Archisimple=False,
                    property="C_hexose_root", vmin=1e-6, vmax=1e-0, log_scale=True,
                    x_center=0, y_center=0, z_center=-1, z_cam=-1,
                    camera_distance=10., step_back_coefficient=0., camera_rotation=False, n_rotation_points = 24*5,
                    recording_images=True,
                    printing_sum=False,
                    recording_sum=False,
                    printing_warnings=False,
                    recording_g=True,
                    recording_g_properties=True,
                    random=True):

    # We convert the time step in seconds:
    time_step_in_seconds = time_step_in_days * 60. * 60. * 24.
    # We calculate the number of steps necessary to reach the end of the simulation period:
    if simulation_period_in_days==0. or time_step_in_days==0.:
        print "WATCH OUT: No simulation was done, as time input was 0."
        n_steps=0
    else:
        n_steps = trunc(simulation_period_in_days / time_step_in_days) + 1

    # We call global variables:
    global time_since_last_adventious_root_emergence
    global adventious_root_emergence

    # We initialize empty variables at t=0:
    step=0
    time = 0.
    total_biomass = 0.
    cumulated_hexose_exudation = 0.
    cumulated_respired_CO2 = 0.
    cumulated_biomass_production = 0.
    sucrose_input_rate =0.
    C_cumulated_in_the_gazeous_phase=0.

    # We initialize empty lists for recording the macro-results:
    time_in_days_series=[]
    sucrose_input_series=[]
    total_living_root_length_series=[]
    total_dead_root_length_series=[]
    total_living_root_surface_series=[]
    total_dead_root_surface_series=[]
    total_living_root_biomass_series=[]
    total_dead_root_biomass_series=[]
    total_sucrose_root_series=[]
    total_hexose_root_series=[]
    total_hexose_soil_series=[]
    total_CO2_series=[]
    total_CO2_root_growth_series=[]
    total_CO2_root_maintenance_series=[]
    total_prod_hexose_series=[]
    total_hexose_exudation_series=[]
    total_hexose_uptake_series=[]
    total_hexose_degradation_series=[]
    total_net_hexose_exudation_series=[]
    C_in_the_root_soil_system_series=[]
    C_cumulated_in_the_gazeous_phase_series=[]

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

    # If the rotation of the camera is allowed:
    if camera_rotation:
        # We calculate the coordinates of the camera on the circle around the center:
        x_coordinates, y_coordinates, z_coordinates = circle_coordinates(z_center=z_cam, radius=camera_distance, n_points=n_rotation_points)
        # We initialize the index for reading each coordinates:
        index_camera=0
    else:
        x_camera=camera_distance
        x_cam=camera_distance
        z_camera=z_cam

    #RECORDING THE INITIAL STATE OF THE MTG:
    #---------------------------------------

    # For recording the graph at each time step to make a video later:
    # -----------------------------------------------------------------
    if recording_images:
        image_name = os.path.join(video_dir, 'root%.4d.png')
        pgl.Viewer.saveSnapshot(image_name % step)

    # For recording the MTG at each time step to load it later on:
    # ------------------------------------------------------------
    if recording_g:
        g_file_name = os.path.join(g_dir, 'root%.4d.pckl')
        with open(g_file_name % step, 'wb') as output:
            pickle.dump(g, output, protocol=2)

    # For recording the properties of g in a csv file:
    # --------------------------------------------------
    if recording_g_properties:
        prop_file_name = os.path.join(prop_dir, 'root%.4d.csv')
        recording_MTG_properties(g, file_name=prop_file_name % step)

    # SUMMING AND PRINTING VARIABLES ON THE ROOT SYSTEM:
    # --------------------------------------------------
    if printing_sum:
        dictionnary = summing(g,
                              printing_total_length=True,
                              printing_total_biomass=True,
                              printing_all=True)
    elif not printing_sum and recording_sum:
        dictionnary = summing(g,
                              printing_total_length=True,
                              printing_total_biomass=True,
                              printing_all=False)
    if recording_sum:
        time_in_days_series.append(time_step_in_days * step)
        sucrose_input_series.append(sucrose_input_rate * time_step_in_seconds)
        total_living_root_length_series.append(dictionnary["total_living_root_length"])
        total_dead_root_length_series.append(dictionnary["total_dead_root_length"])
        total_living_root_biomass_series.append(dictionnary["total_living_root_biomass"])
        total_dead_root_biomass_series.append(dictionnary["total_dead_root_biomass"])
        total_living_root_surface_series.append(dictionnary["total_living_root_surface"])
        total_dead_root_surface_series.append(dictionnary["total_dead_root_surface"])
        total_sucrose_root_series.append(dictionnary["total_sucrose_root"])
        total_hexose_root_series.append(dictionnary["total_hexose_root"])
        total_hexose_soil_series.append(dictionnary["total_hexose_soil"])
        total_CO2_series.append(dictionnary["total_CO2"])
        total_CO2_root_growth_series.append(dictionnary["total_CO2_root_growth"])
        total_CO2_root_maintenance_series.append(dictionnary["total_CO2_root_maintenance"])
        total_prod_hexose_series.append(dictionnary["total_prod_hexose"])
        total_hexose_exudation_series.append(dictionnary["total_hexose_exudation"])
        total_hexose_uptake_series.append(dictionnary["total_hexose_uptake"])
        total_hexose_degradation_series.append(dictionnary["total_hexose_degradation"])
        total_net_hexose_exudation_series.append(dictionnary["total_net_hexose_exudation"])
        C_in_the_root_soil_system_series.append(dictionnary["C_in_the_root_soil_system"])
        C_cumulated_in_the_gazeous_phase += dictionnary["C_emitted_towards_the_gazeous_phase"]
        C_cumulated_in_the_gazeous_phase_series.append(C_cumulated_in_the_gazeous_phase)

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # The code will try to run the following code until it is finished or an error has been raised:
    try:
        # An iteration is done for each time step:
        for step in range(1, n_steps):

            # At the beginning of the time step, we reset the global variable allowing the emergence of adventious roots:
            adventious_root_emergence = "Possible"
            # We keep in memory the value of the global variable time_since_adventious_root_emergence at the beginning of the time steo:
            initial_time_since_adventious_root_emergence = time_since_last_adventious_root_emergence

            # We calculate the current time in hours:
            current_time_in_hours = step * time_step_in_days * 24.

            # DEFINING THE INPUT OF CARBON TO THE ROOTS:
            #-------------------------------------------

            # We look for the first item in the time series of sucrose_input_frame for which time is higher
            # than the current time (a default value of 0 is given otherwise):
            considered_time = next((time for time in sucrose_input_frame.t if time > current_time_in_hours), 0)
            # If the current time in the loop is higher than any time indicated in the dataframe:
            if considered_time > 0:
            #     # Then we use the last value of sucrose unloading rate as the sucrose input rate:
            #     sucrose_input_rate = sucrose_input_frame.Unloading_Sucrose[-1]
            # else:
                # Otherwise, we get the index of the considered time in the list of times from the dataframe:
                time_series=list(sucrose_input_frame.t)
                index=time_series.index(considered_time)
                # If the considered time corresponds to the first item in the list of times from the dataframe:
                if index==0:
                    # Then we use the first item in the list of unloading rates:
                    sucrose_input_rate = sucrose_input_frame.Unloading_Sucrose[index]
                else:
                    # Otherwise, we use a linear function between this considered time and the preceding one,
                    # so that we can calculate a plausible value of sucrose unloading rate at the exact time of the loop:
                    x1 = sucrose_input_frame.t[index-1]
                    x2 = sucrose_input_frame.t[index]
                    y1 = sucrose_input_frame.Unloading_Sucrose[index - 1]
                    y2 = sucrose_input_frame.Unloading_Sucrose[index]
                    a = (y2 - y1)/float(x2 - x1)
                    b = y1 - a*x1
                    sucrose_input_rate = a*current_time_in_hours + b
            # Conversion of sucrose unloading rate from umol_C per hour in mol_sucrose per second:
            sucrose_input_rate = sucrose_input_rate/12.*1e-6/(60.*60.)

            # ALTERNATIVE: WE SET THE SUCROSE SUPPLY SO THAT IT IS NOT AFFECTING THE RESULT:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            sucrose_input_rate = 1e-6
            # sucrose_input_rate = 0.

            print ""
            print "From t =", "{:.2f}".format(Decimal((step-1) * time_step_in_days)), "days to t =", \
                "{:.2f}".format(Decimal(step * time_step_in_days)), "days:"
            print "------------------------------------"
            print "The input rate of sucrose to the root for time=", current_time_in_hours, "h is", \
                "{:.2E}".format(Decimal(sucrose_input_rate)), "mol of sucrose per second, i.e.", \
                "{:.2E}".format(Decimal(sucrose_input_rate*60.*60.*24.)), "mol of sucrose per day."

            print "The root system initially includes", len(g)-1, "root elements."

            # CASE 1: WE REPRODUCE THE GROWTH WITHOUT CONSIDERATIONS OF LOCAL CONCENTRATIONS
            #-------------------------------------------------------------------------------

            if Archisimple:

                # The input of C (gram of C) from shoots is calculated from the input of sucrose:
                C_input = sucrose_input_rate*12*12.01*time_step_in_seconds
                # We assume that only a fraction of this C_input will be used for producing biomass:
                fraction=0.20
                biomass_input = C_input/biomass_C_content*fraction

                # We calculate the potential growth, already based on ArchiSimple rules:
                potential_growth(g, time_step_in_seconds=time_step_in_seconds,
                                 radial_growth=radial_growth,
                                 Archisimple=True, printing_warnings=printing_warnings)

                # We use the function archisimple_growth to adapt the potential growth to the available biomass:
                SC = satisfaction_coefficient(g,biomass_input=biomass_input)
                archisimple_growth(g, SC, time_step_in_seconds, printing_warnings=printing_warnings)

                # We proceed to the segmentation of the whole root system (NOTE: segmentation should always occur AFTER actual growth):
                segmentation_and_primordia_formation(g, time_step_in_seconds, printing_warnings=printing_warnings, random=random)

            else:

            # CASE 2: WE PERFORM THE COMPLETE MODEL WITH C BALANCE IN EACH ROOT ELEMENT
            #--------------------------------------------------------------------------

                # Calculation of potential growth without consideration of available hexose:
                potential_growth(g, time_step_in_seconds=time_step_in_seconds, radial_growth=radial_growth, Archisimple=False)

                # Calculation of actual growth based on the hexose remaining in the roots,
                # and corresponding consumption of hexose in the root:
                g = actual_growth_and_corresponding_respiration(g, time_step_in_seconds=time_step_in_seconds, printing_warnings=printing_warnings)
                # We proceed to the segmentation of the whole root system (NOTE: segmentation should always occur AFTER actual growth):
                segmentation_and_primordia_formation(g, time_step_in_seconds, random=random)
                dist_to_tip(g)

                # Consumption of hexose in the soil:
                soil_hexose_degradation(g, time_step_in_seconds=time_step_in_seconds, printing_warnings=printing_warnings)

                # Transfer of hexose from the root to the soil, consumption of hexose inside the roots:
                root_hexose_exudation(g, time_step_in_seconds=time_step_in_seconds, printing_warnings=printing_warnings)
                # Transfer of hexose from the soil to the root, consumption of hexose in the soil:
                root_hexose_uptake(g, time_step_in_seconds=time_step_in_seconds, printing_warnings=printing_warnings)

                # Consumption of hexose in the root by maintenance respiration:
                maintenance_respiration(g, time_step_in_seconds=time_step_in_seconds, printing_warnings=printing_warnings)

                # Unloading of sucrose from phloem and conversion of sucrose into hexose:
                sucrose_to_hexose(g, time_step_in_seconds=time_step_in_seconds, printing_warnings=printing_warnings)

                # Calculation of the new concentrations in hexose and sucrose once all the processes have been done:
                balance(g, printing_warnings=printing_warnings)

                # Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
                shoot_sucrose_supply_and_spreading(g, sucrose_input_rate=sucrose_input_rate,
                                                   time_step_in_seconds=time_step_in_seconds,
                                                   printing_warnings=printing_warnings)
                # WARNING: The function "shoot_sucrose_supply_and_spreading" must be called AFTER the function "balance",
                # otherwise the deficit in sucrose may be counted twice!!!

            # A the end of the time step, if the global variable "time_since_adventious_root_emergence" has been unchanged:
            if time_since_last_adventious_root_emergence == initial_time_since_adventious_root_emergence:
                #Then we increment it by the time step:
                time_since_last_adventious_root_emergence += time_step_in_seconds
            # Otherwise, the variable has already been reset when the emergence of one adventious root has been allowed.

            # PLOTTING THE MTG:
            #------------------

            # If the rotation of the camera around the root system is required:
            if camera_rotation:
                x_cam = x_coordinates[index_camera]
                y_cam = y_coordinates[index_camera]
                z_cam = z_coordinates[index_camera]
                sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax,
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
                    index_camera=0
            # Otherwise, the camera will stay on a fixed position:
            else:

                sc = plot_mtg(g, prop_cmap=property, lognorm=log_scale, vmin=vmin, vmax=vmax,
                              x_center=x_center,
                              y_center=y_center,
                              z_center=z_center,
                              x_cam=x_camera,
                              y_cam=0,
                              z_cam=z_camera)
                # We move the camera further from the root system:
                x_camera = x_cam + x_cam*step_back_coefficient*step
                z_camera = z_cam + z_cam*step_back_coefficient*step
            # We finally display the MTG on PlantGL:
            pgl.Viewer.display(sc)

            # For recording the graph at each time step to make a video later:
            # -----------------------------------------------------------------
            if recording_images:
                image_name = os.path.join(video_dir, 'root%.4d.png')
                pgl.Viewer.saveSnapshot(image_name % step)

            # For recording the MTG at each time step to load it later on:
            # ------------------------------------------------------------
            if recording_g:
                g_file_name = os.path.join(g_dir, 'root%.4d.pckl')
                with open(g_file_name % step, 'wb') as output:
                    pickle.dump(g, output, protocol=2)

            # For recording the properties of g in a csv file:
            #--------------------------------------------------
            if recording_g_properties:
                prop_file_name = os.path.join(prop_dir, 'root%.4d.csv')
                recording_MTG_properties(g, file_name = prop_file_name % step)

            # SUMMING AND PRINTING VARIABLES ON THE ROOT SYSTEM:
            # --------------------------------------------------
            if printing_sum:
                dictionnary = summing(g,
                                      printing_total_length=True,
                                      printing_total_biomass=True,
                                      printing_all=True)
            elif not printing_sum and recording_sum:
                dictionnary = summing(g,
                                      printing_total_length=True,
                                      printing_total_biomass=True,
                                      printing_all = False)
            if recording_sum:
                time_in_days_series.append(time_step_in_days*step)
                sucrose_input_series.append(sucrose_input_rate*time_step_in_seconds)
                total_living_root_length_series.append(dictionnary["total_living_root_length"])
                total_dead_root_length_series.append(dictionnary["total_dead_root_length"])
                total_living_root_biomass_series.append(dictionnary["total_living_root_biomass"])
                total_dead_root_biomass_series.append(dictionnary["total_dead_root_biomass"])
                total_living_root_surface_series.append(dictionnary["total_living_root_surface"])
                total_dead_root_surface_series.append(dictionnary["total_dead_root_surface"])
                total_sucrose_root_series.append(dictionnary["total_sucrose_root"])
                total_hexose_root_series.append(dictionnary["total_hexose_root"])
                total_hexose_soil_series.append(dictionnary["total_hexose_soil"])
                total_CO2_series.append(dictionnary["total_CO2"])
                total_CO2_root_growth_series.append(dictionnary["total_CO2_root_growth"])
                total_CO2_root_maintenance_series.append(dictionnary["total_CO2_root_maintenance"])
                total_prod_hexose_series.append(dictionnary["total_prod_hexose"])
                total_hexose_exudation_series.append(dictionnary["total_hexose_exudation"])
                total_hexose_uptake_series.append(dictionnary["total_hexose_uptake"])
                total_hexose_degradation_series.append(dictionnary["total_hexose_degradation"])
                total_net_hexose_exudation_series.append(dictionnary["total_net_hexose_exudation"])
                C_in_the_root_soil_system_series.append(dictionnary["C_in_the_root_soil_system"])
                C_cumulated_in_the_gazeous_phase += dictionnary["C_emitted_towards_the_gazeous_phase"]
                C_cumulated_in_the_gazeous_phase_series.append(C_cumulated_in_the_gazeous_phase)

            print "The root system finally includes", len(g)-1, "root elements."

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # At the end of the simulation (or just before an error is about to interrupt the program!):
    #-------------------------------------------------------------------------------------------
    finally:
        print ""
        print "The program has stopped at time t = {:.2f}".format(Decimal(step * time_step_in_days)), "days."
        # We can record all the results in a CSV file:
        if recording_sum:
            # We create a data_frame from the vectors generated in the main program up to this point:
            data_frame = pd.DataFrame({"Time (days)": time_in_days_series,
                                       "Sucrose input (mol of sucrose)": sucrose_input_series,
                                       "Root biomass (g)": total_living_root_biomass_series,
                                       "Root necromass (g)": total_dead_root_biomass_series,
                                       "Root length (m)": total_living_root_length_series,
                                       "Root surface (m2)": total_living_root_surface_series,
                                       "Sucrose in the root (mol of sucrose)": total_sucrose_root_series,
                                       "Hexose in the root (mol of hexose)": total_hexose_root_series,
                                       "Hexose in the soil (mol of hexose)": total_hexose_soil_series,
                                       "CO2 originating from root growth (mol of C)": total_CO2_root_growth_series,
                                       "CO2 originating from root maintenance (mol of C)": total_CO2_root_maintenance_series,
                                       "Hexose produced in the root (mol of hexose)": total_prod_hexose_series,
                                       "Hexose exudated (mol of hexose)": total_hexose_exudation_series,
                                       "Hexose taken up (mol of hexose)": total_hexose_uptake_series,
                                       "Hexose degraded (mol of hexose)": total_hexose_degradation_series,
                                       "Total amount of C present in the root-soil system (mol of C)": C_in_the_root_soil_system_series,
                                       "Total amount of C cumulated in the gazeous phase (mol of C)": C_cumulated_in_the_gazeous_phase_series
                                       },
                                      columns=["Time (days)",
                                               "Sucrose input (mol of sucrose)",
                                               "Total amount of C present in the root-soil system (mol of C)",
                                               "Total amount of C cumulated in the gazeous phase (mol of C)",
                                               "Root biomass (g)",
                                               "Root necromass (g)",
                                               "Root length (m)",
                                               "Root surface (m2)",
                                               "Sucrose in the root (mol of sucrose)",
                                               "Hexose in the root (mol of hexose)",
                                               "Hexose in the soil (mol of hexose)",
                                               "CO2 originating from root growth (mol of C)",
                                               "CO2 originating from root maintenance (mol of C)"])
            # We save the data_frame in a CSV file:
            data_frame.to_csv('simulation_results.csv', na_rep='NA', index=False, header=True)


# RUNNING THE SIMULATION:
#########################

# We set the working directory:
os.chdir('C:\\Users\\frees\\rhizodep\\test')
print "The current directory is:", os.getcwd()

# We record the time when the run starts:
start_time = timeit.default_timer()

# We initiate the properties of the MTG "g":
g = initiate_mtg(random=True)
# We initiate the time variable that will be used to determine the emergence of adventious roots:
time_since_last_adventious_root_emergence = 0.
# We initiate the global variable that corresponds to a possible general deficit in sucrose of the whole root system:
global_sucrose_deficit=0.
# We launch the main simulation program:
main_simulation(g, simulation_period_in_days=50, time_step_in_days=1, radial_growth="Possible", Archisimple=False,
                property="net_hexose_exudation", vmin=1e-10, vmax=1e-5, log_scale=True,
                x_center=0, y_center=0, z_center=-2, z_cam=-5,
                camera_distance = 5, step_back_coefficient=0., camera_rotation=False, n_rotation_points=12*10,
                recording_images=False,
                printing_sum=True,
                recording_sum=True,
                printing_warnings=False,
                recording_g=False,
                recording_g_properties=False,
                random=True)

print ""
print "***************************************************************"
end_time = timeit.default_timer()
print "Run is done! The system took", round(end_time - start_time, 1), "seconds to complete the run."

# We save the final MTG:
with open('g_file.pckl', 'wb') as output:
    pickle.dump(g, output, protocol=2)

print "The whole root system has been saved in the file 'g_file.pckl'."

# To avoid closing PlantGL as soon as the run is done:
raw_input()