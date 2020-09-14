# Todo: Watch the calculation of surface and volume for the apices - if they correspond to cones, the mass balance for segmentation may not be correct!
# Todo: Check why changing the time step significantly changes the results!
# Todo: Impose a delay describing how long the parent segment must sustain the growth of the child

# Importation of functions from the system:
###########################################

from math import sqrt, pi, trunc
from decimal import Decimal
import time
import numpy as np
import pandas as pd
import os

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl

########################################################################################################################
########################################################################################################################
# LIST OF PARAMETERS
########################################################################################################################
########################################################################################################################

# Parameters for root growth:
#----------------------------
# Maximal number of adventitious roots (including primary)(dimensionless):
MNP = 1
# Emission rate of adventious roots (in s-1):
# ER = 0.5 day-1
ER = 0.5 / (60.*60.*24.)
# Tip diameter of the emitted root(in s) ():
# Di=0.5 mm
Di = 0.5 / 1000.
# Slope of the potential elongation rate versus tip diameter (in m m-1 s-1):
# EL = 5 mm mm-1 day-1
EL = 5. / (60.*60.*24.)
# Threshold tip diameter below which there is no possible elongation (diameter of the finest roots)(in m):
# Dmin=0.05 mm
Dmin = 0.05 / 1000.
# Coefficient of growth duration (in s m-2):
# GDs=400. day mm-2
GDs = 400. * (60.*60.*24.) * 1000.**2.
# Gravitropism (dimensionless):
G = 1.
# Delay of emergence of the primordium (in s):
# emergence_delay = 3. days
emergence_delay = 3. * (60.*60.*24.)
# Inter-primordium distance (in m):
# IPD = 7.6 mm
IPD = 7.6 / 1000.
# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
RMD=0.3
# Relative variation of the daughter root diameter (dimensionless):
CVDD=0.15
# Proportionality coefficient between section area of the segment and the sum of distal section areas (dimensionless):
SGC=1.
# Coefficient of the life duration (in s m g-1):
# LDs = 5000. day mm g-1
LDs = 5000. * (60.*60.*24.) / 1000.
# Root tissue density (in g m-3):
# RTD=0.1 g cm-3
RTD = 0.1 * 1000000
density=RTD
# Length of a segment (in m):
segment_length = 3. / 1000.
# C content of biomass (mol of C per g of biomass):
biomass_C_content=0.44/12. # We assume that the biomass contains 44% of C.

# Parameters for growth respiration:
#-----------------------------------
# Growth yield (in mol of CO2 per mol of C used for biomass):
yield_growth = 0.8
# => Explanation: We use the value proposed by Thornley and Cannell (2000)

# Parameters for maintenance respiration:
#----------------------------------------
# Maximal maintenance respiration (in mol of CO2 per g of biomass per s):
resp_maintenance_max = 4.1e-6/20/12.01*0.44
# => Explanation: According to Barillot et al. (2016): km_max = 4.1e-6 umol_C umol_N-1 s-1,
# i.e. 4.1e-6/20*0.44 mol_C g-1 s-1 assuming a C:N molar ratio of root biomass of 20 and a C content of biomass of 44%
# Affinity constant for maintenance respiration (in mol of hexose per g of biomass):
Km_maintenance = 1.67e-3/6.
# => Explanation: We use the value of 1.67e-3 mol_C per g proposed by Barillot et al. (2016)

# Parameters for root hexose exudation:
#--------------------------------------
# Expected exudation (in mol of hexose per m2 per s):
expected_exudation_efflux = 5.2/12.01/6.*1e-6*100.**2./3600.
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour
# Expected sucrose concentration in root (in mol of sucrose per g of root):
expected_C_sucrose_root = 0.0025
# => Explanation: This is a plausible value according to the results of Gauthier (2019, pers. communication)
# Expected hexose concentration in root (in mol of hexose per g of root):
expected_C_hexose_root = 20./12.01/6.*1e-3
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour
# Expected hexose concentration in soil (in mol of hexose per g of root):
expected_C_hexose_soil = expected_C_hexose_root/100.
# => Explanation: We expect the soil concentration to be 2 orders of magnitude lower than the root concentration
# Permeability coefficient (in g of biomass per m2 per s):
Pmax_apex = expected_exudation_efflux /(expected_C_hexose_root-expected_C_hexose_soil)/100.
# => Explanation: We calculate the permeability coefficient according to the expected flux and hexose concentrations.
# Coefficient affecting the decrease of permeability with distance from the apex (adimensional):
gamma_exudation = 0.4

# Parameters for root hexose uptake from soil:
#---------------------------------------------
# Maximum influx of hexose from soil to roots (in mol of hexose per m2 per s):
Imax = expected_exudation_efflux*0.1
# Affinity constant for hexose uptake (in mol of hexose per g of biomass):
Km_influx = expected_C_hexose_root*10.
# => Explanation: We assume that half of the max influx is reached when the soil concentration equals 10 times the expected root concentration

# Parameters for soil hexose degradation:
#----------------------------------------
# Maximum degradation rate of hexose in soil (in mol of hexose per m2 per s):
degradation_max = Imax/2.
# => Explanation: We assume that the maximum degradation rate is 2 times lower than the maximum uptake rate
# Affinity constant for soil hexose degradation (in mol of hexose per g of biomass):
Km_degradation = Km_influx/50.
# => Explanation: We assume that half of the maximum degradation rate can be reached 50 times sooner than half of the maximum uptake rate

# Parameters for sucrose unloading:
#----------------------------------
# Maximum unloading rate of sucrose from the phloem (in mol of sucrose per m2 per s):
unloading_apex = 5e-6
# => Explanation: According to Barillot et al. (2016b), this value is 0.03 umol C g-1 s-1
# Coefficient affecting the decrease of unloading rate with distance from the apex (adimensional):
gamma_unloading=5.
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
    def root_visitor(g, v, turtle):

        # Angle of branching for roots of order 1:
        angle_1=random.randint(80,100)
        # Angle of branching for roots of order 2:
        angle_2=random.randint(35,55)
        # Angle of branching for roots of higher orders:
        angle_sup=random.randint(60,80)
        angles = [angle_1, angle_2] + [angle_sup] * 20

        n = g.node(v)
        # For displaying the radius 5 times larger than in reality:
        radius = n.radius*500
        order = g.order(v)
        length = n.length*100

        # For a lateral root of a given order:
        if g.edge_type(v) == '+':
            # The branching angle is selected from a list of angles:
            angle = angles[order]
            turtle.down(angle)
        elif g.edge_type(v) == "<":
            #Otherwise the angle is very slightly modified:
            angle = random.randint(0,3)
            turtle.down(angle)

        turtle.setId(v)
        turtle.setWidth(radius)
        # For each child of a root segment:
        for c in n.children():
            # If the child is a lateral root:
            if c.edge_type() == '+':
                # We set a random radial angle:
                angle_roll=random.randint(110, 130)
                turtle.rollL(angle_roll)
            elif c.edge_type == "<":
                # Otherwise the angle is very slightly modified:
                angle_roll = random.randint(0, 20)
                turtle.rollL(angle_roll)

        turtle.F(length)

    return root_visitor

def plot_mtg(g, prop_cmap='radius', cmap='jet', lognorm=False):

    visitor = get_root_visitor()

    turtle = turt.PglTurtle()
    turtle.down(180)
    scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)

    # Compute color from radius
    color.colormap(g,prop_cmap, cmap=cmap, lognorm=lognorm)

    shapes = dict((sh.getId(),sh) for sh in scene)

    colors = g.property('color')
    for vid in colors:
        if vid in shapes:
            shapes[vid].appearance = pgl.Material(colors[vid])
    scene = pgl.Scene(shapes.values())
    return scene

# FUNCTIONS FOR CALCULATING PROPERTIES ON THE MTG
#################################################

# Defining the surface of each root element in contact with the soil:
# -------------------------------------------------------------------
def surface_and_volume(element, radius, length):

    """
    The function "surface_and_volume" computes the "external_surface" (m2) and "volume" (m3) of a root element,
    based on the properties radius (m) and length (m).
    If the root element is an apex, the external surface is defined as the surface of a cone of height = "length"
    and radius = "radius", and the volume is the volume of this cone.
    If the root element is a root segment, the external surface is defined as the lateral surface of a cylinder
    of height = "length" and radius = "radius", and the volume corresponds to the volume of this cylinder.
    For each branching on the root segment, the section of the daughter root is subtracted from the cylinder surface
    of the root segment of the mother root.
    """

    n = element
    vid=n.index()
    number_of_children = n.nb_children()


    # WATCH OUT! THE VOLUME OF AN APEX IS THE ONE OF A CYLINDER FOR NOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # If there is no children to the root element:
    if number_of_children==0:
        # Then the root element corresponds to an apex considered as a cone
        # of height = "length" and radius = "radius":
        # external_surface = pi * radius*sqrt(radius**2 + length**2)
        # volume = pi * radius**2 * length / 3
        external_surface = 2 * pi * radius * length
        volume = pi * radius ** 2 * length

    # If there is only one child to the root element:
    elif number_of_children==1:
        # Then the root element is a root segment without branching, considered as a cylinder
        # of height = "length" and radius = "radius":
        external_surface = 2 * pi * radius * length
        volume = pi * radius ** 2 * length
    # If there is one or more lateral roots branched on the root segment:
    else:
        # Then we sum all the sections of the lateral roots branched on the root segment:
        sum_ramif_sections=0
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
        # If the vertex corresponds to a root apex ("leaf" for a MTG):
        #
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #  WATCH OUT: 'if g.is_leaf(vid):' doesn't work - so an apex is not always a leaf in the model? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        vertex=g.node(vid)
        if vertex.label=="Apex":
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

#Calculation of the total amount of sucrose and structural biomass in the root system:
# ------------------------------------------------------------------------------------
def total_root_sucrose_and_biomass(g):

    """
    This function returns two numeric values:
    i) the total amount of sucrose of the root system (total_sucrose_root, in mol of sucrose),
    ii) the total dry biomass of the root system (total_biomass, in g of dry structural biomass).
    """

    # We initialize the values to 0:
    total_sucrose_root = 0.
    total_biomass = 0.

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # We calculate the total amount of sucrose by summing all the local products of concentrations with biomass:
        total_sucrose_root += (n.C_sucrose_root * n.biomass)
        # We calculate the total biomass by summing all the local biomasses:
        total_biomass += n.biomass
    # We return a list of two numeric values:
    return total_sucrose_root, total_biomass

#Calculation of the total amount of sucrose and structural biomass in the root system:
# ------------------------------------------------------------------------------------
def total_print(g):

    """
    This function returns two numeric values:
    i) the total amount of sucrose of the root system (total_sucrose_root, in mol of sucrose),
    ii) the total dry biomass of the root system (total_biomass, in g of dry structural biomass).
    """

    # We initialize the values to 0:
    total_length = 0.
    total_biomass = 0.
    total_sucrose_root = 0.
    total_hexose_root = 0.
    total_hexose_soil = 0.
    total_hexose_exudation = 0.
    total_hexose_uptake = 0.
    total_hexose_degradation = 0.
    total_CO2 = 0.
    total_CO2_root_growth = 0.
    total_CO2_root_maintenance = 0.

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)

        total_length += n.length
        total_biomass += n.biomass

        total_sucrose_root += (n.C_sucrose_root * n.biomass)
        total_hexose_root += (n.C_hexose_root * n.biomass)
        total_hexose_soil +=  (n.C_hexose_soil * n.biomass)
        total_CO2 += n.resp_maintenance + n.resp_growth
        total_CO2_root_growth += n.resp_growth
        total_CO2_root_maintenance += n.resp_maintenance

        total_hexose_exudation += n.hexose_exudation
        total_hexose_uptake += n.hexose_uptake
        total_hexose_degradation += n.hexose_degradation

    print "The total root length is", \
        "{:.1f}".format(Decimal(total_length*100)), "cm."
    print "The total root biomass is", \
        "{:.2E}".format(Decimal(total_biomass)), "g, i.e.", \
        "{:.2E}".format(Decimal(total_biomass*0.44/12.01)), "mol of C."
    print "The total amount of sucrose in the roots is", \
        "{:.2E}".format(Decimal(total_sucrose_root)), "mol of sucrose, i.e.",\
        "{:.2E}".format(Decimal(total_sucrose_root*12)), "mol of C."
    print "The total amount of hexose in the roots is", \
        "{:.2E}".format(Decimal(total_hexose_root)), "mol of hexose, i.e.",\
        "{:.2E}".format(Decimal(total_hexose_root*6)), "mol of C."
    print "The total amount of hexose in the soil is", \
        "{:.2E}".format(Decimal(total_hexose_soil)), "mol of hexose, i.e.",\
        "{:.2E}".format(Decimal(total_hexose_soil*6)), "mol of C."
    print "The total amount of CO2 respired by the roots is", \
        "{:.2E}".format(Decimal(total_CO2)), "mol of C, including", \
        "{:.2E}".format(Decimal(total_CO2_root_growth)),"mol of C for growth and", \
        "{:.2E}".format(Decimal(total_CO2_root_maintenance)),"mol of C for maintenance."
    print "The total net amount of hexose exuded by roots is", \
        "{:.2E}".format(Decimal(total_hexose_exudation-total_hexose_uptake)), "mol of hexose, i.e.", \
        "{:.2E}".format(Decimal((total_hexose_exudation-total_hexose_uptake)*6)), "mol of C."
    print "This corresponds to an exudation rate of",\
        "{:.2E}".format(Decimal((total_hexose_exudation-total_hexose_uptake)*6*12.01/total_biomass*1./time_step_in_days)), \
        "gram of C per gram of biomass per day."
    print "The total net amount of hexose degraded in the soil is", \
        "{:.2E}".format(Decimal(total_hexose_degradation)), "mol of hexose, i.e.", \
        "{:.2E}".format(Decimal(total_hexose_degradation*6)), "mol of C."

########################################################################################################################
########################################################################################################################
# MODULE "SOIL TRANSFORMATION"
########################################################################################################################
########################################################################################################################

# Degradation of hexose in the soil (microbial consumption):
# ---------------------------------------------------------
def soil_hexose_degradation(g, time_step_in_seconds = 1. * (60.*60.*24.)):

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
        n.hexose_degradation = 0.

        # First, we ensure that the element has a positive length and a positive hexose concentration:
        if n.length <= 0. or n.C_hexose_soil <= 0.:
            continue

        # We also check whether the concentration of hexose in soil is positive or not:
        if n.C_hexose_soil <=0.:
            # If the concentration in the soil is negative or 0, then soil degradation process can't occur:
            n.hexose_degradation = 0.
            print "WARNING: No degradation in the soil occured for node", n.index(),\
                "because soil hexose concentration was", n.C_hexose_soil,"mol/g."
            continue
        else:
            # Otherwise, we calculate the potential amount of hexose in the soil that is transformed:
            n.external_surface, n.volume = surface_and_volume(n,n.radius,n.length)
            # hexose_degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
            potential_hexose_degradation = n.external_surface * degradation_max * n.C_hexose_soil \
                                   / (Km_degradation + n.C_hexose_soil) * time_step_in_seconds

        # We modify the concentration of hexose in the soil according to the potential degradation:
        potential_C_hexose_soil = n.C_hexose_soil - potential_hexose_degradation / n.biomass
        # If the potential new concentration is negative:
        if potential_C_hexose_soil < 0.:
            # Then the new concentration is set to 0, and the corresponding degradation is calculated:
            n.hexose_degradation = n.C_hexose_soil * n.biomass
            n.C_hexose_soil = 0.
            print "WARNING: Potential degradation in the soil at node", n.index(),\
                "was too high; it has been limited so that soil hexose concentration becomes 0 mol/g."
        else:
            n.hexose_degradation = potential_hexose_degradation
            n.C_hexose_soil = potential_C_hexose_soil


########################################################################################################################
########################################################################################################################
# MODULE "RHIZODEPOSITION"
########################################################################################################################
########################################################################################################################

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def root_hexose_exudation(g,time_step_in_seconds = 1. * (60.*60.*24.)):

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

        #Just in case - we initialize the exudation of hexose in the element:
        n.hexose_exudation=0.

        # First, we ensure that the element has a positive length and a positive hexose concentration:
        if n.length <= 0 or n.C_hexose_root <=0.:
            # Otherwise we go to the next element in the MTG:
            continue

        # We calculate the surface and volume of the root element:
        n.external_surface, n.volume = surface_and_volume(n, n.radius, n.length)

        # We calculate the permeability coefficient P according to the distance of the element from the apex:
        # OPTION 1 (Personeni et al. 2007): n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip*100) ** gamma_exudation
        # OPTION 2:
        if n.label=="Apex":
            n.permeability_coeff = Pmax_apex
        elif n.lateral_emergence_possibility == "Possible":
            n.permeability_coeff = Pmax_apex/5.
        else:
            n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip*100) ** gamma_exudation

        # hexose_exudation is calculated as an efflux by diffusion:
        potential_hexose_exudation \
            = n.external_surface * n.permeability_coeff * (n.C_hexose_root - n.C_hexose_soil) * time_step_in_seconds
        # We calculate the concentrations of hexose in the root and in the soil according to this potential:
        potential_C_hexose_root = n.C_hexose_root - potential_hexose_exudation / n.biomass

        # If the potential new root concentration is negative:
        if potential_C_hexose_root < 0.:
            # Then we set the concentration in the root to 0 mol/g and adjust the amount exudated:
            n.hexose_exudation = (n.C_hexose_root - 0.) * n.biomass
            n.C_hexose_root = 0.
            print "WARNING: Potential hexose exudation at node", n.index(),"was too high; it has been limited " \
                                                                           "so that root hexose concentration becomes 0 mol/g."
        else:
            # Otherwise, the exudation is done up to the potential calculated:
            n.hexose_exudation = potential_hexose_exudation
            n.C_hexose_root = potential_C_hexose_root

        # In all cases, the new concentration of hexose in the soil is:
        n.C_hexose_soil = n.C_hexose_soil + n.hexose_exudation / n.biomass

# Uptake of hexose from the soil by the root:
# -------------------------------------------
def root_hexose_uptake(g, time_step_in_seconds=1. * (60. * 60. * 24.)):
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

        #Just in case - we initialize the exudation of hexose in the element:
        n.hexose_uptake=0.

        # First, we ensure that the element has a positive length and a positive hexose concentration:
        if n.length <= 0 or n.C_hexose_root <=0.:
            # Otherwise we go to the next element in the MTG:
            continue

        potential_hexose_uptake \
            = n.external_surface * Imax * n.C_hexose_soil / (Km_influx + n.C_hexose_soil) * time_step_in_seconds

        # We calculate the concentrations of hexose in the root and in the soil according to this potential:
        potential_C_hexose_soil = n.C_hexose_soil - potential_hexose_uptake / n.biomass

        # If the potential new root concentration is negative:
        if potential_C_hexose_soil < 0.:
            # Then we set the concentration in the soil to 0 mol/g and adjust the amount taken up by the root:
            n.hexose_uptake = (n.C_hexose_soil - 0.) * n.biomass
            n.C_hexose_soil = 0.
            print "WARNING: Potential hexose uptake at node", n.index(), \
                "was too high; it has been limited so that soil hexose concentration becomes 0 mol/g."
        else:
            # Otherwise, the exudation is done up to the potential calculated:
            n.hexose_uptake = potential_hexose_uptake
            n.C_hexose_soil = potential_C_hexose_soil

        # In all cases, the new concentration of hexose in the root is:
        n.C_hexose_root = n.C_hexose_root + n.hexose_uptake / n.biomass
        n.specific_net_exudation = (n.hexose_exudation - n.hexose_uptake)/n.length

########################################################################################################################
########################################################################################################################
# MODULE "MAINTENANCE RESPIRATION"
########################################################################################################################
########################################################################################################################

# Function calculating maintenance respiration:
def maintenance_respiration(g, time_step_in_seconds=1.*(60.*60.*24.)):
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

        # First, we ensure that the element has a positive length and a positive hexose concentration:
        if n.length <= 0 or n.C_hexose_root <=0.:
            # Otherwise we go to the next element in the MTG:
            continue

        # We calculate the number of moles of CO2 generated by maintenance respiration over the time_step:
        n.resp_maintenance = resp_maintenance_max * n.C_hexose_root / (Km_maintenance + n.C_hexose_root) \
                             * n.biomass * time_step_in_seconds

        # The new concentration of hexose in the root after maintenance respiration is calculated
        # knowing that 1 mol of hexose generates 6 mol of CO2:
        n.C_hexose_root = n.C_hexose_root - n.resp_maintenance / 6.

    return g

########################################################################################################################
########################################################################################################################
# MODULE "POTENTIAL GROWTH"
########################################################################################################################
########################################################################################################################

# DEFINITION OF POTENTIAL GROWTH OF EACH APEX AND SEGMENT:
##########################################################

# Main function of apical development:
#-------------------------------------

def potential_apex_development(apex, time_step_in_seconds = 1. * (60.*60.*24.)):

    # The result of this function, new_apex, is initialized as empty:
    new_apex = []
    # We record the current radius and length prior to growth as the initial radius and length:
    apex.initial_radius=apex.radius
    apex.initial_length=apex.length
    # We initialize the properties "potential_radius" and "potential_length" returned by the function:
    apex.potential_radius=apex.radius
    apex.potential_length=apex.length

    # CASE 1: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A LATERAL ROOT THAT MAY EMERGE FROM A NORMAL ROOT SEGMENT
    if apex.type == "Normal_root_before_emergence":
        # If the time since primordium formation is higher than the delay of emergence:
        if apex.time_since_primordium_formation + time_step_in_seconds >= emergence_delay:
            # The actual time elapsed at the end of this time step since the emergence is calculated:
            apex.potential_time_since_emergence=apex.time_since_primordium_formation + time_step_in_seconds - emergence_delay
            # The corresponding elongation of the apex is calculated:
            elongation = EL * 2. * apex.radius * apex.potential_time_since_emergence
            apex.potential_length = apex.length + elongation
            volume = apex.radius ** 2 * pi * apex.potential_length
            emergence_cost = volume * density * biomass_C_content
            # We select the segment on which the primordium has been formed:
            vid = apex.index()
            index_parent = g.Father(vid, EdgeType='+')
            parent = g.node(index_parent)
            # The possibility of emergence of a lateral root from the parent
            # and the associated biomass C cost are recorded inside the parent:
            parent.lateral_emergence_possibility = "Possible"
            parent.emergence_cost = emergence_cost
            # And the new element returned by the function corresponds to the same apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex
        else:
            # Otherwise the time since primordium formation is simply increased by the time step:
            apex.time_since_primordium_formation += time_step_in_seconds
            # And the new element returned by the function corresponds to the same apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

    # CASE 2: THE APEX CAN FORM A PRIMORDIUM ON ITS SURFACE
    # NOTE: In this configuration, the primordium is formed at the tip of the apex!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
    # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
    # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014):
    potential_radius = np.random.normal(apex.radius * RMD, apex.radius * RMD * CVDD)
    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:

    # WARNING: If more than 1 primordium should be formed because the apex length is too long, the present code only forms 1!!!!!!!!!!!!!!!!!!!
    if apex.dist_to_ramif >= 2 * IPD:
        print "The time step is probably too high, as more than one primordia had to be formed at the same time step on an apex."

    if apex.dist_to_ramif >= IPD and potential_radius >= Dmin:
        # Then we add a primordium of a lateral root at the base of the apex:
        ramif = apex.add_child(edge_type='+',
                               label='Apex',
                               type='Normal_root_before_emergence',
                               lateral_emergence_possibility='Impossible',

                               # The length of the primordium is set to 0:
                               length=0.,
                               radius=potential_radius,
                               external_surface=0.,
                               volume=0.,
                               biomass=0.,

                               # The concentrations should therefore be 0:
                               #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                               C_sucrose_root=apex.C_sucrose_root,
                               C_hexose_root=apex.C_hexose_root,
                               C_hexose_soil=apex.C_hexose_soil,

                               resp_maintenance = 0.,
                               resp_growth = 0.,

                               hexose_exudation=0.,
                               hexose_uptake=0.,
                               hexose_degradation=0.,
                               specific_net_exudation = 0.,

                               growth_duration=GDs * potential_radius * potential_radius * 4,
                               life_duration=LDs * potential_radius * potential_radius * 4 * RTD,
                               # The actual time elapsed since the formation of this primordium is calculated
                               # according to the actual growth of the parent apex since formation:
                               time_since_primordium_formation=(apex.dist_to_ramif - IPD) / (EL * 2. * apex.radius),
                               time_since_emergence=0.,
                               time_since_growth_stopped=0.,
                               time_since_death=0,
                               # The distance between this root apex and the last ramification on this new axis is set to 0:
                               dist_to_ramif=0.)
        # And the new apex now contains the primordium of a lateral root:
        new_apex.append(ramif)
        # And the new distance between the element and the last ramification is calculated as:
        apex.dist_to_ramif = apex.dist_to_ramif - IPD

    # CASE 3: THE APEX CAN ELONGATE
    if apex.type == "Normal_root_after_emergence":
        apex.time_since_primordium_formation += time_step_in_seconds
        apex.time_since_emergence += time_step_in_seconds
        # ELONGATION OF THE APEX:
        # Elongation is calculated following the rules of Pages et al. (2014):
        elongation = EL * 2. * apex.radius * time_step_in_seconds
        # If the length of the apex is smaller than the defined length of a root segment:
        apex.potential_length = apex.length + elongation
        # The new apex is defined as the modified apex:
        new_apex.append(apex)
        # IMPORTANT: THERE IS NO SEGMENT FORMATION AT THIS STAGE!
    return new_apex

# Main function of radial growth and segment death:
#--------------------------------------------------

def potential_segment_development(segment, time_step_in_seconds = 1. * (60.*60.*24.)):

    # Initialization of variables:
    new_segment=[]
    sum_sections = 0.
    death_count = 0.

    # We record the current radius and length prior to growth as the initial radius and length:
    segment.initial_radius=segment.radius
    segment.initial_length=segment.length
    # We initialize the properties "potential_radius" and "potential_length" returned by the function:
    segment.potential_radius=segment.radius
    segment.potential_length=segment.length

    # For each child of the segment:
    for child in segment.children():

        # We add the section of the child to a sum of children sections:
        sum_sections += child.radius**2 * pi

        # If the child is dead:
        if child.type=="Dead":
            # Then we add one dead child to the death count:
            death_count+=1

    # If each child in the list of children has been recognized as dead:
    if death_count == len(segment.children()):
        # Then the segment becomes dead, and has no radius or length anymore:
        segment.type = "Dead"
        segment.radius = 0.
        segment.length = 0.
        # Otherwise, at least one of the children axes is not dead, so the father segment should not die

    # The radius of the root segment is defined according to the pipe model,
    # i.e. the radius is equal to the square root of the sum of the radius of all children
    # proportionally to the coefficient SGC:
    segment.potential_radius = sqrt(SGC * sum_sections / pi)

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

        # We define the list of apices for all vertices labelled as "Apex":
        self._apices = [g.node(v) for v in g.vertices_iter(scale=1) if g.label(v)=='Apex']

        # We define the list of segments for all vertices labelled as "Segment", from the apex to the base:
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = root_gen.next()
        self._segments = [g.node(v) for v in post_order(g, root) if g.label(v) == 'Segment']

    def step(self, time_step_in_seconds=1.*(60.*60.*24.)):
        g = self.g
        # We define "apices" and "segments" as the list of apices and segments in g:
        apices_list = list(self._apices)
        segments_list=list(self._segments)

        # For each apex in the list of apices:
        for apex in apices_list:
            # We define the new list of apices with the function apex_development:
            new_apex = potential_apex_development(apex, time_step_in_seconds)
            # We add these new apices to apex:
            self._apices.extend(new_apex)

        # For each segment in the list of segments:
        for segment in segments_list:
            # We define the new list of apices with the function apex_development:
            new_segment = potential_segment_development(segment, time_step_in_seconds)
            # We add these new apices to apex:
            self._segments.extend(new_segment)

# We finally define the function that calculates the potential growth of the whole MTG at a given time step:
def potential_growth(g, time_step_in_seconds = 1. * (60.*60.*24.)):

    # We simulate the development of all apices and segments in the MTG:
    simulator = Simulate_potential_growth(g)
    simulator.step(time_step_in_seconds=time_step_in_seconds)

########################################################################################################################
########################################################################################################################
# MODULE "ACTUAL GROWTH AND ASSOCIATED RESPIRATION"
########################################################################################################################
########################################################################################################################

# ACTUAL ELONGATION AND RADIAL GROWTH OF ROOT ELEMENTS:
#######################################################

# Function calculating the actual growth and the corresponding growth respiration:
def actual_growth_and_corresponding_respiration(g):
    """
    This function defines how a segment and possibly an emerging root primordium may grow according to the amount
    of hexose present in the segment, taking into account growth respiration based on the model of Thornley and Cannell
    (2000). The calculation is based on the values of potential_radius, potential_length, and emergence_cost defined in
    each segment by the module "POTENTIAL GROWTH".
    The function returns the MTG "g" with modified values of radius and length of each element, the possibility of the
    emergence of lateral roots, and the modified value of C_hexose_root after growth.
    """

    # We have to cover each vertex from the apices up to the base one time:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()

    # We cover all the vertices in the MTG:
    for vid in post_order(g, root):

        # n represents the vertex:
        n = g.node(vid)

        # First, we ensure that the element does not correspond to a primordium that has not emerged,
        # as they are managed through their parent. If the element is a primordium:
        if n.type=="Normal_root_before_emergence":
            # Then we pass to the next element in the iteration:
            continue

        # We calculate the initial surface and volume of the element:
        initial_surface, initial_volume = surface_and_volume(n, n.initial_radius, n.initial_length)
        # The initial biomass of the element is also recorded:
        n.initial_biomass = initial_volume * density
        # We calculate the potential surface and volume of the element based on the potential radius and potential length:
        potential_surface, potential_volume = surface_and_volume(n, n.potential_radius, n.potential_length)

        # If the emergence of a primordium is not possible:
        if n.lateral_emergence_possibility != "Possible":
            # Then the emergence cost is set to 0 (else, the emergence cost has already been calculated):
            n.emergence_cost=0.

        # We calculate the number of moles of C included in the biomass potentially produced over the time_step,
        # where density is the dry structural weight per volume (g m-3), biomass_C_content is the amount of C
        # per gram of dry structural mass (mol_C g-1) and n.emergence_cost the biomass C requirement of a lateral
        # primordium to be emerged, if any:
        potential_growth_demand = (potential_volume - initial_volume) * density * biomass_C_content + n.emergence_cost
        # The amount of hexose required for sustaining the potential growth is calculated by including the respiration
        # cost according to the model of Thornley and Cannell (2000):
        hexose_growth_demand = 1. / 6. * potential_growth_demand / yield_growth

        # The total amount of hexose available at this stage in the root element is calculated:
        hexose_available = n.C_hexose_root * n.initial_biomass

        # CASE 1: THE AMOUNT OF HEXOSE AVAILABLE IS NOT LIMITING THE GROWTH OF THE ELEMENT
        # OR THE EMERGENCE OF A PRIMORDIUM:

        # If there is enough hexose available at this stage to cover all costs related to the growth of the segment:
        if hexose_growth_demand <= hexose_available:

            # Then the length and the radius of the segment are increased up to their potential:
            n.length = n.potential_length
            n.radius = n.potential_radius
            # The surface and volume of the element are automatically calculated:
            n.surface, n.volume= surface_and_volume(n, n.radius, n.length)
            # The new dry structural biomass of the element is calculated from its new volume:
            n.biomass = n.volume * density

            # If the emergence of a primordium is possible:
            if n.lateral_emergence_possibility == "Possible":
                # We get the node corresponding to the primordium to be emerged:
                index_primordium = g.Sons(vid, EdgeType='+')[0]
                primordium = g.node(index_primordium)

                # The emergence of the primordium is done up to the full length:
                primordium.length = primordium.potential_length
                primordium.dist_to_ramif=primordium.length
                # The corresponding external surface and volume of the emerged apex are calculated:
                primordium.surface, primordium.volume = surface_and_volume(primordium, primordium.radius, primordium.length)
                primordium.biomass = primordium.volume * density
                # And the primordium is now defined as an emerged apex:
                primordium.type = "Normal_root_after_emergence"
                n.lateral_emergence_possibility == "Impossible"
                # AT THIS STAGE, THE EMERGED APEX DOES NOT CONTAIN ANY SUCROSE OR HEXOSE!

            # The total respiration cost is calculated and memorized:
            n.resp_growth = (1 - yield_growth) / (yield_growth) * potential_growth_demand

            # The new concentration of sucrose in the root is calculated
            # according to the initial sucrose concentration and the new biomass:
            n.C_sucrose_root = n.C_sucrose_root * n.initial_biomass  / n.biomass
            # The new concentration of hexose in the root is calculated
            # according to the remaining hexose and the new biomass:
            n.C_hexose_root = (hexose_available - hexose_growth_demand) / n.biomass
            # The new concentration of hexose in the soil is calculated
            # according to the initial soil hexose concentration and the new biomass:
            n.C_hexose_soil = n.C_hexose_soil * n.initial_biomass / n.biomass

        # CASE 2: THE AMOUNT OF HEXOSE AVAILABLE DOES LIMIT THE GROWTH OF THE ELEMENT
        # OR THE EMERGENCE OF A PRIMORDIUM:

        # Otherwise, at least one type of growth is limited:
        else:

            # We calculate the maximal volume of the root segment according to all the hexose available:
            volume_max = initial_volume + hexose_available * 6. * yield_growth / (density * biomass_C_content)
            # We calculate the maximal possible length based on a constant radius:
            length_max = volume_max / (pi * n.radius ** 2)

            # SUBCASE 1 - If elongation is limited by the amount of hexose available:
            if length_max <= n.potential_length:
                # Elongation is done according to the amount of hexose available;
                # no radial growth and no primordium emergence is possible:
                n.length = length_max
                print "On segment", n.index() ,"no emergence of primordium was possible because residual hexose was too low !!!"
                print ""

            # SUBCASE 2 - Otherwise, there is enough hexose to perform elongation and at least one other type of growth:
            else:

                # ELONGATION IS DONE FIRST:
                # Elongation is done up to the full potential, and the corresponding new volume is calculated:
                n.length = n.potential_length
                volume_after_elongation = pi * n.initial_radius ** 2 * n.length
                # We calculate the remaining amount of hexose after elongation:
                remaining_hexose = hexose_available \
                                   - 1. / 6. * (volume_after_elongation - initial_volume) * density * biomass_C_content / yield_growth
                # The remaining hexose can still be used for radial growth and primordium emergence.

                # EMERGENCE IS THEN CONSIDERED
                # If the emergence of the primordium has been declared possible by the growth module:
                if n.lateral_emergence_possibility == "Possible":

                    # We get the node corresponding to the primordium:
                    index_primordium = g.Sons(vid, EdgeType='+')[0]
                    primordium = g.node(index_primordium)

                    # If there is enough hexose for primordium emergence:
                    if remaining_hexose - n.emergence_cost * 1. / 6. / yield_growth >= 0.:
                        # The emergence of the primordium is done up to the full length:
                        primordium.length = primordium.potential_length
                        primordium.dist_to_ramif = primordium.length
                        primordium.type = "Normal_root_after_emergence"
                        n.lateral_emergence_possibility == "Impossible"
                        # And the remaining hexose is calculated after the emergence:
                        remaining_hexose = remaining_hexose - n.emergence_cost * 1. / 6. / yield_growth

                    # Otherwise, the emergence is done proportionally to the available hexose, and no radial growth occurs:
                    else:
                        # PRIORITY TO THE EMERGENCE OF PRIMORDIUM:
                        # We calculate the net increase in volume of the emerging primordium:
                        increase_in_volume = remaining_hexose * 6. * yield_growth / (density * biomass_C_content)
                        # The emergence of the primordium is done up a certain length:
                        primordium.length = primordium.length + increase_in_volume / (primordium.radius ** 2 * pi)
                        primordium.dist_to_ramif = primordium.length
                        primordium.type = "Normal_root_after_emergence"
                        n.lateral_emergence_possibility == "Impossible"
                        # No hexose remains anymore:
                        remaining_hexose = 0.

                    # The corresponding external surface and volume of the emerged apex are calculated:
                    primordium.surface, primordium.volume = surface_and_volume(primordium, primordium.radius, primordium.length)
                    primordium.biomass = primordium.volume * density
                    # AT THIS STAGE, THE EMERGED APEX DOES NOT CONTAIN ANY SUCROSE OR HEXOSE!

                # FINALLY, RADIAL GROWTH CAN OCCUR WITH THE REMAINING HEXOSE:
                # The remaining hexose is used for radial growth, if any:
                radial_increase_in_volume = remaining_hexose * 6. * yield_growth / (density * biomass_C_content)
                n.radius = sqrt((n.volume + radial_increase_in_volume) / (n.length * pi))

            # The surface and volume of the element are automatically calculated:
            n.surface, n.volume= surface_and_volume(n, n.radius, n.length)
            # The new dry structural biomass of the element is calculated from its new volume:
            n.biomass = n.volume * density

            # The amount of hexose that has been used for growth respiration is calculated:
            n.resp_growth = (1 - yield_growth) / (yield_growth) * hexose_available * 6.

            if n.biomass<=0:
                n.C_sucrose_root=0.
                n.C_hexose_root=0.
                n.C_hexose_soil=0.
            else:
                # The new concentration of sucrose in the root is calculated
                # according to the initial sucrose concentration and the new biomass:
                n.C_sucrose_root = n.C_sucrose_root * n.initial_biomass  / n.biomass
                # The new concentration of hexose in the root is set to 0 since all hexose has been used:
                n.C_hexose_root = 0.
                # The new concentration of hexose in the soil is calculated
                # according to the initial soil hexose concentration and the new biomass:
                n.C_hexose_soil = n.C_hexose_soil * n.initial_biomass / n.biomass

            # In case the root element corresponds to an apex, the distance to the last ramification is increased:
            if n.label=="Apex":
                n.dist_to_ramif += n.length - n.initial_length

    return g

# SEGMENTATION OF ELONGATED APICES
##################################

def segment_formation(apex):

    # The result of this function, new_apex, is initialized as empty:
    new_apex = []

  # IF THE APEX IS LONGER THAN A SEGMENT AND SHOULD BE SUBDIVIDED INTO SEGMENTS
    if apex.length > segment_length:

        # We first calculate the number of entire segments to be formed within the apex.
        # If the length of the apex does not correspond to an entire number of segments:
        if apex.length / segment_length - trunc(apex.length / segment_length) > 0.:
            # Then the total number of segments to be formed is:
            n_segments = trunc(apex.length / segment_length)
        else:
            # Otherwise, the number fo segments to be formed is decreased by 1,
            # so that the last element corresponds to an apex with a positive length:
            n_segments = trunc(apex.length / segment_length) - 1

        initial_length = apex.length
        initial_biomass = apex.biomass
        initial_resp_maintenance = apex.resp_maintenance
        initial_resp_growth = apex.resp_growth
        initial_hexose_exudation = apex.hexose_exudation
        initial_hexose_uptake = apex.hexose_uptake
        initial_hexose_degradation = apex.hexose_degradation

        # We do an iteration on n-1 segments:
        for i in range(1,n_segments):
            # We define the length of the present element as the length of a segment:
            apex.length = segment_length
            # We modify the geometrical features of the present element accordingly:
            apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
            apex.biomass = apex.volume * density
            # We modify the variables representing total amounts according to the new biomass:
            apex.resp_maintenance = initial_resp_maintenance * apex.biomass / initial_biomass
            apex.resp_growth = initial_resp_growth * apex.biomass / initial_biomass
            apex.hexose_exudation = initial_hexose_exudation * apex.biomass / initial_biomass
            apex.hexose_uptake = initial_hexose_uptake * apex.biomass / initial_biomass
            apex.hexose_degradation = initial_hexose_degradation * apex.biomass / initial_biomass
            # The element is now considered as a segment:
            apex.label = 'Segment'
            # And we add a new apex after this segment with the length of a segment:
            apex=apex.add_child(edge_type='<',
                                label='Apex',
                                type='Normal_root_after_emergence',
                                lateral_emergence_possibility="Impossible",
                                # We initially define the length of the new apex as 0:
                                length=0.,
                                radius=apex.radius,
                                external_surface=0.,
                                volume=0.,
                                biomass=0.,

                                C_sucrose_root=apex.C_sucrose_root,
                                C_hexose_root=apex.C_hexose_root,
                                C_hexose_soil=apex.C_hexose_soil,

                                resp_maintenance = apex.resp_maintenance,
                                resp_growth = apex.resp_growth,
                                hexose_exudation=apex.hexose_exudation,
                                hexose_uptake=apex.hexose_uptake,
                                hexose_degradation=apex.hexose_degradation,
                                specific_net_exudation = apex.specific_net_exudation,

                                time_since_primordium_formation=apex.time_since_primordium_formation,
                                time_since_emergence=apex.time_since_emergence,
                                time_since_growth_stopped=apex.time_since_growth_stopped,
                                growth_duration=apex.growth_duration,
                                life_duration=apex.life_duration,

                                dist_to_ramif=apex.dist_to_ramif + segment_length)

        # Finally, we transform the last apex into a segment one last time:
        apex.length = segment_length
        # We modify the geometrical features of the present element accordingly:
        apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
        apex.biomass = apex.volume * density
        # We modify the variables representing total amounts according to the new biomass:
        apex.resp_maintenance = initial_resp_maintenance * apex.biomass / initial_biomass
        apex.resp_growth = initial_resp_growth * apex.biomass / initial_biomass
        apex.hexose_exudation = initial_hexose_exudation * apex.biomass / initial_biomass
        apex.hexose_uptake = initial_hexose_uptake * apex.biomass / initial_biomass
        apex.hexose_degradation = initial_hexose_degradation * apex.biomass / initial_biomass
        # And the element is now considered as a segment:
        apex.label = 'Segment'
        # And we define a new apex after the new defined segment, with a new length defined as:
        new_length=initial_length - n_segments*segment_length
        apex=apex.add_child(edge_type='<',
                            label='Apex',
                            type='Normal_root_after_emergence',
                            lateral_emergence_possibility="Impossible",
                            length=new_length,
                            radius = apex.radius,

                            C_sucrose_root=apex.C_sucrose_root,
                            C_hexose_root=apex.C_hexose_root,
                            C_hexose_soil=apex.C_hexose_soil,

                            resp_maintenance=apex.resp_maintenance,
                            resp_growth=apex.resp_growth,
                            hexose_exudation=apex.hexose_exudation,
                            hexose_uptake=apex.hexose_uptake,
                            hexose_degradation=apex.hexose_degradation,
                            specific_net_exudation = apex.specific_net_exudation,

                            time_since_primordium_formation=apex.time_since_primordium_formation,
                            time_since_emergence=apex.time_since_emergence,
                            time_since_growth_stopped=0.,
                            growth_duration=apex.growth_duration,
                            life_duration=apex.life_duration,
                            dist_to_ramif = apex.dist_to_ramif + new_length)

        apex.external_surface,apex.volume=surface_and_volume(apex,apex.radius,apex.length)
        apex.biomass = apex.volume * density
        # We modify the variables representing total amounts according to the new biomass:
        apex.resp_maintenance = initial_resp_maintenance * apex.biomass / initial_biomass
        apex.resp_growth = initial_resp_growth * apex.biomass / initial_biomass
        apex.hexose_exudation = initial_hexose_exudation * apex.biomass / initial_biomass
        apex.hexose_uptake = initial_hexose_uptake * apex.biomass / initial_biomass
        apex.hexose_degradation = initial_hexose_degradation * apex.biomass / initial_biomass
        # The new apex is defined as the modified apex:
        new_apex.append(apex)

    return new_apex

# We define a class "Simulate" which is used to simulate the development of apices and segments on the whole MTG "g":
class Simulate_segment_formation(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        """ Simulate on MTG. """
        self.g = g
        # We define the list of apices for all vertices labelled as "Apex":
        self._apices = [g.node(v) for v in g.vertices_iter(scale=1) if g.label(v)=='Apex']

    def step(self):
        g = self.g
        # We define "apices" and "segments" as the list of apices and segments in g:
        apices_list = list(self._apices)

        # For each apex in the list of apices:
        for apex in apices_list:
            # We define the new list of apices with the function apex_development:
            new_apex = segment_formation(apex)
            # We add these new apices to apex:
            self._apices.extend(new_apex)

# We finally define the function that calculates segmentation over the whole MTG at a given time step:
def segmentation(g):

    # We simulate the segmentation of all apices:
    simulator = Simulate_segment_formation(g)
    simulator.step()

########################################################################################################################
########################################################################################################################
# MODULE "CONVERSION OF SUCROSE INTO HEXOSE"
########################################################################################################################
########################################################################################################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def sucrose_to_hexose(g, time_step_in_seconds = 1. * (60.*60.*24.)):

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

        # First, we ensure that the element does not correspond to a primordium that has not emerged,
        # as it would make no sense to calculate fluxes on such element:
        if n.length <= 0. or n.C_sucrose_root <=0.:
            # If the element is a primordium, then we pass to the next element in the iteration:
            continue

        n.external_surface, n.volume = surface_and_volume(n, n.radius, n.length)
        # We calculate the unloading coefficient according to the distance from the apex:
        # OPTION 1 (Personeni et al. 2007): n.unloading_coeff = unloading_apex / (1 + n.dist_to_tip*100) ** gamma_unloading
        # OPTION 2:
        if n.label=="Apex":
            n.unloading_coeff = unloading_apex
        elif n.lateral_emergence_possibility == "Possible":
            n.unloading_coeff = unloading_apex/10. / (1 + n.dist_to_tip*100) ** gamma_unloading
        else:
            n.unloading_coeff = unloading_apex/10. / (1 + n.dist_to_tip*100) ** gamma_unloading

        # We calculate the potential production of hexose (in mol) according to the Michaelis-Menten function:
        potential_prod_hexose = 2. * n.external_surface * n.unloading_coeff * n.C_sucrose_root \
                        / (Km_unloading + n.C_sucrose_root) * time_step_in_seconds
        # The factor 2 originates from the conversion of 1 molecule of sucrose into 2 molecules of hexose.

        # The new potential sucrose concentration after the conversion is calculated:
        potential_C_sucrose_root = n.C_sucrose_root - (potential_prod_hexose / 2.) / n.biomass

        # If this potential sucrose concentration is negative:
        if potential_C_sucrose_root < 0.:
            # Then the production of hexose is limited:
            n.prod_hexose = n.C_sucrose_root * n.biomass
            # And the concentration of sucrose in the root is set to 0:
            n.C_sucrose_root = 0.
            print "WARNING: Potential unloading at node", n.index(), "was too high; it has been limited so that " \
                                                                     "root sucrose concentration becomes 0 mol/g."
        else:
            n.prod_hexose = potential_prod_hexose
            n.C_sucrose_root = potential_C_sucrose_root

        # The concentration of hexose in the root is modified accordingly:
        n.C_hexose_root = n.C_hexose_root + n.prod_hexose / n.biomass

    return g

########################################################################################################################
########################################################################################################################
# MODULE "SUCROSE SUPPLY FROM THE SHOOTS"
########################################################################################################################
########################################################################################################################

# Calculating the net input of sucrose by the aerial parts into the root system:
# ------------------------------------------------------------------------------
def shoot_sucrose_supply_and_spreading(g, sucrose_unloading_rate=1e-9, time_step_in_seconds=1.):

    """
    This function calculates the new root sucrose concentration (mol of sucrose per gram of dry root structural mass)
    AFTER the supply of sucrose from the shoot.
    """

    # The input of sucrose over this time step is calculated
    # from the sucrose transport rate provided as input of the function:
    sucrose_input = sucrose_unloading_rate * time_step_in_seconds

    # We calculate the remaining amount of sucrose in the root system,
    # based on the current sucrose concentration and biomass of each root element:
    total_sucrose_root, total_biomass = total_root_sucrose_and_biomass(g)

    # The new average sucrose concentration in the root system is calculated as:
    C_sucrose_root_after_supply = (total_sucrose_root + sucrose_input) / total_biomass

    new_C_sucrose_root = C_sucrose_root_after_supply
    #new_C_sucrose_root = 0.025 based on Marion Gauthiers' current modelling of vegetative phase

    # We go through the MTG a second time to modify the sugars concentrations:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)
        # The local sucrose concentration in the root is calculated from the new sucrose concentration calculated above:
        n.C_sucrose_root = new_C_sucrose_root

    return g

########################################################################################################################
########################################################################################################################
# MAIN PROGRAM:
########################################################################################################################
########################################################################################################################

# INITIALIZATION OF THE MTG
###########################

def mtg_initialization(density=0.1e6):

    g = MTG()
    # We first add one initial apex:
    apex_id = g.add_component(g.root, label='Apex')
    apex = g.node(apex_id)

    # We define the geometrical properties of the first element:
    apex.type="Normal_root_after_emergence"
    apex.length = 0.01
    apex.radius = Di
    apex.external_surface, apex.volume = surface_and_volume(apex, apex.radius, apex.length)
    apex.biomass = apex.volume * density

    apex.dist_to_ramif = apex.length
    apex.dist_to_tip=0.

    # We define the initial sugar concentrations:
    apex.C_sucrose_root = 0.0025
    apex.C_hexose_root = 0.
    apex.C_hexose_soil = 0.

    apex.lateral_emergence_possibility="Impossible"

    apex.resp_maintenance=0.
    apex.resp_growth=0.
    apex.hexose_exudation=0.
    apex.hexose_uptake = 0.
    apex.hexose_degradation=0.
    apex.specific_net_exudation=0.

    # We define time-related parameters:
    apex.growth_duration= GDs * apex.radius * apex.radius * 4
    apex.life_duration = LDs * apex.radius * apex.radius * 4 * RTD
    apex.time_since_primordium_formation = emergence_delay
    apex.time_since_emergence=0.
    apex.time_since_growth_stopped=0.
    apex.time_since_death=0.
    apex.time_stop = 15

    return g

# SIMULATION OVER TIME:
#######################

# We initiate the properties of the MTG "g":
g=mtg_initialization()

# We can check the current directory path:
print ""
print "The current directory path is:", os.getcwd()

# We read the data showing the unloading rate of sucrose as a function of time from a file:
#------------------------------------------------------------------------------------------
# We first define the path and the file to read as a .csv:
PATH = os.path.join('C:/','Users', 'frees', 'rhizodep','test','organs_states.csv')
# Then we read the file and copy it in a dataframe "df":
df = pd.read_csv(PATH, sep=',')
# We only keep the two columns of interest:
df=df[['t','Unloading_Sucrose']]
# We remove any "NA" in the column of interest:
df= df[pd.notnull(df['Unloading_Sucrose'])]
# We remove any values below 0:
sucrose_input_frame=df[(df.Unloading_Sucrose>=0.)]
# We reset the indices after having filtered the data:
sucrose_input_frame=sucrose_input_frame.reset_index(drop=True)

# We define the time constraints of the simulation:
simulation_period_in_days = 40
time_step_in_days= 2.
time_step_in_seconds = time_step_in_days *60.*60.*24.

# We calculate the number of steps necessary to reach the end of the simulation period:
n_steps=trunc(simulation_period_in_days / time_step_in_days)+1

# An iteration is done for each time step:
for step in range(0,n_steps):

    # Defining the rate of transport of sucrose from shoots to roots:
    #----------------------------------------------------------------
    # We look for the first item in the time series of sucrose_input_frame for which time is higher
    # than the current time (a default value of 0 is given otherwise):
    current_time_in_hours = step * time_step_in_days * 24.
    considered_time = next((time for time in sucrose_input_frame.t if time > current_time_in_hours), 0)
    # If the current time in the loop is higher than any time indicated in the dataframe:
    if considered_time == 0:
        # Then we use the last value of sucrose unloading rate as the sucrose input rate:
        sucrose_input_rate = sucrose_input_frame.Unloading_Sucrose[-1]
    else:
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

    #CHEATING TO BE SURE THAT SUCROSE SUPPLY IS NOT AFFECTING THE RESULT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sucrose_input_rate=1.

    print ""
    print "At t =", "{:.2f}".format(Decimal(step*time_step_in_days)), "days:"
    print "------------------"
    print "The input rate of sucrose to the root for time=", current_time_in_hours, "h is", \
        "{:.2E}".format(Decimal(sucrose_input_rate)), "mol of sucrose per second."
    print ""

    # Using the other functions to simulate the evolution of the MTG:
    #----------------------------------------------------------------

    # 1. Consumption of hexose in the soil:
    soil_hexose_degradation(g, time_step_in_seconds=time_step_in_seconds)

    # 2. Transfer of hexose from the root to the soil, consumption of hexose inside the roots:
    root_hexose_exudation(g, time_step_in_seconds=time_step_in_seconds)
    # 2bis. Transfer of hexose from the soil to the root, consumption of hexose in the soil:
    root_hexose_uptake(g, time_step_in_seconds=time_step_in_seconds)

    # 3. Consumption of hexose in the root by maintenance respiration:
    maintenance_respiration(g, time_step_in_seconds=time_step_in_seconds)

    # 3. Calculation of potential growth without consideration of available hexose:
    potential_growth(g, time_step_in_seconds=time_step_in_seconds)

    # 4. Calculation of actual growth based on the hexose remaining in the roots,
    # and corresponding consumption of hexose in the root:
    actual_growth_and_corresponding_respiration(g)
    segmentation(g)
    dist_to_tip(g)

    # 5. Supply of sucrose from the shoots to the roots and spreading into the whole phloem:
    # Defining the rate of transfer of sucrose from shoots to roots:
    shoot_sucrose_supply_and_spreading(g, sucrose_unloading_rate=sucrose_input_rate, time_step_in_seconds=time_step_in_seconds)

    # 6. Unloading of sucrose from phloem and conversion of sucrose into hexose:
    sucrose_to_hexose(g, time_step_in_seconds=time_step_in_seconds)

    total_print(g)

    sc = plot_mtg(g, prop_cmap='hexose_exudation')
    pgl.Viewer.display(sc)

    # The following code line enables to wait for 0.2 second between each iteration:
    #time.sleep(0.2)

#time.sleep(10)
raw_input()


