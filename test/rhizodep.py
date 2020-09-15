# Todo: Negative concentrations should not exist + introduce an accumulation reservoir?
# Todo: Introduce terms of hexose consumption for growth and maintenance
# Todo: Introduce a growing root system

# Loading all the necessary stuffs:
###################################

from math import sqrt, pi

from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order

import time

from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
import openalea.plantgl.all as pgl

from collections import deque

# class Simulation(object):
#     def __init__(self):
#         g = self.g = MTG()
#         self.root = g.add_component(g.root)
#         self.apices = deque([self.root])
#
#     def growth(self):
#         new_apices = deque([])
#         for aid in apices:
#             new_apex = self.update_apex(aid)
#
#     def update_apex(self):
#         assert(aid in self.g)

# Defining functions for diplaying the root system in a 3D graph in PlantGL:
############################################################################

def get_root_visitor():
    def root_visitor(g, v, turtle):
        angles = [90,45]+[30]*20

        n = g.node(v)
        radius = n.radius*10.
        order = g.order(v)
        length = 1.

        if g.edge_type(v) == '+':
            angle = angles[order]
            turtle.down(angle)

        turtle.setId(v)
        turtle.setWidth(radius)
        for c in n.children():
            if c.edge_type() == '+':
                turtle.rollL(130)

        turtle.F(length)

        # define the color property
        #n.color = random.random()
    return root_visitor

def plot(g, prop_cmap='radius', cmap='jet', lognorm=False):
    """
    Exemple:

        >>> from openalea.plantgl.all import *
        >>> s = plot()
        >>> shapes = dict( (x.getId(), x.geometry) for x in s)
        >>> Viewer.display(s)
    """
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

########################################################
# GENERATION OF A RANDOM MTG DESCRIBING THE ROOT SYSTEM
########################################################

# Defining a function for generating a MTG:
# -----------------------------------------
def generate_mtg(n=6):
    """
    This function generates a random MTG based on the number of total vertices, n.
    """
    g = MTG()
    root = g.add_component(g.root)
    vid = random_tree(g, root, nb_children=2, nb_vertices=n)
    return g

# Defining a list of all the tips from the root system:
# -----------------------------------------------------
def tips(g):
    """
    This function returns the list of vertex ID corresponding to the root tips in the MTG "g".
    """
    return [vid for vid in g.vertices_iter(scale=1) if g.is_leaf(vid)]

# Defining the radius of a root element:
# --------------------------------------
def radius(g, r=0.01,alpha=0.8):

    """
    The function "radius" computes the radius (in meter) of each root segment based on the pipe model,
    where the square of the radius of the root segment is equal to the sum of the square of the radius
    of all the children segments. A parameter alpha (adimensional) is also used to lower the radius
    of the apices depending on their root orders.
    """

    # We initialize an empty dictionary for radius:
    radius = {}
    # We define "root" as  the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # If the vertex corresponds to a root apex ("leaf" for a MTG):
        if g.is_leaf(vid):
            # Then we get the order of the apex (i.e. whether it is a 1st order of root, 2nd order, etc.):
            order = g.order(vid)
            # And the radius of the apex is defined based on the reference radius and the order of the root (the higher the order, the smaller the radius):
            radius[vid] = r*(alpha)**order
        else:
            # Else the radius of the root segment is defined according to the pipe model:
            radius[vid] = sqrt(sum((radius[cid])**2 for cid in g.children(vid)))
            # The radius of a root segment is defined as the square root of the sum of the radius of all children (pipe model)

    # We assign the result "radius" as a new property of each vertex in the MTG "g":
    g.properties()['radius'] = radius
    # We return a modified version of the MTG "g" with a new property "radius":
    return g

# Defining the length of a root element:
# --------------------------------------
def length(g, length=0.01):

    """
    The function "length" computes the length (in meter) of each individual root segment in the MTG "g".
    """

    # We create a dictionnary containing a value of "length" for each vertex in the MTG:
    _length = dict((vid, length) for vid in g.vertices_iter(scale=1))

    # We assign the result "length" as a new property of each vertex in the MTG "g":
    g.properties()['length'] = _length
    # We return a modified version of the MTG "g" with a new property "length":
    return g

##########################################
# DEFINITION OF THE PROPERTIES OF THE MTG
##########################################

# Defining the surface of each root element in contact with the soil:
# -------------------------------------------------------------------
def surface_and_volume(g):

    """
    The function "surface_and_volume" computes the "external_surface" (m2) and "volume" (m3) of each root element
    in the MTG "g", based on the properties radius (m) and length (m) defined in the functions "radius" and "length".
    If the root element is an apex, the external surface is defined as the surface of a cone of height = "length"
    and radius = "radius", and the volume is the volume of this cone.
    If the root element is a root segment, the external surface is defined as the lateral surface of a cylinder
    of height = "length" and radius = "radius", and the volume corresponds to the volume of this cylinder.
    For each branching on the root segment, the section of the daughter root is subtracted from the cylinder surface
    of the root segment of the mother root.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):

        # We calculate the number of children of the root element:
        number_of_children=len(g.children(vid))
        # n represents the vertex:
        n = g.node(vid)
        # If there is no children to the root element:
        if number_of_children==0:
            # Then the root element corresponds to an apex considered as a cone
            # of height = "length" and radius = "radius":
            n.external_surface = pi * n.radius*sqrt(n.radius**2 + n.length**2)
            n.volume = pi * n.radius**2 * n.length / 3
        # If there is only one child to the root element:
        elif number_of_children==1:
            # Then the root element is a root segment without branching, considered as a cylinder
            # of height = "length" and radius = "radius":
            n.external_surface = 2 * pi * n.radius * n.length
            n.volume = pi * n.radius ** 2 * n.length
        # If there is one or more lateral roots branched on the root segment:
        else:
            # Then we sum all the sections of the lateral roots branched on the root segment:
            sum_ramif_sections=0
            for child_vid in g.Sons(vid, EdgeType='+'):
                son = g.node(child_vid)
                sum_ramif_sections += pi * son.radius ** 2
            # And we subtract this sum of sections from the external area of the main cylinder:
            n.external_surface = 2 * pi * n.radius * n.length - sum_ramif_sections
            n.volume = pi * n.radius ** 2 * n.length

# Defining the distance of a vertex from the tip:
# -----------------------------------------------
def dist_to_tip(g):

    """
    The function "dist_to_tip" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices
    [see the function "length"].
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
        if g.is_leaf(vid):
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
    return g

# Defining the structural dry biomass of each element:
# ---------------------------------------------------
def biomass(g, density=0.1e6):

    """
    The function "biomass" computes the dry structural biomass of each root segment (in gram)
    based on its volume (in m3) and its dry bulk density (in g m-3).
    The biomass is defined as the product of the volume of the root segment with the dry bulk density.
    The default dry biomass density is set at 0.1 g cm-3.
    """

    # We use the properties "radius" and "length" of each vertex defined by the functions "radius" and "length":
    volume = g.property('volume')

    # We define a function "biom" expressing the dry structural biomass as a function of the radius and the length of a root segment:
    def biom(vid):
        #The function calculates the dry biomass as the product of the volume of a cylinder and the dry biomass density:
        return volume[vid] * density
    # We cover all the vertices in the MTG using the equation "biom" to calculate biomass:
    _biomass = dict((vid, biom(vid)) for vid in g.vertices_iter(scale=1))
    # We define the result as a new property in the MTG:
    g.properties()['biomass'] = _biomass

# Setting the initial sugar concentrations for each vertex with default values:
# ----------------------------------------------------------------------------
def initial_C_sucrose_root(g, C_sucrose_ref=1e-4):

    """
    This function sets the concentration of sucrose in the phloem of the root system, C_sucrose_root
    (in mol of sucrose per gram of dry root structural mass), to an initial default value, C_sucrose_ref.
    """
    sucrose = dict((vid, C_sucrose_ref) for vid in g.vertices_iter(scale=1))
    g.properties()['C_sucrose_root']=sucrose

def initial_C_hexose_root(g, C_hexose_root_ref=3e-4):

    """
    This function sets the concentration of hexose in the root of the root system, C_hexose_root
    (in mol of sucrose per gram of dry root structural mass), to an initial default value, C_hexose_root_ref.
    """
    hexose_root = dict((vid, C_hexose_root_ref) for vid in g.vertices_iter(scale=1))
    g.properties()['C_hexose_root']=hexose_root

def initial_C_hexose_soil(g, C_hexose_soil_ref=3e-6):

    """
    This function sets the concentration of hexose outside the root of the root system, C_hexose_soil
    (in mol of sucrose per gram of dry root structural mass), to an initial default value, C_hexose_soil_ref.
    """
    hexose_soil = dict((vid, C_hexose_soil_ref) for vid in g.vertices_iter(scale=1))
    g.properties()['C_hexose_soil']=hexose_soil

# Initializing the variables of the MTG:
# --------------------------------------
def init(g):

    """
    This function is used to initialize the properties of the MTG "g", by calling several functions
    and using their default parameters:
    - radius(g)
    - length(g)
    - external_surface(g)
    - dist_to_tip(g)
    - biomass(g)
    - initial_C_sucrose_root(g)
    - initial_C_hexose_root(g)
    - initial_C_hexose_soil(g)
    """

    radius(g)
    length(g)
    surface_and_volume(g)
    dist_to_tip(g)
    biomass(g)
    initial_C_sucrose_root(g)
    initial_C_hexose_root(g)
    initial_C_hexose_soil(g)

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

######################################
# MODULE "DISTRIBUTION-TRANSFORMATION"
######################################

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def sucrose_to_hexose(g, time=1, unloading_apex=1.8e-5, gamma_unloading=0.4, Km_unloading=6e-6):

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
        # We calculate the unloading coefficient according to the distance from the apex:
        n.unloading_coeff = unloading_apex / (1 + n.dist_to_tip) ** gamma_unloading
        # We calculate the production of hexose (in mol) according to the Michaelis-Menten function:
        n.prod_hexose = 2. * n.external_surface * n.unloading_coeff * n.C_sucrose_root / (Km_unloading + n.C_sucrose_root) * time
        # The factor 2 originates from the conversion of 1 molecule of sucrose into 2 molecules of hexose.

    return g

# Calculating the new sugar concentrations inside and outside the roots in connections with other modules:
# --------------------------------------------------------------------------------------------------------
def balance(g, time,sucrose_input):

    """
    This function modifies the concentration of sucrose in the root and the concentrations of hexose
    inside and outside the root (in mol per gram of root dry biomass), depending on different processes:
    i) inputs of sucrose in the roots from the aerial parts (total_supply, in mol of sucrose per second)
    over time (in seconds) and homogeneous distribution of sucrose throughout the root phloem,
    so that the new concentration of sucrose is identical in all root vertices,
    ii) production of hexose from sucrose in the roots and removal by net exudation from the roots,
    iii) inputs of hexose in the soil by net exudation and degradation of hexose in the soil by microorganisms.
    """

    # We go through the MTG one first time to decrease sucrose concentrations in the root through phloem unloading:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)
        # The new sucrose concentration in the root is calculated
        # by substracting locally the sucrose unloaded from the phloem and transformed into hexose:
        n.C_sucrose_root = n.C_sucrose_root - (n.prod_hexose / 2.) / n.biomass

    # Then we calculate the remaining amount of sucrose in the root system:
    total_sucrose_root, total_biomass = total_root_sucrose_and_biomass(g)

    #--------------------------------------------------------------------------------------------------
    # USE OF THE FUNCTION "shoot_sucrose_supply" CALCULATED IN THE MODULE "EXCHANGE WITH AERIAL PARTS":
    #--------------------------------------------------------------------------------------------------
    # The input of sucrose "total_supply" is calculated:
    total_supply = shoot_sucrose_supply(sucrose_input)
    #---------------------------------------------------

    # The new sucrose concentration in the root is calculated by spreading the input of sucrose everywhere
    # in the root system:
    new_C_sucrose_root = (total_sucrose_root + total_supply * time) / total_biomass

    # We go through the MTG a second time to modify the sugars concentrations:
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)

        # The local sucrose concentration in the root is calculated from the new sucrose concentration calculated above:
        n.C_sucrose_root = new_C_sucrose_root
        # We create a new property "amount_sucrose_root" which corresponds to the amount of sucrose in the root element:
        n.amount_sucrose_root = n.C_sucrose_root * n.biomass

        # ----------------------------------------------------------------------------------------------
        # USE OF THE PROPERTY "hexose_exudation" CALCULATED IN THE MODULE "RHIZODEPOSITION":
        # ----------------------------------------------------------------------------------------------
        # The new hexose concentration in the root is calculated by adding to the initial root hexose concentration
        # the production of hexose by sucrose conversion and by subtracting the net exudation of hexose:
        n.C_hexose_root = n.C_hexose_root + n.prod_hexose / n.biomass - n.hexose_exudation / n.biomass
        #-----------------------------------------------------------------------------------------------

        # We create a new property "amount_hexose_root" which corresponds to the amount of hexose in the root element:
        n.amount_hexose_root = n.C_hexose_root * n.biomass
        # We create a new property "hexose_exudation_per_length" which corresponds to the amount of hexose (in mol)
        # exuded to the soil per length of root (in meter):
        n.hexose_exudation_per_length = n.hexose_exudation / n.length
        # We create a new property "hexose_exudation_per_surface" which corresponds to the amount of hexose (in mol)
        # exuded to the soil per surface of root (in square meter):
        n.hexose_exudation_per_surface = n.hexose_exudation / n.external_surface

        # ----------------------------------------------------------------------------------------
        # USE OF THE PROPERTY "hexose_exudation" CALCULATED IN THE MODULE "RHIZODEPOSITION":
        # USE OF THE PROPERTY "hexose_degradation" CALCULATED IN THE MODULE "SOIL TRANSFORMATION":
        # ----------------------------------------------------------------------------------------
        # The new hexose concentration in the soil is calculated by adding to the initial soil hexose concentration
        # the input of hexose by net exudation, and by subtracting the amount of hexose consumed by microorganisms:
        n.C_hexose_soil = n.C_hexose_soil + n.hexose_exudation / n.biomass - n.hexose_degradation / n.biomass
        # -----------------------------------------------------------------------------------------
        # We create a new property "amount_hexose_soil" which corresponds to the amount of hexose outside the root element:
        n.amount_hexose_soil = n.C_hexose_soil * n.biomass

# Transforming the variables:
# ---------------------------
def modify_concentrations(g, time_step, sucrose_supply=1e-2, Km_unloading=6e-6, Imax=1.8e-6, Km_influx=6e-6, Dmax=1.8e-5, Km_degradation=6e-6):

    """
    This function calls several functions affecting the concentrations of sugars inside and outside the roots:
    1) hexose_degradation(g) [module "SOIL TRANSFORMATION"]
    2) hexose_exudation(g) [module "RHIZODEPOSITION"]
    3) sucrose_to_hexose(g) [module "DISTRIBUTION OF ROOT METABOLITES"]
    4) balance(g)
    """

    # -----------------------------------------
    # CALL OF THE MODULE "SOIL TRANSFORMATION":
    # -----------------------------------------
    soil_hexose_degradation(g, time_step, Dmax, Km_degradation)
    #------------------------------------------
    # -------------------------------------
    # CALL OF THE MODULE "RHIZODEPOSITION":
    # -------------------------------------
    hexose_exudation(g, time_step, Imax, Km_influx)
    #--------------------------------------

    sucrose_to_hexose(g, time_step, Km_unloading)
    balance(g, time_step, sucrose_input=sucrose_supply)

###############################################################################################################

######################################
# MODULE "EXCHANGE WITH AERIAL PARTS"
######################################

# Calculating the net input of sucrose by the aerial parts into the root system:
# ------------------------------------------------------------------------------
def shoot_sucrose_supply(total_supply=5e-6):

    """
    This function calculates the net input of sucrose (mol of sucrose per day) into the root system [very simply for the moment!].
    """

    return total_supply

###############################################################################################################

##########################
# MODULE "RHIZODEPOSITION"
##########################

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def hexose_exudation(g,time=1, Pmax_apex=6.67e-3, gamma_exudation=0.4, Imax=1.8e-6,Km_influx=6e-6):

    """
    The function "hexose_exudation" computes the net amount (in mol of hexose) of hexose accumulated
    outside the root over time (in seconds), without considering any degradation process of hexose
    outside the root. This net amount corresponds to the difference between the efflux of hexose from the root
    to the soil by a passive diffusion and the influx of hexose from the soil to the root.
    The efflux by diffusion is calculated from the product of the root external surface (m2),
    the permeability coefficient (g m-2) and the gradient of hexose concentration (mol per gram of dry root
    structural biomass). The influx of hexose is represented as an active process with a substrate-limited
    relationship (Michaelis-Menten function), where Imax (in mol) is the maximal influx, and Km_influx
    (in mol per gram of root structural biomass) represents the hexose concentration for which
    hexose_degradation is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # We calculate the permeability coefficient P according to the distance from the apex:
        n.permeability_coeff = Pmax_apex / (1 + n.dist_to_tip) ** gamma_exudation
        # hexose_exudation is calculated by the difference between the efflux (by diffusion)
        # and the influx (by Michaelis-Menten) and defined as a new property of the MTG:
        n.hexose_exudation = n.external_surface * (n.permeability_coeff * (n.C_hexose_root - n.C_hexose_soil)
                                  - Imax * n.C_hexose_soil / (Km_influx + n.C_hexose_soil))* time

###############################################################################################################

##############################
# MODULE "SOIL TRANSFORMATION"
##############################

# Degradation of hexose in the soil (microbial consumption):
# ---------------------------------------------------------
def soil_hexose_degradation(g, time=1, degradation_max=1.8e-5, Km_degradation=6e-6):

    """
    The function "hexose_degradation" computes the decrease of the concentration of hexose outside the root (in mol of
    hexose per gram of root structural biomass) over time (in seconds). It mimics the uptake of hexose by rhizosphere
    microorganisms, and is therefore described using a substrate-limited function (Michaelis-Menten). g represents the
    MTG     describing the root system, degradation_max is the maximal degradation of hexose (mol m-2),
    and Km_degradation (mol per gram of root structural biomass) represents the hexose concentration for which the rate
    of hexose_degradation is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # hexose_degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.hexose_degradation = n.external_surface * degradation_max * n.C_hexose_soil / (Km_degradation + n.C_hexose_soil) * time

#######################################################################################################################

#########################
# MODULE "ACTUAL GROWTH"
#########################

def maintenance(g,time=1, resp_maintenance_max=1, Km_maintenance=1):

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
        # We calculate the permeability coefficient P according to the distance from the apex:
        n.resp_maintenance = resp_maintenance_max * n.C_hexose_root / (Km_maintenance + n.C_hexose_root) * n.biomass * time
    return g

########################################################################################################################

# MAIN PROGRAM:
###############

# Defining the function "simulation":
# -----------------------------------
# The function "simulation" will run the model over the whole simulation period to be defined by the user:
def simulation(g, simulation_period_in_days = 10, time_step_in_seconds = 60*60*24, daily_sucrose_input = 60. / 12. * 1e-6, parameter_displayed='hexose_exudation'):

    # We initiate the properties of the MTG "g" with the function "init":
    init(g)

    # We calculate the initial profile of sugars in the MTG, and we display the results on screen:
    time_in_days = 0.
    sucrose, biomass = total_root_sucrose_and_biomass(g)
    sucrose_input_per_day=0.
    print "Total root sucrose at", format(time_in_days, ".2g"),"day is", format(sucrose, ".2g"),"mol with a supply of", format(sucrose_input_per_day,".2g"), "mol of sucrose per day."

    # We calculate the adequate number of steps in the iteration below based on the length of the simulation period and the time step:
    n_steps_max = (simulation_period_in_days) * 60 * 60 * 24 / time_step_in_seconds + 1

    # We do an iteration for each time step:
    for step in range(1,n_steps_max):
        time_in_days = step * time_step_in_seconds / (60. * 60. * 24.)
        #Assuming that the input of C to the root is 60 umol of sucrose per day, knowing that one mol of sucrose corresponds to 12 mol of C:
        sucrose_input_per_day = daily_sucrose_input
        # We convert this into an input of sucrose in mol per second:
        sucrose_input_per_second = sucrose_input_per_day / (60.* 60.* 24.)
        #We modifiy the concentration for this step according to the elapsed time from the previous step and the sucrose_input_per_second:
        modify_concentrations(g, time_step=time_step_in_seconds, sucrose_supply=sucrose_input_per_second)
        sc = plot(g, prop_cmap=parameter_displayed)
        pgl.Viewer.display(sc)
        sucrose,biomass=total_root_sucrose_and_biomass(g)
        print "Total root sucrose at", format(time_in_days, ".2g"),"days is", format(sucrose, ".2g"),"mol with a supply of", format(sucrose_input_per_day,".2g"), "mol of sucrose per day."

        # The following code line enables to wait for 0.2 second between each iteration:
        time.sleep(0.2)
        # Alternative - The following code line asks the user to press enter at the end of each iteration:
        # message=input("Press enter to continue: ")

    # For preventing the interpreter to close all windows at the end of the program:
    raw_input()

# Generating the MTG:
# -------------------
# We create a MTG with n vertices:
n= 50
g = generate_mtg(n)

# Running "simulation":
# --------------------
# We can run this program in Conda or in PyCharm with the corresponding conda environment,
# then enter the following command with possibility of modifying some parameters:
simulation(g, simulation_period_in_days=15, time_step_in_seconds=60*60*24, daily_sucrose_input = 60. / 12. * 1e-6, parameter_displayed='hexose_exudation')
