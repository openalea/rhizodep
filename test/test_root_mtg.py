# Loading all the necessary stuffs:
###################################

from math import sqrt, pi

from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from pylab import cm, colorbar
from pylab import plot as pylab_plot
from matplotlib.colors import Normalize, LogNorm

from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
import openalea.plantgl.all as pgl

from collections import deque

class Simulation(object):
    def __init__(self):
        g = self.g = MTG()
        self.root = g.add_component(g.root)
        self.apices = deque([self.root])

    def growth(self):
        new_apices = deque([])
        for aid in apices:
            new_apex = self.update_apex(aid)

    def update_apex(self):
        assert(aid in self.g)

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

# Generating a MTG describing the root system:
##############################################

# Defining a function for generating a MTG:
# -----------------------------------------
def generate_mtg(n=20):
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

# Defining some properties of the root system:
##############################################

# Defining the radius of an element:
# ----------------------------------
def radius(g, r=0.01,alpha=0.8):
    """
    The function "radius" computes the radius (in meter) of each root segment based on the pipe model,
    where the square of the radius of the root segment is equal to the sum of the square of the radius
    of all the children segments. A parameter alpha (adimensional) is also used to lower the radius
    of the apices depending on their root orders.
    """

    # We initialize an empty dictionary for radius:
    radius = {}
    # We define "root" as  the starting point of the loop below (i.e. the first apex in the MTG)(?):
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

# Defining the length of each element:
# ------------------------------------
def length(g, length=0.05):
    """
    The function "length" computes the length (in meter) of each individual root segment in the MTG "g".
    """

    # We create a dictionnary containing a value of "length" for each vertex in the MTG:
    _length = dict((vid, length) for vid in g.vertices_iter(scale=1))

    # We assign the result "length" as a new property of each vertex in the MTG "g":
    g.properties()['length'] = _length
    # We return a modified version of the MTG "g" with a new property "length":
    return g

# Defining the distance of a vertex from the tip:
# -----------------------------------------------
def dist_to_tips(g):
    """
    The function "dist_to_tips" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices
    [see the function "length"].
    """

    # We initialize an empty dictionnary for to_tips:
    to_tips = {}
    # We use the property "length" of each vertex based on the function "length":
    length = g.property('length')

    # We define "root" as  the starting point of the loop below (i.e. the first apex in the MTG)(?):
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

    # We assign the result "dist_to_tips" as a new property of each vertex in the MTG "g":
    g.properties()['dist_to_tips'] = to_tips
    # We return a modified version of the MTG "g" with a new property "dist_to_tips":
    return g

# Defining the evolution of the hexose permeability coefficient along the roots:
# ------------------------------------------------------------------------------
def permeability_coeff(g, max_permeability=6.67e-3, gamma=0.4):
    """
    The function "permeability-coeff" calculates the permeability coefficient of each root segment according
    to its distance from the apex of the corresponding root axis. The permeability coefficient (g m-2) is used
    in the equation describing the efflux of hexose from the root to the soil by a diffusion process.
    The evolution of the permeability coefficient P with the distance from the apex is based on the relationship
    suggested by Personeni et al. (2007): P = P_max / (1 + x)^gamma,
    where P_max is the maximal value of the permeability found in the root apex,
    x the distance from the apex and gamma an adimensional parameter.
    """

    # We use the property "dist_to_tips" of the MTG "g" describing the distance of the root segment from the apex
    # (in meter) defined by the function "dist_to_tips":
    dist = g.property('dist_to_tips')

    #We define a function "eqn" expressing the permeability coefficient as a function of the distance to the apex "dist":
    def eqn(vid):
        return max_permeability/(1+dist[vid])**gamma
    #We cover all the vertices in the MTG and assign the value "perm" using the equation "eqn":
    perm = dict( (vid, eqn(vid)) for vid in g.vertices_iter(scale=1))
    #We define the result as a new property "permeability_coeff" in the MTG "g":
    g.properties()['permeability_coeff'] = perm

# Defining the evolution of the unloading coefficient along the roots:
# --------------------------------------------------------------------
def unloading_coeff(g, max_unloading=1.8e-5, gamma=0.4):
    """
    The function "unloading_coeff" calculates the unloading coefficient of each root segment according to its distance
    from the apex of the corresponding root axis. The unloading coefficient (in mol per square meter, i.e. mol m-2)
    is used in the equation describing the influx of hexose by an active process. The evolution of the unloading
    coefficient U with the distance from the apex is based on the relationship suggested by Personeni et al. (2007)
    for the permeability coefficient (see "permeability_coeff"): U = U_max / (1 + x)^gamma,
    where U_max is the maximal rate of the unloading found in the root apex,
    x the distance from the apex and gamma an adimensional parameter.
    """

    # We use the property "dist_to_tips" of the MTG "g" describing the distance of the root segment from the apex
    # (in meter) defined by the function "dist_to_tips":
    dist = g.property('dist_to_tips')

    # We define a function "eqn" expressing the unloading rate as a function of the distance to the apex "dist":
    def eqn(vid):
        return max_unloading/(1+dist[vid])**gamma
    # We cover all the vertices in the MTG using the equation "eqn":
    unloading = dict( (vid, eqn(vid)) for vid in g.vertices_iter(scale=1))
    # We define the result as a new property in the MTG:
    g.properties()['unloading_coeff'] = unloading

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
    radius = g.property('radius')
    length = g.property('length')

    # We define a function "biom" expressing the dry structural biomass as a function of the radius and the length of a root segment:
    def biom(vid):
        #The function calculates the dry biomass as the product of the volume of a cylinder and the dry biomass density:
        return pi*radius[vid]**2 * length[vid] * density
    # We cover all the vertices in the MTG using the equation "biom" to calculate biomass:
    _biomass = dict((vid, biom(vid)) for vid in g.vertices_iter(scale=1))
    # We define the result as a new property in the MTG:
    g.properties()['biomass'] = _biomass

# Setting the initial sugar concentrations for each vertex with default values:
# ----------------------------------------------------------------------------
def initial_C_sucrose_root(g, C_sucrose_ref=0.6e-3):
    """
    This function sets the concentration of sucrose in the phloem of the root system, C_sucrose_root
    (in mol of sucrose per gram of dry root structural mass), to an initial default value.
    """
    sucrose = dict((vid, C_sucrose_ref) for vid in g.vertices_iter(scale=1))
    g.properties()['C_sucrose_root']=sucrose

def initial_C_hexose_root(g, C_hexose_root_ref=0.3e-3):
    """
    This function sets the concentration of hexose in the root of the root system, C_hexose_root
    (in mol of sucrose per gram of dry root structural mass), to an initial default value.
    """
    hexose_root = dict((vid, C_hexose_root_ref) for vid in g.vertices_iter(scale=1))
    g.properties()['C_hexose_root']=hexose_root

def initial_C_hexose_soil(g, C_hexose_soil_ref=0.3e-5):
    """
    This function sets the concentration of hexose outside the root of the root system, C_hexose_soil
    (in mol of sucrose per gram of dry root structural mass), to an initial default value.
    """
    hexose_soil = dict((vid, C_hexose_soil_ref) for vid in g.vertices_iter(scale=1))
    g.properties()['C_hexose_soil']=hexose_soil

# Degradation of hexose in the soil (microbial consumption):
# ---------------------------------------------------------
def hexose_degradation(g,Dmax=1.8e-5,Km_degradation=6e-6):
    """
    The function "hexose_degradation" computes the decrease (in mol of hexose) in the concentration of hexose
    outside the root. It mimics the uptake of hexose by rhizosphere microorganisms, and is therefore described
    using a substrate-limited function (Michaelis-Menten). g represents the MTG describing the root system,
    Dmax is the maximal degradation of hexose (mol m-2), and Km_degradation (mol per gram of root structural biomass)
    represents the hexose concentration for which hexose_degradation is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        #n represents the vertex:
        n = g.node(vid)
        #S represents the external surface (m2) of the root segment (cylinder) in contact with the soil:
        S = 2 * pi * n.radius * n.length
        #hexose_degradation is defined according to a Michaelis-Menten function as a new property of the MTG:
        n.hexose_degradation = S * Dmax * n.C_hexose_soil/(Km_degradation + n.C_hexose_soil)

# Unloading of sucrose from the phloem and conversion of sucrose into hexose:
# --------------------------------------------------------------------------
def sucrose_to_hexose(g, Km_unloading=6e-6):
    """
    The function "sucrose_to_hexose" simulates the process of sucrose unloading from phloem
    and its immediate conversion into hexose.     2 mol of hexose are produced for 1 mol of sucrose.
    The unloading of sucrose is represented as an active process with a substrate-limited relationship
    (Michaelis-Menten function), where Umax (in mol) is the maximal amount of sucrose unloaded
    and Km_unloading (in mol per gram of root structural biomass) represents the sucrose concentration
    for which sucrose_to_hexose is equal to half of its maximum.
    """

    # We use the properties "C_sucrose_root" and "unloading_coeff" of each vertex
    # defined by the functions "C_sucrose_root" and "unloading_coeff":
    C_suc_root = g.property('C_sucrose_root')
    Umax = g.property('unloading_coeff')
    # We use the properties "radius" and "length" of each vertex defined by the functions "radius" and "length":
    radius = g.property('radius')
    length = g.property('length')

    def eqn(vid):
        #If we include a proportion with the external surface of the root:
        form_hex = 2. * pi * radius[vid] * length[vid] * (2. * Umax[vid] * C_suc_root[vid] / (Km_unloading + C_suc_root[vid]))
        #This corresponds to a Michaelis-Menten function. The factor 2 originates from the conversion
        # of 1 molecule of sucrose into 2 molecules of hexose.
        # If we don't include a proportion with the external surface of the root:
        #form_hex = (2. * Umax[vid] * C_suc_root[vid] / (Km_unloading + C_suc_root[vid]))
        return form_hex

    # We define the result as a new property "prod_hexose" in the MTG:
    delta_hexose = dict((vid, eqn(vid)) for vid in g.vertices_iter(scale=1))
    g.properties()['prod_hexose'] = delta_hexose

# Exudation of hexose from the root into the soil:
# ------------------------------------------------
def hexose_exudation(g,Imax=1.8e-6,Km_influx=6e-6):
    """
    The function "hexose_exudation" computes the net amount (in mol of hexose) of hexose accumulated
    outside the root, without considering any degradation process of hexose outside the root. This net amount
    corresponds to the difference between the efflux of hexose from the root to the soil by a passive diffusion
    and the influx of hexose from the soil to the root. The efflux by diffusion is calculated from the product
    of the root external surface (m2), the permeability coefficient (g m-2) and the gradient of hexose concentration
    (mol per gram of dry root structural biomass). The influx of hexose is represented as an active process
    with a substrate-limited relationship (Michaelis-Menten function), where Imax (in mol) is the maximal influx
    and Km_influx(in mol per gram of root structural biomass) represents the hexose concentration for which
    hexose_degradation is equal to half of its maximum.
    """

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # S is the calculated external surface of the root segment considered as a cylinder:
        S = 2.0 * pi * n.radius * n.length
        # hexose_exudation is calculated by the difference between the efflux (by diffusion)
        # and the influx (by Michaelis-Menten) and defined as a new property of the MTG:
        n.hexose_exudation = S * (n.permeability_coeff * (n.C_hexose_root - n.C_hexose_soil)
                                  - Imax * n.C_hexose_soil / (Km_influx + n.C_hexose_soil))

#Calculation of the new concentration of sucrose which will be applied everywhere in the root system:
# ---------------------------------------------------------------------------------------------------
def total_root_sugars_and_biomass(g):
    """
    This function returns three numeric values, respectively:
    - the total amount of sucrose (total_sucrose_root, in mol of sucrose),
    - the total amount of hexose (total_hexose_root, in mol of hexose),
    - the total dry biomass of the root system (g).
    """

    # We initialize the three values to 0:
    total_sucrose_root = 0.
    total_hexose_root = 0.
    total_biomass = 0.

    # We cover all the vertices in the MTG:
    for vid in g.vertices_iter(scale=1):
        # n represents the vertex:
        n = g.node(vid)
        # We calculate the total amount of sugars by summing all the local products of concentrations with biomass:
        total_sucrose_root += (n.C_sucrose_root * n.biomass)
        total_hexose_root += (n.C_hexose_root * n.biomass)
        # We calculate the total biomass by summing all the local biomass:
        total_biomass += n.biomass
    # We return a list of three numeric values:
    return total_sucrose_root, total_hexose_root, total_biomass

# Balance of sucrose and hexose to calculate the new sugar concentrations:
# ------------------------------------------------------------------------
def balance(g, total_supply=10.):
    """
    This function modifies the concentration of sucrose in the root and the concentrations of hexose
    inside and outside the root (in mol per gram of root dry biomass), depending on different processes:
    i) inputs of sucrose in the roots from the aerial parts (total_supply, in mol of sucrose) and homogeneous
    distribution of sucrose throughout the root phloem so that the new concentration of sucrose is identical
    in all root vertices,
    ii) production of hexose from sucrose in the roots and removal by net exudation from the roots,
    iii) inputs of hexose in the soil by net exudation and degradation of hexose in the soil by microorganisms.
    """
    total_sucrose_root, total_hexose_root, total_biomass = total_root_sugars_and_biomass(g)
    for vid in g.vertices_iter(scale=1):
        n = g.node(vid)
        # The new sucrose concentration in the root is calculated by spreading the input of sucrose everywhere
        # in the root system, and by substracting locally the sucrose unloaded from the phloem and transformed into hexose:
        n.C_sucrose_root = (total_sucrose_root + total_supply) / total_biomass - (n.prod_hexose/2.) / n.biomass
        # The new hexose concentration in the root is calculated by adding to the initial root hexose concentration
        # the production of hexose by sucrose conversion and by substracting the net exudation of hexose:
        n.C_hexose_root = n.C_hexose_root + n.prod_hexose / n.biomass - n.hexose_exudation / n.biomass
        # The new hexose concentration in the soil is calculated by adding to the initial soil hexose concentration
        # the input of hexose by net exudation, and by substracting the amount of hexose consumed by microorganisms:
        n.C_hexose_soil = n.C_hexose_soil + n.hexose_exudation / n.biomass - n.hexose_degradation / n.biomass
        n.amount_hexose_soil = n.C_hexose_soil * n.biomass

####################################################################################################################

# Initializing the variables of the MTG:
# --------------------------------------
def init(g):
    """
    This function is used to initialize the properties of the MTG "g", by calling several functions
    and using their default parameters:
    - radius(g)
    - length(g)
    - dist_to_tips(g)
    - permeability_coeff(g)
    - unloading_coeff(g)
    - biomass(g)
    - initial_C_sucrose_root(g)
    - initial_C_hexose_root(g)
    - initial_C_hexose_soil(g)
    """

    radius(g)
    length(g)
    dist_to_tips(g)
    permeability_coeff(g)
    unloading_coeff(g)
    biomass(g)
    initial_C_sucrose_root(g)
    initial_C_hexose_root(g)
    initial_C_hexose_soil(g)

# Transforming the variables:
# ---------------------------
def modify_concentrations(g, total_supply):
    """
    This function calls several functions affecting the concentrations of sugars inside and outside the roots:
    - hexose_degradation(g)
    - hexose_exudation(g)
    - sucrose_to_hexose(g)
    - balance(g)
    """

    hexose_degradation(g)
    hexose_exudation(g)
    sucrose_to_hexose(g)
    balance(g, total_supply)

# Main program:
###############

# Generating the MTG:
# -------------------
# We create a MTG with n vertices:
n= 100
g = generate_mtg(n)
# We initiate the properties of the MTG, e.g. sugar concentrations:
init(g)

# We have the supply function decrease over time for a given MTG architecture:
for t in range(50):
    modify_concentrations(g, total_supply=100.* 1/(1+t/10000))
    sc = plot(g, prop_cmap='hexose_exudation')
    pgl.Viewer.display(sc)

