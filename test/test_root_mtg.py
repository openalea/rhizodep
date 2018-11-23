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

# g = MTG()

# # Sucrose variation / time step
# delta_C = 1.
# for vid in g.vertices(scale=1):
#     nid = g.node(vid)
#     nid.C_ext = 0.
#     nid.C_int += delta_C


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

def generate_mtg():
    g = MTG()
    root = g.add_component(g.root)
    vid = random_tree(g, root, nb_children=2, nb_vertices=20)
    return g

def tips(g):
    return [vid for vid in g.vertices_iter(scale=1) if g.is_leaf(vid)]

def diameter(g):
    """ Compute diameter of a RSA (Root System Architecture).

    .. todo:: Change radius to diameter
    """
    r=0.01
    alpha = 0.7
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()
    radius = {}
    for vid in post_order(g, root):
        if g.is_leaf(vid):
            order = g.order(vid)
            radius[vid] = r*(alpha)**order
        else:
            radius[vid] = sqrt(sum((radius[cid])**2 for cid in g.children(vid)))

    g.properties()['radius'] = radius
    return g

def length(g, length=0.05):

    _length = dict((vid, length) for vid in g.vertices_iter(scale=1))

    g.properties()['length'] = _length
    return g


def dist_to_tips(g):
    to_tips = {}
    length = g.property('length')
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = root_gen.next()
    for vid in post_order(g, root):
        if g.is_leaf(vid):
            to_tips[vid] = 0.
        else:
            son_id = g.Successor(vid)
            to_tips[vid] = to_tips[son_id] + length[son_id]

    g.properties()['dist_to_tips'] = to_tips
    return g


def permeability_coeff(g, max_permeability=86400., gamma=0.4):
    dist = g.property('dist_to_tips')

    def eqn(vid):
        return max_permeability/(0.01+dist[vid])**gamma

    perm = dict( (vid, eqn(vid)) for vid in g.vertices_iter(scale=1))
    g.properties()['permeability'] = perm

def biomass(g, density=0.1e6):
    """ Biomass : volume * density

    """
    radius = g.property('radius')
    length = g.property('length')

    def biom(vid):
        return pi*radius[vid]**2 * length[vid] * density

    _biomass = dict((vid, biom(vid)) for vid in g.vertices_iter(scale=1))

    g.properties()['biomass'] = _biomass


def C_sucrose_root(g, C_ref=0.3e-3):
    sucrose = dict((vid, C_ref) for vid in g.vertices_iter(scale=1))

def C_hexose_root(g):
    pass

def C_hexose_soil(g):
    pass

##################


###################

def get_root_visitor():
    def root_visitor(g, v, turtle):
        angles = [90,45]+[30]*5

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
    #color.colormap(g,prop_cmap, cmap=cmap, lognorm=lognorm)

    #shapes = dict((sh.getId(),sh) for sh in scene)

    #colors = g.property('color')
    #for vid in colors:
    #    if vid in shapes:
    #        shapes[vid].appearance = pgl.Material(colors[vid])
    #scene = pgl.Scene(shapes.values())
    return scene

################

# Main

g = generate_mtg()
diameter(g)
length(g)
dist_to_tips(g)
permeability_coeff(g)
biomass(g)

