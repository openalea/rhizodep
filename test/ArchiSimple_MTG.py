# Todo: Put all units in SI
# Todo: Show apices as cones instead of cylinders
# Todo: Introduce tropism and directions x,y,z for each vertex (and modify the get_root_visitor accordingly)

# Importation of functions from the system:
###########################################

from math import sqrt, pi, trunc, floor
import time
import random
import numpy as np
import matplotlib

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl

# Setting the randomness of the MTG geometrical properties:
random.seed(3)

######################################################################
# DEFINING FUNCTIONS FOR DISPLAYING THE MTG IN A 3D GRAPH WITH PLANTGL
######################################################################

def get_root_visitor():
    def root_visitor(g, v, turtle):
        n = g.node(v)
        # For displaying the radius 3 times larger than in reality:
        radius = n.radius * 1
        length = n.length * 1
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
    """ Compute a color property based on a given property and a colormap.

    """
    prop = g.property(property_name)
    keys = prop.keys()
    values = np.array(prop.values())
    # m, M = int(values.min()), int(values.max())

    _cmap = color.get_cmap(cmap)
    norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
    values = norm(values)
    # my_colorbar(values, _cmap, norm)

    colors = (_cmap(values)[:, 0:3]) * 255
    colors = np.array(colors, dtype=np.int).tolist()

    g.properties()['color'] = dict(zip(keys, colors))
    return g


def prepareScene(scene, width=1600, height=1200, dist_factor=-10):
    dir = -pgl.Viewer.camera.getPosition()[1]
    # dir.normalize()
    # pgl.Viewer.start()
    # pgl.Viewer.animation( True )
    pgl.Viewer.frameGL.maximize(True)
    # pgl.Viewer.widgetGeometry.setSize(width, height)
    # pgl.Viewer.display(scene)
    bbox = pgl.BoundingBox(scene)
    bbox.lowerLeftCorner = [-10, -10, -10]
    bbox.upperRightCorner = [-10, 10, -10]
    # pgl.Viewer.grids.set(True,True,True,True)
    # pgl.Viewer.camera.setOrthographic()
    d_factor = max(bbox.getXRange(), bbox.getYRange(), bbox.getZRange())

    # pgl.Viewer.camera.position=[0, 0, 0.6]
    pgl.Viewer.camera.lookAt(bbox.getCenter() + dir * (-dist_factor) * d_factor, bbox.getCenter())
    pgl.Viewer.frameGL.setSize(width, height)

    return dir


def plot_mtg(g, prop_cmap='radius', cmap='jet', lognorm=False):
    visitor = get_root_visitor()

    turtle = turt.PglTurtle()
    turtle.down(180)
    scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)

    # Compute color from radius
    color.colormap(g, prop_cmap, cmap=cmap, lognorm=lognorm)

    shapes = dict((sh.getId(), sh) for sh in scene)

    colors = g.property('color')
    for vid in colors:
        if vid in shapes:
            n=g.node(vid)
            # If the node is not dead, its color is displayed according to prop_cmap:
            if n.type!="Dead":
                # shapes[vid].appearance = pgl.Material(colors[vid])
                shapes[vid].appearance = pgl.Material()
            else:
                # Otherwise, if the node is dead, then its color is set to black:
                shapes[vid].appearance = pgl.Material([255,0,0])
    scene = pgl.Scene(shapes.values())
    return scene

# def plot_mtg(g, prop_cmap='hexose_exudation', cmap='jet', lognorm=True, vmin=1e-12, vmax=3e-7):
#     visitor = get_root_visitor()
#
#     turtle = turt.PglTurtle()
#     turtle.down(180)
#     scene = turt.TurtleFrame(g, visitor=visitor, turtle=turtle, gc=False)
#     prepareScene(scene)
#
#     # Compute color from the property "cmap" and the min and max values:
#     my_colormap(g, prop_cmap, cmap=cmap, vmin=vmin, vmax=vmax, lognorm=lognorm)
#
#     shapes = dict((sh.getId(), sh) for sh in scene)
#
#     colors = g.property('color')
#     for vid in colors:
#         if vid in shapes:
#             n = g.node(vid)
#             # If the element is not dead:
#             if n.type != "Dead":
#                 # We color it according to the property cmap defined by the user:
#                 shapes[vid].appearance = pgl.Material(colors[vid])
#             else:
#                 # Otherwise, we print it in black:
#                 shapes[vid].appearance = pgl.Material([0, 0, 0])
#             # If the node is not dead, its color is displayed according to prop_cmap:
#             # if n.type!="Dead":
#             # shapes[vid].appearance = pgl.Material([0,0,200])
#             # # else:
#             #     # Otherwise, if the node is dead, then its color is set to black:
#             #     shapes[vid].appearance = pgl.Material([200,0,0])
#             # if n.label=="Apex":
#             #     shapes[vid].appearance = pgl.Material([0, 200, 0])
#     scene = pgl.Scene(shapes.values())
#     return scene


###########################
# SETTING GROWTH PARAMETERS
###########################

# Emission of adventious/seminal roots:
# --------------------------------------
# Maximal number of adventitious roots (including primary)(dimensionless):
MNP = 5
# Emission rate of adventious roots (day-1):
ER = 0.5
# Tip diameter of the emitted root(s) (mm):
Di = 0.5

# Elongation:
# ------------

# Slope of the potential elongation rate versus tip diameter (in mm mm-1 day-1):
EL = 5.
# Threshold tip diameter below which there is no possible elongation (diameter of the finest roots)(in mm):
Dmin = 0.05
# Coefficient of growth duration (in day mm-2):
GDs = 400.
# Gravitropism (dimensionless):
G = 1.
# Delay of emergence of the primordium (in days):
emergence_delay = 3.

# Branching:
# -----------
# Inter-primordium distance (in mm):
IPD = 7.6
# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
RMD = 0.3
# Relative variation of the daughter root diameter (dimensionless):
CVDD = 0.30

# Radial growth:
# ---------------
# Proportionality coefficient between section area of the segment and the sum of distal section areas (dimensionless):
SGC = 0.1

# Self-pruning:
# --------------
# Coefficient of the life duration (in day mm-1 g-1 cm3):
LDs = 5000.
# Root tissue density (in g cm-3):
RTD = 0.1

# Length of a segment (in mm):
segment_length = 6. # !!!WATCH OUT: THE SEGMENT LENGTH SHOULD ALWAYS BE SMALLER THAN THE PARAMETER IPD !!!!!!!!!!!!!!!!

##############################################
# GENERATION OF INITAL APICES OF PRIMARY ROOTS
##############################################

# The following function initalizes a root system with apices and segments, without any length:

def initiate_mtg():
    g = MTG()
    # We first add one initial apex:
    root = g.add_component(g.root, label='Segment')
    segment = g.node(root)
    segment.type = "Normal_root_after_emergence"
    segment.length = 0.
    segment.radius = Di
    segment.angle_down = 0.
    segment.angle_roll = random.randint(70, 110)
    segment.growth_duration = GDs * segment.radius * segment.radius * 4
    segment.life_duration = LDs * 2 * segment.radius * RTD
    segment.dist_to_ramif = 0.
    segment.time_since_primordium_formation = emergence_delay
    segment.time_since_emergence = 0.
    segment.time_since_growth_stopped = 0.
    segment.time_since_death = 0.

    # If there should be more than one main root (i.e. adventious roots formed at the basis):
    if MNP > 1:
        # Then we form the apices of all possible adventious roots which could emerge:
        for i in range(1, MNP):
            radius_adventious = Di * 0.6
            # We add one new primordium on the previously defined segment:
            apex_adventious = segment.add_child(edge_type='+',
                                                label='Apex',
                                                type="Adventious_root_before_emergence",
                                                time_since_last_adventious=0.,
                                                # The length of the primordium is set to 0:
                                                length=0.,
                                                # The radius of the primordium is Di:
                                                radius=radius_adventious,
                                                angle_down=random.randint(30, 60),
                                                angle_roll=random.randint(0, 5),
                                                growth_duration=GDs * radius_adventious ** 2 * 4,
                                                life_duration=LDs * 2 * segment.radius * RTD,
                                                # The time since this primordium formation is set to 0:
                                                time_since_primordium_formation=0.,
                                                time_since_emergence=0.,
                                                time_since_growth_stopped=0.,
                                                time_since_death=0.,
                                                # The distance between the root apex and this new primordium is set to 0:
                                                dist_to_ramif=0.)
            # And we add one new segment without any length after the previous segment,
            # so that the angle_roll can be changed for the next adventious axis:
            segment = segment.add_child(edge_type='<',
                                        label='Segment',
                                        type="Support_for_adventious_root",
                                        time_since_last_adventious=0.,
                                        # The length of the primordium is set to 0:
                                        length=0.,
                                        # The radius of the primordium is Di:
                                        radius=segment.radius,
                                        angle_down=random.randint(0, 3),
                                        angle_roll=random.randint(70, 110),
                                        growth_duration=GDs * segment.radius ** 2 * 4,
                                        life_duration=LDs * 2 * segment.radius * RTD,
                                        # The time since this primordium formation is set to 0:
                                        time_since_primordium_formation=0.,
                                        time_since_emergence=0.,
                                        time_since_growth_stopped=0.,
                                        time_since_death=0.,
                                        dist_to_ramif=0.)

    # Finally, we add the apex that is going to develop the main axis:
    apex = segment.add_child(edge_type='<',
                             label='Apex',
                             type="Normal_root_before_emergence",
                             length=0.,
                             radius=Di,
                             angle_down=0.,
                             angle_roll=random.randint(110, 130),
                             growth_duration=GDs * segment.radius ** 2 * 4,
                             life_duration=LDs * 2 * segment.radius * RTD,
                             dist_to_ramif=0.,
                             time_since_primordium_formation=emergence_delay,
                             time_since_emergence=0.,
                             time_since_growth_stopped=0.,
                             time_since_death=0.)

    return g

#################################
# DEFINITION OF THE GROWTH MODEL
#################################

# FUNCTION 1: A GIVEN APEX CAN FORM A PRIMORDIUM ON ITS SURFACE
###############################################################

def primordium_formation(apex):

    # We initialize the new_apex that will be returned by the function:
    new_apex = []

    # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
    # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
    # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014):
    potential_radius = np.random.normal((apex.radius-Dmin) * RMD + Dmin, ((apex.radius-Dmin) * RMD + Dmin )* CVDD)
    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:
    if apex.dist_to_ramif >= IPD and potential_radius >= Dmin:
        # We get the order of the current root axis:
        vid = apex.index()
        order = g.order(vid)
        # The angle of the new axis to be formed is calculated according to the order of the parent axis:
        if order == 1:
            primordium_angle_down = random.randint(35, 55)
        else:
            primordium_angle_down = random.randint(60, 80)
        # A specific rolling angle is attributed to the parent apex:
        apex.angle_roll = random.randint(110, 130)
        # Then we add a primordium of a lateral root at the base of the apex:
        ramif = apex.add_child(edge_type='+',
                               label='Apex',
                               type='Normal_root_before_emergence',
                               # The length of the primordium is set to 0:
                               length=0.,
                               radius=potential_radius,
                               angle_down=primordium_angle_down,
                               angle_roll=random.randint(0, 5),
                               growth_duration=GDs * potential_radius * potential_radius * 4,
                               life_duration=LDs * 2. * potential_radius * RTD,
                               # The actual time elapsed since the formation of this primordium is calculated
                               # according to the actual growth of the parent apex since formation:
                               time_since_primordium_formation=(apex.length - (apex.dist_to_ramif - IPD))
                                                               / (EL * 2. * apex.radius),
                               time_since_emergence=0.,
                               time_since_growth_stopped=0.,
                               time_since_death=0,
                               # The distance between this root apex and the last ramification is set to zero:
                               dist_to_ramif=0.)

        # And the new distance between the parent apex and the last ramification is diminished,
        # by taking into account the potential elongation of apex since the child formation:
        apex.dist_to_ramif = EL * 2. * apex.radius * ramif.time_since_primordium_formation
        # We add the apex and its ramif in the list of apices returned by the function:
        new_apex.append(apex)
        new_apex.append(ramif)

        #print "The primordium", ramif.index(), "has been formed on node", apex.index(), "."

    return new_apex

# FUNCTION 2: A GIVEN APEX CAN ELONGATE AND CAN BE TRANSFORMED INTO SEGMENTS; USE OF FUNCTION 1
###############################################################################################

def apex_elongation(apex, elongation_time):

    # We initialize the new_apex that will be returned by the function:
    new_apex = []

    # ELONGATION: Elongation is calculated following the rules of Pages et al. (2014):
    elongation = EL * 2. * apex.radius * elongation_time

    # SEGMENTATION:
    # If the length of the apex is smaller than the defined length of a root segment:
    if apex.length + elongation < segment_length:
        # Then the apex is elongated:
        apex.length += elongation
        # The distance between the tip the last ramification is increased:
        apex.dist_to_ramif += elongation

        #print "The apex", apex.index(), "has been elongated without forming any segment."

        # And we call the function primordium_formation to check whether a primordium should have been formed
        # (Note: we assume that the segment length is always smaller than the inter-branching distance IBD,
        # so that in this case, only 0 or 1 primordium may have been formed - the function is called only once):
        new_apex.append(primordium_formation(apex))

    else:
        # Otherwise, we have to calculate the number of entire segments within the apex.
        # If the final length of the apex does not correspond to an entire number of segments:
        if (apex.length + elongation) / segment_length - floor(apex.length / segment_length) > 0.:
            # Then the total number of segments to be formed is:
            n_segments = floor((apex.length + elongation) / segment_length)
        else:
            # Otherwise, the number of segments to be formed is decreased by 1,
            # so that the last element corresponds to an apex with a positive length:
            n_segments = floor((apex.length + elongation) / segment_length) - 1

        n_segments=int(n_segments)
       # print "The apex", apex.index(), "has been elongated while forming", n_segments, "segment(s)."

        # We keep in memory the initial length of the apex:
        initial_length = apex.length
        # We develop each new segment, except the last one:
        for i in range(1, n_segments):
            # We define the length of the present element as the constant length of a segment:
            apex.length = segment_length
            # We define the new dist to ramif of the current apex once it has elongated until reaching the length of a segment:
            apex.dist_to_ramif += segment_length
            # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
            new_apex.append(primordium_formation(apex))
            # The current apex that has been elongated up to segment_length is now considered as a segment:
            apex.label = 'Segment'
            # And we add a new apex after this segment, initially of length 0:
            apex = apex.add_child(edge_type='<',
                                  label='Apex',
                                  type='Normal_root_after_emergence',
                                  length=0.,
                                  radius=apex.radius,
                                  angle_down=random.randint(0, 10),
                                  angle_roll=random.randint(0, 5),
                                  time_since_primordium_formation=apex.time_since_primordium_formation,
                                  time_since_emergence=apex.time_since_emergence,
                                  time_since_growth_stopped=0.,
                                  growth_duration=apex.growth_duration,
                                  life_duration=apex.life_duration,
                                  dist_to_ramif=apex.dist_to_ramif + segment_length)
        # Finally, we do this operation one last step:
        apex.length = segment_length
        # We define the new dist to ramif of the current apex once it has elongated until reaching the length of a segment:
        apex.dist_to_ramif += segment_length
        # We call the function that can add a primordium on the current apex depending on the new dist_to_ramif:
        new_apex.append(primordium_formation(apex))
        # And the element is now considered as a segment:
        apex.label = 'Segment'
        # And we define a new apex after the new defined segment, with a new length defined as:
        new_length = initial_length + elongation - n_segments * segment_length
        apex = apex.add_child(edge_type='<',
                              label='Apex',
                              type='Normal_root_after_emergence',
                              length=new_length,
                              radius=apex.radius,
                              angle_down=random.randint(0, 10),
                              angle_roll=random.randint(0, 5),
                              time_since_primordium_formation=apex.time_since_primordium_formation,
                              time_since_emergence=apex.time_since_emergence,
                              time_since_growth_stopped=0.,
                              growth_duration=apex.growth_duration,
                              life_duration=apex.life_duration,
                              dist_to_ramif=apex.dist_to_ramif + new_length)
        # And we call the function primordium_formation to check whether a primordium should have been formed
        new_apex.append(primordium_formation(apex))
        # And we add the last apex present at the end of the elongated axis:
        new_apex.append(apex)

    return new_apex

# FUNCTION 3: A GIVEN APEX IS DEVELOPED ACCORDING TO VARIOUS RULES; USE OF FUNCTION 2
#####################################################################################

def apex_development(apex, time_step=1.):

    new_apex=[]

    # CASE 1: THE APEX IS ALREADY DEAD
    if apex.type == "Dead":
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        apex.time_since_growth_stopped += time_step
        apex.time_since_death += time_step
        # The new element returned by the function corresponds to this apex:
        new_apex.append(apex)
        return new_apex

    # CASE 2: THE APEX MUST DIE
    # If the time since the growth of the apex stopped is higher than its prescribed life duration:
    if apex.time_since_growth_stopped + time_step >= apex.life_duration:
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        apex.time_since_growth_stopped += time_step
        apex.time_since_death = apex.time_since_growth_stopped + time_step - apex.life_duration
        # Then the apex is declared "dead":
        apex.type = "Dead"
        # # If needed: we can set its length and radius as 0:
        # apex.radius = 0.
        # apex.length = 0.
        # The new element returned by the function corresponds to this apex:
        new_apex.append(apex)
        return new_apex

    # CASE 3: THE APEX MUST STOP ITS GROWTH
    # If the time since the apex has emerged is higher than the prescribed growth duration:
    if apex.time_since_emergence + time_step > apex.growth_duration:
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        # If the growth had not been stopped before:
        if apex.type != "Stopped":
            apex.type = "Stopped"
            # Then the exact time since growth stopped is calculated:
            apex.time_since_growth_stopped = apex.time_since_emergence + time_step - apex.growth_duration
            # And the apex has been elongated before growth stopped:
            apex.length += EL * 2. * apex.radius * (time_step - apex.time_since_growth_stopped)
        else:
            # Otherwise, the time since growth stopped is simply increased by one time step:
            apex.time_since_growth_stopped += time_step
        # The new element returned by the function corresponds to this apex:
        new_apex.append(apex)
        return new_apex

    # CASE 4: THE APEX CORRESPONDS TO AN EMERGING ADVENTIOUS ROOT
    # An external variable describing the time elapsed since the emergence of the last adventious root is used:
    global time_since_last_adventious
    # If the adventious root has not emerged yet and if the emergence is possible for this time step:
    if apex.type == "Adventious_root_before_emergence":
        # If the time elapsed since the last emergence of adventious root
        # is higher than the prescribed frequency of adventious root emission:
        if time_since_last_adventious + time_step >= 1. / ER:
            apex.time_since_primordium_formation += time_step
            # The adventious root must have emerged, and the actual time elapsed
            # since its emergence over this time step is calculated as:
            apex.time_since_emergence = time_since_last_adventious + time_step - 1. / ER
            # We reset the external time counter at this elapsed time:
            time_since_last_adventious = apex.time_since_emergence
            # The root type is now defined as "Normal":
            apex.type = "Normal_root_after_emergence"
            # And the new axis is then elongated like any other axis:
            new_apex.append(apex_elongation(apex, elongation_time=apex.time_since_emergence))
            # And the function returns this new apex and stops here:
            return new_apex
        else:
            # The new element returned by the function corresponds to this apex:
            new_apex.append(apex)
            return new_apex

    # CASE 5: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A LATERAL ROOT ON A NORMAL ROOT SEGMENT
    if apex.type == "Normal_root_before_emergence":
        # If the time since primordium formation is higher than the delay of emergence:
        if apex.time_since_primordium_formation + time_step >= emergence_delay:
            # Then the root has emerged and the apex type is changed:
            apex.type = "Normal_root_after_emergence"
            apex.time_since_primordium_formation += time_step
            # The actual time elapsed since the emergence at the end of this time step is calculated:
            apex.time_since_emergence = apex.time_since_primordium_formation + time_step - emergence_delay
            # And the new axis is then elongated like any other axis:
            new_apex.append(apex_elongation(apex, elongation_time=apex.time_since_emergence))
            # And the function returns this new apex and stops here:
            return new_apex
        else:
            # Otherwise the time since primordium formation is simply increased by the time step:
            apex.time_since_primordium_formation += time_step
            # The new element returned by the function corresponds to this apex:
            new_apex.append(apex)
            return new_apex

    # CASE 6: THE APEX BELONGS TO AN AXIS THAT HAS ALREADY EMERGED AND CAN CONTINUE ITS GROWTH:
    if apex.type == "Normal_root_after_emergence":
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        # The new axis is simply elongated:
        new_apex.append(apex_elongation(apex, elongation_time=time_step))
        # And the function returns this new apex and stops here:
        return new_apex

    print "WATCH OUT! No case found for apex", apex.index(),"of type", apex.type

# FUNCTION 4: A GIVEN SEGMENT CAN GET THICKER OR DIE
####################################################

def segment_development(segment, radial_growth=True):
    new_segment = []
    son_section=0.
    sum_of_lateral_sections = 0.
    death_count = 0.

    # For each child of the segment:
    for child in segment.children():
        # If the child belongs to the same axis:
        if child.properties()['edge_type']=='<':
            # Then we record the section of this child:
            son_section = child.radius * child.radius * pi
        # Otherwise if the child is the segment of a lateral root AND if this lateral root has already emerged:
        elif child.properties()['edge_type']=='+'and child.length>0.:
            # We add the section of this child to a sum of lateral sections:
            sum_of_lateral_sections += child.radius * child.radius * pi

        # If the child is dead:
        if child.type == "Dead":
            # Then we add one dead child to the death count:
            death_count += 1

    # If each child in the list of children has been recognized as dead:
    if death_count == len(segment.children()):
        # Then the segment becomes dead:
        segment.type = "Dead"
        # Otherwise, at least one of the children axis is not dead, so the father segment should not be dead

    if radial_growth == True:
        # The radius of the root segment is defined according to the pipe model.
        # In ArchiSimp9, the radius is increased by considering the sum of the sections of all the children,
        # by adding a fraction (SGC) of this sum of sections to the current section of the parent segment,
        # and by calculating the new radius that corresponds to this new section of the parent:
        segment.radius = sqrt(son_section / pi + SGC * sum_of_lateral_sections / pi)

    new_segment.append(segment)
    return new_segment


###########################
# SIMULATION OF ROOT GROWTH
###########################

# We define a class "Simulate" that will modify the MTG for each time step:

class Simulate(object):

    # We initiate the object with a list of root apices:
    def __init__(self, g):
        """ Simulate on MTG. """
        self.g = g
        # We define the apices_list of g that contains all vertices labelled as "Apex":
        self.apices_list = [g.node(v) for v in g.vertices_iter(scale=1) if g.label(v)=='Apex']
        # We define the segments_list of g that contains all vertices labelled as "Segments", ranked in a post order:
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = root_gen.next()
        self.segments_list = [g.node(v) for v in post_order(g, root) if g.label(v) == 'Segment']

    def step(self, time_step):
        list_of_apices=list(self.apices_list)
        list_of_segments=list(self.segments_list)
        print "The root system contains", len(list_of_apices) + len(list_of_segments),"different elements."

        # For each apex in the list of apices:
        for apex in list_of_apices:
            new_apices = []
            # We define the new list of apices with the function apex_development:
            new_apices.append(apex_development(apex, time_step))
            # We add these new apices to apex:
            self.apices_list.extend(new_apices)
        # For each segment in the list of segments:
        for segment in list_of_segments:
            new_segments = []
            # We define the new list of apices with the function apex_development:
            new_segments.append(segment_development(segment, radial_growth=True))
            # We add these new apices to apex:
            self.segments_list.extend(new_segments)


##############
# MAIN PROGRAM
##############

# We initiate the properties of g as the simulated MTG of the root system:
g = initiate_mtg()

# We initiate a global variable that controls the possibility of emergence of an adventious root:
time_since_last_adventious = 0.

# ACTUAL SIMULATION OVER TIME:
time_step = 5
final_time_in_days = 60.
n_step = int(final_time_in_days / time_step)

# We do an iteration for each time step:
for step in range(1, n_step + 1):
    print "From t =", (step-1) * time_step, "days to t =", step * time_step, "days:"

    # We simulate the development of all apices and segments in the MTG:
    simulator = Simulate(g)
    simulator.step(time_step=time_step)

    # The global variable time elapsed since the emergence of the last adventious root is increased by the time step:
    time_since_last_adventious += time_step

    # We display the results for this time step:
    sc = plot_mtg(g)
    pgl.Viewer.display(sc)
    # The following code line enables to wait for 0.2 second between each iteration:
    time.sleep(0.2)

# We save the final MTG:
import pickle
with open('g_file.pckl', 'wb') as output:
    pickle.dump(g, output, protocol=2)

# # To open again the MTG file and reading it as "g":
# f = open('g_file.pckl', 'rb')
# g = pickle.load(f)
# f.close()
#
# sc = plot_mtg(g)
# pgl.Viewer.display(sc)

# We ask to press enter before closing the graph:
raw_input()