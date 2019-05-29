# Todo: Introduce tropism and directions x,y,z for each vertex (and modify the get_root_visitor accordingly)
# Todo: Put all units in SI
# Todo: Show apices as cones instead of cylinders
# Todo: Check what is really done for seminal, adventious and aerial roots in ArchiSimple

# Importation of functions from the system:
###########################################

from math import sqrt, pi, trunc
import time
import random
import numpy as np

from openalea.mtg import *
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
from openalea.mtg.traversal import pre_order, post_order
import openalea.plantgl.all as pgl

# Defining functions for diplaying the root system in a 3D graph in PlantGL:
############################################################################

def get_root_visitor():
    def root_visitor(g, v, turtle):

        n = g.node(v)
        # For displaying the radius 3 times larger than in reality:
        radius = n.radius*3
        length = n.length
        angle_down = n.angle_down
        angle_roll = n.angle_roll

        # Moving the turtle:
        turtle.down(angle_down)
        turtle.rollL(angle_roll)
        turtle.setId(v)
        turtle.setWidth(radius)
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

###########################
# SETTING GROWTH PARAMETERS
###########################

# Emission of adventious/seminal roots:
#--------------------------------------
# Maximal number of adventitious roots (including primary)(dimensionless):
MNP=1
# Emission rate of adventious roots (day-1):
ER=0.5
# Tip diameter of the emitted root(s) (mm):
Di=0.5

# Elongation:
#------------

# Slope of the potential elongation rate versus tip diameter (in mm mm-1 day-1):
EL=5.
# Threshold tip diameter below which there is no possible elongation (diameter of the finest roots)(in mm):
Dmin=0.05
# Coefficient of growth duration (in day mm-2):
GDs=400.
# Gravitropism (dimensionless):
G=1.
# Delay of emergence of the primordium (in days):
emergence_delay=3.

# Branching:
#-----------
# Inter-primordium distance (in mm):
IPD=7.6
# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
RMD=0.3
# Relative variation of the daughter root diameter (dimensionless):
CVDD=0.15

# Radial growth:
#---------------
# Proportionality coefficient between section area of the segment and the sum of distal section areas (dimensionless):
SGC=1.

# Self-pruning:
#--------------
# Coefficient of the life duration (in day mm g-1):
LDs=5000.
# Root tissue density (in g cm-3):
RTD=0.1

# Length of a segment (in mm):
segment_length= 3.

#################################
# DEFINITION OF THE GROWTH MODEL
#################################

adventious_root_emergence_for_this_time_step="Possible"
time_since_last_adventious=0.

# Main function of apical development:
def apex_development(apex, time_step=1.):

    # The result of this function, new_apex, is initialized as empty:
    new_apex = []

    # CASE 1: THE APEX IS ALREADY DEAD
    if apex.type=="Dead":
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence+=time_step
        apex.time_since_growth_stopped += time_step
        apex.time_since_death += time_step
        # The new element returned by the function corresponds to this apex:
        new_apex.append(apex)
        # The function stops here and the new apex is returned:
        return new_apex

    # CASE 2: THE APEX MUST DIE
    # If the time since the growth of the apex stopped is higher than its prescribed life duration:
    if apex.time_since_growth_stopped + time_step >= apex.life_duration:
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        apex.time_since_growth_stopped += time_step
        apex.time_since_death = apex.time_since_growth_stopped + time_step - apex.life_duration
        # Then the apex is declared "dead" and has no radius or length:
        apex.type = "Dead"
        apex.radius = 0.
        apex.length = 0.
        # The new element returned by the function corresponds to this apex:
        new_apex.append(apex)
        # The function stops here and the new apex is returned:
        return new_apex

    # CASE 3: THE APEX MUST STOP ITS GROWTH
    # If the time since the apex has emerged is higher than the prescribed growth duration:
    if apex.time_since_emergence + time_step > apex.growth_duration:
        apex.type="Stopped"
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        # If the growth has just been stopped:
        if apex.time_since_growth_stopped == 0:
            # Then the exact time since growth stopped is calculated:
            apex.time_since_growth_stopped=apex.time_since_emergence + time_step - apex.growth_duration
            # And the apex has been elongated before growth stopped:
            apex.length += EL * 2. * apex.radius * (time_step - apex.time_since_growth_stopped)
        # Otherwise, the time since growth stopped is increased by one time step:
        else:
            apex.time_since_growth_stopped+=time_step
        # The new element returned by the function corresponds to this apex:
        new_apex.append(apex)
        # The function stops here and the new apex is returned:
        return new_apex

    # CASE 4: THE APEX CORRESPONDS TO AN EMERGING ADVENTIOUS ROOT
    # External variables are declared within the function:
    # 1) a variable allowing the emergence of an adventious root:
    global adventious_root_emergence_for_this_time_step
    # 2) a variable describing the time elapsed since the emergence of the last adventious root:
    global time_since_last_adventious
    # If the adventious root has not emerged yet and if the emergence is possible for this time step:
    if apex.type == "Adventious_root_before_emergence" and adventious_root_emergence_for_this_time_step=="Possible":
        # If the time elapsed since the last emergence of adventious root
        # is higher than the prescribed frequency of adventious root emission:
        if time_since_last_adventious + time_step >= 1. / ER:
            apex.time_since_primordium_formation += time_step
            # The adventious root must have emerged, and the actual time elapsed
            # since its emergence over this time step is calculated as:
            apex.time_since_emergence = time_since_last_adventious + time_step - 1. / ER
            # The length of the emerged apex is calculated according to the normal elongation rate EL:
            apex.length = EL * 2. * apex.radius * apex.time_since_emergence
            apex.dist_to_ramif= apex.length
            # And the root type is now defined as "Normal":
            apex.type = "Normal_root_after_emergence"
            # We reset the external time counter at this elapsed time:
            time_since_last_adventious = apex.time_since_emergence
            # And we prevent any other adventious root to emerge at the same time step:
            adventious_root_emergence_for_this_time_step = "Impossible"
            # The new element returned by the function corresponds to this apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

    # CASE 5: THE APEX CORRESPONDS TO THE PRIMORDIUM OF A LATERAL ROOT ON A NORMAL ROOT SEGMENT
    if apex.type == "Normal_root_before_emergence":
        # If the time since primordium formation is higher than the delay of emergence:
        if apex.time_since_primordium_formation + time_step >= emergence_delay:
            # Then the root has emerged and the apex type is changed:
            apex.type="Normal_root_after_emergence"
            apex.time_since_primordium_formation += time_step
            # The actual time elapsed since the emergence at the end of this time step is calculated:
            apex.time_since_emergence=apex.time_since_primordium_formation + time_step - emergence_delay
            # The corresponding elongation of the apex is calculated:
            elongation = EL * 2. * apex.radius * apex.time_since_emergence
            apex.length += elongation
            apex.dist_to_ramif += elongation
            # And the new element returned by the function corresponds to the same apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex
        else:
            # Otherwise the time since primordium formation is simply increased by the time step:
            apex.time_since_primordium_formation += time_step
            # And the new element returned by the function corresponds to the same apex:
            new_apex.append(apex)
            # And the function returns this new apex and stops here:
            return new_apex

    # CASE 6: THE APEX CAN FORM A PRIMORDIUM ON ITS SURFACE
    # We first calculate the radius that the primordium may have. This radius is drawn from a normal distribution
    # whose mean is the value of the mother root diameter multiplied by RMD, and whose standard deviation is
    # the product of this mean and the coefficient of variation CVDD (Pages et al. 2014):
    potential_radius = np.random.normal(apex.radius * RMD, apex.radius * RMD * CVDD)
    # If the distance between the apex and the last emerged root is higher than the inter-primordia distance
    # AND if the potential radius is higher than the minimum diameter:
    if apex.dist_to_ramif >= IPD and potential_radius >= Dmin:
        # We get the order of the current root axis:
        vid = apex.index()
        order = g.order(vid)
        # The angle of the new axis to be formed is calculated according to the order of the parent axis:
        if order ==1:
            primordium_angle_down = random.randint(35,55)
        else:
            primordium_angle_down = random.randint(60,80)
        # A specific rolling angle is attributed to the parent apex:
        apex.angle_roll = random.randint(110, 130)
        # Then we add a primordium of a lateral root at the base of the apex:
        ramif = apex.add_child(edge_type='+',
                               label='Apex',
                               type='Normal_root_before_emergence',
                               # The length of the primordium is set to 0:
                               length=0.,
                               radius=potential_radius,
                               angle_down = primordium_angle_down,
                               angle_roll = random.randint(0,5),
                               growth_duration=GDs * potential_radius * potential_radius * 4,
                               life_duration=LDs * potential_radius * potential_radius * 4 * RTD,
                               # The actual time elapsed since the formation of this primordium is calculated
                               # according to the actual growth of the parent apex since formation:
                               time_since_primordium_formation=(apex.length-(apex.dist_to_ramif - IPD))
                                                               /(EL * 2. * apex.radius),
                               time_since_emergence=0.,
                               time_since_growth_stopped=0.,
                               time_since_death=0,
                               # The distance between this root apex and the last ramification is set to zero:
                               dist_to_ramif=0.)
        # And the new apex now contains the primordium of a lateral root:
        new_apex.append(ramif)
        # And the new distance between the element and the last ramification is set to 0:
        apex.dist_to_ramif = 0.

    # CASE 7: THE APEX CAN ELONGATE AND CAN BE TRANSFORMED INTO SEGMENTS
    if apex.type == "Normal_root_after_emergence":
        apex.time_since_primordium_formation += time_step
        apex.time_since_emergence += time_step
        # ELONGATION OF THE APEX:
        # Elongation is calculated following the rules of Pages et al. (2014):
        elongation = EL * 2. * apex.radius * time_step
        # If the length of the apex is smaller than the defined length of a root segment:
        if apex.length + elongation < segment_length:
            # Then the apex is elongated:
            apex.length += elongation
            apex.dist_to_ramif+= elongation
        else:
            # Otherwise, we calculate the number of entire segments within the apex.
            # If the length of the apex does not correspond to an entire number of segments:
            if apex.length / segment_length - trunc(apex.length / segment_length) > 0.:
                # Then the total number of segments to be formed is:
                n_segments = trunc(apex.length / segment_length)
            else:
                # Otherwise, the number fo segments to be formed is decreased by 1,
                # so that the last element corresponds to an apex with a positive length:
                n_segments = trunc(apex.length / segment_length) - 1
            initial_length = apex.length
            for i in range(1,n_segments):
                # We define the length of the present element as the length of a segment:
                apex.length = segment_length
                # The element is now considered as a segment:
                apex.label = 'Segment'
                # And we add a new apex after this segment with the length of a segment:
                apex=apex.add_child(edge_type='<',
                                    label='Apex',
                                    type='Normal_root_after_emergence',
                                    length=0.,
                                    radius=apex.radius,
                                    angle_down=random.randint(0, 3),
                                    angle_roll=random.randint(0, 5),
                                    time_since_primordium_formation=apex.time_since_primordium_formation,
                                    time_since_emergence=apex.time_since_emergence,
                                    time_since_growth_stopped=0.,
                                    growth_duration=apex.growth_duration,
                                    life_duration=apex.life_duration,
                                    dist_to_ramif=apex.dist_to_ramif + segment_length)
            #Finally, we do this operation one last step:
            apex.length = segment_length
            # And the element is now considered as a segment:
            apex.label = 'Segment'
            # And we define a new apex after the new defined segment, with a new length defined as:
            new_length=initial_length + elongation - n_segments*segment_length
            apex.add_child(edge_type='<',
                           label='Apex',
                           type='Normal_root_after_emergence',
                           length=new_length,
                           radius = apex.radius,
                           angle_down=random.randint(0, 3),
                           angle_roll=random.randint(0, 5),
                           time_since_primordium_formation=apex.time_since_primordium_formation,
                           time_since_emergence=apex.time_since_emergence,
                           time_since_growth_stopped=0.,
                           growth_duration=apex.growth_duration,
                           life_duration=apex.life_duration,
                           dist_to_ramif = apex.dist_to_ramif + new_length)
            # The new apex is defined as the modified apex:
            new_apex.append(apex)

    return new_apex

# Main function of radial growth and segment death:
def segment_development(segment, time_step=1.):

    new_segment=[]
    sum_sections = 0.
    death_count = 0.

    # For each child of the segment:
    for child in segment.children():

        # We add the section of the child to a sum of children sections:
        sum_sections += child.radius * child.radius * pi

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
        # Otherwise, at least one of the children axis is not dead, so the father segment should not be dead

    # The radius of the root segment is defined according to the pipe model,
    # i.e. the radius is equal to the square root of the sum of the radius^2 of all children
    # proportionally to the coefficient SGC:
    segment.radius = sqrt(SGC * sum_sections / pi)

    new_segment.append(segment)
    return new_segment

###########################
# SIMULATION OF ROOT GROWTH
###########################

# We define a class "Simulate" which is used to simulate the development of apices and segments on the whole MTG "g":
class Simulate(object):

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

    def step(self, time_step=1.):
        g = self.g
        # We define "apices" and "segments" as the list of apices and segments in g:
        apices_list = list(self._apices)
        segments_list=list(self._segments)

        # For each apex in the list of apices:
        for apex in apices_list:
            # We define the new list of apices with the function apex_development:
            new_apex = apex_development(apex, time_step)
            # We add these new apices to apex:
            self._apices.extend(new_apex)

        # For each segment in the list of segments:
        for segment in segments_list:
            # We define the new list of apices with the function apex_development:
            new_segment = segment_development(segment, time_step)
            # We add these new apices to apex:
            self._segments.extend(new_segment)

##############################################
# GENERATION OF INITAL APICES OF PRIMARY ROOTS
##############################################

def mtg_root():
    g = MTG()
    # We first add one initial apex:
    root = g.add_component(g.root, label='Segment')
    segment = g.node(root)
    segment.type="Normal_root_after_emergence"
    segment.length = 0.
    segment.radius = Di
    segment.angle_down = 0.
    segment.angle_roll = random.randint(70,110)
    segment.growth_duration= GDs * segment.radius * segment.radius * 4
    segment.life_duration = LDs * segment.radius * segment.radius * 4 * RTD
    segment.dist_to_ramif = 0.
    segment.time_since_primordium_formation = emergence_delay
    segment.time_since_emergence=0.
    segment.time_since_growth_stopped=0.
    segment.time_since_death=0.

    # If there should be more than one main root (i.e. adventious roots formed at the basis):
    if MNP>1:
    # Then we form the apices of all possible adventious roots which could emerge:
        for i in range(1,MNP):
            radius_adventious = Di * 0.7
            # We add one new primordium on the previously defined segment:
            apex_adventious = segment.add_child(edge_type='+',
                                  label='Apex',
                                  type="Adventious_root_before_emergence",
                                  time_since_last_adventious=0.,
                                  # The length of the primordium is set to 0:
                                  length=0.,
                                  # The radius of the primordium is Di:
                                  radius=radius_adventious,
                                  angle_down = random.randint(30,60),
                                  angle_roll = random.randint(0,5),
                                  growth_duration = GDs * radius_adventious**2 * 4,
                                  life_duration = LDs * radius_adventious**2 * 4 * RTD,
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
                                  angle_down = random.randint(0,3),
                                  angle_roll = random.randint(70,110),
                                  growth_duration = GDs * segment.radius **2 * 4,
                                  life_duration = LDs * segment.radius **2 * 4 * RTD,
                                  # The time since this primordium formation is set to 0:
                                  time_since_primordium_formation=0.,
                                  time_since_emergence=0.,
                                  time_since_growth_stopped=0.,
                                  time_since_death=0.,
                                  dist_to_ramif=0.)
    
    # Finally, we add the apex that is going to develop the main axis:
    apex = segment.add_child(edge_type='<',
                             label='Apex',
                             type = "Normal_root_before_emergence",
                             length = 0.,
                             radius = Di,
                             angle_down = 0.,
                             angle_roll = random.randint(110, 130),
                             growth_duration = GDs * segment.radius **2 * 4,
                             life_duration = LDs * segment.radius **2 * RTD,
                             dist_to_ramif = 0.,
                             time_since_primordium_formation = emergence_delay,
                             time_since_emergence = 0.,
                             time_since_growth_stopped = 0.,
                             time_since_death = 0.)

    return g

##############
# MAIN PROGRAM
##############

# We initiate the properties of g as the simulated MTG of the root system:
g=mtg_root()

#ACTUAL SIMULATION OVER TIME:

time_step=1.
final_time_in_days=120.
n_step=int(final_time_in_days/time_step)

# We do an iteration for each time step:
for step in range(1, n_step+1):

    print("Day number:",step*time_step)
    # We specify that the emergence of a new adventious root is possible at the beginning of this time step:
    adventious_root_emergence_for_this_time_step = "Possible"

    # We simulate the development of all apices and segments in the MTG:
    simulator=Simulate(g)
    simulator.step(time_step)

    # The time elapsed since the emergence of the last adventious root is increased by the time step:
    time_since_last_adventious += time_step

    # We display the results for this time step:
    sc = plot_mtg(g)
    pgl.Viewer.display(sc)
    # The following code line enables to wait for 0.2 second between each iteration:
    time.sleep(0.2)

# We ask to press enter before closing the graph:
raw_input()