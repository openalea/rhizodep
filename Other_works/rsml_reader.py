import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# NEW CODE """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order
from openalea.mtg import turtle as turt
from openalea.mtg.plantframe import color
import openalea.plantgl.all as pgl

from rhizodep.model import recording_MTG_properties
from rhizodep.tools import my_colormap, plot_mtg
from rhizodep.alternative_plotting import plotting_roots_with_pyvista, fast_plotting_roots_with_pyvista
from math import sqrt, pi, floor, trunc, exp, isnan, cos, acos, sin
import pickle
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


""" 
RSML Reader, by Daniel Leitner (2019) 
"""

########################################################################################################################
# ORIGINAL CODE 'rsml_reader.py' taken from GitHub on 2024-07-19 at:
# https://github.com/RSA-benchmarks/collaborative-comparison/blob/master/rsml_reader.py
########################################################################################################################

def parse_rsml_(organ: ET, polylines: list, properties: dict, functions: dict, parent: int) -> (list, dict, dict):
    """ Recursivly parses the rsml file, used by read_rsml """
    for poly in organ.iterfind('geometry'):  # only one
        polyline = []
        for p in poly[0]:  # 0 is the polyline
            n = p.attrib
            newnode = [float(n['x']), float(n['y']), float(n['z'])]
            polyline.append(newnode)
        polylines.append(polyline)
        properties.setdefault("parent-poly", []).append(parent)

    for prop in organ.iterfind('properties'):
        for p in prop:  # i.e legnth, type, etc..
            try:
                properties.setdefault(str(p.tag), []).append(float(p.attrib['value']))
            except:
                properties.setdefault(str(p.tag), []).append(p.attrib['value'])

    for funcs in organ.iterfind('functions'):
        for fun in funcs:
            samples = []
            for sample in fun.iterfind('sample'):
                try:
                    samples.append(float(sample.attrib['value']))
                except:
                    samples.append(sample.attrib['value'])
            functions.setdefault(str(fun.attrib['name']), []).append(samples)

    pi = len(polylines) - 1
    for elem in organ.iterfind('root'):  # and all laterals
        polylines, properties, functions = parse_rsml_(elem, polylines, properties, functions, pi)

    return polylines, properties, functions


def read_rsml(name: str) -> (list, dict, dict):
    """Parses the RSML file into:

    Args:
    name(str): file name of the rsml file

    Returns:
    (list, dict, dict):
    (1) a (flat) list of polylines, with one polyline per root
    (2) a dictionary of properties, one per root, adds "parent_poly" holding the index of the parent root in the list of polylines
    (3) a dictionary of functions
    """
    root = ET.parse(name).getroot()
    plant = root[1][0]
    polylines = []
    properties = {}
    functions = {}
    for elem in plant.iterfind('root'):
        (polylines, properties, functions) = parse_rsml_(elem, polylines, properties, functions, -1)

    return polylines, properties, functions


def get_segments(polylines: list, props: dict) -> (list, list):
    """ Converts the polylines to a list of nodes and an index list of line segments

    Args:
    polylines(list): flat list of polylines, one polyline per root
    props(dict): dictionary of properties, one per root, must contain "parent-node", (and "parent-poly" that was added by read_rsml)

    Returns:
    (list, list):
    (1) list of nodes
    (2) list of two integer node indices for each line segment
    """
    nodes, offset, segs = [], [], []
    offset.append(0)  # global node index at each polyline
    for p in polylines:
        for n in p:
            nodes.append(n)
        offset.append(offset[-1] + len(p))
    for i, p in enumerate(polylines):
        ni = props["parent-node"][i]
        pi = props["parent-poly"][i]
        if (pi >= 0):
            segs.append([offset[pi] + ni, offset[i]])
        for j in range(0, len(p) - 1):
            segs.append([offset[i] + j, offset[i] + j + 1])
    return nodes, segs


def get_parameter(polylines: list, funcs: dict, props: dict) -> (list, list, list):
    """ Copies radii and creation times, one value per segment
    """
    fdiam = funcs["diameter"]
    fet = funcs["emergence_time"]
    ptype = props["type"]
    radii, cts, types = [], [], []
    for i, p in enumerate(polylines):
        for j in range(0, len(p)):
            radii.append(fdiam[i][j] / 2)
            cts.append(fet[i][j])
            types.append(ptype[i])
    return radii, cts, types


def plot_rsml(polylines: list, prop: list):
    """Plots the polylines in y-z axis with colors given by a root property

    Args:
    polylines(list): flat list of polylines, one polyline per root
    prop(list): a single property, list of scalar value, on per root
    """
    f = matplotlib.colors.Normalize(vmin=min(prop), vmax=max(prop))
    cmap = plt.get_cmap("jet", 256)
    for i, pl in enumerate(polylines):
        nodes = np.array(pl)
        plt.plot(nodes[:, 1], nodes[:, 2], color=cmap(f(prop[i])))
    plt.axis('equal')
    plt.show()


def plot_segs(nodes: list, segs: list, fun: list):
    """Plots the segments in y-z axis (rather slow)

    Args:
    nodes(list): list of nodes
    segs(list): list of two integer node indices for each line segment
    fun(list): a single function, list of scalar value, on per segment, see TODO
    """
    f = matplotlib.colors.Normalize(vmin=min(fun), vmax=max(fun))
    cmap = plt.get_cmap("jet", 256)
    for i, s in enumerate(segs):
        plt.plot([nodes[s[0], 1], nodes[s[1], 1]], [nodes[s[0], 2], nodes[s[1], 2]], color=cmap(f(fun[i])))
    plt.axis('equal')
    plt.show()

# NEW CODE """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def rsml_to_mtg(rsml_file_path,
                converting_to_csv=False, csv_file_path='rsml_data.csv',
                converting_to_Excel=False, Excel_file_path='rsml_data.xlsx'):

    # Example of a root system in RSML available on:
    # 1) Schnepf et al: https://github.com/RSA-benchmarks/collaborative-comparison/tree/master/M3%20Water%20flow%20in%20roots/M3.2%20Root%20system/root_grid
    # 2) Griffith et al.: https://doi.org/10.1002/ppj2.20036, https://zenodo.org/records/5504299/files/wheat_roots_code_data.zip?download=1

    polylines, properties, functions = read_rsml(rsml_file_path)

    # print("List of properties:")
    # for key, v in properties.items():
    #     print("   >", key, ": length =", len(properties[key]), "; values =", v)
    # print("List of functions:")
    # for key, v in functions.items():
    #     print("   >", key, ": length =", len(functions[key]), "; values =", v)
    # print("")

    # CONVERTING THE RSML FILE TO CSV OR EXCEL FILE:
    ################################################

    if converting_to_csv or converting_to_Excel:
        print("Recording a file from the RSML...")
        # We create an empty list that will contain all the data from the RSML file:
        rsml_data = []
        # We first add a list containing the index of each root axis:
        rsml_data.append(range(0, len(polylines)))
        # We then add the polylines values:
        rsml_data.append(polylines)
        # We cover each property and add its values:
        for property in properties.keys():
            rsml_data.append(properties[property])
        # We cover each function and add its values:
        for function in functions.keys():
            rsml_data.append(functions[function])

        # We create a temporary data frame (which will need to be transposed):
        temporary_df = pd.DataFrame(rsml_data)
        # We transpose the data frame to get the correct format:
        data_frame = temporary_df.T
        # We create a list containing the headers of the dataframe:
        column_names = ["root_axis", "polylines"]
        column_names.extend(properties.keys())
        column_names.extend(functions.keys())
        # We replace the column names in the first line of the dataframe:
        data_frame.columns = column_names
        # We record the dataframe as a csv or Excel file:
        if converting_to_csv:
            data_frame.to_csv(csv_file_path, sep=";", na_rep='NA', index=False, header=True)
            print("   > The information contained in the RSML file have been recorded in", csv_file_path)
        if converting_to_Excel:
            data_frame.to_excel(Excel_file_path, na_rep='NA', index=False, header=True)
            print("   > The information contained in the RSML file have been recorded in", Excel_file_path)

    # CREATING AN MTG FILE:
    #######################

    print("Transfering the RSML data to a new MTG...")
    # We create an empty MTG:
    g=MTG()
    # We define the first base element as an empty element:
    id_segment = g.add_component(g.root, label='Segment',
                                 x1=0,
                                 x2=0,
                                 y1=0,
                                 y2=0,
                                 z1=0,
                                 z2=0,
                                 radius1=0,
                                 radius2=0,
                                 radius=0,
                                 length=0
                                 )
    base_segment = g.node(id_segment)

    # We initialize an empty dictionary that will be used to register the vid of the mother elements:
    d = {}
    # We initialize the first mother element:
    mother_element = base_segment

    # For each root axis:
    for l, line in enumerate(polylines):
        # We initialize the first dictionary within the main dictionary:
        d[l] = {}
        last_known_radius = 0.
        # If the root axis is not the main one of the root system:
        if l > 0:
            # We define the mother element of the current lateral axis according to the properties of the RSML file:
            parent_axis_index = properties["parent-poly"][l]
            parent_node_index = properties["parent-node"][l]
            mother_element = g.node(d[parent_axis_index][parent_node_index])
        # For each root element:
        for i in range(1,len(line)):
            # We define the x,y,z coordinates and the radius of the starting point:
            x1, y1, z1 = line[i-1]
            try:
                r1 = functions["diameter"][l][i - 1] / 2.
                last_known_radius = r1
            except:
                r1 = last_known_radius
            # We define the x,y,z coordinates and the radius of the ending point:
            x2, y2, z2 = line[i]
            try:
                r2 = functions["diameter"][l][i]/2.
                last_known_radius = r2
            except:
                r2 = last_known_radius
            # The length of the root element is calculated from the x,y,z coordinates:
            length=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            # We define the edge type ('<': adding a root element on the same axis, '+': adding a lateral root):
            if i==1 and l > 0:
                edgetype="+"
            else:
                edgetype="<"
            # We define the label (Apex or Segment):
            if i == len(line) - 1:
                label="Apex"
            else:
                label="Segment"
            # We finally add the new root element to the previously-defined mother element:
            new_child = mother_element.add_child(edge_type=edgetype,
                                                 label=label,
                                                 x1=x1,
                                                 x2=x2,
                                                 y1=y1,
                                                 y2=y2,
                                                 z1=z1,
                                                 z2=z2,
                                                 radius1=r1,
                                                 radius2=r2,
                                                 radius=(r1+r2)/2,
                                                 length=length)
            # We record the vertex ID of the current root element:
            vid = new_child.index()
            # We add the vid to the dictionary:
            d[l][i]=vid
            # And we now consider current element as the mother element for the next iteration on this axis:
            mother_element = new_child

    print("The MTG has been successfully created!")

    return g

# We shortly define an internal function 'norm' that we will use later on:
def norm(vector):
    try:
        norm_of_vector = sqrt((np.sum(vector ** 2)))
    except:
        print("Problem when calculating the norm for the vector", vector,"- the norm was set to None!")
        norm_of_vector = None
    return norm_of_vector

# We also define a function computing the projection of a vector u on a plane which normal vector is 'normal"
def projection_on_plane(u=np.array([1,1,1]), normal=np.array([1,1,1])):
    proj_of_u_on_plane = u - (np.dot(u, normal) / norm(normal) ** 2) * normal
    return proj_of_u_on_plane

def angle_between_two_vectors (u=np.array([1,0,0]), normal=np.array([0,1,0])):

    if norm(u)==0 or norm(normal)==0:
        # print("ERROR: no angle could be calculated between the vectors", u, "and", normal, "!!!")
        angle_in_degrees = None
    else:
        product = np.dot(u, normal) / (norm(u) * norm(normal))
        # We make sure that the arcosinus function will not receive a number lower than -1 or higher than 1:
        if product < -1.:
            angle_in_rad = pi
        elif product > 1:
            angle_in_rad = 0.
        else:
            # The angle is calculated in rad as:
            angle_in_rad = acos(product)

        # We finally convert the angle into degrees:
        angle_in_degrees = angle_in_rad * 180/pi

    return angle_in_degrees

def rotation(vector = np.array([1,1,1]), angle_yaw = 0., angle_pitch = 0., angle_roll = 0.):

    R_yaw = np.array([[cos(angle_yaw), -sin(angle_yaw), 0],
                     [sin(angle_yaw), cos(angle_yaw), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[cos(angle_pitch), 0, sin(angle_pitch)],
                      [0, 1, 0],
                      [-sin(angle_pitch), 0, cos(angle_pitch)]])

    R_roll = np.array([[1, 0, 0],
                      [0, cos(angle_roll), -sin(angle_roll)],
                      [0, sin(angle_roll), cos(angle_roll)]])

    R_full = np.matmul(np.matmul(R_yaw,R_pitch), R_roll)
    rotated_vector = np.matmul(R_full, vector)

    return rotated_vector

def calculating_angles(g, printing_details = False):

    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the base to the extremity:
    for vid in pre_order(g, root):

        if printing_details:
            print("")
            print("... Considering now the element", vid,"...")
        # We define the current element as n:
        n = g.node(vid)

        # We define whether angle_down should be calculated or not:
        try:
            n.angle_down = n.angle_down * 1.
            angle_down_exists = True
        except:
            angle_down_exists = False
        # We define whether angle_roll should be calculated or not:
        try:
            n.angle_roll = n.angle_roll* 1.
            angle_roll_exists = True
        except:
            angle_roll_exists = False

        # If the element corresponds to the first element of the root system from the base:
        if vid == 1 and not angle_down_exists:
            # Then we set its angle_down to 0:
            n.angle_down = 0.
            if printing_details:
                print("   > For the first element the angle down was set to 0.")

        # If the element has no length, we need to specify an angle_down:
        if n.length <= 0.:
            # If not, the element may correspond to a "flat" element that is here to adjust the angle_roll for a lateral
            # root, or to a primordium of a non-emerged lateral root.
            # If the angle down has not been defined yet, we attribute a nul angle down to it:
            if not angle_down_exists:
                n.angle_down = 0.
                if printing_details:
                    print("   > The angle down was set to 0 as the element", vid, "had no length and no pre-defined angle down!")
                # The angle roll will however depend on the calculation below.

        # If the angle roll has not been defined yet:
        if not angle_roll_exists:
            if printing_details:
                print("   > Calculating angle roll for the element", vid, "...")

            # We initialize a corrector of the final angle to take into account the possible intermediate elements
            # between the actual parent element and the current element:
            cumulative_angle_roll_from_parent = 0.

            if n.length <= 0.:
                # We need to look for a new "parent" vector to calculate the proper angle_roll to assign to the current
                # element. We want to get the closest parent element with a positive length.
                # We initialize some temporary variables:
                index = vid
                parent_id = -1
                parent_length = 0.
                # We loop to find the next parent with positive length:
                while parent_length <= 0. and parent_id is not None:

                    # We try to get the ID of the parent element of the current element on the same axis:
                    parent_id = g.Father(index, EdgeType='<')
                    if parent_id:
                        # We access the parent element:
                        parent = g.node(parent_id)
                        if printing_details:
                            print("[The angle_roll from parent element", parent_id,"was", parent.angle_roll,"]")
                        try:
                            cumulative_angle_roll_from_parent += parent.angle_roll
                            if printing_details:
                                print("[Cumulative angle roll between the actual parent and the current element is "
                                      "increased by", parent.angle_roll, "when considering the element", parent_id,"]")
                        except:
                            cumulative_angle_roll_from_parent += 0.
                    # If unsuccesful, we try to get the ID of the parent element of the current element on a mother axis:
                    if not parent_id:
                        parent_id = g.Father(index, EdgeType='+')
                    # If an actual parent element has been found (whatever its length)::
                    if parent_id:
                        # We access the parent element:
                        parent = g.node(parent_id)
                        # We record its vid for the next iteration of the loop:
                        index = parent_id
                        # Its length is put in memory:
                        parent_length = parent.length
                        # Now the loop will continue until getting a parent element with a positive length,
                        # or no parent element at all.
                # If a parent with positive length has been found:
                if parent_id:
                    # We define the parent's vector according to the [x,y,z] coordinates of the starting and ending point:
                    parent_vector = np.array([parent.x2 - parent.x1, parent.y2 - parent.y1, parent.z2 - parent.z1])
                else:
                    if printing_details:
                        print("ERROR: no angle roll could be calculated for element", vid, "!!!")

            # Otherwise, the current element has a positive length and will now be considered as the parent:
            else:
                parent = n
                # We define the parent's vector according to the [x,y,z] coordinates of the starting and ending point:
                parent_vector = np.array([n.x2 - n.x1, n.y2 - n.y1, n.z2 - n.z1])

            # We want to get all children of the current element, i.e. the next element on the axis
            # and every first element of lateral axes, if any:
            list_of_sons_id = []
            if g.Sons(vid, EdgeType='<') != []:
                list_of_sons_id.append(g.Sons(vid, EdgeType='<')[0])
            if g.Sons(vid, EdgeType='+') !=  []:
                list_of_sons_id.extend(g.Sons(vid, EdgeType='+'))
            # If the list is empty, then this is the last element of the axis (e.g. apex) and there should not be any
            # change in angle_roll and angle_down:
            if not list_of_sons_id:
                if printing_details:
                    print("   > No child was detected for element", vid,"and its angle roll was set to 0.")
                n.angle_roll = 0.
                if not angle_down_exists:
                    print("   > No child was detected for element", vid, "and its angle down was set to 0.")
                    n.angle_down = 0.
                # And we go to the next element in the loop:
                continue
            else:
                if printing_details:
                    print("   > The initial list of children for element", vid, "was:", list_of_sons_id)
            # Otherwise, we will base the following calculations on the respective directions of the parent
            # and its child(ren).

            # We initialize a counter that will be used to assess whether intermediate elements need to be
            # built for adjusting the roll angle:
            counter_for_adding_flat_parents = 0

            # For each child element of the current element:
            for child_id in list_of_sons_id:
                if printing_details:
                    print("   > Considering now the child", child_id,"...")
                # We access the child element:
                child = g.node(child_id)
                # In case the child does not have a positive length:
                if child.length <= 0.:
                    if printing_details:
                        print("   > The child", child_id, "has no length!")
                    # We initialize two checkers and one index:
                    next_child_length = 0.
                    next_child_id = 0
                    index = child_id
                    # We keep searching for the first (grand) child with positive length on the same axis:
                    while next_child_length <= 0. and next_child_id is not None:
                        # We try to get the ID of the first son of the current child on the same axis:
                        next_child_id = g.Successor(index)
                        # If there is a child:
                        if next_child_id is not None:
                            # We access this child element:
                            next_child = g.node(next_child_id)
                            # We record its vid for the next iteration of the loop:
                            index = next_child_id
                            # Its length is put in memory:
                            next_child_length = next_child.length
                            # Now the loop will continue until getting a child element with a positive length,
                            # or no child at all.
                    # After the loop, if a child with positive length has been found:
                    if next_child_id:
                        if printing_details:
                            print("   > The child", child_id, "was replaced by the child", next_child_id, "that has a positive length!")
                        # Then we now consider this child for the subsequent calculation:
                        child = next_child
                    # Otherwise, if there is no child with positive length:
                    else:
                        if printing_details:
                            print("   > No further child with positive length was found after the child", child_id, "!")
                        try:
                            n.angle_roll =  n.angle_roll * 1.
                            print("   > The existing angle_roll for the current element", vid, "was not modified.")
                        except:
                            n.angle_roll = 0.
                            print("   > The angle_roll for the current element", vid, "was set to 0.")
                        # And we move to the next child in the list without considering the calculations below:
                        continue

                # We define the direction vector of the final child:
                child_vector = np.array([child.x2 - child.x1, child.y2 - child.y1, child.z2 - child.z1])

                # a) Calculating angle_roll:
                # --------------------------
                # If the direction of the parent and that of the child vector are identical, their cross product will be
                # null. For checking this, we verify whether the norm of this cross product is below a minimum value:
                if norm(np.cross(parent_vector, child_vector)) < 1e-20:
                    # In such case, their is no new angle_roll and it remains 0:
                    angle_roll = 0.
                # Otherwise, the two vectors are not colinear and we can calculate angles.
                else:
                    # FIRST, WE GET THE ABSOLUTE VALUE OF ANGLE ROLL.
                    # The 'angle_roll' is the angle between the orientation of the parent (i.e. the vector 'w' of the
                    # turtle on the parent) and the projection of the vector of the child element on the plane that is
                    # orthogonal to the direction of the parent element (i.e., the plane which has a normal vector that
                    # is precisely the parent's vector).
                    # The parent w-direction should have been previously recorded, but, in the case of the first element
                    # at the base of the root system, we can initialize the parent-w direction:
                    try:
                        parent.rolling_reference_x = parent.rolling_reference_x * 1.
                    except:
                        if printing_details:
                            print("   > For element", vid, "we initialized a direction w as the vector (1,0,0).")
                        parent.rolling_reference_x = 1.
                        parent.rolling_reference_y = 0.
                        parent.rolling_reference_z = 0.
                    # We get the vector corresponding to direction w of the parent:
                    parent_w = np.array([parent.rolling_reference_x,
                                         parent.rolling_reference_y,
                                         parent.rolling_reference_z])
                    # Now we calculate the projected child vector on the parent's normal plane:
                    projected_vector = projection_on_plane(child_vector, normal=parent_vector)
                    # Finally, we compute the angle_roll from this (in degrees):
                    angle_roll = angle_between_two_vectors(parent_w, projected_vector)
                    if angle_roll is None:
                        if printing_details:
                            print("ERROR: the rolling angle between the vector", parent_w, "from element", vid,
                                  "and the projected vector", projected_vector, "from child", child.index(),
                                  "could not be calculated! We set this angle to 0 by default.")
                        angle_roll = 0.
                    else:
                        if printing_details:
                            print("   > SUCCESS: the rolling angle of element", vid, "between the vector", parent_w,
                                  "from element", parent.index(), "and the projected vector", projected_vector,
                                  "from child", child.index(),
                                  "was calculated as", angle_roll)
                    # SECOND, WE DETERMINE THE SIGN OF THE ANGLE ROLL.
                    # We look at the cross product of the two vectors to assess the sign of the angle:
                    cross_product = np.cross(parent_w, projected_vector)
                    # If the cross product points to the same direction as the parent vector, the angle should be
                    # positive, otherwise it should be negative. We sum the two vectors normalized vectors and check
                    # whether the resulting vector has a norm smaller or larger than the initial parent-vector:
                    vector_sum = parent_vector + cross_product
                    if norm(vector_sum) < norm(parent_vector):
                        # Then the two vectors must have had opposite directions; in such case, the angle roll should
                        # be negative:
                        angle_roll = -angle_roll
                    else:
                        # Then the two vectors must have had a similar direction; in such case, the angle roll should remain positive:
                        angle_roll = angle_roll

                # At this stage, we check that the parent has an angle roll (this should be the case):
                try:
                    parent.angle_roll = parent.angle_roll * 1.
                    parent_angle_roll_exists = True
                except:
                    parent_angle_roll_exists = False

                # If there have been already multiple elements of length zero inserted in the MTG prior to angle calculations:
                if cumulative_angle_roll_from_parent != 0. and parent_angle_roll_exists and angle_roll != 0.:
                    # We adjust the angle_roll:
                    if printing_details:
                        print("   > SPECIAL CASE: The angle roll for element", vid,
                              "which was originally calculated as", angle_roll, "was lowered by",
                              (cumulative_angle_roll_from_parent - parent.angle_roll),
                              "degrees to take into account intermediate flat elements between the parent and the current element!")
                    angle_roll -= (cumulative_angle_roll_from_parent - parent.angle_roll)
                    #TODO:
                    angle_roll -= (cumulative_angle_roll_from_parent)

                # If there has been already a change in angle_roll on the parent when considering the insertion of a
                # first child and if the current element is not a flat element:
                if counter_for_adding_flat_parents >= 1 and n.length > 0.:
                    # Then we insert a "flat" parent of length 0 that will simply be used to adjust the angle_roll
                    # in adequation with the new child:
                    new_flat_parent = child.insert_parent(edge_type='+',
                                                          label = "Base_of_element_for_angle_roll",
                                                          radius = n.radius,
                                                          length = 0.,
                                                          angle_down = 0.,
                                                          angle_roll = angle_roll - parent.angle_roll)
                    if printing_details:
                        print("   > A flat parent element was inserted before element", child.index(),"to account for angle_roll.")
                        print("   > SUCCESS: The angle roll for the new 'flat' element", new_flat_parent.index(), "was",
                              angle_roll - parent.angle_roll,"after correction by the angle", parent.angle_roll,
                              "from the parent element", parent.index())
                else:
                    # Otherwise, the angle_roll is directly associated to the current element.
                    # However, if the "parent" considered above was distinct from the current element, the angle roll
                    # of the current element will need to take into account the change in angle roll between the remote
                    # parent and the current element:
                    if parent != n and parent_angle_roll_exists:
                        # TODO
                        # n.angle_roll = angle_roll - parent.angle_roll
                        n.angle_roll = angle_roll
                        # if printing_details:
                        #     print("   > SUCCESS: The angle roll for element", vid, "was computed as", n.angle_roll, "after correction by the angle", parent.angle_roll,
                        #           "from the actual parent element", parent.index())
                    else:
                        n.angle_roll = angle_roll
                        if printing_details:
                            print("   > SUCCESS: The angle roll for element", vid, "was eventually computed as", angle_roll)
                # In any case, we increment the counter for the possible next round:
                counter_for_adding_flat_parents += 1

                # b) Calculating the new reference orientation w of the vector:
                #--------------------------------------------------------------

                # We also need to record the new rolling reference direction for the child, so that a
                # similar later calculation is made possible for its own children.
                # This rolling reference is actually the vector that is orthogonal to the vector of the child and that
                # is included within the plane formed by the parent's and the child's vector.
                # The latter plane can be defined by the normal vector of the plane that is orthogonal to the two
                # previous vectors, which corresponds to the cross product of the two previous vectors:
                normal_vector = np.cross(parent_vector, child_vector)
                # The direction w of the child is then orthogonal to the normal_vector of the plane and the vector of
                # the child:
                rolling_reference = np.cross(normal_vector, child_vector)

                # The orientation w for the current element is finally recorded:
                child.rolling_reference_x = rolling_reference[0]
                child.rolling_reference_y = rolling_reference[1]
                child.rolling_reference_z = rolling_reference[2]

                # c) Calculating the absolute value of angle_down:
                #-------------------------------------------------
                # Mathematically, an angle between two vectors a and b can be written as:
                # αngle = arccos[ a.b / (norm(a) * norm(b))].
                # The 'angle_down' is the angle between the direction of the parent and the direction of the current node:
                if norm(parent_vector) == 0. or norm(child_vector) == 0.:
                    if printing_details:
                        print("ERROR: the angle_down for vid", vid, "could not be calculated and was set to 0!")
                    child.angle_down = 0.
                else:
                    # We calculate the scalar product of the two vectors divided by the product of their respective norm:
                    product = np.dot(parent_vector, child_vector) / (norm(parent_vector) * norm(child_vector))
                    # We make sure that the arcosinus function will not receive a number lower than -1 or higher than 1:
                    if product < -1.:
                        angle_down = pi
                    elif product > 1:
                        angle_down = 0.
                    else:
                        # The angle_down is calculated in rad as:
                        angle_down = acos(product)

                    # We now convert angle_down into deg and attribute it to the child:
                    angle_down = angle_down * 180/pi
                    child.angle_down = angle_down
                    if printing_details:
                        print("   > SUCCESS: The angle down between the element", parent.index(),"and the child",
                              child.index(),"was determined as", angle_down,"and attributed to the child.")
        else:
            print("   > The angle roll had already been computed for the element", vid,"and no further computation was attempted.")

    return

# This function enables to plot a MTG with PlantGL:
def plot_root_MTG(g,
                   width=1200, height=900,
                   x_center=0, y_center=-0, z_center=0.,
                   x_cam=5., y_cam=5., z_cam=-1,
                   grid=True,
                   background_color=[94, 76, 64],
                   single_color=None,
                   property_name='radius', cmap='jet', vmin=1e-4, vmax=1e-2, lognorm=True,
                   recording_plot=True,
                   plot_name='plot.png'):

    # Sub-function for moving the turtle and create specific shapes for stems and leaves:
    def get_root_visitor():
        """
        This function describes the movement of the 'turtle' along the MTG for creating a root graph on PlantGL.
        :return: root_visitor
        """

        def root_visitor(g, v, turtle):
            n = g.node(v)

            # We look at the geometrical properties already defined within the root element:
            radius = n.radius
            length = n.length
            angle_down = n.angle_down
            angle_roll = n.angle_roll

            # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
            position1 = turtle.getPosition()
            n.x1 = position1[0]
            n.y1 = position1[1]
            n.z1 = position1[2]

            # The direction of the turtle is changed:
            turtle.down(angle_down)
            turtle.rollL(angle_roll)

            # The turtle is moved:
            turtle.setId(v)
            # We define the radius of the cylinder to be displayed:
            turtle.setWidth(radius)
            # We move the turtle by the length of the root segment:
            turtle.F(length)

            # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
            position2 = turtle.getPosition()
            n.x2 = position2[0]
            n.y2 = position2[1]
            n.z2 = position2[2]

        return root_visitor

    # Function for creating a color map for a MTG:
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

        # We make sure that the user did not accidently switch between vmin and vmax:
        if vmin >= vmax:
            raise Exception("Sorry, the vmin and vmax values of the color scale of the graph are wrong!")
        if lognorm and (vmin <= 0 or vmax <= 0):
            raise Exception(
                "Sorry, it is not possible to represent negative values in a log scale - check vmin and vmax!")

        prop = g.property(property_name)
        keys = prop.keys()
        values = list(prop.values())
        # m, M = int(values.min()), int(values.max())

        _cmap = color.get_cmap(cmap)
        norm = color.Normalize(vmin, vmax) if not lognorm else color.LogNorm(vmin, vmax)
        values = norm(values)

        colors = (_cmap(values)[:, 0:3]) * 255
        colors = np.array(colors, dtype=int).tolist()

        g.properties()['color'] = dict(zip(keys, colors))

        return g

    # Function for setting the center anc camera position in a scene:
    def setting_PGLViewer(width=1200, height=1200,
                          x_center=0., y_center=0., z_center=0.,
                          x_cam=0., y_cam=0., z_cam=-0.2,
                          grid=True,
                          background_color=[255, 255, 255]):
        """
        This function sets the center of the graph and the relative position of the camera for displaying a scene in
        PGL.
        :param width: width of the window (pixels)
        :param height: height of the window (pixels)
        :param x_center: x-coordinate of the center at which the camera looks
        :param y_center: y-coordinate of the center at which the camera looks
        :param z_center: z-coordinate of the center at which the camera looks
        :param x_cam: x-coordinate of the camera position
        :param y_cam: y-coordinate of the camera position
        :param z_cam: z-coordinate of the camera position
        :param grid: if True, all grids will be displayed
        :param background_color: a RGB vector corresponding to the background color of the graph
        :return:
        """

        # We define the coordinates of the point cam_target that will be the center of the graph:
        cam_target = pgl.Vector3(x_center, y_center, z_center)
        # We define the coordinates of the point cam_pos that represents the position of the camera:
        cam_pos = pgl.Vector3(x_cam, y_cam, z_cam)
        # We position the camera in the scene relatively to the center of the scene:
        pgl.Viewer.camera.lookAt(cam_pos, cam_target)
        # We define the dimensions of the graph:
        pgl.Viewer.frameGL.setSize(width, height)
        # We define the background of the scene:
        pgl.Viewer.frameGL.setBgColor(background_color[0], background_color[1], background_color[2])
        # We define whether grids are displayed or not:
        pgl.Viewer.grids.set(grid, grid, grid, grid)

        return

    # CREATING THE PLOT:
    #-------------------
    # We initialize a turtle in PlantGL:
    turtle = turt.PglTurtle()
    # If roots are displayed, we move the turtle downwards:
    turtle.down(180)

    # We initialize a scene:
    new_scene = pgl.Scene()

    # SETTING THE NEW SCENE WITH THE TURTLE:
    # We create the scene by moving the turtle along the current MTG:
    scene = turt.TurtleFrame(g, visitor=get_root_visitor(), turtle=turtle, gc=False)

    # SETTING THE COLOR OF THE SHAPES:
    # We compute the colors of the graph:
    my_colormap(g, property_name=property_name, cmap=cmap, vmin=vmin, vmax=vmax, lognorm=lognorm)
    # We use the property 'color' of the MTG calculated by the function 'my_colormap':
    colors = g.property('color')
    # We get a dictionnary of all shapes in the scene:
    shapes = dict((sh.id, sh) for sh in scene)
    # We cover each node of the MTG:
    for vid in colors:
        if vid in shapes:
            n = g.node(vid)
            if single_color is None:
                shapes[vid].appearance = pgl.Material(colors[vid], transparency=0.0)
            else:
                # Otherwise, we can print it with a unique color:
                shapes[vid].appearance = pgl.Material(single_color, transparency=0.0)

    # We add each shape of the current MTG to the final scene:
    for shape in scene:
        new_scene += shape

    # SETTING THE VIEWER OPTIONS:
    setting_PGLViewer(width=width, height=height,
                      x_center=x_center, y_center=y_center, z_center=z_center,
                      x_cam=x_cam, y_cam=y_cam, z_cam=z_cam,
                      grid=grid,
                      background_color=background_color)

    # DISPLAYING THE MTG:
    pgl.Viewer.display(new_scene)

    # RECORDING THE MTG:
    if recording_plot:
        pgl.Viewer.saveSnapshot(plot_name)

    return

########################################################################################################################
########################################################################################################################

if __name__ == '__main__':

    # CREATING A TEST MTG:
    ####################################################################################################################
    # # We create a simple MTG with 3 segments:
    # g = MTG()
    # element_1_id = g.add_component(g.root,
    #                                label='Internode',
    #                                radius=0.05,
    #                                angle_down=0.,
    #                                angle_roll=0.,
    #                                length = 1.,
    #                                x1=0,
    #                                y1=0,
    #                                z1=0)
    # element_1 = g.node(element_1_id)
    # element_2 = element_1.add_child(edge_type='<',
    #                                 label='Base_of_seminal_roots',
    #                                 radius=0.05,
    #                                 length=0.,
    #                                 angle_down=0,
    #                                 angle_roll=60)
    # element_3 = element_2.add_child(edge_type='+',
    #                                 label='Seminal_root',
    #                                 radius=0.07,
    #                                 length=0.5,
    #                                 angle_down=45,
    #                                 angle_roll=0)
    # element_4 = element_2.add_child(edge_type='<',
    #                                 label='Base_of_seminal_roots',
    #                                 radius=0.05,
    #                                 length=0.,
    #                                 angle_down=0,
    #                                 angle_roll=75)
    # element_5 = element_4.add_child(edge_type='+',
    #                                 label='Seminal_root',
    #                                 radius=0.07,
    #                                 length=0.5,
    #                                 angle_down=45,
    #                                 angle_roll=0)
    # element_6 = element_3.add_child(edge_type='<',
    #                                 label='Base_of_seminal_roots',
    #                                 radius=0.05,
    #                                 length=0.,
    #                                 angle_down=0,
    #                                 angle_roll=40)
    # element_7 = element_6.add_child(edge_type='+',
    #                                 label='Seminal_root',
    #                                 radius=0.1,
    #                                 length=0.7,
    #                                 angle_down=65,
    #                                 angle_roll=0)
    # element_8 = element_6.add_child(edge_type='<',
    #                                 label='Internode',
    #                                 radius=0.05,
    #                                 length=0.5,
    #                                 angle_down=60,
    #                                 angle_roll=60)
    # element_9 = element_8.add_child(edge_type='<',
    #                                 label='Internode',
    #                                 radius=0.05,
    #                                 length=1.5,
    #                                 angle_down=20,
    #                                 angle_roll=10)
    # element_10 = element_8.add_child(edge_type='+',
    #                                 label='Lateral',
    #                                 radius=0.075,
    #                                 length=1.,
    #                                 angle_down=80,
    #                                 angle_roll=-60)
    # element_11 = element_8.add_child(edge_type='+',
    #                                 label='Lateral',
    #                                 radius=0.05,
    #                                 length=1.,
    #                                 angle_down=-80,
    #                                 angle_roll=-70)
    # element_12 = element_10.insert_parent(edge_type='+',
    #                                     label="Intermediate",
    #                                     radius = 0.4,
    #                                     length=0.,
    #                                     angle_down = 0,
    #                                     angle_roll = 25)
    # element_13 = element_11.insert_parent(edge_type='+',
    #                                     label="Intermediate",
    #                                     radius = 0.2,
    #                                     length = 0.,
    #                                     angle_down = 0,
    #                                     angle_roll = 45)
    #
    # # The plotting with the turtle automatically calculates the x,y,z coordinates of the two children:
    # plot_root_MTG(g,
    #               width=1200, height=900,
    #               x_center=0., y_center=0., z_center=0.,
    #               x_cam=6., y_cam=-6., z_cam=0.,
    #               grid=True,
    #               background_color=[94, 76, 64],
    #               single_color=None,
    #               property_name="angle_roll", cmap='jet', vmin=0., vmax=180, lognorm=False,
    #               recording_plot=True,
    #               plot_name='plot_original.png')
    # # And we record the MTG for later:
    # with open('root_test.pckl', 'wb') as output:
    #     pickle.dump(g, output, protocol=2)
    # print("The MTG file corresponding to the root system has been recorded.")
    ####################################################################################################################

    # Testing the function "calculating_angles":
    ############################################

    # We load the test MTG:
    # MTG_path = 'root_test.pckl'
    # Or we load a MTG file created by RhizoDep:
    # MTG_path = 'root00125.pckl'
    MTG_path = 'root00238.pckl'
    # MTG_path = 'root00293.pckl'
    # MTG_path = 'root01240.pckl'
    # MTG_path = 'root01745.pckl'


    f = open(MTG_path, 'rb')
    g = pickle.load(f)
    f.close()
    # We record some of the initial properties of the MTG:
    recording_MTG_properties(g, file_name='angles_original.csv', list_of_properties=["label","type","length",
                                                                                     "x1","x2","y1","y2","z1","z2",
                                                                                     "angle_down","angle_roll"])

    # Plotting the new MTG file:
    print("Plotting the original MTG...")

    plot_root_MTG(g,
                  width=1200, height=900,
                  x_center=0., y_center=0., z_center=0.,
                  x_cam=6., y_cam=-6., z_cam=0.,
                  grid=True,
                  background_color=[94, 76, 64],
                  single_color=None,
                  property_name="angle_roll", cmap='jet', vmin=0., vmax=180, lognorm=False,
                  recording_plot=True,
                  plot_name='plot_original.png')

    # We delete the properties "angle_down" and "angle_roll" from the MTG:
    original_angle_down = g.properties().pop('angle_down', None)
    original_angle_roll = g.properties().pop('angle_roll', None)
    # And we recaculate the angles with the customized function:
    print("Computing angles...")
    calculating_angles(g, printing_details=True)

    # We record some of the new properties of the MTG:
    recording_MTG_properties(g, file_name='angles_calculated.csv', list_of_properties=["label","type","length",
                                                                                       "x1","x2","y1","y2","z1","z2",
                                                                                       # "dir_parent_x", "dir_parent_y","dir_parent_z",
                                                                                       # "direction_x", "direction_y", "direction_z",
                                                                                       # "rolling_reference_x","rolling_reference_y","rolling_reference_z",
                                                                                       # "prod_x","prod_y","prod_z", "check",
                                                                                       "angle_down","angle_roll"])

    # Plotting the new MTG file:
    print("Plotting the MTG...")

    plot_root_MTG(g,
                  width=1200, height=900,
                  x_center=0., y_center=0., z_center=0.,
                  x_cam=6., y_cam=-6., z_cam=0.,
                  grid=True,
                  background_color=[94, 76, 64],
                  single_color=None,
                  property_name="angle_roll", cmap='jet', vmin=0., vmax=180, lognorm=False,
                  recording_plot=True,
                  plot_name='plot_calculated.png')

    # my_colormap(g, property_name="angle_roll", cmap='jet', vmin=0., vmax=180, lognorm=False)
    #
    # # We can create the "long-way" plot (TAKES TIME, BUT ALWAYS WORKS!):
    # p = plotting_roots_with_pyvista(g, displaying_root_hairs = False,
    #                                 showing=True, recording_image=True,
    #                                 image_file='C:/Users/frees/rhizodep/saved_outputs/plot.png',
    #                                 background_color = [94, 76, 64],
    #                                 plot_width=800, plot_height=800,
    #                                 camera_x=5., camera_y=5., camera_z=-1.,
    #                                 focal_x=0., focal_y=0., focal_z=-1.)

    # # We can create the "fast-way" plot (QUICK, BUT SOMETIMES BUGS!):
    # p = fast_plotting_roots_with_pyvista(g, displaying_root_hairs = False,
    #                                      showing=True, recording_image=True,
    #                                      image_file='C:/Users/frees/rhizodep/saved_outputs/plot.png',
    #                                      background_color = [94, 76, 64],
    #                                      plot_width=800, plot_height=800,
    #                                      camera_x=5., camera_y=5., camera_z=-1.,
    #                                      focal_x=0., focal_y=0., focal_z=-1.)


