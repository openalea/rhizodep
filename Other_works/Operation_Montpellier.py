
# Loading packages:
import os
import pickle
from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order

########################################################################################################################
# We define a new function for simulating nitrogen uptake for a root system over a certain period of time:
def nitrogen_uptake(g, time_step):

    """
    This function simulates in an absolutely ugly way the amount of nitrogen (mol_N) taken by a root system 'g'.
    :param g: the MTG to be read
    :param time_step: the time step in seconds
    :return: g, the new updated MTG
    """

    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        # We define the amount of nitrogen that has been taken (mol of N) in a perfectly stupid way:
        n.nitrogen_taken += 2*time_step
    print('Nitrogen uptake for the first element of the root system is', n.nitrogen_taken, 'mol of N.')

    return g
########################################################################################################################

# INITIALIZATION:
#################

# We define the path of the directory:
my_path = r'C:\\Users\\frees\\rhizodep\\simulations\\running_scenarios\\outputs\\Scenario_0001\\MTG_files'
os.chdir(my_path)
# We open the MTG file that has already been prepared and contains all RhizoDep's variables:
f = open('root00119.pckl', 'rb')
# We define "g" as the new MTG that has been loaded:
g = pickle.load(f)
f.close()

# We can now investigate the properties of a given node of the MTG, say node Nr. 10:
n = g.node(5)
# print(n.label, n.type)
print('Here are the properties of node Nr. 5:')
print(n.properties())

# We initialize a new property "nitrogen_taken" at each node:
# We define "root" as the starting point of the loop below:
root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
root = next(root_gen)
#  We travel in the MTG from the root tips to the base:
for vid in post_order(g, root):
    # We define the current root element as n:
    n = g.node(vid)
    # And we set this new property "nitrogen_taken" to 0:
    n.nitrogen_taken = 0

# MAIN SIMULATION:
#################

# We run a scenarios over 10 time steps:
n_steps = 10
# We initialize the time at 0:
time = 0
# We calculate things for each time step:
for step in range(0, n_steps):
    # We increment the time:
    time += 1
    print("Calculating at time t =", time, 's...' )
    # We call the function nitrogen uptake for this time step:
    nitrogen_uptake(g, time_step=1)

print('Simulation is done!')