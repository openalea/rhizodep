from openalea.mtg import *

from collections import deque



class Simulation(object):
    def __init__(self):
        g = self.g = MTG()
        self.root = g.add_component(g.root)
        self.apices = deque([self.root])


    def growth(self):
        new_apices = deque([])
        for aid in apices:
            new_apex =
