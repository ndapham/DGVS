from abc import ABC

from extraction import Tube


class AbstractDynamicGraph(ABC):
    def __init__(self, q, h, p):
        self.q = q  # hyperparameter for l(q) graph coloring
        self.h = h  # tolerance of collision
        self.p = p  # Upper bound of the maximum tube number in dynamic graph

        self.tubes_buffer = []  # Buffer stores tubes to push in graph
        self.tubes_in_process = []  # List of tubes to compute the graph
        self.graph = None  # graph of activity tubes
        # self.number_of_collisions = dict()  # list of number collision at each time location

    def updating(self, new_tube):
        raise Exception("Using function for updating the dynamic graph in the abstract class")

    def adding(self, new_tube):
        raise Exception("Using function for adding tube to the dynamic graph in the abstract class")

    def adjusting(self, new_tube):
        raise Exception("Using function for adding the dynamic graph in the abstract class")

    def removing(self) -> Tube:
        raise Exception("Using function for removing tube from dynamic graph in the abstract class")
