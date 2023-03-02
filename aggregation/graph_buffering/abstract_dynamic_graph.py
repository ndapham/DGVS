import torch
from abc import ABC

from extraction import Tube
from aggregation.graph_building.graph import AbstractGraph


class AbstractDynamicGraph(ABC):
    def __init__(self):
        self.graph = None  # graph of activity tubes
        self.h = None  # tolerance of collision
        self.number_of_collisions = list()  # list of number collision at each time location
        self.p = None   # Upper bound of the maximum tube number in dynamic graph

    def updating(self, new_tube, c_min) -> AbstractGraph:
        raise Exception("Using function for updating the graph buffer in the abstract class")

    def adding(self, new_tube, c_min) -> AbstractGraph:
        raise Exception("Using function for adding tube to graph buffer in the abstract class")

    def adjusting(self, new_tube, c_min) -> AbstractGraph:
        raise Exception("Using function for adding the graph buffer in the abstract class")

    def removing(self) -> Tube:
        raise Exception("Using function for removing tube from graph buffer in the abstract class")
