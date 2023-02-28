import torch

from extraction import Tube


class Node:
    """
    Represent a frame which is a  potential collision
    """

    def __init__(self, tag: str, tube: Tube, frame=None):
        self.tag = tag  # tag of the tube which the frame belong to
        self.tube = tube  # the main tube
        self.color = None   # time location assigned to this node
        self.frame = frame


class Graph:

    def __init__(self, tubes: list[Tube]):
        self.tubes = tubes
        self.nodes = {}
        self.edges = []
        self.A = {}
        self._compute_graph()

    def _compute_graph(self):
        pass