import os
import time
from typing import List

from aggregation.graph_buffering.abstract_dynamic_graph import AbstractDynamicGraph
from aggregation.graph_building.graph import RuanGraph
from aggregation.graph_building.graph_coloring import GraphColoration
from extraction import Tube


class GraphBuffer(AbstractDynamicGraph):
    def __init__(self, graph: RuanGraph, h, p):
        super(GraphBuffer, self).__init__(graph=graph, h=h, p=p)
        self.coloration = GraphColoration(graph)
        self.graph = self.coloration.color_graph()
        self.starting_time = self.coloration.tube_starting_time()

    def removing(self):
        tmp = sorted(self.starting_time.items(), key=lambda item: item[1])
        tube_removed = list(tmp.keys())
