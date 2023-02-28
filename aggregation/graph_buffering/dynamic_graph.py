import os
import time

from aggregation.graph_buffering.abstract_dynamic_graph import AbstractDynamicGraph
from extraction import Tube


class GraphBuffer(AbstractDynamicGraph):
    def __init__(self, tubes: list[Tube]):
        pass
