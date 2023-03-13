import os
import time
from typing import List

from aggregation.graph_buffering.abstract_dynamic_graph import AbstractDynamicGraph
from aggregation.graph_building.graph import RuanGraph
from aggregation.graph_building.graph_coloring import GraphColoration
from aggregation.graph_building.relations import RuanRelationsMap
from extraction import Tube


class RuanDynamicGraph(AbstractDynamicGraph):
    def __init__(self, q, h, p):
        super(RuanDynamicGraph, self).__init__(q=q, h=h, p=p)
        self.current_starting_times = None  # the starting times for tubes in current graph (time step t)
        self.graph_coloration = None    # The coloring machine that helps to color the initial graph
        self.c_min = 0  # c_min value in Ruan et al. 2019 - initial value = 0

    def run_pipeline(self):
        """
        Pipeline using to update the graph through time steps.
        Combine the method using potential collisions graph
        """
        while len(self.tubes_buffer):
            # Push a tube from buffer to process
            self.tubes_in_process.append(self.tubes_buffer[0])
            self.tubes_buffer = self.tubes_buffer[1:]

            # If the number of tubes gets up to p then a tube
            # will be selected and fused into synopsis video
            if len(self.tubes_in_process) == self.p:
                if self.graph is None:
                    # Building the graph
                    relation_map = RuanRelationsMap(self.tubes_in_process)
                    self.graph = RuanGraph(self.tubes_in_process, relation_map)

                    # Color the initial graph and get the starting time of each tube
                    self.graph_coloration = GraphColoration(self.q)
                    self.graph = self.graph_coloration.color_graph(self.graph)
                    self.current_starting_times = self.graph_coloration.tube_starting_time(self.graph)
                else:
                    new_tube = self.tubes_in_process[-1]

                    # Remove the tube with minimum starting times
                    tube_del, removed_tag = self.removing()
                    # TODO: records the removed tube for stitching

                    # Update the list of tubes in processing
                    self.tubes_in_process = self.graph.tubes

                    # Update c_min
                    self.c_min = max(self.c_min, removed_tag)
                    self.graph = self.updating(new_tube, self.c_min)
                    # TODO: record the final graph to stitch the rest tubes in synopsis video

    def removing(self):
        """
        Remove the tube with minimum starting time then stitch it to the output video
        """
        tmp = sorted(self.current_starting_times.items(), key=lambda item: item[1])
        removed_tag, remove_starting_time = tmp[0]
        removed_tubes = [tube for tube in self.graph.tubes if tube.tag == removed_tag]
        self.graph.tubes = [tube for tube in self.graph.tubes if tube.tag != removed_tag]
        return removed_tubes, remove_starting_time

    @staticmethod
    def get_color(tube: Tube, n):
        """
        Given the tube  and n is a number
        return the appearance times of the nth frame of that tube
        """
        return tube.color + n - 1

    def updating(self, new_tube, c_min):
        """
        Update the graph with new coming tube.
        Make a compare between two method adding and adjusting, choose the method that give
        better condensation.
        """
        
        return self.graph

    def adding(self, new_tube, c_min):
        # for a_data in new_tube:
        #     for Tb in self.graph.tubes:
        #         for b_data in Tb:
        #             if self.frame_intersect(a_data, b_data):

        return self.graph

    def adjusting(self, new_tube, c_min):
        return self.graph


