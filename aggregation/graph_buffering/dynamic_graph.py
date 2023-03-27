import pprint
from typing import List
import copy

from aggregation.graph_buffering.abstract_dynamic_graph import AbstractDynamicGraph
from aggregation.graph_building.graph import RuanGraph
from aggregation.graph_building.graph_coloring import GraphColoration
from aggregation.graph_building.relations import RuanRelationsMap
from extraction import Tube
from utils.helpers import frame_intersect


class RuanDynamicGraph(AbstractDynamicGraph):
    def __init__(self, q=3, h=1, p=3):
        super(RuanDynamicGraph, self).__init__(q=q, h=h, p=p)
        self.current_starting_times = None  # the starting times for tubes in current graph (time step t)
        self.graph_coloration = None  # The coloring machine that helps to color the initial graph
        self.c_min = 0  # c_min value in Ruan et al. 2019 - available value for new tube stitching in
        self.output_tubes = []

    def run_pipeline(self, tubes):
        """
        Pipeline using to update the graph through time steps.
        Combine with the method using potential collisions graph
        """
        # TODO: define tubes_buffer for further developments for real-time applications
        # Currently, tubes_buffer receive all the whole set of tubes. This can be used for offline synopsis only
        self.tubes_buffer = tubes

        while len(self.tubes_buffer):
            # Push a tube from buffer to process
            self.tubes_in_process.append(self.tubes_buffer[0])
            self.tubes_buffer = self.tubes_buffer[1:]

            # If the number of tubes gets up to p then a tube
            # will be selected and fused into synopsis video
            if len(self.tubes_in_process) >= self.p:
                if self.graph is None:
                    # Building the graph
                    relation_map = RuanRelationsMap(self.tubes_in_process)
                    self.graph = RuanGraph(self.tubes_in_process, relation_map)

                    # Color the initial graph and get the starting time of each tube
                    self.graph_coloration = GraphColoration(self.q)
                    self.graph = self.graph_coloration.color_graph(self.graph)
                    self.current_starting_times = self.graph_coloration.tube_starting_time(self.graph)
                    self.tubes_in_process = self.graph.tubes.copy()

                else:
                    new_tube = self.tubes_in_process[-1]

                    # Remove the tube with minimum starting times
                    tubes_del, remove_starting_times = self.removing()
                    # TODO: records the removed tube for stitching
                    self.output_tubes.extend(tubes_del)

                    # Update the value of c_min
                    self.c_min = max(self.c_min, remove_starting_times)

                    # Update the graph
                    self.graph = self.updating(new_tube)

                    # Update the list of tubes in processing
                    self.tubes_in_process = self.graph.tubes.copy()

        # TODO: record the final graph to stitch the rest tubes in synopsis video
        self.tubes_in_process = sorted(self.tubes_in_process, key=lambda in_prog_tube: in_prog_tube.color)
        self.output_tubes.extend(self.tubes_in_process)

        return self.output_tubes

    def removing(self):
        """
        Remove the tube with minimum starting time then stitch it to the output video
        """
        # Sort to get the tube with the minimum starting time to remove it
        tmp: List[Tube]  # tmp is the sorted list of tubes
        tmp = sorted(self.graph.tubes, key=lambda in_prog_tube: in_prog_tube.color)

        # Get the information of removed_tubes to stitch it to the video synopsis
        removed_tag, remove_starting_time = tmp[0].tag, tmp[0].color
        removed_tubes = [tube for tube in self.graph.tubes if tube.tag == removed_tag]
        assert len(removed_tubes) == 1, \
            f"Expected only one tube be removed each iter but there are {len(removed_tubes)} due to duplicated tube tag"

        # Update the list of rest tube in graph
        self.graph.tubes = [tube for tube in self.graph.tubes if tube.tag != removed_tag]
        return removed_tubes, remove_starting_time

    @staticmethod
    def get_color(tube: Tube, n):
        """
        Given the tube and a number n: index of frame start from 0
        return the appearance times of the nth frame of that tube
        """
        return tube.color + n

    def updating(self, new_tube):
        """
        Update the graph with new coming tube.
        Compare between two method adding and adjusting, choose the method that give
        better condensation.
        """
        # Save the current graph for further updates
        callback_graph = copy.deepcopy(self.graph)

        # Try using the adding method to update the graph
        tmp_graph1 = self.adding(new_tube)
        adding_end_time_location = tmp_graph1.get_end_time_location()

        # Try using the adjusting method to update the graph
        self.graph = callback_graph
        tmp_graph2 = self.adjusting(new_tube)
        adjusting_end_time_location = tmp_graph2.get_end_time_location()

        # Update graph
        self.graph = tmp_graph1 if adding_end_time_location <= adjusting_end_time_location else tmp_graph2

        return self.graph

    def adding(self, new_tube: Tube):
        """
        Adding method described by Ruan et al. 2019
        """
        # TODO: Define NC as a list or a dict? how to manage memory if number_of_collisions as a list
        number_of_collisions = dict()

        # Try to place the new tube in the available graph
        c_tmp: int = 0
        for a_index, a_data in enumerate(new_tube):
            for potential_collision_tube in (self.output_tubes + self.graph.tubes):
                for b_index, b_data in enumerate(potential_collision_tube):
                    if frame_intersect(a_data, b_data):
                        c_tmp = self.get_color(potential_collision_tube, b_index) - a_index
                    if c_tmp >= 0:
                        number_of_collisions[c_tmp] = number_of_collisions.get(c_tmp, 0) + 1

        # Color the new tube based on the list of available places
        color = self.c_min

        while 1:
            if number_of_collisions.get(color, 0) < self.h:
                new_tube.color = color
                break
            color += 1

        # Add new tube to available graph to create new graph G(t+1)
        self.graph.tubes.append(new_tube)
        return self.graph

    def adjusting(self, new_tube: Tube):
        """
        Adjusting method described by Ruan et al. 2019
        """
        # Initialize a queue for adjusting
        queue = []

        # Put new tube at time location of c_min
        new_tube.color = self.c_min
        # Check if new_tube collide with tube in progress
        for potential_collide_tube in self.graph.tubes:
            is_collided_flag = False
            for a_data in new_tube:
                for b_data in potential_collide_tube:
                    # If new tube collides with potential_collide_tube
                    # remove the potential_collide_tube from graph the push it back into queue
                    if frame_intersect(a_data, b_data):
                        is_collided_flag = True
                        # In paper, authors described tube buffer as a queue,
                        # so I wonder if this could make chronological disorders
                        queue.append(potential_collide_tube)
                        self.graph.tubes = [tube for tube in self.graph.tubes if tube.tag != potential_collide_tube.tag]
                        break
                if is_collided_flag:
                    break

        # Add new tube to the available graph to create new graph G(t+1)
        self.graph.tubes.append(new_tube)

        # Add all the tubes in the queue into the graph again
        while len(queue):
            tube_in_queue = queue.pop(0)
            self.graph = self.adding(tube_in_queue)
        return self.graph
