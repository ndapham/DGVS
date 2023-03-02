from abc import ABC
import operator
import numpy as np
import networkx as nx

import tqdm as tqdm
from aggregation.graph_building.graph import RuanGraph
from extraction import Tube

"""
Implement the heuristic graph coloring algorithm for L(q) Graph coloring introduced by He et al 2017
This algorithm is formulated as 2 algorithms: 
- Degree of Saturation algorithm (DSATUR)
- Recursive Largest Fit algorithm (RFL)
"""


class SaturationCache:
    def __init__(self, graph):
        self.graph = graph
        self.s_l = {}  # recorded the saturation of each node based on length of activities
        self.s_app = {}  # recorded the appearing time of activity
        self.s_pc = {}  # recorded number of potential collisions of the tube

        self.tubes = None
        self.num_nodes = None
        self.video_frames = None
        self.max_length_tube = None
        self.init_params()

    def init_params(self):
        """
        Initialize some parameters used in calculating metrics
        """
        self.tubes = self.graph.tubes
        self.num_nodes = len(self.graph.A.keys())
        self.video_frames = max([tube.eframe for tube in self.tubes])
        self.max_length_tube = max([tube.frame_length() for tube in self.tubes])

    def cal_sl(self, node_tag):
        """
        Measure the relative length (in terms of frames) of the
        tube corresponding to the node referenced by node tag. The
        measure is normalized using the tube with maximum length.
        """

        if node_tag in self.s_l:
            return self.s_l[node_tag]
        tube_tag, frame_id = node_tag.split(".")
        tube = self.graph.nodes[tube_tag][frame_id].tube
        self.s_l[node_tag] = tube.frame_length() / self.max_length_tube

        return self.s_l[node_tag]

    def cal_sapp(self, node_tag):
        """
        Measure the relative time of appearance of the tube
        corresponding to the node referenced by node tag.
        """

        if node_tag in self.s_app:
            return self.s_app[node_tag]
        tube_tag, frame_id = node_tag.split(".")
        tube = self.graph.nodes[tube_tag][frame_id].tube
        self.s_app[node_tag] = (self.video_frames - tube.sframe) / self.video_frames

        return self.s_app[node_tag]

    def cal_spc(self, node_tag):
        """
        Measure the number of potential collisions of the tube
        corresponding to the node referenced by node tag
        """

        if node_tag in self.s_pc:
            return self.s_pc[node_tag]
        tube_tag, frame_id = node_tag.split(".")
        num_node_same_tubes = len(self.graph.nodes[tube_tag].keys())
        self.s_pc[node_tag] = num_node_same_tubes / self.num_nodes

        return self.s_pc[node_tag]

    def cal_sd(self, node_tag):
        """
        Classic degree of saturation divided by total number of nodes in order to normalize.
        Count the number of different colors in the adjacent nodes of the node referenced by
        the node tag.

        S_d can not be cached, need to computed it in each iteration while coloring
        """
        different_colors = set()
        adjacent_nodes = self.graph.get_adjacent_node(node_tag)
        for node_tag, node in adjacent_nodes:
            if node.color is not None:
                different_colors.add(node.color)
        return len(different_colors) / self.num_nodes

    def saturation(self, node_tag):
        """
        Calculate the saturation of the node referenced by node tag.
        """
        return self.cal_sd(node_tag) + self.cal_sl(node_tag) + self.cal_spc(node_tag) + self.cal_sapp(node_tag)

    def nodes_saturation(self, node_tags):
        """
        Calculate the degree of saturation for all the nodes in the list node
        """
        return {node_tag: self.saturation(node_tag) for node_tag in node_tags}


class GraphColoration:
    def __init__(self, graph: RuanGraph, q):
        self.saturation_cache = SaturationCache(graph)
        self.graph = graph
        self.q = q

    def q_far_apart(self, proposed_color, node_tag):
        """
        This condition imposes that all the nodes connected by an edge weight 1
        to the node referenced must be at least q far apart
        """
        adjacent_nodes = self.graph.get_adjacent_nodes(node_tag)
        for tag, node in adjacent_nodes:
            if self.graph.A[node_tag][tag] != 1:
                continue
            if node.color is None:
                continue
            if np.abs(node.color - proposed_color) <= self.q:
                return False
        return True

    # def does_not_overlap(self, proposed_color, nodekey):
    #     """
    #     Condition 2 in He et al. 2017 paper which imposed a strict rule to avoid collisions
    #     However, Ruan et al. 2019 not define the Overlapping relation so this is unnecessary
    #     """
    #     if pcg.generated_by_intersection(nodekey) or pcg.isolated_main_node(nodekey):
    #         return True
    #     vi, vj, vip, vjp = pcg.identify_quatern(nodekey)
    #     vi, vj, vip, vjp = pcg.node(vi), pcg.node(vj), pcg.node(vip), pcg.node(vjp)
    #     if vip.color is None or vjp.color is None: return True
    #     tij = vj.frame - vi.frame
    #     return (proposed_color - vip.color) * (proposed_color + tij - vjp.color) > 0

    def ssort(self, nodes_saturation):
        """
        This function sorts the nodes in decreasing order by their degree of saturation.
        The appearance time is used to break ties.
        """
        app = lambda node_tag: self.graph.get_node_by_nodetag(node_tag).tube.sframe
        order_list = [(node_tag, saturation, app(node_tag)) for node_tag, saturation in nodes_saturation]
        order_list.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return [node_tag for node_tag, saturation, appearance in order_list]

    def color_graph(self):
        """
        Implements the graph coloring algorithm introduced by He et al. 2017
        """
        color = 1
        self.graph.clean_colors()
        pbar = tqdm(total=len(self.graph.list_node_tags))

        while len(self.graph.uncolored_nodes()) > 0:
            nodes_not_colored = self.graph.uncolored_nodes()
            nodes_saturation = self.saturation_cache.nodes_saturation(nodes_not_colored)

            order_list = self.ssort(nodes_saturation)
            nodes_saturation = self.saturation_cache.nodes_saturation(nodes_not_colored)

            for node_tag in order_list:
                if self.graph.get_node_by_nodetag(node_tag).color is not None:
                    continue
                if self.q_far_apart(color, node_tag):
                    self.graph.get_node_by_nodetag(node_tag).color = color
                    pbar.update(1)




