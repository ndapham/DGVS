from abc import ABC
from typing import List

from aggregation.graph_building.relations import AbstractRelations
from extraction import Tube


class Node:
    """
    Represent a frame which is a  potential collision
    """

    def __init__(self, tag: str, tube: Tube, frame=None):
        self.tag = tag  # tag of the tube which the frame belong to
        self.tube = tube  # the main tube
        self.color = None  # time location assigned to this node
        self.frame = frame


class AbstractGraph(ABC):
    """
    Graph in general, there are 2 kinds of edges:
    - Undirected edges: connect 2 nodes that have collision
    - Direct nodes: connect 2 nodes in the same tube
    """

    def __init__(self, tubes: List[Tube], relations: AbstractRelations):
        self.tubes = tubes  # list of activity tubes
        self.relations = relations  # RelationsMap used to initialize the graph
        self.nodes = {}  # a dictionary recorded nodes
        self.list_node_tags = []    # a list of the tags of nodes in the graph
        self.edges = []  # list of tuples recorded the start point, end point and weight of the edges
        self.A = {}  # The adjacency matrix
        self.compute_graph()
        self.compute_adjacency_matrix()

    def compute_graph(self):
        raise Exception("Using default function to compute graph from abstract class")

    def color(self):
        raise Exception("Using default function to color the graph nodes from abstract class")

    def compute_adjacency_matrix(self):
        raise Exception("Using default function to compute the adjacency matrix of graph from abstract class")

    def clean_colors(self):
        raise Exception("Using default function to clean colors of nodes in the graph from abstract class")

    # Optional these 2 can be placed under dynamic graph
    def remove_tube(self):
        raise Exception("Using default function to remove tube from abstract class")

    def add_tube(self):
        raise Exception("Using default function to add tube from abstract class")


class RuanGraph(AbstractGraph):
    def __init__(self, tubes: List[Tube], relations: AbstractRelations):
        super(RuanGraph, self).__init__(tubes, relations)

    def compute_graph(self):
        """
        Perform the nodes and edges generation process described in Ruan et al 2019
        I only compute the undirected edges to apply graph coloring algorithm
        with directed edges i represent it through the relation: "in the same tubes"
        """
        self.edges = []
        relations = self.relations.relations_dict

        for tube in self.tubes:
            tube_relations = relations[tube.tag]
            self.nodes[str(tube.tag)] = {}

            # If a tube is isolated, i create a node for its first frame
            if self.check_isolated_node(tube_relations):
                self.nodes[str(tube.tag)][str(0)] = Node(tube.tag, tube)
                continue

            # If the tube collide with some other tubes, i build nodes for frame that witness the collisions
            for tag, relation in tube_relations.items():
                # tag: the tag of src_tube, relation is the list of pairs of frame id that collided

                # if there is no collisions between 2 tubes so keep going through
                if (tag == tube.tag) or (len(relation) == 0):
                    continue

                for src_frame_id, trg_frame_id in relation:
                    src_tag = f"{tube.tag}.{src_frame_id}"
                    trg_tag = f"{tag}.{trg_frame_id}"

                    if src_frame_id not in self.nodes[str(tube.tag)]:
                        self.nodes[str(tube.tag)][str(src_frame_id)] = Node(src_tag, tube, src_frame_id)

                    # if trg_frame_id not in self.nodes[str(tube.tag)]:
                    #     Hehe i do not have target tube to define the target node so hehehe i comment these lines :v
                    #     node_tag = f"{tag}.{trg_frame_id}"
                    #     self.nodes[str(tag)][trg_frame_id] = Node(trg_tag, ....trg_tube, trg_frame_id)

                    # Insert the undirected edge between nodes
                    weight = 1  # i define the weight of the undirected edges as 1 though this is unnecessary
                    self._insert_edge_nodup(src_tag, trg_tag, weight)
        return 0

    def clean_colors(self):
        """
        Reset all the color in nodes to None
        """
        for tube_tag in self.nodes.values():
            for node in self.nodes[tube_tag].values():
                node.color = None
        return

    def check_isolated_node(self, tube_relations):
        """
        If a tube has irrelevant relations with all other tubes, then it can be abstract as an isolated main-node
        """
        return all([relation is None for relation in tube_relations.values()])

    def _insert_edge_nodup(self, from_vertex, to_vertex, weight):
        """
        Insert new edges only - no duplicated
        """
        if (from_vertex, to_vertex, weight) in self.edges:
            return
        self.edges.append((from_vertex, to_vertex, weight))
        return

    def get_node_by_nodetag(self, node_tag):
        tube_tag, frame_id = node_tag.split(".")
        return self.nodes[tube_tag][frame_id]

    def compute_adjacency_matrix(self):
        self.list_node_tags = []
        for tube_tag in self.nodes.keys():
            for node in self.nodes[tube_tag].values():
                self.list_node_tags.append(node.tag)

        for u in self.list_node_tags:
            self.A[u] = {}
            for v in self.list_node_tags:
                self.A[u][v] = 0
        for from_vertex, to_vertex, weight in self.edges:
            self.A[from_vertex][to_vertex] = weight

        return

    def get_adjacent_nodes(self, node_tag):
        """
        Return all the nodes that are adjacent with the original node referenced by node tag
        """
        if node_tag in self.list_node_tags:
            return None

        return [(tag, self.get_node_by_nodetag(tag) for tag in self.list_node_tags if self.A[node_tag][tag] != 0)]

    def uncolored_nodes(self) -> List[str]:
        return [node_tag for node_tag in self.list_node_tags if self.get_node_by_nodetag(node_tag).color is None]