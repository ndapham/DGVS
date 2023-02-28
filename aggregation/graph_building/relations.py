from abc import ABC
from itertools import permutations
import torch


class AbstractRelations(ABC):
    def __init__(self, tubes):
        self.tubes = tubes
        self.relations = {}
        self._fill_with_irrelevant_relations()
        self.compute_relations()

    def _fill_with_irrelevant_relations(self):
        for Ta in self.tubes:
            self.relations[Ta.tag] = {}
            for Tb in self.tubes:
                self.relations[Ta.tag][Tb.tag] = None

    def compute_relations(self):
        raise Exception("Using default compute relations function")


class RuanRelationsMap(AbstractRelations):
    """
    Implementation based on the instruction in Ruan et al. 2019 paper
    """

    def __init__(self, tubes, vectorized_computation=False):
        super(RuanRelationsMap, self).__init__(tubes)
        self.vectorized_computation = vectorized_computation

    def compute_relations(self):
        if self.vectorized_computation:
            self.compute_relations_by_matrix()
        else:
            self.compute_relations_by_loops()

    # Compute relations among tubes using loops
    def compute_relations_by_loops(self):
        n = len(self.tubes)

        # Using 4 nested loops, do not recommend , but it leads to better memory usage compares to matrices
        for Ta, Tb in permutations(self.tubes):
            if Ta == Tb:
                continue
            for a_data in Ta:
                for b_data in Tb:
                    if self._frame_intersect(a_data, b_data):
                        src_frame = a_data[4]  # a_data: x, y, w, h, frame_id
                        trg_frame = b_data[4]  # b_data: x, y, w, h, frame_id
                        if self.relations[Ta.tag][Tb.tab] is None:
                            self.relations[Ta.tag][Tb.tag] = []
                        self.relations[Ta.tag][Tb.tag].append((src_frame, trg_frame))

            #  Utilize computed relations
            if self.relations[Tb.tag][Ta.tag] is not None:
                self.relations[Ta.tag][Tb.tag] = [(item[1], item[0]) for item in self.relations[Ta.tag][Tb.tag]]

    def _frame_intersect(self, src_frame_data, trg_frame_data):
        src_x, src_y, src_w, src_h, _ = src_frame_data
        trg_x, trg_y, trg_w, trg_h, _ = trg_frame_data

        # Check frame overlapping
        src_top_left_x, src_top_left_y = src_x, src_y
        src_bottom_right_x, src_bottom_right_y = src_x + src_w, src_y + src_h

        trg_top_left_x, trg_top_left_y = trg_x, trg_y
        trg_bottom_right_x, trg_bottom_right_y = trg_x + trg_w, trg_y + trg_h

        condition1 = src_top_left_y - tr

    # Compute relations among tubes using tensors
    def compute_relations_by_matrix(self):
        pass
