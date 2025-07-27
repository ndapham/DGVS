import os
from os import path
import time
import datetime
import json
from itertools import permutations

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from pcgvs.extraction import Tube, load_tubes_from_pandas_dataframe
from pcgvs.aggregation.relations import Overlapping, Intersection
from pcgvs.aggregation.graph import PCG
from pcgvs.aggregation.coloring import color_graph
from pcgvs.aggregation.coloring import tubes_starting_time
from pcgvs.aggregation import solve, add_ss_to_dataframe
from pcgvs.synopsis.interpolation import complete_frames
from pcgvs.synopsis import _get_video_shape
from pcgvs.metrics import FR, CR, OR
from pcgvs.utils import get_video_nframes

meta_txt_path = "/home/pducanh/Desktop/pcgvs/data/meta.txt"
output_json_path = "/home/pducanh/Desktop/pcgvs/data/meta.json"

patches_folder = "/home/pducanh/Desktop/cam2/"
metadata_folder = "/home/pducanh/Desktop/pcgvs/data/"
background_path = "/home/pducanh/Desktop/pcgvs/data/background.jpg"
interp = True


def create_frame_json_file(meta_txt_path, output_json_path):
    frames = {}
    frame_idices = []
    frame_order = [0]  # The frame id regard to line
    num_frames = 0

    with open(meta_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            frame_index = line.split()[22][0:5]  # the last number in datetime
            frame_idices.append(frame_index)

        for i in range(0, len(frame_idices) - 1):
            if frame_idices[i] != frame_idices[i + 1]:
                num_frames += 1
                frame_order.append(num_frames)
            else:
                frame_order.append(num_frames)

        for i, num_frames in enumerate(frame_order):
            line = lines[i]
            object_id = line.split()[3][0:5]
            x = line.split()[7][1:-1]
            y = line.split()[8][0:-1]
            w = line.split()[9][0:-1]
            h = line.split()[10][0:-2]
            image_path = line.split()[12].split('/')[-1][:-2]

            if int(num_frames) not in frames.keys():
                frames[int(num_frames)] = []
            frames[int(num_frames)].append([object_id, int(x), int(y), int(w), int(h), image_path])

    with open(output_json_path, "w") as f:
        json.dump(frames, f, indent=4)

    return frames


def load_tubes_from_json_file(json_path):
    columns = ["frame", "tag", "x", "y", "w", "h", "image_path"]
    with open(json_path, "r") as f:
        frames_dict = json.load(f)

    raw_data = []
    for frame_id, items in frames_dict.items():
        for item in items:
            object_id, x, y, h, w, image_path = item[0], item[1], item[2], item[3], item[4], item[5]
            tmp_list = [int(frame_id), int(object_id), int(x), int(y), int(h), int(w), image_path]
            raw_data.append(tmp_list)
    df = pd.DataFrame(raw_data, columns=columns)

    return df


class RelationsMap:

    def __init__(self, tubes):
        self.tubes = tubes
        self.relations = {}
        self._fill_with_irrilevant_relations()
        self._compute()

    # @staticmethod
    # def precomputed_from_json(tubes, path):
    #     try:
    #         with open(path, 'r') as jsonfile:
    #             relations = json.load(jsonfile)
    #             rmap = RelationsMap(tubes)
    #             rmap.relations = relations
    #             return rmap
    #     except:
    #         raise Exception('File not found.')

    def _fill_with_irrilevant_relations(self):
        for Ta in self.tubes:
            self.relations[Ta.tag] = {}
            for Tb in self.tubes:
                self.relations[Ta.tag][Tb.tag] = None

    def _compute(self):
        n = len(self.tubes)
        for Ta, Tb in tqdm(permutations(self.tubes, 2), total=n * (n - 1)):
            if Ta == Tb: continue
            # we focus on tube A and check the intersections with Tube B.
            ffintersec = None  # first frame of intersection.
            lfintersec = None  # last frame of intersection.

            for adata in Ta:
                for bdata in Tb:
                    frame = adata[4]
                    if self._frames_intersect(adata, bdata):
                        ffintersec = frame if ffintersec is None else ffintersec
                        lfintersec = frame

            # In this case, there isn't interaction.
            if lfintersec is None: continue
            # Following the paper recommendations, we
            # set the interaction as overlapping if there
            # are more than 5 intersecting frames.
            delta = lfintersec - ffintersec

            if self.relations[Tb.tag][Ta.tag] is not None:
                # If Tb-Ta relation is computed as INT or OVL, we need to
                # stick with the previous type of relation.
                prel = self.relations[Tb.tag][Ta.tag]
                self.relations[Ta.tag][Tb.tag] = Intersection(ffintersec) \
                    if type(prel) == Intersection \
                    else Overlapping(ffintersec, lfintersec)
            else:
                self.relations[Ta.tag][Tb.tag] = Intersection(ffintersec) \
                    if lfintersec - ffintersec < 5 \
                    else Overlapping(ffintersec, lfintersec)

    def _frames_intersect(self, adata, bdata):
        xa, ya, wa, ha, _ = adata
        xb, yb, wb, hb, _ = bdata
        l_ax, l_ay = xa, ya  # Top-left point of square A
        r_ax, r_ay = xa + wa, ya + ha  # Bottom-right point of square A
        l_bx, l_by = xb, yb  # Top-left point of square B
        r_bx, r_by = xb + wb, yb + hb  # Bottom-right point of square B
        # Check if one square has empty area.
        if l_ax == r_ax or l_ay == r_ay or l_bx == r_bx or l_by == r_by:
            return False
        # Check if one square stands above the other.
        if r_ay < l_by or r_by < l_ay:
            return False
        # Check if one square stands on the left of the other
        if r_ax < l_bx or r_bx < l_ax:
            return False
        # The squares overlap!
        return True

    def as_dict(self):
        return self.relations

    def save_as_json_dict(self, path):
        relations_dict = {}
        for k1 in self.relations.keys():
            relations_dict[int(k1)] = {}
            for k2 in self.relations.keys():
                value = self.relations[k1][k2]
                relations_dict[int(k1)][int(k2)] = value.__str__() if value is not None else "None"
        with open(path, "w") as f:
            json.dump(relations_dict, f, indent=4)

        return relations_dict

    def __str__(self):
        out = ""
        for k1 in self.relations.keys():
            out += f'[{k1}]:\t'
            for k2 in self.relations.keys():
                out += f'({k2}){self.relations[k1][k2]}\t'
            out += '\n'
        return out


def find_q_value(relations_dict, num_object):
    num_ovl_int = 0
    num_none = 0
    for k1 in relations_dict.keys():
        for k2 in relations_dict.keys():
            relation = relations_dict[k1][k2]
            if relation.endswith("OVL") or relation.endswith("INT"):
                num_ovl_int += 1
            else:
                num_none += 1
    q = 3   # for pedestrians as recommended
    overlap_count = num_ovl_int / 2
    if overlap_count > (num_object / 10):
        q = 15
    elif overlap_count > (num_object / 13):
        q = 8
    elif overlap_count > (num_object / 16):
        q = 3

    return q


def save_starting_times_json(starting_times, starting_times_json_path):
    with open(starting_times_json_path, 'w') as f:
        writable_json = {str(k): v for k, v in starting_times.items()}
        json.dump(writable_json, f, indent=4)
    return


def generate_frames(dataframe, patches_path):
    frames = {}
    for idx, row in dataframe.iterrows():
        nf = int(row['newframe'])
        if nf not in frames: frames[nf] = []
        #         patchpath = path.join(patches_path, f'{row["tag"]}_{row["frame"]}.png')
        patchpath = os.path.join(patches_path, f'{row["image_path"]}')
        frames[nf].append({
            'tag': int(row['tag']),
            'file': patchpath,
            'x': int(row['x']),
            'y': int(row['y']),
            'w': int(row['w']),
            'h': int(row['h']),
            'frame': int(row['frame'])
        })
    return frames


def generate_synopsis(frames, output_dir, fps, bgpath, interp=False):
    """
    """
    output = path.join(output_dir, 'synopsis.avi')
    _frames = frames.copy()
    max_frame = max(list(_frames.keys()))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width, height = _get_video_shape(bgpath)

    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    if interp: _frames = complete_frames(_frames)

    for num_frame in range(1, max_frame + 1):
        frame = cv2.imread(bgpath)
        #         print(frame.shape)
        #         print("=" * 50)

        if num_frame in _frames.keys():
            objects = sorted(_frames[num_frame], key=lambda d: d['tag'], reverse=True)
            for obj in objects:
                fr = obj.get('frame')
                raw_img = cv2.imread(obj.get('file'), cv2.IMREAD_UNCHANGED)

                m_img = raw_img[..., -1] / 255
                s_img = raw_img[..., :-1]
                #                 print(m_img.shape)
                #                 print(s_img.shape)

                x = int(obj.get('x'))
                y = int(obj.get('y'))
                w = int(obj.get('w'))
                h = int(obj.get('h'))
                time = str(datetime.timedelta(seconds=int(fr / 30)))
                #                 print(f"y from {y} to {y+s_img.shape[0]} \nx from {x} to {x+s_img.shape[1]}")

                y_start = y
                y_end = max(min(y + s_img.shape[0], 1080), 0)
                x_start = x
                x_end = max(min(x + s_img.shape[1], 1920), 0)
                frame[y_start:y_end, x_start:x_end] = np.uint8(
                    frame[y_start:y_end, x_start:x_end] * (1 - m_img[..., None])[:y_end - y_start,
                                                          :x_end - x_start] / 3 +
                    (m_img[..., None] * raw_img)[:y_end - y_start, :x_end - x_start] * 1.5)

                cv2.rectangle(frame, (x, y), (x_end, y_end), thickness=1, color=(214, 73, 51))
                cv2.putText(frame, time, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (244, 43, 3), 2)
                cv2.putText(frame, str(obj.get("tag")), (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (43, 244, 3), 2)

        try:
            out.write(frame)
        except:
            continue
    out.release()


# DucAnh-Vectorized computation================================================================================================
def padding(vector_tube, max_len):
    """
    params:
    vector_tube: record the location of object through frames-shape:(num_frames, 4)
                (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    max_len: lenght of the vector after padding default-lenght of the whole video,
             it should be max lenght of vectors for better memory usage.
    """
    padded_vector = F.pad(vector_tube, pad=(0, 0, 0, max_len - vector_tube.shape[0]), mode="constant", value=0)
    return padded_vector


def create_matrix_tubes(tubes: list, max_len) -> torch.tensor:
    """
    Given a list of Tube objects
    return a matrix represent all the information about location of objects
    through frames (num_tubes, num_frames, 4)
    """
    matrix_tube = []
    for tube in tubes:
        # Convert list to torch tensor
        x_tensor = torch.tensor(tube.bbX)
        y_tensor = torch.tensor(tube.bbY)
        w_tensor = torch.tensor(tube.bbW)
        h_tensor = torch.tensor(tube.bbH)

        # Create a vector_tube represents an action (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        vector_tube = torch.stack((x_tensor,
                                   y_tensor,
                                   x_tensor + w_tensor,
                                   y_tensor + h_tensor),
                                  dim=0
                                  )
        vector_tube = vector_tube.transpose(0, 1)
        matrix_tube.append(padding(vector_tube, max_len))
    matrix_tube = torch.stack(tuple(matrix_tube), dim=0)
    return matrix_tube


def rearrange(matrix_tube: torch.tensor):
    rearranged_index = torch.tensor([2, 3, 0, 1])
    rearranged_matrix = torch.zeros_like(matrix_tube)

    rearranged_matrix[:, :, rearranged_index] = matrix_tube
    return rearranged_matrix


def check_overlap(matrix_tube: torch.tensor):
    print(matrix_tube.shape)
    num_tubes, num_frames = matrix_tube.shape[:2]

    rearranged_matrix = rearrange(matrix_tube)
    rearranged_matrix = rearranged_matrix.view(-1, 4)
    # Using broadcast instead of loops
    overlaps = matrix_tube.view(-1, 4).unsqueeze(1) - rearranged_matrix

    # Checking overlaps
    overlaps = overlaps.sign()
    overlaps = overlaps[:, :, 0] * overlaps[:, :, 2] + overlaps[:, :, 1] * overlaps[:, :, 3]
    overlaps = overlaps < -1

    # Return the shape of (num_tubes, num_tubes, num_frames, num_frames)
    overlaps = overlaps.view(overlaps.shape[0], num_tubes, num_frames)
    overlaps = overlaps.view(num_tubes, num_frames, num_tubes, num_frames)
    overlaps = overlaps.transpose(1, 2)

    # Remove overlapping between 2 frames in the same tube
    overlaps = (overlaps == True).nonzero(as_tuple=False)
    overlaps = overlaps[overlaps[:, 0] != overlaps[:, 1]]
    return overlaps


class TestRelationsMap(object):
    def __init__(self, tubes, overlap_threshold=5):
        self.tubes = tubes
        self.max_len = None
        self.overlap_threshold = overlap_threshold
        self.relations = {}
        self._init_relations()
        self._compute()

    def _init_relations(self):
        for Ta in self.tubes:
            self.relations[Ta.tag] = {}
            for Tb in self.tubes:
                self.relations[Ta.tag][Tb.tag] = None

    def get_max_len_tube(self):
        get_tube_length = lambda tube: tube.eframe - tube.sframe
        max_len = max(get_tube_length(tube) for tube in self.tubes)
        return max_len

    def _compute(self):
        self.max_len = self.get_max_len_tube()
        st = time.time()
        matrix_tube = create_matrix_tubes(self.tubes, self.max_len)
        et = time.time()

        indices_overlap = check_overlap(matrix_tube)
        print("Check overlap: ", et - st)

        # Using loops :) im stupid :(
        n = len(self.tubes)
        for Ta, Tb in tqdm(permutations(self.tubes, 2), total=n * (n - 1)):
            if Ta == Tb:
                continue

            first_frame_intersect = None
            last_frame_intersect = None
            # frames_intersect = indices_overlap[torch.logical_and(indices_overlap[:, 0] == Ta.tag,
            #                                    indices_overlap[:, 1] == Tb.tag)]
            frames_intersect = indices_overlap[indices_overlap[:, 0] == Ta.tag]
            frames_intersect = frames_intersect[frames_intersect[:, 1] == Tb.tag]
            # If 2 tubes have collisions
            if frames_intersect.shape[0]:
                first_frame_intersect = frames_intersect[0, 2]
                last_frame_intersect = frames_intersect[-1, 2]
            else:
                continue

            # If Tb - Ta relation is already computed, we should stick with the
            # previous relation
            if self.relations[Tb.tag][Ta.tag] is not None:
                pre_relation = self.relations[Tb.tag][Ta.tag]

                self.relations[Ta.tag][Tb.tag] = Intersection(first_frame_intersect) \
                    if type(pre_relation) == Intersection \
                    else Overlapping(first_frame_intersect, last_frame_intersect)
            else:
                self.relations[Ta.tag][Tb.tag] = Intersection(first_frame_intersect) \
                    if (last_frame_intersect - first_frame_intersect) < self.overlap_threshold \
                    else Overlapping(first_frame_intersect, last_frame_intersect)

    def as_dict(self):
        return self.relations

    def save_as_json_dict(self, path):
        relations_dict = {}
        for k1 in self.relations.keys():
            relations_dict[int(k1)] = {}
            for k2 in self.relations.keys():
                value = self.relations[k1][k2]
                relations_dict[int(k1)][int(k2)] = value.__str__() if value is not None else "None"
        with open(path, "w") as f:
            json.dump(relations_dict, f, indent=4)

        return relations_dict

    # For double-checking results only, this is unnecessary
    def __str__(self):
        out = ""
        for k1 in self.relations.keys():
            out += f'[{k1}]:\t'
            for k2 in self.relations.keys():
                out += f'({k2}){self.relations[k1][k2]}\t'
            out += '\n'
        return out


# ================================================================================================

def aggregation_synopsis(meta_txt_path, background_path, patches_folder, metadata_folder, interp=True):
    num_tubes_in_use = 20  # Set this to -1 to run with all the tubes

    # AGGREGATION
    output_json_path = os.path.join(metadata_folder, "meta.json")
    frames = create_frame_json_file(meta_txt_path, output_json_path)
    dataframe = load_tubes_from_json_file(output_json_path)

    print("Computing tubes")
    tubes = load_tubes_from_pandas_dataframe(dataframe)
    tube_tags = [k.tag for k in tubes]
    print("Num_tags: ", len(set(tube_tags)))
    print(tubes[0].__dict__)
    print("Num Tubes: ", len(tubes))

    print("Calculating the relations map")
    relations = TestRelationsMap(tubes[: num_tubes_in_use])
    relations_dict_json_path = os.path.join(metadata_folder, "relations_dict.json")
    relations_dict = relations.save_as_json_dict(relations_dict_json_path)

    print("Finding q value")
    num_object = len(dataframe.index)
    num_object = len(relations_dict.keys())  # Redundancy-Duc Anh

    q = find_q_value(relations_dict, num_object)
    q = 20

    print("Generating the potential collision graph")
    pcg = PCG(tubes[: num_tubes_in_use], relations)

    print("Applying graph coloring algorithm")
    color_graph(pcg, q)
    starting_times = tubes_starting_time(pcg, q)

    starting_times_json_path = os.path.join(metadata_folder, "starting_times.json")
    save_starting_times_json(starting_times, starting_times_json_path)

    # SYNOPSIS
    df = add_ss_to_dataframe(dataframe, tubes[:  num_tubes_in_use], starting_times)
    frames = generate_frames(df, patches_folder)
    generate_synopsis(frames, metadata_folder, 30, background_path, interp)
    print(f'Video synopsis generated with q={q}')

    # spath = './synopsis/synopsis.avi'
    # vpath = './data-simulation/export_20220523_210923.mp4'
    #
    # # metrics
    # _FR = FR(spath, vpath)
    # _CR = CR(spath, frames)
    # _OR = OR(spath, frames)
    #
    # return _FR, _CR, _OR


if __name__ == "__main__":
    aggregation_synopsis(meta_txt_path, background_path, patches_folder, metadata_folder, interp=True)
