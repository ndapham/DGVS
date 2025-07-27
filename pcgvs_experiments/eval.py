import cv2
import numpy as np
from pcgvs.extraction import extract_tubes, extract_patches, extract_background, load_tubes_from_pandas_dataframe, \
    load_tubes_with_pandas
from pcgvs.aggregation import solve, add_ss_to_dataframe
from pcgvs.aggregation.relations import RelationsMap
from pcgvs.aggregation.graph import PCG
from pcgvs.aggregation.coloring import color_graph, tubes_starting_time
from pcgvs.synopsis import generate_frames, generate_synopsis
import pandas as pd

# Input original video
vpath = "/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/Video_input/video.mp4"
# Input synopsis video
spath = "./synopsis/synopsis.avi"
# Input image patches crop from object tracking
patches_path = "/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/patches"

# Extraction
# tubes_path='/Users/nguyenduy/Desktop/pcgvs-main/notebooks/data/test.txt'
# patches_path = extract_patches(source=i, outputdir=o, path_tubes=tubes_path)
# background_path = extract_background(source=i, outputdir=o, path_tubes=tubes_path)


df = pd.read_csv("/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/data-benchmark-simulation/eval.csv", sep="\t")
frames = generate_frames(df, patches_path)


def _get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def _get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return int(w), int(h)


# Frame condensation ratio (FR):
def FR(synopsis_video_path: str, original_video_path: str):
    """ Frame condensation ratio """
    Ts = _get_video_duration(synopsis_video_path)
    Ti = _get_video_duration(original_video_path)
    return Ts / Ti


print(f'FR is {FR(spath, vpath):.3f}')


# Frame compact rate (CR):
def CR(synopsis_video_path: str, frames: dict):
    """ Frame compact rate """
    w, h = _get_video_resolution(synopsis_video_path)
    multiplier = 1 / (w * h * len(frames))
    foreg_area = 0
    for objects in frames.values():
        for obj in objects:
            x, y, _w, _h = obj['x'], obj['y'], obj['w'], obj['h']
            foreg_area += min(_w, w - x) * min(_h, h - y)
    _CR = multiplier * foreg_area
    return _CR


print(f'CR is {CR(spath, frames):.3f}')


# Overlap ratio (OR):
def OR(synopsis_video_path: str, frames: dict):
    """ Overlap ratio """
    w, h = _get_video_resolution(synopsis_video_path)
    multiplier = 1 / (w * h * len(frames))
    overlap_area = 0
    for objects in frames.values():
        F = np.zeros((w, h))
        for obj in objects:
            x, y, _w, _h = obj['x'], obj['y'], obj['w'], obj['h']
            F[x:(x + _w), y:(y + _h)] += np.ones((min(_w, w - x), min(_h, h - y)))
        overlap_area += (F > 1).sum()
    _OR = multiplier * overlap_area
    return _OR


print(f'OR is {OR(spath, frames):.3f}')

# # Aggregation
# dataframe = load_tubes_with_pandas(tubes_path)
# tubes = load_tubes_from_pandas_dataframe(dataframe)
# relations = RelationsMap(tubes)
# pcg = PCG(tubes, relations)
# color_graph(pcg, q)
# starting_times = tubes_starting_time(pcg, q)
