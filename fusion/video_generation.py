from os import access, path
import datetime
import cv2
import numpy as np
from PIL import Image
from fusion.interpolation import complete_frames
from utils.helpers import get_video_shape


def generate_frames(dataframe, patches_path):
    frames = {}
    for idx, row in dataframe.iterrows():
        nf = int(row['newframe'])
        if nf not in frames:
            frames[nf] = []
        patch_path = path.join(patches_path, f'{row["tag"]}_{row["frame"]}.png')
        frames[nf].append({
            'tag': int(row['tag']),
            'file': patch_path,
            'x': int(row['x']),
            'y': int(row['y']),
            'h': int(row['h']),
            'frame': int(row['frame'])
        })
    return frames


def generate_synopsis(frames, output_dir, fps, background_path, interp=False):
    """
    Generate the final video based on frame data which comes from aggregation
    """
    output = path.join(output_dir, "synopsis.avi")
    _frames = frames.copy()
    max_frame = max(list(_frames.keys()))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width, height = get_video_shape(background_path)
