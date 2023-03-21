from os import access, path
import datetime
import cv2
import numpy as np
from PIL import Image
from fusion.interpolation import complete_frames


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
