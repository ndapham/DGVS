import numpy as np

from utils.helpers import get_video_duration, get_video_resolution


def get_fr(synopsis_video_path: str, original_video_path: str):
    """
    Frame condensation ratio
    """
    synopsis_length = get_video_duration(synopsis_video_path)
    original_length = get_video_duration(original_video_path)

    return synopsis_length / original_length


def get_cr(synopsis_video_path: str, frames: dict):
    """Frame compact ratio"""
    w, h = get_video_resolution(synopsis_video_path)
    multiplier = 1 / (w * h * len(frames))
    foreground_area = 0
    for objects in frames.values():
        for obj in objects:
            x, y, _w, _h = obj["x"], obj["y"], obj["w"], obj["h"]
            foreground_area += min(_w, w - x) * min(_h, h - y)
    compact_ratio = multiplier * foreground_area
    return compact_ratio


def get_or(synopsis_video_path: str, frames: dict):
    """Overlap ratio"""
    w, h = get_video_resolution(synopsis_video_path)
    multiplier = 1 / (w * h * len(frames))
    overlap_area = 0
    for objects in frames.values():
        frame_mask = np.zeros((w, h))
        for obj in objects:
            x, y, _w, _h = obj['x'], obj['y'], obj['w'], obj['h']
            frame_mask[x: (x + _w), y: (y + _h)] += np.ones((min(_w, w - x), min(_h, h - y)))
        overlap_area += (frame_mask > 1).sum()
    overlap_ratio = multiplier * overlap_area
    return overlap_ratio
