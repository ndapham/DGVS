import cv2
from PIL import Image


def frame_intersect(src_frame_data, trg_frame_data):
    src_x, src_y, src_w, src_h, _ = src_frame_data
    trg_x, trg_y, trg_w, trg_h, _ = trg_frame_data

    # Check frame overlapping
    src_top_left_x, src_top_left_y = src_x, src_y
    src_bottom_right_x, src_bottom_right_y = src_x + src_w, src_y + src_h

    trg_top_left_x, trg_top_left_y = trg_x, trg_y
    trg_bottom_right_x, trg_bottom_right_y = trg_x + trg_w, trg_y + trg_h

    condition1 = (src_top_left_y - trg_bottom_right_y) * (src_bottom_right_y - trg_top_left_y) < 0
    condition2 = (src_top_left_x - trg_bottom_right_x) * (src_bottom_right_x - trg_top_left_x) < 0
    return condition1 and condition2


def get_video_shape(background_path: str):
    image = Image.open(background_path)
    return image.width, image.height


def get_video_nframes(video_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return int(w), int(h)
