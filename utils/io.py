import json
import pandas as pd
from pandas import DataFrame

from extraction import Tube


def create_json_file(meta_txt_path, frames_dict_json_path):
    """
    In order to extract patches from a video, we create a dictionary call 'frames'
    About frames dictionary:
    - keys are index number of frame
    - values are list of objects that appear in the scene
    """
    frames = {}
    frame_indices = []
    frame_order = [0]  # frame id regarding to line
    num_frames = 0

    with open(meta_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            frame_index = line.split()[22][0:5]  # The last number of datetime
            frame_indices.append(frame_index)

        for i in range(0, len(frame_indices) - 1):
            if frame_indices[i] != frame_indices[i + 1]:
                num_frames += 1
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
    with open(frames_dict_json_path, "w") as f:
        json.dump(frames, f, indent=4)
    return frames


def load_tubes_from_json_file(frames_dict_json_path):
    tubes = []
    with open(frames_dict_json_path, "r") as f:
        frames_dict = json.load(f)

    total_num_frames = int(max(frames_dict.keys()))
    # Create a dictionary whose items follow this format object_id: [[frame_id, x, y, w, h, image_path], ...]
    objects_dict: dict = {}
    for frame_id, data in frames_dict.items():
        for value in data:
            object_id, x, y, w, h, image_path = value[: 6]
            if object_id not in objects_dict:
                objects_dict[object_id] = []
            objects_dict[object_id].append([frame_id, x, y, w, h, image_path])

    for object_id in objects_dict:
        if len(objects_dict[object_id]) < 10:
            continue  # remove shadows which is not an object
        observe_object = sorted(objects_dict[object_id], key=lambda item: item[0])  # sort by frame_id

        # Get the frame index of starting point and ending point
        start_frame = int(observe_object[0][0])
        end_frame = int(observe_object[-1][0])

        if end_frame - start_frame >= total_num_frames - 20:
            continue  # remove stationary objects

        tube = Tube(object_id, start_frame, end_frame)
        tubes.append(tube)

        for data in observe_object:
            tube.next_bounding_box(data[1], data[2], data[3], data[4])

    return tubes


def load_dataframe_from_json_file(json_path):
    columns = ["frame", "tag", "x", "y", "w", "h", "image_path"]
    with open(json_path, "r") as f:
        frames_dict = json.load(f)
    raw_data = []

    for frame_id, items in frames_dict.items():
        for item in items:
            object_id, x, y, w, h, image_path = item[:6]  # Take 6 first values
            tmp_list = [int(frame_id), int(object_id), int(x), int(y), int(w), int(h), image_path]
            raw_data.append(tmp_list)
    df: DataFrame = pd.DataFrame(raw_data, columns=columns)
    return df


def load_tubes_from_pandas_dataframe(df):
    tubes = []
    total_num_frames = df.frame.max()

    for tag in df.tag.unique():
        ob_df = df[df["tag"] == tag]
        if len(ob_df) < 10:
            continue  # remove shadows
        ob_df = ob_df.sort_values(by='frame')
        start_frame = ob_df.frame.min()
        end_frame = ob_df.frame.max()

        # TODO: Fix this check. It's better to determine if an object
        # is stationary using the coordinates instead of this.
        if end_frame - start_frame >= total_num_frames - 20:
            continue  # remove stationary objects

        tube = Tube(tag, start_frame, end_frame)
        tubes.append(tube)
        for _, r in ob_df.iterrows():
            tube.next_bounding_box(r['x'], r['y'], r['w'], r['h'])
    return tubes
