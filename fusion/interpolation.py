import numpy as np


def _unique_tags(frames):
    tags = set()
    for frame_content in frames.values():
        for object_data in frame_content:
            tags.add(object_data.get("tag"))
    return tags


def extract_frames_by_tag(frames, tag):
    keys = list(frames.keys())
    keys.sort()
    tag_frames = {}
    for new_frame in keys:
        for data in frames[new_frame]:
            if tag == data.get("tag"):
                tag_frames[new_frame] = data.copy()
                break

    return tag_frames


def params_to_interpolate_by_tag(tag_frames):
    x, y, f = [], [], []
    for content in tag_frames.values():
        x.append(content.get("x"))
        y.append(content.get("y"))
        f.append(content.get("f"))
    return x, y, f


def complete_frames(frames):
    """
    Complete the frames dictionary interpolating the missing bounding boxes
    """
    tags = _unique_tags(frames)
    last_frame = max(frames.keys())
    interpolated_frames = {i: [] for i in range(1, last_frame + 1)}
    for tag in tags:
        tag_frames = extract_frames_by_tag(frames, tag)
        frames_of_tag = list(tag_frames.keys())
        frames_of_tag.sort()

        # X, Y, F = params_to_interpolate_by_tag(tag_frames)
        for i in range(1, len(frames_of_tag)):
            prev_frame = frames_of_tag[i-1]
            curr_frame = frames_of_tag[i]
            xp = [prev_frame, curr_frame]
            ypX = [tag_frames[prev_frame]["x"], tag_frames[curr_frame]["x"]]
            ypY = [tag_frames[prev_frame]["y"], tag_frames[curr_frame]["y"]]

            if curr_frame == prev_frame + 1:
                continue

            for j in range(prev_frame + 1, curr_frame):
                x_pred = np.interp(j, xp, ypX)
                y_pred = np.interp(j, xp, ypY)

                tag_frames[j] = tag_frames[prev_frame].copy()
                tag_frames[j]["x"] = x_pred
                tag_frames[j]["y"] = y_pred

        for frame in range(1, last_frame):
            if frame not in tag_frames:
                continue
            interpolated_frames[frame].append(tag_frames[frame])
    return interpolated_frames
