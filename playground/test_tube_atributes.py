import pprint

from extraction import Tube
from playground import tube1, tube2, tube3, tube4

list_tubes = [tube1, tube2, tube3, tube4]


def create_tubes(list_tubes):
    """
    Convert tube from dictionary to Tube type
    """
    tubes = []
    for index, tube in enumerate(list_tubes):
        tag = index
        s_frame = list(tube.keys())[0]
        e_frame = list(tube.keys())[-1]
        tmp_tube = Tube(tag, s_frame, e_frame)
        tubes.append(tmp_tube)
        for r in tube.values():
            tmp_tube.next_bounding_box(r[0], r[1], r[2], r[3])
    return tubes


tubes = create_tubes(list_tubes)

for frame_index, frame_data in enumerate(tubes[0]):
    print(type(frame_data))
    print(type(frame_index))
    pprint.pprint(f'{frame_index}: {frame_data}')
