import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import cv2

from extraction import Tube
from aggregation.graph_building.relations import RuanRelationsMap
from aggregation.graph_building.graph import RuanGraph
from aggregation.graph_building.graph_coloring import GraphColoration

# Define 4 tubes as examples, play with toys first hehehe :))
tube1 = {
    0: [96, 80, 32, 64],
    1: [112, 80, 35, 60],
    2: [120, 104, 32, 64],
    3: [136, 112, 32, 63],
    4: [152, 128, 32, 62],
    5: [160, 120, 32, 61],
    6: [170, 128, 30, 60]
}

tube2 = {
    0: [80, 200, 40, 50],
    1: [90, 180, 35, 50],
    2: [120, 160, 40, 55],
    3: [120, 130, 40, 50],
    4: [120, 100, 41, 52],
    5: [100, 70, 41, 50]
}

tube3 = {
    0: [200, 210, 40, 80],
    1: [250, 230, 38, 75],
    2: [280, 245, 40, 75],
    3: [320, 280, 40, 80],
    4: [330, 300, 43, 77],
    5: [350, 310, 40, 75],
    6: [370, 320, 40, 75],
    7: [380, 330, 37, 78]
}
tube4 = {
    0: [210, 170, 30, 70],
    1: [245, 150, 30, 75],
    2: [265, 145, 40, 75],
    3: [270, 130, 40, 80],
    4: [280, 120, 43, 77]
}


# Define background image
def create_background():
    background = np.zeros((500, 500, 3)).astype(np.uint8)
    background.fill(255)
    return background


# Draw tubes
def draw_tubes(tube: dict, input_background, color=(255, 0, 0)):
    tube_image = np.array(input_background)
    for frame_id, coors in tube.items():
        top_left = (coors[0], coors[1])
        bottom_right = (coors[0] + coors[2], coors[1] + coors[3])
        center = (coors[0] + coors[2] // 2, coors[1] + coors[3] // 2)
        tube_image = cv2.circle(tube_image, center, radius=0, color=(125, 125, 125), thickness=3)
        cv2.putText(tube_image, str(frame_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        tube_image = cv2.rectangle(tube_image, top_left, bottom_right, color=color)
    return tube_image


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


if __name__ == "__main__":
    background = create_background()
    # Draw tubes
    # tube1_image = draw_tubes(tube1, background)
    # tube2_image = draw_tubes(tube2, tube1_image, (0, 0, 255))
    # tube3_image = draw_tubes(tube3, tube2_image, (0, 255, 0))
    # tube4_image = draw_tubes(tube4, tube3_image, (255, 0, 255))
    # plt.imshow(tube4_image)
    # plt.show()
    list_tubes = [tube1, tube2, tube3, tube4]
    tubes = create_tubes(list_tubes)
    relation_map = RuanRelationsMap(tubes, vectorized_computation=False)
    graph_coloration = GraphColoration(3)
    graph = RuanGraph(tubes=tubes, relations=relation_map)
    graph = graph_coloration.color_graph(graph=graph)
    print(graph.nodes)
    current_starting_times = graph_coloration.tube_starting_time(graph)
    pp.pprint(current_starting_times)
