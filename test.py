import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import cv2

from playground import playground_tubes, tube1, tube2, tube3, tube4
from aggregation.graph_building.relations import RuanRelationsMap
from aggregation.graph_building.graph import RuanGraph
from aggregation.graph_building.graph_coloring import GraphColoration


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


if __name__ == "__main__":
    background = create_background()
    # Draw tubes
    # tube1_image = draw_tubes(tube1, background)   # red
    # tube2_image = draw_tubes(tube2, tube1_image, (0, 0, 255)) # blue
    # tube3_image = draw_tubes(tube3, tube2_image, (0, 255, 0)) # green
    # tube4_image = draw_tubes(tube4, tube3_image, (255, 0, 255))   # purple
    # plt.imshow(tube4_image)
    # plt.show()
    tubes = playground_tubes
    relation_map = RuanRelationsMap(tubes)
    graph_coloration = GraphColoration(3)
    graph = RuanGraph(tubes=tubes, relations=relation_map)
    graph = graph_coloration.color_graph(graph=graph)
    current_starting_times = graph_coloration.tube_starting_time(graph)
    pp.pprint(current_starting_times)
