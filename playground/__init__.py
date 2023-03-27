from extraction import Tube

# Define 4 tubes as examples
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

list_tubes = [tube1, tube2, tube3, tube4]


def create_tubes(list_tubes, duplicate=1):
    """
    Convert tube from dictionary to Tube type
    """
    tubes = []
    list_tubes = list_tubes * duplicate
    for index, tube in enumerate(list_tubes):
        tag = index
        s_frame = list(tube.keys())[0]
        e_frame = list(tube.keys())[-1]
        tmp_tube = Tube(tag, s_frame, e_frame)
        tubes.append(tmp_tube)
        for r in tube.values():
            tmp_tube.next_bounding_box(r[0], r[1], r[2], r[3])
    return tubes


playground_tubes = create_tubes(list_tubes * 50)
