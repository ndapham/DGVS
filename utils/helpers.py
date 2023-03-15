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
