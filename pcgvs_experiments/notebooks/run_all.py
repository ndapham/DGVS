import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from torch.nn import functional as F
import torch, detectron2

from pcgvs.extraction import extract_patches
import glob
from IPython.display import Image, display
import random
from pcgvs.extraction import extract_background
import json

# input video
# input_vid="/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/Video_input/video4.mp4"

# input image patches crop from object tracking
path = "/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/patches"

# output video synopsis path
output_vid = "/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis"

# object txt input from deepsort
input_txt = "/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/data_input_txt_object/meta.txt"

# input background
input_background = "/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/background.png"

frames = {}
frame_id = []
f1 = [0]
count = 0

with open(input_txt, 'r') as f:
    for line in f:
        f = line.split()[22][0:5]
        frame_id.append(f)

    for i in range(0, len(frame_id) - 1):
        if (frame_id[i] != frame_id[i + 1]):
            count += 1
            f1.append(count)
        else:
            f1.append(count)
    for i in f1:
        id = line.split()[3][0:5]
        x = line.split()[7][1:-1]
        y = line.split()[8][0:-1]
        w = line.split()[9][0:-1]
        h = line.split()[10][0:-2]

        if int(i) not in frames.keys():
            frames[int(i)] = []
        frames[int(i)].append([id, int(x), int(y), int(w), int(h)])

# ectract images form video
# extract_patches(
#     path_tubes=input_txt, 
#     source=input_vid, 
#     outputdir=output_vid
# )

# images = []
# for imageName in glob.glob('/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/patches/*.png'):
#     images.append(imageName)

# # for _ in range(10):
# #     display(
# #         Image(
# #             filename = images[random.randint(0, len(images) - 1)]
# #         )
# # )

# extract_background
# extract_background(
#     path_tubes=input_txt, 
#     source=input_vid, 
#     outputdir=output_vid
# )
# display(Image(filename = '/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/background.png'))


from os import path, getcwd
from pcgvs.extraction import load_tubes_with_pandas

# Generating an OS safe path.
p = path.join(getcwd(), 'synopsis')
p = path.join(p, 'tubes')
p = path.join(p, 'exp')
p = path.join(p, 'tracks')
p = path.join(p, input_txt)

dataframe = load_tubes_with_pandas(p)
# dataframe.head(10)


from pcgvs.extraction import load_tubes_from_pandas_dataframe

tubes = load_tubes_from_pandas_dataframe(dataframe)
# for tube in tubes:
#     print(f'tube of object {tube.tag} starts at frame {tube.sframe} and end at frame {tube.eframe}')

from pcgvs.aggregation.relations import RelationsMap

relations = RelationsMap(tubes)

# write overlap frame for set threshold q
f = open("/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/overlap_frame.txt", "w")
f.write(str(relations))

f = open("/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/overlap_frame.txt", "r")
string = f.read()
list = string.split("\t")
f.close()

list_OVL = []
list_count = []
for i in list:
    list_OVL.append(i[4:])
for j in list_OVL:
    if j == 'OVL' or j == 'VL' or j == 'INT':
        list_count.append('1')

num_object = len(dataframe.index)
overlap_count = len(list_count) / 2
if overlap_count > (num_object / 10):
    q = 15
elif overlap_count > (num_object / 13):
    q = 8
elif overlap_count > (num_object / 16):
    q = 3

# print(q)

from pcgvs.aggregation.graph import PCG

pcg = PCG(tubes, relations)

from pcgvs.aggregation.coloring import color_graph

color_graph(pcg, q)

from pcgvs.aggregation.coloring import tubes_starting_time

starting_times = tubes_starting_time(pcg, q)
# starting_times

import json

with open('/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/starting_times.json', 'w') as ssfile:
    writablejson = {str(k): v for k, v in starting_times.items()}
    json.dump(writablejson, ssfile)

# !nvcc --version
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)


images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.png']

for i in images:

    im = cv2.imread(path + str(i))
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # We can use `Visualizer` to draw the predictions on the image.
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])

    from PIL import Image

    # Get the mask
    masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))
    # Pick an item to the mask

    try:
        item_mask = masks[0]
    except:
        continue

    # Get the true bouding box of the mask
    segmentation = np.where(item_mask - - True)
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))

    # Create croped image from the just the portion of the image
    cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :], mode='RGB')

    # Create a PIL image out of the mask
    mask = Image.fromarray((item_mask * 255).astype('uint8'))

    # Crop the mask to match the cropped image
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    # Load in a backgorund image and choose a paste position
    background = Image.fromarray((im * 0).astype('uint8'))

    # Create a new foreground image as large as the composite and paste the cropped image on top
    new_fg_image = Image.new('RGB', background.size)
    new_fg_image.paste(cropped)

    # Create a new alpha mask as large as the composite and paste the cropped mask
    new_alpha_mask = Image.new('RGB', background.size)
    new_alpha_mask.paste(cropped_mask)

    # Compose the foreground and background using the alpha mask
    composite = Image.composite(new_fg_image, background, mask)

    src = np.array(composite)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    cv2.imwrite(path + str(i[0:-4]) + '.png', dst)

with open('/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/synopsis/starting_times.json', 'r') as ssfile:
    starting_times = json.load(ssfile)
    starting_times = {int(k): v for k, v in starting_times.items()}

# Generating an OS safe path.
from os import path
import os

cwd = os.getcwd()
from pcgvs.extraction import load_tubes_with_pandas
from pcgvs.extraction import load_tubes_from_pandas_dataframe

p = path.join(cwd, 'synopsis')
p = path.join(p, 'tubes')
p = path.join(p, 'exp')
p = path.join(p, 'tracks')
p = path.join(p, input_txt)

dataframe = load_tubes_with_pandas(p)
tubes = load_tubes_from_pandas_dataframe(dataframe)

from pcgvs.aggregation import add_ss_to_dataframe

df = add_ss_to_dataframe(dataframe, tubes, starting_times)
df.to_csv("/Users/nguyenduy/Desktop/pcgvs-main/notebooks/Metadata/data-benchmark-simulation/eval.csv", sep='\t',
          encoding='utf-8')

from pcgvs.aggregation import add_ss_to_dataframe
from pcgvs.synopsis import generate_frames, generate_synopsis
from numpy import interp

frames = generate_frames(df, path)
generate_synopsis(frames, output_vid, 30, input_background, interp)
print('Video synopsis sucessful')
