import pickle

# with open('/mnt/cephfs/xubingye/vlm/VLMEvalKit/eval_res/202506/20250606e1xubing/POINTSV15-API/pkl.pkl', 'rb') as f:
#     data = pickle.load(f)

# breakpoint()
# print(data)


# read jsonl
import os
import json
from typing import List
import numpy as np
from PIL import Image, ImageDraw

from datakit.utils.files import read_jsonl_file
from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box

data = read_jsonl_file("/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/laion5b-en/data/0/000000.jsonl")
breakpoint()
image = decode_base64_image_to_pil(data[0]['base64_image'])
image.save('/mnt/cephfs/xubingye/rubbish/saved_image1.png')
print(image.size)
bbox = [50.401519775390625, 46.9847526550293, 442.79058837890625, 336.489013671875]
bbox[0] = bbox[0] / image.size[0]
bbox[1] = bbox[1] / image.size[1]
bbox[2] = bbox[2] / image.size[0]
bbox[3] = bbox[3] / image.size[1]
image2 = draw_bounding_box(image, bbox)
image2.save('/mnt/cephfs/xubingye/rubbish/saved_image2.png')
bbox = [363.95916748046875, 199.78228759765625, 464.9085693359375, 335.97808837890625]
bbox[0] = bbox[0] / image.size[0]
bbox[1] = bbox[1] / image.size[1]
bbox[2] = bbox[2] / image.size[0]
bbox[3] = bbox[3] / image.size[1]
image3 = draw_bounding_box(image, bbox)
image3.save('/mnt/cephfs/xubingye/rubbish/saved_image3.png')
