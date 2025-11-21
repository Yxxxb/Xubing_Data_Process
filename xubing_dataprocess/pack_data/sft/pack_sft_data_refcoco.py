from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path
import os
import random
from datakit.utils.files import find_all_files, read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box, encode_pil_to_base64_image, draw_bounding_boxes, draw_bounding_boxes_with_labels
from datakit.utils.distributed import get_distributed_env, dist_split_files
from PIL import Image
from datakit.utils.zip import read_all_files_in_zip
from datakit.utils.tar import read_all_files_in_tar, tar_encode_base64_image

jsl = read_jsonl_file("/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/data/grammar_correct/raw_99_3.jsonl")
breakpoint()

sa1b_path = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/segmentation/SA-1B/official_data/sa_000000.tar"
refcoco_path = "/mnt/cephfs/bensenliu/wfs/vlmdatasets/pt/segmentation/refcoco/refcocog.zip"
sa1b = read_all_files_in_tar(sa1b_path, extension='any')
refc = read_all_files_in_zip(refcoco_path, extension='any')
breakpoint()