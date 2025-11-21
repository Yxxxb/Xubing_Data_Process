from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path
import os
import random
from datakit.utils.files import find_all_files, read_jsonl_file, dump_list_to_jsonl_file
from datakit.utils.image import decode_base64_image_to_pil, draw_bounding_box, encode_pil_to_base64_image, draw_bounding_boxes, draw_bounding_boxes_with_labels
from datakit.utils.distributed import get_distributed_env, dist_split_files
from PIL import Image

machine_id = int(os.environ.get('TASK_INDEX', 0))
total_machines = int(os.environ.get('TASK_CNT', 1))
assert total_machines == 100

prompts = [
    "Please identify all the objects of the [class_name] category in this image and return their bounding box coordinates. Format each box as `<box>x_min y_min x_max y_max</box>` using normalized values based on image width and height, with the origin at the top-left corner of the image, rounded to 3 decimal places.",
    "For each instance of [class_name] in the image, provide its bounding box coordinates in the form `<box>x1 y1 x2 y2</box>`, where the values are normalized to the range [0, 1] using the image width and height as the basis, with the origin at the top-left corner. Round all numbers to 3 decimals.",
    "Give me the normalized bounding boxes of all [class_name] objects in this image. Output each in this format: `<box>x_top_left y_top_left x_bottom_right y_bottom_right</box>`, using values between 0 and 1 with three decimal precision. The origin is at the top-left of the image.",
    "Detect every [class_name] object in the image and output their bounding boxes in normalized format as `<box>x1 y1 x2 y2</box>`, with values between 0 and 1 (rounded to 3 decimals), and the origin at the top-left corner of the image.",
    "List all [class_name] objects in this picture and give their bounding boxes. Each should be expressed in the format `<box>x_min y_min x_max y_max</box>` with normalized coordinates between 0 and 1, where the origin is at the top-left corner, and 3 decimal digits.",
    "Please return the bounding box coordinates of every [class_name] object in this image. Use normalized format like `<box>0.123 0.456 0.789 0.987</box>`, with the origin at the top-left corner of the image, and include all instances.",
    "Identify and return the bounding boxes of each [class_name] in the image. Use the tag `<box> </box>` to wrap normalized coordinates (top-left and bottom-right corners), rounded to three decimals, assuming the origin is at the top-left corner of the image.",
    "For each object of category [class_name] detected in this image, output its bounding box as `<box>x1 y1 x2 y2</box>`, normalized to image dimensions with the origin at the top-left corner, and rounded to three decimal places.",
    "Detect all [class_name] objects in the image and provide each bounding box in this format: `<box>x1 y1 x2 y2</box>` with values between 0 and 1, normalized using the image’s width and height, with the origin at the top-left corner, and rounded to 3 decimal digits."
]

coco = COCO('/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/OpenDataLab___Objects365/raw/Objects365/data/train/zhiyuan_objv2_train.json')
names = [x["name"] for x in coco.loadCats(coco.getCatIds())]

sum = 10669307
save_list = []
dealed_num = 0
missed_num = 0

images = []
cats = []
for cid, cat in enumerate(names):
    catIds = coco.getCatIds(catNms=[cat])
    imgIds = coco.getImgIds(catIds=catIds)
    ims = coco.loadImgs(imgIds)
    images.extend(ims)
    cats.extend([cat] * len(ims))
assert len(cats) == len(images)

length = len(images)
chunk_size = length // total_machines
chunked_images = []
chunked_cats = []
for i in range(total_machines):
    start = i * chunk_size
    end = start + chunk_size if i < total_machines - 1 else length
    chunked_images.append(images[start:end])
    chunked_cats.append(cats[start:end])
assert 0 <= machine_id < total_machines, "machine_id 必须在 0 到 total_machines 之间"
c_chuncked_images = chunked_images[machine_id]
c_chunked_cats = chunked_cats[machine_id]

# # dist_split_files
# c_chuncked_images = dist_split_files(images)
# c_chunked_cats = dist_split_files(cats)

# 写入txt文件
# txt_path = f"/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/data/grammar_correct/start_process_{machine_id}.txt"
# with open(txt_path, 'w') as f:
#     f.write(f"start process {len(c_chuncked_images)} images...")

print("Start processing...")
for im, _cat in zip(c_chuncked_images, c_chunked_cats):
    catIds = coco.getCatIds(catNms=[_cat])
    file_name = os.path.join(*im["file_name"].split(os.sep)[-2:])
    _file_names = file_name.split('/')
    assert "patch" in _file_names[0], f"File name {file_name} does not contain 'patch' in the expected position."
    path = os.path.join("/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/OpenDataLab___Objects365/raw/Objects365/data/train", _file_names[0], _file_names[1])  # image filename
    assert path.endswith('.jpg'), f"Path {path} does not end with .jpg"
    if os.path.exists(path):
        annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
        if len(annIds) == 0:
            print(f"Error processing image {path}: len(annIds) == 0")
            missed_num += 1
            continue
        asn_str = ""
        for a in coco.loadAnns(annIds):
            x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
            _id = a['category_id']
            asn_str += f"<box>{max(0, x / im['width']):.3f} {max(0, y / im['height']):.3f} {min(1, (x + w) / im['width']):.3f} {min(1, (y + h) / im['height']):.3f}</box> "
        asn_str = asn_str.strip()
        _prompt = random.choice(prompts).replace('[class_name]', _cat)
        _item_id = f'Obj365-{catIds[0]}-{_cat}-{im["id"]}'
        try:
            pil_image = Image.open(path)
            b64_image = encode_pil_to_base64_image(pil_image)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            missed_num += 1
            continue
        _item = {
            'id': _item_id,
            'base64_image': {_item_id: b64_image},
            'conversations': [
                {'role': 'user', 'text': _prompt.replace('<img>', 'img').replace('<img/>', 'img')},
                {'role': 'assistant', 'text': asn_str.replace('<img>', 'img').replace('<img/>', 'img')}
            ],
        }
        save_list.append(_item)
        dealed_num += 1
        if dealed_num % 10000 == 0:
            print(f"Processed {dealed_num} images.")
            dump_list_to_jsonl_file(os.path.join("/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/data/grammar_correct", f'raw_{machine_id}_{dealed_num // 10000}.jsonl'), save_list)
            save_list = []
    else:
        missed_num += 1

if save_list:
    print(f"Processed {dealed_num} images.")
    dump_list_to_jsonl_file(os.path.join("/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/data/grammar_correct", f'raw_{machine_id}_{dealed_num // 10000}.jsonl'), save_list)

print("##### missed_num: ", missed_num)

# for cid, cat in enumerate(names):
#     catIds = coco.getCatIds(catNms=[cat])
#     imgIds = coco.getImgIds(catIds=catIds)
#     for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
#         file_name = os.path.join(*im["file_name"].split(os.sep)[-2:])
#         _file_names = file_name.split('/')
#         assert "patch" in _file_names[0], f"File name {file_name} does not contain 'patch' in the expected position."
#         path = os.path.join("/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/OpenDataLab___Objects365/raw/Objects365/data/train", _file_names[0], _file_names[1])  # image filename
#         assert path.endswith('.jpg'), f"Path {path} does not end with .jpg"
#         if os.path.exists(path):
#             annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=None)
#             if len(annIds) == 0:
#                 missed_num += 1
#                 continue
#             asn_str = ""
#             # bboxes = []
#             for a in coco.loadAnns(annIds):
#                 x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
#                 _id = a['category_id']
#                 # assert x <= im['width'] and x + w <= im['width'] and y <= im['height'] and y + h <= im['height'], f"x: {x}, y: {y}, w: {w}, h: {h}, im['width']: {im['width']}, im['height']: {im['height']}, path: {path}"
#                 asn_str += f"<box>{max(0, x / im['width']):.3f} {max(0, y / im['height']):.3f} {min(1, (x + w) / im['width']):.3f} {min(1, (y + h) / im['height']):.3f}</box> "
#                 # bboxes.append([max(0, x / im['width']), max(0, y / im['height']), max(0, (x + w) / im['width']), max(0, (y + h) / im['height'])])
#             asn_str = asn_str.strip()
#             _prompt = random.choice(prompts).replace('[class_name]', cat)
#             _item_id = f'Obj365-{catIds[0]}-{cat}-{im["id"]}'
#             pil_image = Image.open(path)
#             b64_image = encode_pil_to_base64_image(pil_image)
#             # boxed_image = draw_bounding_boxes(pil_image, bboxes)
#             # boxed_image.save('/mnt/cephfs/xubingye/tsp/imgs/boxed.jpg')
#             _item = {
#                 'id': _item_id,
#                 'base64_image': {_item_id: b64_image},
#                 'conversations': [
#                     {'role': 'user', 'text': _prompt.replace('<img>', 'img').replace('<img/>', 'img')},
#                     {'role': 'assistant', 'text': asn_str.replace('<img>', 'img').replace('<img/>', 'img')}
#                 ],
#             }
#             save_list.append(_item)
#             dealed_num += 1
#             if dealed_num % 20000 == 0:
#                 print(f"Processed {dealed_num} images.")
#                 dump_list_to_jsonl_file(os.path.join("/mnt/cephfs/xubingye/wfs/datasets/sft-category/Objects365/data/grammar_correct", f'raw_{machine_id}_{dealed_num // 200000}.jsonl'), save_list)
#                 save_list = []
#         else:
#             missed_num += 1

        
    
