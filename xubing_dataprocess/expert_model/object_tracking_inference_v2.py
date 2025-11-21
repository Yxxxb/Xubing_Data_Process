from ultralytics import SAM, YOLO, FastSAM
from deep_sort_realtime.deepsort_tracker import DeepSort
from datakit.utils.distributed import (barrier_all_processes, dist_split_files,
                                       get_distributed_env, gpu_utilization,
                                       kill_process)
from datakit.utils.files import find_all_files, dump_list_to_jsonl_file, mem_efficient_read_jsonl_file, read_jsonl_file
from tqdm import tqdm
from typing import List
import os
import base64
from datakit.utils.mp import multi_process_with_append
import subprocess
from functools import partial
from PIL import Image
from io import BytesIO
from ultralytics.models.sam import SAM2VideoPredictor, SAM2Predictor
import numpy as np
import cv2
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="number of workers for multi-processing",
)

parser.add_argument(
    "--partition", type=str, default=None, help="sub partitions to be processed, e.g. 0-10"
)

parser.add_argument(
    "--sub-task-name", type=str, default="detect_scene_second_clip_main_object_key_caption_qwen25vl", help="task name for saving directory"
)

parser.add_argument(
    "--local-rank", type=int, default=0, help="data parallel local rank"
)

root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/Koala-36M'

MIN_FILE_INDEX = 0
MAX_FILE_INDEX = 30
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="/mnt/cephfs/haichengwang/code/data_processing/sam2.1_b.pt", save_dir="/mnt/cephfs/haichengwang/code/data_processing/runs")
SAM_predictor = SAM2VideoPredictor(overrides=overrides)
model_SAM = FastSAM("/mnt/cephfs/haichengwang/code/data_processing/FastSAM-x.pt")
model = YOLO("/mnt/cephfs/haichengwang/code/data_processing/yolov8x-worldv2.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.model.to(device).eval()
model_SAM.model.to(device).eval()
# SAM_predictor.to(device)

print("finish loading the model")

def base64_to_image(base64_string: str):
    """Base64 to Image"""
    if type(base64_string) != str:
        return base64_string
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def images_to_video(image_list : List[str], 
                    output_root : str, 
                    vid_name: str  = "output_video", 
                    fps: int = 1,
                    ):
    """
       Convert a list of images to a video file.
       Args:
           image_list (List[str]): A list of image base64 format
           output_root (str): The output directory.
           vid_name (str): The name of the output video file.
           fps (int): fps of the output video. Defaults to 2.

        Returns:
            None
    """

    if ".mp4" in vid_name or ".avi" in vid_name or ".mov" in vid_name:
        vid_name = vid_name.split(".")[0]
    output_file = os.path.join(output_root, f'{vid_name}.avi')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not image_list:
        raise ValueError("Image list is empty")

    image_list = [base64_to_image(img) for img in image_list]
    height, width, _ = image_list[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img in image_list:
        video_writer.write(img)

    video_writer.release()
    #os.system(f"""ffmpeg -y -i {output_file} -c:v libx264 -c:a aac -b:a 192k -g 1 -movflags frag_keyframe+empty_moov {output_file.replace('.avi', '.mp4')}""")
    cmd = f"""ffmpeg -y -i {output_file} -c:v libx264 -c:a aac -b:a 192k -g 1 -movflags frag_keyframe+empty_moov {output_file.replace('.avi', '.mp4')}"""
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(output_file)
    return output_file.replace('.avi', '.mp4')

def process_single_jsonl(jsonl_path):
    whole_data = mem_efficient_read_jsonl_file(jsonl_path)
    main_object_data = read_jsonl_file(os.path.join(main_object_folder,os.path.basename(jsonl_path)).replace(".jsonl", f"_filerange_{MIN_FILE_INDEX}_{MAX_FILE_INDEX}.jsonl"))
    output_path = os.path.join(output_root, os.path.basename(jsonl_path)).replace(".jsonl", f"_filerange_{MIN_FILE_INDEX}_{MAX_FILE_INDEX}.jsonl")
    if os.path.exists(output_path):
        return
    
    # model.set_classes(model.names)
    results_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(whole_data)):
            if i not in range(MIN_FILE_INDEX, MAX_FILE_INDEX):
                continue
            main_object_scene = main_object_data[i]
            assert main_object_scene["id"] == data["id"]
            video_detection_list = {"video_name":data["id"], "fast_SAM":[]}
            bboxes = []
            frame_list = list(data["base64_image"].values())
            for j,anno in enumerate(main_object_scene["caption"].values()):
                object_list = anno[0].split("&")
                object_list = [obj.strip()[:76] for obj in object_list]
                timestamps = anno[1]
                # video_path = images_to_video(frame_list[timestamps[0]: timestamps[1]], 
                #                 "/tmp", 
                #                 vid_name=f"{i}_{j}_" + data["id"], 
                #                 fps=data["fps"],
                #                 )
                results = []
                for frame in frame_list[timestamps[0]: timestamps[1]]:
                    results.append(model_SAM(base64_to_image(frame), texts = object_list, device=device, max_det=20, conf=0.3, iou=0.5, agnostic_nms=True))
                for i, result in enumerate(results):
                    bboxes = result[0].boxes.xyxy.cpu().numpy().tolist()
                    confidence = result[0].boxes.conf.tolist()
                    orig_shape = result[0].boxes.orig_shape
                    video_detection_list["fast_SAM"].append({"frame": timestamps[0]+i, "bbox":bboxes, "orig_shape": orig_shape, "confidence": confidence, "object_list": object_list})
                # if os.path.exists(video_path):
                #     os.remove(video_path)
            results_list.append(video_detection_list)
            print("fast SAM finish.")
                # visualize_frame_list = []
                # color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                # for frame, result in zip(frame_list[timestamps[0]: timestamps[1]], results):
                #     image = base64_to_image(frame)
                #     bboxes_list = result[0].boxes.xyxy.cpu().numpy()
                #     for i, bbox in enumerate(bboxes_list):
                #         x1, y1, x2, y2 = map(int, bbox)
                #         cv2.rectangle(image, (x1, y1), (x2, y2), color_list[i % len(color_list)], 2)
                #     visualize_frame_list.append(image)
                # if os.path.exists(video_path):
                #     os.remove(video_path)
                # video_path = images_to_video(visualize_frame_list, 
                #                 "/mnt/cephfs/haichengwang/envs/visualize/detection", 
                #                 vid_name=f"{i}_{j}_" + data["id"], 
                #                 fps=data["fps"],
                #                 )
                # with open(video_path.replace(".mp4", ".txt"),"w") as f:
                #     f.write(str(object_list))
    if len(results_list) > 0:
        dump_list_to_jsonl_file(output_path, results_list)
    return 1

if __name__ == "__main__":
    args = parser.parse_args()
    root = '/mnt/cephfs/bensenliu/wfs/vlmdatasets/sft-wepoints-category-video/CAPTION/Koala-36M'
    sub_folders = os.path.join(root,"original_data/fps1_maxframe300000-correct")
    main_object_folder = os.path.join(root,"caption/fps1_maxframe300000-correct/caption_qwen2_5vl_detect_scene_second_clip_main_object")
    output_root = os.path.join(root,f"object_tracking/fps1_maxframe300000-correct/{args.sub_task_name}")
    os.makedirs(output_root, exist_ok=True)
    all_jsonls = find_all_files(sub_folders, 'jsonl')
    unprocessed_jsons = []
    if args.partition is not None:
        start, end = args.partition.split("-")
        all_jsonls = all_jsonls[int(start):int(end)]
    for jsonl_path in all_jsonls:
        if not os.path.exists(os.path.join(output_root, os.path.basename(jsonl_path)).replace(".jsonl", f"_filerange_{MIN_FILE_INDEX}_{MAX_FILE_INDEX}.jsonl")):
            unprocessed_jsons.append(jsonl_path)
    all_jsonls = unprocessed_jsons
    world_size, rank, _ = get_distributed_env()
    alert_root = "/mnt/cephfs/haichengwang/envs"
    if rank == 0:
        if os.path.exists(os.path.join(alert_root, args.sub_task_name)):
            os.system(f"rm -rf {os.path.join(alert_root, args.sub_task_name)}")
    resume_point = 0
    num_workers = args.num_workers
    sub_jsonls_cur_rank = dist_split_files(all_jsonls)
    sub_jsonls_cur_rank = sub_jsonls_cur_rank[(len(sub_jsonls_cur_rank)//8+1)*args.local_rank:(len(sub_jsonls_cur_rank)//8+1)*(args.local_rank+1)]
    for i, jsonl_file_path in tqdm(enumerate(sub_jsonls_cur_rank)):  # for local debug
        json_dict = process_single_jsonl(jsonl_file_path)
    # success_jsonls = multi_process_with_append(process_single_jsonl, 
    #                                         sub_jsonls_cur_rank, 
    #                                         num_workers=num_workers)

    print('Done')
    barrier_all_processes(task_name=args.sub_task_name, root=alert_root)  # wait for all processes to finish
