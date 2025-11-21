from typing import List

import decord
import numpy as np
import torch
from torchvision.io import read_video

from datakit.utils.image import encode_np_to_base64_image


def extract_frames(video_path: str,
                   num_frames_per_second: int = 1,
                   max_frames: int = 10,
                   time_stamps: List[float] = None) -> List[str]:
    """Extract frames from a video.

    If the duration of the video is more than `max_frames` seconds,
    extract `max_frames` frames using equal intervals. Otherwise,
    extract `num_frames_per_second` frames per second.

    Args:
        video_path (str): Path to the video file.
        num_frames_per_second (int): Number of frames to
            extract per second.
        max_frames (int, optional): Maximum number of frames
            to extract. Defaults to 10.
        time_stamps (List[float], optional): List of time stamps
            to extract frames from. Defaults to None.

    Returns:
        List[str]: List of base64-encoded frames.
    """
    video, _, info = read_video(video_path,
                                start_pts=time_stamps[0],
                                end_pts=time_stamps[1],
                                pts_unit='sec',
                                output_format='THWC')
    duration = video.shape[0] / info['video_fps']
    total_frames = video.shape[0]
    if duration * num_frames_per_second > max_frames:
        num_frames = max_frames
    else:
        num_frames = int(duration * num_frames_per_second)
    idx = torch.linspace(0, total_frames - 1, num_frames).round().long()
    frames = video[idx]
    frames = [frame.numpy() for frame in frames]
    frames = [encode_np_to_base64_image(frame) for frame in frames]
    return frames


def extract_frames_decord(video_path: str,
                          num_frames_per_second: int = 1,
                          max_frames: int = 10,
                          time_stamps: List[float] = None) -> List[str]:
    """Extract frames from a video using decord.

    If the duration of the video is more than `max_frames` seconds,
    extract `max_frames` frames using equal intervals. Otherwise,
    extract `num_frames_per_second` frames per second.

    Args:
        video_path (str): Path to the video file.
        num_frames_per_second (int): Number of frames to
            extract per second.
        max_frames (int, optional): Maximum number of frames to
            extract. Defaults to 10.
        time_stamps (List[float], optional): List of time stamps to
            extract frames from. Defaults to None.

    Returns:
        List[str]: List of base64-encoded frames.
    """
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    start_frame = 0
    end_frame = len(vr)
    if time_stamps is not None:
        start_frame = int(time_stamps[0] * video_fps)
        end_frame = int(time_stamps[1] * video_fps)
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, len(vr))
    video = vr.get_batch(list(range(start_frame, end_frame))).asnumpy()
    duration = video.shape[0] / video_fps
    total_frames = video.shape[0]
    if duration * num_frames_per_second > max_frames:
        num_frames = max_frames
    else:
        num_frames = int(duration * num_frames_per_second)
    idx = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
    frames = video[idx]
    frames = [encode_np_to_base64_image(frame) for frame in frames]
    return frames
