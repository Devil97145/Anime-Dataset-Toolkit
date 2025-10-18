# functions/op_ed_detector.py

import os
import cv2
import numpy as np
from pathlib import Path
from .video_extractor import VideoFrameExtractor

def compute_frame_hash(frame, hash_size=16):
    """计算帧的感知哈希（pHash）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def frames_to_hashes(video_path, frame_interval=15):
    """提取视频关键帧并计算哈希"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hashes = []
    frame_indices = []

    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        h = compute_frame_hash(frame)
        hashes.append(h)
        frame_indices.append(i)

    cap.release()
    return np.array(hashes), frame_indices, fps

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

def find_similar_segments(
    main_video,
    ref_video,
    frame_interval=15,
    max_hamming=10,
    min_consecutive=3
):
    """
    在 main_video 中查找与 ref_video 相似的片段
    返回 [(start_frame, end_frame), ...]
    """
    print(f"[OP/ED] 正在分析参考视频: {os.path.basename(ref_video)}")
    ref_hashes, _, _ = frames_to_hashes(ref_video, frame_interval)
    
    print(f"[OP/ED] 正在扫描主视频: {os.path.basename(main_video)}")
    main_hashes, main_indices, fps = frames_to_hashes(main_video, frame_interval)
    
    matches = []
    i = 0
    while i < len(main_hashes):
        # 找到第一个匹配帧
        dists = [hamming_distance(main_hashes[i], rh) for rh in ref_hashes]
        if min(dists) <= max_hamming:
            start_i = i
            count = 0
            # 统计连续匹配帧数
            while i < len(main_hashes) and count < len(ref_hashes):
                dists = [hamming_distance(main_hashes[i], rh) for rh in ref_hashes]
                if min(dists) <= max_hamming:
                    count += 1
                    i += 1
                else:
                    break
            if count >= min_consecutive:
                start_frame = main_indices[start_i]
                end_frame = main_indices[i - 1] if i - 1 < len(main_indices) else main_indices[-1]
                matches.append((start_frame, end_frame))
                print(f"  ✅ 找到匹配片段: {start_frame} ~ {end_frame} 帧")
            else:
                i += 1
        else:
            i += 1

    return matches, fps