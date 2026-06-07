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

def compute_pairwise_distances(main_hashes, ref_hashes):
    """使用 NumPy 向量化计算所有主哈希与参考哈希之间的距离矩阵"""
    main_hashes = np.array(main_hashes)
    ref_hashes = np.array(ref_hashes)
    distances = np.sum(main_hashes[:, np.newaxis, :] != ref_hashes[np.newaxis, :, :], axis=2)
    return distances

def find_best_matches(distances, max_hamming=10):
    """找到每个主哈希对应的最佳参考哈希索引和距离"""
    min_distances = np.min(distances, axis=1)
    best_ref_indices = np.argmin(distances, axis=1)
    is_match = min_distances <= max_hamming
    return is_match, best_ref_indices, min_distances

def merge_overlapping_segments(segments, merge_threshold=5):
    """合并重叠或接近的片段"""
    if not segments:
        return []
    
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        last = merged[-1]
        if current[0] - last[1] <= merge_threshold:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged

def find_similar_segments(
    main_video,
    ref_video,
    frame_interval=15,
    max_hamming=10,
    min_consecutive=3,
    search_region=None
):
    """
    在 main_video 中查找与 ref_video 相似的片段
    返回 [(start_frame, end_frame), ...]
    search_region: 可选的搜索区域限制 (start_frame, end_frame)
    """
    print(f"[OP/ED] 正在分析参考视频: {os.path.basename(ref_video)}")
    ref_hashes, _, _ = frames_to_hashes(ref_video, frame_interval)
    
    if len(ref_hashes) == 0:
        print(f"  ⚠️ 参考视频没有有效帧")
        return [], 0
    
    print(f"[OP/ED] 正在扫描主视频: {os.path.basename(main_video)}")
    main_hashes, main_indices, fps = frames_to_hashes(main_video, frame_interval)
    
    if len(main_hashes) == 0:
        print(f"  ⚠️ 主视频没有有效帧")
        return [], 0
    
    if search_region is not None:
        start_frame, end_frame = search_region
        filtered = [(h, idx) for h, idx in zip(main_hashes, main_indices) 
                   if start_frame <= idx <= end_frame]
        if not filtered:
            print(f"  ⚠️ 指定搜索区域内没有有效帧")
            return [], 0
        main_hashes, main_indices = zip(*filtered)
    
    distances = compute_pairwise_distances(main_hashes, ref_hashes)
    is_match, best_ref_indices, min_distances = find_best_matches(distances, max_hamming)
    
    matches = []
    i = 0
    n = len(main_hashes)
    
    while i < n:
        if is_match[i]:
            start_i = i
            count = 0
            ref_pos = best_ref_indices[i]
            
            while i < n and count < len(ref_hashes):
                current_ref_pos = (ref_pos + count) % len(ref_hashes)
                if is_match[i] and abs(best_ref_indices[i] - current_ref_pos) <= 2:
                    count += 1
                    i += 1
                else:
                    break
            
            if count >= min_consecutive:
                start_frame = main_indices[start_i]
                end_frame = main_indices[i - 1]
                matches.append((start_frame, end_frame))
                print(f"  ✅ 找到匹配片段: {start_frame} ~ {end_frame} 帧")
            else:
                i += 1
        else:
            i += 1
    
    matches = merge_overlapping_segments(matches)
    
    return matches, fps