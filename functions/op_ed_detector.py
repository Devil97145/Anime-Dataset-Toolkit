# core/op_ed_detector.py
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _extract_frame_at(video_path: str, time_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def _compute_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    if frame1 is None or frame2 is None:
        return 0.0
    # Resize to small size for speed
    h, w = frame1.shape[:2]
    small1 = cv2.resize(frame1, (w // 8, h // 8))
    small2 = cv2.resize(frame2, (w // 8, h // 8))
    diff = cv2.absdiff(small1, small2)
    return 1.0 - np.mean(diff) / 255.0

def _detect_op(video_path: str, max_duration: int, threshold: float) -> Tuple[float, float]:
    """返回 (start, end) in seconds"""
    logger.info("Detecting OP...")
    start_frame = _extract_frame_at(video_path, 1.0)  # 第1秒作为参考
    if start_frame is None:
        return (0.0, 0.0)

    for t in range(5, max_duration + 1, 2):  # 从5秒开始检测，避免静音黑屏
        frame = _extract_frame_at(video_path, float(t))
        if frame is None:
            break
        sim = _compute_frame_similarity(start_frame, frame)
        if sim < threshold:
            return (0.0, float(t))
    return (0.0, float(max_duration))

def _detect_ed(video_path: str, max_duration: int, threshold: float) -> Tuple[float, float]:
    """返回 (start, end) in seconds"""
    logger.info("Detecting ED...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0.0, 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_sec = total_frames / fps
    cap.release()

    end_frame = _extract_frame_at(video_path, total_sec - 1.0)
    if end_frame is None:
        return (total_sec, total_sec)

    for offset in range(5, min(max_duration, int(total_sec)) + 1, 2):
        t = total_sec - offset
        if t < 0:
            break
        frame = _extract_frame_at(video_path, t)
        if frame is None:
            continue
        sim = _compute_frame_similarity(end_frame, frame)
        if sim < threshold:
            return (t, total_sec)
    return (max(0.0, total_sec - max_duration), total_sec)

def detect_op_ed(
    video_path: str,
    max_op_duration: int = 90,
    max_ed_duration: int = 90,
    threshold: float = 0.75
) -> Dict[str, Any]:
    if not video_path or not Path(video_path).exists():
        return {"error": "视频文件不存在"}

    try:
        op_start, op_end = _detect_op(video_path, max_op_duration, threshold)
        ed_start, ed_end = _detect_ed(video_path, max_ed_duration, threshold)

        result = {
            "OP": {
                "start_seconds": round(op_start, 2),
                "end_seconds": round(op_end, 2),
                "duration_seconds": round(op_end - op_start, 2)
            },
            "ED": {
                "start_seconds": round(ed_start, 2),
                "end_seconds": round(ed_end, 2),
                "duration_seconds": round(ed_end - ed_start, 2)
            }
        }
        logger.info(f"Detection result: {result}")
        return result
    except Exception as e:
        logger.exception("Detection failed")
        return {"error": str(e)}