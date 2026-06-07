# ui/op_ed_ui.py

import gradio as gr
import os
import threading
import cv2
from functions.op_ed_detector import find_similar_segments

def get_video_duration(video_path):
    """获取视频总帧数和帧率"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps

def detect_op_ed(main_video, op_ref=None, ed_ref=None, frame_interval=15, 
                 op_search_ratio=0.25, ed_search_ratio=0.25):
    if not main_video:
        return "❌ 请上传主视频"
    
    result = []
    
    def process():
        try:
            total_frames, _ = get_video_duration(main_video)
            
            if op_ref:
                search_end = int(total_frames * op_search_ratio)
                matches, fps = find_similar_segments(
                    main_video, op_ref, frame_interval,
                    search_region=(0, search_end)
                )
                for start, end in matches:
                    start_sec = start / fps
                    end_sec = end / fps
                    result.append(f"🎬 OP: {start_sec:.1f}s - {end_sec:.1f}s")
            
            if ed_ref:
                search_start = int(total_frames * (1 - ed_search_ratio))
                matches, fps = find_similar_segments(
                    main_video, ed_ref, frame_interval,
                    search_region=(search_start, total_frames)
                )
                for start, end in matches:
                    start_sec = start / fps
                    end_sec = end / fps
                    result.append(f"🎬 ED: {start_sec:.1f}s - {end_sec:.1f}s")
            
            if not result:
                result.append("⚠️ 未检测到匹配的 OP/ED 片段")
        except Exception as e:
            result.append(f"❌ 检测失败: {str(e)}")
    
    thread = threading.Thread(target=process, daemon=True)
    thread.start()
    thread.join(timeout=60)
    
    return "\n".join(result) if result else "✅ 检测完成（无结果）"

def create_op_ed_tab():
    with gr.TabItem("🎬 OP/ED 识别"):
        gr.Markdown("## 🎬 自动识别片头（OP）/片尾（ED）")
        gr.Markdown("上传主视频 + OP/ED 参考视频，系统将自动定位相似片段")
        
        with gr.Row():
            with gr.Column():
                main_video = gr.File(label="📽️ 主视频（正片）", file_types=["video"])
                op_ref = gr.File(label="🎵 OP 参考视频（可选）", file_types=["video"])
                ed_ref = gr.File(label="🎵 ED 参考视频（可选）", file_types=["video"])
                frame_interval = gr.Slider(5, 60, value=15, step=5, label="⏱️ 帧采样间隔（越小越准，越慢）")
                op_search_ratio = gr.Slider(0.1, 0.5, value=0.25, step=0.05, 
                                           label="🎬 OP 搜索范围（视频开头比例）")
                ed_search_ratio = gr.Slider(0.1, 0.5, value=0.25, step=0.05, 
                                           label="🎬 ED 搜索范围（视频结尾比例）")
                detect_btn = gr.Button("🔍 开始识别", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="📊 识别结果", lines=8)
        
        detect_btn.click(
            fn=detect_op_ed,
            inputs=[main_video, op_ref, ed_ref, frame_interval, op_search_ratio, ed_search_ratio],
            outputs=output
        )