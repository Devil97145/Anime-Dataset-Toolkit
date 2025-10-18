# ui/op_ed_ui.py

import gradio as gr
import os
import threading
from functions.op_ed_detector import find_similar_segments

def detect_op_ed(main_video, op_ref=None, ed_ref=None, frame_interval=15):
    if not main_video:
        return "❌ 请上传主视频"
    
    result = []
    
    def process():
        try:
            if op_ref:
                matches, fps = find_similar_segments(main_video, op_ref, frame_interval)
                for start, end in matches:
                    start_sec = start / fps
                    end_sec = end / fps
                    result.append(f"🎬 OP: {start_sec:.1f}s - {end_sec:.1f}s")
            
            if ed_ref:
                matches, fps = find_similar_segments(main_video, ed_ref, frame_interval)
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
    thread.join(timeout=30)
    
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
                detect_btn = gr.Button("🔍 开始识别", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="📊 识别结果", lines=8)
        
        detect_btn.click(
            fn=detect_op_ed,
            inputs=[main_video, op_ref, ed_ref, frame_interval],
            outputs=output
        )