# ui/op_ed_ui.py
import gradio as gr
from pathlib import Path
from functions.op_ed_detector import detect_op_ed

def create_op_ed_tab():
    with gr.TabItem("🎬 片头片尾识别"):
        gr.Markdown("### 自动识别视频的片头（OP）与片尾（ED）时间段")
        
        with gr.Row():
            input_video = gr.Video(label="上传视频", interactive=True)
            output_result = gr.JSON(label="检测结果", visible=True)

        with gr.Row():
            max_op_duration = gr.Slider(5, 120, value=90, step=5, label="最大片头时长（秒）")
            max_ed_duration = gr.Slider(5, 120, value=90, step=5, label="最大片尾时长（秒）")
            threshold = gr.Slider(0.1, 0.9, value=0.75, step=0.05, label="相似度阈值（越低越敏感）")

        run_btn = gr.Button("🔍 开始检测")

        run_btn.click(
            fn=detect_op_ed,
            inputs=[input_video, max_op_duration, max_ed_duration, threshold],
            outputs=output_result
        )

        gr.Examples(
            examples=[
                ["text/video/1.mkv"]
            ],
            inputs=[input_video],
            label="示例视频（需存在）"
        )