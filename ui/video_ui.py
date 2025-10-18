# ui/video_ui.py
import gradio as gr
import os
import tempfile
from functions.video_extractor import VideoFrameExtractor

def process_video_folder(input_dir, output_dir, frame_interval, max_threads):
    try:
        if not input_dir or not os.path.isdir(input_dir):
            return "❌ 请选择有效的输入文件夹", None
        if not output_dir:
            return "❌ 请指定输出目录", None
        os.makedirs(output_dir, exist_ok=True)
        extractor = VideoFrameExtractor(input_dir, output_dir, int(frame_interval), int(max_threads))
        extractor.start()
        success = extractor.processed_videos - extractor.failed_videos
        return (
            f"✅ 处理完成!\n成功: {success} 个视频\n失败: {extractor.failed_videos}\n总计: {extractor.total_videos}",
            output_dir
        )
    except Exception as e:
        return f"❌ 处理失败: {str(e)}", None

def process_single_video(video_file, frame_interval, output_dir=None):
    try:
        if not video_file:
            return "❌ 请选择视频文件", None, None
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="video_frames_")
            is_temp = True
        else:
            os.makedirs(output_dir, exist_ok=True)
            is_temp = False
        actual_frames = VideoFrameExtractor.extract_single_video(video_file, output_dir, int(frame_interval))
        frame_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")]
        frame_files.sort()
        temp_hint = "（临时目录，建议手动保存）" if is_temp else ""
        return (
            f"✅ 处理完成!\n提取帧数: {actual_frames}\n输出目录: {output_dir} {temp_hint}",
            frame_files,
            output_dir
        )
    except Exception as e:
        return f"❌ 处理失败: {str(e)}", None, None

def create_video_tab():
    with gr.TabItem("🎬 视频帧提取"):
        gr.Markdown("# 🎬 视频帧提取工具")
        with gr.Tabs():
            with gr.TabItem("📁 批量处理"):
                with gr.Row():
                    with gr.Column():
                        input_dir_batch = gr.Textbox(label="📂 输入文件夹")
                        output_dir_batch = gr.Textbox(label="📁 输出文件夹")
                        frame_interval_batch = gr.Number(label="⏱️ 帧间隔", value=90, precision=0)
                        max_threads_batch = gr.Number(label="🧵 最大线程数", value=4, precision=0)
                        process_btn_batch = gr.Button("🚀 开始批量处理", variant="primary")
                    with gr.Column():
                        output_text_batch = gr.Textbox(label="📊 处理结果", lines=5)
                        output_dir_link = gr.Textbox(label="📂 输出目录", interactive=False)
                process_btn_batch.click(
                    fn=process_video_folder,
                    inputs=[input_dir_batch, output_dir_batch, frame_interval_batch, max_threads_batch],
                    outputs=[output_text_batch, output_dir_link]
                )
            with gr.TabItem("📽️ 单视频处理"):
                with gr.Row():
                    with gr.Column():
                        video_file = gr.File(label="📂 选择视频文件", file_types=["video"])
                        output_dir_single = gr.Textbox(label="📁 自定义输出目录（可选）")
                        frame_interval_single = gr.Number(label="⏱️ 帧间隔", value=90, precision=0)
                        process_btn_single = gr.Button("▶️ 开始处理", variant="primary")
                    with gr.Column():
                        output_text_single = gr.Textbox(label="📊 处理结果", lines=3)
                        gallery = gr.Gallery(label="🖼️ 提取的帧预览", columns=4, height=400)
                process_btn_single.click(
                    fn=process_single_video,
                    inputs=[video_file, frame_interval_single, output_dir_single],
                    outputs=[output_text_single, gallery, gr.State()]
                )
        with gr.Accordion("📘 使用说明", open=False):
            gr.Markdown("（保留你的说明）")