# app.py
import gradio as gr
from ui.video_ui import create_video_tab
from ui.tagging_ui import create_tagging_tab
from ui.filtering_ui import create_filtering_tab
from ui.editor_ui import create_editor_tab
from utils.ffmpeg_checker import check_ffmpeg
from ui.op_ed_ui import create_op_ed_tab
from ui.mikan_ui import create_mikan_tab

def create_app():
    ffmpeg_ok = check_ffmpeg()
    
    with gr.Blocks(title="AI 媒体处理工具箱", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 AI 媒体处理工具箱")
        if not ffmpeg_ok:
            gr.Warning("⚠️ 未检测到 FFmpeg！视频处理功能将无法使用。")
        
        with gr.Tabs():
            create_video_tab()
            create_tagging_tab()
            create_filtering_tab()
            create_editor_tab()
            create_mikan_tab()
            create_op_ed_tab()

        gr.Markdown("---\n## 💡 使用指南\n by Devilworld\n")
    
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        allowed_paths=["F:\动漫数据集工作箱"]
       
    )