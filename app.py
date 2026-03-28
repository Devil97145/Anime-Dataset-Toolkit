# app.py
import gradio as gr
from ui.video_ui import create_video_tab
from ui.tagging_ui import create_tagging_tab
from ui.filtering_ui import create_filtering_tab
from ui.editor_ui import create_editor_tab
from utils.ffmpeg_checker import check_ffmpeg
from ui.op_ed_ui import create_op_ed_tab
from ui.mikan_ui import create_mikan_tab
from ui.vl_caption_ui import create_vl_caption_tab
from ui.asr_caption_ui import create_asr_caption_tab
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app():
    # 设置环境变量，避免TensorFlow警告
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 检查FFmpeg
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        logger.warning("⚠️ 未检测到 FFmpeg！视频处理功能将无法使用。")
    
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
            create_vl_caption_tab()
            create_asr_caption_tab()
        gr.Markdown("---\n## 💡 使用指南\n by Devilworld\n")
    
    return demo

if __name__ == "__main__":
    logger.info("启动AI媒体处理工具箱...")
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=True
        # 移除 temp_dir 参数，因为 Blocks.launch() 不支持此参数
    )