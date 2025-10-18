# ui/tagging_ui.py
import gr
import sys
import traceback
import threading
from functions.wd_tagger_wrapper import run_wd_tagger

def create_tagging_tab():
    with gr.TabItem("🏷️ AI自动打标"):
        gr.Markdown("## 🏷️ AI智能打标")
        with gr.Row():
            with gr.Column(scale=2):
                tagger_image_dir = gr.Textbox(label="📂 图片文件夹路径")
                with gr.Row():
                    tagger_general_threshold = gr.Slider(0, 1, 0.35, label="📊 General标签阈值")
                    tagger_character_threshold = gr.Slider(0, 1, 0.35, label="🎭 Character标签阈值")
                with gr.Row():
                    tagger_remove_underscore = gr.Checkbox(True, label="🔠 自动替换下划线为空格")
                    tagger_recursive = gr.Checkbox(True, label="🔁 递归子文件夹")
                    tagger_append_tags = gr.Checkbox(False, label="➕ 追加模式")
                tagger_undesired_tags = gr.Textbox(label="❌ 排除标签（逗号分隔）")
                tagger_always_first_tags = gr.Textbox(label="🔝 始终置顶标签（逗号分隔）")
                with gr.Accordion("输出设置", open=False):
                    tagger_caption_separator = gr.Textbox(", ", label="🧩 标签分隔符")
                    tagger_caption_extension = gr.Textbox(".txt", label="📝 标签文件扩展名")
                tagger_batch_size = gr.Slider(1, 8, 1, step=1, label="📦 批处理大小")
                tagger_btn = gr.Button("🚀 开始AI打标", variant="primary", size="lg")
            with gr.Column(scale=1):
                tagger_output = gr.Textbox(label="📊 处理结果", lines=12)

        def run_tagger(*args):
            thread = threading.Thread(target=lambda: None, daemon=True)
            def run():
                try:
                    result = run_wd_tagger(*args, debug=False, frequency_tags=True)
                    tagger_output.value = result
                except Exception as e:
                    tagger_output.value = f"❌ 处理失败：{str(e)}"
                    traceback.print_exc()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return "✅ 任务已启动，请查看控制台输出..."

        tagger_btn.click(
            fn=run_tagger,
            inputs=[
                tagger_image_dir,
                "G:/工具箱/wd-vit-tagger-v3",  # 模型路径（建议后续改为配置）
                tagger_general_threshold,
                tagger_character_threshold,
                tagger_caption_extension,
                tagger_remove_underscore,
                tagger_undesired_tags,
                False,  # use_rating_tags
                False,  # use_rating_tags_as_last_tag
                False,  # character_tags_first
                tagger_always_first_tags,
                tagger_caption_separator,
                tagger_append_tags,
                tagger_batch_size,
                tagger_recursive,
            ],
            outputs=tagger_output
        )