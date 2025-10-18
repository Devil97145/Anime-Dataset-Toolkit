# ui/tagging_ui.py

import gradio as gr
import os
import threading
import traceback
from functions.wd_tagger_wrapper import run_wd_tagger

def create_tagging_tab():
    with gr.TabItem("🏷️ AI自动打标"):
        gr.Markdown("## 🏷️ AI智能打标")
        gr.Markdown("使用本地 WD-ViT-TAGGER-v3 模型为图片自动生成高质量标签")

        with gr.Row():
            with gr.Column(scale=2):
                tagger_image_dir = gr.Textbox(
                    label="📂 图片文件夹路径",
                    placeholder="如：G:/images",
                    info="支持子文件夹递归搜索"
                )
                
                with gr.Row():
                    tagger_general_threshold = gr.Slider(
                        0.0, 1.0, value=0.35, step=0.01,
                        label="📊 General标签阈值"
                    )
                    tagger_character_threshold = gr.Slider(
                        0.0, 1.0, value=0.35, step=0.01,
                        label="🎭 Character标签阈值"
                    )
                
                with gr.Row():
                    tagger_remove_underscore = gr.Checkbox(
                        value=True,
                        label="🔠 自动替换下划线为空格"
                    )
                    tagger_recursive = gr.Checkbox(
                        value=True,
                        label="🔁 递归子文件夹"
                    )
                    tagger_append_tags = gr.Checkbox(
                        value=False,
                        label="➕ 追加模式"
                    )
                
                tagger_undesired_tags = gr.Textbox(
                    label="❌ 排除标签（逗号分隔）",
                    placeholder="如：watermark, signature, blurry"
                )
                
                tagger_always_first_tags = gr.Textbox(
                    label="🔝 始终置顶标签（逗号分隔）",
                    placeholder="如：1girl, solo, masterpiece"
                )
                
                with gr.Accordion("输出设置", open=False):
                    tagger_caption_separator = gr.Textbox(
                        value=", ",
                        label="🧩 标签分隔符"
                    )
                    tagger_caption_extension = gr.Textbox(
                        value=".txt",
                        label="📝 标签文件扩展名"
                    )
                
                tagger_batch_size = gr.Slider(
                    1, 8, value=1, step=1,
                    label="📦 批处理大小"
                )
                
                tagger_btn = gr.Button("🚀 开始AI打标", variant="primary", size="lg")

            with gr.Column(scale=1):
                tagger_output = gr.Textbox(
                    label="📊 处理结果",
                    lines=12,
                    interactive=False,
                    placeholder="处理结果将显示在这里..."
                )

        def run_tagger(
            image_dir,
            general_threshold,
            character_threshold,
            caption_extension,
            remove_underscore,
            undesired_tags,
            always_first_tags,
            caption_separator,
            append_tags,
            batch_size,
            recursive
        ):
            result = ["处理中... 请查看控制台输出"]
            def target():
                try:
                    output = run_wd_tagger(
                        image_dir=image_dir,
                        general_threshold=general_threshold,
                        character_threshold=character_threshold,
                        caption_extension=caption_extension,
                        remove_underscore=remove_underscore,
                        undesired_tags=undesired_tags,
                        use_rating_tags=False,
                        use_rating_tags_as_last_tag=False,
                        character_tags_first=False,
                        always_first_tags=always_first_tags,
                        caption_separator=caption_separator,
                        append_tags=append_tags,
                        batch_size=batch_size,
                        recursive=recursive,
                        debug=False,
                        frequency_tags=True
                    )
                    result[0] = output
                except Exception as e:
                    result[0] = f"❌ 处理失败：{str(e)}"
                    traceback.print_exc()

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=1)  # 短暂等待，避免立即返回空
            return result[0] or "✅ 任务已启动，请查看控制台输出..."

        tagger_btn.click(
            fn=run_tagger,
            inputs=[
                tagger_image_dir,
                tagger_general_threshold,
                tagger_character_threshold,
                tagger_caption_extension,
                tagger_remove_underscore,
                tagger_undesired_tags,
                tagger_always_first_tags,
                tagger_caption_separator,
                tagger_append_tags,
                tagger_batch_size,
                tagger_recursive,
            ],
            outputs=tagger_output
        )