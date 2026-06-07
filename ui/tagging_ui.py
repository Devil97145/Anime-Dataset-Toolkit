# ui/tagging_ui.py

import gradio as gr
import os
import threading
import traceback
from functions.wd_tagger_wrapper import (
    run_wd_tagger, 
    WD_TAGGER_MODELS, 
    check_model_exists, 
    download_model,
    get_model_status,
    DEFAULT_DOWNLOAD_DIR
)

def create_tagging_tab():
    with gr.TabItem("🏷️ AI自动打标"):
        gr.Markdown("## 🏷️ AI智能打标")
        gr.Markdown("使用本地 WD-ViT-TAGGER 模型为图片自动生成高质量标签")

        with gr.Row():
            with gr.Column(scale=2):
                # 模型选择
                tagger_model_key = gr.Dropdown(
                    choices=list(WD_TAGGER_MODELS.keys()),
                    value="wd-vit-tagger-v3",
                    label="🤖 选择打标模型",
                    info="不同模型有不同的精度和速度特性"
                )
                
                # 模型状态显示
                model_status_display = gr.HTML(
                    label="📊 模型状态",
                    value=get_model_status_html()
                )
                
                tagger_custom_model_path = gr.Textbox(
                    label="📂 自定义模型路径（仅选择 custom 时使用）",
                    placeholder="如：G:/my_models/wd-tagger-custom",
                    visible=False,
                    info="选择 custom 后填写此项"
                )
                
                # 下载目录设置
                with gr.Row():
                    download_dir_input = gr.Textbox(
                        value=DEFAULT_DOWNLOAD_DIR,
                        label="📥 模型下载目录",
                        info="模型将下载到此目录"
                    )
                
                # 下载按钮
                download_btn = gr.Button("⬇️ 下载/更新模型", variant="secondary")
                download_progress = gr.Textbox(
                    label="📥 下载进度",
                    lines=3,
                    interactive=False,
                    visible=True
                )
                
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
                
                # 模型说明
                gr.Markdown("""
                **模型说明：**
                - `wd-vit-tagger-v3`: 通用推荐，速度和精度平衡
                - `wd-swinv2-tagger-v3`: 更精确，但速度较慢
                - `wd-ovit-tagger-v3`: OViT 架构，中等性能
                - `wd-convnext-tagger-v3`: ConvNeXt，适合复杂场景
                - `custom`: 使用自定义路径的模型
                
                **使用方法：**
                1. 选择或下载所需模型
                2. 设置图片文件夹路径
                3. 点击"开始AI打标"
                """)

        def get_model_status_html():
            """生成模型状态的 HTML 显示"""
            status = get_model_status()
            html = '<div style="background:#f5f5f5;padding:10px;border-radius:8px;">'
            html += '<b>模型安装状态：</b><br>'
            for key, info in status.items():
                if key == "custom":
                    continue
                color = "#28a745" if info["installed"] else "#dc3545"
                status_text = "✅ 已安装" if info["installed"] else "❌ 未安装"
                html += f'<span style="color:{color};">● {info["name"]}: {status_text}</span><br>'
            html += '</div>'
            return html
        
        def refresh_model_status():
            """刷新模型状态显示"""
            return get_model_status_html()
        
        def update_custom_path_visibility(model_key):
            return gr.update(visible=(model_key == "custom"))

        def download_selected_model(model_key, custom_path, download_dir, progress_text):
            """下载选中的模型"""
            def download_thread():
                def progress_callback(progress, message):
                    progress_text[0] = f"[{progress}%] {message}"
                
                if model_key == "custom":
                    if not custom_path:
                        progress_text[0] = "❌ 请填写自定义模型路径"
                        return
                    success, msg, path = download_model(
                        model_key, None, custom_path, progress_callback
                    )
                else:
                    success, msg, path = download_model(
                        model_key, download_dir, None, progress_callback
                    )
                
                if success:
                    progress_text[0] = f"✅ {msg}\n路径: {path}"
                else:
                    progress_text[0] = f"❌ {msg}"
            
            progress_text[0] = f"开始下载 {model_key}..."
            thread = threading.Thread(target=download_thread, daemon=True)
            thread.start()
            thread.join(timeout=120)  # 等待下载完成
            
            if thread.is_alive():
                progress_text[0] += "\n⏳ 下载仍在进行中..."
            
            return progress_text[0]

        def run_tagger(
            image_dir,
            model_key,
            custom_model_path,
            general_threshold,
            character_threshold,
            caption_extension,
            remove_underscore,
            undesired_tags,
            always_first_tags,
            caption_separator,
            append_tags,
            batch_size,
            recursive,
            download_dir
        ):
            result = ["处理中... 请查看控制台输出"]
            def target():
                try:
                    # 创建进度回调
                    def progress_callback(p, m):
                        pass  # 打标过程不需要显示进度
                    
                    output = run_wd_tagger(
                        image_dir=image_dir,
                        model_key=model_key,
                        custom_model_path=custom_model_path if model_key == "custom" else None,
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
                        frequency_tags=True,
                        auto_download=True,
                        download_dir=download_dir,
                        progress_callback=progress_callback
                    )
                    result[0] = output
                except Exception as e:
                    result[0] = f"❌ 处理失败：{str(e)}"
                    traceback.print_exc()

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=1)  # 短暂等待，避免立即返回空
            return result[0] or "✅ 任务已启动，请查看控制台输出..."

        # 模型选择变化时显示/隐藏自定义路径输入框
        tagger_model_key.change(
            fn=update_custom_path_visibility,
            inputs=[tagger_model_key],
            outputs=[tagger_custom_model_path]
        )
        
        # 刷新模型状态按钮
        def on_refresh_click():
            return get_model_status_html()
        
        # 下载按钮点击事件
        download_btn.click(
            fn=download_selected_model,
            inputs=[tagger_model_key, tagger_custom_model_path, download_dir_input, download_progress],
            outputs=download_progress
        )
        
        # 下载完成后刷新模型状态
        download_btn.click(
            fn=on_refresh_click,
            inputs=[],
            outputs=[model_status_display]
        )
        
        tagger_btn.click(
            fn=run_tagger,
            inputs=[
                tagger_image_dir,
                tagger_model_key,
                tagger_custom_model_path,
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
                download_dir_input,
            ],
            outputs=tagger_output
        )
