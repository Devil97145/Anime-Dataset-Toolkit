# ui/nlp_tagging_ui.py
"""
自然语言打标界面
支持 BLIP, Qwen-VL, LLaVA 等视觉语言模型
"""

import gradio as gr
import threading
import traceback
from functions.nlp_tagger import (
    run_nlp_tagger,
    tag_single_image,
    get_available_models,
    check_model_requirements,
    install_model_requirements,
    VisionModelInferrer
)

def create_nlp_tagging_tab():
    with gr.TabItem("🎨 NLP自然语言打标"):
        gr.Markdown("## 🎨 自然语言打标 (NLP)")
        gr.Markdown("使用视觉语言模型为图片自动生成自然语言描述并转换为标签")
        
        # 模型信息显示
        model_info_display = gr.HTML(
            label="📊 模型信息",
            value=get_model_status_html()
        )
        
        with gr.Tabs():
            # ========== 单张图片测试 ==========
            with gr.TabItem("🧪 单张测试"):
                with gr.Row():
                    with gr.Column(scale=1):
                        test_image = gr.Image(
                            label="上传图片", 
                            type="filepath", 
                            height=400,
                            format="png"
                        )
                        
                        test_model = gr.Dropdown(
                            choices=get_model_choices(),
                            value="blip",
                            label="🤖 选择模型",
                            info="不同模型有不同精度和速度"
                        )
                        
                        test_prompt = gr.Textbox(
                            label="📝 自定义提示词（可选）",
                            placeholder="留空使用默认提示词",
                            lines=2
                        )
                        
                        test_btn = gr.Button("🔍 测试打标", variant="primary", size="lg")
                        
                        # 模型安装
                        with gr.Accordion("📦 模型安装", open=False):
                            install_btn = gr.Button("⬇️ 安装依赖", variant="secondary")
                            install_status = gr.Textbox(
                                label="安装状态", 
                                lines=2, 
                                interactive=False
                            )
                            gr.Markdown("""
                            **安装说明：**
                            - BLIP：轻量，无需 GPU，约 1GB
                            - Qwen2-VL (2B)：需要 GPU，约 4GB
                            - LLaVA 1.6：需要 GPU，约 7GB
                            """)
                    
                    with gr.Column(scale=1):
                        test_caption = gr.Textbox(
                            label="📝 生成的描述", 
                            lines=4,
                            interactive=False
                        )
                        
                        test_tags = gr.Textbox(
                            label="🏷️ 提取的标签", 
                            lines=8,
                            interactive=False
                        )
                
                # 事件处理
                def on_test_click(img_path, model_key, prompt):
                    if not img_path:
                        return "请先上传图片", "", get_model_status_html()
                    
                    try:
                        caption, tags = tag_single_image(img_path, model_key, prompt if prompt.strip() else None)
                        return caption, ", ".join(tags) if tags else "", get_model_status_html()
                    except Exception as e:
                        return f"错误: {str(e)}", traceback.format_exc(), get_model_status_html()
                
                test_btn.click(
                    fn=on_test_click,
                    inputs=[test_image, test_model, test_prompt],
                    outputs=[test_caption, test_tags, model_info_display]
                )
                
                def on_install_click(model_key):
                    try:
                        install_model_requirements(model_key)
                        return f"✅ 依赖安装完成！\n\n请选择模型后开始使用。", get_model_status_html()
                    except Exception as e:
                        return f"❌ 安装失败: {str(e)}", get_model_status_html()
                
                install_btn.click(
                    fn=on_install_click,
                    inputs=[test_model],
                    outputs=[install_status, model_info_display]
                )
            
            # ========== 批量打标 ==========
            with gr.TabItem("📦 批量打标"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_image_dir = gr.Textbox(
                            label="📂 图片文件夹路径",
                            placeholder="输入图片文件夹路径"
                        )
                        
                        batch_model = gr.Dropdown(
                            choices=get_model_choices(),
                            value="blip",
                            label="🤖 选择模型"
                        )
                        
                        batch_prompt = gr.Textbox(
                            label="📝 自定义提示词（可选）",
                            placeholder="留空使用默认提示词"
                        )
                        
                        with gr.Row():
                            extraction_method = gr.Radio(
                                choices=["keyword", "nlp"],
                                value="keyword",
                                label="🔍 标签提取方法"
                            )
                        
                        with gr.Row():
                            overwrite = gr.Checkbox(value=False, label="覆盖已有文件")
                            caption_ext = gr.Textbox(value=".txt", label="扩展名", max_lines=1)
                        
                        output_dir = gr.Textbox(
                            label="📁 输出文件夹（可选）",
                            placeholder="留空与图片同目录"
                        )
                        
                        max_threads = gr.Slider(1, 4, value=1, step=1, label="线程数")
                        
                        batch_btn = gr.Button("🚀 开始批量打标", variant="primary", size="lg")
                        
                        batch_output = gr.Textbox(
                            label="📊 处理结果",
                            lines=6,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ## 📖 使用说明
                        
                        ### 支持的模型
                        
                        | 模型 | 精度 | 速度 | 显存需求 | 推荐场景 |
                        |------|------|------|----------|----------|
                        | **BLIP** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | <2GB | 快速处理、日常使用 |
                        | **Qwen2-VL (2B)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~6GB | 详细描述、复杂场景 |
                        | **LLaVA 1.6** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~8GB | 最高精度、专业场景 |
                        
                        ### 使用技巧
                        1. **快速打标**：选择 BLIP，适合大量图片
                        2. **详细描述**：选择 Qwen2-VL 或 LLaVA
                        3. **自定义提示词**：可指定特定风格的描述
                        
                        ### 输出格式
                        ```
                        1girl, solo, long hair, blue eyes, school uniform, smiling
                        ```
                        """)
                
                def on_batch_click(image_dir, model, prompt, method, ext, output, overwrite_flag, threads):
                    if not image_dir:
                        return "❌ 请输入图片文件夹路径", get_model_status_html()
                    
                    result_container = ["⏳ 处理中，请稍候..."]
                    
                    def worker():
                        try:
                            result = run_nlp_tagger(
                                image_dir=image_dir,
                                model_key=model,
                                extraction_method=method,
                                caption_extension=ext,
                                output_dir=output if output.strip() else None,
                                overwrite=overwrite_flag,
                                max_threads=threads,
                                custom_prompt=prompt if prompt.strip() else None
                            )
                            result_container[0] = result
                        except Exception as e:
                            result_container[0] = f"❌ 处理异常: {str(e)}\n{traceback.format_exc()}"
                    
                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    thread.join(timeout=10)
                    
                    if thread.is_alive():
                        result_container[0] += "\n\n⏳ 模型加载中，首次使用可能需要下载..."
                    
                    return result_container[0], get_model_status_html()
                
                batch_btn.click(
                    fn=on_batch_click,
                    inputs=[
                        batch_image_dir,
                        batch_model,
                        batch_prompt,
                        extraction_method,
                        caption_ext,
                        output_dir,
                        overwrite,
                        max_threads
                    ],
                    outputs=[batch_output, model_info_display]
                )
            
            # ========== 标签模板 ==========
            with gr.TabItem("✏️ 标签模板"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📝 标签模板生成")
                        
                        template_selector = gr.Dropdown(
                            choices=[
                                "通用动漫风格",
                                "风景照片",
                                "人物特写",
                                "物体拍摄",
                                "食物摄影"
                            ],
                            value="通用动漫风格",
                            label="选择模板"
                        )
                        
                        template_btn = gr.Button("📋 生成模板")
                        template_tags = gr.Textbox(
                            label="模板标签",
                            lines=5
                        )
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 💡 模板说明
                        
                        - **通用动漫**: 人物、外观、场景基础标签
                        - **风景**: 自然景观、环境标签
                        - **人物特写**: 面部、表情细节标签
                        - **物体拍摄**: 产品、静物摄影标签
                        - **食物摄影**: 美食相关标签
                        
                        *注：模板用于预填充或批量处理参考*
                        """)
                
                def generate_template(choice):
                    templates = {
                        "通用动漫风格": "1girl, solo, looking at viewer, smiling, blue eyes, long hair, school uniform, outdoors, high quality",
                        "风景照片": "outdoors, sky, cloud, tree, grass, sunset, mountain, water, nature, scenic, high quality",
                        "人物特写": "1girl, close-up, portrait, face, detailed eyes, smile, looking at viewer, high quality, detailed",
                        "物体拍摄": "object, still life, simple background, product photo, colorful, studio lighting, high quality",
                        "食物摄影": "food, delicious, appetizing, restaurant style, top view, overhead shot, colorful, high quality"
                    }
                    return templates.get(choice, "")
                
                template_btn.click(
                    fn=generate_template,
                    inputs=[template_selector],
                    outputs=[template_tags]
                )

def get_model_choices():
    """获取可用模型列表"""
    try:
        models = VisionModelInferrer.MODEL_CONFIGS
        choices = []
        for key, config in models.items():
            choices.append((f"{config['name']} ({config['model_size']})", key))
        return choices if choices else [("BLIP (推荐)", "blip")]
    except:
        return [("BLIP (推荐)", "blip")]

def get_model_status_html():
    """生成模型状态 HTML"""
    try:
        requirements = check_model_requirements()
        device = "GPU" if requirements.get("transformers") else "CPU"
        
        html = '<div style="background:#f8f9fa;padding:12px;border-radius:8px;">'
        html += f'<b>当前设备: {device}</b><br><br>'
        
        models = VisionModelInferrer.MODEL_CONFIGS
        for key, config in models.items():
            status = "✅ 可用" if requirements.get("transformers") else "❌ 需安装"
            gpu_note = " (需GPU)" if config.get("requires_gpu") else " (CPU可用)"
            html += f'<span style="color:{"#28a745" if requirements.get("transformers") else "#dc3545"};">'
            html += f'● {config["name"]} [{config["model_size"]}]{gpu_note}: {status}</span><br>'
        
        html += '</div>'
        return html
    except Exception as e:
        return f'<div style="color:red;">模型状态获取失败: {str(e)}</div>'

if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_nlp_tagging_tab()
    demo.launch()
