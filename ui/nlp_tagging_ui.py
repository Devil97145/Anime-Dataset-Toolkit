# ui/nlp_tagging_ui.py
"""
自然语言打标界面
"""

import gradio as gr
import threading
import traceback
from functions.nlp_tagger import (
    run_nlp_tagger,
    tag_single_image,
    get_available_models,
    install_requirements
)

def create_nlp_tagging_tab():
    with gr.TabItem("🎨 NLP自然语言打标"):
        gr.Markdown("## 🎨 自然语言打标 (NLP)")
        gr.Markdown("使用视觉语言模型为图片自动生成自然语言描述并转换为标签")
        
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
                            choices=["blip"],  # 默认只有 BLIP 可用
                            value="blip",
                            label="选择模型"
                        )
                        
                        test_btn = gr.Button("🔍 测试打标", variant="primary", size="lg")
                        
                        # 模型信息
                        with gr.Accordion("模型安装", open=False):
                            install_btn = gr.Button("📦 安装依赖", variant="secondary")
                            install_status = gr.Textbox(
                                label="安装状态", 
                                lines=2, 
                                interactive=False
                            )
                    
                    with gr.Column(scale=1):
                        # 结果展示
                        test_caption = gr.Textbox(
                            label="📝 生成的描述", 
                            lines=3,
                            interactive=False
                        )
                        
                        test_tags = gr.Textbox(
                            label="🏷️ 提取的标签", 
                            lines=6,
                            interactive=False
                        )
                
                # 事件处理
                def on_test_click(img_path, model_key):
                    if not img_path:
                        return "请先上传图片", ""
                    
                    try:
                        caption, tags = tag_single_image(img_path, model_key)
                        return caption, ", ".join(tags) if tags else ""
                    except Exception as e:
                        return f"错误: {str(e)}", traceback.format_exc()
                
                test_btn.click(
                    fn=on_test_click,
                    inputs=[test_image, test_model],
                    outputs=[test_caption, test_tags]
                )
                
                def on_install_click():
                    try:
                        install_requirements()
                        return "✅ 依赖安装完成！请刷新页面检查可用模型。"
                    except Exception as e:
                        return f"❌ 安装失败: {str(e)}"
                
                install_btn.click(
                    fn=on_install_click,
                    inputs=[],
                    outputs=[install_status]
                )
            
            # ========== 批量打标 ==========
            with gr.TabItem("📦 批量打标"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入设置
                        batch_image_dir = gr.Textbox(
                            label="📂 图片文件夹路径",
                            placeholder="输入图片文件夹路径"
                        )
                        
                        batch_model = gr.Dropdown(
                            choices=["blip"],
                            value="blip",
                            label="🤖 选择模型"
                        )
                        
                        with gr.Row():
                            extraction_method = gr.Radio(
                                choices=["keyword", "nlp"],
                                value="keyword",
                                label="🔍 标签提取方法",
                                info="keyword: 关键词匹配 | nlp: 自然语言处理"
                            )
                        
                        with gr.Row():
                            overwrite = gr.Checkbox(
                                value=False,
                                label="覆盖已存在的标签文件"
                            )
                        
                        with gr.Row():
                            caption_ext = gr.Textbox(
                                value=".txt",
                                label="标签文件扩展名",
                                max_lines=1
                            )
                        
                        with gr.Row():
                            output_dir = gr.Textbox(
                                label="输出文件夹（可选）",
                                placeholder="留空则与图片同目录"
                            )
                        
                        with gr.Row():
                            max_threads = gr.Slider(
                                1, 4,
                                value=1,
                                step=1,
                                label="线程数"
                            )
                        
                        batch_btn = gr.Button(
                            "🚀 开始批量打标",
                            variant="primary",
                            size="lg"
                        )
                        
                        # 进度显示
                        batch_output = gr.Textbox(
                            label="📊 处理结果",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        # 说明文档
                        gr.Markdown("""
                        ## 📖 使用说明
                        
                        ### 功能特点
                        - **自动描述生成**: 使用视觉语言模型理解图片内容
                        - **标签提取**: 将自然语言描述转换为可编辑的标签
                        - **批量处理**: 支持大规模图片快速标注
                        
                        ### 支持的模型
                        - **BLIP**: 轻量通用模型，适合大多数场景
                        - **Qwen-VL**: 通义千问（开发中）
                        - **LLaVA**: 开源 LLaVA（开发中）
                        
                        ### 输出格式
                        - 默认生成 `.txt` 文件，与图片在同一目录
                        - 标签格式: `tag1, tag2, tag3, ...`
                        
                        ### 注意事项
                        - 首次使用可能需要下载模型（约 1-2 GB）
                        - 建议使用 GPU 加速，CPU 处理会比较慢
                        - 可在下方查看模型状态和安装依赖
                        """)
                
                # 批量处理逻辑
                def on_batch_click(image_dir, model, method, ext, output, overwrite_flag, threads):
                    if not image_dir:
                        return "❌ 请输入图片文件夹路径"
                    
                    result_container = ["⏳ 处理中..."]
                    
                    def worker():
                        try:
                            result = run_nlp_tagger(
                                image_dir=image_dir,
                                model_key=model,
                                extraction_method=method,
                                caption_extension=ext,
                                output_dir=output if output.strip() else None,
                                overwrite=overwrite_flag,
                                max_threads=threads
                            )
                            result_container[0] = result
                        except Exception as e:
                            result_container[0] = f"❌ 处理异常: {str(e)}\n{traceback.format_exc()}"
                    
                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    
                    return "⏳ 任务已启动，请查看控制台或等待完成..."
                
                def check_result():
                    pass  # 需要更复杂的进度更新机制
                
                batch_btn.click(
                    fn=on_batch_click,
                    inputs=[
                        batch_image_dir,
                        batch_model,
                        extraction_method,
                        caption_ext,
                        output_dir,
                        overwrite,
                        max_threads
                    ],
                    outputs=[batch_output]
                )
            
            # ========== 标签编辑器 ==========
            with gr.TabItem("✏️ 标签编辑"):
                with gr.Row():
                    with gr.Column():
                        # 快速生成标签的辅助工具
                        gr.Markdown("### 📝 标签模板生成")
                        
                        template_selector = gr.Dropdown(
                            choices=[
                                "通用动漫风格",
                                "风景照片",
                                "人物特写",
                                "简单物体"
                            ],
                            value="通用动漫风格",
                            label="选择模板"
                        )
                        
                        template_btn = gr.Button("📋 生成标签模板")
                        template_tags = gr.Textbox(
                            label="模板标签",
                            lines=5,
                            placeholder="选择模板后点击生成..."
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 💡 使用技巧")
                        gr.Markdown("""
                        - **通用动漫**: 包含基础的人物、外观和场景标签
                        - **风景**: 包含自然景观标签
                        - **人物**: 人物特写相关标签
                        - **物体**: 单一物体描述标签
                        
                        *注：模板标签用于批量打标前的预填充或参考。*
                        """)
                
                def generate_template(choice):
                    templates = {
                        "通用动漫风格": "1girl, solo, looking at viewer, smiling, blue eyes, long hair, school uniform, outdoors",
                        "风景照片": "outdoors, sky, cloud, tree, grass, sunset, mountain, water",
                        "人物特写": "1person, close-up, portrait, face, looking at viewer, smile, detailed eyes",
                        "简单物体": "object, still life, simple background, colorful, product photo"
                    }
                    return templates.get(choice, "")
                
                template_btn.click(
                    fn=generate_template,
                    inputs=[template_selector],
                    outputs=[template_tags]
                )

if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_nlp_tagging_tab()
    demo.launch()
