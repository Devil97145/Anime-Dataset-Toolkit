# ui/vl_caption_ui.py
import gradio as gr
from functions.vl_caption import batch_describe_images, describe_single_image, SUPPORTED_VL_MODELS, DEFAULT_VL_MODEL
import os

def create_vl_caption_tab():
    with gr.Tab("🖼️ 图像智能描述（Qwen-VL）"):
        gr.Markdown("### 使用通义千问 VL 模型自动为图片生成中文描述")

        # ========== 单图测试 ==========
        with gr.Accordion("📌 单图测试", open=False):
            with gr.Row():
                single_image = gr.Image(type="filepath", label="上传单张图片")
                single_prompt = gr.Textbox(
                    label="提示词",
                    value="请用一段中文自然语言详细描述这张图片的内容。",
                    placeholder="例如：描述图中的动漫角色外貌和场景"
                )
            with gr.Row():
                single_model = gr.Dropdown(
                    choices=SUPPORTED_VL_MODELS,
                    value=DEFAULT_VL_MODEL,
                    label="模型"
                )
                single_run = gr.Button("🔍 单图描述", variant="secondary")
            single_output = gr.Textbox(label="模型输出", interactive=False, lines=5)

            single_run.click(
                fn=describe_single_image,
                inputs=[single_image, single_model, single_prompt],
                outputs=single_output
            )

        gr.Markdown("---")

        # ========== 批量处理 ==========
        with gr.Row():
            input_dir = gr.Textbox(label="输入图片文件夹", placeholder="例如：E:/数据集/动漫图片")
            output_dir = gr.Textbox(label="输出目录（留空则同目录）", placeholder="可选")

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=SUPPORTED_VL_MODELS,
                value=DEFAULT_VL_MODEL,
                label="模型选择"
            )
            skip_if_exists = gr.Checkbox(label="✔️ 跳过已存在同名 .txt 的图片", value=True)
            overwrite = gr.Checkbox(label="🔄 覆盖已有 .txt 文件", value=False)


        with gr.Row():
            output_format = gr.Radio(
                choices=["txt", "csv", "json"],
                value="txt",
                label="输出格式",
                info="txt：每张图生成同名 .txt；csv/json：额外生成 metadata 文件"
            )

        prompt_input = gr.Textbox(
            label="📝 自定义提示词（Prompt）",
            value="请用一段中文自然语言详细描述这张图片的内容。",
            placeholder="可自定义描述要求"
        )

        run_btn = gr.Button("🚀 开始批量描述", variant="primary")

        result_output = gr.Dataframe(
            headers=["文件名", "状态", "成功", "信息"],
            datatype=["str", "str", "bool", "str"],
            interactive=False,
            max_height=400
        )

        # 覆盖逻辑：如果勾选 overwrite，则 skip_if_exists = False
        def _run_batch(image_dir, out_dir, model, prompt, skip, overwrite_flag, fmt):
            return batch_describe_images(
                image_dir=image_dir,
                output_dir=out_dir,
                model=model,
                prompt=prompt,
                skip_if_exists=skip and not overwrite_flag,  # 关键：overwrite 优先级更高
                output_format=fmt
            )

        run_btn.click(
            fn=_run_batch,
            inputs=[input_dir, output_dir, model_choice, prompt_input, skip_if_exists, overwrite, output_format],
            outputs=result_output
        )

        gr.Markdown("""
        > 💡 **说明**：
        > - **txt**：每张图生成同名 `.txt`（始终执行）
        > - **csv**：额外生成 `metadata.csv`，含 `image, prompt` 两列
        > - **json**：额外生成 `metadata.json`，格式为 `[{"image": "xxx.jpg", "prompt": "xxx"}]`
        > - 勾选 **覆盖** 会强制重新生成所有 `.txt`，即使已存在
        """)