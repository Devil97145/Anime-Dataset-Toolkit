# ui/asr_caption_ui.py
import gradio as gr
from functions.asr_caption import batch_transcribe_audios, transcribe_single_audio, SUPPORTED_ASR_MODELS, DEFAULT_ASR_MODEL
import os

def create_asr_caption_tab():
    with gr.Tab("🎤 语音智能转写（Qwen-ASR）"):
        gr.Markdown("### 使用通义千问 ASR 模型将音频转为中文文本")

        # ========== 单音频测试 ==========
        with gr.Accordion("📌 单音频测试", open=True):
            with gr.Row():
                single_audio = gr.Audio(
                    type="filepath", 
                    label="上传音频文件",
                    format="wav"
                )
            
            with gr.Row():
                # ✅ 关键修复1：使用与批量部分相同的变量名，避免冲突
                single_model_dropdown = gr.Dropdown(  # 改名避免与批量部分冲突
                    choices=SUPPORTED_ASR_MODELS,
                    value=DEFAULT_ASR_MODEL,
                    label="模型",
                    interactive=True,  # 确保可交互
                    elem_id="single_audio_model_dropdown"  # 添加唯一ID
                )
                single_run = gr.Button("🔍 转写音频", variant="primary")
            
            # 状态显示
            single_status = gr.Textbox(label="状态", interactive=False, value="就绪")
            single_output = gr.Textbox(label="转写结果", interactive=False, lines=5)

            # ✅ 关键修复2：正确绑定事件，传入模型参数
            def _single_transcribe(audio_path, model_name):
                """单音频转写函数（带模型参数）"""
                if not audio_path:
                    return "", "❌ 未上传音频文件"
                
                single_status.value = f"⏳ 使用 {model_name} 转写中..."
                try:
                    # ✅ 传入选择的模型
                    result = transcribe_single_audio(audio_path, model=model_name)
                    return result, f"✅ 转写成功 | 模型: {model_name}"
                except Exception as e:
                    error_msg = f"❌ 错误: {str(e)}"
                    logger.error(f"单音频转写失败: {e}")
                    return "", error_msg
                finally:
                    single_status.value = "就绪"

            # ✅ 关键修复3：正确绑定输入和输出
            single_run.click(
                fn=_single_transcribe,
                inputs=[single_audio, single_model_dropdown],  # 传入两个参数
                outputs=[single_output, single_status]
            )

        gr.Markdown("---")


        # ========== 批量处理 ==========
        with gr.Row():
            input_dir = gr.Textbox(label="输入音频文件夹", placeholder="例如：F:/动漫数据集工作箱/音频")
            output_dir = gr.Textbox(label="输出目录（留空则同目录）", placeholder="可选")

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=SUPPORTED_ASR_MODELS,
                value=DEFAULT_ASR_MODEL,
                label="模型选择"
            )
            skip_if_exists = gr.Checkbox(label="✔️ 跳过已存在同名 .txt 的音频", value=True)
            overwrite = gr.Checkbox(label="🔄 覆盖已有 .txt 文件", value=False)

        with gr.Row():
            output_format = gr.Radio(
                choices=["txt", "csv", "json"],
                value="txt",
                label="输出格式",
                info="txt：每段音频生成同名 .txt；csv/json：额外生成 metadata 文件"
            )

        run_btn = gr.Button("🚀 开始批量转写", variant="primary")

        result_output = gr.Dataframe(
            headers=["文件名", "状态", "成功", "信息"],
            datatype=["str", "str", "bool", "str"],
            interactive=False,
            max_height=400
        )

        def _run_batch(audio_dir, out_dir, model, skip, overwrite_flag, fmt):
            return batch_transcribe_audios(
                audio_dir=audio_dir,
                output_dir=out_dir,
                model=model,
                skip_if_exists=skip and not overwrite_flag,
                output_format=fmt
            )

        run_btn.click(
            fn=_run_batch,
            inputs=[input_dir, output_dir, model_choice, skip_if_exists, overwrite, output_format],
            outputs=result_output
        )

        gr.Markdown("""
        > 💡 **说明**：
        > - **txt**：每段音频生成同名 `.txt`
        > - **csv**：生成 `metadata.csv`，含 `audio, transcription` 两列（UTF-8-BOM 编码）
        > - **json**：生成 `metadata.json`，格式为 `[{"audio": "xxx.mp3", "transcription": "xxx"}]`
        > - 勾选 **覆盖** 会强制重新生成所有 `.txt`
        """)