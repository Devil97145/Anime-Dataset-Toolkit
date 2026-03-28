# functions/asr_caption.py
import os
import json
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dashscope import MultiModalConversation
import dashscope

logger = logging.getLogger(__name__)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

SUPPORTED_ASR_MODELS = [
    "qwen3-asr-flash",
    "qwen-audio-asr"
]
DEFAULT_ASR_MODEL = "qwen3-asr-flash"


def _normalize_path(file_path: str) -> str:

    abs_path = os.path.abspath(file_path)
    if os.name == 'nt':  # Windows
        return abs_path.replace('\\', '/')
    return abs_path


def transcribe_single_audio(
    audio_path: str,
    model: str = SUPPORTED_ASR_MODELS,
    enable_lid: bool = True,
    enable_itn: bool = False
) -> str:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")


    normalized_path = _normalize_path(audio_path)
    logger.info(f"使用标准化路径: {normalized_path}")

    messages = [
        {
            "role": "user",
            "content": [{"audio": normalized_path}]  # 直接传路径
        }
    ]

    response = MultiModalConversation.call(
        model=model,
        messages=messages,
        asr_options={
            "enable_lid": enable_lid,
            "enable_itn": enable_itn
        },
        timeout=60
    )

    if response and response.output and response.output.choices:
        text = response.output.choices[0].message.content[0]["text"]
        if isinstance(text, dict):
            return text.get("text", "")
        return str(text)
    else:
        raise ValueError(f"ASR 模型返回错误: {response}")

def batch_transcribe_audios(
    audio_dir: str,
    output_dir: Optional[str] = None,
    model: str = DEFAULT_ASR_MODEL,
    skip_if_exists: bool = True,
    output_format: str = "txt",
    enable_lid: bool = True,
    enable_itn: bool = False
) -> List[Tuple[str, str, bool, str]]:
    if not os.path.isdir(audio_dir):
        return [("❌", f"目录不存在: {audio_dir}", False, "")]

    if not output_dir or output_dir.strip() == "":
        output_dir = audio_dir
    else:
        output_dir = output_dir.strip()
    os.makedirs(output_dir, exist_ok=True)

    # 支持常见音频格式
    extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
    audio_files = sorted([
        f for f in os.listdir(audio_dir)
        if Path(f).suffix.lower() in extensions
    ])

    results = []
    transcriptions = []

    for filename in audio_files:
        input_path = os.path.join(audio_dir, filename)
        txt_path = os.path.join(output_dir, Path(filename).stem + ".txt")

        need_call = True
        existing_text = ""
        if os.path.exists(txt_path):
            if skip_if_exists:
                need_call = False
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    existing_text = f.read().strip()
                results.append((filename, "已存在", True, existing_text[:50] + "..."))
            else:
                need_call = True

        if need_call:
            try:
                transcription = transcribe_single_audio(
                    input_path,
                    model=model,
                    enable_lid=enable_lid,
                    enable_itn=enable_itn
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
                results.append((filename, "成功", True, transcription[:50] + "..."))
                transcriptions.append((filename, transcription))
            except Exception as e:
                error_msg = str(e)[:100]
                results.append((filename, "失败", False, error_msg))
                logger.warning(f"ASR 失败: {filename} - {e}")
                transcriptions.append((filename, ""))
        else:
            transcriptions.append((filename, existing_text))

    # 生成 CSV/JSON（与图像模块完全一致）
    if output_format == "csv":
        csv_path = os.path.join(output_dir, "metadata.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['audio', 'transcription'])
            for audio_file, text in transcriptions:
                writer.writerow([audio_file, text])
        results.append(("✅", f"CSV 已保存至: {csv_path}", True, ""))

    elif output_format == "json":
        json_path = os.path.join(output_dir, "metadata.json")
        data = [{"audio": audio, "transcription": txt} for audio, txt in transcriptions]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        results.append(("✅", f"JSON 已保存至: {json_path}", True, ""))

    return results