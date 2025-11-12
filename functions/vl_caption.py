# functions/vl_caption.py
import os
import json
import csv
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dashscope import MultiModalConversation
from PIL import Image
import dashscope

logger = logging.getLogger(__name__)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

SUPPORTED_VL_MODELS = [
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen2.5-vl-72b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-3b-instruct",
    "qwen-vl-ocr",
    "qwen3-vl-32b-thinking",
    "qwen3-vl-32b-instruct"
]

# 默认模型（可统一维护）
DEFAULT_VL_MODEL = "qwen3-vl-plus"

# 创建全局临时目录（避免重复建）
_temp_dir = tempfile.mkdtemp(prefix="vl_caption_")
logger.info(f"创建VL临时目录: {_temp_dir}")


def compress_image_if_large(image_path: str, max_size_mb: int = 10) -> str:
    """
    如果图片 > max_size_mb，压缩为 JPEG 并保存到临时目录，返回新路径
    否则返回原路径
    """
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb <= max_size_mb:
        return image_path

    try:
        with Image.open(image_path) as img:
            # 转为 RGB（RGBA/PNG 会报错）
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            # 生成临时文件名
            temp_path = os.path.join(_temp_dir, Path(image_path).stem + "_compressed.jpg")
            # 压缩质量（可调）
            img.save(temp_path, "JPEG", quality=85, optimize=True)
            logger.info(f"压缩图片: {image_path} → {temp_path} ({size_mb:.1f}MB → {os.path.getsize(temp_path)/(1024*1024):.1f}MB)")
            return temp_path
    except Exception as e:
        logger.warning(f"压缩失败，使用原图: {e}")
        return image_path


def describe_single_image(
    image_path: str,
    model: str = "qwen3-vl-plus",
    prompt: str = "请用一段中文自然语言详细描述这张图片的内容。"
) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")

    # 关键：压缩大图（仅用于上传）
    upload_path = compress_image_if_large(image_path)
    url = f"file://{os.path.abspath(upload_path)}"
    messages = [{"role": "user", "content": [{"image": url}, {"text": prompt}]}]

    response = MultiModalConversation.call(model=model, messages=messages, timeout=30)
    if response and response.output and response.output.choices:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise ValueError("模型返回空结果")


def batch_describe_images(
    image_dir: str,
    output_dir: Optional[str] = None,
    model: str = "qwen3-vl-plus",
    prompt: str = "请用一段中文自然语言详细描述这张图片的内容。",
    skip_if_exists: bool = True,
    output_format: str = "txt",
) -> List[Tuple[str, str, bool, str]]:
    if not os.path.isdir(image_dir):
        return [("❌", f"目录不存在: {image_dir}", False, "")]

    if not output_dir or output_dir.strip() == "":
        output_dir = image_dir
    else:
        output_dir = output_dir.strip()
    os.makedirs(output_dir, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in extensions
    ])

    results = []
    descriptions = []

    for filename in image_files:
        input_path = os.path.join(image_dir, filename)
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
                description = describe_single_image(input_path, model=model, prompt=prompt)
                # 重点：txt 文件写回原图目录（或指定 output_dir）
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(description)
                results.append((filename, "成功", True, description[:50] + "..."))
                descriptions.append((filename, description))
            except Exception as e:
                error_msg = str(e)[:100]
                results.append((filename, "失败", False, error_msg))
                logger.warning(f"处理失败: {filename} - {e}")
                descriptions.append((filename, ""))
        else:
            descriptions.append((filename, existing_text))

    # 生成 CSV/JSON
    if output_format == "csv":
        csv_path = os.path.join(output_dir, "metadata.csv")
        # 写入 UTF-8-BOM，确保 Excel 正确识别中文
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image', 'prompt'])
            for img_file, text in descriptions:
                writer.writerow([img_file, text])
        results.append(("✅", f"CSV 已保存至: {csv_path}", True, ""))

    elif output_format == "json":
        json_path = os.path.join(output_dir, "metadata.json")
        data = [{"image": img, "prompt": txt} for img, txt in descriptions]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        results.append(("✅", f"JSON 已保存至: {json_path}", True, ""))

    return results