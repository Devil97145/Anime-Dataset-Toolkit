# functions/wd_tagger_wrapper.py

import csv
import os
import requests
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import re
import pandas as pd

# ======================
# 模型配置
# ======================

# 可用打标模型列表
WD_TAGGER_MODELS = {
    "wd-vit-tagger-v3": {
        "path": "G:/工具箱/wd-vit-tagger-v3",
        "description": "通用图像打标（推荐）",
        "version": "v3",
        "hf_model_id": "SmilingWolf/wd-vit-tagger-v3",
        "files": ["model.onnx", "selected_tags.csv"]
    },
    "wd-swinv2-tagger-v3": {
        "path": "G:/工具箱/wd-swinv2-tagger-v3",
        "description": "SwinV2 模型，更精确但更慢",
        "version": "v3",
        "hf_model_id": "SmilingWolf/wd-swinv2-tagger-v3",
        "files": ["model.onnx", "selected_tags.csv"]
    },
    "wd-ovit-tagger-v3": {
        "path": "G:/工具箱/wd-ovit-tagger-v3",
        "description": "OViT 模型，平衡速度和精度",
        "version": "v3",
        "hf_model_id": "SmilingWolf/wd-ovit-tagger-v3",
        "files": ["model.onnx", "selected_tags.csv"]
    },
    "wd-vit-tagger-v2": {
        "path": "G:/工具箱/wd-vit-tagger-v2",
        "description": "旧版 ViT 模型",
        "version": "v2",
        "hf_model_id": "SmilingWolf/wd-vit-tagger-v2",
        "files": ["model.onnx", "selected_tags.csv"]
    },
    "wd-convnext-tagger-v3": {
        "path": "G:/工具箱/wd-convnext-tagger-v3",
        "description": "ConvNeXt 模型，适合复杂场景",
        "version": "v3",
        "hf_model_id": "SmilingWolf/wd-convnext-tagger-v3",
        "files": ["model.onnx", "selected_tags.csv"]
    },
    "custom": {
        "path": "",
        "description": "自定义模型路径",
        "version": "custom",
        "hf_model_id": None,
        "files": []
    }
}

# 默认模型
DEFAULT_MODEL_KEY = "wd-vit-tagger-v3"

# 模型缓存
_model_cache = {}

# 默认下载目录
DEFAULT_DOWNLOAD_DIR = "G:/工具箱"

# 临时添加当前路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import library.train_util as train_util
    from library.utils import setup_logging
    setup_logging()
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    class DummyLogger:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}", flush=True)
        @staticmethod
        def error(msg): print(f"[ERROR] {msg}", flush=True)
        @staticmethod
        def warning(msg): print(f"[WARNING] {msg}", flush=True)
    logger = DummyLogger()

    def glob_images_pathlib(dir_path, recursive):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
        dir_path = Path(dir_path)
        if recursive:
            return [p for p in dir_path.rglob("*") if p.suffix.lower() in image_extensions]
        else:
            return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]

    train_util = type('train_util', (), {'glob_images_pathlib': glob_images_pathlib})


# ======================
# 图像预处理工具函数
# ======================

def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if isinstance(img, str) and img.endswith(".gif"):
        pil_img = Image.open(img)
        pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(img, flag)


def smart_24bit(img):
    if img.dtype == np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def smart_resize(img, size):
    if img.shape[0] > size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


# ======================
# 标签后处理
# ======================

tag_escape_pattern = re.compile(r'([\\()])')

def postprocess_tags(
    tags: dict,
    threshold=0.35,
    additional_tags=None,
    exclude_tags=None,
    sort_by_alphabetical_order=False,
    add_confident_as_weight=False,
    replace_underscore=False,
    replace_underscore_excludes=None,
    escape_tag=False
) -> dict:
    if additional_tags is None:
        additional_tags = []
    if exclude_tags is None:
        exclude_tags = []
    if replace_underscore_excludes is None:
        replace_underscore_excludes = []

    for t in additional_tags:
        tags[t] = 1.0

    filtered_tags = {
        t: c for t, c in sorted(
            tags.items(),
            key=lambda i: i[0] if sort_by_alphabetical_order else i[1],
            reverse=not sort_by_alphabetical_order
        )
        if c >= threshold and t not in exclude_tags
    }

    new_tags = {}
    for tag, conf in filtered_tags.items():
        new_tag = tag

        if replace_underscore and tag not in replace_underscore_excludes:
            new_tag = new_tag.replace('_', ' ')

        if escape_tag:
            new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

        if add_confident_as_weight:
            new_tag = f'({new_tag}:{conf:.3f})'

        new_tags[new_tag] = conf

    return new_tags


# ======================
# ONNX 模型封装类
# ======================

class WaifuDiffusionInterrogator:
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        model_location=None
    ):
        self.name = name
        self.model_path = model_path
        self.tags_path = tags_path
        self.model_location = model_location
        self.model = None
        self.tags = None

    def load(self):
        if self.model_location is None:
            raise ValueError("model_location must be specified")

        model_full_path = Path(self.model_location) / self.model_path
        tags_full_path = Path(self.model_location) / self.tags_path

        if not model_full_path.exists():
            raise FileNotFoundError(f"ONNX模型文件不存在：{model_full_path}")
        if not tags_full_path.exists():
            raise FileNotFoundError(f"标签文件不存在：{tags_full_path}")

        import torch  # noqa: F401
        from onnxruntime import InferenceSession

        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info(f"可用的执行提供者: {available}")
        except Exception as e:
            logger.warning(f"无法获取 ONNX Runtime 提供者: {e}")
            available = ['CPUExecutionProvider']

        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'ROCMExecutionProvider' in available:
            providers.append('ROCMExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.model = InferenceSession(str(model_full_path), providers=providers)
        logger.info(f"已加载模型 {self.name}，使用提供者: {self.model.get_providers()}")

        self.tags = pd.read_csv(tags_full_path)

    def unload(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            logger.info(f"已卸载模型 {self.name}")
        if hasattr(self, 'tags'):
            del self.tags

    def interrogate(self, image: Image.Image):
        if self.model is None:
            self.load()

        _, height, _, _ = self.model.get_inputs()[0].shape

        image = image.convert('RGBA')
        bg = Image.new('RGBA', image.size, 'WHITE')
        bg.paste(image, mask=image)
        image = bg.convert('RGB')
        image_np = np.asarray(image)
        image_bgr = image_np[:, :, ::-1]

        image_square = make_square(image_bgr, height)
        image_resized = smart_resize(image_square, height)
        image_input = image_resized.astype(np.float32)
        image_input = np.expand_dims(image_input, 0)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        confidents = self.model.run([output_name], {input_name: image_input})[0]

        tags_df = self.tags[['name']].copy()
        tags_df['confidents'] = confidents[0]

        ratings = dict(tags_df[:4].values)
        tags = dict(tags_df[4:].values)

        return ratings, tags


# ======================
# 数据集与批处理
# ======================

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        try:
            image = Image.open(img_path).convert("RGB")
            return (image, img_path)
        except Exception as e:
            logger.error(f"无法加载图像: {img_path}, 错误: {e}")
            return None


def collate_fn_remove_corrupted(batch):
    return [x for x in batch if x is not None]


# ======================
# 模型下载功能
# ======================

def get_model_file_url(hf_model_id, filename):
    """获取 Hugging Face 模型文件的直接下载 URL"""
    return f"https://huggingface.co/{hf_model_id}/resolve/main/{filename}"


def download_file(url, dest_path, desc=""):
    """下载文件并显示进度"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=desc or os.path.basename(dest_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def check_model_exists(model_key, custom_path=None):
    """检查模型是否已存在"""
    if model_key == "custom":
        return os.path.exists(custom_path) if custom_path else False
    
    model_config = WD_TAGGER_MODELS.get(model_key)
    if not model_config:
        return False
    
    model_path = model_config["path"]
    
    # 检查必需的文件是否存在
    for filename in model_config.get("files", ["model.onnx", "selected_tags.csv"]):
        file_path = os.path.join(model_path, filename)
        if not os.path.exists(file_path):
            return False
    
    return True


def download_model(model_key, download_dir=None, custom_path=None, progress_callback=None):
    """
    下载模型到指定目录
    
    Args:
        model_key: 模型标识符
        download_dir: 下载目录（仅用于非 custom 模型）
        custom_path: 自定义路径（仅用于 custom 模型）
        progress_callback: 进度回调函数 callback(progress, message)
    
    Returns:
        tuple: (success, message, model_path)
    """
    if model_key == "custom":
        if not custom_path:
            return False, "自定义路径不能为空", None
        target_dir = custom_path
        model_config = {"files": ["model.onnx", "selected_tags.csv"]}
    else:
        model_config = WD_TAGGER_MODELS.get(model_key)
        if not model_config:
            return False, f"未知模型: {model_key}", None
        
        if download_dir:
            target_dir = os.path.join(download_dir, model_key)
        else:
            target_dir = model_config["path"]
    
    hf_model_id = model_config.get("hf_model_id")
    files = model_config.get("files", [])
    
    if not hf_model_id or not files:
        return False, "此模型不支持下载", None
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 下载每个文件
    total_files = len(files)
    for i, filename in enumerate(files):
        dest_path = os.path.join(target_dir, filename)
        
        # 如果文件已存在，跳过
        if os.path.exists(dest_path):
            logger.info(f"文件已存在: {dest_path}")
            if progress_callback:
                progress_callback(
                    int((i + 1) / total_files * 100),
                    f"文件已存在，跳过: {filename}"
                )
            continue
        
        if progress_callback:
            progress_callback(
                int(i / total_files * 100),
                f"正在下载 {filename}..."
            )
        
        url = get_model_file_url(hf_model_id, filename)
        logger.info(f"正在下载: {url}")
        
        success = download_file(url, dest_path, desc=filename)
        if not success:
            return False, f"下载 {filename} 失败", None
        
        if progress_callback:
            progress_callback(
                int((i + 1) / total_files * 100),
                f"下载完成: {filename}"
            )
    
    return True, f"模型下载完成: {model_key}", target_dir


# ======================
# 辅助函数
# ======================

def get_model_path(model_key, custom_path=None, auto_download=True, download_dir=None, progress_callback=None):
    """获取模型路径，可选自动下载"""
    if model_key == "custom":
        if not custom_path or not os.path.exists(custom_path):
            raise ValueError("自定义模型路径无效或不存在")
        return custom_path
    
    model_config = WD_TAGGER_MODELS.get(model_key)
    if not model_config:
        raise ValueError(f"未知模型: {model_key}")
    
    model_path = model_config["path"]
    
    # 检查模型是否存在
    if not check_model_exists(model_key):
        if auto_download:
            logger.info(f"模型不存在，正在下载: {model_key}")
            success, msg, downloaded_path = download_model(
                model_key, download_dir, None, progress_callback
            )
            if success:
                # 更新配置中的路径
                WD_TAGGER_MODELS[model_key]["path"] = downloaded_path
                return downloaded_path
            else:
                raise ValueError(f"模型下载失败: {msg}")
        else:
            raise ValueError(f"模型文件夹不存在: {model_path}")
    
    return model_path


def get_available_models():
    """获取可用的模型列表"""
    available = []
    for key, config in WD_TAGGER_MODELS.items():
        if key == "custom":
            continue
        if check_model_exists(key):
            available.append(key)
    return available


def get_model_status():
    """获取所有模型的状态"""
    status = {}
    for key, config in WD_TAGGER_MODELS.items():
        if key == "custom":
            status[key] = {
                "name": config["description"],
                "installed": None,
                "path": ""
            }
            continue
        
        model_path = config["path"]
        installed = check_model_exists(key)
        status[key] = {
            "name": config["description"],
            "installed": installed,
            "path": model_path
        }
    
    return status


# ======================
# 主打标函数
# ======================

def run_wd_tagger(
    image_dir,
    model_key=None,
    custom_model_path=None,
    general_threshold=0.35,
    character_threshold=0.35,
    caption_extension=".txt",
    remove_underscore=True,
    undesired_tags="",
    use_rating_tags=False,
    use_rating_tags_as_last_tag=False,
    character_tags_first=False,
    always_first_tags=None,
    caption_separator=", ",
    append_tags=False,
    batch_size=1,
    recursive=False,
    debug=False,
    frequency_tags=False,
    auto_download=True,
    download_dir=None,
    progress_callback=None
):
    """
    使用 WD-ViT-TAGGER 模型为图像生成标签
    
    Args:
        model_key: 模型标识符（如 "wd-vit-tagger-v3"）
        custom_model_path: 当 model_key="custom" 时的自定义路径
        auto_download: 当模型不存在时是否自动下载
        download_dir: 下载目录
        progress_callback: 进度回调函数
    """
    if not os.path.exists(image_dir):
        return f"❌ 图片文件夹不存在：{image_dir}"
    
    # 获取模型路径
    try:
        if model_key is None:
            model_key = DEFAULT_MODEL_KEY
        model_path = get_model_path(
            model_key, custom_model_path, 
            auto_download=auto_download,
            download_dir=download_dir,
            progress_callback=progress_callback
        )
    except ValueError as e:
        return f"❌ {str(e)}"

    logger.info(f"使用模型: {model_key} @ {model_path}")

    try:
        interrogator = WaifuDiffusionInterrogator(
            f'wd-tagger-{model_key}',
            model_location=model_path
        )
        logger.info("模型加载成功！")
    except Exception as e:
        return f"❌ 模型加载失败：{str(e)}"

    undesired_tags_list = [t.strip() for t in undesired_tags.split(",") if t.strip()] if undesired_tags else []
    always_first_tags_list = [t.strip() for t in always_first_tags.split(",") if t.strip()] if always_first_tags else []

    image_paths = train_util.glob_images_pathlib(image_dir, recursive)
    logger.info(f"找到 {len(image_paths)} 张图片")
    if not image_paths:
        return "❌ 未找到任何图片文件"

    tag_freq = {}

    def run_batch(path_imgs):
        for image, image_path in path_imgs:
            try:
                ratings, tags = interrogator.interrogate(image)

                processed_general = postprocess_tags(
                    tags,
                    threshold=general_threshold,
                    exclude_tags=undesired_tags_list,
                    replace_underscore=remove_underscore,
                    sort_by_alphabetical_order=False,
                    add_confident_as_weight=False,
                    escape_tag=False
                )

                character_tags = {
                    tag: conf for tag, conf in tags.items()
                    if conf >= character_threshold and tag not in undesired_tags_list
                }
                if remove_underscore:
                    character_tags = {tag.replace('_', ' '): conf for tag, conf in character_tags.items()}
                else:
                    character_tags = dict(character_tags)

                combined_tags = list(processed_general.keys())

                for tag in character_tags:
                    if tag not in combined_tags:
                        if character_tags_first:
                            combined_tags.insert(0, tag)
                        else:
                            combined_tags.append(tag)

                if use_rating_tags or use_rating_tags_as_last_tag:
                    rating_names = list(ratings.keys())
                    if rating_names:
                        top_rating = rating_names[0]
                        if top_rating not in undesired_tags_list:
                            rating_tag = top_rating.replace('_', ' ') if remove_underscore else top_rating
                            if use_rating_tags:
                                combined_tags.insert(0, rating_tag)
                            else:
                                combined_tags.append(rating_tag)

                if always_first_tags_list:
                    processed_first = [
                        t.replace('_', ' ') if remove_underscore else t
                        for t in always_first_tags_list
                    ]
                    for tag in processed_first:
                        if tag in combined_tags:
                            combined_tags.remove(tag)
                        combined_tags.insert(0, tag)

                for tag in combined_tags:
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1

                tag_text = caption_separator.join(combined_tags)

                caption_file = os.path.splitext(image_path)[0] + caption_extension
                if append_tags and os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        existing = f.read().strip()
                    if existing:
                        existing_tags = [t.strip() for t in existing.split(caption_separator) if t.strip()]
                        new_tags = [t for t in combined_tags if t not in existing_tags]
                        tag_text = caption_separator.join(existing_tags + new_tags)

                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")

                if debug:
                    logger.info(f"{image_path}: {len(combined_tags)} tags")

            except Exception as e:
                logger.error(f"处理 {image_path} 时出错: {e}")

    dataset = ImageLoadingPrepDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_remove_corrupted
    )

    processed_count = 0
    for batch in tqdm(dataloader, desc="AI打标中"):
        if batch:
            run_batch(batch)
            processed_count += len(batch)

    interrogator.unload()

    model_name = model_key if model_key else DEFAULT_MODEL_KEY
    if frequency_tags:
        top20 = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        freq_info = "\n".join([f"{tag}: {freq}" for tag, freq in top20])
        return f"✅ AI打标完成！\n📦 使用模型: {model_name}\n🖼️ 共处理 {processed_count} 张图片。\n\n前20个高频标签：\n{freq_info}"
    else:
        return f"✅ AI打标完成！\n📦 使用模型: {model_name}\n🖼️ 共处理 {processed_count} 张图片，标签文件已生成。"
