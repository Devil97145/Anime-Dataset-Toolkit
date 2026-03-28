import csv
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import re
import pandas as pd
import shutil
import time
import requests

# 下载相关库 + 兼容性处理
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[WARNING] 安装huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except ImportError:
    print("[WARNING] 安装modelscope...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope>=1.9.5"])  # 指定兼容版本
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

# 网络配置优化（关键修复）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 强制国内镜像
os.environ["REQUESTS_CA_BUNDLE"] = ""  # 解决证书问题（部分环境）

# 默认配置
DEFAULT_MODEL_PATH = r"./models/wd-vit-tagger-v3"
MODEL_CONFIG = {
    "hubface_repo": "SmilingWolf/wd-vit-tagger-v3",
    "cn_mirror_url": "https://hf-mirror.com/SmilingWolf/wd-vit-tagger-v3/resolve/main/",  # 直接下载URL
    "modelscope_repo": "smilingwolf/wd-vit-tagger-v3",
    "required_files": ["model.onnx", "selected_tags.csv"],
    "download_timeout": 300,
    "max_retries": 2
}

# 临时路径添加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 日志类（不变）
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
    logger = DummyLogger()

    def glob_images_pathlib(dir_path, recursive):
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
        dir_path = Path(dir_path)
        return [p for p in (dir_path.rglob("*") if recursive else dir_path.iterdir()) 
                if p.is_file() and p.suffix.lower() in image_extensions]
    train_util = type('train_util', (), {'glob_images_pathlib': glob_images_pathlib})


# ======================
# 模型下载修复（核心解决参数和连接问题）
# ======================

def check_model_exists(model_dir: str) -> bool:
    """检查模型文件是否存在且非空"""
    model_dir = Path(model_dir)
    for file_name in MODEL_CONFIG["required_files"]:
        file_path = model_dir / file_name
        if not file_path.exists() or file_path.stat().st_size < 1024:  # 至少1KB
            logger.info(f"缺失有效文件：{file_name}")
            return False
    return True

def download_file_with_requests(url: str, save_path: str) -> bool:
    """使用requests直接下载文件（绕过API限制）"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=MODEL_CONFIG["download_timeout"])
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return os.path.getsize(save_path) > 1024  # 验证文件大小
    except Exception as e:
        logger.error(f"直接下载失败：{str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def download_model_from_hubface(model_dir: str) -> bool:
    """从Hugging Face下载（结合直接URL下载）"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 先尝试API下载
    def _api_download():
        snapshot_download(
            repo_id=MODEL_CONFIG["hubface_repo"],
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            allow_patterns=MODEL_CONFIG["required_files"],
            ignore_patterns=["*.*.md5", "*.*.sha256"]
        )
        return check_model_exists(str(model_dir))
    
    # API失败则尝试直接文件下载
    if not _api_download():
        logger.info("尝试直接URL下载...")
        all_success = True
        for file_name in MODEL_CONFIG["required_files"]:
            file_url = f"{MODEL_CONFIG['cn_mirror_url']}{file_name}"
            save_path = model_dir / file_name
            if not download_file_with_requests(file_url, str(save_path)):
                all_success = False
                break
        return all_success
    return True

def auto_download_model(model_dir: str) -> bool:
    """自动下载主函数 + 手动下载指引"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理无效文件
    for file in model_dir.glob("*"):
        if file.name in MODEL_CONFIG["required_files"] and file.stat().st_size < 1024:
            file.unlink()
    
    if check_model_exists(str(model_dir)):
        logger.info("模型已就绪")
        return True
    
    # 1. 尝试Hugging Face镜像（含直接下载）
    if download_model_from_hubface(str(model_dir)):
        logger.info("Hugging Face镜像下载成功")
        return True
     
    # 所有方法失败：提供手动下载指引
    logger.error("\n====== 手动下载指引 ======")
    logger.error(f"请手动下载以下文件到 {model_dir} 目录：")
    for file_name in MODEL_CONFIG["required_files"]:
        logger.error(f"- {MODEL_CONFIG['cn_mirror_url']}{file_name}")
    logger.error("下载后重新运行程序即可")
    return False


# ======================
# 图像预处理、标签处理等函数（不变）
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
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])

def smart_resize(img, size):
    if img.shape[0] > size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

tag_escape_pattern = re.compile(r'([\\()])')
def postprocess_tags(tags: dict, threshold=0.35, additional_tags=None, exclude_tags=None,
                    sort_by_alphabetical_order=False, add_confident_as_weight=False,
                    replace_underscore=False, replace_underscore_excludes=None, escape_tag=False) -> dict:
    additional_tags = additional_tags or []
    exclude_tags = exclude_tags or []
    replace_underscore_excludes = replace_underscore_excludes or []
    for t in additional_tags:
        tags[t] = 1.0
    filtered_tags = {t:c for t,c in sorted(tags.items(), key=lambda i: i[0] if sort_by_alphabetical_order else i[1], reverse=not sort_by_alphabetical_order)
                    if c >= threshold and t not in exclude_tags}
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
# 模型封装与打标函数（不变）
# ======================

class WaifuDiffusionInterrogator:
    def __init__(self, name: str, model_path='model.onnx', tags_path='selected_tags.csv', model_location=None):
        self.name = name
        self.model_path = model_path
        self.tags_path = tags_path
        self.model_location = model_location
        self.model = None
        self.tags = None

    def load(self, device='auto'):
        if self.model_location is None:
            raise ValueError("model_location must be specified")
        model_full_path = Path(self.model_location) / self.model_path
        tags_full_path = Path(self.model_location) / self.tags_path

        if not model_full_path.exists() or not tags_full_path.exists():
            logger.info("启动模型下载...")
            if not auto_download_model(self.model_location):
                raise FileNotFoundError("模型文件缺失")

        try:
            from onnxruntime import InferenceSession, SessionOptions
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            logger.info(f"可用执行提供者: {available_providers}")
        except Exception as e:
            raise RuntimeError(f"ONNX Runtime加载失败：{e}")

        providers = []
        if device == 'auto':
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            if 'ROCMExecutionProvider' in available_providers:
                providers.append('ROCMExecutionProvider')
            providers.append('CPUExecutionProvider')
        elif device == 'cuda':
            if 'CUDAExecutionProvider' not in available_providers:
                raise RuntimeError("未检测到CUDA环境")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            raise ValueError(f"不支持的设备：{device}")

        options = SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = InferenceSession(str(model_full_path), options, providers=providers)
        logger.info(f"模型 {self.name} 加载成功，使用设备：{device}")
        self.tags = pd.read_csv(tags_full_path)

    def unload(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tags'):
            del self.tags

    def interrogate(self, image: Image.Image):
        if self.model is None:
            raise RuntimeError("模型未加载")
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


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        try:
            return (Image.open(img_path).convert("RGB"), img_path)
        except Exception as e:
            logger.error(f"加载失败：{img_path}, {e}")
            return None

def collate_fn_remove_corrupted(batch):
    return [x for x in batch if x is not None]


def run_wd_tagger(
    image_dir,
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
    device='auto'
):
    if not os.path.exists(image_dir):
        return f"❌ 图片文件夹不存在：{image_dir}"
    
    model_path = DEFAULT_MODEL_PATH
    if not auto_download_model(model_path):
        return "❌ 模型准备失败，请按指引手动下载"

    try:
        interrogator = WaifuDiffusionInterrogator('wd-vit-tagger-v3', model_location=model_path)
        interrogator.load(device=device)
    except Exception as e:
        return f"❌ 模型加载失败：{str(e)}"

    undesired_tags_list = [t.strip() for t in undesired_tags.split(",") if t.strip()]
    always_first_tags_list = [t.strip() for t in (always_first_tags or "").split(",") if t.strip()]
    image_paths = train_util.glob_images_pathlib(image_dir, recursive)
    if not image_paths:
        return "❌ 未找到图片文件"

    tag_freq = {}
    def run_batch(path_imgs):
        for image, image_path in path_imgs:
            try:
                ratings, tags = interrogator.interrogate(image)
                processed_general = postprocess_tags(tags, general_threshold, exclude_tags=undesired_tags_list,
                                                    replace_underscore=remove_underscore)
                character_tags = {tag.replace('_',' ') if remove_underscore else tag: conf 
                                for tag, conf in tags.items() 
                                if conf >= character_threshold and tag not in undesired_tags_list}
                combined_tags = list(processed_general.keys())
                for tag in character_tags:
                    if tag not in combined_tags:
                        combined_tags.insert(0, tag) if character_tags_first else combined_tags.append(tag)
                if use_rating_tags or use_rating_tags_as_last_tag:
                    rating_names = list(ratings.keys())
                    if rating_names and rating_names[0] not in undesired_tags_list:
                        rating_tag = rating_names[0].replace('_',' ') if remove_underscore else rating_names[0]
                        if use_rating_tags:
                            combined_tags.insert(0, rating_tag)
                        else:
                            combined_tags.append(rating_tag)
                for tag in always_first_tags_list:
                    tag = tag.replace('_',' ') if remove_underscore else tag
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
            except Exception as e:
                logger.error(f"处理失败：{image_path}, {e}")

    dataset = ImageLoadingPrepDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_remove_corrupted
    )

    processed_count = 0
    for batch in tqdm(dataloader, desc=f"AI打标中（设备：{device}）"):
        if batch:
            run_batch(batch)
            processed_count += len(batch)

    interrogator.unload()
    if frequency_tags:
        top20 = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        freq_info = "\n".join([f"{tag}: {freq}" for tag, freq in top20])
        return f"✅ 完成！处理 {processed_count} 张（设备：{device}）。\n前20标签：\n{freq_info}"
    else:
        return f"✅ 完成！处理 {processed_count} 张（设备：{device}），标签已生成。"

