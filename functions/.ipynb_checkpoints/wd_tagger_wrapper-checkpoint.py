# functions/wd_tagger_wrapper.py

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

# 临时添加当前路径（用于兼容 train_util）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import library.train_util as train_util
    from library.utils import setup_logging
    setup_logging()
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    # 降级处理：创建简化版 train_util 和 logger
    class DummyLogger:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}", flush=True)
        @staticmethod
        def error(msg): print(f"[ERROR] {msg}", flush=True)
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

    # 添加额外标签（置信度=1.0）
    for t in additional_tags:
        tags[t] = 1.0

    # 过滤 + 排序
    filtered_tags = {
        t: c for t, c in sorted(
            tags.items(),
            key=lambda i: i[0] if sort_by_alphabetical_order else i[1],
            reverse=not sort_by_alphabetical_order
        )
        if c >= threshold and t not in exclude_tags
    }

    # 格式化标签
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

        # 加载 torch 以确保 CUDA 库可用（ONNX Runtime 依赖）
        import torch  # noqa: F401
        from onnxruntime import InferenceSession

        # 自动选择执行提供者
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
        providers.append('CPUExecutionProvider')  # fallback

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

        # 转 RGBA → 白底 RGB
        image = image.convert('RGBA')
        bg = Image.new('RGBA', image.size, 'WHITE')
        bg.paste(image, mask=image)
        image = bg.convert('RGB')
        image_np = np.asarray(image)

        # RGB → BGR
        image_bgr = image_np[:, :, ::-1]

        # 预处理
        image_square = make_square(image_bgr, height)
        image_resized = smart_resize(image_square, height)
        image_input = image_resized.astype(np.float32)
        image_input = np.expand_dims(image_input, 0)

        # 推理
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        confidents = self.model.run([output_name], {input_name: image_input})[0]

        # 构建标签 DataFrame
        tags_df = self.tags[['name']].copy()
        tags_df['confidents'] = confidents[0]

        # 前4个是 rating（general, sensitive, questionable, explicit）
        ratings = dict(tags_df[:4].values)
        # 其余是普通标签
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
# 主打标函数
# ======================

def run_wd_tagger(
    image_dir,
    model_path="G:/工具箱/wd-vit-tagger-v3",
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
    frequency_tags=False
):
    """
    使用 WD-ViT-TAGGER-v3 (ONNX) 为图像生成标签
    """
    if not os.path.exists(image_dir):
        return f"❌ 图片文件夹不存在：{image_dir}"
    if not os.path.exists(model_path):
        return f"❌ 模型文件夹不存在：{model_path}"

    # 初始化模型
    try:
        interrogator = WaifuDiffusionInterrogator('wd-vit-tagger-v3', model_location=model_path)
        logger.info("模型加载成功！")
    except Exception as e:
        return f"❌ 模型加载失败：{str(e)}"

    # 解析标签参数
    undesired_tags_list = [t.strip() for t in undesired_tags.split(",") if t.strip()] if undesired_tags else []
    always_first_tags_list = [t.strip() for t in always_first_tags.split(",") if t.strip()] if always_first_tags else []

    # 获取图像列表
    image_paths = train_util.glob_images_pathlib(image_dir, recursive)
    logger.info(f"找到 {len(image_paths)} 张图片")
    if not image_paths:
        return "❌ 未找到任何图片文件"

    tag_freq = {}

    def run_batch(path_imgs):
        for image, image_path in path_imgs:
            try:
                ratings, tags = interrogator.interrogate(image)

                # 处理 general 标签（使用 general_threshold）
                processed_general = postprocess_tags(
                    tags,
                    threshold=general_threshold,
                    exclude_tags=undesired_tags_list,
                    replace_underscore=remove_underscore,  # ✅ 修复：使用参数
                    sort_by_alphabetical_order=False,
                    add_confident_as_weight=False,
                    escape_tag=False
                )

                # 处理 character 标签（使用 character_threshold）
                character_tags = {
                    tag: conf for tag, conf in tags.items()
                    if conf >= character_threshold and tag not in undesired_tags_list
                }
                # 替换下划线（如果启用）
                if remove_underscore:
                    character_tags = {tag.replace('_', ' '): conf for tag, conf in character_tags.items()}
                else:
                    character_tags = dict(character_tags)

                # 合并标签
                combined_tags = list(processed_general.keys())

                # 添加 character 标签
                for tag in character_tags:
                    if tag not in combined_tags:
                        if character_tags_first:
                            combined_tags.insert(0, tag)
                        else:
                            combined_tags.append(tag)

                # 处理 rating 标签
                if use_rating_tags or use_rating_tags_as_last_tag:
                    rating_names = list(ratings.keys())
                    if rating_names:
                        top_rating = rating_names[0]  # 最高置信度
                        if top_rating not in undesired_tags_list:
                            rating_tag = top_rating.replace('_', ' ') if remove_underscore else top_rating
                            if use_rating_tags:
                                combined_tags.insert(0, rating_tag)
                            else:
                                combined_tags.append(rating_tag)

                # 处理 always_first_tags
                if always_first_tags_list:
                    processed_first = [
                        t.replace('_', ' ') if remove_underscore else t
                        for t in always_first_tags_list
                    ]
                    for tag in processed_first:
                        if tag in combined_tags:
                            combined_tags.remove(tag)
                        combined_tags.insert(0, tag)

                # 统计频率
                for tag in combined_tags:
                    tag_freq[tag] = tag_freq.get(tag, 0) + 1

                # 构建最终标签文本
                tag_text = caption_separator.join(combined_tags)

                # 追加模式
                caption_file = os.path.splitext(image_path)[0] + caption_extension
                if append_tags and os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        existing = f.read().strip()
                    if existing:
                        existing_tags = [t.strip() for t in existing.split(caption_separator) if t.strip()]
                        new_tags = [t for t in combined_tags if t not in existing_tags]
                        tag_text = caption_separator.join(existing_tags + new_tags)

                # 保存
                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")

                if debug:
                    logger.info(f"{image_path}: {len(combined_tags)} tags")

            except Exception as e:
                logger.error(f"处理 {image_path} 时出错: {e}")

    # 创建 DataLoader
    dataset = ImageLoadingPrepDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_remove_corrupted
    )

    # 执行批处理
    processed_count = 0
    for batch in tqdm(dataloader, desc="AI打标中"):
        if batch:
            run_batch(batch)
            processed_count += len(batch)

    # 清理
    interrogator.unload()

    # 返回结果
    if frequency_tags:
        top20 = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        freq_info = "\n".join([f"{tag}: {freq}" for tag, freq in top20])
        return f"✅ AI打标完成！共处理 {processed_count} 张图片。\n\n前20个高频标签：\n{freq_info}"
    else:
        return f"✅ AI打标完成！共处理 {processed_count} 张图片，标签文件已生成。"