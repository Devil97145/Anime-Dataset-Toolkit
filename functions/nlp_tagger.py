# functions/nlp_tagger.py
"""
轻量化自然语言打标模块
支持从自然语言描述生成图片标签
"""

import os
import sys
import threading
import torch
from queue import Queue
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import traceback
import re
import json
import hashlib

# 日志函数
def log_info(msg):
    print(f"[NLP打标] {msg}", flush=True)

def log_error(msg):
    print(f"[NLP打标] ❌ {msg}", flush=True)

# ========== 全局缓存和模型管理 ==========
class NLPTaggerModel:
    """
    轻量化自然语言打标模型包装器
    """
    
    # 可用模型配置
    MODEL_CONFIGS = {
        "blip": {
            "name": "BLIP (Salesforce)",
            "description": "轻量视觉语言模型，支持图像描述生成",
            "model_name": "Salesforce/blip-image-captioning-base",
            "requires": ["transformers", "torch"],
            "is_available": False
        },
        "qwen-vl": {
            "name": "Qwen-VL",
            "description": "通义千问视觉语言模型",
            "model_name": "Qwen/Qwen-VL-Chat-Int4",
            "requires": ["transformers", "torch", "accelerate"],
            "is_available": False
        },
        "llava": {
            "name": "LLaVA",
            "description": "开源视觉语言模型",
            "model_name": "llava-hf/llava-1.5-7b-hf",
            "requires": ["transformers", "torch", "accelerate"],
            "is_available": False
        }
    }
    
    # 默认使用的模型
    DEFAULT_MODEL = "blip"
    
    def __init__(self, model_key=None):
        self.model_key = model_key or self.DEFAULT_MODEL
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self._is_loaded = False
        self._init_model_configs()
    
    def _init_model_configs(self):
        """初始化模型配置的可用性检测"""
        for key, config in self.MODEL_CONFIGS.items():
            config["is_available"] = self._check_requirements(config["requires"])
    
    def _check_requirements(self, reqs):
        """检查依赖是否已安装"""
        try:
            for req in reqs:
                __import__(req)
            return True
        except ImportError:
            return False
    
    def _get_device(self):
        """获取可用设备"""
        if torch.cuda.is_available():
            log_info("使用 GPU (CUDA)")
            return "cuda"
        elif torch.backends.mps.is_available():
            log_info("使用 GPU (MPS)")
            return "mps"
        else:
            log_info("使用 CPU")
            return "cpu"
    
    def is_available(self):
        """检查当前模型是否可用"""
        config = self.MODEL_CONFIGS.get(self.model_key)
        return config is not None and config["is_available"]
    
    def load(self):
        """加载模型"""
        if self._is_loaded:
            return True
        
        if not self.is_available():
            raise RuntimeError(f"模型 {self.model_key} 不可用，依赖未安装")
        
        log_info(f"正在加载模型: {self.MODEL_CONFIGS[self.model_key]['name']}")
        
        try:
            if self.model_key == "blip":
                self._load_blip()
            elif self.model_key == "qwen-vl":
                self._load_qwen_vl()
            elif self.model_key == "llava":
                self._load_llava()
            
            self._is_loaded = True
            log_info("✅ 模型加载成功")
            return True
        except Exception as e:
            log_error(f"模型加载失败: {str(e)}")
            return False
    
    def _load_blip(self):
        """加载 BLIP 模型"""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        model_name = self.MODEL_CONFIGS["blip"]["model_name"]
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
    
    def _load_qwen_vl(self):
        """加载 Qwen-VL 模型"""
        # 简化版实现，实际可能需要更复杂的配置
        log_info("Qwen-VL 加载功能开发中")
        raise NotImplementedError("Qwen-VL 暂时不可用")
    
    def _load_llava(self):
        """加载 LLaVA 模型"""
        log_info("LLaVA 加载功能开发中")
        raise NotImplementedError("LLaVA 暂时不可用")
    
    def unload(self):
        """卸载模型"""
        if self._is_loaded:
            self.model = None
            self.processor = None
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_loaded = False
            log_info("模型已卸载")
    
    def generate_caption(self, image, prompt=None):
        """
        生成图像描述
        
        Args:
            image: PIL Image 或路径
            prompt: 可选的提示词
        
        Returns:
            str: 生成的描述文本
        """
        if not self._is_loaded:
            self.load()
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        try:
            if self.model_key == "blip":
                return self._generate_blip_caption(image, prompt)
            # 其他模型类似处理
            else:
                return "Model not supported yet"
        except Exception as e:
            log_error(f"生成描述失败: {str(e)}")
            traceback.print_exc()
            return ""
    
    def _generate_blip_caption(self, image, prompt=None):
        """使用 BLIP 生成图像描述"""
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=100, num_beams=3)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    @staticmethod
    def get_available_models():
        """获取所有可用模型列表"""
        available = []
        for key, config in NLPTaggerModel.MODEL_CONFIGS.items():
            if config["is_available"]:
                available.append((key, config["name"], config["description"]))
        return available


# ========== 描述文本转标签 ==========
class DescriptionToTags:
    """
    将自然语言描述转换为标签系统
    """
    
    # 通用标签词库（英文，适合 Danbooru 风格）
    CATEGORIES = {
        "character": [
            "girl", "boy", "woman", "man", "child", "person", 
            "1girl", "solo", "2girls", "multiple girls", "1boy", "2boys",
            "solo focus", "looking at viewer", "smiling", "serious"
        ],
        "appearance": [
            "blue hair", "red hair", "blonde hair", "black hair", "brown hair", "white hair",
            "long hair", "short hair", "twintails", "ponytail",
            "blue eyes", "red eyes", "green eyes", "yellow eyes", "purple eyes", "black eyes",
            "blush", "glasses", "expressionless"
        ],
        "clothing": [
            "school uniform", "swimsuit", "dress", "shirt", "skirt", "pants",
            "white shirt", "black skirt", "ribbon", "tie", "socks", "shoes",
            "bikini", "one-piece swimsuit", "casual", "jacket", "hoodie"
        ],
        "scene": [
            "outdoors", "indoors", "school", "beach", "park", "forest", "city",
            "night", "day", "sunset", "sky", "cloud", "tree", "building"
        ],
        "action": [
            "sitting", "standing", "walking", "lying", "eating", "drinking", "reading",
            "holding object", "looking away", "smile", "frown", "waving"
        ]
    }
    
    # 同义词映射
    SYNONYMS = {
        "female": "girl", "male": "boy",
        "kid": "child", "lady": "woman",
        "sunset": "sunset sky",
        "water": "ocean", "sea", "lake", "river",
        "tree": "trees",
        "happy": "smile", "sad": "frown"
    }
    
    def __init__(self, lang="en"):
        self.lang = lang
    
    def extract_tags(self, description, method="keyword"):
        """
        从描述中提取标签
        
        Args:
            description: 自然语言描述
            method: 提取方法 ('keyword', 'nlp')
        
        Returns:
            list: 提取的标签列表
        """
        if not description or not description.strip():
            return []
        
        tags = []
        
        if method == "keyword":
            tags = self._extract_by_keywords(description)
        elif method == "nlp":
            tags = self._extract_by_nlp(description)
        
        # 后处理
        tags = self._postprocess_tags(tags)
        
        return tags
    
    def _extract_by_keywords(self, description):
        """基于关键词匹配提取标签"""
        tags = []
        desc_lower = description.lower()
        
        # 从词库中匹配
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in desc_lower:
                    tags.append(keyword)
        
        # 简单的名词提取（补充）
        words = re.findall(r'\b[a-z]{2,}s?\b', desc_lower)
        for word in words:
            # 去重并添加不在词库但可能有用的词
            if word not in tags and len(word) >= 3:
                # 排除停用词
                stop_words = {
                    "the", "a", "an", "is", "are", "was", "were", 
                    "be", "to", "for", "of", "and", "with", "on", 
                    "in", "at", "by", "that", "this", "from"
                }
                if word not in stop_words:
                    # 保持原始大小写（尽可能）
                    original_word = self._find_original_word(description, word)
                    if original_word:
                        tags.append(original_word.lower())
        
        return tags
    
    def _find_original_word(self, text, word_lower):
        """找到原文中的对应单词"""
        pattern = re.compile(r'\b' + re.escape(word_lower) + r'\b', re.IGNORECASE)
        match = pattern.search(text)
        return match.group() if match else word_lower
    
    def _extract_by_nlp(self, description):
        """使用简单 NLP 方法提取（需要 spaCy 等）"""
        # 简化版，暂时使用关键词方法
        return self._extract_by_keywords(description)
    
    def _postprocess_tags(self, tags):
        """标签后处理：去重、排序、规范化"""
        # 去重
        seen = set()
        unique_tags = []
        for tag in tags:
            normalized = tag.strip().lower()
            if normalized not in seen and normalized:
                seen.add(normalized)
                # 同义词替换
                for syn, target in self.SYNONYMS.items():
                    if normalized == syn:
                        normalized = target
                        break
                unique_tags.append(normalized)
        
        # 排序
        unique_tags.sort()
        
        return unique_tags


# ========== 批量打标处理器 ==========
class NLPTaggerBatchProcessor:
    """
    批量自然语言打标处理器
    """
    
    def __init__(
        self, 
        image_dir, 
        model_key="blip",
        extraction_method="keyword",
        caption_extension=".txt",
        output_dir=None,
        overwrite=False,
        max_threads=1,
        auto_clean=True
    ):
        self.image_dir = image_dir
        self.model_key = model_key
        self.extraction_method = extraction_method
        self.caption_extension = caption_extension
        self.output_dir = output_dir or image_dir
        self.overwrite = overwrite
        self.max_threads = max_threads
        self.auto_clean = auto_clean
        
        # 初始化组件
        self.tagger_model = NLPTaggerModel(model_key)
        self.description_converter = DescriptionToTags()
        
        # 状态变量
        self.image_paths = []
        self.task_queue = Queue()
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # 支持的图片扩展名
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        
        if not os.path.exists(self.image_dir):
            raise ValueError(f"图片文件夹不存在: {self.image_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def scan_images(self):
        """扫描图片"""
        self.image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(self.image_extensions):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
        
        log_info(f"找到 {len(self.image_paths)} 张图片")
        return len(self.image_paths) > 0
    
    def process_image(self, img_path):
        """处理单张图片"""
        try:
            img_filename = os.path.basename(img_path)
            base_name = os.path.splitext(img_filename)[0]
            
            # 确定输出文件
            relative_path = os.path.relpath(img_path, self.image_dir)
            output_subdir = os.path.dirname(relative_path)
            output_path = os.path.join(self.output_dir, output_subdir, base_name + self.caption_extension)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 检查是否跳过
            if os.path.exists(output_path) and not self.overwrite:
                log_info(f"跳过已存在: {img_filename}")
                return True, "Skipped (exists)"
            
            # 1. 生成描述
            log_info(f"正在处理: {img_filename}")
            caption = self.tagger_model.generate_caption(img_path)
            
            if not caption:
                return False, "No caption generated"
            
            # 2. 转换为标签
            tags = self.description_converter.extract_tags(caption, self.extraction_method)
            
            # 3. 保存标签
            tag_text = ", ".join(tags)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(tag_text)
            
            log_info(f"✅ {img_filename}: {len(tags)} 标签")
            return True, f"Generated {len(tags)} tags"
        
        except Exception as e:
            log_error(f"处理失败 {os.path.basename(img_path)}: {str(e)}")
            traceback.print_exc()
            return False, str(e)
    
    def worker(self):
        """工作线程"""
        while not self.stop_event.is_set():
            try:
                if self.task_queue.empty():
                    threading.Event().wait(0.5)
                    continue
                
                img_path = self.task_queue.get(timeout=5)
                success, msg = self.process_image(img_path)
                
                with self.lock:
                    self.processed_count += 1
                    if success:
                        self.success_count += 1
                    else:
                        self.failed_count += 1
                
                self.task_queue.task_done()
            
            except Exception as e:
                if self.task_queue.empty():
                    break
                log_error(f"工作线程错误: {str(e)}")
    
    def start(self):
        """开始批量处理"""
        if not self.scan_images():
            log_info("没有找到图片")
            return
        
        # 加载模型
        if not self.tagger_model.load():
            log_error("模型加载失败")
            return
        
        # 填充任务队列
        for img_path in self.image_paths:
            self.task_queue.put(img_path)
        
        log_info(f"开始处理 {len(self.image_paths)} 张图片...")
        
        # 启动工作线程
        threads = []
        for _ in range(self.max_threads):
            thread = threading.Thread(target=self.worker)
            thread.daemon = False
            thread.start()
            threads.append(thread)
        
        # 等待完成
        self.task_queue.join()
        self.stop_event.set()
        
        for thread in threads:
            thread.join(timeout=10)
        
        # 清理
        if self.auto_clean:
            self.tagger_model.unload()
        
        # 统计
        log_info("\n" + "="*50)
        log_info(f"处理完成！成功: {self.success_count} | 失败: {self.failed_count}")
        log_info("="*50)


# ========== 模块级导出函数 ==========
def get_available_models():
    """获取可用模型列表"""
    return NLPTaggerModel.get_available_models()


def run_nlp_tagger(
    image_dir,
    model_key="blip",
    extraction_method="keyword",
    caption_extension=".txt",
    output_dir=None,
    overwrite=False,
    max_threads=1,
    auto_clean=True
):
    """
    模块级函数：运行 NLP 打标
    """
    try:
        processor = NLPTaggerBatchProcessor(
            image_dir=image_dir,
            model_key=model_key,
            extraction_method=extraction_method,
            caption_extension=caption_extension,
            output_dir=output_dir,
            overwrite=overwrite,
            max_threads=max_threads,
            auto_clean=auto_clean
        )
        
        processor.start()
        
        return f"✅ NLP 打标完成！成功: {processor.success_count} 张，失败: {processor.failed_count} 张"
    
    except Exception as e:
        log_error(f"打标失败: {str(e)}")
        traceback.print_exc()
        return f"❌ NLP 打标失败: {str(e)}"


def tag_single_image(image_path, model_key="blip"):
    """
    模块级函数：单张图片打标测试
    
    Returns:
        tuple: (caption, tags)
    """
    try:
        tagger = NLPTaggerModel(model_key)
        tagger.load()
        
        caption = tagger.generate_caption(image_path)
        converter = DescriptionToTags()
        tags = converter.extract_tags(caption)
        
        tagger.unload()
        
        return caption, tags
    
    except Exception as e:
        log_error(f"单张打标失败: {str(e)}")
        traceback.print_exc()
        return str(e), []


# ========== 自动安装依赖（可选） ==========
def install_requirements(requirements=None):
    """
    尝试安装所需依赖
    
    Args:
        requirements: 可选的依赖列表，默认安装 BLIP 需要的
    """
    if requirements is None:
        requirements = ["transformers", "torch", "pillow"]
    
    import subprocess
    import sys
    
    log_info("尝试安装依赖...")
    for req in requirements:
        try:
            __import__(req)
            log_info(f"✅ {req} 已安装")
        except ImportError:
            log_info(f"正在安装 {req}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                log_info(f"✅ {req} 安装成功")
            except Exception as e:
                log_error(f"❌ {req} 安装失败: {str(e)}")
