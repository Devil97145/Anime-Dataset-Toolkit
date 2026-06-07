# functions/nlp_tagger.py
"""
轻量化自然语言打标模块
支持 BLIP, Qwen-VL, LLaVA 等视觉语言模型
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

def log_warning(msg):
    print(f"[NLP打标] ⚠️ {msg}", flush=True)


# ========== 模型管理器 ==========
class ModelManager:
    """统一的模型管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.loaded_models = {}
        self.device = self._get_device()
        log_info(f"使用设备: {self.device}")
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def get_model(self, model_key):
        """获取或加载模型"""
        if model_key not in self.loaded_models:
            self.loaded_models[model_key] = self._load_model(model_key)
        return self.loaded_models[model_key]
    
    def _load_model(self, model_key):
        """加载指定模型"""
        if model_key == "blip":
            return self._load_blip_model()
        elif model_key == "qwen-vl":
            return self._load_qwen_vl_model()
        elif model_key == "llava":
            return self._load_llava_model()
        elif model_key == "qwen2-vl":
            return self._load_qwen2_vl_model()
        else:
            raise ValueError(f"不支持的模型: {model_key}")
    
    def _load_blip_model(self):
        """加载 BLIP 模型"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            log_info("正在加载 BLIP 模型...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            model = model.to(self.device)
            
            return {
                "processor": processor,
                "model": model,
                "type": "blip"
            }
        except Exception as e:
            log_error(f"BLIP 加载失败: {str(e)}")
            raise
    
    def _load_qwen_vl_model(self):
        """加载 Qwen-VL 模型"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            log_info("正在加载 Qwen2-VL 模型...")
            model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
            
            # 加载处理器和模型
            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            return {
                "processor": processor,
                "model": model,
                "type": "qwen2-vl"
            }
        except Exception as e:
            log_error(f"Qwen2-VL 加载失败: {str(e)}")
            # 回退到 CPU 友好版本
            log_info("尝试加载轻量版本...")
            return self._load_qwen_vl_light()
    
    def _load_qwen_vl_light(self):
        """加载 Qwen-VL 轻量版本"""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            log_info("正在加载 Qwen2-VL 轻量版本...")
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            
            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu"
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            return {
                "processor": processor,
                "model": model,
                "type": "qwen2-vl"
            }
        except Exception as e:
            log_error(f"Qwen-VL 轻量版加载失败: {str(e)}")
            raise
    
    def _load_qwen2_vl_model(self):
        """加载 Qwen2-VL (使用 Qwen2 架构)"""
        return self._load_qwen_vl_model()
    
    def _load_llava_model(self):
        """加载 LLaVA 模型"""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            log_info("正在加载 LLaVA 模型...")
            
            # 使用 LLaVA 1.6 版本
            model_name = "llava-hf/llava-1.6-mistral-7b-hf"
            
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            return {
                "processor": processor,
                "model": model,
                "type": "llava"
            }
        except Exception as e:
            log_error(f"LLaVA 加载失败: {str(e)}")
            # 回退到更轻量的版本
            return self._load_llava_light()
    
    def _load_llava_light(self):
        """加载 LLaVA 轻量版本"""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            log_info("正在加载 LLaVA 轻量版本...")
            model_name = "llava-hf/llava-1.5-7b-hf"
            
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu"
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            return {
                "processor": processor,
                "model": model,
                "type": "llava"
            }
        except Exception as e:
            log_error(f"LLaVA 轻量版加载失败: {str(e)}")
            raise
    
    def unload_model(self, model_key):
        """卸载指定模型"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_info(f"已卸载模型: {model_key}")
    
    def unload_all(self):
        """卸载所有模型"""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_info("已卸载所有模型")


# 全局模型管理器实例
_model_manager = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ========== 视觉语言模型推理器 ==========
class VisionModelInferrer:
    """
    统一的多模型推理接口
    """
    
    # 支持的模型及其配置
    MODEL_CONFIGS = {
        "blip": {
            "name": "BLIP",
            "description": "轻量通用模型，Salesforce出品，速度快",
            "default_prompt": "a photography of",
            "requires_gpu": False,
            "model_size": "~1GB",
            "is_available": None,  # 动态检测
        },
        "qwen2-vl": {
            "name": "Qwen2-VL (2B)",
            "description": "通义千问2视觉版，7B参数，适合详细描述",
            "default_prompt": "描述这张图片的内容，包括主体、背景、风格等细节。",
            "requires_gpu": True,
            "model_size": "~4GB",
            "is_available": None,
        },
        "qwen-vl": {
            "name": "Qwen-VL",
            "description": "通义千问视觉版，7B参数",
            "default_prompt": "描述这张图片的内容。",
            "requires_gpu": True,
            "model_size": "~8GB",
            "is_available": None,
        },
        "llava": {
            "name": "LLaVA 1.6",
            "description": "开源多模态模型，7B参数",
            "default_prompt": "Describe this image in detail.",
            "requires_gpu": True,
            "model_size": "~7GB",
            "is_available": None,
        }
    }
    
    def __init__(self, model_key="blip"):
        self.model_key = model_key
        self.config = self.MODEL_CONFIGS.get(model_key, self.MODEL_CONFIGS["blip"])
        self.model_manager = get_model_manager()
        self.model_data = None
        
        # 检查模型可用性
        self._check_availability()
    
    def _check_availability(self):
        """检查模型及其依赖是否可用"""
        try:
            if self.model_key == "blip":
                import transformers
                self.config["is_available"] = True
            elif self.model_key in ["qwen2-vl", "qwen-vl"]:
                import transformers
                self.config["is_available"] = True
            elif self.model_key == "llava":
                import transformers
                self.config["is_available"] = True
            else:
                self.config["is_available"] = False
        except ImportError as e:
            log_warning(f"缺少依赖: {str(e)}")
            self.config["is_available"] = False
    
    def load(self):
        """加载模型"""
        if not self.config["is_available"]:
            raise RuntimeError(f"模型 {self.model_key} 不可用，缺少必要依赖")
        
        log_info(f"正在加载模型: {self.config['name']}")
        self.model_data = self.model_manager.get_model(self.model_key)
        return True
    
    def unload(self):
        """卸载模型"""
        self.model_manager.unload_model(self.model_key)
        self.model_data = None
    
    def generate_caption(self, image, prompt=None):
        """
        生成图像描述
        
        Args:
            image: PIL Image 或图像路径
            prompt: 可选的提示词
        
        Returns:
            str: 生成的描述文本
        """
        if self.model_data is None:
            self.load()
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        model_type = self.model_data.get("type", self.model_key)
        
        try:
            if model_type == "blip":
                return self._generate_blip_caption(image, prompt)
            elif model_type in ["qwen2-vl", "qwen-vl"]:
                return self._generate_qwen_caption(image, prompt)
            elif model_type == "llava":
                return self._generate_llava_caption(image, prompt)
            else:
                return self._generate_blip_caption(image, prompt)  # 默认回退
        except Exception as e:
            log_error(f"生成描述失败: {str(e)}")
            traceback.print_exc()
            return ""
    
    def _generate_blip_caption(self, image, prompt=None):
        """使用 BLIP 生成描述"""
        processor = self.model_data["processor"]
        model = self.model_data["model"]
        
        if prompt:
            inputs = processor(image, text=prompt, return_tensors="pt").to(self.model_manager.device)
        else:
            inputs = processor(image, return_tensors="pt").to(self.model_manager.device)
        
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=3,
            do_sample=True,
            temperature=0.7
        )
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _generate_qwen_caption(self, image, prompt=None):
        """使用 Qwen2-VL 生成描述"""
        processor = self.model_data["processor"]
        model = self.model_data["model"]
        
        default_prompt = self.config.get("default_prompt", "详细描述这张图片的内容。")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt or default_prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.model_manager.device)
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return caption
    
    def _generate_llava_caption(self, image, prompt=None):
        """使用 LLaVA 生成描述"""
        processor = self.model_data["processor"]
        model = self.model_data["model"]
        
        default_prompt = self.config.get("default_prompt", "Describe this image in detail.")
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt or default_prompt}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model_manager.device)
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return caption
    
    @staticmethod
    def get_available_models(device=None):
        """获取可用模型列表"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        available = []
        for key, config in VisionModelInferrer.MODEL_CONFIGS.items():
            if config["is_available"] is None:
                # 动态检测
                try:
                    inferrer = VisionModelInferrer(key)
                    inferrer._check_availability()
                except:
                    config["is_available"] = False
            
            # GPU 模型在没有 GPU 时标记为可选
            if config["requires_gpu"] and device == "cpu":
                continue
            
            if config["is_available"]:
                available.append({
                    "key": key,
                    "name": config["name"],
                    "description": config["description"],
                    "model_size": config["model_size"],
                    "requires_gpu": config["requires_gpu"]
                })
        
        return available


# ========== 描述文本转标签 ==========
class DescriptionToTags:
    """
    将自然语言描述转换为标签系统
    """
    
    # 通用标签词库
    CATEGORIES = {
        "character": [
            "1girl", "solo", "2girls", "multiple girls", "1boy", "2boys",
            "solo focus", "looking at viewer", "smiling", "serious",
            "portrait", "close-up", "cowboy shot", "full body"
        ],
        "appearance": [
            "blue hair", "red hair", "blonde hair", "black hair", "brown hair", "white hair", "pink hair", "purple hair",
            "long hair", "short hair", "twintails", "ponytail", "braid", "hair bun",
            "blue eyes", "red eyes", "green eyes", "yellow eyes", "purple eyes", "heterochromia",
            "blush", "glasses", "expressionless", "angry", "sad", "happy"
        ],
        "clothing": [
            "school uniform", "swimsuit", "dress", "shirt", "skirt", "pants", "shorts",
            "white shirt", "black skirt", "ribbon", "tie", "socks", "shoes", "boots",
            "bikini", "one-piece swimsuit", "casual", "jacket", "hoodie", "coat",
            "maid outfit", "kimono", "traditional clothing", "military uniform"
        ],
        "scene": [
            "outdoors", "indoors", "school", "beach", "park", "forest", "city",
            "night", "day", "sunset", "sunrise", "sky", "cloud", "tree", "building",
            "room", "bedroom", "bathroom", "kitchen", "classroom", "garden"
        ],
        "style": [
            "anime style", "realistic", "photorealistic", "cartoon", "semi-realistic",
            "colorful", "monochrome", "greyscale", "vibrant colors", "pastel colors"
        ],
        "quality": [
            "high quality", "masterpiece", "best quality", "detailed", "simple background",
            "white background", "transparent background", "complex background"
        ]
    }
    
    # 中文到英文标签的映射
    CN_TO_EN = {
        "女孩": "1girl", "男孩": "1boy", "女性": "1girl", "男性": "1boy",
        "长发": "long hair", "短发": "short hair",
        "蓝发": "blue hair", "红发": "red hair", "金发": "blonde hair", "黑发": "black hair", "白发": "white hair",
        "蓝眼": "blue eyes", "红眼": "red eyes", "绿眼": "green eyes",
        "微笑": "smiling", "微笑": "smiling", "微笑": "smiling",
        "学校": "school", "校服": "school uniform",
        "户外": "outdoors", "室内": "indoors",
        "天空": "sky", "云": "cloud", "树": "tree",
        "水": "water", "海": "ocean", "沙滩": "beach",
        "动漫风格": "anime style", "写实": "realistic",
        "高质量": "high quality", " masterpiece": "masterpiece"
    }
    
    # 同义词映射
    SYNONYMS = {
        "female": "1girl", "male": "1boy",
        "kid": "child", "lady": "woman",
        "sunset": "sunset sky", "dusk": "sunset",
        "water": "ocean", "sea": "ocean", "lake": "lake",
        "person": "1person", "people": "multiple people"
    }
    
    def __init__(self, lang="auto"):
        self.lang = lang
    
    def extract_tags(self, description, method="keyword"):
        """从描述中提取标签"""
        if not description or not description.strip():
            return []
        
        tags = []
        
        if method == "keyword":
            tags = self._extract_by_keywords(description)
        elif method == "nlp":
            tags = self._extract_by_nlp(description)
        
        tags = self._postprocess_tags(tags)
        return tags
    
    def _extract_by_keywords(self, description):
        """基于关键词匹配提取标签"""
        tags = []
        desc_lower = description.lower()
        
        # 中文转换
        for cn, en in self.CN_TO_EN.items():
            if cn in description:
                tags.append(en)
        
        # 从词库中匹配
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in desc_lower:
                    tags.append(keyword)
        
        # 简单名词提取
        words = re.findall(r'\b[a-z]{2,}s?\b', desc_lower)
        for word in words:
            if word not in tags and len(word) >= 3:
                stop_words = {
                    "the", "a", "an", "is", "are", "was", "were", 
                    "be", "to", "for", "of", "and", "with", "on", 
                    "in", "at", "by", "that", "this", "from", "has",
                    "have", "had", "but", "not", "with", "only"
                }
                if word not in stop_words:
                    tags.append(word)
        
        return tags
    
    def _extract_by_nlp(self, description):
        """使用简单 NLP 方法提取"""
        return self._extract_by_keywords(description)
    
    def _postprocess_tags(self, tags):
        """标签后处理"""
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
        
        unique_tags.sort()
        return unique_tags


# ========== 批量打标处理器 ==========
class NLPTaggerBatchProcessor:
    """批量自然语言打标处理器"""
    
    def __init__(
        self, 
        image_dir, 
        model_key="blip",
        extraction_method="keyword",
        caption_extension=".txt",
        output_dir=None,
        overwrite=False,
        max_threads=1,
        auto_clean=True,
        custom_prompt=None
    ):
        self.image_dir = image_dir
        self.model_key = model_key
        self.extraction_method = extraction_method
        self.caption_extension = caption_extension
        self.output_dir = output_dir or image_dir
        self.overwrite = overwrite
        self.max_threads = max_threads
        self.auto_clean = auto_clean
        self.custom_prompt = custom_prompt
        
        # 初始化组件
        self.inferrer = VisionModelInferrer(model_key)
        self.description_converter = DescriptionToTags()
        
        # 状态变量
        self.image_paths = []
        self.task_queue = Queue()
        self.processed_count = 0
        self.success_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
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
            
            relative_path = os.path.relpath(img_path, self.image_dir)
            output_subdir = os.path.dirname(relative_path)
            output_path = os.path.join(self.output_dir, output_subdir, base_name + self.caption_extension)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if os.path.exists(output_path) and not self.overwrite:
                log_info(f"跳过已存在: {img_filename}")
                return True, "Skipped (exists)"
            
            log_info(f"正在处理: {img_filename}")
            caption = self.inferrer.generate_caption(img_path, self.custom_prompt)
            
            if not caption:
                return False, "No caption generated"
            
            tags = self.description_converter.extract_tags(caption, self.extraction_method)
            
            tag_text = ", ".join(tags)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(tag_text)
            
            log_info(f"✅ {img_filename}: {len(tags)} 标签 | 描述: {caption[:50]}...")
            return True, f"Generated {len(tags)} tags"
        
        except Exception as e:
            log_error(f"处理失败 {os.path.basename(img_path)}: {str(e)}")
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
        
        if not self.inferrer.load():
            log_error("模型加载失败")
            return
        
        for img_path in self.image_paths:
            self.task_queue.put(img_path)
        
        log_info(f"开始处理 {len(self.image_paths)} 张图片...")
        
        threads = []
        for _ in range(self.max_threads):
            thread = threading.Thread(target=self.worker)
            thread.daemon = False
            thread.start()
            threads.append(thread)
        
        self.task_queue.join()
        self.stop_event.set()
        
        for thread in threads:
            thread.join(timeout=10)
        
        if self.auto_clean:
            self.inferrer.unload()
        
        log_info("\n" + "="*50)
        log_info(f"处理完成！成功: {self.success_count} | 失败: {self.failed_count}")
        log_info("="*50)


# ========== 模块级导出函数 ==========
def get_available_models():
    """获取可用模型列表"""
    return VisionModelInferrer.get_available_models()


def run_nlp_tagger(
    image_dir,
    model_key="blip",
    extraction_method="keyword",
    caption_extension=".txt",
    output_dir=None,
    overwrite=False,
    max_threads=1,
    auto_clean=True,
    custom_prompt=None
):
    """运行 NLP 打标"""
    try:
        processor = NLPTaggerBatchProcessor(
            image_dir=image_dir,
            model_key=model_key,
            extraction_method=extraction_method,
            caption_extension=caption_extension,
            output_dir=output_dir,
            overwrite=overwrite,
            max_threads=max_threads,
            auto_clean=auto_clean,
            custom_prompt=custom_prompt
        )
        
        processor.start()
        
        return f"✅ NLP 打标完成！成功: {processor.success_count} 张，失败: {processor.failed_count} 张"
    
    except Exception as e:
        log_error(f"打标失败: {str(e)}")
        traceback.print_exc()
        return f"❌ NLP 打标失败: {str(e)}"


def tag_single_image(image_path, model_key="blip", custom_prompt=None):
    """单张图片打标测试"""
    try:
        inferrer = VisionModelInferrer(model_key)
        inferrer.load()
        
        caption = inferrer.generate_caption(image_path, custom_prompt)
        converter = DescriptionToTags()
        tags = converter.extract_tags(caption)
        
        inferrer.unload()
        
        return caption, tags
    
    except Exception as e:
        log_error(f"单张打标失败: {str(e)}")
        traceback.print_exc()
        return str(e), []


def check_model_requirements():
    """检查模型依赖"""
    results = {}
    
    # 基础依赖
    try:
        import transformers
        results["transformers"] = True
    except:
        results["transformers"] = False
    
    # 特定模型依赖
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        results["blip"] = True
    except:
        results["blip"] = False
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        results["qwen2-vl"] = True
    except:
        results["qwen2-vl"] = False
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        results["llava"] = True
    except:
        results["llava"] = False
    
    return results


def install_model_requirements(model_key=None):
    """安装模型依赖"""
    import subprocess
    import sys
    
    if model_key is None:
        # 安装所有可能的依赖
        packages = ["transformers>=4.30.0", "torch", "accelerate"]
    elif model_key == "blip":
        packages = ["transformers", "torch"]
    elif model_key in ["qwen2-vl", "qwen-vl"]:
        packages = ["transformers>=4.45.0", "torch>=2.4.0", "accelerate>=0.26.0"]
    elif model_key == "llava":
        packages = ["transformers>=4.45.0", "torch>=2.4.0", "accelerate>=0.26.0"]
    else:
        packages = ["transformers", "torch"]
    
    log_info(f"正在安装依赖: {packages}")
    
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            log_info(f"✅ {pkg} 安装成功")
        except Exception as e:
            log_error(f"❌ {pkg} 安装失败: {str(e)}")
    
    return True
