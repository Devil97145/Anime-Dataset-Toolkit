# functions/txt_image_processor.py

import os
import sys
import shutil
import threading
from queue import Queue
import traceback
import re
from collections import Counter

class TxtWithImageProcessor:
    def __init__(self, source_dir, keyword, action, target_dir=None, case_sensitive=False, 
                 fuzzy_match=False, max_threads=4, match_mode='or'):
        """
        match_mode: 'or' 表示任一关键词匹配即命中，'and' 表示所有关键词必须都出现
        """
        self.source_dir = source_dir
        self.keyword = keyword
        self.action = action
        self.target_dir = target_dir
        self.case_sensitive = case_sensitive
        self.fuzzy_match = fuzzy_match
        self.max_threads = max_threads
        self.match_mode = match_mode.lower()  # 支持 'or' / 'and'
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        
        self.txt_file_list = []
        self.matched_txt_list = []
        self.total_txt_files = 0
        self.scanned_txt_files = 0
        self.task_queue = Queue()
        self.processed_txt = 0
        self.success_txt = 0
        self.success_images = 0
        self.failed_files = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        if not os.path.exists(self.source_dir):
            raise ValueError(f"源文件夹不存在：{self.source_dir}")
        if self.action not in ["move", "delete"]:
            raise ValueError("动作仅支持 'move' 或 'delete'")
        if self.action == "move" and (not self.target_dir or not os.path.exists(self.target_dir)):
            raise ValueError(f"目标文件夹不存在：{self.target_dir}")
        if self.match_mode not in ['or', 'and']:
            raise ValueError("匹配模式仅支持 'or' 或 'and'")

        # 预处理关键词列表
        self.keyword_list = self._parse_keywords(keyword)
        if not self.keyword_list:
            raise ValueError("关键词不能为空")

    def _parse_keywords(self, keyword_str):
        """解析关键词字符串，支持多种分隔符，但保留关键词内部空格"""
        if not keyword_str.strip():
            return []
        separators = r'[,;，；\n]+'
        parts = re.split(separators, keyword_str.strip())
        keywords = [part.strip() for part in parts if part.strip()]
        return keywords

    def load_txt_files(self):
        self.txt_file_list = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    txt_path = os.path.join(root, file)
                    self.txt_file_list.append(txt_path)
        
        self.total_txt_files = len(self.txt_file_list)
        print(f"[信息] 扫描完成：共找到 {self.total_txt_files} 个TXT文件", flush=True)
        print(f"[信息] 关键词列表（{self.match_mode.upper()}模式）：{self.keyword_list}", flush=True)
        
        if self.total_txt_files == 0:
            print("[信息] 未找到任何TXT文件，无需处理", flush=True)
            return False
        
        for txt_path in self.txt_file_list:
            self.task_queue.put(txt_path)
        return True

    def _scan_txt_content(self, txt_path):
        filename = os.path.basename(txt_path)
        try:
            content = ""
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(txt_path, "r", encoding="gbk", errors="replace") as f:
                    content = f.read()

            text = content
            keywords_to_match = self.keyword_list[:]
            if not self.case_sensitive:
                text = text.lower()
                keywords_to_match = [kw.lower() for kw in keywords_to_match]

            if self.fuzzy_match:
                text = re.sub(r'[_\-]+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                keywords_to_match = [re.sub(r'[_\-]+', ' ', kw) for kw in keywords_to_match]
                keywords_to_match = [re.sub(r'\s+', ' ', kw).strip() for kw in keywords_to_match]

            matched_keywords = []
            for kw in keywords_to_match:
                if kw in text:
                    matched_keywords.append(kw)

            is_matched = False
            if self.match_mode == 'or':
                is_matched = len(matched_keywords) > 0
            elif self.match_mode == 'and':
                is_matched = len(matched_keywords) == len(keywords_to_match)

            if is_matched:
                txt_dir = os.path.dirname(txt_path)
                txt_prefix = os.path.splitext(filename)[0]
                related_images = []

                for file in os.listdir(txt_dir):
                    file_ext = os.path.splitext(file)[1].lower()
                    file_prefix = os.path.splitext(file)[0]
                    if file_ext in self.image_extensions and file_prefix == txt_prefix:
                        img_path = os.path.join(txt_dir, file)
                        related_images.append(img_path)

                return True, txt_path, related_images
            else:
                return False, txt_path, []

        except Exception as e:
            print(f"[错误] 读取{filename}失败：{str(e)}", file=sys.stderr, flush=True)
            return False, txt_path, []

    def scan_content_worker(self):
        thread_id = threading.get_ident()
        print(f"[调试] 扫描线程 {thread_id} 启动", flush=True)

        while not self.stop_event.is_set():
            try:
                if self.task_queue.empty():
                    threading.Event().wait(0.5)
                    continue

                txt_path = self.task_queue.get(timeout=5)
                self.scanned_txt_files += 1
                filename = os.path.basename(txt_path)

                is_matched, _, related_images = self._scan_txt_content(txt_path)
                if is_matched:
                    with self.lock:
                        self.matched_txt_list.append({"txt": txt_path, "images": related_images})
                    img_count = len(related_images)
                    print(f"[匹配成功] {filename} | 关联图片：{img_count} 张", flush=True)
                else:
                    print(f"[匹配失败] {filename} | 未匹配关键词", flush=True)

                progress = (self.scanned_txt_files / self.total_txt_files) * 100
                print(f"[扫描进度] {progress:.1f}% | 已扫描 {self.scanned_txt_files}/{self.total_txt_files} 个TXT", flush=True)

            except Exception as e:
                if self.task_queue.empty():
                    print(f"[调试] 扫描线程 {thread_id} 任务完成，退出", flush=True)
                    break
                print(f"[调试] 扫描线程 {thread_id} 临时错误：{str(e)}", flush=True)
            finally:
                if 'txt_path' in locals():
                    self.task_queue.task_done()

    def scan_all_content(self):
        print(f"[启动] 开始扫描TXT内容（关键词：{self.keyword_list}，模式：{self.match_mode.upper()}），自动查找同名图片", flush=True)

        scan_threads = min(self.max_threads, self.total_txt_files, os.cpu_count() * 2)
        actual_threads = max(1, scan_threads)
        print(f"[调试] 启动 {actual_threads} 个扫描线程", flush=True)

        threads = []
        for _ in range(actual_threads):
            thread = threading.Thread(target=self.scan_content_worker)
            thread.daemon = False
            thread.start()
            threads.append(thread)

        self.task_queue.join()
        self.stop_event.set()

        for thread in threads:
            thread.join(timeout=10)
            if thread.is_alive():
                print(f"[警告] 扫描线程未正常退出", flush=True)

        matched_count = len(self.matched_txt_list)
        total_related_images = sum(len(item["images"]) for item in self.matched_txt_list)
        print(f"\n[扫描结果] 共匹配 {matched_count} 个TXT文件，关联 {total_related_images} 张图片", flush=True)
        return matched_count > 0

    def _get_target_path(self, source_path, base_prefix=None):
        if self.action != "move":
            return None

        source_filename = os.path.basename(source_path)
        source_ext = os.path.splitext(source_filename)[1]
        
        if base_prefix:
            target_filename = f"{base_prefix}{source_ext}"
        else:
            target_filename = source_filename

        target_path = os.path.join(self.target_dir, target_filename)
        counter = 1

        while os.path.exists(target_path):
            if base_prefix:
                target_filename = f"{base_prefix}_{counter}{source_ext}"
            else:
                name = os.path.splitext(source_filename)[0]
                target_filename = f"{name}_{counter}{source_ext}"
            target_path = os.path.join(self.target_dir, target_filename)
            counter += 1

        return target_path

    def process_related_files(self, txt_item):
        txt_path = txt_item["txt"]
        img_paths = txt_item["images"]
        txt_filename = os.path.basename(txt_path)
        current_success = 0
        current_failed = 0

        try:
            if self.action == "move":
                txt_target = self._get_target_path(txt_path)
                new_txt_prefix = os.path.splitext(os.path.basename(txt_target))[0]
                shutil.move(txt_path, txt_target)
                print(f"[成功] 移动TXT：{txt_filename} → {os.path.basename(txt_target)}", flush=True)
                current_success += 1

                for img_path in img_paths:
                    img_filename = os.path.basename(img_path)
                    img_target = self._get_target_path(img_path, base_prefix=new_txt_prefix)
                    shutil.move(img_path, img_target)
                    print(f"[成功] 移动图片：{img_filename} → {os.path.basename(img_target)}", flush=True)
                    current_success += 1
                    with self.lock:
                        self.success_images += 1

            elif self.action == "delete":
                os.remove(txt_path)
                print(f"[成功] 删除TXT：{txt_filename}", flush=True)
                current_success += 1

                for img_path in img_paths:
                    img_filename = os.path.basename(img_path)
                    os.remove(img_path)
                    print(f"[成功] 删除图片：{img_filename}", flush=True)
                    current_success += 1
                    with self.lock:
                        self.success_images += 1

        except Exception as e:
            print(f"[错误] 处理{txt_filename}失败：{str(e)}", file=sys.stderr, flush=True)
            current_failed += 1

        return current_success, current_failed

    def process_worker(self):
        thread_id = threading.get_ident()
        print(f"[调试] 处理线程 {thread_id} 启动", flush=True)

        while not self.stop_event.is_set():
            try:
                if self.task_queue.empty():
                    threading.Event().wait(0.5)
                    continue

                txt_item = self.task_queue.get(timeout=5)
                txt_filename = os.path.basename(txt_item["txt"])
                print(f"[调试] 线程 {thread_id} 开始处理：{txt_filename}", flush=True)

                success_cnt, failed_cnt = self.process_related_files(txt_item)

                with self.lock:
                    self.processed_txt += 1
                    self.success_txt += 1 if success_cnt > 0 else 0
                    self.failed_files += failed_cnt

                total_to_process = len(self.matched_txt_list)
                progress = (self.processed_txt / total_to_process) * 100
                print(f"[处理进度] {progress:.1f}% | 已处理 {self.processed_txt}/{total_to_process} 个TXT", flush=True)

            except Exception as e:
                if self.task_queue.empty():
                    print(f"[调试] 处理线程 {thread_id} 任务完成，退出", flush=True)
                    break
                print(f"[调试] 处理线程 {thread_id} 临时错误：{str(e)}", flush=True)
            finally:
                if 'txt_item' in locals():
                    self.task_queue.task_done()

    def process_matched_files(self):
        total_to_process = len(self.matched_txt_list)
        if total_to_process == 0:
            print("[信息] 没有匹配的文件需要处理", flush=True)
            return

        self.task_queue = Queue()
        for txt_item in self.matched_txt_list:
            self.task_queue.put(txt_item)

        total_related_images = sum(len(item["images"]) for item in self.matched_txt_list)
        print(f"[启动] 开始处理 {total_to_process} 个TXT + {total_related_images} 张图片", flush=True)

        process_threads = min(self.max_threads, total_to_process, os.cpu_count() * 2)
        actual_threads = max(1, process_threads)
        print(f"[调试] 启动 {actual_threads} 个处理线程", flush=True)

        self.stop_event.clear()
        threads = []
        for _ in range(actual_threads):
            thread = threading.Thread(target=self.process_worker)
            thread.daemon = False
            thread.start()
            threads.append(thread)

        self.task_queue.join()
        self.stop_event.set()

        for thread in threads:
            thread.join(timeout=10)
            if thread.is_alive():
                print(f"[警告] 处理线程未正常退出", flush=True)

        print(f"\n[处理结果] 共处理 {self.processed_txt} 个TXT，成功 {self.success_txt} 个，失败 {self.failed_files} 个")
        print(f"[处理结果] 共处理图片 {self.success_images} 张", flush=True)

    def start(self):
        try:
            if self.action == "delete":
                print("⚠️  ⚠️  ⚠️  警告：你正在使用 DELETE 模式，文件将被永久删除！")
                print("⚠️  请确认操作，确保已备份重要数据！")
                print("")

            if not self.load_txt_files():
                return

            self.stop_event.clear()
            if not self.scan_all_content():
                return

            self.process_matched_files()

        except KeyboardInterrupt:
            self.stop_event.set()
            print("\n[停止] 已手动终止处理", flush=True)
        except Exception as e:
            print(f"\n[错误] 程序异常：{str(e)}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    @staticmethod
    def get_top_keywords_from_dir(source_dir, top_n=20):
        """
        扫描目录下所有TXT，按逗号分隔提取标签（保留空格），统计高频词组
        适用于：1girl, red eyes, blue scrunchie 格式
        返回：[(词组, 频次), ...]
        """
        from collections import Counter
        word_counter = Counter()
        txt_count = 0

        if not os.path.exists(source_dir):
            return []

        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(".txt"):
                    txt_path = os.path.join(root, file)
                    try:
                        content = ""
                        try:
                            with open(txt_path, "r", encoding="utf-8") as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            with open(txt_path, "r", encoding="gbk", errors="replace") as f:
                                content = f.read()

                        raw_tags = content.split(',')
                        cleaned_tags = [
                            tag.strip()
                            for tag in raw_tags
                            if tag.strip()
                            and not tag.strip().isdigit()
                            and len(tag.strip()) >= 2
                        ]

                        word_counter.update(cleaned_tags)
                        txt_count += 1
                    except Exception as e:
                        print(f"[词频分析] 读取 {file} 失败：{str(e)}", flush=True)

        print(f"[词频分析] 共扫描 {txt_count} 个TXT文件，提取高频词中...", flush=True)
        top_words = word_counter.most_common(top_n)
        return top_words


# ========== 模块级函数（方便从外部直接导入）==========
def get_top_keywords_from_dir(source_dir, top_n=20):
    """
    顶层函数：扫描目录下所有TXT，返回高频标签列表
    """
    return TxtWithImageProcessor.get_top_keywords_from_dir(source_dir, top_n)