# video_extractor.py

import os
import subprocess
import glob
import sys
import threading
from queue import Queue
import traceback
import re

class VideoFrameExtractor:
    def __init__(self, input_dir, output_dir, frame_interval=90, max_threads=4):
        """
        初始化视频帧提取器
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.max_threads = max_threads
        
        # 视频文件列表
        self.all_video_files = []
        self.valid_video_files = []
        self.video_total_frames = {}  # {video_path: total_frames}
        
        # 统计信息
        self.total_videos = 0
        self.processed_videos = 0
        self.failed_videos = 0
        
        # 线程控制
        self.task_queue = Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # 支持的视频格式
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[初始化] 🎬 视频帧提取器已创建", flush=True)
        print(f"[初始化] 📂 输入路径: {input_dir}", flush=True)
        print(f"[初始化] 📁 输出路径: {output_dir}", flush=True)

    def load_video_files(self):
        """
        加载视频文件并获取总帧数
        """
        print(f"[加载] 🔍 开始扫描视频文件...", flush=True)
        
        if not os.path.exists(self.input_dir):
            print(f"[错误] ❌ 输入路径不存在: {self.input_dir}", file=sys.stderr, flush=True)
            return False

        if not os.path.isdir(self.input_dir):
            print(f"[错误] ❌ 路径不是文件夹: {self.input_dir}", file=sys.stderr, flush=True)
            return False

        try:
            # 获取所有文件
            all_items = os.listdir(self.input_dir)
            print(f"[加载] 📁 文件夹内共有 {len(all_items)} 个项目", flush=True)
        except Exception as e:
            print(f"[错误] ❌ 无法读取目录 {self.input_dir}: {str(e)}", file=sys.stderr, flush=True)
            return False

        # 筛选视频文件
        self.all_video_files = []
        for item in all_items:
            item_path = os.path.join(self.input_dir, item)
            if not os.path.isfile(item_path):
                continue
            
            _, ext = os.path.splitext(item)
            if ext.lower() in self.video_extensions:
                # 使用绝对路径，统一为正斜杠
                full_path = os.path.abspath(item_path).replace("\\", "/")
                self.all_video_files.append(full_path)
                print(f"[加载] ✅ 找到视频: {full_path}", flush=True)

        self.total_videos = len(self.all_video_files)
        if self.total_videos == 0:
            print(f"[错误] ❌ 未找到任何视频文件！支持格式: {self.video_extensions}", file=sys.stderr, flush=True)
            return False

        print(f"[加载] 📊 找到 {self.total_videos} 个视频文件，开始获取总帧数...", flush=True)
        
        # 获取每个视频的总帧数
        for video_path in self.all_video_files:
            video_name = os.path.basename(video_path)
            print(f"[加载] 🎞️ 正在获取 {video_name} 的总帧数...", flush=True)
            
            # 优先使用 ffprobe
            total_frames = self._get_video_total_frames_ffprobe(video_path)
            if total_frames == -1:
                # 备选使用 ffmpeg
                total_frames = self._get_video_total_frames_ffmpeg(video_path)

            if total_frames == -1:
                print(f"[警告] ⚠️ 无法获取 {video_name} 总帧数，将使用视频级进度", file=sys.stderr, flush=True)
                self.valid_video_files.append(video_path)
                self.video_total_frames[video_path] = -1
            else:
                self.valid_video_files.append(video_path)
                self.video_total_frames[video_path] = total_frames
                print(f"[加载] ✅ {video_name}: 总帧数 = {total_frames}", flush=True)

        self.total_videos = len(self.valid_video_files)
        if self.total_videos == 0:
            print(f"[错误] ❌ 所有视频都无法识别帧数，无法继续", file=sys.stderr, flush=True)
            return False

        print(f"[加载] ✅ 成功加载 {self.total_videos} 个可处理视频文件", flush=True)
        return True

    def _get_video_total_frames_ffprobe(self, video_path):
        """使用 ffprobe 获取视频总帧数"""
        try:
            print(f"[FFPROBE] 📊 尝试获取 {os.path.basename(video_path)} 帧数...", flush=True)
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_packets', '-show_entries', 'stream=nb_read_packets',
                '-of', 'default=nokey=1:noprint_wrappers=1', video_path
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15, encoding='utf-8'
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                frame_count = int(result.stdout.strip())
                print(f"[FFPROBE] ✅ 成功获取帧数: {frame_count}", flush=True)
                return frame_count
            else:
                print(f"[FFPROBE] ❌ 失败: {result.stderr.strip()}", flush=True)
            return -1
        except Exception as e:
            print(f"[FFPROBE ERROR] ❌ {str(e)}", file=sys.stderr, flush=True)
            return -1

    def _get_video_total_frames_ffmpeg(self, video_path):
        """使用 ffmpeg 获取视频总帧数（备选方案）"""
        try:
            print(f"[FFMPEG] 📊 尝试获取 {os.path.basename(video_path)} 帧数...", flush=True)
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', video_path, '-vcodec', 'copy', '-f', 'null', '-'
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15, encoding='utf-8'
            )
            frame_match = re.search(r'frame=\s*(\d+)', result.stderr)
            if frame_match:
                frame_count = int(frame_match.group(1))
                print(f"[FFMPEG] ✅ 成功获取帧数: {frame_count}", flush=True)
                return frame_count
            else:
                print(f"[FFMPEG] ❌ 未匹配到帧数: {result.stderr.strip()}", flush=True)
            return -1
        except Exception as e:
            print(f"[FFMPEG ERROR] ❌ {str(e)}", file=sys.stderr, flush=True)
            return -1

    def process_video(self):
        """处理单个视频文件"""
        thread_id = threading.get_ident()
        print(f"[线程] 🧵 线程 {thread_id} 启动", flush=True)

        while not self.stop_event.is_set():
            try:
                if self.task_queue.empty():
                    threading.Event().wait(0.5)
                    continue

                video_idx, video_path, total_frames = self.task_queue.get(timeout=5)
                video_name = os.path.basename(video_path)
                print(f"[处理] 🎬 线程 {thread_id} 开始处理: {video_name}", flush=True)

                # 输出文件模式
                output_pattern = os.path.join(self.output_dir, f"{video_idx}_%d.jpg")
                expected_frames = total_frames // self.frame_interval if total_frames != -1 else -1

                # FFmpeg 命令
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'warning',
                    '-i', video_path,
                    '-map', '0:v',  # 只处理视频流
                    '-vf', f"select=not(mod(n\\,{self.frame_interval}))",  # 每N帧提取一帧
                    '-fps_mode', 'vfr',  # 可变帧率
                    '-q:v', '2',  # 图片质量 (1-31, 1=最好)
                    '-progress', 'pipe:1',  # 输出进度信息
                    output_pattern
                ]

                try:
                    print(f"[FFMPEG] 🚀 执行命令: {' '.join(cmd)}", flush=True)
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1, universal_newlines=True, encoding='utf-8'
                    )

                    current_frame = 0
                    # 读取进度输出
                    for line in process.stdout:
                        line = line.strip()
                        if total_frames != -1 and line.startswith('frame='):
                            try:
                                current_frame = int(line.split('=')[1])
                                # 计算视频内进度
                                video_progress = min((current_frame / expected_frames) * 100, 100) if expected_frames > 0 else 0
                                # 计算整体进度
                                with self.lock:
                                    overall_progress = (self.processed_videos * 100 + video_progress) / self.total_videos
                                print(f"[进度] 📈 {overall_progress:.1f}% | {video_name} (帧: {current_frame}/{expected_frames})", flush=True)
                            except Exception as e:
                                print(f"[解析错误] ❌ 解析进度失败: {str(e)}", flush=True)

                    # 等待进程结束
                    return_code = process.wait(timeout=300)
                    if return_code != 0:
                        raise Exception(f"FFmpeg返回非零码: {return_code}")

                    # 更新统计
                    with self.lock:
                        self.processed_videos += 1
                    overall_progress = (self.processed_videos / self.total_videos) * 100
                    print(f"[成功] ✅ {overall_progress:.1f}% | 成功处理: {video_name}", flush=True)

                except subprocess.TimeoutExpired:
                    process.kill()
                    with self.lock:
                        self.processed_videos += 1
                        self.failed_videos += 1
                    overall_progress = (self.processed_videos / self.total_videos) * 100
                    print(f"[超时] ❌ {overall_progress:.1f}% | {video_name} 处理超时", file=sys.stderr, flush=True)

                except Exception as e:
                    with self.lock:
                        self.processed_videos += 1
                        self.failed_videos += 1
                    overall_progress = (self.processed_videos / self.total_videos) * 100
                    print(f"[错误] ❌ {overall_progress:.1f}% | {video_name} 处理失败: {str(e)}", file=sys.stderr, flush=True)

                finally:
                    self.task_queue.task_done()
                    print(f"[线程] ✅ 线程 {thread_id} 完成任务: {video_name}", flush=True)

            except Exception as e:
                if self.task_queue.empty():
                    print(f"[线程] 🧵 线程 {thread_id} 任务完成，退出", flush=True)
                    break
                print(f"[线程错误] ❌ 线程 {thread_id} 错误: {str(e)}", flush=True)

    def start(self):
        """启动视频处理"""
        print(f"=== [启动] 🚀 视频抽帧处理开始 ===", flush=True)
        
        # 检查 FFmpeg 是否可用
        try:
            print(f"[FFMPEG检查] 🔍 检查 FFmpeg 是否可用...", flush=True)
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, encoding='utf-8')
            if result.returncode == 0:
                print(f"[FFMPEG检查] ✅ FFmpeg 正常", flush=True)
            else:
                print(f"[FFMPEG检查] ❌ FFmpeg 返回错误", file=sys.stderr, flush=True)
                return
        except Exception as e:
            print(f"[FFMPEG检查] ❌ 未找到 FFmpeg: {str(e)}", file=sys.stderr, flush=True)
            return

        # 加载视频文件
        if not self.load_video_files():
            print(f"[中止] ❌ 加载视频文件失败，中止处理", flush=True)
            return

        # 填充任务队列
        for idx, video_path in enumerate(self.valid_video_files, 1):
            total_frames = self.video_total_frames[video_path]
            self.task_queue.put((idx, video_path, total_frames))
        print(f"[队列] 📥 任务队列已填充 {self.task_queue.qsize()} 个任务", flush=True)

        # 启动处理线程
        actual_threads = min(self.max_threads, self.total_videos)
        print(f"[线程] 🧵 启动 {actual_threads} 个处理线程", flush=True)

        threads = []
        for i in range(actual_threads):
            thread = threading.Thread(target=self.process_video, name=f"Worker-{i+1}")
            thread.daemon = False
            thread.start()
            threads.append(thread)
            print(f"[线程] ▶️ 启动线程: {thread.name}", flush=True)

        # 等待所有任务完成
        try:
            print(f"[等待] ⏳ 等待所有任务完成...", flush=True)
            self.task_queue.join()
            self.stop_event.set()

            # 等待线程结束
            for thread in threads:
                thread.join(timeout=10)
                if thread.is_alive():
                    print(f"[警告] ⚠️ 线程 {thread.name} 未正常退出", flush=True)

            # 输出最终结果
            success_count = self.processed_videos - self.failed_videos
            print(f"=== [完成] 🎉 处理完成 ===", flush=True)
            print(f"[结果] ✅ 成功: {success_count} | ❌ 失败: {self.failed_videos} | 📂 总计: {self.total_videos}", flush=True)

        except KeyboardInterrupt:
            self.stop_event.set()
            print("[停止] ⚠️ 已手动终止处理", flush=True)
        except Exception as e:
            print(f"[崩溃] ❌ 程序异常: {str(e)}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    @staticmethod
    def extract_single_video(video_path, output_dir, frame_interval=90):
        """
        静态方法：处理单个视频文件
        """
        print(f"=== [单视频] 🎬 开始处理单视频 ===", flush=True)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ 视频文件不存在：{video_path}")

        supported_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        if not any(video_path.lower().endswith(ext) for ext in supported_exts):
            raise ValueError(f"❌ 不支持的视频格式，支持：{', '.join(supported_exts)}")

        os.makedirs(output_dir, exist_ok=True)

        # 获取总帧数
        total_frames = VideoFrameExtractor._get_video_total_frames_ffprobe_static(video_path)
        if total_frames == -1:
            total_frames = VideoFrameExtractor._get_video_total_frames_ffmpeg_static(video_path)

        video_name = os.path.basename(video_path)
        print(f"[单视频] 📹 处理: {video_name}", flush=True)
        print(f"[单视频] 📊 总帧数: {total_frames if total_frames != -1 else '未知'}", flush=True)

        # 输出文件模式
        output_pattern = os.path.join(output_dir, "frame_%d.jpg")
        expected_frames = total_frames // frame_interval if total_frames != -1 else -1

        # FFmpeg 命令
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'info',
            '-i', video_path,
            '-vf', f"select=not(mod(n\\,{frame_interval}))",
            '-fps_mode', 'vfr',
            '-q:v', '2',
            output_pattern
        ]

        try:
            print(f"[单视频命令] 🚀 执行: {' '.join(cmd)}", flush=True)
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, universal_newlines=True, encoding='utf-8'
            )

            current_frame = 0
            # 读取进度输出
            for line in process.stderr:
                line = line.strip()
                if total_frames != -1 and 'frame=' in line:
                    match = re.search(r'frame=\s*(\d+)', line)
                    if match:
                        current_frame = int(match.group(1))
                        if expected_frames > 0:
                            progress = (current_frame / expected_frames) * 100
                            print(f"[单视频进度] 📈 {progress:.1f}% | 已提取 {current_frame} 帧", flush=True)

            # 等待进程结束
            return_code = process.wait(timeout=3600)
            if return_code != 0:
                raise Exception(f"❌ FFmpeg处理失败，返回码：{return_code}")

            # 统计实际生成的帧数
            frame_files = [f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")]
            actual_frames = len(frame_files)

            print(f"[单视频完成] ✅ 共提取 {actual_frames} 帧到 {output_dir}", flush=True)
            return actual_frames

        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("⏰ 处理超时（超过1小时）")
        except Exception as e:
            raise Exception(f"❌ 处理失败：{str(e)}")

    @staticmethod
    def _get_video_total_frames_ffprobe_static(video_path):
        """静态方法：使用 ffprobe 获取帧数"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_packets', '-show_entries', 'stream=nb_read_packets',
                '-of', 'default=nokey=1:noprint_wrappers=1', video_path
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15, encoding='utf-8'
            )
            if result.returncode == 0 and result.stdout.strip().isdigit():
                return int(result.stdout.strip())
            return -1
        except:
            return -1

    @staticmethod
    def _get_video_total_frames_ffmpeg_static(video_path):
        """静态方法：使用 ffmpeg 获取帧数"""
        try:
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', video_path, '-vcodec', 'copy', '-f', 'null', '-'
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15, encoding='utf-8'
            )
            frame_match = re.search(r'frame=(\d+)', result.stderr)
            if frame_match:
                return int(frame_match.group(1))
            return -1
        except:
            return -1