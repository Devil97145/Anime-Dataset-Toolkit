import cv2
import numpy as np
import imagehash
from PIL import Image
import subprocess
import json
import os
import tempfile
import time
import shutil
from typing import List, Tuple, Dict, Optional, Union
import logging
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileLockManager:
    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()
    
    def get_lock(self, file_path):
        with self.lock:
            if file_path not in self.locks:
                self.locks[file_path] = threading.Lock()
            return self.locks[file_path]
    
    @contextmanager
    def file_lock(self, file_path):
        lock = self.get_lock(file_path)
        try:
            lock.acquire()
            yield
        finally:
            lock.release()

class OPEDDetector:
    def __init__(self, frame_interval: int = 15, max_hamming: int = 10, min_consecutive: int = 5,
                 use_gpu: bool = False, similarity_threshold: float = 0.85, max_workers: int = None):
        """
        初始化OP/ED检测器
        
        Args:
            max_workers: 最大工作线程数，None表示使用CPU核心数
        """
        self.frame_interval = frame_interval
        self.max_hamming = max_hamming
        self.min_consecutive = min_consecutive
        self.use_gpu = use_gpu
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers or max(1, os.cpu_count() - 1)
        
        # 文件锁管理
        self.file_lock_manager = FileLockManager()
        
        # 检查GPU可用性
        if use_gpu and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("GPU acceleration enabled")
        else:
            self.use_gpu = False
            logger.info("GPU acceleration disabled, using CPU")
        
        logger.info(f"初始化检测器，最大工作线程数: {self.max_workers}")
    
    def _safe_copy_file(self, source_path: str, dest_path: str, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
        """安全复制文件，处理Windows权限问题"""
        with self.file_lock_manager.file_lock(source_path):
            for attempt in range(max_retries):
                try:
                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # 尝试复制文件
                    if os.path.exists(dest_path):
                        try:
                            os.remove(dest_path)
                        except PermissionError:
                            time.sleep(0.2)
                    
                    shutil.copy2(source_path, dest_path)
                    
                    # 验证文件是否成功复制
                    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                        logger.info(f"文件复制成功: {source_path} -> {dest_path} (尝试 {attempt + 1})")
                        return True
                    
                except PermissionError as e:
                    logger.warning(f"权限错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # 指数退避
                        continue
                    else:
                        logger.error(f"文件复制失败，达到最大重试次数: {source_path}")
                        return False
                except Exception as e:
                    logger.error(f"文件复制错误: {e}")
                    return False
            
            return False
    
    def _get_video_info(self, video_path: str) -> Dict:
        """获取视频基本信息"""
        for attempt in range(3):
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {video_path}")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                cap.release()
                
                return {
                    'fps': fps,
                    'frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'duration': duration,
                    'file_size': os.path.getsize(video_path) / 1024 / 1024  # MB
                }
            except Exception as e:
                logger.warning(f"获取视频信息失败 (尝试 {attempt + 1}): {e}")
                time.sleep(0.5)
        
        raise ValueError(f"无法获取视频信息: {video_path}")
    
    def _extract_frames(self, video_path: str, frame_interval: int = None) -> List[Tuple[int, float, np.ndarray]]:
        """提取视频帧"""
        frame_interval = frame_interval or self.frame_interval
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                # 调整帧大小以提高处理速度
                target_height = 360
                if frame.shape[0] > target_height:
                    scale = target_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    frame = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)
                frames.append((frame_count, timestamp, frame))
            
            frame_count += 1
        
        cap.release()
        logger.info(f"从 {video_path} 提取了 {len(frames)} 帧")
        return frames
    
    def _calculate_frame_hash(self, frame: np.ndarray) -> imagehash.ImageHash:
        """计算帧的感知哈希"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            return imagehash.phash(pil_image, hash_size=8)
        except Exception as e:
            logger.error(f"计算帧哈希失败: {e}")
            return imagehash.phash(Image.new('RGB', (8, 8)))
    
    def _compare_segments(self, main_frames: List[Tuple[int, float, np.ndarray]], 
                         ref_frames: List[Tuple[int, float, np.ndarray]]) -> List[Dict]:
        """比较主视频和参考视频的片段"""
        if not main_frames or not ref_frames:
            return []
        
        # 预计算参考帧的哈希
        ref_hashes = [self._calculate_frame_hash(frame) for _, _, frame in ref_frames]
        ref_duration = ref_frames[-1][1] - ref_frames[0][1]
        
        matches = []
        consecutive_count = 0
        current_match = None
        
        for i, (main_frame_num, main_timestamp, main_frame) in enumerate(main_frames):
            main_hash = self._calculate_frame_hash(main_frame)
            
            # 与参考帧比较
            best_similarity = 0
            for ref_hash in ref_hashes:
                try:
                    hamming_distance = main_hash - ref_hash
                    similarity = 1 - (hamming_distance / 64)  # 8x8 hash = 64 bits
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                except Exception as e:
                    logger.warning(f"哈希比较错误: {e}")
                    continue
            
            # 检查是否匹配
            if best_similarity >= self.similarity_threshold:
                if consecutive_count == 0:
                    # 新匹配开始
                    current_match = {
                        'start_frame': main_frame_num,
                        'start_time': main_timestamp,
                        'similarity_sum': best_similarity,
                        'frame_count': 1
                    }
                
                consecutive_count += 1
                current_match['similarity_sum'] += best_similarity
                current_match['frame_count'] += 1
                
                # 检查是否达到最小连续帧数
                if consecutive_count >= self.min_consecutive:
                    # 计算平均相似度
                    avg_similarity = current_match['similarity_sum'] / current_match['frame_count']
                    
                    # 保存匹配
                    match_end_frame = main_frame_num
                    match_end_time = main_timestamp
                    
                    matches.append({
                        'start_frame': current_match['start_frame'],
                        'end_frame': match_end_frame,
                        'start_time': current_match['start_time'],
                        'end_time': match_end_time,
                        'similarity': avg_similarity,
                        'duration': match_end_time - current_match['start_time']
                    })
                    
                    # 重置，避免重叠匹配
                    consecutive_count = 0
                    current_match = None
            
            else:
                # 重置连续计数
                consecutive_count = 0
                current_match = None
        
        # 合并重叠或接近的匹配
        merged_matches = self._merge_overlapping_matches(matches, ref_duration)
        
        return merged_matches
    
    def _merge_overlapping_matches(self, matches: List[Dict], ref_duration: float) -> List[Dict]:
        """合并重叠或接近的匹配片段"""
        if not matches:
            return []
        
        # 按开始时间排序
        matches.sort(key=lambda x: x['start_time'])
        merged = []
        
        for match in matches:
            if not merged:
                merged.append(match)
                continue
            
            last_match = merged[-1]
            gap = match['start_time'] - last_match['end_time']
            
            # 如果重叠或间隔很小，合并
            if gap < ref_duration * 0.5:  # 间隔小于参考片段时长的一半
                merged[-1]['end_time'] = max(last_match['end_time'], match['end_time'])
                merged[-1]['end_frame'] = max(last_match['end_frame'], match['end_frame'])
                merged[-1]['similarity'] = max(last_match['similarity'], match['similarity'])
                merged[-1]['duration'] = merged[-1]['end_time'] - merged[-1]['start_time']
            else:
                merged.append(match)
        
        return merged
    
    def detect_op_ed(self, main_video_path: str, op_ref_path: str = None, ed_ref_path: str = None,
                    output_json: str = None) -> Dict[str, Union[List[Dict], str]]:
        """
        检测OP和ED
        
        Returns:
            包含检测结果和文件信息的字典
        """
        logger.info(f"开始检测OP/ED - 主视频: {main_video_path}")
        
        try:
            # 获取视频信息
            main_info = self._get_video_info(main_video_path)
            logger.info(f"主视频信息: {main_info}")
            
            # 提取主视频帧
            main_frames = self._extract_frames(main_video_path)
            
            results = {
                'op_segments': [],
                'ed_segments': [],
                'video_info': main_info,
                'filename': os.path.basename(main_video_path),
                'success': True
            }
            
            # 检测OP
            if op_ref_path and os.path.exists(op_ref_path):
                logger.info(f"检测OP - 参考视频: {op_ref_path}")
                try:
                    op_frames = self._extract_frames(op_ref_path)
                    results['op_segments'] = self._compare_segments(main_frames, op_frames)
                    logger.info(f"找到 {len(results['op_segments'])} 个OP片段")
                except Exception as e:
                    logger.error(f"OP检测失败: {e}")
                    results['op_error'] = str(e)
            
            # 检测ED
            if ed_ref_path and os.path.exists(ed_ref_path):
                logger.info(f"检测ED - 参考视频: {ed_ref_path}")
                try:
                    ed_frames = self._extract_frames(ed_ref_path)
                    results['ed_segments'] = self._compare_segments(main_frames, ed_frames)
                    logger.info(f"找到 {len(results['ed_segments'])} 个ED片段")
                except Exception as e:
                    logger.error(f"ED检测失败: {e}")
                    results['ed_error'] = str(e)
            
            # 保存结果到JSON
            if output_json:
                os.makedirs(os.path.dirname(output_json), exist_ok=True)
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"结果已保存到: {output_json}")
            
            return results
            
        except Exception as e:
            logger.error(f"检测过程中出错: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'filename': os.path.basename(main_video_path)
            }
    
    def batch_detect_op_ed(self, main_video_paths: List[str], op_ref_path: str = None, ed_ref_path: str = None,
                          output_dir: str = None, progress_callback=None) -> Dict[str, Dict]:
        """
        批量检测OP/ED
        
        Args:
            main_video_paths: 主视频文件路径列表
            op_ref_path: OP参考视频路径
            ed_ref_path: ED参考视频路径
            output_dir: 结果输出目录
            progress_callback: 进度回调函数 (current, total, filename)
        
        Returns:
            {filename: results_dict, ...}
        """
        logger.info(f"开始批量检测OP/ED - 共 {len(main_video_paths)} 个视频")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results_dict = {}
        total_files = len(main_video_paths)
        
        # 预加载参考视频帧（如果存在）
        op_frames = None
        ed_frames = None
        
        if op_ref_path and os.path.exists(op_ref_path):
            try:
                op_frames = self._extract_frames(op_ref_path)
                logger.info(f"预加载OP参考帧: {len(op_frames)} 帧")
            except Exception as e:
                logger.error(f"预加载OP参考帧失败: {e}")
        
        if ed_ref_path and os.path.exists(ed_ref_path):
            try:
                ed_frames = self._extract_frames(ed_ref_path)
                logger.info(f"预加载ED参考帧: {len(ed_frames)} 帧")
            except Exception as e:
                logger.error(f"预加载ED参考帧失败: {e}")
        
        def process_single_video(video_path, index):
            """处理单个视频"""
            try:
                if progress_callback:
                    progress_callback(index + 1, total_files, os.path.basename(video_path))
                
                # 获取视频信息
                main_info = self._get_video_info(video_path)
                main_frames = self._extract_frames(video_path)
                
                results = {
                    'op_segments': [],
                    'ed_segments': [],
                    'video_info': main_info,
                    'filename': os.path.basename(video_path),
                    'success': True
                }
                
                # 检测OP
                if op_frames:
                    try:
                        results['op_segments'] = self._compare_segments(main_frames, op_frames)
                    except Exception as e:
                        logger.error(f"OP检测失败 ({os.path.basename(video_path)}): {e}")
                        results['op_error'] = str(e)
                
                # 检测ED
                if ed_frames:
                    try:
                        results['ed_segments'] = self._compare_segments(main_frames, ed_frames)
                    except Exception as e:
                        logger.error(f"ED检测失败 ({os.path.basename(video_path)}): {e}")
                        results['ed_error'] = str(e)
                
                # 保存单个结果
                if output_dir:
                    json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_results.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                
                return video_path, results
                
            except Exception as e:
                logger.error(f"处理视频 {video_path} 时出错: {e}", exc_info=True)
                return video_path, {
                    'success': False,
                    'error': str(e),
                    'filename': os.path.basename(video_path)
                }
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(process_single_video, video_path, idx): video_path 
                for idx, video_path in enumerate(main_video_paths)
            }
            
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_path, result = future.result()
                    results_dict[os.path.basename(video_path)] = result
                except Exception as e:
                    logger.error(f"线程处理失败 {video_path}: {e}")
                    results_dict[os.path.basename(video_path)] = {
                        'success': False,
                        'error': str(e),
                        'filename': os.path.basename(video_path)
                    }
        
        # 保存批量结果汇总
        if output_dir:
            summary_path = os.path.join(output_dir, 'batch_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_files': total_files,
                    'successful_files': sum(1 for r in results_dict.values() if r.get('success', False)),
                    'failed_files': sum(1 for r in results_dict.values() if not r.get('success', False)),
                    'results': results_dict,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"批量检测完成，结果汇总保存到: {summary_path}")
        
        logger.info(f"批量检测完成 - 成功: {sum(1 for r in results_dict.values() if r.get('success', False))}/{total_files}")
        return results_dict
    
    def _safe_ffmpeg_command(self, cmd: list, max_retries: int = 2) -> bool:
        """安全执行ffmpeg命令，处理可能的权限问题"""
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"执行ffmpeg命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"ffmpeg命令执行成功")
                    return True
                
                logger.warning(f"ffmpeg命令失败 (尝试 {attempt + 1}/{max_retries + 1}): {result.stderr}")
                
                if attempt < max_retries:
                    # 等待一段时间后重试
                    time.sleep(1.0)
                    continue
                
            except subprocess.TimeoutExpired:
                logger.error(f"ffmpeg命令超时")
            except Exception as e:
                logger.error(f"执行ffmpeg命令时出错: {e}")
        
        return False
    
    def remove_op_ed_segments(self, input_video: str, op_segments: List[Dict], ed_segments: List[Dict], 
                            output_video: str, dry_run: bool = False) -> Dict[str, Union[bool, str, float]]:
        """
        删除OP/ED片段并生成新视频
        
        Returns:
            包含操作结果和统计信息的字典
        """
        logger.info(f"开始删除OP/ED片段 - 输入: {input_video}, 输出: {output_video}")
        
        start_time = time.time()
        result_info = {
            'success': False,
            'output_path': output_video,
            'processing_time': 0,
            'original_size': 0,
            'output_size': 0,
            'size_ratio': 0
        }
        
        try:
            # 获取原始文件大小
            if os.path.exists(input_video):
                result_info['original_size'] = os.path.getsize(input_video) / 1024 / 1024  # MB
            
            # 合并所有要删除的时间段
            remove_segments = []
            remove_segments.extend(op_segments)
            remove_segments.extend(ed_segments)
            
            if not remove_segments:
                logger.warning("没有要删除的片段，跳过处理")
                if not dry_run:
                    try:
                        shutil.copy2(input_video, output_video)
                        result_info['success'] = True
                    except Exception as e:
                        logger.error(f"复制文件失败: {e}")
                else:
                    result_info['success'] = True
                
                return result_info
            
            # 按开始时间排序
            remove_segments.sort(key=lambda x: x['start_time'])
            
            # 获取视频信息
            video_info = self._get_video_info(input_video)
            total_duration = video_info['duration']
            
            # 计算要保留的时间段
            keep_segments = []
            current_time = 0.0
            
            for segment in remove_segments:
                start_time_seg = segment['start_time']
                end_time_seg = segment['end_time']
                
                # 确保时间段有效
                start_time_seg = max(0.0, min(start_time_seg, total_duration))
                end_time_seg = max(0.0, min(end_time_seg, total_duration))
                
                if start_time_seg > current_time:
                    keep_segments.append((current_time, start_time_seg))
                
                current_time = max(current_time, end_time_seg)
            
            if current_time < total_duration:
                keep_segments.append((current_time, total_duration))
            
            logger.info(f"保留时间段: {keep_segments}")
            
            if not keep_segments:
                logger.error("没有要保留的片段，操作失败")
                return result_info
            
            if dry_run:
                logger.info("Dry run mode - 不执行实际剪辑")
                result_info['success'] = True
                return result_info
            
            try:
                # 创建输出目录
                os.makedirs(os.path.dirname(output_video), exist_ok=True)
                
                # 使用ffmpeg进行剪辑
                if len(keep_segments) == 1:
                    # 只需要一个时间段，直接裁剪
                    start_time_seg, end_time_seg = keep_segments[0]
                    cmd = [
                        'ffmpeg',
                        '-i', input_video,
                        '-ss', str(start_time_seg),
                        '-to', str(end_time_seg),
                        '-c', 'copy',
                        '-y',
                        output_video
                    ]
                    
                    success = self._safe_ffmpeg_command(cmd)
                    if not success:
                        # 尝试使用不同的方法
                        cmd = [
                            'ffmpeg',
                            '-i', input_video,
                            '-ss', str(start_time_seg),
                            '-t', str(end_time_seg - start_time_seg),
                            '-c', 'copy',
                            '-y',
                            output_video
                        ]
                        success = self._safe_ffmpeg_command(cmd)
                    
                    result_info['success'] = success
                
                else:
                    # 需要多个时间段，使用concat
                    temp_dir = tempfile.mkdtemp(prefix='oped_ffmpeg_')
                    segment_files = []
                    
                    for i, (start_time_seg, end_time_seg) in enumerate(keep_segments):
                        segment_file = os.path.join(temp_dir, f'segment_{i:03d}.mp4')
                        cmd = [
                            'ffmpeg',
                            '-i', input_video,
                            '-ss', str(start_time_seg),
                            '-to', str(end_time_seg),
                            '-c', 'copy',
                            '-y',
                            segment_file
                        ]
                        
                        if not self._safe_ffmpeg_command(cmd):
                            logger.warning(f"片段 {i} 生成失败，跳过")
                            continue
                        
                        if os.path.exists(segment_file) and os.path.getsize(segment_file) > 1024:  # 大于1KB
                            segment_files.append(segment_file)
                        else:
                            logger.warning(f"片段文件无效或为空: {segment_file}")
                    
                    if not segment_files:
                        logger.error("没有生成有效的片段文件")
                        return result_info
                    
                    # 创建文件列表
                    list_file = os.path.join(temp_dir, 'filelist.txt')
                    with open(list_file, 'w', encoding='utf-8') as f:
                        for segment_file in segment_files:
                            # 使用相对路径
                            rel_path = os.path.relpath(segment_file, temp_dir).replace('\\', '/')
                            f.write(f"file '{rel_path}'\n")
                    
                    # 合并片段
                    cmd = [
                        'ffmpeg',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', list_file,
                        '-c', 'copy',
                        '-y',
                        output_video
                    ]
                    
                    success = self._safe_ffmpeg_command(cmd)
                    result_info['success'] = success
                    
                    # 清理临时文件
                    try:
                        for file in segment_files + [list_file]:
                            if os.path.exists(file):
                                os.remove(file)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
            
            except Exception as e:
                logger.error(f"ffmpeg处理过程中出错: {e}", exc_info=True)
                result_info['success'] = False
            
            # 获取输出文件信息
            if result_info['success'] and os.path.exists(output_video):
                result_info['output_size'] = os.path.getsize(output_video) / 1024 / 1024  # MB
                if result_info['original_size'] > 0:
                    result_info['size_ratio'] = result_info['output_size'] / result_info['original_size']
            
            result_info['processing_time'] = time.time() - start_time
            return result_info
        
        except Exception as e:
            logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
            result_info['processing_time'] = time.time() - start_time
            return result_info
    
    def batch_remove_op_ed(self, video_results: Dict[str, Dict], input_dir: str, output_dir: str,
                          progress_callback=None) -> Dict[str, Dict]:
        """
        批量删除OP/ED
        
        Args:
            video_results: batch_detect_op_ed返回的结果
            input_dir: 输入视频目录
            output_dir: 输出目录
            progress_callback: 进度回调函数 (current, total, filename)
        
        Returns:
            {filename: processing_result, ...}
        """
        logger.info(f"开始批量删除OP/ED - 共 {len(video_results)} 个视频")
        os.makedirs(output_dir, exist_ok=True)
        
        processing_results = {}
        total_files = len(video_results)
        current_count = 0
        
        for filename, results in video_results.items():
            current_count += 1
            if progress_callback:
                progress_callback(current_count, total_files, filename)
            
            if not results.get('success', False):
                logger.warning(f"跳过失败的视频: {filename}")
                processing_results[filename] = {
                    'success': False,
                    'error': '检测失败，跳过处理',
                    'filename': filename
                }
                continue
            
            input_video_path = os.path.join(input_dir, filename)
            output_video_path = os.path.join(output_dir, f"no_op_ed_{filename}")
            
            if not os.path.exists(input_video_path):
                logger.error(f"输入视频不存在: {input_video_path}")
                processing_results[filename] = {
                    'success': False,
                    'error': f'输入视频不存在: {input_video_path}',
                    'filename': filename
                }
                continue
            
            # 删除OP/ED
            processing_result = self.remove_op_ed_segments(
                input_video=input_video_path,
                op_segments=results.get('op_segments', []),
                ed_segments=results.get('ed_segments', []),
                output_video=output_video_path
            )
            
            processing_result['filename'] = filename
            processing_results[filename] = processing_result
            
            if processing_result['success']:
                logger.info(f"成功处理 {filename} -> {output_video_path}")
            else:
                logger.error(f"处理 {filename} 失败")
        
        # 保存批量处理结果
        summary_path = os.path.join(output_dir, 'batch_processing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': total_files,
                'successful_files': sum(1 for r in processing_results.values() if r.get('success', False)),
                'failed_files': sum(1 for r in processing_results.values() if not r.get('success', False)),
                'results': processing_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"批量处理完成，结果汇总保存到: {summary_path}")
        return processing_results