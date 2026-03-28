import gradio as gr
import os
import json
import tempfile
import time
from typing import List, Dict, Tuple, Optional, Union
import logging
from functions.op_ed_detector import OPEDDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OPEDUI:
    def __init__(self):
        self.detector = OPEDDetector(
            frame_interval=15,
            max_hamming=10,
            min_consecutive=5,
            use_gpu=False,
            similarity_threshold=0.85,
            max_workers=4  # 限制线程数，避免资源耗尽
        )
        # 使用更安全的临时目录
        self.temp_dir = tempfile.mkdtemp(prefix='oped_ui_')
        logger.info(f"创建临时目录: {self.temp_dir}")
        self.batch_temp_dir = tempfile.mkdtemp(prefix='oped_batch_')
        logger.info(f"创建批量临时目录: {self.batch_temp_dir}")
    
    def __del__(self):
        """清理临时文件"""
        try:
            for temp_dir in [self.temp_dir, self.batch_temp_dir]:
                if os.path.exists(temp_dir):
                    # 尝试清理临时文件
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                if os.path.exists(file_path):
                                    os.chmod(file_path, 0o666)
                                    os.remove(file_path)
                            except Exception as e:
                                logger.warning(f"清理文件失败 {file}: {e}")
                    
                    # 尝试删除目录
                    try:
                        os.rmdir(temp_dir)
                        logger.info(f"清理临时目录: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"删除临时目录失败: {e}")
        except Exception as e:
            logger.warning(f"清理临时目录时出错: {e}")
    
    def _format_segment_info(self, segments: List[Dict], segment_type: str) -> str:
        """格式化片段信息为可读文本"""
        if not segments:
            return f"未检测到{segment_type}"
        
        result = f"检测到 {len(segments)} 个{segment_type}片段:\n"
        for i, seg in enumerate(segments, 1):
            duration = seg['end_time'] - seg['start_time']
            result += f"片段 {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s (时长: {duration:.2f}s, 相似度: {seg['similarity']:.3f})\n"
        return result
    
    def _format_batch_results(self, batch_results: Dict[str, Dict]) -> str:
        """格式化批量检测结果"""
        if not batch_results:
            return "没有检测结果"
        
        summary = f"## 📊 批量检测结果汇总\n\n"
        summary += f"**总文件数:** {len(batch_results)}\n"
        summary += f"**成功检测:** {sum(1 for r in batch_results.values() if r.get('success', False))}\n"
        summary += f"**检测失败:** {sum(1 for r in batch_results.values() if not r.get('success', False))}\n\n"
        
        summary += "## 📋 详细结果\n\n"
        
        for filename, results in batch_results.items():
            summary += f"### 🎬 {filename}\n"
            
            if not results.get('success', False):
                summary += f"❌ **检测失败**: {results.get('error', '未知错误')}\n\n"
                continue
            
            # 视频信息
            video_info = results.get('video_info', {})
            if video_info:
                summary += f"**视频信息:**\n"
                summary += f"- 时长: {video_info.get('duration', 0):.2f} 秒\n"
                summary += f"- 分辨率: {video_info.get('width', 0)}x{video_info.get('height', 0)}\n"
                summary += f"- 文件大小: {video_info.get('file_size', 0):.2f} MB\n\n"
            
            # OP结果
            op_segments = results.get('op_segments', [])
            summary += f"**OP检测结果:** {len(op_segments)} 个片段\n"
            for i, seg in enumerate(op_segments, 1):
                duration = seg['end_time'] - seg['start_time']
                summary += f"- 片段 {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s (时长: {duration:.2f}s, 相似度: {seg['similarity']:.3f})\n"
            
            if not op_segments:
                summary += "- 未检测到OP片段\n"
            
            # ED结果
            ed_segments = results.get('ed_segments', [])
            summary += f"\n**ED检测结果:** {len(ed_segments)} 个片段\n"
            for i, seg in enumerate(ed_segments, 1):
                duration = seg['end_time'] - seg['start_time']
                summary += f"- 片段 {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s (时长: {duration:.2f}s, 相似度: {seg['similarity']:.3f})\n"
            
            if not ed_segments:
                summary += "- 未检测到ED片段\n"
            
            summary += "\n---\n\n"
        
        return summary
    
    def _get_file_path(self, file) -> Optional[str]:
        """安全获取文件路径，处理Gradio文件对象"""
        if not file:
            return None
        
        try:
            if hasattr(file, 'name') and os.path.exists(file.name):
                return file.name
            elif isinstance(file, str) and os.path.exists(file):
                return file
            else:
                logger.error(f"无效的文件输入: {file}")
                return None
        except Exception as e:
            logger.error(f"获取文件路径失败: {e}")
            return None
    
    def _copy_to_safe_location(self, source_path: str, prefix: str = '') -> Optional[str]:
        """安全复制文件到临时目录，处理权限问题"""
        if not source_path or not os.path.exists(source_path):
            return None
        
        try:
            filename = os.path.basename(source_path)
            if prefix:
                filename = f"{prefix}_{filename}"
            
            dest_path = os.path.join(self.temp_dir, filename)
            
            # 使用安全的复制方法
            success = self.detector._safe_copy_file(source_path, dest_path)
            
            if success and os.path.exists(dest_path):
                logger.info(f"安全复制文件: {source_path} -> {dest_path}")
                return dest_path
            else:
                logger.error(f"文件复制失败: {source_path} -> {dest_path}")
                return None
                
        except Exception as e:
            logger.error(f"文件复制过程中出错: {e}", exc_info=True)
            return None
    
    def _get_video_files_from_directory(self, directory_path: str) -> List[str]:
        """获取目录中的视频文件"""
        if not directory_path or not os.path.exists(directory_path):
            return []
        
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
        video_files = []
        
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(directory_path, file))
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        return video_files
    
    def batch_detect_progress(self, current: int, total: int, filename: str) -> str:
        """批量检测进度回调"""
        progress = (current / total) * 100
        return f"🔍 检测进度: {current}/{total} ({progress:.1f}%) - 当前文件: {filename}"
    
    def batch_process_progress(self, current: int, total: int, filename: str) -> str:
        """批量处理进度回调"""
        progress = (current / total) * 100
        return f"✂️ 处理进度: {current}/{total} ({progress:.1f}%) - 当前文件: {filename}"
    
    def batch_detect_op_ed(self, video_directory: str, op_ref_video, ed_ref_video,
                          similarity_threshold: float, min_duration: float) -> Tuple[str, str, str]:
        """
        批量检测OP/ED
        
        Returns:
            (results_markdown, json_path, status)
        """
        if not video_directory or not os.path.exists(video_directory):
            return "错误: 无效的视频目录", "", "错误: 无效的视频目录"
        
        try:
            # 更新检测器参数
            self.detector.similarity_threshold = similarity_threshold
            
            # 获取视频文件列表
            video_files = self._get_video_files_from_directory(video_directory)
            if not video_files:
                return "错误: 目录中没有找到视频文件", "", "错误: 目录中没有找到视频文件"
            
            # 获取参考文件路径
            op_ref_path = self._get_file_path(op_ref_video) if op_ref_video else None
            ed_ref_path = self._get_file_path(ed_ref_video) if ed_ref_video else None
            
            # 安全复制参考文件
            safe_op_ref_path = self._copy_to_safe_location(op_ref_path, 'batch_op_ref') if op_ref_path else None
            safe_ed_ref_path = self._copy_to_safe_location(ed_ref_path, 'batch_ed_ref') if ed_ref_path else None
            
            # 创建输出目录
            batch_output_dir = os.path.join(self.batch_temp_dir, 'batch_detection_results')
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # 批量检测
            batch_results = self.detector.batch_detect_op_ed(
                main_video_paths=video_files,
                op_ref_path=safe_op_ref_path,
                ed_ref_path=safe_ed_ref_path,
                output_dir=batch_output_dir,
                progress_callback=self.batch_detect_progress
            )
            
            # 格式化结果
            results_markdown = self._format_batch_results(batch_results)
            
            # 保存JSON结果
            json_path = os.path.join(batch_output_dir, 'batch_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            # 生成状态信息
            status = f"批量检测完成!\n"
            status += f"总文件数: {len(video_files)}\n"
            status += f"成功检测: {sum(1 for r in batch_results.values() if r.get('success', False))}\n"
            status += f"检测失败: {sum(1 for r in batch_results.values() if not r.get('success', False))}"
            
            return results_markdown, json_path, status
            
        except Exception as e:
            logger.error(f"批量检测过程中出错: {str(e)}", exc_info=True)
            return f"批量检测错误: {str(e)}", "", f"错误: {str(e)}"
    
    def batch_remove_op_ed(self, video_directory: str, detection_results_json: str, 
                          output_directory: str) -> Tuple[str, str]:
        """
        批量删除OP/ED
        
        Returns:
            (results_markdown, status)
        """
        if not video_directory or not os.path.exists(video_directory):
            return "错误: 无效的视频目录", "错误: 无效的视频目录"
        
        if not detection_results_json or not os.path.exists(detection_results_json):
            return "错误: 无效的检测结果文件", "错误: 无效的检测结果文件"
        
        if not output_directory:
            output_directory = os.path.join(video_directory, 'no_op_ed_output')
        
        try:
            # 加载检测结果
            # 处理Gradio文件对象
            if hasattr(detection_results_json, 'name'):
                detection_results_json = detection_results_json.name
            
            with open(detection_results_json, 'r', encoding='utf-8') as f:
                detection_results = json.load(f)
            
            # 确保输出目录存在
            os.makedirs(output_directory, exist_ok=True)
            
            # 批量处理
            processing_results = self.detector.batch_remove_op_ed(
                video_results=detection_results,
                input_dir=video_directory,
                output_dir=output_directory,
                progress_callback=self.batch_process_progress
            )
            
            # 格式化处理结果
            results_markdown = self._format_batch_processing_results(processing_results)
            
            # 生成状态信息
            status = f"批量处理完成!\n"
            status += f"总文件数: {len(processing_results)}\n"
            status += f"成功处理: {sum(1 for r in processing_results.values() if r.get('success', False))}\n"
            status += f"处理失败: {sum(1 for r in processing_results.values() if not r.get('success', False))}\n"
            status += f"输出目录: {output_directory}"
            
            return results_markdown, status
            
        except Exception as e:
            logger.error(f"批量处理过程中出错: {str(e)}", exc_info=True)
            return f"批量处理错误: {str(e)}", f"错误: {str(e)}"
    
    def _format_batch_processing_results(self, processing_results: Dict[str, Dict]) -> str:
        """格式化批量处理结果"""
        if not processing_results:
            return "没有处理结果"
        
        summary = f"## 📊 批量处理结果汇总\n\n"
        summary += f"**总文件数:** {len(processing_results)}\n"
        summary += f"**成功处理:** {sum(1 for r in processing_results.values() if r.get('success', False))}\n"
        summary += f"**处理失败:** {sum(1 for r in processing_results.values() if not r.get('success', False))}\n\n"
        
        summary += "## 📋 详细结果\n\n"
        
        for filename, results in processing_results.items():
            summary += f"### 🎬 {filename}\n"
            
            if not results.get('success', False):
                summary += f"❌ **处理失败**: {results.get('error', '未知错误')}\n\n"
                continue
            
            summary += f"✅ **处理成功**\n"
            summary += f"- 处理时间: {results.get('processing_time', 0):.2f} 秒\n"
            summary += f"- 原始大小: {results.get('original_size', 0):.2f} MB\n"
            summary += f"- 输出大小: {results.get('output_size', 0):.2f} MB\n"
            summary += f"- 压缩比例: {results.get('size_ratio', 0):.1%}\n"
            summary += f"- 输出文件: {results.get('output_path', '')}\n\n"
        
        return summary
    
    def detect_op_ed(self, main_video, op_ref_video, ed_ref_video, 
                    similarity_threshold: float, min_duration: float) -> Tuple[str, str, str, str, str]:
        """
        检测OP/ED
        """
        if not main_video:
            return "请上传主视频文件", "", "", "", "错误: 未上传主视频"
        
        try:
            # 更新检测器参数
            self.detector.similarity_threshold = similarity_threshold
            
            # 获取文件路径
            main_path = self._get_file_path(main_video)
            op_ref_path = self._get_file_path(op_ref_video) if op_ref_video else None
            ed_ref_path = self._get_file_path(ed_ref_video) if ed_ref_video else None
            
            if not main_path or not os.path.exists(main_path):
                return "错误: 无法获取主视频文件路径", "", "", "", "错误: 无法获取主视频文件路径"
            
            logger.info(f"开始检测 - 主视频: {main_path}, OP参考: {op_ref_path}, ED参考: {ed_ref_path}")
            
            # 安全复制文件到临时位置
            safe_main_path = self._copy_to_safe_location(main_path, 'main')
            safe_op_ref_path = self._copy_to_safe_location(op_ref_path, 'op_ref') if op_ref_path else None
            safe_ed_ref_path = self._copy_to_safe_location(ed_ref_path, 'ed_ref') if ed_ref_path else None
            
            if not safe_main_path:
                return "错误: 无法复制主视频文件到临时位置", "", "", "", "错误: 无法复制主视频文件"
            
            # 检测OP/ED
            start_time = time.time()
            results = self.detector.detect_op_ed(
                main_video_path=safe_main_path,
                op_ref_path=safe_op_ref_path,
                ed_ref_path=safe_ed_ref_path
            )
            detection_time = time.time() - start_time
            
            # 过滤掉时长太短的片段
            if min_duration > 0:
                results['op_segments'] = [seg for seg in results['op_segments'] 
                                        if seg['end_time'] - seg['start_time'] >= min_duration]
                results['ed_segments'] = [seg for seg in results['ed_segments'] 
                                        if seg['end_time'] - seg['start_time'] >= min_duration]
            
            # 格式化结果
            op_info = self._format_segment_info(results['op_segments'], 'OP')
            ed_info = self._format_segment_info(results['ed_segments'], 'ED')
            
            # 保存JSON结果
            json_path = os.path.join(self.temp_dir, 'detection_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 生成状态信息
            status = f"检测完成! 耗时: {detection_time:.2f}秒\n"
            status += f"OP片段: {len(results['op_segments'])} 个, ED片段: {len(results['ed_segments'])} 个"
            
            return op_info, ed_info, json_path, "", status
            
        except Exception as e:
            logger.error(f"检测过程中出错: {str(e)}", exc_info=True)
            return f"检测错误: {str(e)}", "", "", "", f"错误: {str(e)}"
    
    def remove_op_ed(self, main_video, op_ref_video, ed_ref_video, 
                    similarity_threshold: float, min_duration: float,
                    custom_op_times: str, custom_ed_times: str,
                    output_filename: str) -> Tuple[str, str, str]:
        """
        删除OP/ED并生成最终视频
        """
        if not main_video:
            return "", "错误: 未上传主视频", ""
        
        try:
            # 获取文件路径
            main_path = self._get_file_path(main_video)
            op_ref_path = self._get_file_path(op_ref_video) if op_ref_video else None
            ed_ref_path = self._get_file_path(ed_ref_video) if ed_ref_video else None
            
            if not main_path or not os.path.exists(main_path):
                return "", "错误: 无法获取主视频文件路径", ""
            
            # 安全复制文件
            safe_main_path = self._copy_to_safe_location(main_path, 'main')
            safe_op_ref_path = self._copy_to_safe_location(op_ref_path, 'op_ref') if op_ref_path else None
            safe_ed_ref_path = self._copy_to_safe_location(ed_ref_path, 'ed_ref') if ed_ref_path else None
            
            if not safe_main_path:
                return "", "错误: 无法复制主视频文件到临时位置", ""
            
            # 先进行检测
            results = self.detector.detect_op_ed(
                main_video_path=safe_main_path,
                op_ref_path=safe_op_ref_path,
                ed_ref_path=safe_ed_ref_path
            )
            
            # 应用最小时长过滤
            if min_duration > 0:
                results['op_segments'] = [seg for seg in results['op_segments'] 
                                        if seg['end_time'] - seg['start_time'] >= min_duration]
                results['ed_segments'] = [seg for seg in results['ed_segments'] 
                                        if seg['end_time'] - seg['start_time'] >= min_duration]
            
            # 应用自定义时间段
            if custom_op_times and isinstance(custom_op_times, str) and custom_op_times.strip():
                try:
                    results['op_segments'] = self._parse_custom_times(custom_op_times, 'OP')
                except ValueError as e:
                    return "", f"OP时间段格式错误: {str(e)}", ""
            
            if custom_ed_times and isinstance(custom_ed_times, str) and custom_ed_times.strip():
                try:
                    results['ed_segments'] = self._parse_custom_times(custom_ed_times, 'ED')
                except ValueError as e:
                    return "", f"ED时间段格式错误: {str(e)}", ""
            
            # 生成输出路径
            if not output_filename or not output_filename.strip():
                output_filename = 'output_no_op_ed.mp4'
            elif not output_filename.endswith('.mp4'):
                output_filename = output_filename + '.mp4'
            
            output_path = os.path.join(self.temp_dir, output_filename)
            
            # 删除OP/ED
            start_time = time.time()
            success = self.detector.remove_op_ed_segments(
                input_video=safe_main_path,
                op_segments=results['op_segments'],
                ed_segments=results['ed_segments'],
                output_video=output_path
            )
            processing_time = time.time() - start_time
            
            if not success or not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:  # 小于1KB
                error_msg = "处理失败: 无法生成输出文件"
                if os.path.exists(output_path):
                    error_msg += f", 文件大小: {os.path.getsize(output_path)} bytes"
                return "", error_msg, ""
            
            # 生成状态信息
            total_size = os.path.getsize(safe_main_path) / 1024 / 1024
            output_size = os.path.getsize(output_path) / 1024 / 1024
            size_ratio = output_size / total_size if total_size > 0 else 0
            
            status = f"处理成功! 耗时: {processing_time:.2f}秒\n"
            status += f"原文件大小: {total_size:.2f}MB, 输出文件大小: {output_size:.2f}MB ({size_ratio:.1%})\n"
            status += f"删除了 {len(results['op_segments'])} 个OP片段和 {len(results['ed_segments'])} 个ED片段"
            
            # 保存最终结果JSON
            json_path = os.path.join(self.temp_dir, 'final_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'input_video': safe_main_path,
                    'output_video': output_path,
                    'processing_time': processing_time,
                    'file_size_ratio': size_ratio,
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            return output_path, status, json_path
            
        except Exception as e:
            logger.error(f"处理过程中出错: {str(e)}", exc_info=True)
            return "", f"处理错误: {str(e)}", ""
    
    def _parse_custom_times(self, time_str: str, segment_type: str) -> List[Dict]:
        """
        解析自定义时间段字符串
        """
        segments = []
        time_str = time_str.strip()
        
        if not time_str:
            return segments
        
        try:
            for segment_str in time_str.split(','):
                segment_str = segment_str.strip()
                if not segment_str:
                    continue
                
                parts = segment_str.split('-')
                if len(parts) != 2:
                    raise ValueError(f"{segment_type}时间段格式错误: '{segment_str}'，应为'开始-结束'")
                
                start_str, end_str = parts[0].strip(), parts[1].strip()
                
                try:
                    start_time = float(start_str)
                    end_time = float(end_str)
                except ValueError:
                    raise ValueError(f"{segment_type}时间格式错误: '{segment_str}'，时间必须是数字")
                
                if start_time < 0 or end_time < 0:
                    raise ValueError(f"{segment_type}时间不能为负数: '{segment_str}'")
                
                if start_time >= end_time:
                    raise ValueError(f"{segment_type}开始时间必须小于结束时间: '{segment_str}'")
                
                segments.append({
                    'start_frame': 0,
                    'end_frame': 0,
                    'start_time': start_time,
                    'end_time': end_time,
                    'similarity': 1.0,
                    'duration': end_time - start_time
                })
            
            return segments
        
        except Exception as e:
            raise ValueError(f"{segment_type}时间段解析错误: {str(e)}")

def create_op_ed_tab():
    """创建OP/ED处理标签页"""
    op_ed_ui = OPEDUI()
    
    with gr.TabItem("🎬 OP/ED 处理"):
        gr.Markdown("""
        ### 🎬 动漫OP/ED检测与删除工具
        
        **支持功能:**
        - 🔍 单文件检测与处理
        - 📦 批量检测与处理
        - ⚙️ 高级参数调整
        - 📊 详细结果统计
        
        **使用说明:**
        1. 选择单文件处理或批量处理
        2. 上传视频文件或选择视频目录
        3. （可选）上传OP/ED参考片段
        4. 调整参数或使用默认值
        5. 开始检测和处理
        """)
        
        with gr.Tabs():
            # 单文件处理标签页
            with gr.Tab("🎯 单文件处理"):
                with gr.Tabs():
                    # 检测标签页
                    with gr.Tab("🔍 检测OP/ED"):
                        with gr.Row():
                            with gr.Column():
                                main_video_input = gr.Video(label="主视频文件 (必填)", sources=["upload"], interactive=True)
                                with gr.Row():
                                    op_ref_input = gr.Video(label="OP参考片段 (可选)", sources=["upload"], interactive=True)
                                    ed_ref_input = gr.Video(label="ED参考片段 (可选)", sources=["upload"], interactive=True)
                                
                                with gr.Accordion("高级参数", open=False):
                                    similarity_slider = gr.Slider(
                                        minimum=0.7, maximum=0.95, value=0.85, step=0.01,
                                        label="相似度阈值", 
                                        info="值越高要求越严格，0.85为推荐值"
                                    )
                                    min_duration_slider = gr.Slider(
                                        minimum=0, maximum=120, value=10, step=1,
                                        label="最小片段时长(秒)", 
                                        info="过滤掉时长小于此值的片段"
                                    )
                                
                                detect_btn = gr.Button("🔍 检测OP/ED", variant="primary")
                            
                            with gr.Column():
                                op_output = gr.Textbox(label="OP检测结果", lines=6, interactive=False)
                                ed_output = gr.Textbox(label="ED检测结果", lines=6, interactive=False)
                                status_output = gr.Textbox(label="状态信息", lines=3, interactive=False)
                                json_output = gr.File(label="检测结果JSON")
                    
                    # 处理标签页
                    with gr.Tab("✂️ 删除并生成最终视频"):
                        with gr.Row():
                            with gr.Column():
                                main_video_input2 = gr.Video(label="主视频文件 (必填)", sources=["upload"], interactive=True)
                                with gr.Row():
                                    op_ref_input2 = gr.Video(label="OP参考片段 (可选)", sources=["upload"], interactive=True)
                                    ed_ref_input2 = gr.Video(label="ED参考片段 (可选)", sources=["upload"], interactive=True)
                                
                                with gr.Accordion("自定义时间段 (覆盖自动检测结果)", open=False):
                                    custom_op_times = gr.Textbox(
                                        label="自定义OP时间段", 
                                        placeholder="格式: 0-90,120-180 (多个用逗号分隔)",
                                        info="例如: '0-92.5' 表示0秒到92.5秒"
                                    )
                                    custom_ed_times = gr.Textbox(
                                        label="自定义ED时间段", 
                                        placeholder="格式: 1200-1292.5",
                                        info="如果自动检测不准确，可以手动指定"
                                    )
                                
                                output_filename = gr.Textbox(
                                    label="输出文件名", 
                                    value="output_no_op_ed.mp4",
                                    info="必须以.mp4结尾"
                                )
                                
                                with gr.Accordion("高级参数", open=False):
                                    similarity_slider2 = gr.Slider(
                                        minimum=0.7, maximum=0.95, value=0.85, step=0.01,
                                        label="相似度阈值"
                                    )
                                    min_duration_slider2 = gr.Slider(
                                        minimum=0, maximum=120, value=10, step=1,
                                        label="最小片段时长(秒)"
                                    )
                                
                                process_btn = gr.Button("✂️ 删除并生成最终视频", variant="primary")
                            
                            with gr.Column():
                                final_video_output = gr.Video(label="最终输出视频")
                                final_status = gr.Textbox(label="处理状态", lines=4, interactive=False)
                                final_json_output = gr.File(label="最终结果JSON")
            
            # 批量处理标签页
            with gr.Tab("📦 批量处理"):
                with gr.Tabs():
                    # 批量检测标签页
                    with gr.Tab("🔍 批量检测OP/ED"):
                        with gr.Row():
                            with gr.Column():
                                video_directory = gr.Textbox(
                                    label="视频目录路径",
                                    placeholder="例如: F:\\动漫数据集工作箱\\某番剧",
                                )
                                gr.Markdown("ℹ️ **提示**: 请确保目录中只包含需要处理的视频文件")
                                
                                with gr.Row():
                                    op_ref_batch = gr.Video(label="OP参考片段 (可选)", sources=["upload"], interactive=True)
                                    ed_ref_batch = gr.Video(label="ED参考片段 (可选)", sources=["upload"], interactive=True)
                                
                                gr.Markdown("**OP/ED参考说明**:")
                                gr.Markdown("- 如果不上传参考片段，将只进行基础分析")
                                gr.Markdown("- 参考片段应选择画面稳定、无字幕干扰的部分")
                                
                                with gr.Accordion("高级参数", open=False):
                                    batch_similarity_slider = gr.Slider(
                                        minimum=0.7, maximum=0.95, value=0.85, step=0.01,
                                        label="相似度阈值", 
                                    )
                                    batch_min_duration_slider = gr.Slider(
                                        minimum=0, maximum=120, value=10, step=1,
                                        label="最小片段时长(秒)", 
                                    )
                                
                                batch_detect_btn = gr.Button("🔍 开始批量检测", variant="primary")
                                batch_progress = gr.Textbox(label="检测进度", lines=2, interactive=False)
                            
                            with gr.Column():
                                batch_results_md = gr.Markdown(label="检测结果", value="等待检测结果...")
                                batch_json_output = gr.File(label="检测结果JSON")
                                batch_status = gr.Textbox(label="状态信息", lines=3, interactive=False)
                    
                    # 批量处理标签页
                    with gr.Tab("✂️ 批量删除OP/ED"):
                        with gr.Row():
                            with gr.Column():
                                processing_video_directory = gr.Textbox(
                                    label="视频目录路径",
                                    placeholder="例如: F:\\动漫数据集工作箱\\某番剧",
                                )
                                gr.Markdown("ℹ️ **提示**: 此目录应与批量检测时使用的目录相同")
                                
                                detection_results_file = gr.File(
                                    label="检测结果JSON文件",
                                    file_types=[".json"]
                                )
                                gr.Markdown("📄 **检测结果文件**: 从批量检测结果中下载的JSON文件")
                                
                                output_directory = gr.Textbox(
                                    label="输出目录",
                                    placeholder="例如: F:\\动漫数据集工作箱\\某番剧\\no_op_ed",
                                )
                                gr.Markdown("📁 **输出目录说明**:")
                                gr.Markdown("- 如果目录不存在，程序会自动创建")
                                gr.Markdown("- 确保有足够的磁盘空间")
                                
                                batch_process_btn = gr.Button("✂️ 开始批量处理", variant="primary")
                                batch_process_progress = gr.Textbox(label="处理进度", lines=2, interactive=False)
                            
                            with gr.Column():
                                batch_processing_results_md = gr.Markdown(label="处理结果", value="等待处理结果...")
                                batch_processing_status = gr.Textbox(label="状态信息", lines=3, interactive=False)
        
        # 底部信息
        gr.Markdown("""
        ---
        **技术说明:**
        - 使用感知哈希(pHash)算法进行帧匹配
        - 自动处理Windows文件权限问题
        - 支持多线程并行处理，提高效率
        - 智能合并重叠的检测结果
        - 详细的进度显示和结果统计
        
        **注意事项:**
        - ⚠️ 批量处理会占用较多内存和CPU资源
        - 📁 确保有足够的磁盘空间存储输出文件
        - ⏰ 处理4K视频或长视频可能需要较长时间
        - 🔒 临时文件会在程序退出时自动清理
        - 🔄 失败的文件可以单独重新处理
        """)

        # 事件处理
        # 单文件检测
        detect_btn.click(
            fn=lambda *args: op_ed_ui.detect_op_ed(*args),
            inputs=[
                main_video_input, op_ref_input, ed_ref_input,
                similarity_slider, min_duration_slider
            ],
            outputs=[
                op_output, 
                ed_output, 
                json_output, 
                gr.Textbox(visible=False),  # 隐藏预览视频输出
                status_output
            ],
            show_progress=True
        )
        
        # 单文件处理
        process_btn.click(
            fn=lambda *args: op_ed_ui.remove_op_ed(*args),
            inputs=[
                main_video_input2, op_ref_input2, ed_ref_input2,
                similarity_slider2, min_duration_slider2,
                custom_op_times, custom_ed_times,
                output_filename
            ],
            outputs=[final_video_output, final_status, final_json_output],
            show_progress=True
        )
        
        # 批量检测
        batch_detect_btn.click(
            fn=lambda *args: op_ed_ui.batch_detect_op_ed(*args),
            inputs=[
                video_directory, op_ref_batch, ed_ref_batch,
                batch_similarity_slider, batch_min_duration_slider
            ],
            outputs=[batch_results_md, batch_json_output, batch_status],
            show_progress=True
        )
        
        # 批量处理
        batch_process_btn.click(
            fn=lambda *args: op_ed_ui.batch_remove_op_ed(*args),
            inputs=[
                processing_video_directory, detection_results_file, output_directory
            ],
            outputs=[batch_processing_results_md, batch_processing_status],
            show_progress=True
        )