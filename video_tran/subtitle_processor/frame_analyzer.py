# video_tran/subtitle_processor/frame_analyzer.py

import time
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from typing import List, Dict, Any, Generator, Tuple
import os

from ..utils.data_structures import PreciseSubtitle

from paddleocr import PaddleOCR

class FrameAnalyzer:
    """
    负责分析视频的指定时间区间，通过变化检测和OCR，
    重建高精度的字幕数据。
    """
    def __init__(self, config: Dict[str, Any]):
        from ..utils.similarity_provider import SSIMSimilarityProvider, MSESimilarityProvider
        import torchvision.transforms as T
        import torch

        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = self.config.get('batch_size', 32)
        self.ocr_batch_size = self.config.get('ocr_batch_size', 8)
        self.preprocess = T.Compose([T.ToTensor()])
        
        print("初始化 PaddleOCR 引擎...")
        # lang config is nested under remover_v3 now, let's access it correctly
        paddle_config = self.config.get('paddle_ocr', {})
        self.ocr_engine = PaddleOCR(use_gpu=True, show_log=False, use_angle_cls=False, lang=paddle_config.get('lang', 'en'))
        
        sim_config = self.config.get('similarity', {})
        algo_name = sim_config.get('algorithm', 'ssim')
        print(f"初始化相似度检测器，使用算法: {algo_name.upper()}")
        if algo_name == 'ssim':
            self.similarity_provider = SSIMSimilarityProvider()
        elif algo_name == 'mse':
            self.similarity_provider = MSESimilarityProvider()
        else:
            raise ValueError(f"未知的相似度算法: {algo_name}")
        
        self.y_axis_config = self.config.get('y_axis_detection', {})
        self.y_axis_similarity_threshold = self.y_axis_config.get('similarity_threshold', 0.2)

        self.subtitle_area = None
        print(f"帧分析器已在设备 {self.device} 上初始化")

    def set_subtitle_area(self, area: Tuple[int, int, int, int]):
        """设置字幕区域以供后续使用"""
        print(f"分析器已设置字幕区域: {area}")
        self.subtitle_area = area

    def determine_subtitle_area(self, video_path: str, initial_subtitles: List[Dict[str, Any]], sample_count: int = 20) -> Tuple[int, int, int, int]:
        """
        通过对长字幕进行采样和OCR，智能确定视频中字幕的主要Y轴范围。
        """
        import cv2
        from typing import List, Dict, Any, Tuple
        from difflib import SequenceMatcher
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        sorted_subs = sorted(initial_subtitles, key=lambda s: len(s.get('text', '')), reverse=True)
        samples = sorted_subs[:sample_count]
        
        all_found_coords = []

        def time_str_to_seconds(time_str):
            time_str = time_str.replace(',', '.')
            parts = time_str.split(':')
            if len(parts) == 3: return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])

        for i, sub in enumerate(samples):
            start_sec = time_str_to_seconds(sub['startTime'])
            end_sec = time_str_to_seconds(sub['endTime'])
            
            mid_frame_idx = int(((start_sec + end_sec) / 2) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue

            result = self.ocr_engine.ocr(frame, cls=False)
            if not (result and result[0]):
                continue
            
            target_text = sub.get('text', '').strip()
            if not target_text:
                continue

            best_match_ratio = 0.0
            best_match_box = None
            
            for res in result[0]:
                ocr_text = res[1][0]
                ratio = SequenceMatcher(None, target_text, ocr_text).ratio()
                
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_match_box = res[0]

            if best_match_ratio > self.y_axis_similarity_threshold:
                min_y = min(p[1] for p in best_match_box)
                max_y = max(p[1] for p in best_match_box)
                all_found_coords.append((min_y, max_y))

        cap.release()

        if len(all_found_coords) < 3:
            raise RuntimeError(f"无法通过OCR样本自动确定字幕区域！有效样本数过少({len(all_found_coords)}), 请检查视频内容或调整y_axis_detection配置。")

        min_ys = np.array([c[0] for c in all_found_coords])
        max_ys = np.array([c[1] for c in all_found_coords])

        median_min_y = np.median(min_ys)
        median_max_y = np.median(max_ys)
        std_min_y = np.std(min_ys)
        std_max_y = np.std(max_ys)

        filtered_coords = []
        for min_y, max_y in all_found_coords:
            if abs(min_y - median_min_y) < 2 * std_min_y and abs(max_y - median_max_y) < 2 * std_max_y:
                filtered_coords.append((min_y, max_y))
        
        if not filtered_coords:
            filtered_coords = all_found_coords

        avg_min_y = sum(c[0] for c in filtered_coords) / len(filtered_coords)
        avg_max_y = sum(c[1] for c in filtered_coords) / len(filtered_coords)
        
        padding = 10
        final_min_y = max(0, int(avg_min_y - padding))
        final_max_y = int(avg_max_y + padding)
        
        final_area = (0, final_min_y, video_width, final_max_y)
        return final_area

    def _extract_frame_batches(self, video_path: str, time_range: tuple[float, float], initial_sub: Dict[str, Any]) -> Generator[Tuple[torch.Tensor, List[float], int], None, None]:
            import cv2
            import torch
            import numpy as np
            from typing import Generator, Tuple, List

            if not hasattr(self, 'subtitle_area') or self.subtitle_area is None:
                raise ValueError("错误: 在提取帧之前必须先调用 set_subtitle_area() 设置字幕区域。")

            sub_id = initial_sub.get('id')
            is_debug_sub = (sub_id == 1)
            debug_dir = None
            if is_debug_sub:
                debug_dir = f'workspace/temp/sub_{sub_id}_full_frames'
                os.makedirs(debug_dir, exist_ok=True)
                print(f"  [调试] 已为ID {sub_id} 启用精确OCR框图，将保存到: {os.path.abspath(debug_dir)}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            x1, y1, x2, y2 = self.subtitle_area
            crop_y_start_for_coords = y1

            start_frame_idx = int(time_range[0] * fps)
            end_frame_idx = int(time_range[1] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            
            frames_buffer, timestamps_buffer = [], []
            for i in range(start_frame_idx, end_frame_idx + 1):
                ret, frame = cap.read()
                if not ret: break

                cropped_frame = frame[y1:y2, x1:x2]

                if is_debug_sub:
                    frame_with_box = frame.copy()
                    # 对当前帧的裁剪区域进行OCR，以获得最精确的边界框
                    ocr_result = self.ocr_engine.ocr(cropped_frame, cls=False)
                    if ocr_result and ocr_result[0]:
                        for res in ocr_result[0]:
                            box = res[0]
                            # 将相对坐标转换回绝对坐标
                            abs_box = np.array([[[p[0] + x1, p[1] + y1] for p in box]], dtype=np.int32)
                            cv2.polylines(frame_with_box, [abs_box], isClosed=True, color=(0, 0, 255), thickness=2)
                    
                    timestamp = i / fps
                    debug_filename = os.path.join(debug_dir, f"frame_{timestamp:.3f}.png")
                    cv2.imwrite(debug_filename, frame_with_box)
                
                frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(self.preprocess(frame_rgb))
                timestamps_buffer.append(i / fps)

                if len(frames_buffer) == self.batch_size:
                    yield torch.stack(frames_buffer).to(self.device), timestamps_buffer, crop_y_start_for_coords
                    frames_buffer, timestamps_buffer = [], []
            
            if frames_buffer:
                yield torch.stack(frames_buffer).to(self.device), timestamps_buffer, crop_y_start_for_coords
            
            total_frames = end_frame_idx - start_frame_idx + 1
            print(f"  [调试-帧提取] 完成字幕ID {sub_id} 的帧提取。共处理 {total_frames} 帧，时间戳范围 {time_range[0]:.3f} 到 {time_range[1]:.3f}。")
            cap.release()

    def _normalize_text(self, text: str) -> str:
        """将文本转为小写并移除常见标点符号，用于更宽松的比较。"""
        import string
        # 移除所有标点
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 转为小写并移除空白
        return text.lower().strip()

    def _merge_duplicate_subtitles(self, subtitles: List[PreciseSubtitle]) -> List[PreciseSubtitle]:
        if not subtitles:
            return []
        
        merged_subtitles = [subtitles[0]]
        for i in range(1, len(subtitles)):
            # 使用归一化后的文本进行比较
            current_norm_text = self._normalize_text(subtitles[i].text)
            prev_norm_text = self._normalize_text(merged_subtitles[-1].text)

            if current_norm_text == prev_norm_text:
                # 如果文本内容相似，则合并时间
                merged_subtitles[-1].end_time = subtitles[i].end_time
            elif not current_norm_text:
                # 如果当前文本为空，则跳过，不添加到合并列表
                continue
            else:
                # 否则，添加为新的字幕条
                merged_subtitles.append(subtitles[i])
        return merged_subtitles

    def analyze_time_range(self, video_path: str, time_range: tuple[float, float], initial_speaker: str, initial_sub: Dict[str, Any]) -> List[PreciseSubtitle]:
        import time
        import torch
        import cv2
        from ..utils.data_structures import PreciseSubtitle

        total_start_time = time.time()
        print(f"正在分析 {video_path}，时间从 {time_range[0]:.2f}秒 到 {time_range[1]:.2f}秒...")
        
        all_strips_tensors = []
        all_timestamps = []
        crop_y_start = self.subtitle_area[1]

        for batch_tensors, frame_timestamps, _ in self._extract_frame_batches(video_path, time_range, initial_sub):
            all_strips_tensors.append(batch_tensors)
            all_timestamps.extend(frame_timestamps)
        
        if not all_strips_tensors:
            return []
        
        full_tensor = torch.cat(all_strips_tensors)
        print(f"  - 第1轮: 提取了 {len(full_tensor)} 帧字幕条。")

        # --- 混合变化检测开始 ---

        # 1. 基于相似度的主要变化检测
        sim_config = self.config.get('similarity', {})
        threshold = sim_config.get('threshold', 0.4)
        change_scores = self.similarity_provider.compare_batch(full_tensor)
        change_flags = [score > threshold for score in change_scores]
        cut_point_indices = {i+1 for i, flag in enumerate(change_flags) if flag} # change_scores[i] 是 i 和 i+1 帧的比较，所以断点是 i+1
        print(f"  - 第2轮(a): 相似度检测发现 {len(cut_point_indices)} 个潜在断句点。")

        # --- 新增的调试代码 --- #
        if cut_point_indices:
            print(f"  [调试-相似度] 相似度断句点位于以下时间戳:")
            for idx in sorted(list(cut_point_indices)):
                if idx > 0:
                    ts = all_timestamps[idx]
                    prev_ts = all_timestamps[idx-1]
                    score = change_scores[idx-1] # change_scores的索引比帧索引小1
                    print(f"    - 索引 {idx} (时间戳: {ts:.3f}秒), 从前一帧 (时间戳: {prev_ts:.3f}秒) 过渡, 变化得分: {score:.4f}")
        # --- 调试代码结束 --- #

        # 2. 基于标准差的轻量级空白帧检测
        remover_config = self.config.get('remover_v3', {})
        blank_threshold = remover_config.get('blank_detection_threshold', 0.01)
        stds = torch.std(full_tensor, dim=[1, 2, 3])
        is_blank_list = (stds < blank_threshold).tolist()

        # --- 新增的调试代码 --- #
        print("  [调试-标准差] 所有帧的像素标准差:")
        for i, std_val in enumerate(stds):
            ts = all_timestamps[i]
            is_blank_by_threshold = is_blank_list[i]
            print(f"    - 索引 {i}, 时间戳: {ts:.3f}秒, 标准差: {std_val:.4f}, 是否空白(阈值{blank_threshold}): {is_blank_by_threshold}")
        # --- 调试代码结束 --- #

        # 3. 寻找空白/非空白的转换点，并补充到断句点中
        blank_transition_points = set()
        for i in range(1, len(is_blank_list)):
            if is_blank_list[i] != is_blank_list[i-1]:
                blank_transition_points.add(i)
        
        original_points_count = len(cut_point_indices)
        cut_point_indices.update(blank_transition_points)
        print(f"  - 第2轮(b): 空白检测补充了 {len(cut_point_indices) - original_points_count} 个新的断句点。")

        # --- 混合变化检测结束 ---

        ocr_indices = sorted(list(set([0] + list(cut_point_indices))))
        texts_at_indices = {}
        for idx in ocr_indices:
            frame_tensor = full_tensor[idx]
            frame_np_rgb = frame_tensor.permute(1, 2, 0).cpu().numpy() * 255
            frame_np_bgr = cv2.cvtColor(frame_np_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            ocr_result = self.ocr_engine.ocr(frame_np_bgr, cls=False)
            text = ""
            bbox = (0, 0, 0, 0)
            if ocr_result and ocr_result[0]:
                text = " ".join([res[1][0] for res in ocr_result[0]])
                all_boxes = [res[0] for res in ocr_result[0]]
                x_min = min(p[0] for box in all_boxes for p in box)
                y_min = min(p[1] for box in all_boxes for p in box)
                x_max = max(p[0] for box in all_boxes for p in box)
                y_max = max(p[1] for box in all_boxes for p in box)
                bbox = (int(x_min), int(y_min + crop_y_start), int(x_max), int(y_max + crop_y_start))

            texts_at_indices[idx] = (text.strip(), bbox)
            frame_timestamp = all_timestamps[idx]
            print(f"  [调试-OCR] 关键帧索引: {idx}, 时间戳: {frame_timestamp:.3f}, OCR文本: '{text.strip() if text.strip() else '[空]'}'', Bbox: {bbox}")
        print(f"  - 第3轮: 在 {len(ocr_indices)} 个关键帧上完成OCR。")

        precise_subtitles = []
        if not ocr_indices:
            return []

        segment_start_idx = ocr_indices[0]
        current_text, current_bbox = texts_at_indices[segment_start_idx]

        for i in range(1, len(ocr_indices)):
            next_idx = ocr_indices[i]
            next_text, next_bbox = texts_at_indices[next_idx]
            
            if self._normalize_text(next_text) != self._normalize_text(current_text):
                if current_text:
                    start_time = all_timestamps[segment_start_idx]
                    end_time = all_timestamps[next_idx - 1]
                    if end_time > start_time:
                        new_sub = PreciseSubtitle(id=0, start_time=start_time, end_time=end_time, text=current_text, speaker=initial_speaker, coordinates=current_bbox)
                        print(f"  [调试-分段] 新建字幕分段: 开始={new_sub.start_time:.3f}, 结束={new_sub.end_time:.3f}, 文本='{new_sub.text}'")
                        precise_subtitles.append(new_sub)
                
                segment_start_idx = next_idx
                current_text = next_text
                current_bbox = next_bbox

        if current_text:
            start_time = all_timestamps[segment_start_idx]
            end_time = all_timestamps[-1]
            if end_time > start_time:
                new_sub = PreciseSubtitle(id=0, start_time=start_time, end_time=end_time, text=current_text, speaker=initial_speaker, coordinates=current_bbox)
                print(f"  [调试-分段] 新建字幕分段 (最终): 开始={new_sub.start_time:.3f}, 结束={new_sub.end_time:.3f}, 文本='{new_sub.text}'")
                precise_subtitles.append(new_sub)
        
        print(f"  - 第4轮: 生成了 {len(precise_subtitles)} 条字幕。分段耗时: {time.time() - total_start_time:.2f}s")
        return precise_subtitles
