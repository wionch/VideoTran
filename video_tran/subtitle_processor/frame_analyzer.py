# video_tran/subtitle_processor/frame_analyzer.py

import time
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from typing import List, Dict, Any, Generator, Tuple

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

        print("--- 开始智能检测Y轴字幕区域 ---")
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

        print(f"选取了 {len(samples)} 条最长字幕作为样本进行分析...")
        for i, sub in enumerate(samples):
            start_sec = time_str_to_seconds(sub['startTime'])
            end_sec = time_str_to_seconds(sub['endTime'])
            
            mid_frame_idx = int(((start_sec + end_sec) / 2) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"  - 样本 {i+1}: 无法读取帧 at index {mid_frame_idx}")
                continue

            result = self.ocr_engine.ocr(frame, cls=False)
            if not (result and result[0]):
                print(f"  - 样本 {i+1}: 在帧 {mid_frame_idx} 未找到任何文本。")
                continue
            
            target_text = sub.get('text', '').strip()
            if not target_text:
                print(f"  - 样本 {i+1}: 原始字幕文本为空，跳过。")
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
                print(f"  - 样本 {i+1}: 匹配成功 (相似度: {best_match_ratio:.2f})，Y轴区域: [{int(min_y)}, {int(max_y)}]")
            else:
                full_ocr_text = " | ".join([res[1][0] for res in result[0]])
                print(f"  - 样本 {i+1}: 匹配失败。目标: '{target_text}', OCR识别内容: '{full_ocr_text}', 最高相似度: {best_match_ratio:.2f}")

        cap.release()

        if len(all_found_coords) < 3:
            raise RuntimeError(f"无法通过OCR样本自动确定字幕区域！有效样本数过少({len(all_found_coords)}), 请检查视频内容或调整y_axis_detection配置。")

        print(f"--- 第一轮筛选完成，找到 {len(all_found_coords)} 个有效Y轴坐标。开始统计分析... ---")

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
        
        print(f"--- 第二轮离群值过滤完成，剩余 {len(filtered_coords)} 个高置信度坐标。 ---")

        if not filtered_coords:
            print("警告: 离群值过滤后无可用坐标，将使用第一轮所有结果进行计算。")
            filtered_coords = all_found_coords

        avg_min_y = sum(c[0] for c in filtered_coords) / len(filtered_coords)
        avg_max_y = sum(c[1] for c in filtered_coords) / len(filtered_coords)
        
        padding = 10
        final_min_y = max(0, int(avg_min_y - padding))
        final_max_y = int(avg_max_y + padding)
        
        final_area = (0, final_min_y, video_width, final_max_y)
        print(f"--- Y轴区域检测完成。最终确定区域: {final_area} ---")
        return final_area

    def _extract_frame_batches(self, video_path: str, time_range: tuple[float, float]) -> Generator[Tuple[torch.Tensor, List[float], int], None, None]:
            import cv2
            import torch
            from typing import Generator, Tuple, List

            if not hasattr(self, 'subtitle_area') or self.subtitle_area is None:
                raise ValueError("错误: 在提取帧之前必须先调用 set_subtitle_area() 设置字幕区域。")

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
                
                frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(self.preprocess(frame_rgb))
                timestamps_buffer.append(i / fps)

                if len(frames_buffer) == self.batch_size:
                    yield torch.stack(frames_buffer).to(self.device), timestamps_buffer, crop_y_start_for_coords
                    frames_buffer, timestamps_buffer = [], []
            
            if frames_buffer:
                yield torch.stack(frames_buffer).to(self.device), timestamps_buffer, crop_y_start_for_coords
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

    def analyze_time_range(self, video_path: str, time_range: tuple[float, float], initial_speaker: str) -> List[PreciseSubtitle]:
        import time
        import torch
        import cv2
        from ..utils.data_structures import PreciseSubtitle

        total_start_time = time.time()
        print(f"正在分析 {video_path}，时间从 {time_range[0]:.2f}秒 到 {time_range[1]:.2f}秒...")
        
        all_strips_tensors = []
        all_timestamps = []
        crop_y_start = self.subtitle_area[1]

        for batch_tensors, frame_timestamps, _ in self._extract_frame_batches(video_path, time_range):
            all_strips_tensors.append(batch_tensors)
            all_timestamps.extend(frame_timestamps)
        
        if not all_strips_tensors:
            return []
        
        full_tensor = torch.cat(all_strips_tensors)
        print(f"  - 第1轮: 提取了 {len(full_tensor)} 帧字幕条。")

        sim_config = self.config.get('similarity', {})
        threshold = sim_config.get('threshold', 0.4)
        change_scores = self.similarity_provider.compare_batch(full_tensor)
        change_flags = [score > threshold for score in change_scores]
        cut_point_indices = [i for i, flag in enumerate(change_flags) if flag]
        print(f"  - 第2轮: 发现 {len(cut_point_indices)} 个潜在断句点。")

        ocr_indices = sorted(list(set([0] + cut_point_indices)))
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
        print(f"  - 第3轮: 在 {len(ocr_indices)} 个关键帧上完成OCR。")

        precise_subtitles = []
        if not ocr_indices:
            return []

        # --- 核心逻辑重构 ---
        # 从第一个OCR点开始
        segment_start_idx = ocr_indices[0]
        current_text, current_bbox = texts_at_indices[segment_start_idx]

        for i in range(1, len(ocr_indices)):
            next_idx = ocr_indices[i]
            next_text, next_bbox = texts_at_indices[next_idx]
            
            # 如果文本内容发生变化，则结束当前片段
            if next_text != current_text:
                # 只有当旧文本有效时，才创建片段
                if current_text:
                    start_time = all_timestamps[segment_start_idx]
                    # 片段的结束帧是下一个片段开始的前一帧
                    end_time = all_timestamps[next_idx - 1]
                    if end_time > start_time:
                        precise_subtitles.append(PreciseSubtitle(id=0, start_time=start_time, end_time=end_time, text=current_text, speaker=initial_speaker, coordinates=current_bbox))
                    
                    # 开启新片段
                    segment_start_idx = next_idx
                    current_text = next_text
                    current_bbox = next_bbox

        # 保存最后一个片段
        if current_text:
            start_time = all_timestamps[segment_start_idx]
            # 最后一个片段的结束时间是整个分析范围的结束时间
            end_time = all_timestamps[-1]
            if end_time > start_time:
                precise_subtitles.append(PreciseSubtitle(id=0, start_time=start_time, end_time=end_time, text=current_text, speaker=initial_speaker, coordinates=current_bbox))
        
        print(f"  - 第4轮: 生成了 {len(precise_subtitles)} 条字幕。分段耗时: {time.time() - total_start_time:.2f}s")
        # 移除局部的合并调用，后续将进行全局合并
        return precise_subtitles