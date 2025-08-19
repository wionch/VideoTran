好的，遵照您的指示。我将以“零上下文”原则，提供一份完整的、未经精简的实施清单。此清单包含所有必要的代码和操作，您可以完全依赖它来完成整个重构任务。

[MODE: PLAN]

**最终目标：** 将项目代码重构为符合“方案C”设计、拥有可扩展的GPU相似度检测架构、并且可以完整运行的最终形态。

---

### **实施清单 (IMPLEMENTATION CHECKLIST) - 完整无精简版**

#### **第一部分：环境与配置准备 (Setup)**

*   **清单 1.1：安装新的依赖库**
    *   **目标：** 为即将使用的SSIM算法添加所需的 `piq` 库。
    *   **文件：** `D:\Python\Project\VideoTran\requirements.txt`
    *   **操作：** 打开该文件，在文件末尾添加新的一行：
        ```
        piq
        ```
    *   **验收标准：** 在您的项目环境中运行 `pip install -r requirements.txt` 命令，可以成功安装 `piq` 库而无错误。

*   **清单 1.2：更新配置文件**
    *   **目标：** 在配置文件中添加对新相似度架构的支持，使其可配置。
    *   **文件：** `D:\Python\Project\VideoTran\configs\config.yaml`
    *   **操作：** 打开该文件，在文件末尾添加以下配置内容：
        ```yaml
        similarity:
          algorithm: ssim  # 算法选择, 可选: ssim, mse
          threshold: 0.1   # 变化检测的阈值。SSIM建议用0.1左右, MSE建议用1e-5左右
        ```
    *   **验收标准：** `config.yaml` 文件被成功修改并保存，包含了新的 `similarity` 配置节。

---

#### **第二部分：构建可扩展的相似度检测模块 (Architecture Refactoring)**

*   **清单 2.1：创建新的相似度提供者模块**
    *   **目标：** 建立一个全新的、可扩展的相似度计算模块，以取代旧的实现。
    *   **操作：**
        1.  在 `D:\Python\Project\VideoTran\video_tran\utils\` 目录下创建一个新文件，名为 `similarity_provider.py`。
        2.  将以下完整内容复制并粘贴到新建的 `similarity_provider.py` 文件中：
            ```python
            # D:\\Python\\Project\\VideoTran\\video_tran\\utils\\similarity_provider.py
            import torch
            import piq
            from abc import ABC, abstractmethod
            from typing import List, Tuple

            class BaseSimilarityProvider(ABC):
                """定义了相似度提供者的统一接口。"""
                @abstractmethod
                def compare_batch(self, frame_tensors: torch.Tensor) -> List[float]:
                    """
                    比较一个批次中的相邻帧，返回一个“变化分数”的列表。
                    分数越高，表示相邻帧之间的变化越大。
                    列表的第一个元素总是0，因为第一帧没有可比较的对象。
                    """
                    pass

            class SSIMSimilarityProvider(BaseSimilarityProvider):
                """使用结构相似性(SSIM)进行比较的提供者。更符合人类视觉感知。"""
                def compare_batch(self, frame_tensors: torch.Tensor) -> List[float]:
                    if not frame_tensors.is_cuda:
                        raise ValueError("输入的张量必须在GPU上。")
                    if frame_tensors.shape[0] < 2:
                        return [0.0] * frame_tensors.shape[0]
                    
                    frames_a = frame_tensors[:-1]
                    frames_b = frame_tensors[1:]
                    
                    # SSIM值域为[0, 1]，1表示完全相同。我们用 1.0 - ssim 作为变化分数。
                    ssim_scores = piq.ssim(frames_a, frames_b, data_range=1.0, reduction='none')
                    change_scores = 1.0 - ssim_scores
                    
                    return [0.0] + change_scores.cpu().tolist()

            class MSESimilarityProvider(BaseSimilarityProvider):
                """使用均方误差(MSE)进行比较的提供者。速度快但对感知不敏感。"""
                def compare_batch(self, frame_tensors: torch.Tensor) -> List[float]:
                    if not frame_tensors.is_cuda:
                        raise ValueError("输入的张量必须在GPU上。")
                    if frame_tensors.shape[0] < 2:
                        return [0.0] * frame_tensors.shape[0]
                    
                    frames_a = frame_tensors[:-1]
                    frames_b = frame_tensors[1:]
                    
                    mse_losses = torch.nn.functional.mse_loss(frames_a, frames_b, reduction='none')
                    mse_per_pair = mse_losses.mean(dim=[1, 2, 3])
                    
                    return [0.0] + mse_per_pair.cpu().tolist()
            ```
    *   **验收标准：** `D:\Python\Project\VideoTran\video_tran\utils\similarity_provider.py` 文件被成功创建并包含上述完整代码。

*   **清单 2.2：删除旧的相似度模块**
    *   **目标：** 移除过时且不再需要的旧文件。
    *   **文件：** `D:\Python\Project\VideoTran\video_tran\utils\image_similarity.py`
    *   **操作：** 在您的文件管理器中，找到并**删除**这个文件。
    *   **验收标准：** `D:\Python\Project\VideoTran\video_tran\utils\` 目录下不再存在 `image_similarity.py` 文件。

---

#### **第三部分：核心分析器 `FrameAnalyzer` 的重构 (Logic Implementation)**

*   **清单 3.1：替换 `FrameAnalyzer` 的 `__init__` 方法**
    *   **目标：** 修改 `FrameAnalyzer` 的初始化逻辑，使其能够根据配置文件动态加载并使用我们新创建的相似度提供者。
    *   **文件：** `D:\Python\Project\VideoTran\video_tran\subtitle_processor\frame_analyzer.py`
    *   **操作：** 在 `FrameAnalyzer` 类中，找到 `__init__` 方法，并将其**整体替换**为以下代码：
        ```python
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
            self.ocr_engine = PaddleOCR(use_gpu=True, show_log=False, use_angle_cls=False, lang=self.config['paddle_ocr']['lang'])
            
            sim_config = self.config.get('similarity', {})
            algo_name = sim_config.get('algorithm', 'ssim')
            print(f"初始化相似度检测器，使用算法: {algo_name.upper()}")
            if algo_name == 'ssim':
                self.similarity_provider = SSIMSimilarityProvider()
            elif algo_name == 'mse':
                self.similarity_provider = MSESimilarityProvider()
            else:
                raise ValueError(f"未知的相似度算法: {algo_name}")
            
            self.subtitle_area = None # 为Y轴区域初始化一个占位符
            print(f"帧分析器已在设备 {self.device} 上初始化")
        ```
    *   **验收标准：** `FrameAnalyzer` 的 `__init__` 方法被成功替换为新代码。

*   **清单 3.2：添加Y轴检测及设置方法**
    *   **目标：** 在`FrameAnalyzer`中，实现“方案C”描述的智能Y轴定位逻辑。
    *   **文件：** `D:\Python\Project\VideoTran\video_tran\subtitle_processor\frame_analyzer.py`
    *   **操作：** 在`FrameAnalyzer`类中（例如，在 `__init__` 方法之后），添加以下**两个完整的新方法** `set_subtitle_area` 和 `determine_subtitle_area`：
        ```python
        def set_subtitle_area(self, area: Tuple[int, int, int, int]):
            """设置字幕区域以供后续使用"""
            print(f"分析器已设置字幕区域: {area}")
            self.subtitle_area = area

        def determine_subtitle_area(self, video_path: str, initial_subtitles: List[Dict[str, Any]], sample_count: int = 7) -> Tuple[int, int, int, int]:
            """
            通过对长字幕进行采样和OCR，智能确定视频中字幕的主要Y轴范围。
            """
            import cv2
            from typing import List, Dict, Any, Tuple
            
            print("--- 开始智能检测Y轴字幕区域 ---")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件: {video_path}")
            
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)

            sorted_subs = sorted(initial_subtitles, key=lambda s: len(s.get('text', '')), reverse=True)
            samples = sorted_subs[:sample_count]
            
            y_coords = []

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
                
                if not ret: continue

                result = self.ocr_engine.ocr(frame, cls=False)
                if not (result and result[0]): continue
                
                from difflib import SequenceMatcher
                best_match_ratio = 0.7
                
                full_ocr_text = " ".join([res[1][0] for res in result[0]])
                match = SequenceMatcher(None, sub['text'], full_ocr_text).find_longest_match()
                ratio = match.size / len(sub['text']) if len(sub['text']) > 0 else 0

                if ratio > best_match_ratio:
                    min_y, max_y = float('inf'), float('-inf')
                    for res in result[0]:
                        box = res[0]
                        current_min_y = min(p[1] for p in box)
                        current_max_y = max(p[1] for p in box)
                        min_y = min(min_y, current_min_y)
                        max_y = max(max_y, current_max_y)
                    
                    if min_y != float('inf'):
                        y_coords.append((min_y, max_y))
                        print(f"  - 样本 {i+1}: 找到匹配区域 Y=[{int(min_y)}, {int(max_y)}]")

            cap.release()

            if not y_coords:
                raise RuntimeError("无法通过OCR样本自动确定字幕区域！")

            avg_min_y = sum(item[0] for item in y_coords) / len(y_coords)
            avg_max_y = sum(item[1] for item in y_coords) / len(y_coords)
            
            final_min_y = max(0, int(avg_min_y) - 10)
            final_max_y = int(avg_max_y) + 10
            
            final_area = (0, final_min_y, video_width, final_max_y)
            print(f"--- Y轴区域检测完成。最终确定区域: {final_area} ---")
            return final_area
        ```
    *   **验收标准：** `FrameAnalyzer`类中成功添加了`determine_subtitle_area`和`set_subtitle_area`这两个新方法。

*   **清单 3.3：替换 `_extract_frame_batches` 方法**
    *   **目标：** 让帧提取逻辑使用动态计算出的Y轴区域，而不是硬编码。
    *   **文件：** `D:\Python\Project\VideoTran\video_tran\subtitle_processor\frame_analyzer.py`
    *   **操作：** 在 `FrameAnalyzer` 类中，找到 `_extract_frame_batches` 方法，并将其**整体替换**为以下代码：
        ```python
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
        ```
    *   **验收标准：** `_extract_frame_batches` 方法被成功替换，不再包含硬编码的裁剪逻辑。

*   **清单 3.4：替换 `analyze_time_range` 方法**
    *   **目标：** 彻底重写核心分析方法，以完全实现“方案C”所描述的，基于“相似度优先”的高效断句逻辑。
    *   **文件：** `D:\Python\Project\VideoTran\video_tran\subtitle_processor\frame_analyzer.py`
    *   **操作：** 在 `FrameAnalyzer` 类中，找到 `analyze_time_range` 方法，并将其**整体替换**为以下代码：
        ```python
        def analyze_time_range(self, video_path: str, time_range: tuple[float, float], initial_speaker: str) -> List[PreciseSubtitle]:
            import time
            import torch
            import cv2
            from ..utils.data_structures import PreciseSubtitle

            total_start_time = time.time()
            print(f"正在分析 {video_path}，时间从 {time_range[0]:.2f}秒 到 {time_range[1]:.2f}秒...")
            
            all_strips_tensors = []
            all_timestamps = []
            for batch_tensors, frame_timestamps, _ in self._extract_frame_batches(video_path, time_range):
                all_strips_tensors.append(batch_tensors)
                all_timestamps.extend(frame_timestamps)
            
            if not all_strips_tensors:
                return []
            
            full_tensor = torch.cat(all_strips_tensors)
            print(f"  - 第1轮: 提取了 {len(full_tensor)} 帧字幕条。")

            sim_config = self.config.get('similarity', {})
            threshold = sim_config.get('threshold', 0.1)
            change_scores = self.similarity_provider.compare_batch(full_tensor)
            change_flags = [score > threshold for score in change_scores]
            cut_point_indices = [i for i, flag in enumerate(change_flags) if flag]
            print(f"  - 第2轮: 发现 {len(cut_point_indices)} 个潜在断句点。")

            ocr_indices = sorted(list(set([0] + cut_point_indices)))
            texts_at_indices = {}
            for idx in ocr_indices:
                frame_tensor = full_tensor[idx]
                frame_np_rgb = frame_tensor.permute(1, 2, 0).cpu().numpy() * 255
                frame_np_bgr = cv2.cvtColor(frame_np_rgb.astype(np.uint8), cv2.COLOR_RGB_BGR)
                
                ocr_result = self.ocr_engine.ocr(frame_np_bgr, cls=False)
                text = " ".join([res[1][0] for res in ocr_result[0]]) if ocr_result and ocr_result[0] else ""
                texts_at_indices[idx] = text.strip()
            print(f"  - 第3轮: 在 {len(ocr_indices)} 个关键帧上完成OCR。")

            precise_subtitles = []
            if not texts_at_indices.get(0):
                return []

            start_idx = 0
            current_text = texts_at_indices[0]
            for i in range(1, len(ocr_indices)):
                end_idx = ocr_indices[i]
                next_text = texts_at_indices[end_idx]
                if current_text:
                    start_time = all_timestamps[start_idx]
                    end_time = all_timestamps[end_idx - 1]
                    if end_time > start_time:
                        precise_subtitles.append(PreciseSubtitle(id=0, start_time=start_time, end_time=end_time, text=current_text, speaker=initial_speaker, coordinates=(0,0,0,0)))
                start_idx = end_idx
                current_text = next_text

            if current_text:
                start_time = all_timestamps[start_idx]
                end_time = all_timestamps[-1]
                if end_time > start_time:
                    precise_subtitles.append(PreciseSubtitle(id=0, start_time=start_time, end_time=end_time, text=current_text, speaker=initial_speaker, coordinates=(0,0,0,0)))
            
            print(f"  - 第4轮: 生成了 {len(precise_subtitles)} 条字幕。分段耗时: {time.time() - total_start_time:.2f}s")
            return self._merge_duplicate_subtitles(precise_subtitles)
        ```
    *   **验收标准：** `analyze_time_range` 方法被成功替换，其逻辑现在是“提取所有->批量比较->少量OCR->重建”。

---

#### **第四部分：主入口脚本的整合与清理 (Integration & Finalization)**

*   **清单 4.1：重构主脚本 `main` 函数**
    *   **目标：** 串联所有新逻辑，并移除所有调试中断点，使程序能完整运行。
    *   **文件：** `D:\Python\Project\VideoTran\remove_subtitle_v3.py`
    *   **操作：** 在 `main` 函数中，找到 `if args.subtitle_input:` 这个代码块，并将其**从 `print("--- 分析模式启动 ---")` 开始，一直到 `break` 的整块代码**，替换为以下内容：
        ```python
        print("--- 分析模式启动 ---")
        print(f"加载配置文件: {args.config_path}")
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        remover_config = config.get('remover_v3', {})

        print(f"读取初始字幕文件: {args.subtitle_input}")
        with open(args.subtitle_input, 'r', encoding='utf-8') as f:
            initial_subtitles = json.load(f)

        print("初始化帧分析器...")
        analyzer = FrameAnalyzer(remover_config)

        # Y轴区间确定
        if args.subtitle_area:
            subtitle_area = tuple(args.subtitle_area)
            print(f"使用手动提供的Y轴区间: {subtitle_area}")
        else:
            print("未提供Y轴区间，开始自动检测...")
            try:
                subtitle_area = analyzer.determine_subtitle_area(args.video_input, initial_subtitles)
            except Exception as e:
                print(f"自动检测Y轴区域失败: {e}")
                return # 失败则中断
        
        analyzer.set_subtitle_area(subtitle_area)

        all_precise_subtitles: List[PreciseSubtitle] = []
        print("开始逐段进行精准字幕分析...")

        def time_str_to_seconds(time_str):
            time_str = time_str.replace(',', '.')
            parts = time_str.split(':')
            seconds = 0
            if len(parts) == 3:
                seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                seconds = float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 1:
                seconds = float(parts[0])
            return seconds

        for sub in initial_subtitles:
            try:
                start_seconds = time_str_to_seconds(sub['startTime'])
                end_seconds = time_str_to_seconds(sub['endTime'])
            except (ValueError, IndexError):
                print(f"警告: 无法解析时间戳 '{sub['startTime']}' 或 '{sub['endTime']}'. 跳过此片段。")
                continue
            
            time_range = (start_seconds, end_seconds)
            speaker = sub.get('speaker', 'SPEAKER_UNKNOWN')
            
            print(f"\n--- 正在处理分段 ID {sub.get('id', 'N/A')} ({sub['startTime']} -> {sub['endTime']}) ---")
            precise_subs = analyzer.analyze_time_range(args.video_input, time_range, speaker)
            all_precise_subtitles.extend(precise_subs)
        ```
    *   **验收标准：** `main` 函数中的分析逻辑被完整替换，不再包含任何 `return` 或 `break` 形式的调试中断点。脚本现在可以从头到尾完整地运行。

---

此清单是最终的、完整的实施指南。请在批准后，从清单1.1开始，逐步执行。

**请确认此最终实施清单。批准后，我将开始引导您完成每一步。(y/n)**