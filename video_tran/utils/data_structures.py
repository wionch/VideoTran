# video_tran/utils/data_structures.py

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PreciseSubtitle:
    """
    用于存储通过视频帧分析得到的高精度字幕信息的数据类。
    """
    id: int
    start_time: float
    end_time: float
    text: str
    speaker: str
    # 坐标格式为 [x_min, y_min, x_max, y_max]
    coordinates: Tuple[int, int, int, int]
    # 置信度，可选字段
    confidence: float = 1.0
