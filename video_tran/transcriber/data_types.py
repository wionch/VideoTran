# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: data_types.py
@time: 2025/8/15 16:50
"""
from dataclasses import dataclass, asdict
import json
from typing import List, Optional


@dataclass
class Segment:
    """
    代表一个语音片段的数据结构。
    """
    id: int
    startTime: float
    endTime: float
    text: str
    speaker: Optional[str] = None


def srt_time_to_seconds(time_str):
    """
    将SRT时间格式 (HH:MM:SS,ms 或 MM:SS,ms) 转换为秒 (浮点数)。
    支持字符串和数字类型的输入。
    """
    # 如果已经是数字类型，直接返回
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    # 如果是字符串但不包含逗号，尝试直接转换为浮点数
    if isinstance(time_str, str) and ',' not in time_str:
        try:
            return float(time_str)
        except ValueError:
            raise ValueError(f"无效的时间格式: {time_str}")
    
    # 处理SRT格式时间字符串
    if not isinstance(time_str, str):
        raise ValueError(f"无效的时间格式类型: {type(time_str)}")
        
    # 分割时间部分和毫秒部分
    time_parts_str, ms_str = time_str.split(',')
    milliseconds = int(ms_str)

    # 分割时间部分的小时、分钟、秒
    time_components = [int(x) for x in time_parts_str.split(':')]

    hours = 0
    minutes = 0
    seconds = 0

    # 根据组件数量确定格式
    if len(time_components) == 3: # HH:MM:SS
        hours, minutes, seconds = time_components
    elif len(time_components) == 2: # MM:SS
        minutes, seconds = time_components
    elif len(time_components) == 1: # SS (不常见，但为了健壮性考虑)
        seconds = time_components[0]
    else:
        raise ValueError(f"无效的SRT时间格式 (冒号分隔部分过多或过少): {time_str}")

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def segments_to_json(segments: List[Segment], file_path: str):
    """
    将Segment列表序列化为JSON文件。

    Args:
        segments (List[Segment]): Segment对象列表。
        file_path (str): 输出的JSON文件路径。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in segments], f, ensure_ascii=False, indent=4)


def segments_from_json(file_path: str) -> List[Segment]:
    """
    从JSON文件反序列化为Segment列表。
    自动处理时间格式转换（SRT格式字符串转换为秒数）。

    Args:
        file_path (str): 输入的JSON文件路径。

    Returns:
        List[Segment]: Segment对象列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = []
    for item in data:
        # 转换时间格式（如果需要的话）
        if 'startTime' in item:
            item['startTime'] = srt_time_to_seconds(item['startTime'])
        if 'endTime' in item:
            item['endTime'] = srt_time_to_seconds(item['endTime'])
        
        segments.append(Segment(**item))
    
    return segments
