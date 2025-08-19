# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: srt_utils.py
@time: 2025/8/15 18:30
"""
from typing import List
from video_tran.transcriber.data_types import Segment
import datetime

def to_srt_time_format(total_seconds: float) -> str:
    """
    将秒数转换为 SRT 字幕的时间格式 (HH:MM:SS,ms)。
    """
    # 将 total_seconds 转换为 timedelta 对象
    td = datetime.timedelta(seconds=total_seconds)
    
    # 从 timedelta 对象中提取时、分、秒和微秒
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"


def create_srt_file(segments: List[Segment], output_path: str):
    """
    根据 Segment 列表创建 SRT 字幕文件。

    Args:
        segments (List[Segment]): 包含翻译后文本和时间戳的 Segment 列表。
        output_path (str): 输出的 .srt 文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = to_srt_time_format(segment.start)
            end_time = to_srt_time_format(segment.end)
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment.text}\n\n")
