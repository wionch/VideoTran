# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: audio_utils.py
@time: 2025/8/15 18:00
"""
from pydub import AudioSegment
from pydub.effects import speedup
from typing import List, Tuple

def adjust_silence(audio_path: str, speech_timestamps: List[Tuple[int, int]], target_duration_ms: int) -> AudioSegment:
    """
    通过调整静音部分的时长来精确控制音频总长，不改变语音部分的语速。

    Args:
        audio_path (str): 输入音频文件的路径。
        speech_timestamps (List[Tuple[int, int]]): VAD检测到的语音活动时间戳。
        target_duration_ms (int): 目标音频总时长（毫秒）。

    Returns:
        AudioSegment: 调整后但尚未导出的 Pydub 音频对象。如果无法调整，则返回 None。
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"无法从 {audio_path} 加载音频: {e}")
        return None

    total_speech_duration = sum(end - start for start, end in speech_timestamps)
    
    if total_speech_duration > target_duration_ms:
        # 语音部分已经超过目标总长，此方法无法处理
        return None

    # 提取所有的语音片段
    speech_segments = [audio[start:end] for start, end in speech_timestamps]

    # 计算需要填充的总静音时长
    total_silence_needed = target_duration_ms - total_speech_duration
    
    # 计算静音段的数量（语音段之间 + 开头 + 结尾）
    num_silence_segments = len(speech_segments) + 1
    if num_silence_segments == 0:
        return AudioSegment.silent(duration=target_duration_ms)

    # 平均分配静音时长
    silence_duration_per_segment = total_silence_needed / num_silence_segments
    
    # 创建一个空的音频作为画布
    final_audio = AudioSegment.empty()

    # 依次拼接静音和语音
    for speech_segment in speech_segments:
        final_audio += AudioSegment.silent(duration=silence_duration_per_segment)
        final_audio += speech_segment
    
    # 在末尾添加最后一个静音段
    final_audio += AudioSegment.silent(duration=silence_duration_per_segment)

    return final_audio


def force_speed(audio_path: str, target_duration_ms: int) -> AudioSegment:
    """
    通过强制调整整个音频的语速来匹配目标时长。
    这是一个备用方案，可能会影响音质和自然度。

    Args:
        audio_path (str): 输入音频文件的路径。
        target_duration_ms (int): 目标音频总时长（毫秒）。

    Returns:
        AudioSegment: 调整后但尚未导出的 Pydub 音频对象。
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"无法从 {audio_path} 加载音频: {e}")
        return None
        
    current_duration_ms = len(audio)
    if current_duration_ms == 0:
        return audio

    speed_ratio = current_duration_ms / target_duration_ms
    
    # pydub.effects.speedup 的 playback_speed 参数是 > 1 加速, < 1 减速
    # 但它实现的是移除或复制帧，所以我们需要用 speedup
    if speed_ratio > 1.0: # 需要加速
        return speedup(audio, playback_speed=speed_ratio)
    elif speed_ratio < 1.0: # 需要减速
        # pydub 没有直接的减速函数，但可以通过加速一个更慢的版本来模拟
        # 这里我们用一个简化的方法，直接返回，因为减速效果通常不好
        # 一个更复杂的实现是使用 sox 或其他库
        print(f"警告: 请求减速 (ratio: {speed_ratio})，Pydub 的 speedup 不支持。返回原始音频。")
        return audio
    else:
        return audio

def slice_audio(audio_path: str, start_ms: int, end_ms: int, output_path: str) -> bool:
    """
    切割音频文件。

    Args:
        audio_path (str): 输入音频文件路径。
        start_ms (int): 开始时间（毫秒）。
        end_ms (int): 结束时间（毫秒）。
        output_path (str): 切割后音频的保存路径。

    Returns:
        bool: 是否成功。
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
        sliced_audio = audio[start_ms:end_ms]
        sliced_audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"切割音频时出错: {e}")
        return False
