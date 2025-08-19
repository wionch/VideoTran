# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: vad_utils.py
@time: 2025/8/15 17:50
"""
import webrtcvad
from pydub import AudioSegment
from typing import List, Tuple


def get_speech_timestamps(audio_path: str, aggressiveness: int = 1) -> List[Tuple[int, int]]:
    """
    使用 VAD (Voice Activity Detection) 分析音频，返回所有语音活动的时间戳。

    Args:
        audio_path (str): 输入音频文件的路径。
        aggressiveness (int): VAD 的敏感度，范围从 0 (最不敏感) 到 3 (最敏感)。

    Returns:
        List[Tuple[int, int]]: 一个包含 (开始时间, 结束时间) 元组的列表，单位为毫秒。
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"无法从 {audio_path} 加载音频: {e}")
        return []

    # VAD 要求是 16-bit 单声道 PCM，采样率 8k, 16k, 32k, or 48k
    # 我们将音频转换为 VAD 兼容的格式
    if audio.sample_width != 2:
        audio = audio.set_sample_width(2)
    if audio.channels != 1:
        audio = audio.set_channels(1)
    if audio.frame_rate not in [8000, 16000, 32000, 48000]:
        # 选择一个接近的、受支持的采样率
        if audio.frame_rate > 32000:
            audio = audio.set_frame_rate(48000)
        elif audio.frame_rate > 16000:
            audio = audio.set_frame_rate(32000)
        else:
            audio = audio.set_frame_rate(16000)

    vad = webrtcvad.Vad(aggressiveness)

    # VAD 处理的帧必须是 10, 20, or 30 ms long
    frame_duration_ms = 30
    samples_per_frame = int(audio.frame_rate * (frame_duration_ms / 1000.0))
    bytes_per_frame = samples_per_frame * audio.sample_width

    speech_timestamps = []
    is_speech = False
    start_time = 0

    for i in range(0, len(audio.raw_data), bytes_per_frame):
        frame = audio.raw_data[i:i+bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break

        current_time_ms = int((i / bytes_per_frame) * frame_duration_ms)

        try:
            if vad.is_speech(frame, audio.frame_rate):
                if not is_speech:
                    start_time = current_time_ms
                    is_speech = True
            else:
                if is_speech:
                    speech_timestamps.append((start_time, current_time_ms))
                    is_speech = False
        except webrtcvad.Error as e:
            print(f"VAD error processing frame at {current_time_ms}ms. Frame length (bytes): {len(frame)}, Sample rate: {audio.frame_rate}. Error: {e}")
            continue

    # 如果音频以语音结束
    if is_speech:
        speech_timestamps.append((start_time, len(audio)))

    return speech_timestamps
