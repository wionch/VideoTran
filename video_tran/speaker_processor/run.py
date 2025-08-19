# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/08/16
"""
import argparse
import os
import sys
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pydub import AudioSegment
from video_tran.transcriber.data_types import segments_from_json
from tqdm import tqdm

def process_speakers(input_json_path: str, input_audio_path: str, output_dir: str):
    """
    Processes the diarized audio to create a reference audio file for each speaker.

    Args:
        input_json_path (str): Path to the JSON file with diarized segments.
        input_audio_path (str): Path to the source audio file (vocals).
        output_dir (str): Directory to save the speaker reference audio files.
    """
    print("正在加载音频文件... (这可能需要一些时间)")
    try:
        vocal_audio = AudioSegment.from_wav(input_audio_path)
    except FileNotFoundError:
        print(f"错误: 音频文件未找到 at '{input_audio_path}'", file=sys.stderr)
        sys.exit(1)

    segments = segments_from_json(input_json_path)
    if not segments:
        print(f"错误: 未能从 '{input_json_path}' 加载或解析任何片段。", file=sys.stderr)
        sys.exit(1)

    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for segment in segments:
        if segment.speaker:
            speaker_segments[segment.speaker].append(segment)

    if not speaker_segments:
        print("警告: JSON文件中未找到任何带有说话人标签的片段。无法生成参考音。", file=sys.stderr)
        return

    print(f"找到了 {len(speaker_segments)} 个说话人。正在为每个说话人生成参考音频...")

    # Create a reference audio for each speaker
    for speaker, speaker_segs in tqdm(speaker_segments.items(), desc="处理说话人"):
        # Concatenate all audio segments for the current speaker
        combined_audio = AudioSegment.empty()
        for seg in speaker_segs:
            # pydub uses milliseconds
            start_ms = int(seg.startTime * 1000)
            end_ms = int(seg.endTime * 1000)
            audio_slice = vocal_audio[start_ms:end_ms]
            combined_audio += audio_slice

        # Export the combined audio to a file
        output_path = os.path.join(output_dir, f"{speaker}.wav")
        print(f"正在导出说话人 {speaker} 的参考音频到: {output_path}")
        combined_audio.export(output_path, format="wav")

    print("\n所有说话人的参考音频已成功生成。")


def main():
    """
    Command-line entry point for the speaker processor module.
    """
    parser = argparse.ArgumentParser(description="为每个说话人生成参考音频。 সন")
    parser.add_argument("--input-json", required=True, help="包含说话人信息的转录JSON文件路径。 সন")
    parser.add_argument("--input-audio", required=True, help="原始人声音频文件路径 (vocals.wav)。 সন")
    parser.add_argument("--output-dir", required=True, help="用于保存每个说话人参考音频的目录。 সন")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    process_speakers(args.input_json, args.input_audio, args.output_dir)


if __name__ == "__main__":
    main()
