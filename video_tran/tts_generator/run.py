# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 18:20
@description: [REFACTORED V3] 高效的语音合成与时长校准模块。
             此版本实现了音频加速（替代截断）和分说话人音量精准匹配功能。
"""
import argparse
import sys
import os
from tqdm import tqdm
from pydub import AudioSegment
import json

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.config import load_config
from video_tran.transcriber.data_types import segments_from_json
from video_tran.utils.audio_utils import slice_audio
from video_tran.tts_generator.tts_wrapper import TTSWrapper

def preload_speaker_references(segments, config, ref_audio_dir):
    """
    为所有说话人准备参考音频路径。
    """
    print("准备说话人参考音频路径...")
    speaker_ref_paths = {}
    unique_speakers = sorted(list(set(seg.speaker for seg in segments if seg.speaker)))

    for speaker in unique_speakers:
        ref_path = os.path.join(ref_audio_dir, f"{speaker}.wav")
        if os.path.exists(ref_path):
            speaker_ref_paths[speaker] = ref_path
        else:
            print(f"警告: 说话人 {speaker} 的参考音频不存在于 '{ref_path}'")
            speaker_ref_paths[speaker] = None

    print("说话人参考路径准备完成。")
    return speaker_ref_paths

def preload_speaker_volumes(speaker_ref_paths: dict) -> dict:
    """
    预加载并缓存所有参考音频的响度（dBFS）。
    """
    print("正在预加载参考音音量...")
    speaker_volumes = {}
    for speaker, path in speaker_ref_paths.items():
        if path and os.path.exists(path):
            try:
                audio = AudioSegment.from_file(path)
                speaker_volumes[speaker] = audio.dBFS
            except Exception as e:
                print(f"警告: 无法读取或分析参考音 {path} 的音量: {e}")
    print("参考音音量预加载完成。")
    return speaker_volumes

def main():
    """
    语音合成与时长校准模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="高效的语音合成与时长校准模块。" )
    parser.add_argument("--input-json", required=True, help="输入的翻译后 segments JSON 文件路径。" )
    parser.add_argument("--ref-audio", required=True, help="用于提取音色的原始音频文件或包含参考音的目录。" )
    parser.add_argument("--output-audio", required=True, help="最终生成的完整配音音轨的路径。" )
    parser.add_argument("--temp-dir", required=True, help="存放临时文件的目录路径。" )
    parser.add_argument("--target-lang", required=True, help="TTS的目标语言, e.g., 'EN' or 'ZH'." )
    parser.add_argument("--config-path", default="configs/config.yaml", help="配置文件的路径。" )
    parser.add_argument("--use-speaker-ref", action="store_true", help="若设置，则ref-audio被视为包含说话人参考音频的目录。" )
    parser.add_argument("--align-duration", action="store_true", help="若设置，则启用时长对齐（加速或填充静音）。" )
    parser.add_argument("--target-dbfs", type=float, help="手动设定目标音量(dBFS)来覆盖自动匹配。" )

    args = parser.parse_args()

    config = load_config(args.config_path)
    if not config:
        print(f"无法加载配置文件: {args.config_path}")
        sys.exit(1)

    segments = segments_from_json(args.input_json)
    if not segments:
        print(f"未能从 {args.input_json} 加载或解析出任何片段。" )
        sys.exit(1)

    os.makedirs(args.temp_dir, exist_ok=True)

    speaker_ref_paths = {}
    speaker_volumes = {}

    if args.use_speaker_ref:
        speaker_ref_paths = preload_speaker_references(segments, config, args.ref_audio)
        # 仅在用户未手动指定 master volume 时，才预加载音量用于匹配
        if args.target_dbfs is None:
            speaker_volumes = preload_speaker_volumes(speaker_ref_paths)

    print("正在初始化TTS引擎... (可能需要一些时间)")
    try:
        tts_wrapper = TTSWrapper(config)
        print("TTS引擎初始化成功。" )
    except Exception as e:
        print(f"致命错误: 无法初始化TTS引擎: {e}")
        sys.exit(1)

    final_dub_track = AudioSegment.silent(duration=0)
    last_segment_end_time = 0.0

    print("开始生成、校准并拼接配音片段...")
    for i, segment in enumerate(tqdm(segments, desc="语音生成与拼接进度")):
        ref_path = None
        if args.use_speaker_ref:
            if segment.speaker and segment.speaker in speaker_ref_paths:
                ref_path = speaker_ref_paths[segment.speaker]
                if not ref_path or not os.path.exists(ref_path):
                    print(f"警告: 片段 {i} 的说话人 '{segment.speaker}' 参考音频路径无效，跳过。" )
                    continue
            else:
                print(f"警告: 片段 {i} 的说话人 '{segment.speaker}' 未知或未预加载，跳过。" )
                continue
        else:
            ref_path = os.path.join(args.temp_dir, f"ref_slice_{i}.wav")
            slice_audio(args.ref_audio, segment.startTime * 1000, segment.endTime * 1000, ref_path)
            if not os.path.exists(ref_path):
                print(f"警告: 未能切割参考音片段 {i}，跳过。" )
                continue

        gap_duration = (segment.startTime - last_segment_end_time) * 1000
        if gap_duration > 0:
            final_dub_track += AudioSegment.silent(duration=gap_duration)

        temp_dub_path = os.path.join(args.temp_dir, f"temp_dub_{i}.wav")

        generated_path = tts_wrapper.generate(
            text=segment.text,
            ref_audio_path=ref_path,
            output_path=temp_dub_path,
            language=args.target_lang
        )

        processed_audio = None
        if generated_path and os.path.exists(generated_path):
            generated_audio = AudioSegment.from_file(generated_path)
            target_duration_ms = (segment.endTime - segment.startTime) * 1000
            generated_duration_ms = len(generated_audio)

            # 1. 时长对齐：加速或填充
            if args.align_duration:
                if generated_duration_ms > target_duration_ms and target_duration_ms > 0:
                    speed_factor = generated_duration_ms / target_duration_ms
                    print(f"\n片段 {i} 过长 ({generated_duration_ms}ms > {target_duration_ms}ms)，将加速 {speed_factor:.2f}x")
                    aligned_audio = generated_audio.speedup(playback_speed=speed_factor)
                else:
                    silence = AudioSegment.silent(duration=target_duration_ms - generated_duration_ms)
                    aligned_audio = generated_audio + silence
            else:
                aligned_audio = generated_audio

            # 2. 音量标准化
            target_db = None
            if args.target_dbfs is not None:
                target_db = args.target_dbfs
            elif segment.speaker in speaker_volumes:
                target_db = speaker_volumes[segment.speaker]

            if target_db is not None:
                change_in_dbfs = target_db - aligned_audio.dBFS
                processed_audio = aligned_audio.apply_gain(change_in_dbfs)
            else:
                processed_audio = aligned_audio
        else:
            print(f"\n警告: 未能为片段 {i} 生成有效的语音文件，将填充等长静音。" )
            target_duration_ms = (segment.endTime - segment.startTime) * 1000
            processed_audio = AudioSegment.silent(duration=target_duration_ms)

        final_dub_track += processed_audio
        last_segment_end_time = segment.endTime

    print("所有片段处理完毕，正在导出...")

    # 仅在未使用说话人参考且未手动设定音量时，才对整个音轨进行默认的最终标准化
    if not args.use_speaker_ref and args.target_dbfs is None:
        print("未指定说话人参考且未手动设定音量，对整个音轨进行默认标准化..." )
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - final_dub_track.dBFS
        final_dub_track = final_dub_track.apply_gain(change_in_dBFS)

    final_dub_track.export(args.output_audio, format="wav")

    print(f"语音合成与校准全部完成，最终音轨已保存到: {args.output_audio}")

if __name__ == "__main__":
    main()
