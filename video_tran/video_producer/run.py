# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 18:40
"""
import argparse
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.utils.shell_utils import run_command
from video_tran.video_producer.srt_utils import create_srt_file
from video_tran.transcriber.data_types import segments_from_json

def main():
    """
    视频生成模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="视频生成模块：合并音轨并生成最终视频。")
    parser.add_argument("--original-video", required=True, help="原始视频文件的路径。")
    parser.add_argument("--dubbed-audio", required=True, help="新的配音音轨（只有人声）的路径。")
    parser.add_argument("--bg-audio", required=True, help="原始的背景音轨的路径。")
    parser.add_argument("--segments-json", required=True, help="包含翻译后文本的 segments JSON 文件路径，用于生成字幕。")
    parser.add_argument("--output-video", required=True, help="最终输出视频的路径。")
    parser.add_argument("--output-srt", required=True, help="最终输出 SRT 字幕的路径。")
    parser.add_argument("--temp-dir", required=True, help="存放临时文件的目录路径。")
    parser.add_argument("--normalize-volume", action="store_true", help="启用音量均衡，让人声更清晰。")

    args = parser.parse_args()
    
    os.makedirs(args.temp_dir, exist_ok=True)

    dubbed_audio_path = args.dubbed_audio

    # --- 音量均衡逻辑 ---
    if args.normalize_volume:
        print("启用音量均衡...")
        try:
            from pydub import AudioSegment
            
            dubbed_audio = AudioSegment.from_file(dubbed_audio_path)
            bg_audio = AudioSegment.from_file(args.bg_audio)

            # 目标：让配音的平均音量比背景音高6dB
            target_dbfs_diff = 6
            
            # 计算音量差异
            dbfs_diff = bg_audio.dBFS - dubbed_audio.dBFS
            
            # 计算需要施加的增益
            gain = dbfs_diff + target_dbfs_diff
            
            print(f"背景音量: {bg_audio.dBFS:.2f} dBFS, 配音音量: {dubbed_audio.dBFS:.2f} dBFS")
            print(f"将对配音施加 {gain:.2f} dB 增益。")

            normalized_dubbed_audio = dubbed_audio.apply_gain(gain)
            
            # 导出音量均衡后的配音
            normalized_dub_path = os.path.join(args.temp_dir, "normalized_dub.wav")
            normalized_dubbed_audio.export(normalized_dub_path, format="wav")
            
            # 更新后续步骤要使用的配音文件路径
            dubbed_audio_path = normalized_dub_path

        except Exception as e:
            print(f"音量均衡失败: {e}。将使用原始音量进行合并。")
            dubbed_audio_path = args.dubbed_audio

    # 1. 合并配音和背景音
    final_audio_path = os.path.join(args.temp_dir, "final_audio.wav")
    print("正在合并配音音轨和背景音轨...")
    command_mix = (
        f'ffmpeg -i "{dubbed_audio_path}" -i "{args.bg_audio}" '
        f'-filter_complex "[0:a][1:a]amix=inputs=2:duration=longest" '
        f'"{final_audio_path}" -y'
    )
    success, stdout, stderr = run_command(command_mix)
    if not success:
        print("合并音轨失败。")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # 2. 将最终音轨与原视频（的画面）合并
    print("正在将新的完整音轨合并到视频中...")
    command_merge = (
        f'ffmpeg -i "{args.original_video}" -i "{final_audio_path}" '
        f'-map 0:v:0 -map 1:a:0 -c:v copy -shortest '
        f'"{args.output_video}" -y'
    )
    success, stdout, stderr = run_command(command_merge)
    if not success:
        print("合并最终视频失败。")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # 3. 生成 SRT 字幕文件
    print("正在生成 SRT 字幕文件...")
    segments = segments_from_json(args.segments_json)
    if segments:
        create_srt_file(segments, args.output_srt)
    else:
        print(f"警告: 未能从 {args.segments_json} 加载片段，无法生成字幕。")

    print("视频生成成功完成！")
    print(f"最终视频: {args.output_video}")
    print(f"最终字幕: {args.output_srt}")


if __name__ == "__main__":
    main()

"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 18:40
"""
import argparse
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.utils.shell_utils import run_command
from video_tran.video_producer.srt_utils import create_srt_file
from video_tran.transcriber.data_types import segments_from_json


def main():
    """
    视频生成模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="视频生成模块：合并音轨并生成最终视频。")
    parser.add_argument("--original-video", required=True, help="原始视频文件的路径。")
    parser.add_argument("--dubbed-audio", required=True, help="新的配音音轨（只有人声）的路径。")
    parser.add_argument("--bg-audio", required=True, help="原始的背景音轨的路径。")
    parser.add_argument("--segments-json", required=True, help="包含翻译后文本的 segments JSON 文件路径，用于生成字幕。")
    parser.add_argument("--output-video", required=True, help="最终输出视频的路径。")
    parser.add_argument("--output-srt", required=True, help="最终输出 SRT 字幕的路径。")
    parser.add_argument("--temp-dir", required=True, help="存放临时文件的目录路径。")

    args = parser.parse_args()
    
    # 创建临时目录
    os.makedirs(args.temp_dir, exist_ok=True)

    # 1. 合并配音和背景音
    final_audio_path = os.path.join(args.temp_dir, "final_audio.wav")
    print("正在合并配音音轨和背景音轨...")
    # -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest" 表示将两个输入音频混合
    command_mix = (
        f'ffmpeg -i "{args.dubbed_audio}" -i "{args.bg_audio}" '
        f'-filter_complex "[0:a][1:a]amix=inputs=2:duration=longest" '
        f'"{final_audio_path}" -y'
    )
    success, stdout, stderr = run_command(command_mix)
    if not success:
        print("合并音轨失败。")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # 2. 将最终音轨与原视频（的画面）合并
    print("正在将新的完整音轨合并到视频中...")
    # -map 0:v:0 表示使用第一个输入（视频）的视频流
    # -map 1:a:0 表示使用第二个输入（音频）的音频流
    # -c:v copy 表示直接复制视频流，不重新编码，速度最快
    command_merge = (
        f'ffmpeg -i "{args.original_video}" -i "{final_audio_path}" '
        f'-map 0:v:0 -map 1:a:0 -c:v copy -shortest '
        f'"{args.output_video}" -y'
    )
    success, stdout, stderr = run_command(command_merge)
    if not success:
        print("合并最终视频失败。")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        sys.exit(1)

    # 3. 生成 SRT 字幕文件
    print("正在生成 SRT 字幕文件...")
    segments = segments_from_json(args.segments_json)
    if segments:
        create_srt_file(segments, args.output_srt)
    else:
        print(f"警告: 未能从 {args.segments_json} 加载片段，无法生成字幕。")

    print("视频生成成功完成！")
    print(f"最终视频: {args.output_video}")
    print(f"最终字幕: {args.output_srt}")


if __name__ == "__main__":
    main()
