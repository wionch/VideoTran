# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 17:00
"""
import argparse
import json
import sys
import os
from typing import List

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.utils.shell_utils import run_command
# Segment class now has an optional 'speaker' field
from video_tran.transcriber.data_types import Segment, segments_to_json
from video_tran.config import load_config


def parse_whisperx_json(json_path: str) -> List[Segment]:
    """
    解析 whisperX 的 JSON 输出文件，现在支持说话人信息。

    Args:
        json_path (str): whisperX 生成的 JSON 文件的路径。

    Returns:
        List[Segment]: Segment 对象列表。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        # whisperX 在diarize模式下，会将speaker信息添加到每个word中，我们需要从word中聚合
        # 或者，更简单的做法是直接使用最终的segments，它会有一个speaker字段
        for item in data.get('segments', []):
            start_time = item.get('start')
            end_time = item.get('end')
            text = item.get('text', '').strip()
            speaker = item.get('speaker') # 直接获取speaker字段

            if start_time is not None and end_time is not None and text:
                segments.append(Segment(start=start_time, end=end_time, text=text, speaker=speaker))
        
        return segments
    except FileNotFoundError:
        print(f"错误: whisperX JSON 文件未找到 at '{json_path}'")
        return []
    except Exception as e:
        print(f"解析 whisperX JSON 文件时出错: {e}")
        return []


def main():
    """
    语音转录模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="使用 whisperX 转录音频。")
    parser.add_argument("--input-audio", required=True, help="要转录的输入音频文件的路径。")
    parser.add_argument("--lang", required=True, help="音频的语言代码 (例如, 'zh')。")
    parser.add_argument("--output-json", required=True, help="输出的转录结果 JSON 文件的路径。")
    parser.add_argument("--config-path", required=True, help="配置文件的路径。")
    # Add the diarize flag
    parser.add_argument("--diarize", action="store_true", help="执行说话人识别。")

    args = parser.parse_args()

    config = load_config(args.config_path)
    if not config:
        print(f"错误: 无法加载配置文件 at '{args.config_path}'", file=sys.stderr)
        sys.exit(1)

    t_config = config.get('transcriber', {})
    model = t_config.get('model', 'large-v2')
    batch_size = t_config.get('batch_size', 16)
    compute_type = t_config.get('compute_type', 'float16')

    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)

    # 构建 whisperX 命令
    command = (
        f'whisperx "{args.input_audio}" '
        f'--model {model} '
        f'--language {args.lang} '
        f'--output_format json '
        f'--output_dir "{output_dir}" '
        f'--batch_size {batch_size} '
        f'--compute_type {compute_type}'
    )

    # 如果启用了说话人识别，添加相应参数
    if args.diarize:
        # Diarization requires alignment model
        command += ' --align_model WAV2VEC2_ASR_LARGE_LV60K_960H'
        command += ' --diarize'
        # Check for Hugging Face token
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token:
            command += f' --hf_token {hf_token}'
        else:
            print("警告: 未找到 HUGGING_FACE_TOKEN 环境变量。如果 pyannote/speaker-diarization 模型需要认证，执行可能会失败。")


    print(f"执行 whisperX 命令: {command}")
    success, stdout, stderr = run_command(command)

    if not success:
        print("whisperX 执行失败。", file=sys.stderr)
        print("STDOUT:", stdout, file=sys.stderr)
        print("STDERR:", stderr, file=sys.stderr)
        sys.exit(1)

    input_basename = os.path.splitext(os.path.basename(args.input_audio))[0]
    whisperx_json_path = os.path.join(output_dir, f"{input_basename}.json")

    if not os.path.exists(whisperx_json_path):
        print(f"错误: 未找到 whisperX 的输出文件: {whisperx_json_path}", file=sys.stderr)
        sys.exit(1)

    segments = parse_whisperx_json(whisperx_json_path)
    if not segments:
        print("未能从 whisperX 的输出中解析出任何语音片段。")
        sys.exit(1)

    segments_to_json(segments, args.output_json)

    print(f"转录完成，结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
