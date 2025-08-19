# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 16:35
"""
import argparse
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.audio_processor.processor import AudioProcessor
from video_tran.config import load_config


def main():
    """
    音频处理模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="音频处理模块：提取和分离音频。")
    parser.add_argument("--video-path", required=True, help="输入视频文件的路径。")
    parser.add_argument("--output-audio-path", required=True, help="提取出的原始音频的保存路径。")
    parser.add_argument("--output-vocals-path", required=True, help="分离出的人声音频的保存路径。")
    parser.add_argument("--output-background-path", required=True, help="分离出的背景声的保存路径。")
    parser.add_argument("--config-path", default="configs/config.yaml", help="配置文件的路径。")

    args = parser.parse_args()

    # 加载配置
    # 注意：在实际运行前，需要将 config.yaml.template 复制为 config.yaml
    config = load_config(args.config_path)
    if not config:
        print(f"无法加载配置文件: {args.config_path}")
        sys.exit(1)

    processor = AudioProcessor(config)

    # 1. 提取音频
    success = processor.extract_audio(args.video_path, args.output_audio_path)
    if not success:
        print("提取音频失败，流程终止。")
        sys.exit(1)

    # 2. 分离人声
    success = processor.separate_vocals(
        args.output_audio_path,
        args.output_vocals_path,
        args.output_background_path
    )
    if not success:
        print("分离人声失败，流程终止。")
        sys.exit(1)

    print("音频处理成功完成。")


if __name__ == "__main__":
    main()
