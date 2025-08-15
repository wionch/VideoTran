# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: main.py
@time: 2025/8/15 19:00
"""
import argparse
from orchestrator import Orchestrator


def main():
    """
    应用程序的主入口点。
    """
    parser = argparse.ArgumentParser(description="自动化视频语音翻译工具。")
    parser.add_argument("-i", "--input-video", required=True, help="要处理的输入视频文件的路径。")
    parser.add_argument("-sl", "--source-language", required=True, help="视频的源语言代码 (例如, 'zh' 表示中文)。")
    parser.add_argument("-tl", "--target-language", required=True, help="要翻译成的目标语言 (例如, 'en' 表示英文)。")
    parser.add_argument("-c", "--config", default="configs/config.yaml", help="配置文件的路径。")

    args = parser.parse_args()

    try:
        orchestrator = Orchestrator(args.config)
        orchestrator.run(
            video_path=args.input_video,
            src_lang=args.source_language,
            target_lang=args.target_language
        )
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")


if __name__ == "__main__":
    main()
