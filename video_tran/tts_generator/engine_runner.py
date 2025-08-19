# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: engine_runner.py
@time: 2025/8/17
"""
import argparse
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.config import load_config
from video_tran.tts_generator.tts_wrapper import TTSWrapper

def main():
    parser = argparse.ArgumentParser(description="TTS Engine Runner")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", required=True, help="Path to reference audio file")
    parser.add_argument("--output-path", required=True, help="Path to save the output audio")
    parser.add_argument("--lang", required=True, help="Language of the text (e.g., 'ZH', 'EN')")
    parser.add_argument("--config-path", required=True, help="Path to the main config.yaml file")
    
    args = parser.parse_args()

    config = load_config(args.config_path)
    if not config:
        print(f"Error: Could not load config from {args.config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        tts_wrapper = TTSWrapper(config)
        result_path = tts_wrapper.generate(
            text=args.text,
            ref_audio_path=args.ref_audio,
            output_path=args.output_path,
            language=args.lang
        )
        
        if result_path and os.path.exists(result_path):
            print(f"SUCCESS:{result_path}")
            sys.exit(0)
        else:
            print(f"Error: TTS generation failed.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
