# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 17:40
"""
import argparse
import sys
import os
import asyncio
import aiohttp
from tqdm import tqdm

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.utils.llm_client import DeepSeekClient
from video_tran.transcriber.data_types import segments_from_json, segments_to_json, Segment


async def main_async():
    """
    异步主函数，用于并行处理文本翻译。
    """
    parser = argparse.ArgumentParser(description="文本翻译模块：使用 DeepSeek API 翻译文本。")
    parser.add_argument("--input-json", required=True, help="输入的校正后 segments JSON 文件路径。")
    parser.add_argument("--output-json", required=True, help="输出的翻译后 segments JSON 文件路径。")
    parser.add_argument("--target-lang", required=True, help="目标翻译语言 (例如, 'English')。")

    args = parser.parse_args()

    segments = segments_from_json(args.input_json)
    if not segments:
        print(f"未能从 {args.input_json} 加载或解析出任何片段。")
        sys.exit(1)

    try:
        client = DeepSeekClient()
    except ValueError as e:
        print(f"初始化 DeepSeekClient 时出错: {e}")
        sys.exit(1)

    print(f"开始将文本并行翻译为 {args.target_lang}...")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for seg in segments:
            duration = float(seg.endTime) - float(seg.startTime)
            task = client.translate_text_async(session, seg.text, args.target_lang, duration)
            tasks.append(task)
        
        all_translated_texts = await asyncio.gather(*tasks)

    translated_segments = []
    for i, translated_text in enumerate(all_translated_texts):
        original_segment = segments[i]
        final_text = translated_text

        if not translated_text:
            final_text = original_segment.text
            print(f"警告: 未能翻译片段 (start: {original_segment.startTime}): '{original_segment.text}'。将使用原文作为回退。")

        translated_segments.append(
            Segment(id=original_segment.id, startTime=original_segment.startTime, endTime=original_segment.endTime, text=final_text, speaker=original_segment.speaker)
        )

    segments_to_json(translated_segments, args.output_json)

    print(f"文本翻译完成，结果已保存到: {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main_async())
