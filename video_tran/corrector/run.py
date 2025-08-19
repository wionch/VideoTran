# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 17:25
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
    异步主函数，用于并行处理字幕校正。
    """
    parser = argparse.ArgumentParser(description="LLM 字幕校正模块：使用 DeepSeek API 校正字幕文本。")
    parser.add_argument("--input-json", required=True, help="输入的 segments JSON 文件路径。")
    parser.add_argument("--output-json", required=True, help="输出的校正后 segments JSON 文件路径。")

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

    corrected_segments = [None] * len(segments)

    print("开始并行校正字幕...")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, segment in enumerate(segments):
            # 为每个请求创建一个异步任务
            task = client.correct_text_async(session, segment.text)
            tasks.append(task)

        # 使用 tqdm 显示异步任务的进度
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="校正进度"):
            results.append(await f)

    # 将结果放回原始的位置
    # 注意：asyncio.as_completed 不保证顺序，但如果我们需要保持顺序，
    # 我们可以通过将索引与任务关联起来解决，或者直接使用 asyncio.gather
    # 这里我们假设顺序无关紧要，或者通过其他方式重建
    # 为了简单和健壮，我们直接用返回的结果更新原始segment
    for i, corrected_text in enumerate(results):
        # 假设返回结果的顺序与tasks创建顺序一致 (gather保证，as_completed不保证)
        # 为了安全起见，我们还是用 gather
        pass # 下面的代码块将使用 gather

    # 使用 asyncio.gather 来保证结果的顺序
    async with aiohttp.ClientSession() as session:
        tasks = [client.correct_text_async(session, seg.text) for seg in segments]
        all_corrected_texts = await asyncio.gather(*tasks)

    for i, corrected_text in enumerate(all_corrected_texts):
        segments[i].text = corrected_text

    segments_to_json(segments, args.output_json)

    print(f"字幕校正完成，结果已保存到: {args.output_json}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main_async())
