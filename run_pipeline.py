# run_pipeline.py
import argparse
import os
import time
import json
from pipeline.pipeline import VideoSubtitleExtractorPipeline

def main():
    """
    新的、简化的流水线执行入口。
    """
    parser = argparse.ArgumentParser(description="工业级视频字幕提取流水线")
    parser.add_argument("-i", "--video_input", type=str, required=True, help="输入视频文件的路径")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="字幕JSON文件的输出目录。如果未提供，则默认保存在视频文件旁边。")
    parser.add_argument("--config", type=str, default=None, help="自定义流水线配置文件的路径 (YAML格式)。")

    args = parser.parse_args()

    if not os.path.exists(args.video_input):
        print(f"错误: 输入的视频文件不存在: {args.video_input}")
        return

    print("--- 开始执行高性能字幕提取流水线 ---")
    
    # 1. 初始化流水线
    pipeline = VideoSubtitleExtractorPipeline(config_path=args.config)

    # 2. 执行流水线
    precise_subtitles = pipeline.run(args.video_input)

    # 3. 保存结果
    if precise_subtitles:
        video_basename = os.path.basename(args.video_input)
        video_name, _ = os.path.splitext(video_basename)
        
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.video_input)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 使用 .precise.json 后缀以保持兼容性
        output_path = os.path.join(output_dir, f"{video_name}.precise.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(precise_subtitles, f, ensure_ascii=False, indent=4)
        print(f"\n处理成功！高精度字幕已保存到: {os.path.abspath(output_path)}")
    else:
        print("\n处理完成，但未能从视频中提取到任何有效字幕。")

if __name__ == "__main__":
    script_start_time = time.time()
    try:
        main()
    finally:
        print(f"\n脚本总执行时间: {time.time() - script_start_time:.2f} 秒。")