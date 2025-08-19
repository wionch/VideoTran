# remove_subtitle_v3.py

import argparse
import json
import yaml, time
import os
import sys
from typing import List

# --- 集成 video-subtitle-remover --- #
remover_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'video-subtitle-remover'))
sys.path.insert(0, remover_path)

from video_tran.subtitle_processor.frame_analyzer import FrameAnalyzer
from video_tran.utils.data_structures import PreciseSubtitle
from backend.main import SubtitleRemover

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="精准字幕移除工具 V3")
    parser.add_argument("-i", "--video_input", type=str, required=True, help="输入视频文件的路径")
    parser.add_argument("-o", "--output_path", type=str, required=False, help="处理后视频的输出路径")
    parser.add_argument("-s", "--subtitle_input", type=str, help="[分析模式] 初始ASR字幕JSON文件的路径")
    parser.add_argument("--use_existing_json", type=str, help="[移除模式] 提供已生成的精准字幕JSON文件路径，跳过分析，直接移除。")
    parser.add_argument("-c", "--config_path", type=str, default="configs/config.yaml", help="配置文件的路径")
    # 新增算法选择参数
    parser.add_argument("--remover_model", type=str, default='lama', choices=['lama', 'sttn'], help="选择用于字幕移除的修复算法: lama 或 sttn。")
    parser.add_argument("--only_convert", action="store_true", help="只执行字幕转换，不进行视频字幕移除。")
    parser.add_argument("--subtitle_area", type=int, nargs=2, help="[调试] 手动提供字幕的Y轴区间 (y_min, y_max)，跳过自动检测。")
    args = parser.parse_args()

    # 自定义参数验证
    if not args.only_convert and not args.output_path:
        parser.error("当不使用 --only_convert 参数时，必须提供 -o/--output_path。")

    if not args.subtitle_input and not args.use_existing_json:
        parser.error("必须提供 --subtitle_input (分析模式) 或 --use_existing_json (移除模式) 其中之一。")

    precise_subtitle_path = ""

    if args.subtitle_input:
        # --- 阶段一：分析并生成JSON ---
        print("--- 分析模式启动 ---")
        print(f"加载配置文件: {args.config_path}")
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 将顶级配置和remover_v3子配置合并，以兼容新旧配置格式
        analyzer_config = config.get('remover_v3', {})
        analyzer_config.update(config)

        print(f"读取初始字幕文件: {args.subtitle_input}")
        with open(args.subtitle_input, 'r', encoding='utf-8') as f:
            initial_subtitles = json.load(f)

        print("初始化帧分析器...")
        analyzer = FrameAnalyzer(analyzer_config)

        # Y轴区间确定
        if args.subtitle_area:
            import cv2
            cap = cv2.VideoCapture(args.video_input)
            if not cap.isOpened():
                raise IOError(f"无法打开视频 {args.video_input} 以获取宽度信息")
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            subtitle_area = (0, args.subtitle_area[0], video_width, args.subtitle_area[1])
            print(f"使用手动提供的Y轴区间: {subtitle_area}")
        else:
            print("未提供Y轴区间，开始自动检测...")
            try:
                y_axis_config = analyzer_config.get('y_axis_detection', {})
                sample_count = y_axis_config.get('sample_count', 20)
                subtitle_area = analyzer.determine_subtitle_area(args.video_input, initial_subtitles, sample_count=sample_count)
            except Exception as e:
                print(f"自动检测Y轴区域失败: {e}")
                return # 失败则中断
        
        analyzer.set_subtitle_area(subtitle_area)

        all_precise_subtitles: List[PreciseSubtitle] = []
        print("开始逐段进行精准字幕分析...")

        def time_str_to_seconds(time_str):
            time_str = time_str.replace(',', '.')
            parts = time_str.split(':')
            seconds = 0
            if len(parts) == 3:
                seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                seconds = float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 1:
                seconds = float(parts[0])
            return seconds

        for sub in initial_subtitles:
            try:
                start_seconds = time_str_to_seconds(sub['startTime'])
                end_seconds = time_str_to_seconds(sub['endTime'])
            except (ValueError, IndexError):
                print(f"警告: 无法解析时间戳 '{sub['startTime']}' 或 '{sub['endTime']}'. 跳过此片段。")
                continue
            
            time_range = (start_seconds, end_seconds)
            speaker = sub.get('speaker', 'SPEAKER_UNKNOWN')
            
            print(f"\n--- 正在处理分段 ID {sub.get('id', 'N/A')} ({sub['startTime']} -> {sub['endTime']}) ---")
            precise_subs = analyzer.analyze_time_range(args.video_input, time_range, speaker)
            all_precise_subtitles.extend(precise_subs)
        
        print("\n--- 所有片段分析完成。开始全局合并... ---")
        merged_subtitles = analyzer._merge_duplicate_subtitles(all_precise_subtitles)
        print(f"全局合并完成，字幕数从 {len(all_precise_subtitles)} 条减少到 {len(merged_subtitles)} 条。")

        final_subtitles_for_json = []
        for i, sub in enumerate(merged_subtitles):
            x1, y1, x2, y2 = sub.coordinates
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            final_subtitles_for_json.append({
                'id': i + 1,
                'startTime': sub.start_time,
                'endTime': sub.end_time,
                'text': sub.text,
                'speaker': sub.speaker,
                'bbox': bbox
            })
        print(f"执行最终重新编号和格式转换，共 {len(final_subtitles_for_json)} 条字幕。")

        video_basename = os.path.basename(args.video_input)
        video_name, _ = os.path.splitext(video_basename)
        precise_subtitle_path = os.path.join(os.path.dirname(args.video_input), f"{video_name}.precise.json")

        print(f"保存高精度字幕到: {precise_subtitle_path}")
        with open(precise_subtitle_path, 'w', encoding='utf-8') as f:
            json.dump(final_subtitles_for_json, f, ensure_ascii=False, indent=4)
        if args.only_convert:
            print("已完成字幕转换，根据 --only_convert 参数，脚本将退出。")
            return # Exit the main function
    
    elif args.use_existing_json:
        # --- 直接使用现有JSON的模式 ---
        print("--- 移除模式启动 ---")
        if not os.path.exists(args.use_existing_json):
            print(f"错误: 指定的JSON文件不存在: {args.use_existing_json}")
            return
        precise_subtitle_path = args.use_existing_json
        print(f"使用已存在的精准字幕文件: {precise_subtitle_path}")

    # --- 阶段二：调用工具执行视频清理 ---
    print(f"\n--- 开始执行字幕移除，使用算法: {args.remover_model.upper()} ---")
    
    with open(precise_subtitle_path, 'r', encoding='utf-8') as f:
        subs_to_remove = json.load(f)
    
    def seconds_to_time_str(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds * 1000) % 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    for sub in subs_to_remove:
        sub['startTime'] = seconds_to_time_str(sub['startTime'])
        sub['endTime'] = seconds_to_time_str(sub['endTime'])
    
    temp_json_path = precise_subtitle_path + ".tmp.json"
    with open(temp_json_path, 'w', encoding='utf-8') as f:
        json.dump(subs_to_remove, f, ensure_ascii=False, indent=4)

    remover_instance = SubtitleRemover(
        video_path=args.video_input,
        subtitle_json_path=temp_json_path,
        output_path=args.output_path,
        remover_model=args.remover_model # 传递选择的算法
    )
    remover_instance.run()

    os.remove(temp_json_path)

    print(f"\n处理完成！最终视频已生成于 {args.output_path}")

if __name__ == "__main__":
    script_start_time = time.time()
    try:
        main()
    finally:
        print(f"\n脚本总执行时间: {time.time() - script_start_time:.2f} 秒。")
