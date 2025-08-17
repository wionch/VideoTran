import json
import re
import os

def srt_time_to_seconds(time_str):
    """
    将SRT时间格式 (HH:MM:SS,ms 或 MM:SS,ms) 转换为秒 (浮点数)。
    """
    # 确保字符串包含毫秒的逗号
    if ',' not in time_str:
        raise ValueError(f"无效的SRT时间格式 (缺少毫秒逗号): {time_str}")

    # 分割时间部分和毫秒部分
    time_parts_str, ms_str = time_str.split(',')
    milliseconds = int(ms_str)

    # 分割时间部分的小时、分钟、秒
    time_components = [int(x) for x in time_parts_str.split(':')]

    hours = 0
    minutes = 0
    seconds = 0

    # 根据组件数量确定格式
    if len(time_components) == 3: # HH:MM:SS
        hours, minutes, seconds = time_components
    elif len(time_components) == 2: # MM:SS
        minutes, seconds = time_components
    elif len(time_components) == 1: # SS (不常见，但为了健壮性考虑)
        seconds = time_components[0]
    else:
        raise ValueError(f"无效的SRT时间格式 (冒号分隔部分过多或过少): {time_str}")

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def convert_srt_to_videotran_json(srt_file_path, output_json_path):
    """
    将SRT文件转换为VideoTran转录模块所需的JSON格式。

    Args:
        srt_file_path (str): 输入SRT文件的路径。
        output_json_path (str): 输出JSON文件的保存路径。
    """
    segments = []
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: SRT文件未找到: {srt_file_path}")
        return
    except Exception as e:
        print(f"读取SRT文件时出错: {e}")
        return

    # 将SRT内容分割成独立的字幕块
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2: # 至少需要序号和时间行
            continue

        # 第一行通常是序号，第二行是时间，其余是文本
        time_line = lines[1]
        text_lines = lines[2:]

        time_parts = time_line.split(' --> ')
        if len(time_parts) != 2:
            print(f"警告: 跳过格式不正确的时间行: {time_line}")
            continue

        try:
            start_time = srt_time_to_seconds(time_parts[0].strip())
            end_time = srt_time_to_seconds(time_parts[1].strip())
            text = ' '.join(text_lines).strip()

            if text: # 只添加有实际文本的片段
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": None # SRT通常不包含说话人信息，因此设置为None
                })
        except ValueError as e:
            print(f"警告: 跳过因时间解析错误而导致的字幕块: {e} 在块中:\n{block}")
            continue
        except Exception as e:
            print(f"警告: 处理字幕块时发生未知错误: {e} 在块中:\n{block}")
            continue

    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)
        print(f"转换完成。JSON文件已保存到: {output_json_path}")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")

# --- 如何使用此脚本 ---
# 1. 将上述代码保存为一个Python文件，例如：convert_srt.py
# 2. 在终端中运行：
#    python convert_srt.py
# 转换执行环境 conda activate videotran_env
# 示例用法（请根据您的实际路径修改）
srt_file = "D:\\Python\\Project\\VideoTran\\videos\\223.srt"
output_json = "D:\\Python\\Project\\VideoTran\\videos\\223.json" # 建议保存到output目录
convert_srt_to_videotran_json(srt_file, output_json)
