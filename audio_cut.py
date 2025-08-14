"""
audio_cut.py

一个使用 FFmpeg 实现的高效音频截取脚本。

功能:
- 通过调用外部 FFmpeg 程序，可以从指定的音频文件中根据开始和结束时间快速截取片段。
- 使用 '-c copy' 参数，避免了音频的重新编码，实现了近乎无损的快速剪辑。

依赖:
- 用户系统环境中必须已安装 FFmpeg，并将其添加到了系统PATH中。
"""

import subprocess
import os

def cut_audio_ffmpeg(input_path, output_path, start_time, end_time):
    """
    使用 FFmpeg 从音频文件中截取一个片段。

    Args:
        input_path (str): 原始音频文件的路径。
        output_path (str): 截取后音频的保存路径。
        start_time (str or int): 截取的开始时间 (格式可以是 'HH:MM:SS' 或纯秒数)。
        end_time (str or int): 截取的结束时间 (格式可以是 'HH:MM:SS' 或纯秒数)。
    
    Returns:
        bool: 如果成功则返回 True，否则返回 False。
    """
    print(f"准备截取音频: {input_path}")
    print(f"开始时间: {start_time}, 结束时间: {end_time}")
    print(f"输出到: {output_path}")

    # 步骤 1: 检查并创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 步骤 2: 构建 FFmpeg 命令
    # -y 参数表示如果输出文件已存在，则无需询问直接覆盖
    # -c copy 表示直接复制编解码器，不进行重新编码，速度极快
    command = [
        'ffmpeg',
        '-i',
        input_path,
        '-ss',
        str(start_time),
        '-to',
        str(end_time),
        '-c',
        'copy',
        '-y',
        output_path
    ]

    # 步骤 3: 执行命令并进行异常处理
    try:
        print(f"\\n正在执行 FFmpeg 命令:\n{' '.join(command)}\\n")
        
        # 使用 subprocess.run 执行命令
        result = subprocess.run(
            command, 
            check=True,        # 如果ffmpeg返回非零退出码（表示错误），则会引发CalledProcessError
            capture_output=True, # 捕获标准输出和标准错误输出
            text=True          # 以文本形式解码输出
        )
        
        print("--- FFmpeg 输出 ---")
        print(result.stderr) # FFmpeg 通常将进度和摘要信息输出到 stderr
        print("---------------------")
        print(f"\\n成功！音频片段已保存到: {output_path}")
        return True

    except FileNotFoundError:
        print("\\n!! 错误: 'ffmpeg' 命令未找到。\\n")
        print("!! 请确保您已在系统中安装了 FFmpeg，并将其添加到了环境变量PATH中。")
        return False
    except subprocess.CalledProcessError as e:
        print("\\n!! 错误: FFmpeg 执行失败。\\n")
        print(f"!! 退出码: {e.returncode}")
        print("--- FFmpeg 错误输出 ---")
        print(e.stderr) # 打印 ffmpeg 的错误信息
        print("------------------------")
        return False
    except Exception as e:
        print(f"\\n!! 发生未知错误: {e}")
        return False

# --- 主函数入口 ---
if __name__ == '__main__':
    # ###################################################################
    # # 配置区域
    # ###################################################################
    
    # 输入文件
    INPUT_AUDIO = r"D:\\Python\\Project\\VideoTran\\videos\\Vocals.wav"
    
    # 输出文件
    OUTPUT_AUDIO = r"D:\\Python\\Project\\VideoTran\\output\\b.wav"
    
    # 截取的开始和结束时间 (可以是秒，也可以是 'HH:MM:SS.ms' 格式)
    START_TIME = "00:04:04" # 从第5秒开始
    END_TIME = "00:04:13"   # 到第15秒结束 (总时长10秒)
    
    # ###################################################################
    # # 执行流程
    # ###################################################################
    print("--- 开始执行音频截取脚本 [audio_cut.py] ---")
    
    cut_audio_ffmpeg(INPUT_AUDIO, OUTPUT_AUDIO, START_TIME, END_TIME)
    
    print("\\n--- 脚本执行完毕 ---")
