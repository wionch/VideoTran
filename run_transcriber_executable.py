# -*- coding: utf-8 -*-
import subprocess
import os

def run_speaker_transcription(audio_path, hf_token):
    """
    通过调用 faster-whisper-xxl.exe 来执行带说话人识别的语音转录。

    :param audio_path: 输入音频文件的完整路径。
    :param hf_token: Hugging Face 的 Read Token。
    """
    # --- 基础路径配置 ---
    project_root = os.path.dirname(os.path.abspath(__file__))
    executable_path = os.path.join(project_root, "whisper-standalone-win/Faster-Whisper-XXL", "faster-whisper-xxl.exe")
    output_dir = os.path.join(project_root, "output")

    # --- 检查路径和 Token ---
    if not os.path.exists(executable_path):
        print(f"错误: 未找到核心程序: {executable_path}")
        return
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        return
    if not hf_token or "hf_" not in hf_token:
        print("错误: 请提供有效的 Hugging Face Token。")
        print("访问 https://huggingface.co/settings/tokens 获取。")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # --- 构建命令行参数 ---
    # 修正: 移除了 --align_model 和 --hf_token，添加了 --word_timestamps
    command = [
        executable_path,
        audio_path,
        "--model", "large-v2",
        "--language", "zh",
        "--diarize", "pyannote_v3.1",# "pyannote_v3.1",
        # "--word_timestamps", "True", # 启用词级别时间戳
        "--output_dir", output_dir,
        "--output_format", "all",
        "--compute_type", "float16",
        "--device", "cuda",
        # "--verbose", "True"
    ]

    print("--- 准备执行以下命令 ---")
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))
    print("--------------------------")

    try:
        # --- 设置环境变量 ---
        # 修正: 将 HF Token 通过环境变量传递给子进程
        env = os.environ.copy()
        env["HF_TOKEN"] = hf_token
        
        # --- 执行命令 ---
        # 将 env 参数传递给 subprocess.run
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            env=env
        )
        
        print("--- 命令执行成功 ---")
        print("标准输出:")
        print(result.stdout)
        
        output_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}.srt"
        print(f"--- 处理完成！结果已保存到 {os.path.join(output_dir, output_filename)} ---")

    except subprocess.CalledProcessError as e:
        print("--- 命令执行失败 ---")
        print(f"返回码: {e.returncode}")
        print("标准输出:")
        print(e.stdout)
        print("标准错误:")
        print(e.stderr)
    except FileNotFoundError:
        print(f"错误: 无法找到命令: {executable_path}。请检查路径是否正确。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")

if __name__ == '__main__':
    # --- 用户配置 ---
    # 在这里填入你的 Hugging Face Read Token
    HUGGING_FACE_TOKEN = "" 
    
    # 指定要处理的音频文件
    audio_file_to_process = r"D:\Python\Project\VideoTran\videos\Volcal.wav"

    # --- 执行主函数 ---
    run_speaker_transcription(audio_file_to_process, HUGGING_FACE_TOKEN)