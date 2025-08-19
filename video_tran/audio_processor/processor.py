# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: processor.py
@time: 2025/8/15 16:28
"""
import os
from video_tran.utils.shell_utils import run_command


class AudioProcessor:
    """
    负责处理音频，包括从视频中提取音频和分离人声。
    """

    def __init__(self, config):
        """
        初始化 AudioProcessor。

        Args:
            config: 配置对象，包含模型路径等信息。
        """
        self.config = config

    def extract_audio(self, video_path: str, output_audio_path: str) -> bool:
        """
        从视频文件中提取完整的音轨。

        Args:
            video_path (str): 输入视频文件的路径。
            output_audio_path (str): 输出音频文件的保存路径。

        Returns:
            bool: 如果提取成功则返回 True，否则返回 False。
        """
        print(f"从 {video_path} 中提取音频...")
        command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{output_audio_path}" -y'
        success, stdout, stderr = run_command(command)
        if not success:
            print(f"音频提取失败: {stderr}")
            return False
        print(f"音频已成功提取到: {output_audio_path}")
        return True

    def separate_vocals(self, audio_path: str, output_vocals_path: str, output_background_path: str) -> bool:
        """
        将音轨分离为人声和背景声。

        Args:
            audio_path (str): 输入音频文件的路径。
            output_vocals_path (str): 分离出的人声音频的保存路径。
            output_background_path (str): 分离出的背景声频的保存路径。

        Returns:
            bool: 如果分离成功则返回 True，否则返回 False。
        """
        print(f"正在从 {audio_path} 中分离人声...")
        # 注意: 此处的 'audio-separator' 命令需要根据实际安装的工具和模型进行调整。
        # 这里使用一个假设的命令行格式。
        model_name = self.config.get('models', {}).get('uvr_model', 'UVR-MDX-NET-Inst_HQ_3.onnx')
        model_name_base = os.path.splitext(model_name)[0]
        output_dir = os.path.dirname(output_vocals_path)

        command = (
            f'audio-separator "{audio_path}" '
            f'--model_filename {model_name} '
            f'--output_dir "{output_dir}" '
            f'--output_format WAV'
        )

        success, stdout, stderr = run_command(command)
        if not success:
            print(f"人声分离失败: {stderr}")
            return False

        # audio-separator 可能会生成固定命名格式的文件，需要重命名
        # 这部分逻辑需要根据 audio-separator 的实际行为来确定
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        generated_vocals = os.path.join(output_dir, f'{base_name}_(Vocals)_{model_name_base}.wav')
        generated_background = os.path.join(output_dir, f'{base_name}_(Instrumental)_{model_name_base}.wav')

        try:
            os.rename(generated_vocals, output_vocals_path)
            os.rename(generated_background, output_background_path)
            print(f"人声已成功分离到: {output_vocals_path}")
            print(f"背景声已成功分离到: {output_background_path}")
            return True
        except FileNotFoundError as e:
            print(f"重命名分离出的文件时出错: {e}")
            print("请检查 'audio-separator' 工具的实际输出文件名格式。")
            return False
