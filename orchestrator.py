# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: orchestrator.py
@time: 2025/8/15 18:50
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime

from video_tran.config import load_config
from video_tran.utils.shell_utils import run_command


class Orchestrator:
    """
    负责编排整个视频翻译流程的控制器。
    """

    def __init__(self, config_path: str):
        """
        初始化 Orchestrator。

        Args:
            config_path (str): 配置文件的路径。
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        if not self.config:
            raise ValueError(f"无法加载配置文件: {config_path}")
        self.project_root = os.path.abspath(os.path.dirname(__file__))


    def run(self, video_path: str, src_lang: str, target_lang: str, mode: str = 'dub', no_cleanup: bool = False):
        """
        执行端到端的视频翻译流程。

        Args:
            video_path (str): 输入视频的路径。
            src_lang (str): 源语言代码 (例如, 'zh')。
            target_lang (str): 目标语言 (例如, 'English')。
            mode (str): 处理模式: 'dub', 'dub_v2', 'transcribe', 'translate'。
            no_cleanup (bool): 如果为 True，则不清理临时工作目录。
        """
        # 创建一个唯一的临时工作目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        tasks_dir = os.path.join(self.project_root, 'tasks')  # 改为 tasks 目录
        os.makedirs(tasks_dir, exist_ok=True)
        work_dir = os.path.join(tasks_dir, video_name)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        print(f"工作目录已创建: {work_dir}")

        try:
            # --- 定义文件路径 ---
            # T1 Outputs
            original_audio = os.path.join(work_dir, "original_audio.wav")
            vocals_audio = os.path.join(work_dir, "vocals.wav")
            background_audio = os.path.join(work_dir, "background.wav")
            # T2 Outputs
            transcribed_json = os.path.join(work_dir, "transcribed.json")
            # T2.5 Outputs
            corrected_json = os.path.join(work_dir, "corrected.json")
            # T3 Outputs
            translated_json = os.path.join(work_dir, "translated.json")
            # T4 Outputs
            dubbed_vocals_audio = os.path.join(work_dir, "dubbed_vocals.wav")
            # T5 Outputs
            output_video_name = f"{video_name}_dubbed_{target_lang}.mp4"
            output_srt_name = f"{video_name}_dubbed_{target_lang}.srt"
            output_dir = self.config.get('paths', {}).get('output', 'output')
            os.makedirs(output_dir, exist_ok=True)
            final_video_path = os.path.join(output_dir, output_video_name)
            final_srt_path = os.path.join(output_dir, output_srt_name)

            # --- 执行流水线 ---
            if mode == 'dub_v2':
                # --- dub_v2 专属文件路径 ---
                diarized_json = os.path.join(work_dir, "diarized.json")
                speaker_ref_dir = os.path.join(work_dir, "speaker_references")
                os.makedirs(speaker_ref_dir, exist_ok=True)

                # --- dub_v2 执行流水线 ---
                self._run_module("T1: 音频处理", self._get_t1_command(video_path, original_audio, vocals_audio, background_audio))
                self._run_module("T2: 语音转录与说话人识别", self._get_t2_command(vocals_audio, src_lang, diarized_json, self.config_path, diarize=True))
                self._run_module("T2.6: 生成说话人参考音", self._get_t2_6_command(diarized_json, vocals_audio, speaker_ref_dir))
                self._run_module("T2.5: LLM校正", self._get_t2_5_command(diarized_json, corrected_json))
                self._run_module("T3: 文本翻译", self._get_t3_command(corrected_json, target_lang, translated_json))
                self._run_module("T4: 增强型语音合成", self._get_t4_command(translated_json, speaker_ref_dir, dubbed_vocals_audio, work_dir, target_lang, use_speaker_ref=True, align_duration=True))
                self._run_module("T5: 增强型视频生成", self._get_t5_command(video_path, dubbed_vocals_audio, background_audio, translated_json, final_video_path, final_srt_path, work_dir, normalize_volume=True))

            else:  # 'dub', 'transcribe', 'translate' modes
                self._run_module("T1: 音频处理", self._get_t1_command(video_path, original_audio, vocals_audio, background_audio))
                self._run_module("T2: 语音转录", self._get_t2_command(vocals_audio, src_lang, transcribed_json, self.config_path))
                if mode == 'transcribe':
                    print(f"
>>> 转录完成! <<<")
                    print(f"结果已保存到: {transcribed_json}")
                    return

                self._run_module("T2.5: LLM校正", self._get_t2_5_command(transcribed_json, corrected_json))
                self._run_module("T3: 文本翻译", self._get_t3_command(corrected_json, target_lang, translated_json))
                if mode == 'translate':
                    print(f"
>>> 翻译完成! <<<")
                    print(f"结果已保存到: {translated_json}")
                    return
                
                # This is the 'dub' mode
                self._run_module("T4: 语音合成", self._get_t4_command(translated_json, vocals_audio, dubbed_vocals_audio, work_dir, target_lang))
                self._run_module("T5: 视频生成", self._get_t5_command(video_path, dubbed_vocals_audio, background_audio, translated_json, final_video_path, final_srt_path, work_dir))

            print("\n>>> 流程执行成功! <<<")
            print(f"最终视频文件: {final_video_path}")
            print(f"最终字幕文件: {final_srt_path}")

finally:
            # 清理临时文件
            if not no_cleanup:
                shutil.rmtree(work_dir)
                print(f"工作目录已清理: {work_dir}")
            else:
                print(f"工作目录保留在: {work_dir}，以便检查中间文件。")


    def _run_module(self, module_name: str, command: str):
        """
        执行一个流程模块并检查结果。
        """
        print(f"
--- 开始执行模块: {module_name} ---")
        print(f"命令: {command}")
        
        success, stdout, stderr = run_command(command)
        
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        
        if not success:
            print(f"!!! 模块 {module_name} 执行失败 !!!")
            sys.exit(1)
        print(f"--- 模块 {module_name} 执行成功 ---
")

    def _get_command_prefix(self, module_key: str) -> str:
        """获取模块的环境激活命令"""
        return self.config.get('environments', {}).get(module_key, '')

    def _get_t1_command(self, video_path, original_audio, vocals_audio, background_audio):
        cmd_prefix = self._get_command_prefix('main')
        script_path = os.path.join(self.project_root, 'video_tran', 'audio_processor', 'run.py')
        return f"{cmd_prefix} && python {script_path} --video-path \"{video_path}\" --output-audio-path \"{original_audio}\" --output-vocals-path \"{vocals_audio}\" --output-background-path \"{background_audio}\""

    def _get_t2_command(self, vocals_audio, src_lang, transcribed_json, config_path, diarize=False):
        cmd_prefix = self._get_command_prefix('transcriber')
        script_path = os.path.join(self.project_root, 'video_tran', 'transcriber', 'run.py')
        command = f"{cmd_prefix} && python {script_path} --input-audio \"{vocals_audio}\" --lang {src_lang} --output-json \"{transcribed_json}\" --config-path \"{config_path}\""
        if diarize:
            command += " --diarize"
        return command

    def _get_t2_5_command(self, transcribed_json, corrected_json):
        cmd_prefix = self._get_command_prefix('main')
        script_path = os.path.join(self.project_root, 'video_tran', 'corrector', 'run.py')
        return f"{cmd_prefix} && python {script_path} --input-json \"{transcribed_json}\" --output-json \"{corrected_json}\""

    def _get_t2_6_command(self, diarized_json, vocals_audio, speaker_ref_dir):
        cmd_prefix = self._get_command_prefix('main')
        script_path = os.path.join(self.project_root, 'video_tran', 'speaker_processor', 'run.py')
        return f"{cmd_prefix} && python {script_path} --input-json \"{diarized_json}\" --input-audio \"{vocals_audio}\" --output-dir \"{speaker_ref_dir}\""

    def _get_t3_command(self, corrected_json, target_lang, translated_json):
        cmd_prefix = self._get_command_prefix('main')
        script_path = os.path.join(self.project_root, 'video_tran', 'translator', 'run.py')
        return f"{cmd_prefix} && python {script_path} --input-json \"{corrected_json}\" --target-lang \"{target_lang}\" --output-json \"{translated_json}\""

    def _get_t4_command(self, translated_json, ref_path, dubbed_vocals_audio, temp_dir, target_lang, use_speaker_ref=False, align_duration=False):
        cmd_prefix = self._get_command_prefix('tts_generator')
        script_path = os.path.join(self.project_root, 'video_tran', 'tts_generator', 'run.py')
        command = f"{cmd_prefix} && python {script_path} --input-json \"{translated_json}\" --ref-audio \"{ref_path}\" --output-audio \"{dubbed_vocals_audio}\" --temp-dir \"{temp_dir}\" --target-lang \"{target_lang}\""
        if use_speaker_ref:
            command += " --use-speaker-ref"
        if align_duration:
            command += " --align-duration"
        return command

    def _get_t5_command(self, original_video, dubbed_vocals, bg_audio, segments_json, output_video, output_srt, temp_dir, normalize_volume=False):
        cmd_prefix = self._get_command_prefix('main')
        script_path = os.path.join(self.project_root, 'video_tran', 'video_producer', 'run.py')
        command = f"{cmd_prefix} && python {script_path} --original-video \"{original_video}\" --dubbed-audio \"{dubbed_vocals}\" --bg-audio \"{bg_audio}\" --segments-json \"{segments_json}\" --output-video \"{output_video}\" --output-srt \"{output_srt}\" --temp-dir \"{temp_dir}\""
        if normalize_volume:
            command += " --normalize-volume"
        return command


    