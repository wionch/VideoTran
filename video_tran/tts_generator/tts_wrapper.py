# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: tts_wrapper.py
@time: 2025/8/17
"""
import os
import torch
from typing import Optional
import sys

# --- Base Class for all TTS Engines ---
class BaseTTSEngine:
    """
    所有TTS引擎的通用接口基类。
    """
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate(self, text: str, ref_audio_path: str, output_path: str, language: str) -> Optional[str]:
        """
        生成语音的核心方法。

        Args:
            text (str): 要转换的文本。
            ref_audio_path (str): 参考音色音频的路径。
            output_path (str): 生成的语音文件的保存路径。
            language (str): 目标语言。

        Returns:
            Optional[str]: 成功则返回生成文件的路径，否则返回None。
        """
        raise NotImplementedError("子类必须实现 generate 方法")

# --- OpenVoice Engine Implementation ---
class OpenVoiceEngine(BaseTTSEngine):
    """
    封装对 OpenVoiceV2 的调用。
    """
    def __init__(self, config):
        super().__init__(config)
        print("Initializing OpenVoice Engine...")
        from openvoice.api import ToneColorConverter
        from openvoice.se_extractor import get_se

        openvoice_root = self.config.get('openvoice', {}).get('model_dir', 'OpenVoice')
        checkpoints_v2_dir = os.path.join(openvoice_root, 'checkpoints_v2')
        ckpt_converter = os.path.join(checkpoints_v2_dir, 'converter')
        
        self.tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=self.device)
        self.tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))
        self.source_se = torch.load(os.path.join(checkpoints_v2_dir, 'base_speakers', 'ses', 'ZH.pth'), map_location=self.device)
        self.get_se = get_se

    def generate(self, text: str, ref_audio_path: str, output_path: str, language: str) -> Optional[str]:
        from melo.api import TTS
        melo_tts = TTS(language=language.upper(), device=self.device)
        speaker_ids = melo_tts.hps.data.spk2id
        default_speaker_id = list(speaker_ids.values())[0]

        try:
            target_se, _ = self.get_se(ref_audio_path, self.tone_color_converter, target_dir='processed', vad=True)
            temp_tts_path = os.path.join(os.path.dirname(output_path), f'_temp_tts_{os.getpid()}.wav')
            melo_tts.tts_to_file(text, default_speaker_id, temp_tts_path, speed=1.0)

            self.tone_color_converter.convert(
                audio_src_path=temp_tts_path,
                src_se=self.source_se,
                tgt_se=target_se,
                output_path=output_path,
                tau=0.5,
                message="@MyShell"
            )
            if os.path.exists(temp_tts_path):
                os.remove(temp_tts_path)
            return output_path
        except Exception as e:
            print(f"[OpenVoice] Error during generation: {e}")
            return None

# --- IndexTTS Engine Implementation ---
class IndexTTSEngine(BaseTTSEngine):
    """
    封装对 IndexTTS 的调用。
    """
    def __init__(self, config):
        super().__init__(config)
        print("Initializing IndexTTS Engine...")
        
        # 1. 使用正确的模块导入
        from indextts.infer import IndexTTS
        
        # 2. 显式指定模型和配置路径进行初始化，使其更加稳健
        model_dir = os.path.join("index-tts", "checkpoints")
        config_path = os.path.join(model_dir, "config.yaml")
        
        if not os.path.exists(model_dir) or not os.path.exists(config_path):
            raise FileNotFoundError(f"IndexTTS model/config not found in '{os.path.abspath(model_dir)}'. Please ensure the 'checkpoints' directory is inside the 'index-tts' directory.")
            
        self.model = IndexTTS(model_dir=model_dir, cfg_path=config_path)

    def generate(self, text: str, ref_audio_path: str, output_path: str, language: str) -> Optional[str]:
        try:
            # 3. 修正 infer 方法的参数顺序并移除无效的 language 参数
            # 正确顺序: (voice, text, output_path)
            self.model.infer(ref_audio_path, text, output_path)
            return output_path
        except Exception as e:
            print(f"[IndexTTS] Error during generation: {e}")
            return None

# --- TTS Wrapper Factory ---
class TTSWrapper:
    """
    一个工厂类，根据配置创建并持有正确的TTS引擎实例。
    """
    def __init__(self, config):
        self.config = config
        engine_name = self.config.get('tts_engine', 'openvoice').lower()
        print(f"Selected TTS Engine: {engine_name}")

        if engine_name == 'indextts':
            self.engine = IndexTTSEngine(self.config)
        elif engine_name == 'openvoice':
            self.engine = OpenVoiceEngine(self.config)
        else:
            raise ValueError(f"Unsupported TTS engine: {engine_name}")

    def generate(self, text: str, ref_audio_path: str, output_path: str, language: str) -> Optional[str]:
        """
        将调用委托给当前加载的引擎。
        """
        return self.engine.generate(text, ref_audio_path, output_path, language)
