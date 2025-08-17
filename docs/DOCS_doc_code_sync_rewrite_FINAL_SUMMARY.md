# VideoTran 项目总结与操作手册

## 1. 项目概述 (Project Overview)

本次任务的核心目标是重构 `VideoTran` 项目的文档体系，特别是将 `MANUAL.md` 的内容与代码实现进行严格同步。通过引入模块化的 `video_tran` 包，我们提升了代码的组织性、可维护性和可扩展性。旧的脚本被重构为独立的、功能内聚的模块，并通过一个统一的编排器 (`orchestrator.py`) 来驱动，从而实现了更清晰、更鲁棒的视频处理流程。

## 2. 最终交付物 (Final Deliverables)

以下是本次任务修改或创建的核心文件的最终代码。

---

### **文件 1: `D:\Python\Project\VideoTran\MANUAL.md`**

```markdown
# VideoTran - 自动化视频语音翻译工具 | 操作手册

## 1. 项目概述

VideoTran 是一个功能强大的自动化视频翻译与配音工具。它采用先进的AI模型，能够将视频中的源语言语音（如中文）替换为指定目标语言（如英文）的语音，同时保留原始语音片段的音色。最终生成一个音轨被替换、并附带精确翻译字幕的全新视频文件。

## 2. 核心特性

- **端到端自动化**: 只需一条命令，即可完成从原始视频到翻译后视频的全过程。
- **片段级音色保留**: 采用“孤岛”策略，为每个语音片段独立处理和保留音色，在保证角色身份100%正确的前提下，实现高保真音色转换。
- **智能时长校准**: 采用创新的VAD（语音活动检测）混合策略，在保证语音自然度的前提下，智能调整翻译后语音的时长，以最大程度对齐原始口型。
- **高质量翻译与校正**: 集成大型语言模型（LLM），不仅翻译准确，还能对语音识别（ASR）的初步结果进行润色和校正。
- **模块化架构**: 系统采用高度模块化的代码库，便于扩展、维护和对单个模块进行测试。

## 3. 系统架构

本工具由一个核心**编排器 (`Orchestrator`)** 和一系列**处理模块**构成。编排器在单一Python环境中，通过直接调用模块类来驱动流水线：

1.  **音频处理 (`AudioProcessor`)**: 从视频中提取音频，并将其分离为**人声**和**背景声**。
2.  **语音转录 (`Transcriber`)**: 使用 `whisperX` 将人声音频转录为带精确时间戳的文本。
3.  **文本校正 (`Corrector`)**: 使用LLM对转录文本进行标点和语法优化。
4.  **文本翻译 (`Translator`)**: 使用LLM将校正后的文本翻译为目标语言。
5.  **语音合成 (`TTSGenerator`)**: 使用 `OpenVoice`，结合原始语音片段的音色，将翻译文本合成为新的人声音频。此阶段会执行核心的**时长校准**算法。
6.  **视频生成 (`VideoProducer`)**: 将新的人声音频与原始背景声混合，替换原视频音轨，并生成SRT字幕文件。

## 4. 环境准备 (Prerequisites)

在开始安装之前，请确保您的系统（已在Windows下验证）已安装以下软件：

- **Python**: 版本 `3.10` 或更高。
- **Conda**: 用于管理Python虚拟环境。
- **FFmpeg**: 一个强大的音视频处理工具。请从其[官网](https://ffmpeg.org/download.html)下载，并**务必将其可执行文件路径添加到系统的环境变量 `Path` 中**。

## 5. 安装与配置

请严格按照以下步骤在**单一环境**中进行安装和配置。

### 步骤 1: 创建并配置Conda环境

1.  **创建并激活Conda环境**。打开终端，执行以下命令：
    ```bash
    conda create -n videotran_env python=3.10 -y
    conda activate videotran_env
    ```

2.  **安装核心依赖**。在已激活的环境中，执行以下命令安装所有必需的Python包。这将一次性安装所有依赖，包括PyTorch, whisperX, OpenVoice等。
    ```bash
    # 1. 安装PyTorch (GPU版本，请根据您的CUDA版本调整)
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

    # 2. 安装whisperX
    pip install git+https://github.com/m-bain/whisperX.git

    # 3. 安装OpenVoice
    pip install openvoice-cli

    # 4. 安装其他依赖
    pip install moviepy imageio-ffmpeg "audio-separator[gpu]" pyyaml webrtcvad pydub ollama
    ```

### 步骤 2: 设置API密钥

本项目需要调用LLM API进行文本校正和翻译。请根据您在`config.yaml`中配置的`translator`服务商，设置对应的环境变量。

1.  获取您的API密钥。
2.  将其设置为系统环境变量。**请勿将密钥硬编码到任何文件中**。
    - **示例 (Windows PowerShell)**:
      ```powershell
      # 如果使用DeepSeek
      $env:DEEPSEEK_TOKEN="sk-xxxxxxxxxxxxxxxxxxxx"

      # 如果使用Ollama (通常无需密钥)
      # 无需操作
      ```

### 步骤 3: 准备配置文件

1.  在项目根目录中，找到 `configs/` 文件夹。
2.  将 `config.yaml.template` 文件复制一份，并重命名为 `config.yaml`。
3.  打开 `config.yaml` 文件。通常情况下，您只需关心`translator`和`tts_generator`的配置，其他可保持默认。

## 6. 如何使用

一切准备就绪后，您可以开始进行视频翻译。

1.  **激活环境**:
    每次运行前，请确保您在已创建的环境 (`videotran_env`) 下。
    ```bash
    conda activate videotran_env
    ```

2.  **执行命令**:
    打开终端，进入项目根目录，使用以下命令格式启动程序：
    ```bash
    python main.py -i <视频路径> -sl <源语言> -tl <目标语言> -m <处理模式>
    ```

    **参数说明**:
    - `-i, --video_path`: **[必需]** 您的原始视频文件的完整路径。
    - `-sl, --source_language`: **[必需]** 视频中的原始语言 (例如: `en`, `zh`)。
    - `-tl, --target_language`: **[必需]** 您希望翻译成的目标语言 (例如: `zh`, `en`)。
    - `-c, --config_path`: **[可选]** 配置文件的路径。默认为 `configs/config.yaml`。
    - `-m, --mode`: **[可选]** 处理模式。默认为 `dub`。
        - `transcribe`: 只执行到语音转录。
        - `translate`: 执行到文本翻译。
        - `dub`: 执行完整流程，生成配音视频。

    **具体示例**:
    ```bash
    python main.py -i "D:\MyVideos\project_promo.mp4" -sl "en" -tl "zh" -m "dub"
    ```

## 7. 产出说明

流程成功执行后，您将在 `output/` 目录（或您在配置文件中指定的其他输出目录）中找到最终产出物。根据您选择的模式，可能包含：
- **`*_transcribed.json`**: 转录结果。
- **`*_translated.json`**: 翻译结果。
- **`*_dubbed.mp4`**: 最终的配音视频文件。
- **`*_dubbed.srt`**: 与视频同步的SRT格式字幕文件。

此外，在项目根目录下会生成一个临时的 `videotran_*` 文件夹，其中包含了所有中间处理文件，方便您进行检查和调试。

## 8. 故障排查

- **`ffmpeg: command not found`**: 说明 `ffmpeg` 未正确安装或其路径未添加到系统环境变量 `Path` 中。
- **`未提供...API 密钥...`**: 说明对应的环境变量未被正确设置。请重新检查或设置后重启您的终端。
- **模块执行失败**: 检查终端中打印的日志信息。由于所有模块在同一进程中运行，错误信息会直接显示在控制台，据此进行排查。

## 9. 手动与分步执行

本系统设计为模块化，允许您独立执行流水线的每一个步骤。这对于调试或手动提供中间文件（例如，您自己准备的翻译稿）非常有用。

**重要**:
-   执行任何步骤前，请务必激活对应的Conda环境。不同的模块可能需要不同的环境。
-   所有路径参数（如 `--video-path`, `--output-json`）都应使用**绝对路径**或相对于您当前终端工作目录的**正确相对路径**。
-   以下命令中的 `<...>` 表示您需要替换为实际文件路径的占位符。
-   建议将所有中间文件都存放在一个统一的工作目录中，例如 `tasks/your_video_name/`。

### 步骤 T1: 音频处理 (提取并分离音轨)

-   **环境**: `main` (或您在 `config.yaml` 中为 `main` 指定的环境)
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/audio_processor/run.py --video-path "<视频路径>" --output-audio-path "<输出原始音频路径.wav>" --output-vocals-path "<输出人声路径.wav>" --output-background-path "<输出背景声路径.wav>"
    ```

### 步骤 T2: 语音转录

-   **环境**: `transcriber` (或您在 `config.yaml` 中为 `transcriber` 指定的环境)
-   **命令**:
    ```bash
    conda activate whisper_env && python video_tran/transcriber/run.py --input-audio "<输入人声路径.wav>" --lang <源语言> --output-json "<输出转录JSON路径.json>" --config-path "configs/config.yaml"
    ```

### 步骤 T2.5: LLM校正

-   **环境**: `main`
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/corrector/run.py --input-json "<输入转录JSON路径.json>" --output-json "<输出校正后JSON路径.json>"
    ```

### 步骤 T3: 文本翻译

-   **环境**: `main`
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/translator/run.py --input-json "<输入校正后JSON路径.json>" --target-lang <目标语言> --output-json "<输出翻译后JSON路径.json>"
    ```

### 步骤 T4: 语音合成

-   **环境**: `tts_generator` (或您在 `config.yaml` 中为 `tts_generator` 指定的环境)
-   **命令**:
    ```bash
    conda activate openvoice_env && python video_tran/tts_generator/run.py --input-json "<输入翻译后JSON路径.json>" --ref-audio "<原始人声路径.wav>" --output-audio "<输出配音人声路径.wav>" --temp-dir "<工作目录>" --target-lang <目标语言>
    ```

### 步骤 T5: 视频生成

-   **环境**: `main`
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/video_producer/run.py --original-video "<原始视频路径>" --dubbed-audio "<配音人声路径.wav>" --bg-audio "<背景声路径.wav>" --segments-json "<翻译后JSON路径.json>" --output-video "<最终视频输出路径.mp4>" --output-srt "<最终字幕输出路径.srt>" --temp-dir "<工作目录>"
    ```
```

### **文件 2: `D:\Python\Project\VideoTran\orchestrator.py`**

```python
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
                    print(f"\n>>> 转录完成! <<<")
                    print(f"结果已保存到: {transcribed_json}")
                    return

                self._run_module("T2.5: LLM校正", self._get_t2_5_command(transcribed_json, corrected_json))
                self._run_module("T3: 文本翻译", self._get_t3_command(corrected_json, target_lang, translated_json))
                if mode == 'translate':
                    print(f"\n>>> 翻译完成! <<<")
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
                print(f"工作目录保留在: {work_dir}，以便检查中间文件。" )


    def _run_module(self, module_name: str, command: str):
        """
        执行一个流程模块并检查结果。
        """
        print(f"\n--- 开始执行模块: {module_name} ---")
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
```

### **文件 3: `D:\Python\Project\VideoTran\main.py`**

```python
# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: main.py
@time: 2025/8/15 19:00
"""
import argparse
from orchestrator import Orchestrator


def main():
    """
    应用程序的主入口点。
    """
    parser = argparse.ArgumentParser(description="自动化视频语音翻译工具。" )
    parser.add_argument("-i", "--input-video", required=True, help="要处理的输入视频文件的路径。" )
    parser.add_argument("-sl", "--source-language", required=True, help="视频的源语言代码 (例如, 'zh' 表示中文)。" )
    parser.add_argument("-tl", "--target-language", required=True, help="要翻译成的目标语言 (例如, 'en' 表示英文)。" )
    parser.add_argument("-c", "--config", default="configs/config.yaml", help="配置文件的路径。" )
    # 新增 mode 参数
    parser.add_argument("-m", "--mode", choices=['dub', 'dub_v2', 'transcribe', 'translate'], default='dub', help="处理模式: 'dub' (标准配音), 'dub_v2' (增强型配音), 'transcribe' (仅转录), 'translate' (仅翻译)。" )
    # 新增 no_cleanup 参数
    parser.add_argument("--no-cleanup", action="store_true", help="执行后不清理临时工作目录。" )

    args = parser.parse_args()

    try:
        orchestrator = Orchestrator(args.config)
        orchestrator.run(
            video_path=args.input_video,
            src_lang=args.source_language,
            target_lang=args.target_language,
            mode=args.mode,
            no_cleanup=args.no_cleanup
        )
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")


if __name__ == "__main__":
    main()
```

### **文件 4: `D:\Python\Project\VideoTran\video_tran\audio_processor\run.py`**

```python
# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 16:35
"""
import argparse
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.audio_processor.processor import AudioProcessor
from video_tran.config import load_config


def main():
    """
    音频处理模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="音频处理模块：提取和分离音频。" )
    parser.add_argument("--video-path", required=True, help="输入视频文件的路径。" )
    parser.add_argument("--output-audio-path", required=True, help="提取出的原始音频的保存路径。" )
    parser.add_argument("--output-vocals-path", required=True, help="分离出的人声音频的保存路径。" )
    parser.add_argument("--output-background-path", required=True, help="分离出的背景声的保存路径。" )
    parser.add_argument("--config-path", default="configs/config.yaml", help="配置文件的路径。" )

    args = parser.parse_args()

    # 加载配置
    # 注意：在实际运行前，需要将 config.yaml.template 复制为 config.yaml
    config = load_config(args.config_path)
    if not config:
        print(f"无法加载配置文件: {args.config_path}")
        sys.exit(1)

    processor = AudioProcessor(config)

    # 1. 提取音频
    success = processor.extract_audio(args.video_path, args.output_audio_path)
    if not success:
        print("提取音频失败，流程终止。" )
        sys.exit(1)

    # 2. 分离人声
    success = processor.separate_vocals(
        args.output_audio_path,
        args.output_vocals_path,
        args.output_background_path
    )
    if not success:
        print("分离人声失败，流程终止。" )
        sys.exit(1)

    print("音频处理成功完成。" )


if __name__ == "__main__":
    main()
```

---

### **文件 5: `D:\Python\Project\VideoTran\video_tran\transcriber\run.py`**

```python
# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: run.py
@time: 2025/8/15 17:00
"""
import argparse
import json
import sys
import os
from typing import List

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from video_tran.utils.shell_utils import run_command
# Segment class now has an optional 'speaker' field
from video_tran.transcriber.data_types import Segment, segments_to_json
from video_tran.config import load_config


def parse_whisperx_json(json_path: str) -> List[Segment]:
    """
    解析 whisperX 的 JSON 输出文件，现在支持说话人信息。

    Args:
        json_path (str): whisperX 生成的 JSON 文件的路径。

    Returns:
        List[Segment]: Segment 对象列表。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        # whisperX 在diarize模式下，会将speaker信息添加到每个word中，我们需要从word中聚合
        # 或者，更简单的做法是直接使用最终的segments，它会有一个speaker字段
        for item in data.get('segments', []):
            start_time = item.get('start')
            end_time = item.get('end')
            text = item.get('text', '').strip()
            speaker = item.get('speaker') # 直接获取speaker字段

            if start_time is not None and end_time is not None and text:
                segments.append(Segment(start=start_time, end=end_time, text=text, speaker=speaker))
        
        return segments
    except FileNotFoundError:
        print(f"错误: whisperX JSON 文件未找到 at '{json_path}'")
        return []
    except Exception as e:
        print(f"解析 whisperX JSON 文件时出错: {e}")
        return []


def main():
    """
    语音转录模块的命令行入口点。
    """
    parser = argparse.ArgumentParser(description="使用 whisperX 转录音频。" )
    parser.add_argument("--input-audio", required=True, help="要转录的输入音频文件的路径。" )
    parser.add_argument("--lang", required=True, help="音频的语言代码 (例如, 'zh')。" )
    parser.add_argument("--output-json", required=True, help="输出的转录结果 JSON 文件的路径。" )
    parser.add_argument("--config-path", required=True, help="配置文件的路径。" )
    # Add the diarize flag
    parser.add_argument("--diarize", action="store_true", help="执行说话人识别。" )

    args = parser.parse_args()

    config = load_config(args.config_path)
    if not config:
        print(f"错误: 无法加载配置文件 at '{args.config_path}'", file=sys.stderr)
        sys.exit(1)

    t_config = config.get('transcriber', {})
    model = t_config.get('model', 'large-v2')
    batch_size = t_config.get('batch_size', 16)
    compute_type = t_config.get('compute_type', 'float16')

    output_dir = os.path.dirname(args.output_json)
    os.makedirs(output_dir, exist_ok=True)

    # 构建 whisperX 命令
    command = (
        f'whisperx "{args.input_audio}" ' 
        f'--model {model} ' 
        f'--language {args.lang} ' 
        f'--output_format json ' 
        f'--output_dir "{output_dir}" ' 
        f'--batch_size {batch_size} ' 
        f'--compute_type {compute_type}'
    )

    # 如果启用了说话人识别，添加相应参数
    if args.diarize:
        # Diarization requires alignment model
        command += ' --align_model WAV2VEC2_ASR_LARGE_LV60K_960H'
        command += ' --diarize'
        # Check for Hugging Face token
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token:
            command += f' --hf_token {hf_token}'
        else:
            print("警告: 未找到 HUGGING_FACE_TOKEN 环境变量。如果 pyannote/speaker-diarization 模型需要认证，执行可能会失败。" )


    print(f"执行 whisperX 命令: {command}")
    success, stdout, stderr = run_command(command)

    if not success:
        print("whisperX 执行失败。", file=sys.stderr)
        print("STDOUT:", stdout, file=sys.stderr)
        print("STDERR:", stderr, file=sys.stderr)
        sys.exit(1)

    input_basename = os.path.splitext(os.path.basename(args.input_audio))[0]
    whisperx_json_path = os.path.join(output_dir, f"{input_basename}.json")

    if not os.path.exists(whisperx_json_path):
        print(f"错误: 未找到 whisperX 的输出文件: {whisperx_json_path}", file=sys.stderr)
        sys.exit(1)

    segments = parse_whisperx_json(whisperx_json_path)
    if not segments:
        print("未能从 whisperX 的输出中解析出任何语音片段。" )
        sys.exit(1)

    segments_to_json(segments, args.output_json)

    print(f"转录完成，结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
```

---

### **文件 6: `D:\Python\Project\VideoTran\video_tran\corrector\run.py`**

```python
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
    parser = argparse.ArgumentParser(description="LLM 字幕校正模块：使用 DeepSeek API 校正字幕文本。" )
    parser.add_argument("--input-json", required=True, help="输入的 segments JSON 文件路径。" )
    parser.add_argument("--output-json", required=True, help="输出的校正后 segments JSON 文件路径。" )

    args = parser.parse_args()

    segments = segments_from_json(args.input_json)
    if not segments:
        print(f"未能从 {args.input_json} 加载或解析出任何片段。" )
        sys.exit(1)

    try:
        client = DeepSeekClient()
    except ValueError as e:
        print(f"初始化 DeepSeekClient 时出错: {e}")
        sys.exit(1)

    corrected_segments = [None] * len(segments)

    print("开始并行校正字幕..." )

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, segment in enumerate(segments):
            # 为每个请求创建一个异步任务
            task = client.correct_text_async(session, segment.text)
            tasks.append(task)

        # 使用 tqdm 显示异步任务的进度
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="校正进度" ):
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
```