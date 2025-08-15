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