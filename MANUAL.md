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
        - `dub`: 执行完整流程，生成配音视频。适用于单人或不需精细说话人区分的场景。
        - `dub_v2`: **增强型配音模式**。在 `dub` 模式基础上，增加说话人识别和说话人参考音批量提取，并优化时长对齐策略，适用于多说话人或对配音质量要求更高的场景。

    **具体示例**:
    ```bash
    # 标准配音模式 (dub)
    python main.py -i "D:\MyVideos\project_promo.mp4" -sl "en" -tl "zh" -m "dub"

    # 增强型配音模式 (dub_v2)
    python main.py -i "D:\MyVideos\multi_speaker_interview.mp4" -sl "en" -tl "zh" -m "dub_v2"
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

### 9.1 标准配音模式 (`dub`) 步骤

此模式适用于单人或不需精细说话人区分的场景。

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

### 9.2 增强型配音模式 (`dub_v2`) 步骤

此模式适用于多说话人或对配音质量要求更高的场景。它在标准模式的基础上，增加了说话人识别和说话人参考音批量提取，并优化了时长对齐策略。

### 步骤 T1: 音频处理 (提取并分离音轨)

-   **环境**: `main` (或您在 `config.yaml` 中为 `main` 指定的环境)
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/audio_processor/run.py --video-path "<视频路径>" --output-audio-path "<输出原始音频路径.wav>" --output-vocals-path "<输出人声路径.wav>" --output-background-path "<输出背景声路径.wav>"
    ```

### 步骤 T2: 语音转录 (带说话人识别)

-   **环境**: `transcriber` (或您在 `config.yaml` 中为 `transcriber` 指定的环境)
-   **注意：** 需添加 `--diarize` 参数。
-   **命令**:
    ```bash
    conda activate whisper_env && python video_tran/transcriber/run.py --input-audio "<输入人声路径.wav>" --lang <源语言> --output-json "<输出转录JSON路径.json>" --config-path "configs/config.yaml" --diarize
    ```

### 步骤 T2.5: LLM校正

-   **环境**: `main`
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/corrector/run.py --input-json "<输入转录JSON路径.json>" --output-json "<输出校正后JSON路径.json>"
    ```

### 步骤 T2.6: 生成说话人参考音

-   **环境**: `main` (或您在 `config.yaml` 中为 `main` 指定的环境)
-   **说明**: 此步骤会根据带有说话人信息的 JSON 文件和原始人声音频，为每个说话人生成单独的参考音频文件，存储在一个目录中。
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/speaker_processor/run.py --input-json "<输入包含说话人信息的JSON路径>" --input-audio "<输入人声路径.wav>" --output-dir "<输出说话人参考音目录路径>"
    ```

### 步骤 T3: 文本翻译

-   **环境**: `main`
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/translator/run.py --input-json "<输入校正后JSON路径.json>" --target-lang <目标语言> --output-json "<输出翻译后JSON路径.json>"
    ```

### 步骤 T4: 语音合成 (增强型)

-   **环境**: `tts_generator` (根据您在 `config.yaml` 中选择的TTS引擎，激活对应的环境，例如 `indextts_env` 或 `openvoice_env`)
-   **说明**: 此脚本经过重构，性能大幅提升，并包含了更智能的音频处理逻辑。
-   **核心功能**:
    -   **性能优化**: 脚本启动时一次性加载TTS模型，避免了为每个片段重复加载，处理速度极快。
    -   **智能时长对齐**: 当使用 `--align-duration` 时，如生成音频过长，脚本会对其进行**音频加速**而非截断，以保留完整语句。
    -   **精准音量匹配**: 默认情况下，脚本会分析每个说话人的参考音量，并将生成的语音片段**自动匹配**到对应的音量，以保留不同角色间的音量动态。
-   **命令**:
    ```bash
    # 1. 激活环境 (以 index-tts 为例)
    conda activate indextts_env

    # 2. 执行脚本
    python video_tran/tts_generator/run.py --input-json "<输入翻译后JSON路径.json>" --ref-audio "<说话人参考音目录路径>" --output-audio "<输出配音人声路径.wav>" --temp-dir "<工作目录>" --target-lang <目标语言> --use-speaker-ref --align-duration

    # (可选) 如果您想手动覆盖自动音量匹配，让所有人都统一音量，请使用 --target-dbfs
    python video_tran/tts_generator/run.py --input-json "..." --ref-audio "..." --output-audio "..." --temp-dir "..." --target-lang zh --use-speaker-ref --align-duration --target-dbfs -18.0
    ```
-   **关键参数**:
    -   `--use-speaker-ref`: **(必需)** 启用此模式，`--ref-audio` 必须指向包含各说话人 `.wav` 文件的目录。
    -   `--align-duration`: **(推荐)** 启用智能时长对齐（加速/填充）。
    -   `--target-dbfs <数值>`: **(可选)** 设置一个全局目标音量 (dBFS)。如果提供此参数，将**覆盖**默认的分说话人音量自动匹配逻辑。

### 步骤 T5: 视频生成

-   **环境**: `main`
-   **命令**:
    ```bash
    conda activate main_env && python video_tran/video_producer/run.py --original-video "<原始视频路径>" --dubbed-audio "<配音人声路径.wav>" --bg-audio "<背景声路径.wav>" --segments-json "<翻译后JSON路径.json>" --output-video "<最终视频输出路径.mp4>" --output-srt "<最终字幕输出路径.srt>" --temp-dir "<工作目录>"
    ```