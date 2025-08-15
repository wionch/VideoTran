# 施工蓝图 (Task v7 - 实施同步版): AUTOMATED_VIDEO_TRANSLATION

**版本说明: v7版本根据最终代码实现重写了环境设置和编排器任务，准确反映了项目的当前状态。**

## T-1: 项目环境设置 (Project Environment Setup)
- **目标**: 创建并配置项目运行所需的**单一环境**和外部工具。
- **子任务**:
    - **T-1.1**: **安装FFmpeg**。请根据您的操作系统（Windows）从官方网站下载并安装FFmpeg，并确保将其路径添加到系统环境变量`Path`中。
    - **T-1.2**: **创建并激活Conda环境**。打开终端，执行以下命令创建一个新的Conda环境。
        ```bash
        conda create -n videotran_env python=3.10 -y
        conda activate videotran_env
        ```
    - **T-1.3**: **安装核心依赖**。在已激活的环境中，执行以下命令安装所有必需的Python包。请注意，这将一次性安装所有依赖，包括PyTorch, whisperX, OpenVoice等。
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
    - **T-1.4**: **设置环境变量**。请根据您的操作系统说明，设置以下环境变量，用于存放API密钥：
        - `DEEPSEEK_TOKEN` (如果使用DeepSeek)
        - `OPENAI_API_KEY` (如果使用OpenAI)
        - (请根据您在`config.yaml`中选择的LLM服务商，设置对应的密钥)

---

## T0: 项目脚手架
- **依赖**: T-1
- **目标**: 搭建项目的基础目录结构和配置文件。
- **子任务**:
    - **T0.1**: 创建根目录 `video_tran`。
    - **T0.2**: 在 `video_tran` 内创建子目录: `audio_processor`, `corrector`, `transcriber`, `translator`, `tts_generator`, `video_producer`, `utils`。
    - **T0.3**: 在 `video_tran` 和所有子目录中创建空的 `__init__.py` 文件。
    - **T0.4**: 创建顶层文件: `main.py`, `orchestrator.py`, `config.py`。
    - **T0.5**: 创建 `config/config.yaml.template`，内容包含非敏感配置项（如`model_paths`）和所需环境变量的注释说明。
    - **T0.6**: 创建 `.gitignore`，忽略 `__pycache__/`, `*.pyc`, `*.env`, `temp/`, `output/`, `venv/`, `.*env`。

---

## T1: 音频处理模块
- **依赖**: T0
- **目标**: 实现从视频提取并分离出人声和背景声的功能。
- **子任务**:
    - **T1.1**: 在 `video_tran/utils/` 中创建 `shell_utils.py`，定义函数 `run_command(command: str) -> (bool, str, str)`。
    - **T1.2**: 在 `video_tran/audio_processor/` 中创建 `processor.py`，定义 `AudioProcessor` 类，包含 `extract_audio` 和 `separate_vocals` 方法。
    - **T1.3**: 创建 `video_tran/audio_processor/run.py` 作为CLI入口 (可选，用于独立测试)。

---

## T2: 语音转录模块
- **依赖**: T1
- **目标**: 将人声音频转录为带时间戳的文本。
- **子任务**:
    - **T2.1**: 在 `video_tran/transcriber/` 中创建 `data_types.py`，定义 `Segment` 数据类：`@dataclass class Segment: start: float; end: float; text: str`。
    - **T2.2**: 创建 `video_tran/transcriber/run.py` (CLI)，调用 `whisperX` 并关闭diarize选项。
    - **T2.3**: 读取 `whisperX` 的JSON结果，解析并保存为 `List[Segment]` 格式。

---

## T2.5: LLM字幕校正模块
- **依赖**: T2
- **目标**: 使用LLM修正ASR结果中的错别字和语法。
- **子任务**:
    - **T2.5.1**: 创建 `video_tran/corrector/run.py` (CLI)。
    - **T2.5.2**: 在 `video_tran/utils/` 中创建 `llm_client.py`，定义 `LLMClient`，从环境变量读取密钥。
    - **T2.5.3**: 在CLI脚本中，读取 `segments.json`，对每条`text`调用`LLMClient`进行校正。
    - **T2.5.4**: 将结果写入 `corrected_segments.json`。

---

## T3: 文本翻译模块
- **依赖**: T2.5
- **目标**: 将校正后的文本翻译成目标语言，并初步考虑时长。
- **子任务**:
    - **T3.1**: 创建 `video_tran/translator/run.py` (CLI)。
    - **T3.2**: 复用 `utils.llm_client`。
    - **T3.3**: 读取 `corrected_segments.json`，构造时长感知的Prompt调用LLM。
    - **T3.4**: 将结果写入 `translated_segments.json`。

---

## T4: 语音合成与时长校准模块
- **依赖**: T1, T3
- **目标**: 生成语音，并采用VAD混合策略确保时长精确对齐。
- **子任务**:
    - **T4.1**: 在 `video_tran/utils/` 中创建 `vad_utils.py`，定义 `get_speech_timestamps(audio_path)` 函数（使用 `py-webrtcvad`）。
    - **T4.2**: 在 `video_tran/utils/` 中创建 `audio_utils.py`，定义 `adjust_silence(audio_path, ...)`。
    - **T4.3**: 在 `video_tran/tts_generator/` 中创建 `tts_wrapper.py`，封装 `OpenVoice` 调用，**需支持语速调节参数**。
    - **T4.4**: 重构 `video_tran/tts_generator/run.py` (CLI) 的核心逻辑：
        - **For each segment:**
        - 1. 切割参考音 `ref.wav`。
        - 2. 调用 `tts_wrapper` **以自然语速**生成 `temp_dub.wav`。
        - 3. 调用 `vad_utils.get_speech_timestamps` 分析 `temp_dub.wav`。
        - 4. **If** 语音总长 <= 目标时长, **then** 调用 `audio_utils.adjust_silence` 生成最终的 `seg_dub.wav`。
        - 5. **Else**, 调用 `tts_wrapper` **并传入语速参数**，重新生成 `seg_dub.wav`。
    - **T4.5**: 所有 `seg_dub.wav` 生成后，用`ffmpeg`拼接成一个完整的音轨 `dubbed_vocals.wav`。

---

## T5: 视频生成模块
- **依赖**: T4
- **目标**: 将新生成的音轨和字幕合并到原视频中。
- **子任务**:
    - **T5.1**: 在 `video_tran/video_producer/` 中创建 `srt_utils.py`，定义 `create_srt_file` 函数。
    - **T5.2**: 在 `video_tran/video_producer/run.py` (CLI) 中，调用`ffmpeg`合并音轨、替换视频音轨，并调用`create_srt_file`生成字幕。

---

## T6: 核心编排器
- **依赖**: T0-T5
- **目标**: 串联所有模块，实现端到端的自动化流程。
- **子任务**:
    - **T6.1**: 在 `config.py` 中实现 `load_config(path)`。
    - **T6.2**: 在 `orchestrator.py` 中定义 `Orchestrator` 类。主流程将**直接导入并实例化**各模块的Python类（`AudioProcessor`, `Transcriber`等），并按顺序（T1->T2->T2.5->T3->T4->T5）调用其`run`方法。
    - **T6.3**: 在 `main.py` 中，解析命令行参数，实例化并运行 `Orchestrator`。