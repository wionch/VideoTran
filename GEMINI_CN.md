# Gemini 代码助手上下文

本文档概述了 `VideoTran` 项目，这是一个用于音频和视频处理的基于 Python 的工具集合。该项目采用单一代码库（monorepo）的结构，包含三个主要组件：`audiocraft`、`index-tts` 和 `whisperX`。

## 项目概述

`VideoTran` 项目结合了三个强大的 AI 驱动工具，为语音和音频处理提供了一个完整的流程：

*   **`whisperX`**：用于准确、快速的自动语音识别 (ASR)，并提供词级时间戳和说话人分离功能。
*   **`index-tts`**：用于文本到语音 (TTS) 合成，可将转录的文本生成语音。
*   **`audiocraft`**：用于音频生成，包括音乐和音效，可用于丰富视频的音轨。

这些工具可以单独使用，也可以串联起来创建一个完整的视频翻译和配音流程。例如，可以使用 `whisperX` 转录视频中的音频，翻译文本，然后使用 `index-tts` 生成目标语言的新音频。`audiocraft` 则可用于生成新的配乐或音效。

## 构建和运行

每个子项目都有其自己的一组依赖项和运行说明。建议为每个项目创建一个独立的 Python 虚拟环境，以避免依赖冲突。

### whisperX

*   **用途**：自动语音识别 (ASR)，具有词级时间戳和说话人分离功能。
*   **主要依赖**：`faster-whisper`、`pyannote-audio`、`torch`、`torchaudio`。
*   **运行**：该项目可以使用 `whisperx` 脚本从命令行运行。

```bash
# 示例用法：
whisperx <video_file> --model large-v2 --language en --align-model WAV2VEC2_ASR_LARGE_LV60K_960H --diarize
```

### index-tts

*   **用途**：文本到语音 (TTS) 合成。
*   **主要依赖**：`accelerate`、`transformers`、`gradio`、`cn2an`、`jieba`。
*   **运行**：该项目提供了一个基于 Gradio 的 Web 界面，以供交互使用。

```bash
# 运行 Web 用户界面：
python webui.py
```

### audiocraft

*   **用途**：音频生成，包括音乐和音效。
*   **主要依赖**：`torch`、`transformers`、`gradio`、`encodec`。
*   **运行**：该项目为其不同模型提供了几个基于 Gradio 的演示。

```bash
# 运行 MusicGen 演示：
python demos/musicgen_app.py
```

## 开发约定

*   所有三个项目都基于 Python，并使用 `pip` 进行包管理。
*   每个项目都有其自己的 `requirements.txt` 或 `pyproject.toml` 文件来管理依赖项。
*   这些项目利用了 PyTorch 和 Transformers 等流行的深度学习库。
*   Gradio 用于为模型创建交互式的基于 Web 的演示。

## 交互约束

为了确保工作流程的顺畅和高效，请在我们的交互中遵守以下约束：

1.  **默认语言**: 在未指定语言的情况下，所有文本输出都必须使用中文。
2.  **格式化**: 所有文本内容都应使用 Markdown 格式以保证清晰和可读性。
3.  **代码注释**: 所有生成的代码片段都必须提供详细的中文注释，以解释其逻辑和功能。
4.  **文档完整性**: 修改文档时，不允许出现精简内容。必须始终输出完整的文档，以防止内容丢失。
