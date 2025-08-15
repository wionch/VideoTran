# VideoTran - Automated Video Translation and Dubbing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, AI-powered tool to automatically translate the speech in a video to a different language, preserving the original speaker's voice.

## Overview

VideoTran is a powerful command-line tool that automates the entire video dubbing process. It takes a video file with speech in a source language (e.g., Chinese), and produces a new video where the speech is replaced with a target language (e.g., English), complete with accurate voice cloning and synchronized subtitles.

The pipeline is built with a modular architecture, leveraging state-of-the-art AI models for each step of the process.

## Core Features

- **End-to-End Automation**: Go from a source video to a dubbed video with a single command.
- **Segment-Level Voice Cloning**: Utilizes an "island" strategy to process each speech segment independently, ensuring 100% character-identity correctness while preserving the voice timbre.
- **Intelligent Lip-Sync**: Employs an innovative VAD (Voice Activity Detection) hybrid strategy to intelligently adjust the duration of translated speech, maximizing alignment with the original lip movements.
- **High-Quality Translation & Correction**: Integrates Large Language Models (LLMs) to not only provide accurate translations but also to refine and correct the initial speech-to-text transcription.
- **Modular Architecture**: The system is built as a modular library, making it easy to maintain, extend, or test individual components.

## Installation

This project is designed to run in a single, consolidated Conda environment.

### Prerequisites

- **Python**: Version `3.10` or higher.
- **Conda**: For managing the Python environment.
- **FFmpeg**: A powerful audio/video processing tool. Download it from the [official website](https://ffmpeg.org/download.html) and **ensure its executable path is added to your system's `Path` environment variable**.

### Setup Steps

1.  **Create and Activate Conda Environment**:
    Open your terminal and run the following commands:
    ```bash
    conda create -n videotran_env python=3.10 -y
    conda activate videotran_env
    ```

2.  **Install Dependencies**:
    In the activated `videotran_env` environment, run the following commands to install all necessary packages.

    ```bash
    # 1. Install PyTorch (GPU version recommended, adjust for your CUDA version)
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

    # 2. Install whisperX
    pip install git+https://github.com/m-bain/whisperX.git

    # 3. Install OpenVoice
    pip install openvoice-cli

    # 4. Install other dependencies
    pip install moviepy imageio-ffmpeg "audio-separator[gpu]" pyyaml webrtcvad pydub ollama
    ```

3.  **Set API Keys**:
    The project requires API keys for LLM services. Set them as environment variables. For example, in Windows PowerShell:
    ```powershell
    # If using DeepSeek
    $env:DEEPSEEK_TOKEN="your_deepseek_api_key"

    # If using another service, set the corresponding variable
    ```

4.  **Prepare Configuration File**:
    - Navigate to the `configs/` directory.
    - Make a copy of `config.yaml.template` and rename it to `config.yaml`.
    - Edit `config.yaml` to select your desired `translator` provider and other settings if needed.

## Usage

Ensure you have activated the Conda environment before running the tool:
```bash
conda activate videotran_env
```

Execute the main script from the project root directory with the following command:

```bash
python main.py -i <path_to_video> -sl <source_lang> -tl <target_lang> -m <mode>
```

### Arguments

- `-i, --video_path`: **(Required)** Full path to your source video file.
- `-sl, --source_language`: **(Required)** Source language of the video (e.g., `en`, `zh`).
- `-tl, --target_language`: **(Required)** Target language for translation (e.g., `zh`, `en`).
- `-c, --config_path`: (Optional) Path to the configuration file. Defaults to `configs/config.yaml`.
- `-m, --mode`: (Optional) Processing mode. Defaults to `dub`.
    - `transcribe`: Executes transcription only.
    - `translate`: Executes up to the translation step.
    - `dub`: Executes the full pipeline to generate a dubbed video.

### Example

```bash
python main.py -i "C:\videos\my_talk.mp4" -sl "en" -tl "zh" -m "dub"
```

## Output

The final files will be placed in the `output/` directory (or as configured in your `config.yaml`). Depending on the mode, this can include transcribed text (`.json`), translated text (`.json`), and the final dubbed video (`.mp4`) and subtitle file (`.srt`).

A temporary workspace folder (`videotran_*`) will also be created in the project root for intermediate files.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
