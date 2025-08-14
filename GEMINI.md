
## Project Overview

The `VideoTran` project combines three powerful AI-driven tools to provide a complete pipeline for speech and audio processing:

- **`whisperX`**: for accurate and fast automatic speech recognition (ASR) with word-level timestamps and speaker diarization.
- **`index-tts`**: for text-to-speech (TTS) synthesis, allowing for the generation of speech from transcribed text.
- **`audiocraft`**: for audio generation, including music and sound effects, which can be used to enrich the audio track of a video.

These tools can be used individually or chained together to create a complete video translation and dubbing pipeline. For example, one could use `whisperX` to transcribe the audio from a video, translate the text, and then use `index-tts` to generate new audio in the target language. `audiocraft` could then be used to generate a new soundtrack or sound effects.

## Building and Running

Each of the sub-projects has its own set of dependencies and instructions for running. It is recommended to create a separate Python virtual environment for each project to avoid dependency conflicts.

### whisperX

- **Purpose**: Automatic Speech Recognition (ASR) with word-level timestamps and speaker diarization.
- **Key Dependencies**: `faster-whisper`, `pyannote-audio`, `torch`, `torchaudio`.
- **Running**: The project can be run from the command line using the `whisperx` script.

```bash
# Example usage:
whisperx <video_file> --model large-v2 --language en --align-model WAV2VEC2_ASR_LARGE_LV60K_960H --diarize
```

### index-tts

- **Purpose**: Text-to-Speech (TTS) synthesis.
- **Key Dependencies**: `accelerate`, `transformers`, `gradio`, `cn2an`, `jieba`.
- **Running**: The project provides a Gradio-based web interface for interactive use.

```bash
# To run the web UI:
python webui.py
```

### audiocraft

- **Purpose**: Audio generation, including music and sound effects.
- **Key Dependencies**: `torch`, `transformers`, `gradio`, `encodec`.
- **Running**: The project includes several Gradio-based demos for its different models.

```bash
# To run the MusicGen demo:
python demos/musicgen_app.py
```

## Development Conventions

- All three projects are Python-based and use `pip` for package management.
- Each project has its own `requirements.txt` or `pyproject.toml` file to manage dependencies.
- The projects make use of popular deep learning libraries like PyTorch and Transformers.
- Gradio is used to create interactive web-based demos for the models.

