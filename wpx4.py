# -*- coding: utf-8 -*-
"""
【更高精度的说话人字幕生成脚本 - wpx4.py】

核心设计:
本脚本是 wpx3 的改进版，旨在解决声线相似说话人识别不准的问题。
采用“引导式日志” (Guided Diarization) 方案，是 pyannote.audio 的标准推荐用法之一。

工作流程:
1.  **生成参考声纹**: 与wpx3相同，为每位已知说话人生成一个高精度的参考声纹向量。
2.  **标准说话人分离 (盲分)**: 使用 `pyannote/speaker-diarization-3.1` 流水线对整个音频进行一次匿名的说话人分离，得到如 `SPEAKER_00`, `SPEAKER_01` 等的时间轴。这一步的切分质量更高。
3.  **建立映射关系**: 遍历每一个匿名说话人（如 `SPEAKER_00`）的所有语音片段，提取其平均声纹，然后与所有已知的参考声纹进行比对，找到最相似的真实说话人，从而建立一个映射表（e.g., {"SPEAKER_00": "主持人"}）。
4.  **应用映射并生成字幕**: 将 `faster-whisper` 生成的原始字幕，与我们刚刚校正过的、带有真实姓名的时间轴进行匹配，为每一句字幕标注上最准确的说话人，并保持原始分句和时间轴不变。
"""

import os
import torch
import torchaudio
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines.utils.hook import ProgressHook
from faster_whisper import WhisperModel
import srt
from datetime import timedelta
from scipy.spatial.distance import cdist

# ==================================================================================
# 任务 1: 生成参考声纹 (与wpx3类似，但作为独立函数)
# ==================================================================================

def generate_reference_embeddings(speaker_samples: dict, embed_model: Model, device: torch.device):
    """
    为已知的说话人生成参考声纹。
    """
    print("---" + "[任务 1/4] 生成参考声纹" + "---")
    ref_embeddings = {}
    with torch.no_grad():
        for speaker_name, sample_path in speaker_samples.items():
            if not os.path.exists(sample_path):
                print(f"!! 警告: 说话人 '{speaker_name}' 的样本文件不存在于: {sample_path}。将跳过此说话人。")
                continue
            
            try:
                print(f"-> 正在为说话人 '{speaker_name}' 生成参考声纹...")
                waveform, _ = torchaudio.load(sample_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                embedding = embed_model(waveform.to(device).unsqueeze(0))
                normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                ref_embeddings[speaker_name] = normalized_embedding
                print(f"-> 说话人 '{speaker_name}' 的参考声纹已生成。")

            except Exception as e:
                print(f"!! 处理说话人 '{speaker_name}' 的样本时出错: {e}")
                continue
    
    if not ref_embeddings:
        print("!! 严重错误: 未能成功生成任何说话人的参考声纹。 ")
        return None

    print("---" + "---" + "[任务 1/4] 完成" + "---" + "---" + "\n")
    return ref_embeddings


# ==================================================================================
# 任务 2: 引导式日志 (Guided Diarization) - [逻辑已修正 v3]
# ==================================================================================

def run_guided_diarization(audio_path: str, ref_embeddings: dict, embed_model: Model, hf_token: str, device: torch.device):
    """
    [逻辑修正 v3] 增加了对过短音频片段的安全检查，避免运行时错误。
    """
    print("---" + "[任务 2/4] 执行引导式日志 (v3 - 带安全检查)" + "---")
    try:
        print("-> 正在加载 pyannote/speaker-diarization-3.1 流水线...")
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diarization_pipeline.to(device)
        print("-> 流水线加载成功。")

        print("-> 正在对完整音频进行匿名分离以获取时间轴...")
        with ProgressHook() as hook:
            diarization_result = diarization_pipeline(audio_path, hook=hook)
        print("-> 匿名分离完成，现在开始逐段识别...")

        final_timeline = []
        full_waveform, sr = torchaudio.load(audio_path)

        for turn, _, _ in diarization_result.itertracks(yield_label=True):
            # 安全检查：跳过时长过短的片段，防止模型报错
            if turn.duration < 0.5:
                print(f"  -> 片段 [{turn.start:.2f}s - {turn.end:.2f}s] 时长过短，标记为 UNKNOWN。")
                final_timeline.append({
                    "speaker": "UNKNOWN",
                    "start": turn.start,
                    "end": turn.end,
                    "distance": 1.0  # 设为最大距离
                })
                continue

            # 提取当前独立片段的波形
            segment_waveform = full_waveform[0, int(turn.start * sr):int(turn.end * sr)].unsqueeze(0)

            # 计算该独立片段的声纹
            with torch.no_grad():
                segment_embedding = embed_model(segment_waveform.to(device).unsqueeze(0))
                segment_embedding = torch.nn.functional.normalize(segment_embedding, p=2, dim=1)

            # 与所有参考声纹计算距离，找到最匹配的说话人
            min_distance = float('inf')
            best_speaker_name = "UNKNOWN"
            for speaker_name, ref_emb in ref_embeddings.items():
                distance = cdist(segment_embedding.cpu().numpy(), ref_emb.cpu().numpy(), metric='cosine')[0, 0]
                if distance < min_distance:
                    min_distance = distance
                    best_speaker_name = speaker_name
            
            # 将识别结果添加到最终时间轴
            final_timeline.append({
                "speaker": best_speaker_name,
                "start": turn.start,
                "end": turn.end,
                "distance": min_distance
            })
            print(f"  -> 片段 [{turn.start:.2f}s - {turn.end:.2f}s]: 分配给 '{best_speaker_name}' (距离: {min_distance:.4f})")

        print("---" + "[任务 2/4] 完成" + "---" + "\n")
        return final_timeline

    except Exception as e:
        print(f"!! 引导式日志过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def transcribe_and_segment_audio(audio_path: str, asr_model: WhisperModel, language: str = 'zh'):
    """
    使用 faster-whisper 模型转录音频并返回带时间戳的句子分段。
    """
    print("---" + "[(并行) 音频转录与分段]" + "---")
    try:
        print(f"-> 正在使用 faster-whisper (VAD) 转录音频: {audio_path}")
        segments, info = asr_model.transcribe(audio_path, language=language, vad_filter=True)
        segments_list = [{"text": s.text, "start": s.start, "end": s.end} for s in segments]
        print(f"-> 转录完成，共 {len(segments_list)} 个分段。")
        return segments_list, info
    except Exception as e:
        print(f"!! 音频转录过程中发生错误: {e}")
        return None, None

# ==================================================================================
# 任务 3: 将高精度时间轴映射到字幕
# ==================================================================================

def map_timeline_to_subtitles(transcription_segments: list, speaker_timeline: list):
    """
    将带有真实姓名的说话人时间轴映射到原始的转录分段上。
    """
    print("---" + "[任务 3/4] 映射时间轴到字幕" + "---")
    final_subtitles = []

    for i, segment in enumerate(transcription_segments):
        sub_start, sub_end = segment['start'], segment['end']
        speaker_durations = {}

        for turn in speaker_timeline:
            turn_start, turn_end = turn['start'], turn['end']
            overlap = max(0, min(sub_end, turn_end) - max(sub_start, turn_start))
            if overlap > 0:
                speaker = turn['speaker']
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap

        if speaker_durations:
            dominant_speaker = max(speaker_durations, key=speaker_durations.get)
        else:
            dominant_speaker = "UNKNOWN"

        content = f"[{dominant_speaker}]: {segment['text'].strip()}"
        new_subtitle = srt.Subtitle(
            index=(i + 1),
            start=timedelta(seconds=sub_start),
            end=timedelta(seconds=sub_end),
            content=content
        )
        final_subtitles.append(new_subtitle)
        print(f"-> 字幕 {i+1}/{len(transcription_segments)}: 分配给 '{dominant_speaker}'")

    print(f"---" + "[任务 3/4] 完成" + "---" + "\n")
    return final_subtitles


# ==================================================================================
# (复用) SRT文件生成函数
# ==================================================================================

def generate_srt_file(final_subtitles: list, output_srt_path: str):
    """
    将最终的字幕条目列表写入到SRT文件中。
    """
    print("---" + "[任务 4/4] 生成最终SRT文件" + "---")
    try:
        final_srt_content = srt.compose(final_subtitles)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(final_srt_content)
        print(f"-> 成功生成SRT文件: {output_srt_path}")
        return True
    except Exception as e:
        print(f"!! 写入SRT文件时发生错误: {e}")
        return False

# ==================================================================================
# 主流程编排
# ==================================================================================

def run_wpx4_pipeline(audio_path: str, speaker_samples: dict, output_path: str, hf_token: str):
    """
    执行 wpx4 的完整流程。
    """
    print("=== 开始执行 wpx4 高精度说话人分离流程 ===")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    try:
        embed_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        embed_model.to(device)
        asr_model = WhisperModel("large-v2", device=str(device), compute_type="float16" if use_cuda else "int8")
    except Exception as e:
        print(f"!! 模型加载失败: {e}")
        return

    ref_embeddings = generate_reference_embeddings(speaker_samples, embed_model, device)
    if not ref_embeddings:
        print("=== 流程因参考声纹生成失败而中止 ===")
        return

    speaker_timeline = run_guided_diarization(audio_path, ref_embeddings, embed_model, hf_token, device)
    if not speaker_timeline:
        print("=== 流程因引导式日志失败而中止 ===")
        return

    transcription_segments, _ = transcribe_and_segment_audio(audio_path, asr_model)
    if not transcription_segments:
        print("=== 流程因音频转录失败而中止 ===")
        return

    final_subtitles = map_timeline_to_subtitles(transcription_segments, speaker_timeline)
    if not final_subtitles:
        print("=== 流程因字幕映射失败而中止 ===")
        return

    if generate_srt_file(final_subtitles, output_path):
        print(f"\n=== 流程成功执行完毕！最终文件已保存至: {output_path} ===")
    else:
        print("=== 流程因文件写入失败而中止 ===")


if __name__ == '__main__':
    # ###################################################################
    # # 1. 配置区域
    # ###################################################################
    HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
    AUDIO_FILE = r"D:\Python\Project\VideoTran\videos\Vocals.wav"
    SPEAKER_SAMPLES = {
        "嘉宾": r"D:\Python\Project\VideoTran\output\a.wav",
        "主持人": r"D:\Python\Project\VideoTran\output\b.wav"
    }
    OUTPUT_SRT_PATH = os.path.join(os.path.dirname(AUDIO_FILE), "..", "output", "Vocals_wpx_4_guided.srt")
    os.makedirs(os.path.dirname(OUTPUT_SRT_PATH), exist_ok=True)

    # ###################################################################
    # # 2. 执行流程
    # ###################################################################
    if not HUGGING_FACE_TOKEN:
        print("!! 严重错误: 请设置 HUGGING_FACE_TOKEN 环境变量。流程无法继续。" )
    else:
        run_wpx4_pipeline(
            audio_path=AUDIO_FILE,
            speaker_samples=SPEAKER_SAMPLES,
            output_path=OUTPUT_SRT_PATH,
            hf_token=HUGGING_FACE_TOKEN
        )
