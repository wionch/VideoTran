# -*- coding: utf-8 -*-
"""
【高精度说话人字幕生成脚本 - wpx3.py】

核心设计:
本脚本采用“强监督、段落级声纹匹配”方案，旨在解决无监督说话人分离准确率不高的问题。

工作流程:
1.  **加载与初始化**: 加载 `faster-whisper` 转录模型和 `pyannote` 声纹模型，并根据用户提供的音频样本，为每位已知说话人生成一个高精度的参考声纹向量。
2.  **转录与分段**: 使用 `faster-whisper` 对整个音频进行转录，利用其VAD（Voice Activity Detection）功能，将音频切分为一个个独立的语音段落（句子）。
3.  **逐段匹配**: 遍历每一个语音段落，提取其对应的音频数据，并计算出该段落的声纹。
4.  **相似度比对**: 将段落声纹与所有已知的参考声纹进行余弦相似度比对，将相似度最高的说话人身份赋予该段落。
5.  **结果优化与合并**: 对初步标记的结果进行平滑处理，消除孤立的错误识别，然后将连续的、属于同一说话人的段落合并，形成最终的字幕条目。
6.  **生成SRT**: 将优化后的字幕条目列表，使用 `srt` 库生成为格式标准的 `.srt` 文件。
"""

import os
import torch
import torchaudio
from pyannote.audio import Model
from faster_whisper import WhisperModel
import srt
from datetime import timedelta

# ==================================================================================
# 任务 1: `01_setup_and_init` - 环境设置与模型加载
# ==================================================================================

def initialize_models_and_generate_embeddings(speaker_samples: dict, hf_token: str):
    """
    初始化所有需要的模型，并为已知的说话人生成参考声纹。
    """
    print("--- [任务 1/6] 初始化模型与参考声纹 ---")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    compute_type = "float16" if use_cuda else "int8"
    print(f"-> 使用设备: {device}, 计算类型: {compute_type}")
    if use_cuda:
        print(f"-> 检测到 GPU: {torch.cuda.get_device_name(0)}")

    models = {}
    try:
        print("-> 正在加载 faster-whisper ASR 模型 (large-v2)...")
        models['asr_model'] = WhisperModel("large-v2", device=str(device), compute_type=compute_type)
        print("-> 正在加载 pyannote/embedding 声纹模型...")
        if not hf_token:
            raise ValueError("错误: 未提供 Hugging Face Token，无法加载 pyannote/embedding 声纹模型。" )
        models['embed_model'] = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        if models['embed_model'] is None:
             raise RuntimeError("错误: pyannote/embedding 模型加载失败，返回值为 None。请检查 Token 和网络。" )
        models['embed_model'].to(device)
        print("-> 所有模型加载成功。")
    except Exception as e:
        print(f"!! 模型加载过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    ref_embeddings = {}
    with torch.no_grad():
        for speaker_name, sample_path in speaker_samples.items():
            if not os.path.exists(sample_path):
                print(f"!! 警告: 说话人 '{speaker_name}' 的样本文件不存在于: {sample_path}。将跳过此说话人。" )
                continue
            try:
                print(f"-> 正在为说话人 '{speaker_name}' 生成参考声纹...")
                waveform, _ = torchaudio.load(sample_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                embedding = models['embed_model'](waveform.to(device).unsqueeze(0))
                normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                ref_embeddings[speaker_name] = normalized_embedding
                print(f"-> 说话人 '{speaker_name}' 的参考声纹已生成。" )
            except Exception as e:
                print(f"!! 处理说话人 '{speaker_name}' 的样本时出错: {e}")
                continue
    if not ref_embeddings:
        print("!! 严重错误: 未能成功生成任何说话人的参考声纹。流程中止。" )
        return None, None, None
    print("--- [任务 1/6] 完成 ---\n")
    return models, ref_embeddings, device

# ==================================================================================
# 任务 2: `02_transcribe` - 音频转录与分段
# ==================================================================================

def transcribe_and_segment_audio(audio_path: str, asr_model: WhisperModel, language: str = 'zh'):
    """
    使用 faster-whisper 模型转录音频并返回带时间戳的句子分段。
    """
    print("--- [任务 2/6] 音频转录与分段 ---")
    try:
        print(f"-> 正在使用 faster-whisper (VAD) 转录音频: {audio_path}")
        segments, info = asr_model.transcribe(audio_path, language=language, vad_filter=True)
        segments_list = [{"text": s.text, "start": s.start, "end": s.end} for s in segments]
        print(f"-> 转录完成。检测到语言: {info.language} (置信度: {info.language_probability:.2f})，共 {len(segments_list)} 个分段。" )
        print("-> 正在加载完整音频波形...")
        full_audio_waveform, _ = torchaudio.load(audio_path)
        print("-> 音频波形加载成功。" )
        print("--- [任务 2/6] 完成 ---\n")
        return segments_list, full_audio_waveform
    except Exception as e:
        print(f"!! 音频转录过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================================================================================
# 任务 3: `03_match_speakers` - 核心声纹匹配逻辑
# ==================================================================================

def match_speakers_to_segments(segments: list, full_audio_waveform: torch.Tensor, ref_embeddings: dict, embed_model: Model, device: torch.device, similarity_threshold: float = 0.5):
    """
    为每个语音分段匹配最相似的说话人。
    """
    print("--- [任务 3/6] 核心声纹匹配 ---")
    annotated_segments = []
    sample_rate = 16000
    with torch.no_grad():
        for i, segment in enumerate(segments):
            start_time, end_time = segment['start'], segment['end']
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            segment_waveform = full_audio_waveform[0, start_frame:end_frame].unsqueeze(0)
            if segment_waveform.shape[1] < sample_rate * 0.5:
                segment['speaker'] = "UNKNOWN"
                segment['similarity'] = 0.0
                annotated_segments.append(segment)
                print(f"-> 分段 {i+1}/{len(segments)} 过短，跳过匹配。" )
                continue
            segment_embedding = embed_model(segment_waveform.to(device).unsqueeze(0))
            segment_embedding = torch.nn.functional.normalize(segment_embedding, p=2, dim=1)
            max_similarity = -1.0
            best_speaker = "UNKNOWN"
            for speaker_name, ref_emb in ref_embeddings.items():
                similarity = torch.nn.functional.cosine_similarity(segment_embedding, ref_emb).item()
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_speaker = speaker_name
            if max_similarity >= similarity_threshold:
                segment['speaker'] = best_speaker
                segment['similarity'] = max_similarity
            else:
                segment['speaker'] = "UNKNOWN"
                segment['similarity'] = max_similarity
            annotated_segments.append(segment)
            print(f"-> 分段 {i+1}/{len(segments)}: 匹配到 '{segment['speaker']}' (相似度: {segment['similarity']:.4f})")
    print("--- [任务 3/6] 完成 ---\n")
    return annotated_segments

# ==================================================================================
# 任务 4: `04_refine_results` - 结果平滑与格式化 (已修改)
# ==================================================================================

def refine_and_format_segments(annotated_segments: list, smoothing_window: int = 1):
    """
    [格式修正] 对已标注的分段进行平滑处理，并为每个分段生成独立的字幕条目（不合并）。

    Args:
        annotated_segments (list): 来自任务3的已标注分段列表。
        smoothing_window (int, optional): 平滑窗口大小。默认为 1。

    Returns:
        list: 一个最终的字幕条目列表，每个条目是一个 srt.Subtitle 对象。
    """
    print("--- [任务 4/6] 结果平滑与格式化 ---")
    if not annotated_segments:
        return []

    # 1. 平滑处理 (逻辑保持不变)
    num_segments = len(annotated_segments)
    if num_segments > 2 * smoothing_window:
        print(f"-> 正在进行平滑处理，窗口大小: {smoothing_window}")
        for i in range(smoothing_window, num_segments - smoothing_window):
            current_speaker = annotated_segments[i]['speaker']
            prev_speakers = {annotated_segments[j]['speaker'] for j in range(i - smoothing_window, i)}
            next_speakers = {annotated_segments[j]['speaker'] for j in range(i + 1, i + 1 + smoothing_window)}
            if len(prev_speakers) == 1 and len(next_speakers) == 1 and list(prev_speakers)[0] == list(next_speakers)[0]:
                neighbor_speaker = list(prev_speakers)[0]
                if current_speaker != neighbor_speaker:
                    print(f"  -> 平滑修正: 分段 {i+1} 的说话人从 '{current_speaker}' 修正为 '{neighbor_speaker}'")
                    annotated_segments[i]['speaker'] = neighbor_speaker

    # 2. 逐句格式化 (新逻辑，不再合并)
    print("-> 正在为每个分段生成独立的字幕条目...")
    final_subtitles = []
    for i, segment in enumerate(annotated_segments):
        speaker_tag = f"[{segment['speaker']}]"
        content = f"{speaker_tag}: {segment['text'].strip()}"
        
        new_subtitle = srt.Subtitle(
            index=(i + 1),
            start=timedelta(seconds=segment['start']),
            end=timedelta(seconds=segment['end']),
            content=content
        )
        final_subtitles.append(new_subtitle)

    print(f"-> 格式化完成，共生成 {len(final_subtitles)} 个独立的字幕条目。")
    print("--- [任务 4/6] 完成 ---\n")
    return final_subtitles

# ==================================================================================
# 任务 5: `05_generate_srt` - SRT文件生成
# ==================================================================================

def generate_srt_file(final_subtitles: list, output_srt_path: str):
    """
    将最终的字幕条目列表写入到SRT文件中。
    """
    print("--- [任务 5/6] 生成最终SRT文件 ---")
    try:
        final_srt_content = srt.compose(final_subtitles)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(final_srt_content)
        print(f"-> 成功生成SRT文件: {output_srt_path}")
        print("--- [任务 5/6] 完成 ---\n")
        return True
    except Exception as e:
        print(f"!! 写入SRT文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================================================================================
# 任务 6: `06_create_wpx3` - 组装完整脚本
# ==================================================================================

def run_speaker_diarization_pipeline(audio_path: str, speaker_samples: dict, output_path: str, hf_token: str):
    """
    执行完整的高精度说话人分离与字幕生成流程。
    """
    print("=== 开始执行高精度说话人分离流程 (wpx3) ===")
    models, ref_embeddings, device = initialize_models_and_generate_embeddings(speaker_samples, hf_token)
    if not models:
        print("=== 流程因模型初始化失败而中止 ===")
        return
    segments, waveform = transcribe_and_segment_audio(audio_path, models['asr_model'])
    if not segments:
        print("=== 流程因音频转录失败而中止 ===")
        return
    annotated_segments = match_speakers_to_segments(segments, waveform, ref_embeddings, models['embed_model'], device)
    if not annotated_segments:
        print("=== 流程因声纹匹配失败而中止 ===")
        return
    # 更新函数调用
    final_subtitles = refine_and_format_segments(annotated_segments)
    if not final_subtitles:
        print("=== 流程因结果格式化失败而中止 ===")
        return
    success = generate_srt_file(final_subtitles, output_path)
    if success:
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
    OUTPUT_SRT_PATH = os.path.join(os.path.dirname(AUDIO_FILE), "..", "output", "Vocals_wpx_3_refined.srt")
    os.makedirs(os.path.dirname(OUTPUT_SRT_PATH), exist_ok=True)

    # ###################################################################
    # # 2. 执行流程
    # ###################################################################
    if not HUGGING_FACE_TOKEN:
        print("!! 严重错误: 请设置 HUGGING_FACE_TOKEN 环境变量。流程无法继续。" )
    else:
        run_speaker_diarization_pipeline(
            audio_path=AUDIO_FILE,
            speaker_samples=SPEAKER_SAMPLES,
            output_path=OUTPUT_SRT_PATH,
            hf_token=HUGGING_FACE_TOKEN
        )
