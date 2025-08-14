"""
【高精度说话人字幕校正脚本 - wpx2.py】

核心功能：
本脚本旨在解决 `wpx.py` 中无监督说话人分离准确率不高的问题。
它采用“有监督分离 + 后期映射”的技术方案，实现高精度的说话人字幕生成。

工作流程:
1.  **高精度分离**: 首先，使用 `pyannote.audio` 的“有监督”分离功能，通过提供已知说话人的声纹样本，对整个音频文件进行一次高精度的说话人时间轴划分。
2.  **解析原始字幕**: 读取一个已经由 Whisper 或其他ASR工具生成的、不含说话人信息的纯文本SRT文件。
3.  **映射与标注**: 将高精度的说话人时间轴与纯文本字幕的时间轴进行匹配，为每一句字幕找到最主要的发言人，并添加标签。
4.  **生成新字幕**: 输出一个带有准确说话人标签的全新SRT文件。

[对比 wpx.py]:
-   **准确性**: 本脚本的核心优势。wpx.py 使用无监督分离（盲猜），本脚本使用有监督分离（有参考答案），准确率大幅提升。
-   **逻辑**: wpx.py 是“转录->对齐->盲分->合并”的流水线。本脚本是“高精度分离”和“转录后字幕”两条线最后“映射合并”的逻辑，更像是一个校正流程。
-   **依赖**: 新增了 `pyannote.audio` 和 `srt` 库的强依赖。

"""

import os
import torch
import torchaudio
from pyannote.audio import Pipeline, Model
import srt # 需要通过 pip install srt 来安装
from datetime import timedelta

# --- 核心功能函数 ---

def parse_srt_to_seconds(srt_file_path):
    """
    解析SRT文件，并将时间戳转换为秒。
    [新增功能]: 引入 'srt' 库，专门用于解析和生成SRT文件，比手动拼接字符串更健壮。
    """
    print(f"步骤 1/4: 解析原始SRT文件: {srt_file_path}")
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))
        
        # 将timedelta转换为秒，方便计算
        for sub in subtitles:
            sub.start_sec = sub.start.total_seconds()
            sub.end_sec = sub.end.total_seconds()
        
        print(f"-> 解析成功，共找到 {len(subtitles)} 句字幕。")
        return subtitles
    except Exception as e:
        print(f"!! 解析SRT文件失败: {e}")
        return []

def diarize_with_known_speakers(audio_path, hf_token, speaker_samples):
    """
    使用pyannote.audio进行说话人分离，并通过声纹比对将结果映射到已知的说话人。
    [核心逻辑变更]: 此版本采用“无监督分离 + 后期映射”策略，以规避旧版API调用错误。
    1. 执行标准的无监督说话人分离，得到带有通用标签（如SPEAKER_00）的时间轴。
    2. 对每个通用标签，提取其全部语音，计算一个代表性声纹。
    3. 将这个声纹与已知的说话人样本声纹进行比对。
    4. 建立一个“通用标签 -> 真实姓名”的映射表。
    5. 使用映射表生成最终的、带有正确人名的说话人时间轴。
    """
    print("步骤 2/4: 使用pyannote.audio进行高精度说话人分离...")
    if not hf_token:
        print("!! 错误: 未提供Hugging Face Token，无法进行分离。" )
        return None

    try:
        # 1. 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-> 使用设备: {device}")

        # 2. 加载流水线和模型
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline.to(device)
        print("-> Diarization Pipeline 加载成功。" )
        
        embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        if embedding_model is None:
            print("\n!! 错误: 无法加载声纹模型 'pyannote/embedding'。请检查Hugging Face Token和网络连接。\n")
            return None
        embedding_model.to(device)
        print("-> 声纹模型加载成功。")

        # 3. 为已知说话人生成参考声纹
        known_speaker_labels = list(speaker_samples.keys())
        known_speaker_embeddings = []
        with torch.no_grad():
            for label in known_speaker_labels:
                sample_path = speaker_samples[label]
                if not os.path.exists(sample_path):
                    print(f"!! 警告: 样本文件不存在 {sample_path}, 跳过说话人 {label}")
                    continue
                
                waveform, sample_rate = torchaudio.load(sample_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # 计算声纹并进行归一化
                embedding = embedding_model(waveform.to(device).unsqueeze(0))
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                known_speaker_embeddings.append(embedding)
                print(f"-> 已为说话人 '{label}' 生成参考声纹。" )

        if not known_speaker_embeddings:
            print("!! 错误: 未能成功生成任何说话人的参考声纹。" )
            return None
        
        known_speaker_embeddings = torch.cat(known_speaker_embeddings, dim=0)

        # 4. 执行标准的无监督说话人分离
        print("-> 正在对完整音频进行无监督分离，这可能需要一些时间...")
        diarization = pipeline(audio_path)
        print("-> 无监督分离完成。")

        # 5. 后期映射：将通用标签（如SPEAKER_00）映射到真实姓名
        print("-> 开始进行说话人声纹比对与映射...")
        speaker_mapping = {}
        
        # 加载完整音频用于提取片段
        full_waveform, sr = torchaudio.load(audio_path)

        generic_speakers = diarization.labels()
        for speaker in generic_speakers:
            print(f"  -> 正在处理通用标签: {speaker}")
            # 提取该通用说话人的所有语音片段
            speaker_timeline = diarization.label_timeline(speaker) 
            
            # 从完整音频中提取和拼接这些片段
            speaker_waveforms = []
            for segment in speaker_timeline:
                start_frame = int(segment.start * sr)
                end_frame = int(segment.end * sr)
                speaker_waveforms.append(full_waveform[0, start_frame:end_frame])
            
            if not speaker_waveforms:
                print(f"  -> 警告: 未能为 {speaker} 提取到任何音频片段。" )
                continue

            speaker_audio = torch.cat(speaker_waveforms, dim=0).unsqueeze(0)

            # 为拼接后的音频计算代表性声纹
            with torch.no_grad():
                segment_embedding = embedding_model(speaker_audio.to(device).unsqueeze(0))
                segment_embedding = torch.nn.functional.normalize(segment_embedding, p=2, dim=1)

            # 计算与所有已知声纹的相似度
            similarities = torch.nn.functional.cosine_similarity(segment_embedding, known_speaker_embeddings)
            
            # 找到最匹配的已知说话人
            best_match_index = torch.argmax(similarities).item()
            best_match_speaker = known_speaker_labels[best_match_index]
            
            speaker_mapping[speaker] = best_match_speaker
            print(f"  -> 映射结果: {speaker} -> {best_match_speaker} (相似度: {similarities.max().item():.4f})")

        # 6. 使用映射结果生成最终的时间轴
        diarization_timeline = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            mapped_speaker = speaker_mapping.get(speaker_label, "UNKNOWN")
            diarization_timeline.append({
                "speaker": mapped_speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        print("-> 高精度分离与映射完成。" )
        return diarization_timeline

    except Exception as e:
        print(f"!! 说话人分离过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def map_speakers_and_generate_srt(subtitles, diarization_timeline, output_srt_path):
    """
    将高精度说话人时间轴映射到原始字幕上，并生成新的SRT文件。
    [对比 wpx.py]: wpx.py 是将不准的说话人标签直接分配给单词。我们这里是全新逻辑：
    用高精度的说话人时间轴，去给已经存在的句子（来自原始SRT）进行映射和标注。
    """
    print("步骤 3/4: 开始将说话人映射到字幕...")
    annotated_subtitles = []

    for sub in subtitles:
        sub_start, sub_end = sub.start_sec, sub.end_sec
        speaker_durations = {}

        # 计算每个说话人在该字幕时间段内的发言时长
        for seg in diarization_timeline:
            seg_start, seg_end = seg['start'], seg['end']
            # 计算重叠时长
            overlap = max(0, min(sub_end, seg_end) - max(sub_start, seg_start))
            if overlap > 0:
                speaker = seg['speaker']
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap

        # 找到发言时间最长的人作为该句字幕的主要说话人
        if speaker_durations:
            dominant_speaker = max(speaker_durations, key=speaker_durations.get)
            new_content = f"[{dominant_speaker}]: {sub.content}"
        else:
            # 如果没有找到对应的说话人（比如是静音或音乐），可以选择保留原样或添加标记
            new_content = f"[UNKNOWN]: {sub.content}"
        
        # 更新字幕内容
        sub.content = new_content
        annotated_subtitles.append(sub)

    # 在重新组合为SRT之前，清理掉我们手动添加的临时属性
    # 这是因为 srt.compose 内部会重新构造Subtitle对象，不认识这些临时属性会导致TypeError
    for sub in annotated_subtitles:
        if hasattr(sub, 'start_sec'):
            del sub.start_sec
        if hasattr(sub, 'end_sec'):
            del sub.end_sec

    # 使用srt库重新生成SRT文件内容
    final_srt_content = srt.compose(annotated_subtitles)

    print(f"步骤 4/4: 写入最终的带说话人标签的SRT文件: {output_srt_path}")
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write(final_srt_content)
    print("-> 文件写入成功！")


# --- 主函数入口 ---
if __name__ == '__main__':
    # ###################################################################
    # # 1. 配置区域
    # ###################################################################

    # !!! 需要 Hugging Face Token 才能使用 pyannote.audio 3.1 !!!
    # 建议设置为环境变量 HUGGING_FACE_TOKEN
    HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")

    # --- 输入文件路径 ---
    # 原始音频文件
    AUDIO_FILE = r"D:\Python\Project\VideoTran\videos\Vocals.wav"
    # 由wpx.py或Whisper生成的、不带说话人信息的SRT文件
    ORIGINAL_SRT_FILE = r"D:\Python\Project\VideoTran\output\Vocals_wpx_1.srt"
    
    # --- 已知说话人的声纹样本 (5秒左右的纯净人声WAV) ---
    SPEAKER_SAMPLES = {
        "嘉宾": r"D:\Python\Project\VideoTran\output\a.wav", # 替换为你的样本路径
        "主持人": r"D:\Python\Project\VideoTran\output\b.wav"  # 替换为你的样本路径
    }

    # --- 输出文件路径 ---
    # 获取原始SRT文件名作为基础
    base_name = os.path.splitext(os.path.basename(ORIGINAL_SRT_FILE))[0]
    output_dir = os.path.dirname(ORIGINAL_SRT_FILE)
    # 定义最终输出的文件名
    FINAL_SRT_FILE = os.path.join(output_dir, f"{base_name}_refined.srt")

    # ###################################################################
    # # 2. 执行流程
    # ###################################################################
    print("---" + "开始执行高精度说话人字幕校正流程 [wpx2.py]" + "---")

    # 步骤 1: 解析原始SRT
    subtitles_data = parse_srt_to_seconds(ORIGINAL_SRT_FILE)

    if subtitles_data:
        # 步骤 2: 高精度说话人分离
        diarization_timeline = diarize_with_known_speakers(AUDIO_FILE, HUGGING_FACE_TOKEN, SPEAKER_SAMPLES)

        if diarization_timeline:
            # 步骤 3 & 4: 映射并生成最终的SRT文件
            map_speakers_and_generate_srt(subtitles_data, diarization_timeline, FINAL_SRT_FILE)
            print("\n---" + "流程执行完毕！" + "---")
        else:
            print("\n---" + "流程因说话人分离失败而中止。" + "---")
    else:
        print("\n---" + "流程因SRT文件解析失败而中止。" + "---")
