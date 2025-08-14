"""
【最终混合式方案 V7 - 带对齐功能】
新的说话人识别与文本生成模块。

本模块采用混合式架构，各取所长：
1. 转录: 使用 faster-whisper 进行，以获得最准确的文本和断句。
2. 对齐/说话人分离: 将 faster-whisper 的准确结果传递给 whisperx 的后续模块处理。
"""
import os
import torch
import whisperx
from faster_whisper import WhisperModel
import whisperx.diarize
from whisperx.utils import get_writer

import sys
print("---" + "[DEBUG]" + "---")
print(f"Python Executable: {sys.executable}")
print(f"PyTorch Version: {torch.__version__}")
print("---" + "[DEBUG]" + "---")

def recognize_speakers_and_generate_text(audio_path, hf_token):
    """
    使用混合式方案执行转录和后续处理。
    """
    # --- 1. 参数设置 ---
    use_cuda_env = os.getenv('USE_CUDA', '1')
    use_cuda = (use_cuda_env == '1') and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    compute_type = 'float16' if use_cuda else 'int8'
    language = 'zh'

    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在于路径: {audio_path}")
        return

    output_dir = os.path.join(os.path.dirname(os.path.dirname(audio_path)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # --- 2. 转录与保存 ---
        print(f"---" + "使用设备: {device}, 计算类型: {compute_type}" + "---")
        if use_cuda:
            print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")

        print("步骤 1/4: 使用 faster-whisper 加载模型并执行转录...")
        fw_model = WhisperModel("large-v2", device=device, compute_type=compute_type)
        segments, info = fw_model.transcribe(audio_path, language=language, vad_filter=True)
        
        # 将生成器转为列表以复用
        segments_list = list(segments) 
        transcription_result = {"segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments_list], "language": info.language}
        print(f"转录完成。检测到语言: {info.language} (置信度: {info.language_probability:.2f})")

        # --- 3. 对齐阶段 (使用 whisperx) ---
        print("\n步骤 2/4: 使用 whisperx 加载对齐模型并对齐结果...")
        align_model, align_metadata = whisperx.load_align_model(language_code=info.language, device=device)
        # 注意：这里传递 transcription_result['segments'] 而不是原始的 segments_list
        result_aligned = whisperx.align(transcription_result['segments'], align_model, align_metadata, audio_path, device)
        result_aligned["language"] = info.language  # Manually add language info
        print("对齐完成。")

        # --- 4. 保存结果文件 ---
        print("\n步骤 3/5: 初始化文件写入器...")
        srt_writer = get_writer("srt", output_dir)
        json_writer = get_writer("json", output_dir)

        # 获取不带扩展名的基本文件名，用于构造新的输出文件名
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]

        # 定义通用SRT写入选项
        writer_options = {
            "max_line_width": None,
            "max_line_count": None,
            "highlight_words": False
        }

        # 4.1 保存原始转录结果 (Volcal_wpx_1.srt)
        temp_audio_path_1 = os.path.join(os.path.dirname(audio_path), f"{audio_basename}_wpx_1.wav")
        srt_writer(transcription_result, temp_audio_path_1, writer_options)
        print(f"-> 原始转录结果已保存到: {os.path.join(output_dir, f'{audio_basename}_wpx_1.srt')}")

        # 4.2 保存对齐后的句子级别结果 (Volcal_wpx_2.srt)
        temp_audio_path_2 = os.path.join(os.path.dirname(audio_path), f"{audio_basename}_wpx_2.wav")
        srt_writer(result_aligned, temp_audio_path_2, writer_options)
        print(f"-> 句子级对齐结果已保存到: {os.path.join(output_dir, f'{audio_basename}_wpx_2.srt')}")

        # 4.3 保存对齐后的词级别JSON结果 (Volcal_wpx_3.json)
        temp_audio_path_3 = os.path.join(os.path.dirname(audio_path), f"{audio_basename}_wpx_3.wav")
        json_writer(result_aligned, temp_audio_path_3, {})
        print(f"-> 词级对齐JSON结果已保存到: {os.path.join(output_dir, f'{audio_basename}_wpx_3.json')}")

        # --- 5. 说话人分离阶段 ---
        print("\n步骤 4/5: 使用 whisperx 进行说话人分离...")
        if hf_token:
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio_path, num_speakers=2)
            result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)
            result_final["language"] = info.language # Add language for writer

            # 5.1 保存带说话人标签的SRT (Volcal_wpx_4.srt)
            temp_audio_path_4 = os.path.join(os.path.dirname(audio_path), f"{audio_basename}_wpx_4.wav")
            srt_writer(result_final, temp_audio_path_4, writer_options)
            print(f"-> 说话人分离结果已保存到: {os.path.join(output_dir, f'{audio_basename}_wpx_4.srt')}")
        else:
            print("-> 未提供Hugging Face Token，跳过说话人分离步骤。" )

        print("\n步骤 5/5: 所有文件生成完毕。")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # ###################################################################
    # # 提示：说话人分离功能需要 Hugging Face Token
    # # HUGGING_FACE_TOKEN = "hf_YOUR_TOKEN_HERE"
    # ###################################################################
    huggingface_token = os.environ.get("HUGGING_FACE_TOKEN")

    HUGGING_FACE_TOKEN = huggingface_token 

    test_audio_file = r"D:\Python\Project\VideoTran\videos\Vocals.wav"
    print("---" + "开始独立测试 wpx.py [V7 - 带对齐功能]" + "---")
    recognize_speakers_and_generate_text(test_audio_file, hf_token=HUGGING_FACE_TOKEN)
    print("---" + "独立测试完成" + "---")
