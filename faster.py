
import os
import torch
from faster_whisper import WhisperModel

def format_timestamp(seconds):
    """将秒数转换为 SRT 时间戳格式 (HH:MM:SS,ms)"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe_with_faster_whisper(audio_path):
    """
    使用 faster-whisper 执行音频转录并生成 SRT 文件。

    :param audio_path: 输入音频文件的路径。
    """
    # --- 1. 参数设置 ---
    use_cuda_env = os.getenv('USE_CUDA', '1')
    use_cuda = (use_cuda_env == '1') and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    compute_type = 'float16' if use_cuda else 'int8'

    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在于路径: {audio_path}")
        return

    output_dir = os.path.join(os.path.dirname(os.path.dirname(audio_path)), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    try:
        # --- 2. 流程开始 ---
        print(f"--- 使用设备: {device}, 计算类型: {compute_type} ---")
        if use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到 GPU: {gpu_name}")

        # --- 3. 加载 Whisper 模型 ---
        print("步骤 1/3: 加载 Whisper 模型 (faster-whisper)...")
        model = WhisperModel("large-v2", device=device, compute_type=compute_type)

        # --- 4. 执行转录 ---
        print(f"步骤 2/3: 开始音频转录: {audio_path}...")
        segments, info = model.transcribe(audio_path, language="zh", vad_filter=True)
        
        print(f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})")

        # --- 5. 生成 SRT 内容 ---
        print("步骤 3/3: 生成 SRT 字幕文件...")
        srt_content = []
        for i, segment in enumerate(segments):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()
            
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("") # 添加空行
            # print(text)

        # --- 6. 结果保存 ---
        output_basename = os.path.splitext(os.path.basename(audio_path))[0]
        srt_path = os.path.join(output_dir, f"{output_basename}_faster.srt")
        print("srt_content:", srt_content)
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_content))
            
        print(f"--- 处理全部完成！结果已保存到: {srt_path} ---")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # --- 用于独立测试的配置 ---
    test_audio_file = r"D:\Python\Project\VideoTran\videos\Volcal.wav"
    
    print("--- 开始独立测试 faster.py ---")
    transcribe_with_faster_whisper(test_audio_file)
    print("--- 独立测试完成 ---")
