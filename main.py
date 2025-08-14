"""
主程序入口
"""
import os
import time


def process_video(video_path, hf_token=None):
    from audio_extractor import extract_audio
    from vocal_separator import separate_vocals
    from wpx import recognize_speakers_and_generate_text
    """
    处理指定视频文件的完整流程。

    1. 提取音频
    2. 分离人声与背景音
    3. 对人声进行语音识别和说话人分析

    :param video_path: 视频文件的完整路径。
    :param hf_token: Hugging Face Hub 的认证令牌，用于说话人分离。
    """
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在于路径: {video_path}")
        return

    # --- 1. 定义路径 ---
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # 音频提取路径
    extracted_audio_path = os.path.join(video_dir, f"{video_basename}.mp3")
    
    
    print(f"--- 开始处理视频: {video_path} ---")
    total_start_time = time.time()

    # --- 2. 提取音频 ---
    print("\n--- 步骤 1/3: 提取音频 ---")
    step_start_time = time.time()
    extract_audio(video_path, extracted_audio_path)
    step_end_time = time.time()
    print(f"步骤 1 完成，耗时: {step_end_time - step_start_time:.2f} 秒")

    # --- 3. 分离人声和背景音 ---
    print("\n--- 步骤 2/3: 分离人声和背景音 ---")
    step_start_time = time.time()
    # separate_vocals 返回一个包含所有输出文件路径的列表
    separate_vocals(extracted_audio_path)
    step_end_time = time.time()
    print(f"步骤 2 完成，耗时: {step_end_time - step_start_time:.2f} 秒")


    # 查找包含 "Vocals" 的文件路径
    vocal_track_path = r'./videos/Vocals.wav'

    if not os.path.exists(vocal_track_path):
        print("错误: 未能从分离结果中找到人声文件。")
        return
        
    print(f"找到人声文件: {vocal_track_path}")

    # --- 4. 提取说话人文本 ---
    print("\n--- 步骤 3/3: 识别说话人并生成文本 ---")
    step_start_time = time.time()
    recognize_speakers_and_generate_text(vocal_track_path, hf_token=hf_token)
    step_end_time = time.time()
    print(f"步骤 3 完成，耗时: {step_end_time - step_start_time:.2f} 秒")

    total_end_time = time.time()
    print(f"\n--- 视频处理完成: {video_path} ---")
    print(f"总耗时: {total_end_time - total_start_time:.2f} 秒")



if __name__ == "__main__":
    # ###################################################################
    # # 提示：说话人分离功能需要 Hugging Face Token
    # # HUGGING_FACE_TOKEN = "hf_YOUR_TOKEN_HERE"
    # ###################################################################
    huggingface_token = os.environ.get("HUGGING_FACE_TOKEN")
    print('huggingface_token:', huggingface_token)
    HUGGING_FACE_TOKEN = huggingface_token 

    # 指定要处理的视频文件
    video_to_process = r"D:\Python\Project\VideoTran\videos\7.mp4"
    
    process_video(video_to_process, hf_token=HUGGING_FACE_TOKEN)

    
