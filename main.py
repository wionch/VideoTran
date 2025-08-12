import os
import sys
os.environ['TRANSFORMERS_SKIP_TORCH_LOAD_SAFETY_CHECK'] = '1'

# 解决Conda环境下DLL加载失败的问题
# 获取当前Python解释器的路径，并构建Conda环境的关键路径
if sys.platform == 'win32':
    conda_env_path = sys.executable
    conda_env_root = os.path.dirname(os.path.dirname(conda_env_path))
    # 将关键的DLL路径添加到PATH环境变量中
    dll_paths = [
        os.path.join(conda_env_root, 'Library', 'bin'),
        os.path.join(conda_env_root, 'bin'),
        os.path.join(conda_env_root, 'Scripts')
    ]
    for path in dll_paths:
        if os.path.isdir(path):
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']

# 导入拆分后的模块
import torch  # 用于检测 GPU 可用性
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# from audio_extractor import extract_audio
# from vocal_separator import separate_vocals
from wpx import recognize_speakers_and_generate_text

if __name__ == "__main__":
    # ----------------------------------------------------------------
    # 请在下方引号内替换为您的 Hugging Face Access Token
    # 获取步骤:
    # 1. 访问 https://huggingface.co/pyannote/speaker-diarization-3.1 并同意协议
    # 2. 对 https://huggingface.co/pyannote/segmentation-3.0 也进行同样的操作
    # 3. 访问 https://huggingface.co/settings/tokens 创建一个 read token 并复制到此处
    # ----------------------------------------------------------------
    HUGGING_FACE_TOKEN = ""

    # 定义文件路径
    video_file = r"D:\Python\Project\VideoTran\videos\333.mkv"
    audio_file = r"D:\Python\Project\VideoTran\videos\audio.wav"
    vocal_audio_file = r"D:\Python\Project\VideoTran\videos\Volcal.wav"

    # --- 工作流程 ---
    # 添加文件存在性检查
    if not os.path.exists(vocal_audio_file):
        print(f"错误: 人声音频文件不存在: {vocal_audio_file}")
        print("请确保已经完成音频提取和人声分离步骤")
        sys.exit(1)

    # 步骤 0: 提取音频
    # print("--- 步骤 0: 提取音频 ---")
    # extract_audio(video_file, audio_file)
    # print("--- 步骤 0 完成 ---")

    # 步骤 1: 分离人声 (如果需要，取消下面的注释)
    # print("--- 步骤 1: 分离人声 ---")
    # separate_vocals(video_file)
    # print("--- 步骤 1 完成 ---")

    # # 步骤 2: 识别说话人并生成文本
    print("--- 步骤 2: 识别说话人并生成文本 ---")
    recognize_speakers_and_generate_text(vocal_audio_file, hf_token=HUGGING_FACE_TOKEN)
    print("--- 步骤 2 完成 ---")