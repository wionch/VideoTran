# 导入必要的库
import torch
import torchaudio
from pyannote.audio import Pipeline

# --- 准备工作 ---
# 1. 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 加载“一体化”的说话人分离管道 (把它想象成一个包含所有工具的工具箱)
# 这个管道对象本身，就包含了VAD、声纹提取、聚类等所有必要的模型组件。
# 所以我们只需要加载这一个 "pyannote/speaker-diarization-3.1"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HUGGINGFACE_TOKEN" # 替换为你的Hugging Face Token
)
pipeline.to(device)

# 定义你的音频文件路径
long_audio_path = "videos\Vocals.wav"
sample_a_path = "path/to/your/sample_A_5s.wav"
sample_b_path = "path/to/your/sample_B_5s.wav"


# --- 核心步骤 ---

# 3. 从“工具箱”中取出我们需要的“声纹提取器” (Speaker Embedding Model)
# 我们不需要再单独加载一个嵌入模型，可以直接从主管道对象中访问它。
# `pipeline.embedding_model` 就是我们所说的 voiceprint extractor。
embedding_model = pipeline.embedding_model

# 加载并处理音频样本
waveform_a, sample_rate_a = torchaudio.load(sample_a_path)
waveform_b, sample_rate_b = torchaudio.load(sample_b_path)
# (实际应用中请确保采样率与模型要求一致，通常为16kHz)

# 4. 使用取出的声纹提取器为A和B生成声纹
with torch.no_grad():
    # 为A生成声纹
    embedding_a = embedding_model(waveform_a.to(device).unsqueeze(0))
    embedding_a = torch.nn.functional.normalize(embedding_a, p=2, dim=1)
    
    # 为B生成声纹
    embedding_b = embedding_model(waveform_b.to(device).unsqueeze(0))
    embedding_b = torch.nn.functional.normalize(embedding_b, p=2, dim=1)

# 5. 将生成的声纹和对应标签组合起来
known_speaker_embeddings = torch.cat([embedding_a, embedding_b], dim=0)
known_speaker_labels = ["A", "B"]

# 6. 运行完整的管道，并将包含已知声纹信息的“特殊指令”传入
# 管道接收到这个 `speaker_embeddings` 参数后，
# 就会在内部使用这些声纹来识别和标记说话人。
diarization = pipeline(
    long_audio_path,
    speaker_embeddings=known_speaker_embeddings,
    speaker_labels=known_speaker_labels
)

# --- 查看结果 ---
# 输出的说话人标签将直接是 'A' 和 'B'
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"在 {turn.start:.2f}s 到 {turn.end:.2f}s, 说话人 {speaker} 在说话。")