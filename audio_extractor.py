"""
音频提取模块
用于从视频文件中提取音频
"""
import moviepy.video.io.VideoFileClip as VideoFileClip


def extract_audio(video_path, audio_path):
    """
    从视频文件中提取音频并保存为MP3文件。

    :param video_path: 输入视频文件的路径。
    :param audio_path: 输出MP3文件的路径。
    """
    try:
        print(f"正在从 {video_path} 提取音频...")
        video = VideoFileClip.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        print(f"成功提取音频到: {audio_path}")
    except Exception as e:
        print(f"提取音频时发生错误: {e}")


if __name__ == "__main__":
    # 独立测试代码示例
    video_file = r"D:\Python\Project\VideoTran\videos\6.mkv"
    output_audio = r"D:\Python\Project\VideoTran\videos\6.mp3"
    
    extract_audio(video_file, output_audio)
