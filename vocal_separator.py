"""
人声分离模块
使用 python-audio-separator 进行人声与背景音分离
"""
import os
import logging
from audio_separator.separator import Separator


def separate_vocals(video_path):
    """
    通过`python-audio-separator`将视频文件进行人声和非人声的分轨提取。

    :param video_path: 输入视频文件的路径。
    """
    try:
        output_dir = os.path.dirname(video_path)
        separator = Separator(output_dir=output_dir, log_level=logging.INFO)
        model_name = 'UVR-MDX-NET-Inst_HQ_5.onnx'
        print(f"正在加载人声分离模型: {model_name}")
        separator.load_model(model_filename=model_name)
        print(f"正在分离人声和背景音: {video_path}")
        output_files = separator.separate(video_path)
        print(f"分离完成！输出文件: {', '.join(output_files)}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    # 独立测试代码示例
    video_file = r"D:\Python\Project\VideoTran\videos\333.mkv"
    
    separate_vocals(video_file)
