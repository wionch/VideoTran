# D:\Python\Project\VideoTran\video_tran\utils\similarity_provider.py
import torch
import piq
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseSimilarityProvider(ABC):
    """定义了相似度提供者的统一接口。"""
    @abstractmethod
    def compare_batch(self, frame_tensors: torch.Tensor) -> List[float]:
        """
        比较一个批次中的相邻帧，返回一个“变化分数”的列表。
        分数越高，表示相邻帧之间的变化越大。
        列表的第一个元素总是0，因为第一帧没有可比较的对象。
        """
        pass

class SSIMSimilarityProvider(BaseSimilarityProvider):
    """使用结构相似性(SSIM)进行比较的提供者。更符合人类视觉感知。"""
    def compare_batch(self, frame_tensors: torch.Tensor) -> List[float]:
        if not frame_tensors.is_cuda:
            raise ValueError("输入的张量必须在GPU上。")
        if frame_tensors.shape[0] < 2:
            return [0.0] * frame_tensors.shape[0]
        
        frames_a = frame_tensors[:-1]
        frames_b = frame_tensors[1:]
        
        # SSIM值域为[0, 1]，1表示完全相同。我们用 1.0 - ssim 作为变化分数。
        ssim_scores = piq.ssim(frames_a, frames_b, data_range=1.0, reduction='none')
        change_scores = 1.0 - ssim_scores
        
        return [0.0] + change_scores.cpu().tolist()

class MSESimilarityProvider(BaseSimilarityProvider):
    """使用均方误差(MSE)进行比较的提供者。速度快但对感知不敏感。"""
    def compare_batch(self, frame_tensors: torch.Tensor) -> List[float]:
        if not frame_tensors.is_cuda:
            raise ValueError("输入的张量必须在GPU上。")
        if frame_tensors.shape[0] < 2:
            return [0.0] * frame_tensors.shape[0]
        
        frames_a = frame_tensors[:-1]
        frames_b = frame_tensors[1:]
        
        mse_losses = torch.nn.functional.mse_loss(frames_a, frames_b, reduction='none')
        mse_per_pair = mse_losses.mean(dim=[1, 2, 3])
        
        return [0.0] + mse_per_pair.cpu().tolist()