经过评估发现`AudioCraft`并不能实现在复杂音频(人声+背景音乐+环境音+噪音)中进行人声和非人声的分轨提取.

经过手动测试发现`URV5(Ultimate Vocal Remover v5)`通过`MDX-Net`方式, 加载模型`UVR-MDX-NET Inst HQ5`可以完美实现此功能.  

但是URV5是GUI桌面工具, 并不适合整合到本项目实现代码调用.

通过搜索发现了[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator#)

这个项目支持命令行整合URV5模型来实现对应的功能. 所以需要进行评估:

1. `python-audio-separator`代码我已经clone到了目录 @python-audio-separator 下, 请阅读此项目, 确定是否真的可以实现URV5相应的功能
2. 如果可行,请将想 @思路.md 文档中的音轨分离的部分更新成 `python-audio-separator` + `URV5` 相应的内容

---
---
---
---

执行日志:

```text
2025-08-11 17:08:07,502 - INFO - separator - Operating System: Windows 10.0.19044
2025-08-11 17:08:07,502 - INFO - separator - System: Windows Node: DESKTOP-AG8II45 Release: 10 Machine: AMD64 Proc: Intel64 Family 6 Model 63 Stepping 2, GenuineIntel
2025-08-11 17:08:07,502 - INFO - separator - Python Version: 3.11.13
2025-08-11 17:08:07,502 - INFO - separator - PyTorch Version: 2.8.0+cpu
2025-08-11 17:08:07,558 - INFO - separator - FFmpeg installed: ffmpeg version 2025-08-07-git-fa458c7243-full_build-www.gyan.dev Copyright (c) 2000-2025 the FFmpeg developers
2025-08-11 17:08:07,561 - INFO - separator - ONNX Runtime GPU package installed with version: 1.22.0
2025-08-11 17:08:07,562 - INFO - separator - No hardware acceleration could be configured, running in CPU mode
正在加载模型: UVR-MDX-NET-Inst_HQ_5.onnx
2025-08-11 17:08:07,563 - INFO - separator - Loading model UVR-MDX-NET-Inst_HQ_5.onnx...
2025-08-11 17:08:07,655 - INFO - separator - Hash of model file /tmp/audio-separator-models/UVR-MDX-NET-Inst_HQ_5.onnx 
is cb790d0c913647ced70fc6b38f5bea1a
2025-08-11 17:08:11,082 - INFO - separator - Load model duration: 00:00:03
正在处理文件: D:\Python\Project\VideoTran\videos\333.mkv
```

日志中显示运行的还是cpu模式. 请排查

## 解决方案

`````markdown
很好！从您提供的终端输出可以看到：

## **已知事实**

1. ✅ **NVIDIA 驱动已安装**：Driver Version: 560.94
2. ✅ **GPU 硬件可用**：NVIDIA GeForce GTX 1060 6GB
3. ✅ **CUDA 运行时版本**：CUDA Version: 12.6
4. ❌ **CUDA Toolkit 未安装**：`nvcc` 命令不可用

## **分析结果**

**[分析]** 您的情况很典型：
- 有 NVIDIA GPU 和驱动程序
- `nvidia-smi` 显示的 "CUDA Version: 12.6" 是驱动程序支持的最高 CUDA 版本
- 但没有安装完整的 CUDA Toolkit（所以 `nvcc` 不可用）

**[建议]** 对于您的音频分离项目，**不需要**安装完整的 CUDA Toolkit！只需要安装支持 CUDA 的 PyTorch 版本即可。

## **解决方案**

请按以下步骤操作：

### 1. 检查当前 PyTorch 版本
```bash
conda activate videotran
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 卸载当前的 CPU 版本 PyTorch
```bash
pip uninstall torch torchvision torchaudio
```

### 3. 安装 CUDA 版本的 PyTorch
由于您的驱动支持 CUDA 12.6，建议安装 CUDA 12.1 版本的 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 验证安装
```bash
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch CUDA版本: {torch.version.cuda}')
"
```

### 5. 重新运行您的程序
完成后，重新运行：
```bash
python main.py
```

您应该会看到类似这样的日志：
```
CUDA is available in Torch, setting Torch device to CUDA
ONNXruntime has CUDAExecutionProvider available, enabling acceleration
```

## **为什么不需要安装 CUDA Toolkit？**

**[分析]** PyTorch 的 CUDA 版本已经包含了运行所需的 CUDA 库，它可以直接使用您现有的 NVIDIA 驱动程序。只有在需要自己编译 CUDA 代码时才需要完整的 CUDA Toolkit。

请先执行步骤1检查当前 PyTorch 版本，然后按照上述步骤操作。如果遇到任何问题，请告诉我具体的错误信息。
`````

---
---
---
---
