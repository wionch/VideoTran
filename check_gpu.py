# import torch

# if torch.cuda.is_available():
#     print("CUDA is available. GPU will be used.")
#     print(f"Device count: {torch.cuda.device_count()}")
#     print(f"Current device: {torch.cuda.current_device()}")
#     print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# else:
#     print("CUDA is not available. The script will run on CPU.")


# check_gpu.py
import paddle
import sys

print("--- PaddlePaddle GPU Environment Check ---")
try:
    print(f"Python Version: {sys.version}")
    print(f"PaddlePaddle Version: {paddle.__version__}")
    
    # 检查是否安装了GPU版本的PaddlePaddle
    is_gpu_version = paddle.is_compiled_with_cuda()
    print(f"Is GPU Version: {is_gpu_version}")

    if is_gpu_version:
        print(f"CUDA Version (Compiled with): {paddle.version.cuda()}")
        print(f"cuDNN Version (Compiled with): {paddle.version.cudnn()}")
        
        # 尝试获取GPU设备信息
        try:
            gpu_count = paddle.device.cuda.device_count()
            print(f"Number of GPUs Detected: {gpu_count}")
            if gpu_count > 0:
                # 检查当前设备是否为GPU
                current_device = paddle.get_device()
                print(f"Current Device Used by Paddle: {current_device}")
                if 'gpu' in current_device:
                    print("\n✅ GPU environment seems OK for PaddlePaddle.")
                else:
                    print("\n❌ WARNING: PaddlePaddle is using CPU. There might be a driver or CUDA version mismatch.")
            else:
                print("\n❌ ERROR: No GPU device found by PaddlePaddle, even though it's a GPU version.")

        except Exception as e:
            print(f"\n❌ ERROR during GPU device check: {e}")
            print("   This often means there's a problem with the NVIDIA driver or CUDA installation.")

    else:
        print("\n❌ ERROR: You have installed the CPU-only version of PaddlePaddle.")
        print("   Please uninstall 'paddlepaddle' and install 'paddlepaddle-gpu'.")

except Exception as e:
    print(f"An error occurred: {e}")

print("------------------------------------")

