import torch
import warnings

# DEVICE = 'mps'
DEVICE = 'cuda'

def get_gpu_memory_usage():
    if DEVICE == 'mps':
        warnings.warn("【MAC用户警告】对于很多broadcast op操作，Pytorch当前还并不支持，你也许可以尝试开启PYTORCH_ENABLE_MPS_FALLBACK=1，这样会使用CPU来代替MPS作为backend，暂不清楚会不会和cuda有区别，请确保最终代码在CUDA上也能正常运行")
        allocated = torch.mps.current_allocated_memory()
        reserved = torch.mps.driver_allocated_memory()
    elif DEVICE == 'cuda':
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    else:
        raise ValueError(f"Invalid device: {DEVICE}")
    return allocated / (1024 ** 2), reserved / (1024 ** 2)  # 转换为 MB

def output_gpu_memory_usage(logger_str = ""):
    allocated_mb, reserved_mb = get_gpu_memory_usage()
    print(f"{logger_str} 已分配的显存: {allocated_mb:.2f} MB, 已预留的显存: {reserved_mb:.2f} MB")
