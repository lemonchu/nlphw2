import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from utils import output_gpu_memory_usage
import torch.nn as nn
import time
def setup(rank, world_size):
    # 使用GLOO后端初始化torch.distributed
    # 这里我们使用GLOO后端，因为他可以兼容CPU和GPU，你可以在CPU上调试，再在GPU上运行
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def all_reduce_example(rank, world_size):
    tensor = torch.arange(5, device='cpu', dtype=torch.float32) * (rank + 1)  # 张量内容取决于rank
    print(f"[Rank {rank}] before all_reduce: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}] after all_reduce: {tensor}")
    
def all_gather_example(rank, world_size):
    tensor = torch.arange(5, device='cpu', dtype=torch.float32) * (rank + 1)
    print(f"[Rank {rank}] before all_gather: {tensor}")
    output = [torch.zeros_like(tensor, dtype=torch.float32) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    print(f"[Rank {rank}] after all_gather: {output}")

def reduce_scatter_example(rank, world_size):

    tensor = torch.arange(6, device='cpu', dtype=torch.float32) * (rank + 1)
    output_tensor = torch.zeros(3, device='cpu', dtype=torch.float32)
    print(f"[Rank {rank}] before reduce_scatter: {tensor}")
    
    dist.reduce_scatter_tensor(output_tensor, tensor)
    print(f"[Rank {rank}] after reduce_scatter: {output_tensor}")

def detect_gpu_memory(rank):
    output_gpu_memory_usage(f"[Rank {rank}] before create model")
    linear = nn.Linear(1000, 1000)
    output_gpu_memory_usage(f"[Rank {rank}] after create model")
    linear = linear.to(device='cuda')
    output_gpu_memory_usage(f"[Rank {rank}] after move model to gpu")
    # 删除模型
    del linear
    output_gpu_memory_usage(f"[Rank {rank}] delete model")
    torch.cuda.empty_cache()
    output_gpu_memory_usage(f"[Rank {rank}] after empty cache")
def main(rank, world_size):
    setup(rank, world_size)
    if rank == 0:
        print("###ALL REDUCE EXAMPLE###\n")
    all_reduce_example(rank, world_size)
    time.sleep(1)
    
    if rank == 0:
        print("###ALL GATHER EXAMPLE###\n")
    all_gather_example(rank, world_size)
    time.sleep(1)

    if rank == 0:
        print("###REDUCE SCATTER EXAMPLE###\n")
    reduce_scatter_example(rank, world_size)
    time.sleep(1)

    if rank == 0:
        print("###DETECT GPU MEMORY###\n")
    detect_gpu_memory(rank)
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # 模拟两个"GPU"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(main, args=(world_size,), nprocs=world_size)
