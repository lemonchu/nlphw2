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
    
    group_1 = dist.new_group(ranks=[0,1])
    group_2 = dist.new_group(ranks=[2,3])
    group_3 = dist.new_group(ranks=[0,2])
    group_4 = dist.new_group(ranks=[1,3])
    groups = [group_1, group_2, group_3, group_4]
    return groups

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

def send_point_to_point(rank, world_size,groups):
    if rank == 0:
        tensor = torch.arange(5, device='cpu', dtype=torch.float32) * (rank + 1)
        dist.send(tensor, dst=rank + 1, group=dist.group.WORLD)
        dist.send(tensor, dst=2, group=groups[2])
        print(f"[Rank {rank}] after send: {tensor}")
    elif rank == 1:
        tensor = torch.zeros(5, device='cpu', dtype=torch.float32)
        print(tensor)
        dist.recv(tensor, src=rank - 1, group=dist.group.WORLD)
        print(f"[Rank {rank}] after recv: {tensor}")
    elif rank == 2:
        tensor = torch.zeros(5, device='cpu', dtype=torch.float32)
        dist.recv(tensor, src=0, group=groups[2])
        print(f"[Rank {rank}] after recv: {tensor}")
        dist.send(tensor, dst=3, group=groups[1])
    elif rank == 3:
        tensor = torch.zeros(5, device='cpu', dtype=torch.float32)
        dist.recv(tensor, src=2, group=groups[1])
        print(f"[Rank {rank}] after recv: {tensor}")

def loss_backward_through_pp_fail_exp(rank, world_size,groups):
    if rank == 3:
        return 
    elif rank == 2:
        time.sleep(1)
        print("###rank 2 baseline   ###")
        x = torch.randn(10, device='cpu', requires_grad=True)
        y = torch.randn(10, device='cpu', requires_grad=True)
        z = x + y
        z = z.sum()
        z.backward()
        print(f"[Rank {rank}] x.grad",x.grad)
        print(f"[Rank {rank}] y.grad",y.grad)
    else:
        if rank == 0:
            x = torch.randn(10, device='cpu', requires_grad=True)
            dist.send(x, dst=1, group=groups[0])
            print(f"[Rank {rank}] after send: {x}")
            print(f"[Rank {rank}] x.grad",x.grad)
        else:
            x = torch.zeros(10, device='cpu', requires_grad=True)
            dist.recv(x, src=0, group=groups[0])
            y = torch.randn(10, device='cpu', requires_grad=True)
            z = x + y
            z = z.sum()
            z.backward()
            print(f"[Rank {rank}] x.grad",x.grad)
            print(f"[Rank {rank}] y.grad",y.grad)
        
def loss_backward_through_pp_success_exp(rank, world_size,groups):
    if rank == 3:
        return 
    elif rank == 2:
        time.sleep(1)
        print("###rank 2 baseline   ###")
        x = torch.randn(10, device='cpu', requires_grad=True)
        y = torch.randn(10, device='cpu', requires_grad=True)
        z = x + y
        z = z.sum()
        z.backward()
        print(f"[Rank {rank}] x.grad",x.grad)
        print(f"[Rank {rank}] y.grad",y.grad)
    else:
        if rank == 0:
            
            
            a = torch.randn(10, device='cpu', requires_grad=True)

            x = a * 2

            dist.send(x, dst=1, group=groups[0])
            print(f"[Rank {rank}] after send: {x}")
            print(f"[Rank {rank}] x.grad",x.grad)
            grad = torch.randn(10, device='cpu', requires_grad=True)
            dist.recv(grad, src=1, group=groups[0])
            x.grad = grad
            torch.autograd.backward(x,x.grad)
            print(f"[Rank {rank}] after recv: {x}")
            print(f"[Rank {rank}] x.grad",x.grad)
            print(f"[Rank {rank}] a.grad",a.grad)
        else:
            x = torch.zeros(10, device='cpu', requires_grad=True)
            dist.recv(x, src=0, group=groups[0])
            print(f"[Rank {rank}] after recv: {x}")
            y = torch.randn(10, device='cpu', requires_grad=True)
            z = x + y
            z = z.sum()
            z.backward()
            print(f"[Rank {rank}] x.grad",x.grad)
            print(f"[Rank {rank}] y.grad",y.grad)
            dist.send(x.grad, dst=0, group=groups[0])
            print(f"[Rank {rank}] after send: {x}")
        

def forward_and_backward_through_tp(rank, world_size, groups):
    if rank == 2 or rank == 3:
        return 
    else:   
        x = torch.randn(5, device='cpu', requires_grad=True)
        linear = nn.Linear(5, 10) 
        linear.weight.data = torch.ones(5,10)
        y = linear(x)
        torch.all_reduce(y, op=dist.ReduceOp.SUM, group=groups[2])
        print(f"[Rank {rank}] y",y)
        y.sum().backward()
        print(f"[Rank {rank}] x.grad",x.grad)
        print(f"[Rank {rank}] linear.weight.grad",linear.weight.grad)

def consecutive_pipeline_experiment(rank, world_size, groups):
    if rank == 2 or rank == 3:
        return 
    else:
        print(f"[Rank {rank}] start experiment")
        if rank == 0:
            x = torch.randn(5, device='cpu', requires_grad=True,dtype=torch.float32)
            x1 = torch.randn(5, device='cpu', requires_grad=True,dtype=torch.float32) * 2
            print(f"[Rank {rank}] x",x)
            time.sleep(1)
            dist.send(x, dst=1, group=groups[0])
            dist.send(x1, dst=1, group=groups[0])
            print(f"[Rank {rank}] x",x)
            print(f"[Rank {rank}] x1",x1)
        else:   
            x = torch.randn(5, device='cpu', requires_grad=True,dtype=torch.float32)
            x1 = torch.randn(5, device='cpu', requires_grad=True,dtype=torch.float32) * 2
            print(f"[Rank {rank}] x",x)
            dist.recv(x, src=0, group=groups[0])
            dist.recv(x1, src=0, group=groups[0])
            print(f"[Rank {rank}] x",x)
            print(f"[Rank {rank}] x1",x1)

def main(rank, world_size):
    groups = setup(rank, world_size)

    
    # if rank == 0:
    #     print("###ALL REDUCE EXAMPLE###\n")
    # all_reduce_example(rank, world_size)
    # time.sleep(1)
    
    # if rank == 0:
    #     print("###ALL GATHER EXAMPLE###\n")
    # all_gather_example(rank, world_size)
    # time.sleep(1)

    # if rank == 0:
    #     print("###REDUCE SCATTER EXAMPLE###\n")
    # reduce_scatter_example(rank, world_size)
    # time.sleep(1)

    # if rank == 0:
    #     print("###DETECT GPU MEMORY###\n")
    # detect_gpu_memory(rank)

    # if rank == 0:
    #     print("###try sending point to point###\n")
    # send_point_to_point(rank, world_size, groups)
    # time.sleep(1)

    # if rank == 0:
    #     print("###try loss backward through pp fail experiment###\n")
    # loss_backward_through_pp_fail_exp(rank, world_size, groups)
    # time.sleep(1)

    # if rank == 0:
    #     print("###try loss backward through pp success experiment###\n")
    # loss_backward_through_pp_success_exp(rank, world_size, groups)
    # time.sleep(1)

    # if rank == 0:
    #     print("###try forward and backward through tp###\n")
    # forward_and_backward_through_tp(rank, world_size, groups)
    # time.sleep(1)

    if rank == 0:
        print("###try consecutive pipeline experiment###\n")
    consecutive_pipeline_experiment(rank, world_size, groups)
    time.sleep(1)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4  # 模拟两个"GPU"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(main, args=(world_size,), nprocs=world_size)
