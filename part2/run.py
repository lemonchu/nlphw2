import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import numpy as np

from utils import output_gpu_memory_usage
import argparse

from optimizer import SimpleOptimizer, ZeroOptimizer
from model_simple import CustomTransformer
from dataset import FixedRandomDataset, SimpleDataLoader, DataParallelDataLoader

from partition import partition_model
from pipeline import NoPipe, SimplePipe, GPipe

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def init_3d_parallel_group(rank, world_size, tp_size, pp_size, dp_size):
    # Ensure world size matches the 3D partitioning
    assert world_size == tp_size * pp_size * dp_size, "world_size must equal tp_size * pp_size * dp_size"
    
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)

    rank = dist.get_rank()
    
    # Calculate ranks for tensor, pipeline, and data parallel groups
    tp_rank = rank % tp_size
    pp_rank = (rank // tp_size) % pp_size
    dp_rank = rank // (tp_size * pp_size)

    print(f'[Rank {rank}] init tp rank {tp_rank}/{tp_size}')
    print(f'[Rank {rank}] init pp rank {pp_rank}/{pp_size}')
    print(f'[Rank {rank}] init dp rank {dp_rank}/{dp_size}')

    # Initialize tensor parallel groups
    tp_groups = []
    for i in range(dp_size):
        for j in range(pp_size):
            tp_group_ranks = [i * pp_size * tp_size + j * tp_size + k for k in range(tp_size)]
            tp_group = dist.new_group(ranks=tp_group_ranks)
            tp_groups.append(tp_group)
            if rank in tp_group_ranks:
                my_tp_group = tp_group
            print(f'[Rank {rank}] initialized tp_group with ranks: {tp_group_ranks}')

    # Initialize pipeline parallel groups
    pp_groups = []
    for i in range(dp_size):
        for k in range(tp_size):
            pp_group_ranks = [i * pp_size * tp_size + j * tp_size + k for j in range(pp_size)]
            pp_group = dist.new_group(ranks=pp_group_ranks)
            pp_groups.append(pp_group)
            if rank in pp_group_ranks:
                my_pp_group = pp_group
            print(f'[Rank {rank}] initialized pp_group with ranks: {pp_group_ranks}')

    # Initialize data parallel groups
    dp_groups = []
    for j in range(pp_size):
        for k in range(tp_size):
            dp_group_ranks = [i * pp_size * tp_size + j * tp_size + k for i in range(dp_size)]
            dp_group = dist.new_group(ranks=dp_group_ranks)
            dp_groups.append(dp_group)
            if rank in dp_group_ranks:
                my_dp_group = dp_group
            print(f'[Rank {rank}] initialized dp_group with ranks: {dp_group_ranks}')

    print('all groups initialized')
    return my_tp_group, my_pp_group, my_dp_group



def main(rank, world_size, args):
    # 初始化进程组
    print(f'[rank {rank}] init process group')
    
    tp_group, pp_group, dp_group = init_3d_parallel_group(rank, world_size, args.tp_size, args.pp_size, args.dp_size)

    is_last_pp = pp_group.rank() == pp_group.size() - 1
    is_last_dp = dp_group.rank() == dp_group.size() - 1
    is_last_tp = tp_group.rank() == tp_group.size() - 1

    # 创建数据集和数据加载器，确保数据在进程间分割
    dataset = FixedRandomDataset(num_samples=80, seq_length=128, vocab_size=4000, seed=42)
   
    if dp_group.size() == 1:
        data_loader = SimpleDataLoader(dataset, batch_size=args.micro_batch_size) 
    else:
        data_loader = DataParallelDataLoader(dataset, batch_size=args.micro_batch_size, rank=dp_group.rank() if dp_group is not None else 0, dp_group = dp_group)
    # print(len(data_loader))
    # 初始化模型
    model = CustomTransformer(
        embed_size=256, 
        num_layers=16, 
        num_heads=8, 
        ff_size=1024, 
        vocab_size=4000,
        tp_group=tp_group
    ).to(device)
    torch.save(model.state_dict(), f"ckpt/model.pth")
    dict_state = torch.load(f"ckpt/model.pth",weights_only=True)
    model.load_state_dict(dict_state)
    for param in model.parameters():
        
        last_dp_rank = (dp_group.size() - 1) * pp_group.size() * tp_group.size() + pp_group.rank() * tp_group.size() + tp_group.rank()
        last_pp_rank = dp_group.rank() * pp_group.size() * tp_group.size() + (pp_group.size() - 1) * tp_group.size() + tp_group.rank()

        dist.broadcast(param.data, src=last_dp_rank, group=dp_group)
        
        dist.broadcast(param.data, src=last_pp_rank, group=pp_group)
    loss_criterion = nn.CrossEntropyLoss()
    if pp_group.size() > 1:
        model = partition_model(model.core, pp_group.rank(), pp_group.size())
        if args.pipeline == 'simplepipe':
            pipe = SimplePipe(model, loss_criterion, rank=rank, pp_group=pp_group, tp_size=args.tp_size)
        else:
            pipe = GPipe(model, loss_criterion, rank=rank, pp_group=pp_group, tp_size=args.tp_size)
    else:
        pipe = NoPipe(model, loss_criterion)
    output_gpu_memory_usage(f"[rank {rank}] after init model")
    # 确保所有进程的模型参数相同
    
    gradient_accumulation_steps = args.global_batch_size // args.micro_batch_size // args.dp_size
    if dp_group.size() == 1:
        optimizer = SimpleOptimizer(model, lr=1e-3, weight_decay=0.0, gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        optimizer = ZeroOptimizer(model, dp_group=dp_group, lr=0.001, weight_decay=0.0, stage=args.stage, gradient_accumulation_steps=gradient_accumulation_steps)
    
    output_gpu_memory_usage(f"[rank {rank}] after init optimizer")
    
    loss_log = []
    
    total_steps = 0
    for epoch in range(args.epoch):
        data_loader.set_epoch(epoch)  # 确保每个 epoch 的数据顺序一致
        # 训练循环
        loss_accum = 0
        samples_accum = []
        for i, sample in enumerate(data_loader):
            sample = {k: v.to(device) for k, v in sample.items()}
            if pp_group.size() == 1:
                loss = pipe.forward_backward(**sample)
                optimizer.step()
                total_steps += 1
                loss_temp = loss.detach().cpu()
                dist.all_reduce(loss_temp, group=dp_group, op=dist.ReduceOp.SUM)
                loss_temp = loss_temp / args.dp_size
                loss_accum += loss_temp
                # print(loss_accum)
                if total_steps % gradient_accumulation_steps == 0:
                    loss_to_log = loss_accum / gradient_accumulation_steps
                    if is_last_dp and is_last_pp and is_last_tp:
                        print(f"Rank {rank} - Epoch {epoch} - Step {total_steps} - Loss:", loss_to_log)
                    loss_log.append(loss_to_log.item())
                    
                    loss_accum = 0
            else:
                ### TODO: Implement the loss calculation for pipeline parallel
                ### Hint: Only last pp rank will calculate the loss
                if args.pipeline == 'simplepipe':
                    loss = pipe.forward_backward(**sample)
                    optimizer.step()
                    total_steps += 1
                    if is_last_pp and is_last_tp:
                        loss_temp = loss.detach().cpu()
                        dist.all_reduce(loss_temp, group=dp_group, op=dist.ReduceOp.SUM)
                        loss_temp = loss_temp / args.dp_size
                        loss_accum += loss_temp
                        if total_steps % gradient_accumulation_steps == 0:
                            loss_to_log = loss_accum / gradient_accumulation_steps
                            if is_last_dp and is_last_pp and is_last_tp:
                                print(f"Rank {rank} - Epoch {epoch} - Step {total_steps} - Loss:", loss_to_log)
                            loss_log.append(loss_to_log.item())
                            loss_accum = 0  
                else:
                    samples_accum.append(sample)
                    if len(samples_accum) == gradient_accumulation_steps:
                        losses = pipe.forward_backward(samples_accum)
                        for loss in losses:
                            optimizer.step()
                            total_steps += 1
                        samples_accum = []
                        if is_last_pp and is_last_tp:
                            loss_temp = torch.mean(torch.tensor(losses))
                            dist.all_reduce(loss_temp, group=dp_group, op=dist.ReduceOp.SUM)
                            loss_temp = loss_temp / args.dp_size
                            loss_accum += loss_temp
                            if total_steps % gradient_accumulation_steps == 0:
                                loss_to_log = loss_accum
                                if is_last_dp and is_last_pp and is_last_tp:
                                    print(f"Rank {rank} - Epoch {epoch} - Step {total_steps} - Loss:", loss_to_log)
                                loss_log.append(loss_to_log.item())
                                loss_accum = 0  

                
                ### TODOEND
                dist.barrier()
                
        output_gpu_memory_usage(f"[rank {rank}] after epoch {epoch}")
    # 清理进程组

    # save the loss log
    if is_last_dp and is_last_tp and is_last_pp:
        with open(f"loss_log_DP{args.dp_size}_PP{args.pp_size}_TP{args.tp_size}_stage{args.stage}_pipeline{args.pipeline}.txt", "w") as f:
            for loss in loss_log:
                loss = round(loss, 3)
                f.write(f"{loss}\n")

    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--micro_batch_size', type=int, default=2)
    parser.add_argument('--global_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--pipeline', type=str, choices=['simplepipe', 'gpipe'], default='simplepipe')
    args = parser.parse_args()
    assert args.world_size % (args.tp_size * args.pp_size) == 0
    args.dp_size = args.world_size // (args.tp_size * args.pp_size)

    return args

if __name__ == '__main__':
    # 设置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()
    
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)
