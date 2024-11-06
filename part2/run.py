import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from utils import output_gpu_memory_usage
import argparse

from optimizer import SimpleOptimizer, ZeroOptimizer
from model import CustomTransformer
from dataset import FixedRandomDataset, SimpleDataLoader, DataParallelDataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def init_3d_parallel_group(world_size, tp_size, pp_size, dp_size):
    # Ensure world size matches the 3D partitioning
    assert world_size == tp_size * pp_size * dp_size, "world_size must equal tp_size * pp_size * dp_size"
    
    rank = dist.get_rank()
    
    # Calculate ranks for tensor, pipeline, and data parallel groups
    tp_rank = rank % tp_size
    pp_rank = (rank // tp_size) % pp_size
    dp_rank = rank // (tp_size * pp_size)

    # Initialize tensor parallel group
    if tp_size > 1:
        tp_group_ranks = [r for r in range(rank - tp_rank, rank - tp_rank + tp_size)]
        tp_group = dist.new_group(ranks=tp_group_ranks)
    else:
        tp_group = None

    # Initialize pipeline parallel group
    if pp_size > 1:
        pp_group_ranks = [r for r in range( dp_rank * (tp_size * pp_size) + tp_rank, dp_rank * (tp_size * pp_size) + tp_rank + tp_size*pp_size, tp_size)]
        pp_group = dist.new_group(ranks=pp_group_ranks)
    else:
        pp_group = None

    # Initialize data parallel group
    if dp_size > 1:
        dp_group_ranks = [r for r in range( pp_rank * tp_size + tp_rank, world_size , tp_size)]
        dp_group = dist.new_group(ranks=dp_group_ranks)
    else:
        dp_group = None

    # 输出当前rank所在的各个并行组信息
    print(f"[Rank {rank}] TP Group: {tp_group_ranks if tp_size > 1 else None}")
    print(f"[Rank {rank}] PP Group: {pp_group_ranks if pp_size > 1 else None}")  
    print(f"[Rank {rank}] DP Group: {dp_group_ranks if dp_size > 1 else None}")

    return tp_group, pp_group, dp_group


def main(rank, world_size, args):
    # 初始化进程组
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)

    tp_group, pp_group, dp_group = init_3d_parallel_group(world_size, args.tp_size, args.pp_size, args.dp_size)

    # 创建数据集和数据加载器，确保数据在进程间分割
    dataset = FixedRandomDataset(num_samples=80, seq_length=128, vocab_size=4000, seed=42)
   
    if dp_group is None:
        data_loader = SimpleDataLoader(dataset, batch_size=args.micro_batch_size) 
    else:
        data_loader = DataParallelDataLoader(dataset, batch_size=args.micro_batch_size, rank=dp_group.rank() if dp_group is not None else 0, dp_group = dp_group)
    # print(len(data_loader))
    # 初始化模型
    model = CustomTransformer(
        embed_size=256, 
        num_layers=16, 
        num_heads=16, 
        ff_size=1024, 
        vocab_size=4000,
        tp_group=tp_group,
        pp_group=pp_group
    ).to(device)

    output_gpu_memory_usage(f"[rank {rank}] after init model")
    # 确保所有进程的模型参数相同
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    gradient_accumulation_steps = args.global_batch_size // args.micro_batch_size // world_size
    if dp_group is None:
        optimizer = SimpleOptimizer(model, lr=0.001, weight_decay=0.0, gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        optimizer = ZeroOptimizer(model, dp_group=dp_group, lr=0.001, weight_decay=0.0, stage=args.stage, gradient_accumulation_steps=gradient_accumulation_steps)
    
    loss_criterion = nn.CrossEntropyLoss()
    
    output_gpu_memory_usage(f"[rank {rank}] after init optimizer")
    
    loss_log = []
    
    total_steps = 0
    for epoch in range(args.epoch):
        data_loader.set_epoch(epoch)  # 确保每个 epoch 的数据顺序一致
        # 训练循环
        loss_accum = 0
        for i, sample in enumerate(data_loader):
            sample = {k: v.to(device) for k, v in sample.items()}
            output = model(sample['input_ids'])
            if pp_group is None:
                loss = loss_criterion(output.view(-1, output.size(-1)), sample['labels'].view(-1))
                optimizer.step(loss)
                total_steps += 1
                loss_temp = loss.detach().cpu()
                dist.all_reduce(loss_temp, group=dp_group, op=dist.ReduceOp.SUM)
                loss_temp = loss_temp / args.dp_size
                loss_accum += loss_temp
                # print(loss_accum)
                if total_steps % gradient_accumulation_steps == 0:
                    loss_to_log = loss_accum / gradient_accumulation_steps
                    if rank == 0:
                        print(f"Rank {rank} - Epoch {epoch} - Step {total_steps} - Loss:", loss_to_log)
                    loss_log.append(loss_to_log.item())
                    
                    loss_accum = 0
            else:
                ### TODO: Implement the loss calculation for pipeline parallel
                ### Hint: Only last pp rank will calculate the loss
                raise NotImplementedError
                ### TODOEND
                
        output_gpu_memory_usage(f"[rank {rank}] after epoch {epoch}")
    # 清理进程组

    # save the loss log
    if rank == 0:
        with open(f"loss_log_DP{args.dp_size}_PP{args.pp_size}_TP{args.tp_size}_stage{args.stage}.txt", "w") as f:
            for loss in loss_log:
                loss = round(loss, 3)
                f.write(f"{loss}\n")

    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--tp-size', type=int, default=1)
    parser.add_argument('--pp-size', type=int, default=1)
    parser.add_argument('--micro_batch_size', type=int, default=2)
    parser.add_argument('--global_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--stage', type=int, default=0)
    args = parser.parse_args()
    assert args.world_size % (args.tp_size * args.pp_size) == 0
    args.dp_size = args.world_size // (args.tp_size * args.pp_size)

    return args

if __name__ == '__main__':
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    args = parse_args()
    
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)
