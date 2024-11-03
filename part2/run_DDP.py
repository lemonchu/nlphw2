import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import CustomTransformer
from dataset import FixedRandomDataset
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from utils import output_gpu_memory_usage

def main(rank, world_size):
    # 初始化进程组
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)

    # 创建数据集和数据加载器，确保数据在进程间分割
    dataset = FixedRandomDataset(num_samples=20, seq_length=128, vocab_size=30522, seed=42)
    
    global_batch_size = 8
    micro_batch_size = 2
    gradient_accumulation_steps = global_batch_size // micro_batch_size // world_size
    
    data_loader = DataLoader(dataset, batch_size=micro_batch_size, rank=rank, world_size=world_size)
    
    # 初始化模型
    model = CustomTransformer(
        embed_size=128, 
        num_layers=6, 
        num_heads=8, 
        ff_size=512, 
        vocab_size=30522, 
        max_length=128, 
        dropout=0.1
    ).to('mps')
    output_gpu_memory_usage(f"[rank {rank}] after init model")
    # 确保所有进程的模型参数相同
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    output_gpu_memory_usage(f"[rank {rank}] after init optimizer")
    for epoch in range(100):
        data_loader.set_epoch(epoch)  # 确保每个 epoch 的数据顺序一致
        # 训练循环
        for i, sample in enumerate(data_loader):
            
            sample = {k: v.to('mps') for k, v in sample.items()}
            
            output = model(sample['input_ids'], sample['attention_mask'])
            loss = loss_criterion(output.view(-1, output.size(-1)), sample['labels'].view(-1))
            print(f"Rank {rank} - Epoch {epoch} - Sample {i} - Loss:", loss.item())
            
            loss.backward()

            # 梯度累积
            if (i + 1) % gradient_accumulation_steps == 0:
                # 手动同步梯度
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size  # 取平均梯度

                optimizer.step()
                optimizer.zero_grad()
        output_gpu_memory_usage(f"[rank {rank}] after epoch {epoch}")

    # 清理进程组
    dist.destroy_process_group()

class DataLoader:
    def __init__(self, dataset, batch_size=1, rank=0, world_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.total_size = len(self.dataset)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        # 确保所有进程使用相同的随机顺序
        g = torch.Generator()
        g.manual_seed(epoch)
        indices = torch.randperm(self.total_size, generator=g)
        # 将索引划分给各个进程
        self.indices = indices[self.rank::self.world_size]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch = [self.dataset[idx.item()] for idx in batch_indices]
            yield self.collate_fn(batch)
                
    def collate_fn(self, batch):
        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = torch.stack([item[key] for item in batch])
        return new_batch

if __name__ == '__main__':
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = 2  # 进程数量，可根据需要调整
    mp.spawn(main, args=(2,), nprocs=world_size, join=True)
