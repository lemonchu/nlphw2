import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import CustomTransformer
from dataset import FixedRandomDataset
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from utils import output_gpu_memory_usage

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def align_size(data, bucket_size):
    shape = data.shape
    data = data.view(-1)
    if data.numel() % bucket_size != 0:
        data = torch.cat([data, torch.zeros(bucket_size - data.numel() % bucket_size, dtype=data.dtype, device=data.device)])
    return data

class ZeroOptimizer:
    def __init__(self, model, group=None, lr=0.001, weight_decay=0.0,gradient_accumulation_steps=1):
        self.model = model
        if group is None:
            self.group = dist.group.WORLD
        else:
            self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
        self.pbuckets = []
       
        self.lr = lr
        self.weight_decay = weight_decay
        self.partition_parameters()
        self.optimizer = torch.optim.AdamW(self.pbuckets, lr=self.lr, weight_decay=self.weight_decay)
        self.gradient_accumulation_steps = gradient_accumulation_steps
    def partition_parameters(self):
        
        # Partition parameters equally among processes
        
        for param in self.model.parameters():
            data = param.data.view(-1)
    
            data = align_size(data, self.group.size())
            # only select the data of this rank, and copy to the param_shard
            data = data.view(self.group.size(), -1)
            param_shard = data[self.rank].clone()

            self.pbuckets.append(torch.nn.Parameter(param_shard, requires_grad=True))
        
    def step(self):

        for (pbucket, param) in zip(self.pbuckets, self.model.parameters()):
            
            grad = torch.zeros_like(pbucket,requires_grad=False)

            data = param.grad.view(-1).clone()
            data = align_size(data, self.group.size())

            data = data / self.group.size() / self.gradient_accumulation_steps
            dist.reduce_scatter_tensor(grad, data, group=self.group)
             
            pbucket.grad = grad

        self.optimizer.step()
        
        for (pbucket, param) in zip(self.pbuckets, self.model.parameters()):
            shape = param.shape
            model_param = torch.zeros_like(param, dtype=param.dtype, device=param.device)
            model_param = align_size(model_param, self.group.size()).view(-1)

            dist.all_gather_into_tensor(model_param, pbucket.data.view(-1), group=self.group)
            # print(f"[rank {self.rank}] data: {pbucket.data[:10]}")
            # print(f"[rank {self.rank}] model_param: {model_param[:10]}")
            param.data = model_param[:param.numel()].clone().view(shape)
            
        self.optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

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
    ).to(device)
    output_gpu_memory_usage(f"[rank {rank}] after init model")
    # 确保所有进程的模型参数相同
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    loss_criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = ZeroOptimizer(model, group=dist.group.WORLD, lr=0.001, weight_decay=0.0)
    output_gpu_memory_usage(f"[rank {rank}] after init optimizer")
    for epoch in range(100):
        data_loader.set_epoch(epoch)  # 确保每个 epoch 的数据顺序一致
        # 训练循环
        for i, sample in enumerate(data_loader):
            
            sample = {k: v.to(device) for k, v in sample.items()}
            
            output = model(sample['input_ids'], sample['attention_mask'])
            loss = loss_criterion(output.view(-1, output.size(-1)), sample['labels'].view(-1))
            print(f"Rank {rank} - Epoch {epoch} - Sample {i} - Loss:", loss.item())
            
            loss.backward()
            
            # 梯度累积
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
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

    world_size = 4  # 进程数量，可根据需要调整
    mp.spawn(main, args=(4,), nprocs=world_size, join=True)
