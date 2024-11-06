import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import CustomTransformer
# 定义dummy数据集类
from dataset import FixedRandomDataset

from utils import output_gpu_memory_usage

class DataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        torch.manual_seed(epoch)
        indices = torch.randperm(len(self.dataset))
        self.indices = indices

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = [self.dataset[idx] for idx in self.indices[i:i+self.batch_size]]
            yield self.collate_fn(batch)
            
    def collate_fn(self, batch):
        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = torch.stack([item[key] for item in batch])
        return new_batch
# 创建固定随机数据集实例

device = 'cuda:0'

dataset = FixedRandomDataset(num_samples=20, seq_length=128, vocab_size=30522, seed=42)

global_batch_size = 8
micro_batch_size = 2

data_loader = DataLoader(dataset, batch_size=micro_batch_size)
gradient_accumulation_steps = global_batch_size // micro_batch_size

output_gpu_memory_usage("before init model")

model = CustomTransformer(
    embed_size=128, 
    num_layers=6, 
    num_heads=8, 
    ff_size=512, 
    vocab_size=30522, 
    max_length=128, 
    dropout=0.1
).to(device)
output_gpu_memory_usage("after init model")
loss_criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
output_gpu_memory_usage("after init optimizer")
for epoch in range(100):
    data_loader.set_epoch(epoch)
    # 检查数据
    for i, sample in enumerate(data_loader):
        # print(f"Sample {i} - Attention Mask:", sample['attention_mask'])
        
        sample = {k: v.to(device) for k, v in sample.items()}
        
        output = model(sample['input_ids'], sample['attention_mask'])
        loss = loss_criterion(output.view(-1, output.size(-1)), sample['labels'].view(-1))
        print(f"Epoch {epoch} - Sample {i} - Loss:", loss.item())
        
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
 
    output_gpu_memory_usage(f"after epoch {epoch}")
    