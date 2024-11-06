import torch
from torch.utils.data import Dataset

class FixedRandomDataset(Dataset):
    def __init__(self, num_samples=5, seq_length=128, vocab_size=4000, seed=42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed
        # 生成固定的随机input_ids和attention_mask
        torch.manual_seed(self.seed)
        self.input_ids = torch.randint(0, self.vocab_size, (self.num_samples, self.seq_length))
      
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx][:-1],  
            'labels': self.input_ids[idx][1:]
        }
    

class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = None
        self.set_epoch(0)

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

class DataParallelDataLoader:
    def __init__(self, dataset, batch_size=1, rank=0, dp_group=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dp_group = dp_group
        self.dp_rank = dp_group.rank()
        self.total_size = len(self.dataset)
        # print(self.dp_group.size())
        self.set_epoch(0)
    def set_epoch(self, epoch):
        ### TODO: make sure all ranks have the same random order
        ### this function is called before each epoch
        torch.manual_seed(epoch)
        indices = torch.randperm(len(self.dataset))
        self.indices = indices[self.dp_rank::self.dp_group.size()]
        ### TODOEND

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        ### TODO: make sure each rank only get a subset of the dataset
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch = [self.dataset[idx.item()] for idx in batch_indices]
            yield self.collate_fn(batch)
        ### TODO END
                
    def collate_fn(self, batch):
        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = torch.stack([item[key] for item in batch])
        return new_batch