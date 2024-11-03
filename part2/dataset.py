import torch
from torch.utils.data import Dataset

class FixedRandomDataset(Dataset):
    def __init__(self, num_samples=5, seq_length=128, vocab_size=30522, seed=42):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed
        # 生成固定的随机input_ids和attention_mask
        torch.manual_seed(self.seed)
        self.input_ids = torch.randint(0, self.vocab_size, (self.num_samples, self.seq_length))
        self.attention_mask = torch.randint(0, 2, (self.num_samples, self.seq_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx][:-1],
            'attention_mask': self.attention_mask[idx][:-1],
            'labels': self.input_ids[idx][1:]
        }