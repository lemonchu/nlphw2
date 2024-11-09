import torch
import torch.nn as nn
import torch.nn.functional as F

from tp_forward import column_parallel_linear_forward, row_parallel_linear_forward, row_parallel_embedding_forward

from tp_forward import parallel_attention_forward

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, tp_group=None):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.head_dim = embed_size // num_heads
        self.num_heads = num_heads // tp_group.size()
        self.tp_group = tp_group

        if self.tp_group.size() == 1:
            self.query = nn.Linear(embed_size, embed_size)
            self.key = nn.Linear(embed_size, embed_size)
            self.value = nn.Linear(embed_size, embed_size)
            self.fc_out = nn.Linear(embed_size, embed_size)
        else:
            ### TODO: Implement tensor parallel attention
            pass
            self.query = nn.Linear(embed_size, embed_size // self.tp_group.size())
            self.key = nn.Linear(embed_size, embed_size // self.tp_group.size())
            self.value = nn.Linear(embed_size, embed_size // self.tp_group.size())
            self.fc_out = nn.Linear(embed_size//self.tp_group.size(), embed_size)
            ### TODOEND
    
    def forward(self, x):
        batch_size, seq_length, embed_size = x.size()
        
        # Linear transformations
        if self.tp_group.size() == 1:
        
            query = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            key = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_length, seq_length]
            attention = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attention, value)
            # Concatenate heads and put through final linear layer
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)
            return self.fc_out(attn_output)
        else:
            ### TODO: Implement tensor parallel attention
            query = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            key = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_length, seq_length]
            attention = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attention, value)
            # Concatenate heads and put through final linear layer
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size // self.tp_group.size())

            output = self.fc_out(attn_output)
            output = parallel_attention_forward.apply(output, self.tp_group)
            return output
            ### TODOEND

class OutputHead(nn.Module):

    def __init__(self, in_features, out_features, tp_group):
        super(OutputHead, self).__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x):
        x = self.linear(x)
        return x
    
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super(ColumnParallelLinear, self).__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        if self.tp_group.size() == 1:
            return self.linear(x)
        else:
            args = [x, self.linear.weight, self.linear.bias, self.tp_group]
            x = column_parallel_linear_forward.apply(*args)
            return x

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super(RowParallelLinear, self).__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        if self.tp_group.size() == 1:
            return self.linear(x)
        else:
            args = [x, self.linear.weight, self.linear.bias, self.tp_group]
            x = row_parallel_linear_forward.apply(*args)
            return x

class MLP(nn.Module):
    def __init__(self, embed_size, ff_size, tp_group=None):
        super(MLP, self).__init__()
        self.tp_group = tp_group
        
        self.fc1 = ColumnParallelLinear(embed_size, ff_size // self.tp_group.size(), self.tp_group)
        self.fc2 = RowParallelLinear(ff_size // self.tp_group.size(), embed_size, self.tp_group)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size, tp_group=None):
        super(TransformerLayer, self).__init__()
        self.tp_group = tp_group
        
        self.attention = MultiHeadAttention(embed_size, num_heads, self.tp_group)
        self.feed_forward = MLP(embed_size, ff_size, self.tp_group)
    
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output

        # Feed forward network with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
    
        return x

class CustomTransformer(nn.Module):
    def __init__(self, embed_size, 
                 num_layers, 
                 num_heads, 
                 ff_size, 
                 vocab_size,
                 tp_group=None):
        super(CustomTransformer, self).__init__()
        self.tp_group = tp_group
        self.embed_size = embed_size
        word_embedding = RowParallelEmbedding(vocab_size//self.tp_group.size(), self.embed_size, self.tp_group)
        
        layers = [
            TransformerLayer(embed_size, num_heads, ff_size, self.tp_group)
            for _ in range(num_layers)
        ]
        embedding_to_logits = OutputHead(self.embed_size, vocab_size, self.tp_group)

        self.core = nn.Sequential(
            word_embedding,
            *layers,
            embedding_to_logits
        ) 
    def forward(self, x):

        # We ignore position embedding for simplisity
        return self.core(x)
    
    def __len__(self):
        return len(self.core)

class RowParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, tp_group):
        super(RowParallelEmbedding, self).__init__()
        self.tp_group = tp_group
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.vocab_start_index = self.vocab_size * self.tp_group.rank()
        self.vocab_end_index = self.vocab_start_index + self.vocab_size
    def forward(self, x):
        ### TODO: Implement tensor parallel embedding
        if self.tp_group.size() == 1:
            x = nn.functional.one_hot(x, num_classes=self.vocab_size, ignore_index=-1).float()
            x = self.embedding(x)
            return x
        else:
            
            input_mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)

            # Adjust indices to be relative to this partition
            x_adjusted = x - self.vocab_start_index

            # Replace indices not in this partition with zero (will be masked out)
            x_adjusted = x_adjusted.masked_fill(~input_mask, 0)

            # Perform embedding lookup for the adjusted indices
            output_parallel = self.embedding(x_adjusted)

            # Zero out embeddings for tokens not in this partition
            output_parallel = output_parallel * input_mask.unsqueeze(-1).float()

            # All-reduce to sum embeddings from all partitions
            x = row_parallel_embedding_forward.apply(output_parallel, self.tp_group)
            return x
