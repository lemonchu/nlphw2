import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SimpleEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
    def forward(self, x):
        return self.embedding(x)
    
class SimpleOutputHead(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(SimpleOutputHead, self).__init__()
        self.linear = nn.Linear(embed_size, vocab_size, bias=False)
    def forward(self, x):
        return self.linear(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, tp_group=None):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.head_dim = embed_size // num_heads
        self.num_heads = num_heads
        self.tp_group = tp_group

        if self.tp_group.size() == 1:
            self.query = nn.Linear(embed_size, embed_size)
            self.key = nn.Linear(embed_size, embed_size)
            self.value = nn.Linear(embed_size, embed_size)
            self.fc_out = nn.Linear(embed_size, embed_size)
        else:
            ### TODO: Implement tensor parallel attention
            pass
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
            raise NotImplementedError
            ### TODOEND

class ColumnParallelOutputHead(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super(ColumnParallelOutputHead, self).__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x):
        x = self.linear(x)
        x = torch.cat(torch.split(x, self.tp_group.size()), dim=1)
        return x

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super(ColumnParallelLinear, self).__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        x = self.linear(x)
        x = torch.cat(torch.split(x, self.tp_group.size()), dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, embed_size, ff_size, tp_group=None):
        super(MLP, self).__init__()
        self.tp_group = tp_group
        if self.tp_group.size() == 1:
            self.fc1 = nn.Linear(embed_size, ff_size)
            self.fc2 = nn.Linear(ff_size, embed_size)
        else:
            ### TODO: Implement tensor parallel MLP
            self.fc1 = ColumnParallelLinear(embed_size, ff_size // 2, self.tp_group)
            self.fc2 = RowParallelLinear(ff_size // 2, embed_size, self.tp_group)
            raise NotImplementedError
            ### TODOEND

    def forward(self, x):
        if self.tp_group.size() == 1:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
        else:
            ### TODO: Implement tensor parallel MLP
            raise NotImplementedError
            ### TODOEND

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size, tp_group=None):
        super(TransformerLayer, self).__init__()
        self.tp_group = tp_group
        if self.tp_group.size() == 1:
            self.attention = MultiHeadAttention(embed_size, num_heads, self.tp_group)
            self.feed_forward = MLP(embed_size, ff_size, self.tp_group)
        else:
            ### TODO: Implement tensor parallel attention
            raise NotImplementedError
            ### TODOEND
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        
        if self.tp_group.size() == 1:
            attn_output = self.attention(x)
            x = x + attn_output
            x = self.norm1(x)

            # Feed forward network with residual connection
            ff_output = self.feed_forward(x)
            x = x + ff_output
            x = self.norm2(x)
        else:
            ### TODO: Implement tensor parallel attention
            raise NotImplementedError
            ### TODOEND
        return x

class CustomTransformer(nn.Module):
    def __init__(self, embed_size, 
                 num_layers, 
                 num_heads, 
                 ff_size, 
                 vocab_size, 
                 tp_group=None,
                 pp_group=None,
                 rank=None):
        super(CustomTransformer, self).__init__()
        self.embed_size = embed_size

        self.tp_group = tp_group
        self.pp_group = pp_group
        self.rank = rank
        self.is_first_pp = self.pp_group.rank() == 0
        self.is_last_pp = self.pp_group.rank() == self.pp_group.size() - 1

        if self.is_first_pp:
            if self.tp_group.size() == 1:
                self.word_embedding = SimpleEmbedding(vocab_size, embed_size)
            else:
                self.word_embedding = RowParallelEmbedding(vocab_size//self.tp_group.size(), embed_size, self.tp_group)
            
        each_rank_layer_size = num_layers // self.pp_group.size()
        self.layers_id = [f"layer_{i}" for i in  range(each_rank_layer_size*self.pp_group.rank(), each_rank_layer_size*(self.pp_group.rank()+1))]
        self.layers = nn.ModuleDict({
            layer_id: TransformerLayer(embed_size, num_heads, ff_size, self.tp_group)
            for layer_id in self.layers_id
        })

        if self.is_last_pp:
            if self.tp_group.size() == 1:
                self.embedding_to_logits = SimpleOutputHead(embed_size, vocab_size)
            else:
                self.embedding_to_logits = ColumnParallelOutputHead(embed_size, vocab_size//self.tp_group.size(), self.tp_group)
                
            # raise NotImplementedError
            ### TODOEND 
    def forward(self, x):

        # We ignore position embedding for simplisity
        if self.pp_group.size() == 1:
            print(x.shape,x.dtype)
            x = self.word_embedding(x)
            for layer_id in self.layers_id:
                x = self.layers[layer_id](x)
            output = self.embedding_to_logits(x)
            x = None
        else:
            if not self.is_first_pp:
                zeros = torch.zeros([x.shape[0], x.shape[1], self.embed_size], device=x.device, dtype=torch.float32)
                x = zeros
                # print(f'[rank {self.rank}] recv')
                torch.distributed.recv(x, src=self.rank - self.tp_group.size(), group=self.pp_group)
                # print(f'[rank {self.rank}] recv done')
            else: 
                # print(f'[rank {self.rank}] embedding')
                x = self.word_embedding(x)
                # print(f'[rank {self.rank}] embedding done')
            
            for layer_id in self.layers_id:
                # print(f'[rank {self.rank}] layer {layer_id}')
                x = self.layers[layer_id](x)
                # print(f'[rank {self.rank}] layer {layer_id} done')
            
            if not self.is_last_pp:
                # print(f'[rank {self.rank}] send')
                torch.distributed.send(x, dst=self.rank + self.tp_group.size(), group=self.pp_group)
                output = None
                # print(f'[rank {self.rank}] send done')
            else:   
                output = self.embedding_to_logits(x)
        return output

class RowParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, tp_group):
        super(RowParallelEmbedding, self).__init__()
        self.tp_group = tp_group
        self.embedding = nn.Embedding(vocab_size, embed_size)
    def forward(self, x):
        ### TODO: Implement tensor parallel embedding
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = torch.cat(torch.split(x, self.tp_group.size()), dim=1)
        ### TODOEND
        return x
    
class ColumnParallelOutputHead(nn.Module):
    def __init__(self, embed_size, vocab_size, tp_group):
        super(ColumnParallelOutputHead, self).__init__()
        self.tp_group = tp_group
        self.embedding = nn.Embedding(vocab_size, embed_size)
    def forward(self, x):
        ### TODO: Implement tensor parallel embedding
        x = self.embedding(x)
        x = torch.cat(torch.split(x, self.tp_group.size()), dim=1)
        ### TODOEND
        return x
