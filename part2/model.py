import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, tp_group=None):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.head_dim = embed_size // num_heads
        self.num_heads = num_heads
        self.tp_group = tp_group

        if self.tp_group is None:
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
        if self.tp_group is None:
        
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

class MLP(nn.Module):
    def __init__(self, embed_size, ff_size, tp_group=None):
        super(MLP, self).__init__()
        self.tp_group = tp_group
        if self.tp_group is None:
            self.fc1 = nn.Linear(embed_size, ff_size)
            self.fc2 = nn.Linear(ff_size, embed_size)
        else:
            ### TODO: Implement tensor parallel MLP
            raise NotImplementedError
            ### TODOEND

    def forward(self, x):
        if self.tp_group is None:
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
        if self.tp_group is None:
            self.attention = MultiHeadAttention(embed_size, num_heads)
            self.feed_forward = MLP(embed_size, ff_size)
        else:
            ### TODO: Implement tensor parallel attention
            raise NotImplementedError
            ### TODOEND
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        
        if self.tp_group is None:

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
                 pp_group=None):
        super(CustomTransformer, self).__init__()
        self.tp_group = tp_group
        self.pp_group = pp_group

        self.embed_size = embed_size
        
        if self.pp_group is None:
            if self.tp_group is None:
                self.word_embedding = nn.Embedding(vocab_size, embed_size)
                self.embedding_to_logits = nn.Linear(embed_size, vocab_size)
            else:
                ### TODO[Optional]: you can change the code for your own implementation
                self.word_embedding = RowParallelEmbedding(vocab_size//self.tp_group.size(), embed_size, self.tp_group)
                self.embedding_to_logits = ColumnParallelEmbedding(embed_size, vocab_size//self.tp_group.size(), self.tp_group)
                ### TODOEND
            self.layers = nn.ModuleList([
                TransformerLayer(embed_size, num_heads, ff_size, self.tp_group)
                for _ in range(num_layers)
            ])
        else:
            ### TODO[Optional]: Implement pipeline parallel embedding(you can change the code)
            raise NotImplementedError
            ### TODOEND 

    def forward(self, x):

        # We ignore position embedding for simplisity
        if self.pp_group is None:
            x = self.word_embedding(x) 
            for layer in self.layers:
                x = layer(x)
            output = self.embedding_to_logits(x)
        else:
            ### TODO: Implement pipeline parallel embedding
            raise NotImplementedError
            ### TODOEND
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
    
class ColumnParallelEmbedding(nn.Module):
    def __init__(self, embed_size, vocab_size, tp_group):
        super(ColumnParallelEmbedding, self).__init__()
        self.tp_group = tp_group
        self.embedding = nn.Embedding(vocab_size, embed_size)
    def forward(self, x):
        ### TODO: Implement tensor parallel embedding
        x = self.embedding(x)
        x = torch.cat(torch.split(x, self.tp_group.size()), dim=1)
        ### TODOEND
        return x
