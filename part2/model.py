import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / d_k**0.5
        
        # print(scores.shape)
        # print(mask.shape)

        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.head_dim = embed_size // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.size()
        
        # Linear transformations
        query = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Adjust mask dimensions
        # if mask is not None:
        #     mask = mask.unsqueeze(1).unsqueeze(2) 

        # Apply attention
        attn_output, attn_weights = self.attention(query, key, value, mask)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)
        return self.fc_out(attn_output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_size, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_size)
        self.fc2 = nn.Linear(ff_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_size, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_size, ff_size, dropout)
        
        # Layer Norm and Dropout
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed forward network with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class CustomTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, ff_size, vocab_size, max_length, dropout=0.1):
        super(CustomTransformer, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        # Stack of Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_size, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        mask = None
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(x.device)
        
        # Add word and position embeddings
        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Pass through each Transformer layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final linear layer
        output = self.fc_out(x)
        return output
