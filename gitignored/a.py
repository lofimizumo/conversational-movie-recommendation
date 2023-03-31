import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import os

class AttentionBlock(nn.Module):
    #generate the doc
    """
    Multi-head attention block.
    Args:
        embed_dim: (int) Embedding dimension.
        heads: (int) Number of attention heads.
        dropout: (float) Dropout probability.
    """

    def __init__(self, embed_dim, heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        assert (self.head_dim * heads == embed_dim), "Embedding dimension not divisible by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # split embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, heads_dim)
        queries = self.queries(queries)  # (N, query_len, heads, heads_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # x.shape: (N, query_len, heads * heads_dim)
        out = self.dropout(self.fc_out(out))
        # x.shape: (N, query_len, embed_dim)
        return out


#test this attention block
# x = torch.randn((2, 3, 512))
# sample_attention = AttentionBlock(embed_dim=512, heads=8)
# sample_attention(x, x, x, None).shape

#define a sort function 
def sort_by_seq_len(x, seq_len):
    #generate the doc
    """
    Sort a batch of sequences by their length.
    Args:
        x: (batch_size, max_seq_len, embed_dim)
        seq_len: (batch_size)
    Returns:
        sorted_x: (batch_size, max_seq_len, embed_dim)
        sorted_seq_len: (batch_size)
        recover_idx: (batch_size)
    """
    #sort the sequence length
    sorted_seq_len, perm_idx = seq_len.sort(0, descending=True)
    _, recover_idx = perm_idx.sort(0, descending=False)
    sorted_x = x[perm_idx]
    return sorted_x, sorted_seq_len, recover_idx





if __name__ == "__main__":
    x = torch.randn((2, 3, 512))
    seq_len = torch.tensor([3, 2])
    sorted_x, sorted_seq_len, recover_idx = sort_by_seq_len(x, seq_len)
    print(sorted_x)

