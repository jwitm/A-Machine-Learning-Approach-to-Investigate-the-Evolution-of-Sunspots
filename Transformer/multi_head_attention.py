import math
from torch import nn
from torch.nn.modules import MultiheadAttention
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_mech = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        # single input u_t will be chunked into n_heads, so must be devisable (orange)
        assert (d_model % n_heads == 0), "dimensions do not work"

    def forward(self, q, k, v, mask=None):
        '''
        Input: q, k, v --> (seq_length, batch_size, d_model)
        Output: out --> (seq_length, batch_size, d_model)
                attention_weights --> (batch_size, n_heads, seq_len, seq_len)
        '''

        # multi-headed attention expects input to be of shape (batch_size, seq_len, d_model)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention_weights = self.attention_mech(
            q, k, v, mask=mask, dropout=self.dropout)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # restore to input shape (seq_length, batch_size, d_model)
        out = out.permute(1, 0, 2)

        return out, attention_weights

    def split(self, tensor):
        '''
        split tensor by number of head
        input --> (batch_size, seq_len, d_model)
        output --> (batch_size, n_heads, seq_len, d_head)
        '''
        batch_size, seq_len, d_model = tensor.size()

        d_head = d_model // self.n_heads
        tensor = tensor.view(batch_size, seq_len,
                             self.n_heads, d_head).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        '''
        inverse function of self.split(tensor : torch.Tensor)
        input --> (batch_size, n_heads, seq_len, d_head)
        return --> (batch_size, seq_len, d_model)
        '''
        batch_size, n_heads, seq_len, d_heads = tensor.size()
        d_model = n_heads * d_heads
        tensor = tensor.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        return tensor

    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask=None, dropout=None):
        '''
        Input:
            q, k, v --> (batch_size, n_heads, seq_len, d_head)
        Output:
            v --> (batch_size, n_heads, seq_len, d_head)
            attention_weights --> (batch_size, n_heads, seq_len, seq_len)
        '''
        batch_size, n_heads, seq_len, d_head = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        # (batch_size, n_heads, seq_len, seq_len)
        attention_weights = (q @ k_t) / math.sqrt(d_head)

        # 2. apply masking (opt)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask.eq(0), -10000)

        # 3. pass them softmax to make [0, 1] range
        # (batch_size, n_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 4. pass dropout (opt)
        if dropout is not None:
            # (batch_size, n_heads, seq_len, seq_len)
            attention_weights = self.attention_dropout(attention_weights)

        v = attention_weights @ v  # (batch_size, n_heads, seq_len, d_head)

        return v, attention_weights