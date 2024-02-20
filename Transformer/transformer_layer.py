from torch import nn
from ..Transformer.multi_head_attention import MultiHeadAttention

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    '''
    (seq_len, batch_size, d_model) --> (seq_len, batch_size, d_model)
    This transformer encoder layer block is made up of self-attention and a feedforward network (no activation).
    It differs from the TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm
    Input
    -----
        d_model --> the number of expected features in the input (required).
        n_head --> the number of heads in the multiheadattention models (required).
        dim_feedforward --> the dimension of the feedforward network model (default=256).
        dropout --> the dropout value (default=0.1).
        activation --> the activation function of intermediate layer, relu or gelu (default=relu).
    '''
    def __init__(self, d_model, n_heads, dim_feedforward=256, dropout=0.1, activation='relu'):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        # normalizes each feature across batch samples and time steps
        self.norm1 = nn.BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = nn.BatchNorm1d(d_model, eps=1e-5)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model))
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        '''
        Input
        -----
            src: the sequence to the encoder layer (required) (seq_len, batch_size, d_model)
            src_mask: the mask for the src sequence (optional) (seq_len, batch_size, d_model)
            src_key_padding_mask: the mask for the src keys per batch (optional)
            
        Output --> (seq_len, batch_size, d_model)
        ------
        '''
        att, attention_weights = self.attention(src, src, src)
        src = src + self.dropout1(att)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        forward = self.feed_forward(src)
        out = src + self.dropout2(forward)  # (seq_len, batch_size, d_model)
        out = out.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        out = self.norm2(out)
        out = out.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return out
    

class TransformerLayerNormEncoderLayer(nn.modules.Module):
    '''
    (seq_len, batch_size, d_model) --> (seq_len, batch_size, d_model)
    This transformer encoder layer block is made up of self-attention and a feedforward network (no activation).
    Input
    -----
        d_model --> the number of expected features in the input (required).
        n_head --> the number of heads in the multiheadattention models (required).
        dim_feedforward --> the dimension of the feedforward network model (default=256).
        dropout --> the dropout value (default=0.1).
        activation --> the activation function of intermediate layer, relu or gelu (default=relu).
    '''
    def __init__(self, d_model, n_heads, dim_feedforward=256, dropout=0.1, activation='relu'):
        super(TransformerLayerNormEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 =  nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 =  nn.LayerNorm(d_model, eps=1e-6)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model))
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        '''
        Input
        -----
            src: the sequence to the encoder layer (required) (seq_len, batch_size, d_model)
            src_mask: the mask for the src sequence (optional) (seq_len, batch_size, d_model)
            src_key_padding_mask: the mask for the src keys per batch (optional)
            
        Output --> (seq_len, batch_size, d_model)
        ------
        '''
        att, attention_weights = self.attention(src, src, src)
        src = src + self.dropout1(att)  # (seq_len, batch_size, d_model)
        # src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        # src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        forward = self.feed_forward(src)
        out = src + self.dropout2(forward)  # (seq_len, batch_size, d_model)
        # out = out.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        out = self.norm2(out)
        # out = out.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return out
    
