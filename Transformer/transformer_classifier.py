import math
import torch
from torch import nn

from Transformer.positional_encodings import get_pos_encoder
from Transformer.transformer_layer import TransformerBatchNormEncoderLayer
from Transformer.transformer_layer import TransformerLayerNormEncoderLayer
from Transformer.transformer_encoder import TransformerEncoder
from Transformer.positional_encodings import _get_activation_fn


class TSTransformerEncoderClassiregressor(nn.Module):
    ''''
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    We use batchnorm (reasons satted in the paper)
    Input:
        feat_dim --> feature dimension (SWAN-SF --> 25)
        max_len --> lenght of the longest sequnce in time over the entire dataset (for SWAN-SF --> 60 and all the same)
        d_model --> model dimension after projecting x_t, i.e., first Linear layer output dim (SWAN-SF --> try 64)
        n_heads --> Use standard 8 heads like in original paper
        num_layers --> number of transformer blocks in sucession (SWAN-SF --> try 3)
        dim_feedforward --> the dimension of the feedforward network model (SWAN-SF --> 256)
        num_classes --> (SWAN-SF --> 2)
        pos_encoding --> fiexed sinasoidal encodeings ore learnable (SWAN-SF --> try learnable)
        activation --> relu or gelu
        norm_mode --> either layer or batch, "batch_norm" is a self made function for time series data while "layer_norm" is a pytorch modelue for NLP tasks
    '''
    def __init__(self, feat_dim=1, max_len=1, d_model=1, n_heads=8, num_layers=1, dim_feedforward=1, num_classes=1,
                 dropout=0.1, pos_encoding='learnable', activation='gelu', norm_mode='layer_norm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm_mode == 'layer_norm': 
            encoder_layer = TransformerLayerNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Sequential(
        nn.Linear(d_model * max_len, num_classes),
        nn.Sigmoid()
        )
        return output_layer

    def forward(self, X, padding_masks=False):
        '''
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        '''
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim].
        # padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)

        # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(inp, src_mask=~padding_masks)

        # NOTE: If you want to use the attention maps, uncomment the part below
        # attention_maps = self.transformer_encoder.get_attention_maps(inp, src_mask=~padding_masks)
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(output)
        output = output.permute(1, 0, 2) # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # zero-out padding embeddings
        output = output * padding_masks.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.output_layer(output)  # (batch_size, num_classes)
        return output#, torch.stack(attention_maps) # Note: if you want to use attention maps, uncomment this part