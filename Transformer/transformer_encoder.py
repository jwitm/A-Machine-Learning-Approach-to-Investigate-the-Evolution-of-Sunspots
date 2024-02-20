from torch import nn

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, src_mask=None):
        for l in self.layers:
            x = l(x, src_mask=src_mask)
        return x

    def get_attention_maps(self, x, src_mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.attention(x, x, x)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps