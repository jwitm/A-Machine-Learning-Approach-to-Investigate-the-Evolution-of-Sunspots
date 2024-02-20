import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from Transformer.transformer_classifier import TSTransformerEncoderClassiregressor
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
    
class Hybrid(nn.Module):
    def __init__(self):
        super(Hybrid, self).__init__()
        self.convnet = vgg16(weights = VGG16_Weights.DEFAULT)

        # If you want to load the weights from a file, use the following code
        # state_dict = torch.load("/path/to/your/vgg16.pth") 
        # self.convnet.load_state_dict(state_dict)

        self.max_len = 148 # change this according to your data

        self.transformer = TSTransformerEncoderClassiregressor(feat_dim=1000, max_len=self.max_len, d_model=32, n_heads=4, num_layers=3, dim_feedforward=2*32, num_classes=1,
                 dropout=0.5, pos_encoding='fixed', activation='relu', norm_mode='layer_norm', freeze=False)     
        

    def forward(self, x):
        device = next(self.convnet.parameters()).device
        B, C, T, H, W = x.shape

        x = x.reshape(B*T,C,H,W)            # reshape to feed to vgg16
        
        x = checkpoint(self.convnet, x, use_reentrant = False)                # [T*B,1000]
        x = x.reshape(B,T,-1)               # [B, T, 1000]

        if T<self.max_len:
            pad_size = self.max_len - T
            x = F.pad(x, (0,0,0,pad_size), "constant", 0)
            padding_masks = torch.ones(B, T, dtype=torch.bool, device=x.device)
            padding_masks = F.pad(padding_masks, (0, pad_size), "constant", 0)
        else:
            padding_masks = torch.ones(B, T, dtype=torch.bool, device=x.device)

        x = self.transformer(x,padding_masks) # input should be [batch_size, seq_length, feat_dim].

        return x#, attention_maps # NOTE: If you want to use the attention maps, uncomment the part in the forward function