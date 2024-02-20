from torchvision.models import vgg16
import torch.nn as nn

class Sunspot_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained vgg16 model
        self.part1 = vgg16(pretrained=True)

        # add fcl to perform binary classification
        self.fully_connected = nn.Sequential(
            nn.Linear(1000, 100),
            nn.LayerNorm(normalized_shape=[(100)]),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.part1(x)
        x = self.fully_connected(x)
        return x