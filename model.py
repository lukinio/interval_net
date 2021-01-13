import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import LinearInterval, Conv2dInterval


class SmallIntervalNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dInterval(1, 6, kernel_size=5, stride=2),
            nn.ReLU(),
            Conv2dInterval(6, 16, kernel_size=5),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            LinearInterval(16*8*8, 120),
            nn.ReLU(),
            LinearInterval(120, 50),
            nn.ReLU(),
        )
        self.bounds = None
        self.last_layer = LinearInterval(50, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        self.bounds = (x[:, :x.size(1) // 2], x[:, x.size(1) // 2:])
        x = self.last_layer(x)
        return x[:, :x.size(1) // 2], x[:, x.size(1) // 2:]
