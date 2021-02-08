import torch
import torch.nn as nn
import torch.nn.functional as f
from interval_layers import LinearInterval, Conv2dInterval


class Small(nn.Module):

    def __init__(self, eps=0):
        super().__init__()
        self.conv1 = Conv2dInterval(1, 16, 4, 2, eps=eps, input_layer=True)
        self.conv2 = Conv2dInterval(16, 32, 4, 1, eps=eps)
        self.fc1 = LinearInterval(32*10*10, 100, eps=eps)
        self.last_layer = LinearInterval(100, 10, eps=eps)

        self.bounds = None

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        self.save_bounds(x)
        x = self.last_layer(x)
        return x[:, :x.size(1)//3]
    
    def save_bounds(self, x):
        s = x.size(1) // 3
        self.bounds = x[:, s:2*s], x[:, 2*s:]
    
    def set_eps(self, eps):
        for layer in self.children():
            layer.eps = eps
