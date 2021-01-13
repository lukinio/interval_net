import torch
import torch.nn as nn
import torch.nn.functional as f


class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        x_lower = x[:, :x.size(1) // 2]
        x_upper = x[:, x.size(1) // 2:]
        center = (x_upper + x_lower) / 2
        radius = (x_upper - x_lower) / 2
        new_center = f.linear(center, self.weight, self.bias)
        new_radius = f.linear(radius, self.weight.abs(), None)
        return torch.cat([new_center - new_radius, new_center + new_radius], dim=1)


class Conv2dInterval(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

    def forward(self, x):
        x_lower = x[:, :x.size(1) // 2]
        x_upper = x[:, x.size(1) // 2:]
        center = (x_upper + x_lower) / 2
        radius = (x_upper - x_lower) / 2
        new_center = f.conv2d(center, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
        new_radius = f.conv2d(radius, torch.abs(self.weight), None, self.stride,
                              self.padding, self.dilation, self.groups)
        return torch.cat([new_center - new_radius, new_center + new_radius], dim=1)
