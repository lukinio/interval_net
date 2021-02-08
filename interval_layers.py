import torch
import torch.nn as nn
import torch.nn.functional as f


class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=0, input_layer=False):
        super().__init__(in_features, out_features, bias)
        self.eps = eps
        self.input_layer = input_layer

    def forward(self, x):
        if self.input_layer:
            x = torch.cat((x, x, x), dim=1)

        s = x.size(1) // 3
        x_middle = x[:, :s]
        x_lower = x[:, s:2 * s]
        x_upper = x[:, 2 * s:]

        middle = super().forward(x_middle)

        w_lower_pos = (self.weight - self.eps).clamp(min=0).t()
        w_lower_neg = (self.weight - self.eps).clamp(max=0).t()
        w_upper_pos = (self.weight + self.eps).clamp(min=0).t()
        w_upper_neg = (self.weight + self.eps).clamp(max=0).t()

        lower = x_lower @ w_lower_pos + x_upper @ w_lower_neg + self.bias
        upper = x_upper @ w_upper_pos + x_lower @ w_upper_neg + self.bias

        return torch.cat((middle, lower, upper), dim=1)


class Conv2dInterval(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=0, input_layer=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.eps = eps
        self.input_layer = input_layer

    def forward(self, x):
        if self.input_layer:
            x = torch.cat((x, x, x), dim=1)

        s = x.size(1) // 3
        x_middle = x[:, :s]
        x_lower = x[:, s:2*s]
        x_upper = x[:, 2*s:]

        middle = super().forward(x_middle)

        w_lower_pos = (self.weight - self.eps).clamp(min=0)
        w_lower_neg = (self.weight - self.eps).clamp(max=0)
        w_upper_pos = (self.weight + self.eps).clamp(min=0)
        w_upper_neg = (self.weight + self.eps).clamp(max=0)

        lower = (f.conv2d(x_lower, w_lower_pos, None, self.stride,
                          self.padding, self.dilation, self.groups) +
                 f.conv2d(x_upper, w_lower_neg, None, self.stride,
                          self.padding, self.dilation, self.groups) +
                 self.bias[None, :, None, None])

        upper = (f.conv2d(x_upper, w_upper_pos, None, self.stride,
                          self.padding, self.dilation, self.groups) +
                 f.conv2d(x_lower, w_upper_neg, None, self.stride,
                          self.padding, self.dilation, self.groups) +
                 self.bias[None, :, None, None])

        return torch.cat((middle, lower, upper), dim=1)


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, 2)
        self.conv2 = nn.Conv2d(16, 32, 4, 1)
        self.fc1 = nn.Linear(32*10*10, 100)
        self.last_layer = nn.Linear(100, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.last_layer(x)
        return x


if __name__ == '__main__':

    t = nn.Sequential(
        Conv2dInterval(1, 16, 4, 2, eps=0, input_layer=True),
        Conv2dInterval(16, 32, 4, 1, eps=0)
    )
    x = torch.randn(20, 1, 28, 28)
    print(t(x).size())
