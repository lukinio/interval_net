import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter
from math import sqrt
from copy import deepcopy


# class LinearInterval(nn.Module):
#     def __init__(self, in_features, out_features, bias=True, eps=0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.eps = eps
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         self.lower_weight = Parameter(torch.Tensor(out_features, in_features))
#         self.upper_weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
#         nn.init.kaiming_uniform_(self.lower_weight, a=sqrt(5))
#         nn.init.kaiming_uniform_(self.upper_weight, a=sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lower_weight)
#             bound = 1 / sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, x):
#         x_lower = x[:, :x.size(1) // 2]
#         x_upper = x[:, x.size(1) // 2:]
#
#         l_weight = self.lower_weight.t()
#         u_weight = self.upper_weight.t()
#         lower_pos, lower_neg = l_weight.clamp(min=0), l_weight.clamp(max=0)
#         upper_pos, upper_neg = u_weight.clamp(min=0), u_weight.clamp(max=0)
#
#         lower = x_lower @ lower_pos + x_upper @ lower_neg + self.bias
#         upper = x_upper @ upper_pos + x_lower @ upper_neg + self.bias
#
#         return torch.cat([lower, upper], dim=1)
#
#
#         self.lower_weight = (self.weight.clamp(min=0) @ x +
#                              self.weight.clamp(max=0) @ x +
#                              self.bias[:, None]).t()
#         self.upper_weight = (self.weight.clamp(min=0) @ self.upper_weight.t() +
#                              self.weight.clamp(max=0) @ self.lower_weight.t() +
#                              self.bias[:, None]).t()
#
#         # x_lower = x[:, :x.size(1) // 2]
#         # x_upper = x[:, x.size(1) // 2:]
#         # lower = f.linear(x_lower, self.lower_weight.data, self.bias)
#         # upper = f.linear(x_upper, self.upper_weight, self.bias)
#         # return torch.cat([lower, upper], dim=1)
#
#         x_lower = x[:, :x.size(1) // 2]
#         x_upper = x[:, x.size(1) // 2:]
#         center = (x_upper + x_lower) / 2
#         radius = (x_upper - x_lower) / 2
#         new_center = f.linear(center, self.weight, self.bias)
#         new_radius = f.linear(radius, self.weight.abs(), None)
#         return torch.cat([new_center - new_radius, new_center + new_radius], dim=1)
#


class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        # x na wejściu to tensor postaci torch.cat([x-eps, x+eps], dim=1)
        x_lower = x[:, :x.size(1) // 2]
        x_upper = x[:, x.size(1) // 2:]

        # następnie wyznaczam środek i promień
        center = (x_upper + x_lower) / 2
        radius = (x_upper - x_lower) / 2

        # propaguje środek i progmien zgodnie ze wzorami
        new_center = f.linear(center, self.weight, self.bias)
        new_radius = f.linear(radius, self.weight.abs(), None)

        # zwracam nowy tensor, który zawiera lewy i prawy interwał
        # jako środek +/- promień
        return torch.cat([new_center - new_radius, new_center + new_radius], dim=1)


class LinearIntervalLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lower_weight = Parameter(torch.Tensor(out_features, in_features))
        self.upper_weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lower_weight, a=sqrt(5))
        self.upper_weight = deepcopy(self.lower_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lower_weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x na wejściu to tensor postaci torch.cat([x-eps, x+eps], dim=1)
        x_lower = x[:, :x.size(1) // 2]
        x_upper = x[:, x.size(1) // 2:]

        # wyznaczam dodatnie/ujemne wagi dla lewego końca
        w_lower_pos = self.lower_weight.clamp(min=0).t()  # dodatnie
        w_lower_neg = self.lower_weight.clamp(max=0).t()  # ujemne

        # wyznaczam dodatnie/ujemne wagi dla prawego końca
        w_upper_pos = self.upper_weight.clamp(min=0).t()  # dodatnie
        w_upper_neg = self.upper_weight.clamp(max=0).t()  # ujemne

        # wyliczam lewy koniec jako sumę 3 składników:
        # @ - mnożenie macierzowe
        # 1. lewy koniec @ dodatnie wagi z lewego końca
        # 2. prawy koniec @ ujemne wagi z prawego końca
        # 3. bias
        lower = x_lower @ w_lower_pos + x_upper @ w_upper_neg + self.bias
        # analogicznie prawy koniec liczę jako sumę 3 składników:
        # @ - mnożenie macierzowe
        # 1. prawy koniec @ dodatnie wagi z prawego końca
        # 2. lewy koniec @ ujemne wagi z lewego końca
        # 3. bias
        upper = x_upper @ w_upper_pos + x_lower @ w_lower_neg + self.bias

        # zwracam jako jednen tensor
        return torch.cat([lower, upper], dim=1)



class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(2*in_features, 2*out_features, bias)

    def forward(self, x):
        x_lower = x[:, :x.size(1) // 2]
        x_upper = x[:, x.size(1) // 2:]
        center = (x_upper + x_lower) / 2
        radius = (x_upper - x_lower) / 2

        w_lower, w_upper = self.weight.data
        w_center = (w_upper + w_lower) / 2
        w_radius = (w_upper - w_lower) / 2

        new_center = f.linear(center, self.weight, self.bias)
        new_radius = f.linear(radius, self.weight.abs(), None)
        return torch.cat([new_center - new_radius, new_center + new_radius], dim=1)

# class LinearInterval(nn.Linear):
#
#     def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
#         super().__init__(in_features, out_features, bias)
#
#     def bounds(self, x, eps):
#         x = f.linear(x, self.weight, self.bias)
#         w = self.weight.abs().sum(dim=1)
#         lower, upper = x - eps*w, x + eps*w
#         return lower.clamp(max=0), upper.clamp(min=0)

# class LinearInterval(nn.Linear):
#
#     def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
#         super().__init__(in_features, out_features, bias)
#         self.lower, self.upper = self.weight, self.weight
#
#     def perturb(self, eps):
#         w = self.weight.abs()
#         self.lower = (self.weight - w * eps).clamp(min=0)
#         self.upper = (self.weight + w * eps).clamp(max=0)
#
#     def bounds(self):
#         return self.lower, self.upper


class Conv2dInterval(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(2*in_channels, 2*out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

    def forward(self, x):
        return f.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
