import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter
from math import sqrt
from copy import deepcopy


# INTERWAŁY NA WAGACH
class LinearInterval(nn.Module):
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

    def perturb(self, eps):
        self.lower_weight = self.lower_weight - eps
        self.upper_weight = self.upper_weight + eps

    def forward(self, x):
        # x na wejściu to tensor zawierający prawy i lewy koniec
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
        # 2. prawy koniec @ ujemne wagi z lewego końca
        # 3. bias
        lower = x_lower @ w_lower_pos + x_upper @ w_lower_neg + self.bias
        # analogicznie prawy koniec liczę jako sumę 3 składników:
        # @ - mnożenie macierzowe
        # 1. prawy koniec @ dodatnie wagi z prawego końca
        # 2. lewy koniec @ ujemne wagi z prawego końca
        # 3. bias
        upper = x_upper @ w_upper_pos + x_lower @ w_upper_neg + self.bias

        # zwracam jako jednen tensor
        return torch.cat([lower, upper], dim=1)
