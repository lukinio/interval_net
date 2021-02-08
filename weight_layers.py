import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter
from math import sqrt
from copy import deepcopy


# INTERWAŁY NA WAGACH
class LinearInterval(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=0):
        super().__init__(in_features, out_features, bias)
        self.eps = eps

    def forward(self, x):
        # x na wejściu to tensor zawierający prawy i lewy koniec
        x_lower = x[:, :x.size(1) // 2]
        x_upper = x[:, x.size(1) // 2:]

        # wyznaczam dodatnie/ujemne wagi dla lewego końca
        w_lower_pos = (self.weight - self.eps).clamp(min=0).t()  # dodatnie
        w_lower_neg = (self.weight - self.eps).clamp(max=0).t()  # ujemne

        # wyznaczam dodatnie/ujemne wagi dla prawego końca
        w_upper_pos = (self.weight + self.eps).clamp(min=0).t()  # dodatnie
        w_upper_neg = (self.weight + self.eps).clamp(max=0).t()  # ujemne

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

