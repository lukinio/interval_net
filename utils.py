from functools import wraps
from datetime import datetime as dt

import torch
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = dt.now().replace(microsecond=0)
        result = f(*args, **kw)
        end = dt.now().replace(microsecond=0)
        print(f"function: {f.__name__} took: {end-start}")
        return result
    return wrap

def copy_model(Model, old):
    new_model = Model().to(device)
    for layer, old_layer in zip(new_model.children(), old.children()):
        print(layer)
        layer.weight.data = old_layer.weight.data.clone()
    return new_model

def clean_acc(model, loader):
    clean_correct = 0.
    model.eval()
    with torch.no_grad():
        for k, (X, y) in enumerate(loader, 1):
            start = dt.now().replace(microsecond=0)
            X, y = X.to(device), y.to(device)  
            clean_correct += (model(X).argmax(dim=1) == y).sum().item()
            end = dt.now().replace(microsecond=0)
            print(f"iter: {k}/{len(loader)} time: {end-start}", end="\r")
        print(" " * 50, end="\r")
        acc = round(clean_correct / len(loader.dataset), 4)
        print(f"clean accuracy: {acc}")
    return acc

def adv_acc(model, loader, loss_fn, attack, attack_params):
    model.eval()
    adv_correct = [0] * len(attack_params)
    for i, (name, params) in enumerate(attack_params):
        for k, (X, y) in enumerate(loader, 1):
            start = dt.now().replace(microsecond=0)
            X, y = X.to(device), y.to(device)
            noise = attack(model, X, y, loss_fn, **params)
            out = model(X+noise)
            adv_correct[i] += (out.argmax(dim=1) == y).sum().item()
            end = dt.now().replace(microsecond=0)
            print(f"iter: {k}/{len(loader)} time: {end-start}", end="\r")
        print(" " * 50, end="\r")
        print(f"name: {name} accuracy: {(adv_correct[i] / len(loader.dataset)):.4f}")
    return [round(a/len(loader.dataset), 4) for a in adv_correct]


def print_table(models, attacks, clean, adv):
    d = {'model': models, 'clean image': clean}
    for i, (name, p) in enumerate(attacks):
        d[name] = [adv[j][i] for j, _ in enumerate(adv)]

    return pd.DataFrame(data=d)
