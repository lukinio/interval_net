# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import optim

from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

from trainer import Trainer, plot_history
from model import SmallIntervalNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':

    tr_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    vl_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    ds_train = MNIST('../data', train=True, download=True, transform=tr_transform)
    ds_test = MNIST('../data', train=False, download=True, transform=vl_transform)

    trainer = Trainer(ds_train, ds_test, batch_size=(200, 200))
    model = SmallIntervalNet()
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-5,
                                                     patience=2, verbose=True)

    history = trainer.train(model, loss_fn, optimizer, scheduler, epochs=80, patience=6)
    plot_history(history)
    torch.save(model.state_dict(), "saved/test_interval_net.pt")
