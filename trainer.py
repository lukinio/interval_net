import os
from datetime import datetime as dt
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import timing


class Trainer:

    def __init__(self, model_name, train_dl, test_dl, eps_scheduler, kappa_scheduler,
                 num_classes=10):
        self.model_name = model_name
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.eps_scheduler = eps_scheduler
        self.kappa_scheduler = kappa_scheduler

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.checkpoint_path = "saved/{}/{}_{}_checkpoint.pt"
        os.makedirs(f"saved/{model_name}/", exist_ok=True)
        self.logs = {}

        self.C = [-torch.eye(num_classes).to(self.device) for _ in range(num_classes)]
        for y0 in range(num_classes):
            self.C[y0][y0, :] += 1

    def _interval_based_bound(self, model, y0, idx):
        # requires last layer to be linear
        cW = self.C[y0].t() @ (model.last_layer.weight - self.eps_scheduler.current)
        cb = self.C[y0].t() @ model.last_layer.bias
        l, u = model.bounds
        return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()

    @timing
    def _train_test_epoch(self, model, loader, loss_fn, optimizer=None):
        robust_err = 0
        total_loss, accuracy = 0., 0.
        for k, (X, y) in enumerate(loader, 1):
            X, y = X.to(self.device), y.to(self.device)
            out = model(X)
            fit_loss = self.kappa_scheduler.current * loss_fn(out, y)
            accuracy += (out.argmax(dim=1) == y).sum().item()

            robust_loss = 0
            for y0 in range(10):
                if (y == y0).sum().item() > 0:
                    lower_bound = self._interval_based_bound(model, y0, y == y0)
                    robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y == y0]) / X.shape[0]

                    # increment when true label is not winning
                    robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

            # combined losss
            combined_loss = self.kappa_scheduler.current * fit_loss + \
                            (1 - self.kappa_scheduler.current) * robust_loss

            if optimizer is not None:
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                self.eps_scheduler.step()
                self.kappa_scheduler.step()
                model.set_eps(self.eps_scheduler.current)

            total_loss += combined_loss.data.item() * X.size(0)

        return total_loss / len(loader.dataset), accuracy / len(loader.dataset), robust_err / len(loader.dataset)

    def _train(self, model, loss_fn, optimizer):
        model.train()
        with torch.enable_grad():
            return self._train_test_epoch(model, self.train_dl, loss_fn, optimizer)

    def _test(self, model, loss_fn):
        model.eval()
        with torch.no_grad():
            return self._train_test_epoch(model, self.test_dl, loss_fn)

    def train(self, model, loss_fn, optimizer, scheduler=None, epochs=30):
        self.logs = {
            'loss': {"train": [], "test": []},
            'accuracy': {"train": [], "test": []},
            'robust error': {"train": [], "test": []},
        }
        model = model.to(self.device)

        for e in range(1, epochs + 1):
            print(f"epoch start with: eps: {self.eps_scheduler.current:.8f}, "
                  f"kappa: {self.kappa_scheduler.current:.8f}")
            train_loss, train_acc, train_err = self._train(model, loss_fn, optimizer)
            test_loss, test_acc, test_err = self._test(model, loss_fn)
            followed_metric = test_loss
            if scheduler is not None:
                scheduler.step(followed_metric)

            out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, robust err: {:.4f}"
            print(out.format(e, test_loss, test_acc, test_err), end="\n\n")

            # Update logs
            self.logs['loss']["train"].append(train_loss)
            self.logs['loss']["test"].append(test_loss)
            self.logs['accuracy']["train"].append(train_acc)
            self.logs['accuracy']["test"].append(test_acc)
            self.logs['robust error']["train"].append(train_err)
            self.logs['robust error']["test"].append(test_err)

            # if e % 10 == 0:
            torch.save(model.state_dict(), self.checkpoint_path.format(self.model_name, self.model_name, str(e)))

        return self.logs


def plot_history(model_name, hists):
    x = np.arange(1, len(hists["loss"]["test"]) + 1)
    f, axes = plt.subplots(nrows=1, ncols=len(hists), figsize=(15, 5))
    for ax, (name, hist) in zip(axes, hists.items()):
        for label, h in hist.items():
            ax.plot(x, h, label=label)

        ax.set_title("Model " + name)
        ax.set_xlabel('epochs')
        ax.set_ylabel(name)
        ax.legend(loc="best")

    f.savefig(f"{model_name}.png", dpi=f.dpi)
    plt.show()
