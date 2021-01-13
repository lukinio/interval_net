import re
import os
from datetime import datetime as dt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    checkpoint_path = "saved/{}_checkpoint.pt"

    def __init__(self, train_dataset, test_dataset, batch_size=(200, 200)):
        self.batch_size = batch_size
        self.train_dl = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
        self.test_dl = DataLoader(test_dataset, batch_size=batch_size[1], shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.logs = {}
        self.eps, self.kappa = 0, 1
        self.interval_train = False
        os.makedirs("saved", exist_ok=True)

    @staticmethod
    def _interval_based_bound(model, c, bounds, idx):
        # requires last layer to be linear
        cW = c.t() @ model.last_layer.weight
        cb = c.t() @ model.last_layer.bias
        l, u = bounds
        return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()

    def _train_test_epoch(self, model, loader, loss_fn, optimizer=None):
        C = [-torch.eye(10).to(self.device) for _ in range(10)]
        for y0 in range(10):
            C[y0][y0, :] += 1

        robust_err = 0
        total_loss, accuracy = 0., 0.
        for k, (X, y) in enumerate(loader, 1):
            start = dt.now().replace(microsecond=0)
            X, y = X.to(self.device), y.to(self.device)

            lower_out, upper_out = model(torch.cat([X - self.eps, X + self.eps], dim=1))
            fit_loss = self.kappa * loss_fn(lower_out, y)
            accuracy += (lower_out.argmax(dim=1) == y).sum().item()

            robust_loss = 0
            for y0 in range(10):
                if sum(y == y0) > 0:
                    lower_bound = self._interval_based_bound(model, C[y0], model.bounds, y == y0)
                    robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound, y[y == y0]) / X.shape[0]

                    # increment when true label is not winning
                    robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

            # combined losss
            combined_loss = self.kappa * fit_loss + (1 - self.kappa) * robust_loss

            if optimizer is not None:
                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

            total_loss += combined_loss.data.item() * X.size(0)
            end = dt.now().replace(microsecond=0)
            # print(f"{self.phase_name} iteration: {k}/{len(loader)} time: {end - start}", end="\r")

            self.eps += 0.00001
            if self.kappa > 0.5:
                self.kappa -= 0.0001
        # print(" " * 80, end="\r")

        return total_loss/len(loader.dataset), accuracy/len(loader.dataset), robust_err/len(loader.dataset)

    def _train(self, model, loss_fn, optimizer):
        model.train()
        self.phase_name = "train"
        with torch.enable_grad():
            return self._train_test_epoch(model, self.train_dl, loss_fn, optimizer)

    def _test(self, model, loss_fn):
        model.eval()
        self.phase_name = "test"
        with torch.no_grad():
            return self._train_test_epoch(model, self.test_dl, loss_fn)

    def train(self, model, loss_fn, optimizer, scheduler=None, epochs=30, patience=4):
        self.logs = {'loss': {"train": [], "test": []}, 'accuracy': {"train": [], "test": []}}
        model = model.to(self.device)
        epochs_no_improve, min_loss = np.inf, float('inf')
        model_name = re.sub(r'\W+', '', str(model.__class__).split(".")[-1])

        for e in range(1, epochs+1):
            start = dt.now().replace(microsecond=0)
            print(f"epoch usage eps: {self.eps}, kappa: {self.kappa}")
            train_loss, train_acc, train_err = self._train(model, loss_fn, optimizer)
            test_loss, test_acc, test_err = self._test(model, loss_fn)
            if scheduler is not None:
                scheduler.step(test_loss)
            end = dt.now().replace(microsecond=0)
            out = "Epoch: {} Validation Loss: {:.4f} accuracy: {:.4f}, robust err: {:.4f}, time: {}"
            print(out.format(e, train_loss, train_acc, train_err, end - start))
            print(out.format(e, test_loss, test_acc, test_err, end - start))

            # Update logs
            self.logs['loss']["train"].append(train_loss)
            self.logs['loss']["test"].append(test_loss)
            self.logs['accuracy']["train"].append(train_acc)
            self.logs['accuracy']["test"].append(test_acc)

            # Early stopping
            if test_loss < min_loss:
                min_loss = test_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), Trainer.checkpoint_path.format(model_name))
            else:
                epochs_no_improve += 1
                print(f"epochs_no_improve: {epochs_no_improve}/{patience}")
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break

        model.load_state_dict(torch.load(Trainer.checkpoint_path.format(model_name),
                                         map_location=self.device))
        return self.logs


def plot_history(hists):
    x = np.arange(1, len(hists["loss"]["test"]) + 1)
    f, axes = plt.subplots(nrows=1, ncols=len(hists), figsize=(15, 5))
    for ax, (name, hist) in zip(axes, hists.items()):
        for label, h in hist.items():
            ax.plot(x, h, label=label)

        ax.set_title("Model " + name)
        ax.set_xlabel('epochs')
        ax.set_ylabel(name)
        ax.legend(loc="best")

    f.savefig("model.png", dpi=f.dpi)
