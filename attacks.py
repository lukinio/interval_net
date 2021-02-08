import torch


def fgsm(model, X, y, loss_fn, epsilon=0.1):
    noise = torch.zeros_like(X, requires_grad=True)
    loss = loss_fn(model(X + noise), y)
    loss.backward()
    return epsilon * noise.grad.detach().sign()


def pgd(model, X, y, loss_fn, epsilon=0.1, alpha=0.01, num_iter=20):
    delta = torch.zeros_like(X, requires_grad=True)
    for _ in range(num_iter):
        loss = loss_fn(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    return delta.detach()
