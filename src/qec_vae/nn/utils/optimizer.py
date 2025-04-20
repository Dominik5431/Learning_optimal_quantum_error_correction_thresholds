import torch


def make_optimizer(lr):
    return lambda params: torch.optim.Adam(params, lr=lr, weight_decay=1e-5)