import torch
import torch.nn as nn
from pathlib import Path


class Net(nn.Module):
    """
    Base class for all used machine learning models.
    Implements custom save() and load() function.
    """
    def __init__(self, name: str, cluster: bool = False):
        super(Net, self).__init__()
        self.name = name
        self.cluster = cluster

    def save(self):
        torch.save(self.state_dict(), "data/net_{0}.pt".format(self.name))

    def load(self):
        self.load_state_dict(torch.load("data/net_{0}.pt".format(self.name)))

