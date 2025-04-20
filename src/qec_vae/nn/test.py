import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .data.qecdata import QECDataset
from .net.net import Net
from .net.vae import VariationalAutoencoder
from typing import Type


def test_model_reconstruction_error(model: nn.Module, dataset: QECDataset,
                                    loss_func: torch.nn.MSELoss()):
    """
    Evaluates the model reconstruction error on the test set.
    :param model: VAE model, type nn.Module
    :param dataset: test set
    :param loss_func: MSE loss function
    :return: tuple, mean and variance of reconstruction error
    """
    # no KL divergence loss when evaluating reconstruction error!

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():  # not necessary, but improves performance since no gradients are computed
        loss = 0
        for (batch_idx, batch) in enumerate(test_loader):
            if len(batch) == 3:
                output, mean, log_var = model.forward([batch[0].to(device), batch[1].to(device)])
                flips = batch[2]
            else:
                output, mean, log_var = model.forward(batch[0].to(device))
                flips = batch[1]
            loss += loss_func(output, batch[0].to(
                device))  # default reduction='mean' for MSELoss --> reduction mean: takes average over whole batch
            loss = torch.mean(loss, dim=[1, 2, 3])
    return loss, flips


def test_model_latent_space(model: VariationalAutoencoder, dataset: Type[QECDataset]):
    """
    Evaluates the latent space on the test set.
    :param model: VAE model, type VariationalAutoencoder
    :param dataset: test set for a single noise strength
    :return: latent space mean, variance, variable and information about if the syndrome was flipped
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    flips = None
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            if len(batch) == 3:
                z_mean, z_log_var, z, *_ = model.encoder.forward([batch[0].to(device), batch[1].to(device)])
                flips = batch[2]
            elif len(batch) == 2:
                z_mean, z_log_var, z, *_ = model.encoder.forward(batch[0].to(device))
                flips = batch[1]
            else:
                z_mean, z_log_var, z, *_ = model.encoder.forward(batch[0].to(device))
    return z_mean, z_log_var, z, flips


def test_latent_space_TraVAE(model, test_set, device):
    """
    Evaluates the latent space on the test set for a model based on Transformers.
    :param model: TraVAE model
    :param test_set: test dataset
    :param device: torch.device
    :return: latent space mean
    """
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    mean = torch.tensor([]).to(device)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            mean = torch.cat((mean, model.partial_forward(batch.to(device))), dim=0)
    return torch.as_tensor(mean, device=device)


def test_model_predictions(model: Type[Net], dataset: Type[QECDataset]):
    """
    Evaluates the encoder predictions into the latent space on the test set.
    :param model: VAE model
    :param dataset: samples to encode into the latent space
    :return:
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for (batch_idx, batch) in enumerate(test_loader):
            z = model.forward(batch[0])
    return z
