from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model: nn.Module, init_optimizer: Callable[[Any], Optimizer], loss: Callable, epochs, batch_size,
          dataset: Dataset, val_set: Dataset, distance: int, b: float) -> nn.Module:
    """
    Trains a VAE model (based upon convolutions and linear layers) on syndrome measurement,
    :param model: VAE model
    :param init_optimizer: function to initialize the optimizer
    :param loss: loss function
    :param epochs: number of epochs
    :param batch_size: batch size
    :param dataset: train dataset
    :param val_set: validation dataset
    :return: trained model
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    print("Training will run on {0}".format(device))

    model = model.to(device)
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)
    optimizer = init_optimizer((model.parameters()))

    # Writes training summary to external file
    writer = SummaryWriter('logs/train')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_loss = float("inf")

    k = 0.09
    '''
    if "skip" in model.name:
        b = 11
    else:
        b = 9
    '''
    # b = 6.5
    counter = 0

    for e in range(epochs):
        # beta = (50 + np.exp(-k * e + b))
        beta = b * distance**2 * batch_size  # 0.01 slightly more latent space
        # d=25: 0.01 -> latent space, 0.05 -> reconstruction error
        avg_loss = 0
        num_batches = 0

        # Training
        model.train()
        with tqdm(train_loader, unit="batch") as epoch_pbar:
            for batch in epoch_pbar:
                # print(batch_idx)
                optimizer.zero_grad()
                if len(batch) >= 2:
                    output, mean, log_var = model.forward([batch[0].to(device), batch[1].to(device)])
                    batch_loss = loss(output, mean, log_var, [batch[0].to(device), batch[1].to(device)], beta=beta)
                else:
                    output, mean, log_var = model.forward(batch[0].to(device))
                    reconstruction_loss, kldiv_loss = loss(output, mean, log_var, batch[0].to(device), beta=beta)
                    batch_loss = beta * reconstruction_loss + kldiv_loss
                batch_loss.backward()
                avg_loss += batch_loss
                optimizer.step()
                num_batches += 1

                od = OrderedDict()
                od["MSE"] = reconstruction_loss.item()
                od["KLD"] = kldiv_loss.item()
                epoch_pbar.set_postfix(od)

        avg_loss /= num_batches
        writer.add_scalar('training loss', avg_loss, global_step=e)
        print(f'Epoch {e + 1}/{epochs}, Loss: {avg_loss:.4f}')

        # Validation
        avg_val_loss = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(val_loader):
                if len(batch) >= 2:
                    val_output, val_mean, val_log_var = model.forward([batch[0].to(device), batch[1].to(device)])
                    val_loss = loss(val_output, val_mean, val_log_var, [batch[0].to(device), batch[1].to(device)],
                                    beta=1)
                else:
                    val_output, val_mean, val_log_var = model.forward(batch[0].to(device))
                    val_loss, val_kldiv_loss = loss(val_output, val_mean, val_log_var, batch[0].to(device),
                                                    beta=1)
                avg_val_loss += val_loss
                num_batches += 1
            avg_val_loss /= num_batches
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            model.save()

            scheduler.step()
            print(scheduler.get_last_lr())
            if scheduler.get_last_lr()[0] < 2e-8:
                counter += 1
            if counter > 10:
                break

            writer.add_scalar('validation loss', avg_val_loss, global_step=e)
            print(f'Epoch {e + 1}/{epochs}, Validation loss: {avg_val_loss:.4f}')
    return model


def train_TraVAE(model: nn.Module, init_optimizer: Callable[[Any], Optimizer], loss_func: Callable, epochs, batch_size,
                 dataset: Dataset, val_set: Dataset) -> nn.Module:
    """
    Trains a transformer-based variational autoencoder on syndrome measurements.
    :param model: transformer-based VAE
    :param init_optimizer: function to initialize the optimizer
    :param loss_func: loss function
    :param epochs: number of epochs
    :param batch_size: batch size
    :param dataset: train dataset
    :param val_set: validation dataset
    :return: trained model
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    print("Training will run on {0}".format(device))

    model = model.to(device)
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)

    optimizer = init_optimizer((model.parameters()))
    writer = SummaryWriter('logs/train')
    val_loss_increase = 0
    previous_val_loss = float('inf')
    best_val_loss = float("inf")

    for e in range(epochs):
        k = 0.5
        b = 6.5
        beta = (1 + np.exp(-k * e + b))
        # beta = 100
        avg_loss = 0
        num_batches = 0
        model.train()
        for (batch_idx, batch) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output, mean, log_var, z = model.forward(batch.to(device))
            loss = loss_func(output, mean, log_var, z, batch.to(device), beta)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            num_batches += 1
        avg_loss /= (num_batches * batch_size)
        writer.add_scalar('training loss', avg_loss, global_step=e)
        print(f'Epoch {e + 1}/{epochs}, Loss: {avg_loss:.4f}')

        # Validation
        avg_val_loss = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for (batch_idx, batch) in enumerate(val_loader):
                val_output, val_mean, val_log_var, val_z = model.forward(batch.to(device))
                val_loss = loss_func(val_output, val_mean, val_log_var, val_z, batch.to(device), beta=500)
                avg_val_loss += val_loss
                num_batches += 1
            avg_val_loss /= (num_batches * batch_size)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save()
            if avg_val_loss >= previous_val_loss:
                val_loss_increase += 1
            else:
                val_loss_increase = 0
            previous_val_loss = avg_val_loss
            writer.add_scalar('validation loss', avg_val_loss, global_step=e)
            print(f'Epoch {e + 1}/{epochs}, Validation loss: {avg_val_loss:.4f}')
            if val_loss_increase > 4:
                break
    return model
