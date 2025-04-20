import torch
import torch.nn.functional as F


def loss_func(output, mean, log_var, target: torch.Tensor, beta, z=None, lambda_l1=None, sl_weight=0):
    # start_token_value = 1
    # start_token = torch.full((target.size(0), 1), start_token_value, dtype=torch.long, device=device)
    # target = torch.cat((start_token, target), dim=1)

    # Reconstruction loss
    reconstruction_loss_func = torch.nn.MSELoss(reduction='mean')

    if type(output) is tuple:
        reconstruction_loss = sl_weight * (reconstruction_loss_func(output[0], target[0])
                                           + reconstruction_loss_func(output[1], target[1]))
    else:
        # output = output.squeeze()
        target = target.float()
        reconstruction_loss = reconstruction_loss_func(output, target)

    # KL divergence loss
    kl_loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    # L1 regularization on latent space
    if lambda_l1 is not None:
        l1 = lambda_l1 * torch.sum(torch.abs(z))
    else:
        l1 = 0

    # Beta-VAE: reweighting the reconstruction loss vs. the KL divergence loss, beta ~500. For small beta the network
    # will just focus on minimzing the KL divergence loss and therefore output latents close to zero.
    return reconstruction_loss, kl_loss

