from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warnings

from ..data.results_wrapper import ResultsWrapper
from .functions import smooth, simple_bootstrap
import torch

"""
This file contains various methods to plot the results and evaluating the obtained data during inference.
"""


def final_plot(latents_bitflip, latents_depolarizing, recon_bitflip, recon_depolarizing):
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    fig, ax = plt.subplots(4, 2, figsize=(12, 18))  # figsize=(8.27, 11.69))

    dists = list(latents_bitflip.keys())
    for k, dist in enumerate(dists):
        noises = list(latents_bitflip[dist].keys())
        latent = list(latents_bitflip[dist].values())
        zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))

        flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(),
                                  latent)))  # gets the information whether a sample has been flipped

        # convert noise strength to temperature scale
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        der = np.zeros(len(noises))  # susceptibility
        means = np.zeros(len(noises))
        unc = []
        for i in range(1, len(noises)):
            vals = zs[i][np.where(flips[i] == -1)]
            error_bars = simple_bootstrap(vals, np.mean, r=100)
            means[i] = error_bars[0]
            unc.append((error_bars[1], error_bars[2]))
            der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
        der = smooth(der, 9)
        unc = list(map(list, zip(*unc)))
        print(unc)

        ax[0, 0].errorbar(noises[1:], means[1:] / np.max(means[1:]), yerr=unc, color=coloring[k], marker='o',
                          markersize=1,
                          linestyle='solid', label='d={}'.format(dist))
        ax[1, 0].plot(noises[1:], der[1:], color=coloring[k], marker='o', markersize=1, linestyle='solid',
                      label='d={}'.format(dist))

        ax[0, 0].vlines(0.951, 0, 1, colors='red', linestyles='dashed')
        ax[1, 0].vlines(0.951, 0, 0.01, colors='red', linestyles='dashed')

        # plot threshold lines

    # one more time for label --> label only during last plotting

    ax[0, 0].vlines(0.951, 0, 1, colors='red', linestyles='dashed', label='threshold')
    ax[1, 0].vlines(0.951, 0, 0.01, colors='red', linestyles='dashed', label='threshold')

    ax[0, 0].set_ylabel(r'$\mu_\mathrm{op}$; single branch')
    ax[1, 0].set_ylabel(r'susceptibility $\chi(\mu_\mathrm{op})$')

    # For depolarizing
    dists = list(latents_depolarizing.keys())
    for k, dist in enumerate(dists):
        noises = list(latents_depolarizing[dist].keys())
        latent = list(latents_depolarizing[dist].values())
        zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))

        flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(),
                                  latent)))  # gets the information whether a sample has been flipped

        # convert noise strength to temperature scale
        noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

        der = np.zeros(len(noises))  # susceptibility
        means = np.zeros(len(noises))
        unc = []
        for i in range(1, len(noises)):
            vals = zs[i][np.where(flips[i] == -1)]
            error_bars = simple_bootstrap(vals, np.mean, r=100)
            means[i] = error_bars[0]
            unc.append((error_bars[1], error_bars[2]))
            der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
        der = smooth(der, 9)
        unc = list(map(list, zip(*unc)))
        print(unc)

        ax[0, 1].errorbar(noises[1:], means[1:] / np.max(means), yerr=unc, color=coloring[k], marker='o', markersize=1,
                          linestyle='solid', label='d={}'.format(dist))
        ax[1, 1].plot(noises[1:], der[1:], color=coloring[k], marker='o', markersize=1, linestyle='solid',
                      label='d={}'.format(dist))

        ax[0, 1].vlines(1.565, 0, 1, colors='red', linestyles='dashed')
        # ax[0, 1].vlines(1.373, 0, 1, colors='grey', linestyles='dashed')
        ax[1, 1].vlines(1.565, 0, 0.013, colors='red', linestyles='dashed')
        ax[1, 1].vlines(1.373, 0, 0.013, colors='grey', linestyles='dashed')

        # plot threshold lines

    # one more time for label --> label only during last plotting

    ax[0, 1].vlines(1.565, 0, 1, colors='red', linestyles='dashed', label='threshold')
    ax[1, 1].vlines(1.373, 0, 0.01, colors='grey', linestyles='dashed',
                    label=r'$\frac{3}{2} \cdot$ bit-flip threshold')
    ax[1, 1].vlines(1.565, 0, 0.013, colors='red', linestyles='dashed', label='threshold')
    # ax[0, 1].vlines(1.373, 0, 1, colors='grey', linestyles='dashed',
    #                label=r'$\frac{3}{2} \cdot$ bit-flip threshold')

    ax[0, 1].set_ylabel(r'$\mu_\mathrm{op}$; single branch')
    ax[1, 1].set_ylabel(r'susceptibility $\chi(\mu_\mathrm{op})$')

    # RECONSTRUCTION ERROR

    dists = list(recon_bitflip.keys())
    for k, dist in enumerate(dists):
        noises = list(recon_bitflip[dist].keys())
        recon = list(recon_bitflip[dist].values())
        zs = np.array(list(map(lambda x: x[0].cpu().detach().numpy(), recon)))

        flips = np.array(list(map(lambda x: x[1].cpu().detach().numpy(),
                                  recon)))  # gets the information whether a sample has been flipped

        # convert noise strength to temperature scale
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))

        der = np.zeros(len(noises))  # susceptibility
        means = np.zeros(len(noises))
        unc = []
        for i in range(1, len(noises)):
            vals = zs[i][np.where(flips[i] == -1)]
            error_bars = simple_bootstrap(vals, np.mean, r=100)
            means[i] = error_bars[0]
            unc.append((error_bars[1], error_bars[2]))
            der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
        der = smooth(der, 9)
        unc = list(map(list, zip(*unc)))
        print(unc)

        ax[2, 0].errorbar(noises[1:], means[1:], yerr=unc, color=coloring[k], marker='o', markersize=1,
                          linestyle='solid', label='d={}'.format(dist))
        ax[3, 0].plot(noises[1:], der[1:], color=coloring[k], marker='o', markersize=1, linestyle='solid',
                      label='d={}'.format(dist))

        ax[2, 0].vlines(0.951, 0, 1, colors='red', linestyles='dashed')
        ax[3, 0].vlines(0.951, 0, 0.008, colors='red', linestyles='dashed')

        # plot threshold lines

    # one more time for label --> label only during last plotting

    ax[2, 0].vlines(0.951, 0, 1, colors='red', linestyles='dashed', label='threshold')
    ax[3, 0].vlines(0.951, 0, 0.008, colors='red', linestyles='dashed', label='threshold')

    ax[2, 0].set_ylabel(r'MSE')
    ax[3, 0].set_ylabel(r'Var(MSE)')

    # For depolarizing
    dists = list(recon_depolarizing.keys())
    for k, dist in enumerate(dists):
        noises = list(recon_depolarizing[dist].keys())
        recon = list(recon_depolarizing[dist].values())
        zs = np.array(list(map(lambda x: x[0].cpu().detach().numpy(), recon)))

        flips = np.array(list(map(lambda x: x[1].cpu().detach().numpy(),
                                  recon)))  # gets the information whether a sample has been flipped

        # convert noise strength to temperature scale
        noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

        der = np.zeros(len(noises))  # susceptibility
        means = np.zeros(len(noises))
        unc = []
        for i in range(1, len(noises)):
            vals = zs[i][np.where(flips[i] == -1)]
            error_bars = simple_bootstrap(vals, np.mean, r=100)
            means[i] = error_bars[0]
            unc.append((error_bars[1], error_bars[2]))
            der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
        der = smooth(der, 9)
        unc = list(map(list, zip(*unc)))
        print(unc)

        ax[2, 1].errorbar(noises[1:], means[1:], yerr=unc, color=coloring[k], marker='o', markersize=1,
                          linestyle='solid', label='d={}'.format(dist))
        ax[3, 1].plot(noises[1:], der[1:], color=coloring[k], marker='o', markersize=1, linestyle='solid',
                      label='d={}'.format(dist))

        ax[2, 1].vlines(1.565, 0, 1, colors='red', linestyles='dashed')
        # ax[2, 1].vlines(1.373, 0, 1, colors='grey', linestyles='dashed')
        ax[3, 1].vlines(1.565, 0, 0.006, colors='red', linestyles='dashed')
        # ax[3, 1].vlines(1.373, 0, 0.01, colors='grey', linestyles='dashed')

        # plot threshold lines

    # one more time for label --> label only during last plotting

    ax[2, 1].vlines(1.565, 0, 1, colors='red', linestyles='dashed', label='threshold')
    # ax[3, 1].vlines(1.373, 0, 0.01, colors='grey', linestyles='dashed')
    ax[3, 1].vlines(1.565, 0, 0.006, colors='red', linestyles='dashed')
    # ax[2, 1].vlines(1.373, 0, 1, colors='grey', linestyles='dashed',
    #                 label=r'$\frac{3}{2} \cdot$ bit-flip threshold')

    ax[2, 1].set_ylabel(r'MSE')
    ax[3, 1].set_ylabel(r'Var(MSE)')

    ax[0, 0].set_title('bit-flip', fontsize=16)
    ax[0, 1].set_title('depolarizing', fontsize=16)

    # Add labels in the upper left corner
    labels = ['(a)', '(e)', '(b)', '(f)', '(c)', '(g)', '(d)', '(h)']
    for i, a in enumerate(ax.flatten()):
        a.text(-0.2, 0.98, labels[i], transform=a.transAxes, fontsize=14,
               verticalalignment='top', horizontalalignment='left')
        if labels[i] == '(d)' or labels[i] == '(h)':
            a.set_xlabel(r'associated temperature $T_\mathrm{NL}$')
        if i == 5:
            a.legend(loc='upper left')
        else:
            a.legend()
    plt.tight_layout()
    plt.savefig(f'plots/VAE_results.svg')
    plt.show()


def final_plot2(latents_bitflip, latents_depolarizing, dist):
    dists = list(latents_bitflip.keys())
    assert dist in dists
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    noises = list(latents_bitflip[dist].keys())
    latent = list(latents_bitflip[dist].values())
    zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))
    mean_raw = np.array(list(map(lambda x: torch.mean(x[-2]).cpu(), latent)))
    sus = np.array(list(map(lambda x: np.mean(x[-1]), latent)))
    sus = smooth(sus, 7)
    flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(),
                              latent)))  # gets the information whether a sample has been flipped

    # convert noise strength to temperature scale
    noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))

    der = np.zeros(len(noises))  # susceptibility
    means = np.zeros(len(noises))
    unc = []
    for i in range(1, len(noises)):
        vals = zs[i][np.where(flips[i] == -1)]
        error_bars = simple_bootstrap(vals, np.mean, r=100)
        means[i] = error_bars[0]
        unc.append((error_bars[1], error_bars[2]))
        der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
    der = smooth(der, 7)
    unc = list(map(list, zip(*unc)))
    print(unc)

    axs[0, 0].errorbar(noises[1:], means[1:] / np.max(means[1:]), yerr=unc, color=coloring[0], marker='o', markersize=1,
                       linestyle='solid', label='d={}'.format(dist))
    axs[0, 0].plot(noises[1:], mean_raw[1:], color='black', linestyle='dashed', label=r"M=$\frac{1}{N} \sum_i s_i$")

    axs[1, 0].plot(noises[1:], der[1:] / (np.max(means[1:]) ** 2), color=coloring[0], marker='o', markersize=1,
                   linestyle='solid',
                   label='d={}'.format(dist))
    axs[1, 0].plot(noises[1:], sus[1:], color='black', linestyle='dashed', label=r'$\chi(M)$')

    axs[0, 0].vlines(0.951, 0, 1, colors='red', linestyles='dashed', label='threshold')
    axs[1, 0].vlines(0.951, 0, 0.006, colors='red', linestyles='dashed', label='threshold')

    axs[1, 0].set_xlabel(r'associated temperature $T_\mathrm{NL}$')
    axs[0, 0].set_ylabel(r'$\mu_\mathrm{op}$; single branch')
    axs[1, 0].set_ylabel(r'susceptibility $\chi(\mu_\mathrm{op})$')
    plt.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles[::-1], labels[::-1])
    axs[1, 0].legend()

    # DEPOLARIZING

    noises = list(latents_depolarizing[dist].keys())
    latent = list(latents_depolarizing[dist].values())
    zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))
    mean_raw = np.array(list(map(lambda x: torch.mean(x[-2]).cpu(), latent)))
    sus = np.array(list(map(lambda x: np.mean(x[-1]), latent)))
    sus = smooth(sus, 7)
    flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(),
                              latent)))  # gets the information whether a sample has been flipped

    # convert noise strength to temperature scale
    noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

    der = np.zeros(len(noises))  # susceptibility
    means = np.zeros(len(noises))
    unc = []
    for i in range(1, len(noises)):
        vals = zs[i][np.where(flips[i] == -1)]
        error_bars = simple_bootstrap(vals, np.mean, r=100)
        means[i] = error_bars[0]
        unc.append((error_bars[1], error_bars[2]))
        der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
    der = smooth(der, 7)
    unc = list(map(list, zip(*unc)))
    print(unc)

    axs[0, 1].errorbar(noises[1:], means[1:] / np.max(means[1:]), yerr=unc, color=coloring[0], marker='o', markersize=1,
                       linestyle='solid', label='d={}'.format(dist))
    axs[0, 1].plot(noises[1:], mean_raw[1:], color='black', linestyle='dashed', label=r"M=$\frac{1}{N} \sum_i s_i$")

    axs[1, 1].plot(noises[1:], der[1:] / (np.max(means[1:]) ** 2), color=coloring[0], marker='o', markersize=1,
                   linestyle='solid',
                   label='d={}'.format(dist))
    axs[1, 1].plot(noises[1:], sus[1:], color='black', linestyle='dashed', label=r'$\chi(M)$')

    axs[0, 1].vlines(1.565, 0, 1, colors='red', linestyles='dashed', label='threshold')
    axs[1, 1].vlines(1.373, 0, 0.005, colors='grey', linestyles='dashed',
                     label=r'$\frac{3}{2} \cdot$ bit-flip threshold')
    axs[1, 1].vlines(1.565, 0, 0.005, colors='red', linestyles='dashed', label='threshold')
    # axs[0, 1].vlines(1.373, 0, 1, colors='grey', linestyles='dashed', label=r'$\frac{3}{2} \cdot$ bit-flip threshold')

    axs[1, 1].set_xlabel(r'associated temperature $T_\mathrm{NL}$')
    axs[0, 1].set_ylabel(r'$\mu_\mathrm{op}$; single branch')
    axs[1, 1].set_ylabel(r'susceptibility $\chi(\mu_\mathrm{op})$')
    plt.tight_layout()
    handles, labels = axs[0, 1].get_legend_handles_labels()
    axs[0, 1].legend(handles[::-1], labels[::-1])
    axs[1, 1].legend()

    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, a in enumerate(axs.flatten()):
        a.text(-0.3, 0.98, labels[i], transform=a.transAxes, fontsize=16,
               verticalalignment='top', horizontalalignment='left')
    plt.tight_layout()

    # ax2.legend()
    plt.savefig(f'plots/mean_op.svg')
    plt.show()


def plot_latent_mean(latents: dict, random_flip: bool, structure: str):
    """
    Plots the mean of the latent distribution
    :param latents: dictionary: key: distance, value: dictionary: key: noise, value: latent space mean
    :param random_flip: bool, whether random_flip is activated or not
    :param structure: structure of the VAE network
    :return:
    """
    warnings.warn("Only implemented for bit-flip noise.")

    # Plot latent space mean
    dists = list(latents.keys())
    for dist in dists:
        noises = list(latents[dist].keys())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))  # convert noises to temperature scale
        latent = list(latents[dist].values())  # latent mean values

        zs = np.array(list(map(lambda x: x[0].cpu().detach().numpy(), latent)))[:, :, 0]
        flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(), latent)))

        if not random_flip:
            means = list(map(lambda x: torch.mean(x[2]).cpu().detach().numpy(), latent))
        else:
            means = list(map(lambda x: torch.mean(torch.abs(x[0])).cpu().detach().numpy(), latent))
            # means = smooth(means, 5)
            # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
        plt.plot(noises, means, label=str(dist))
    plt.xlabel('associated temperature')
    plt.ylabel(r'mean $\langle | \mu | \rangle$')
    # plt.xscale('log')
    if random_flip:
        plt.title(structure + " + random flip")
    else:
        plt.title(structure)
    plt.legend()
    plt.show()

    # plot latent space log variance
    for dist in dists:
        noises = list(latents[dist].keys())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        latent = list(latents[dist].values())
        sigmas = list(map(lambda x: torch.mean(x[1]).cpu().detach().numpy(), latent))
        plt.plot(noises, sigmas, label=str(dist))
    plt.xlabel('associated temperature')
    plt.ylabel(r'mean $\langle \log \sigma^2 \rangle$')
    plt.legend()
    if random_flip:
        plt.title(structure + " + random flip")
    else:
        plt.title(structure)
    plt.show()


def plot_latent_susceptibility(latents: dict, random_flip: bool, structure: str, noise_model: str, show=True,
                               surface: bool = False):
    """
    Plots the susceptibility of the latent space order parameter.
    :param latents: dictionary: key: distance, value: dictionary: key: noise, value: latent space mean
    :param random_flip: bool, whether random_flip is activated or not
    :param structure: structure of the VAE network
    :param noise_model: specifies the noise model, supported: 'BitFlip' or 'Depolarizing'
    :param show: bool, show plot?
    :param surface: bool, specifies whether sequential data is used
    :return:
    """

    dists = list(latents.keys())
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    # ax2 = ax1.twinx()

    if surface:
        structure = 'TraVAE'
        for k, dist in enumerate(dists):
            noises = list(latents[dist].keys())
            print(noises)
            latent = list(latents[dist].values())
            print('here', latent[0].size())
            zs = np.array(list(map(lambda x: torch.sum(torch.abs(x), dim=1).cpu().detach().numpy(), latent)))

            noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

            means = np.mean(zs, axis=1)
            var = np.var(zs, axis=1, ddof=1)

            # var = smooth(var, 5)

            ax1.plot(noises[1:], means[1:], color=coloring[k], linestyle='dashed')
            ax2.plot(noises[1:], var[1:], color=coloring[k], label=str(dist))
            # plt.vlines(0.189, 0, max(var), colors='red', linestyles='dashed')
            plt.vlines(1.56, 0, max(var), colors='red', linestyles='dashed')  # Vertical line at the threshold
    else:
        for k, dist in enumerate(dists):
            # if dist == 7 or dist == 9 or dist == 39:
            #     continue
            # if dist > 11:
            #     continue
            try:
                noises = list(latents[dist].keys())
                latent = list(latents[dist].values())
                if random_flip:
                    zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))
                else:
                    zs = -1 * np.array(list(map(lambda x: torch.sum(x[0], dim=1).cpu().detach().numpy(), latent)))
                mean_raw = np.array(list(map(lambda x: torch.mean(x[-2]).cpu(), latent)))
                sus = np.array(list(map(lambda x: np.mean(x[-1]), latent)))

            except KeyError:  # some latent dicts might be saved in another way: This except block handles those.
                noises = list(latents[dist][dist].keys())
                latent = list(latents[dist][dist].values())
                zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))
            try:
                flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(),
                                          latent)))  # gets the information whether a sample has been flipped
            except AttributeError:
                flips = -1 * np.ones_like(zs)

            # convert noise strength to temperature scale
            if noise_model == 'BitFlip':
                noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
            else:
                noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

            der = np.zeros(len(noises))  # susceptibility
            means = np.zeros(len(noises))
            unc = []
            for i in range(1, len(noises)):
                vals = zs[i][np.where(flips[i] == -1)]
                error_bars = simple_bootstrap(vals, np.mean, r=100)
                means[i] = error_bars[0]
                unc.append((error_bars[1], error_bars[2]))
                der[i] = np.mean(vals ** 2) - np.mean(vals) ** 2
            der = smooth(der, 7)
            sus = smooth(sus, 7)

            scaling = max(means) - min(means)

            unc = list(map(list, zip(*unc)))
            print(unc)

            ax1.errorbar(noises[1:], means[1:], yerr=unc, color=coloring[k], marker='o', markersize=1,
                         linestyle='solid', label='d={}'.format(dist))
            ax1.plot(noises[1:], mean_raw[1:], color='black', linestyle='dashed')

            ax2.plot(noises[1:], der[1:] / scaling ** 2, color=coloring[k], marker='o', markersize=1, linestyle='solid',
                     label='d={}'.format(dist))
            ax2.plot(noises[1:], sus[1:], color='black', linestyle='dashed')

            # plot threshold lines
            if noise_model == 'BitFlip':
                ax1.vlines(0.951, 0, 1.1 * max(means), colors='red', linestyles='dashed')
                ax2.vlines(0.951, 0, 1.1 * max(der), colors='red', linestyles='dashed')
            else:
                ax1.vlines(1.565, -0.1, 1.1 * max(means), colors='red', linestyles='dashed')
                ax2.vlines(1.373, 0, 1.1 * max(der) / scaling ** 2, colors='grey', linestyles='dashed')
                ax2.vlines(1.565, 0, 1.1 * max(der) / scaling ** 2, colors='red', linestyles='dashed')
                ax1.vlines(1.373, -0.1, 1.1 * max(means), colors='grey', linestyles='dashed')

    # one more time for label --> label only during last plotting
    if noise_model == 'BitFlip':
        ax1.vlines(0.951, 0, 1.1 * max(means), colors='red', linestyles='dashed', label='threshold')
        ax2.vlines(0.951, 0, 1.1 * max(der), colors='red', linestyles='dashed', label='threshold')
    else:
        ax1.vlines(1.565, -0.1, 1.1 * max(means), colors='red', linestyles='dashed', label='threshold')
        ax2.vlines(1.373, 0, 1.1 * max(der) / scaling ** 2, colors='grey', linestyles='dashed',
                   label=r'$\frac{3}{2} \cdot$ bit-flip threshold')
        ax2.vlines(1.565, 0, 1.1 * max(der) / scaling ** 2, colors='red', linestyles='dashed', label='threshold')
        ax1.vlines(1.373, -0.1, 1.1 * max(means), colors='grey', linestyles='dashed',
                   label=r'$\frac{3}{2} \cdot$ bit-flip threshold')
    # ax1.tick_params(axis='y', labelcolor='black')
    # ax1.set_xlabel('associated temperature')
    ax2.set_xlabel(r'associated temperature $T_\mathrm{NL}$')
    ax1.set_ylabel(r'$\mu_\mathrm{op}$; single branch')
    ax2.set_ylabel(r'susceptibility $\chi(\mu_\mathrm{op})$')
    plt.tight_layout()
    ax1.legend()
    ax2.legend()

    # Add labels in the upper left corner
    ax1.text(-0.16, 0.98, '(a)', transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left')

    ax2.text(-0.16, 0.98, '(b)', transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='left')

    # ax2.legend()
    plt.savefig('depolarizing_no_rf.pdf')
    if show:
        plt.show()


def plot_correlation():
    """
    Test function: plots correlation between latent space order parameter and susceptibility and the respective
    quantities of the raw data.
    :return:
    """
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    noise_model = 'BitFlip'

    dist = 21
    name_dict_latent = "latents_bitflip_simple_dim1f7"

    # name_dict_latent = "latents_BitFlip_standard_rf_21ff1_1"

    latents = ResultsWrapper(name=name_dict_latent).load().get_dict()
    noises = list(latents[dist].keys())
    latent = list(latents[dist].values())
    zs = np.array(list(map(lambda x: torch.sum(torch.abs(x[0]), dim=1).cpu().detach().numpy(), latent)))
    flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(), latent)))

    if noise_model == 'BitFlip':
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
    else:
        noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

    der = np.zeros(len(noises))
    means = np.zeros(len(noises))
    unc = []
    for i in range(1, len(noises)):
        vals = zs[i][np.where(flips[i] == -1)]
        error_bars = simple_bootstrap(vals, np.mean, r=100)

        means[i] = error_bars[0]
        if i == 1:
            scaling = means[i]
        means[i] /= scaling
        unc.append((error_bars[1], error_bars[2]))
        der[i] = np.mean((vals / scaling) ** 2) - np.mean(vals / scaling) ** 2
    # der = smooth(der, 7)
    unc = list(map(list, zip(*unc)))
    print(unc)

    ax1.errorbar(noises[1:], means[1:], yerr=unc, color=coloring[3], marker='o', markersize=1, linestyle='solid',
                 label='d={}'.format(dist))

    ax2.plot(noises[1:], der[1:], color=coloring[3], marker='o', markersize=1, linestyle='solid',
             label='d={}'.format(dist))

    mean = np.load('mean.npy')
    sus = np.load('sus.npy')

    temperature = np.arange(0.02, 3, 0.02)
    noises = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))

    ax1.plot(2 / np.log((1 - noises) / noises), mean, color='blue', marker='o', markersize=1,
             linestyle='solid', label=r'$\langle M \rangle$')
    ax1.plot(2 / np.log((1 - noises) / noises), 1 - 2 * 4 * ((1 - noises) * noises ** 3 + (1 - noises) ** 3 * noises),
             color='black',
             label='theory')

    ax2.plot(2 / np.log((1 - noises) / noises), sus, color='blue', marker='o', markersize=1,
             linestyle='solid', label=r'$\chi (M)$')

    if noise_model == 'BitFlip':
        ax1.vlines(0.951, 0, 1.1 * max(means), colors='red', linestyles='dashed', label='threshold')
        ax2.vlines(0.951, 0, 1.1 * max(der), colors='red', linestyles='dashed', label='threshold')

    ax2.set_xlabel(r'associated temperature $T_\mathrm{NL}$')
    ax1.set_ylabel(r' mean; $\mu_\mathrm{op}$, single branch, resized')
    ax2.set_ylabel(r'susceptibility $\chi$, resized')
    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.savefig('plots/bitflip_vs_mean.svg')
    plt.show()


if __name__ == '__main__':
    # Calls correlation plot
    plot_correlation()


def plot_binder_cumulant(latents: dict, random_flip: bool, structure: str, noise_model: str, show=True):
    """
    For the random-bond Ising model, the Binder cumulant is often used to determine the phase transition. This method
    plots the binder-cumulant of the latent space order parameter
    :param latents: dictionary: key: distance, value: dictionary: key: noise, value: latent space mean
    :param random_flip: bool, whether random_flip is activated or not
    :param structure: structure of the VAE network
    :param noise_model: str, specifies which noise model is used, supported: 'BitFlip' or 'Depolarizing'
    :param show: bool, whether to show the resulting plot
    :return:
    """
    dists = list(latents.keys())
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for k, dist in enumerate(dists):
        if dist == 7 or dist == 9:
            continue
        noises = list(latents[dist].keys())
        print(noises)
        latent = list(latents[dist].values())
        zs = np.array(list(map(lambda x: x[0].cpu().detach().numpy(), latent)))[:, :, 0]
        # print(np.shape(zs))
        flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(), latent)))
        # print(np.shape(flips))
        if noise_model == 'BitFlip':
            noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        else:
            noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))

        der = np.zeros(len(noises))
        means = np.zeros(len(noises))
        unc = np.zeros(len(noises))
        for i in range(1, len(noises)):
            # vals = np.zeros(np.sum(idxs[i]))
            # vals = zs[i][idx][:, 0]
            vals = zs[i][np.where(flips[i] == -1)]
            means[i] = np.mean(vals)
            # unc[i] = simple_bootstrap(vals, np.mean, r=100)
        means /= means[1]
        for i in range(1, len(noises)):
            vals = zs[i][np.where(flips[i] == -1)]
            der[i] = 1 - 3 * np.mean((vals / means[1]) ** 4) / (np.mean((vals / means[1]) ** 2) ** 2)
        der = smooth(der, 7)

        ax1.errorbar(noises[1:], means[1:], yerr=unc[1:], color=coloring[k], linestyle='dashed')

        ax2.plot(noises[1:], dist * der[1:], color=coloring[k], label=str(dist))
        if noise_model == 'BitFlip':
            plt.vlines(0.951, -250, 0, colors='red', linestyles='dashed')
        else:
            plt.vlines(1.565, 0, max(dist * der), colors='red', linestyles='dashed')
            plt.vlines(1.373, 0, max(dist * der), colors='red', linestyles='dashed')
            # plt.vlines(0.109, 0, max(dist * der), colors='red', linestyles='dashed')

    plt.title("Structure: " + structure)
    # ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlabel('associated temperature')
    # ax1.set_xlabel('bitflip probability p')
    ax1.set_ylabel(r'mean $\langle \mu \rangle$ single branch normalized')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel(r'$d \cdot $ binder cumulant', color='blue')
    temperatures = np.arange(0., 2., 0.01)
    noises = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), temperatures)))
    p_nts = 4 * (noises * (1 - noises) ** 3 + noises ** 3 * (1 - noises))
    exp = 1 - 2 * p_nts
    ax1.plot(temperatures, exp)
    plt.tight_layout()
    plt.legend()
    if show:
        plt.show()


def scatter_latent_var(latents: dict, random_flip: bool, structure: str):
    """
    Scatters data points in the latent space
    :param latents: dictionary: key: distance, value: dictionary: key: noise, value: latent space mean
    :param random_flip: bool, whether random_flip is activated or not
    :param structure: structure of the VAE network
    :return:
    """
    warnings.warn("Only supported for 1d latent space.")
    dists = list(latents.keys())
    for dist in dists:
        noises = list(latents[dist].keys())
        latent = list(latents[dist].values())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        for j, noise in enumerate(noises):
            if not j % 5 == 0:
                continue
            for h in range(0, len(latent[j][0].cpu()), int(len(latent[j][0].cpu()) / 10)):
                plt.scatter(noise, latent[j][0].cpu()[h], s=3, color='black')
        #plt.ylim([-0.00001, 0.00001])
        plt.xlabel('associated temperature')
        plt.ylabel(r'latent space mean $\mu$')
        # plt.xscale('log')
        plt.xlim(0, 2)
        if random_flip:
            plt.title(structure + " + random flip")
        else:
            plt.title(structure)
        plt.show()


def plot_reconstruction_error(reconstructions: dict, random_flip: bool, structure: str):
    """
    Plots the reconstruction error vs. the noise strength
    :param reconstructions: dictionary: key: distance, value: dictionary: key: noise, value: latent space mean
    :param random_flip: bool, whether random_flip is activated or not
    :param structure: structure of the VAE network
    :return:
    """
    dists = list(reconstructions.keys())
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for k, dist in enumerate(dists):
        if dist == 7 or dist == 9:
            continue
        noises = list(reconstructions[dist].keys())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        reconstruction = list(reconstructions[dist].values())
        rec_mean = np.array(list(map(lambda x: np.array(x[0].cpu().data), reconstruction)))
        rec_var = np.array(list(map(lambda x: np.array(x[1].cpu().data), reconstruction)))
        # rec_mean = np.array(list(map(lambda x: np.array(x.cpu().data), reconstruction)))
        ax1.plot(noises, rec_mean, color=coloring[k], linestyle='dashed')
        ax2.plot(noises[15:], rec_var[15:], color=coloring[k], label=str(dist))
        plt.vlines(1.373, 0, 1, colors='red', linestyles='dashed')
        # plt.vlines(0.109, 0, max(rec_var), colors='red', linestyles='dashed')
    ax1.set_xlabel('associated temperature')
    # ax1.set_xlabel('bitflip probability p')
    ax1.set_ylabel('reconstruction error')
    ax2.set_ylabel('reconstruction variance', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.legend()
    plt.title("Structure: " + structure)
    # plt.xscale('log')
    plt.tight_layout()
    plt.show()


def plot_reconstruction_derivative(reconstructions: dict, random_flip: bool, structure: str):
    """
    Plots the derivative of the reconstruction error vs. the noise strength
    :param reconstructions: dictionary: key: distance, value: dictionary: key: noise, value: latent space mean
    :param random_flip: bool, whether random_flip is activated or not
    :param structure: structure of the VAE network
    :return:
    """
    dists = list(reconstructions.keys())
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
    for k, dist in enumerate(dists):
        if dist == 7 or dist == 9:
            continue
        noises = list(reconstructions[dist].keys())
        # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        reconstruction = list(reconstructions[dist].values())
        reconstruction = np.array(list(map(lambda x: np.array(x[0].cpu().data), reconstruction)))
        # reconstruction = smooth(reconstruction, 7)
        # der = np.zeros(len(reconstruction))
        # der[0] = (reconstruction[1] - reconstruction[0]) / (noises[1] - noises[0])
        # for i in np.arange(1, len(reconstruction) - 1):
        #     der[i] = (reconstruction[i + 1] - reconstruction[i - 1]) / (noises[i + 1] - noises[i - 1])
        # der[-1] = (reconstruction[-1] - reconstruction[-2]) / (noises[-1] - noises[-2])
        # der = np.gradient(reconstruction, noises)

        act_noises = np.exp(-2 / noises) / (1 + np.exp(-2 / noises))
        syndrome_p = 4 * (act_noises ** 3 * (1 - act_noises) + act_noises * (1 - act_noises) ** 3)
        ax1.plot(noises, reconstruction, color=coloring[k], linestyle='dashed')
        # ax2.plot(noises[8:-8], der[8:-8], color=coloring[k], label=str(dist))

        Ts = np.arange(0, 2, 0.01)
        noises = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))
        ps = np.array(list(map(lambda x: 4 * (x ** 3 * (1 - x) + x * (1 - x) ** 3), noises)))
        Ts = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))

        # ax1.set_ylim(0, 1)
        # plt.vlines(0.95, -0.1, max(der[8:-8]), colors='red', linestyles='dashed')
        # plt.vlines(1.373, -0.1, max(der[8:-8]), colors='red', linestyles='dashed')

    ax1.tick_params(axis='y', labelcolor='black')
    # ax1.set_xlabel('bitflip probability p')
    ax1.set_xlabel('associated temperature')
    ax1.set_ylabel('reconstruction error')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('reconstruction derivative', color='blue')
    plt.legend()
    # plt.xscale('log')
    # plt.xlim(0, 2)
    plt.title("Structure: " + structure)
    plt.tight_layout()
    plt.show()


def plot_reconstruction(data, noise, distance, model):
    """
    Plots a reconstruction for some given sample.
    :param data: Data sample
    :param noise: Noise strength
    :param distance: distance of the code
    :param model: VAE model
    :return:
    """
    model.load()
    model.eval()
    sample = data[0]
    syndrome = sample[0]  # [0]
    print(syndrome.size())
    print(torch.mean(syndrome))
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    im = axs[0, 0].imshow(np.reshape(syndrome[0].cpu().numpy(), (distance, distance)), cmap='inferno')

    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    bar = fig.colorbar(im, ticks=[-1, 1], shrink=0.6)
    #try:
    im = axs[0, 1].imshow(np.reshape(syndrome[1].cpu().numpy(), (distance, distance)), cmap='inferno')
    # plt.colorbar(im)

    bar = fig.colorbar(im, ticks=[-1, 1], shrink=0.6)

    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    # Adjust the axis limits to center the image
    padding = 1  # Adjust this value to control how much space you want around the image
    axs[0, 0].set_xlim(-padding, distance)
    axs[0, 0].set_ylim(distance, -padding)  # Flip the y-axis to keep origin='upper'

    axs[0, 1].set_xlim(-padding, distance)
    axs[0, 1].set_ylim(distance, -padding)  # Flip the y-axis to keep origin='upper'

    # Draw lines that extend beyond the image
    for i in range(-padding, distance + 1):
        axs[0, 0].vlines(i - 0.5, -padding, distance, colors='black')
        axs[0, 0].hlines(i - 0.5, -padding, distance, colors='black')
        axs[0, 1].vlines(i, -padding, distance, colors='black')
        axs[0, 1].hlines(i, -padding, distance, colors='black')
    # plt.savefig('syndrome_bitflip_for_recon.svg')
    # plt.show()

    with torch.no_grad():
        output, mean, log_var = model.forward(data.get_syndromes())
        # output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    recon_syn = output[0]
    # recon_log = output[1][0]
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im = axs[1, 0].imshow(np.reshape(recon_syn[0].cpu().numpy(), (distance, distance)), cmap='magma')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    fig.colorbar(im, shrink=0.6)

    im = axs[1, 1].imshow(np.reshape(recon_syn[1].cpu().numpy(), (distance, distance)), cmap='magma')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    fig.colorbar(im, shrink=0.6)
    loss = torch.nn.MSELoss(reduction='mean')
    print('Loss: ', loss(syndrome, recon_syn))
    print('Ideal loss: ', loss(syndrome, torch.mean(syndrome) * torch.ones_like(syndrome)))
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    # Adjust the axis limits to center the image
    padding = 1  # Adjust this value to control how much space you want around the image
    axs[1, 0].set_xlim(-padding, distance)
    axs[1, 0].set_ylim(distance, -padding)  # Flip the y-axis to keep origin='upper'

    axs[1, 1].set_xlim(-padding, distance)
    axs[1, 1].set_ylim(distance, -padding)  # Flip the y-axis to keep origin='upper'

    # Draw lines that extend beyond the image
    for i in range(-padding, distance + 1):
        axs[1, 0].vlines(i - 0.5, -padding, distance, colors='black')
        axs[1, 0].hlines(i - 0.5, -padding, distance, colors='black')
        axs[1, 1].vlines(i, -padding, distance, colors='black')
        axs[1, 1].hlines(i, -padding, distance, colors='black')

    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, a in enumerate(axs.flatten()):
        a.text(-0.15, 0.98, labels[i], transform=a.transAxes, fontsize=12,
               verticalalignment='top', horizontalalignment='left')

    plt.subplots_adjust(wspace=0.2, hspace=-0.1)
    # plt.tight_layout()
    # plt.savefig('reconstruction_depolarizing.svg')
    plt.show()
    print(torch.mean(recon_syn))


def plot_mean_variance_samples(raw, distance, noise_model):
    """
    Plots the mean and variance of raw data samples
    :param raw: raw data samples
    :param distance: distance of the code
    :param noise_model: noise model under which the data samples were obtained
    :return:
    """
    results = raw.get(distance)
    noises = list(results.keys())
    # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
    # noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
    if noise_model == 'BitFlip':
        for noise in noises:
            # mean = torch.mean(results[noise][0])
            for i in np.arange(0, len(results[noise][0]), int(len(results[noise][0]) / 2)):
                plt.scatter(2 / (np.log((1 - noise) / noise)), results[noise][0][i].cpu().detach().numpy(), c='black',
                            s=3)
    elif noise_model == 'Depolarizing':
        for noise in noises:
            mean = torch.mean(results[noise][0], dim=1)
            for i in np.arange(0, len(mean), int(len(mean) / 5)):
                plt.scatter(2 / (np.log((1 - noise) / noise)), mean[i].cpu().detach().numpy(), c='black',
                            s=3)
    # plt.xscale('log')
    plt.show()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if noise_model == 'BitFlip':
        means = np.array(list(map(lambda x: torch.mean(torch.abs(results[x][0])).cpu().detach().numpy(), noises)))
        vars = np.array(list(map(lambda x: (torch.mean(results[x][0] ** 2).cpu().detach().numpy() - torch.mean(
            torch.abs(results[x][0])).cpu().detach().numpy() ** 2), noises)))
        print(results[noises[0]][0].cpu().detach().numpy())
        print(np.var(results[noises[0]][0].cpu().detach().numpy(), ddof=1))
        vars2 = np.array(list(map(lambda x: np.var(results[x][0].cpu().detach().numpy(), ddof=1), noises)))
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        # vars = smooth(vars, 11)
        ax1.plot(noises, means, c='black', label='mean of syndrome')
        ax2.plot(noises, vars, c='blue', label='susceptibility')
        ax2.plot(noises, vars2, c='red', label='variance of syndrome2')
        Ts = np.arange(0, 3, 0.01)
        noises = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))
        p_nts = np.array(list(map(lambda x: 4 * (x ** 3 * (1 - x) + x * (1 - x) ** 3), noises)))

        ax1.plot(Ts, 1 - 2 * p_nts, label=r'$1-2p_{NTS}$')

        # plt.plot(Ts, max(vars) / max(np.gradient(p_nts, Ts)) * np.gradient(p_nts, Ts), label='derivative')
        # ax1.plot(Ts, (1 - (1 - 2 * p_nts) ** 2), label='theoretical variance')

        def p(noise):
            return 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)

        def pnn(noise):
            return 6 * noise * (1 - noise) ** 6 + 20 * noise ** 3 * (1 - noise) ** 4 + 6 * noise ** 5 * (
                    1 - noise) ** 2 + 6 * noise ** 2 * (1 - noise) ** 5 + 20 * noise ** 4 * (
                    1 - noise) ** 3 + 6 * noise ** 6 * (1 - noise)

        def po(noise):
            return 8 * noise * (1 - noise) ** 7 + 56 * noise ** 3 * (1 - noise) ** 5 + 56 * noise ** 5 * (
                    1 - noise) ** 3 + 8 * noise ** 7 * (1 - noise)

        def var(noise):
            # return 1 / distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_o(noise))) - (
            #            1 - 2 * p_nts(noise)) ** 2
            return 1 / distance ** 2 * (
                    1 + 4 * (1 - 2 * pnn(noise)) + (distance ** 2 - 5) * (1 - 2 * p(noise)) ** 2) - (
                    1 - 2 * p(noise)) ** 2

        ax2.plot(Ts, var(noises), label='total variance')
        plt.vlines(0.95, 0, max(var(noises)), colors='red', linestyles='dashed')
    elif noise_model == 'Depolarizing':
        mean = np.array(list(map(lambda x: torch.mean(results[x][0], dim=1).cpu().detach().numpy(), noises)))
        means = np.array(list(map(lambda x: np.mean(np.abs(x)), mean)))
        vars = np.array(list(map(lambda x: (np.mean(x ** 2) - np.mean(
            np.abs(x)) ** 2), mean)))
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        ax1.plot(noises, means, c='black')
        ax2.plot(noises, vars, c='blue')

        plt.vlines(1.373, 0, max(vars), colors='red', linestyles='dashed')
        plt.vlines(1.225, 0, max(vars), colors='red', linestyles='dashed')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlabel('bitflip probability p')
    ax1.set_ylabel(r'mean $\langle | \mu | \rangle$')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('susceptibility')
    plt.legend()
    # ax1.set_xlim(0, 2)
    plt.tight_layout()
    # fig.legend(loc = (0.53, 0.77))
    plt.show()


def plot_loss(history, distance, val=True, save=False):
    """
    Plots training history and validation loss.
    :param history:
    :param distance:
    :param val:
    :param save:
    :return:
    """
    epochs = len(history.history['accuracy'])
    # summarize history for loss
    plt.plot(np.arange(epochs), history.history['loss'])
    plt.plot(np.arange(epochs), history.history['val_loss'])
    plt.title('model loss L={0}'.format(distance))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    if save:
        plt.savefig(str(Path().resolve().parent) + "/data/loss.pdf", bbox_inches='tight')
    plt.show()


def plot_collapsed(dictionary, noises, pc, nu):
    """
    Plot data collapse.
    :param nu:
    :param pc:
    :param noises:
    :param dictionary: dictionary: keys: distances, labels: tuple (noises, predictions)
    :return:
    """
    dists_str = dictionary.keys()
    dists = np.array(list(dists_str)).astype(int)
    markers = ['o', 'x', 's', 'v', '+', '^']
    colors = ['blue', 'black', 'green', 'red', 'orange', 'purple']
    for i, dist in enumerate(dists):
        predictions, errors = dictionary[str(dist)]
        noises_temp = np.array(list(map(lambda x: (x - pc) * dist ** (1 / nu), noises)))
        plt.plot(noises_temp, predictions[:, 0], color=colors[i], marker=markers[i], linewidth=1, label=dist)
        plt.plot(noises_temp, predictions[:, 1], color=colors[i], marker=markers[i], linewidth=1, label=dist)
    plt.legend()
    plt.xlabel(r'noise probability $p$')
    plt.ylabel(r'ordered/unordered')
    plt.show()
