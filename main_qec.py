import os
import sys
from collections import Counter
import seaborn as sns

import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
from tqdm import tqdm
import h5py

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset

from src.qec_transformer.utils import simple_bootstrap, get_pr, find_crossing, \
    find_crossing2, get_scaling, function
from src.qec_transformer.optimizer import make_optimizer
from src.qec_transformer.loops import training_loop, online_training, eval_log_error_rate, val_loss_uncertainty

from src.qec_transformer.code_capacity.nn.qectransformer import QecTransformer
from src.qec_transformer.code_capacity.nn.qecVT import QecVT
from src.qec_transformer.circuit_level.nn.r_qecVT import RQecVT
from src.qec_transformer.circuit_level.nn.r_qectransformer import RQecTransformer

from src.qec_transformer.code_capacity.data.dataset import DepolarizingSurfaceData, BitflipSurfaceData
from src.qec_transformer.circuit_level.data.dataset import CircuitLevelSurfaceData, PhenomenologicalSurfaceData

import re

torch.set_printoptions(precision=3, sci_mode=False)
torch.set_printoptions(threshold=10_000)

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 14

load_pretrained = True
logical = 'maximal'
readout = 'transformer-decoder'
patch_distance = 3
online_learning = True
from_checkpoint = True
finite_size_scaling = False

n_layers_dict = {3: 3, 5: 3, 7: 3, 9: 3, 11: 3, 13: 3}
d_model_dict = {3: 128, 5: 128, 7: 128, 9: 256, 11: 256, 13: 256}
d_ff_dict = {3: 128, 5: 128, 7: 128, 9: 256, 11: 256, 13: 256}


@njit
def decimal_to_binary(decimals, bit_width):
    # Prepare the output array
    n = len(decimals)
    binary_array = np.zeros((n, bit_width), dtype=np.uint8)

    # Loop through each bit position
    for j in range(bit_width):
        # Create a bit mask for the j-th bit
        mask = 1 << (bit_width - j - 1)

        # Extract the j-th bit across all decimal values and store it
        binary_array[:, j] = (decimals & mask) >> (bit_width - j - 1)

    return binary_array


@njit
def binary_to_decimal(binary):
    # Get the number of binary numbers and their bit width
    binary = binary.ravel().astype(np.uint8)
    decimal_value = 0  # Initialize as an integer
    for j in np.arange(binary.size):
        decimal_value |= binary[j] << (binary.size - j - 1)
    return decimal_value


def main_qec(distance, task, noise_model, noise):
    lr = 1e-4
    num_epochs = 150
    batch_size = 1000
    data_size = 1000000
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    mode = noise_model

    model_dict = {
        'n': distance ** 2,
        'k': 1,
        'd_model': d_model_dict[distance],
        'd_ff': d_ff_dict[distance],
        'n_layers': n_layers_dict[distance],
        'n_heads': 8,
        'device': device,
        'dropout': 0.,
        'max_seq_len': distance ** 2 - 1 + 2 * distance,
        'noise_model': mode
    }

    if distance < 7:
        model_type = 'qectransformer'
    else:
        model_type = 'qecVT'

    if noise_model == 'depolarizing':
        run_depolarizing(task=task, noise=noise, distance=distance, model_type=model_type, lr=lr, device=device,
                         num_epochs=num_epochs, batch_size=batch_size, model_dict=model_dict, data_size=data_size)
    elif noise_model == 'phenomenological':
        run_phenomenological(task=task, noise=noise, distance=distance, model_type=model_type, lr=lr, device=device,
                             num_epochs=num_epochs, batch_size=batch_size, model_dict=model_dict, data_size=data_size)
    elif noise_model == 'circuit-level':
        run_circuit_level(task=task, noise=noise, distance=distance, model_type=model_type, lr=lr, device=device,
                          num_epochs=num_epochs, batch_size=batch_size, model_dict=model_dict, data_size=data_size)
    else:
        raise ValueError(f'Noise model {noise_model} is not specified for mode transformer.')


def log_error_rate(checkpoint, model, thresholds, noise, device, mode, data, iteration, distance):
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        model = model.load()
    for threshold in thresholds:
        error_rate_data = {
            noise: eval_log_error_rate(model, data, distance, noise, device, mode=mode, threshold=threshold,
                                       increased_num=noise < 2e-3)}
        print(f'Threshold {threshold}: {error_rate_data}')
        torch.save(error_rate_data,
                   "data/log_error_rate_{0}_{1}_{2}_{3}".format(iteration, distance, noise, threshold))


def estimate_ci(checkpoint, model, noise, device, mode, data, iteration, distance):
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        model = model.load()
    d = val_loss_uncertainty(iteration, distance, noise, model, data, device)
    ci = (1 - 2 * d[0] / np.log(2), 2 / np.log(2) * d[1], 2 / np.log(2) * d[2])
    torch.save(ci, "data/{0}_{1}_{2}_{3}".format('val_loss', iteration, distance, noise))


def merge_dictionaries(distances, noise_vals, eval_what, iteration, thresholds=None):
    if eval_what == 'val_loss':
        for dist in distances:
            super_dict = {}
            for i, n in enumerate(noise_vals):
                try:
                    print("data/{0}_{1}_{2}_{3}".format(eval_what, iteration, dist, n))
                    d = torch.load("data/{0}_{1}_{2}_{3}".format(eval_what, iteration, dist, n),
                                   map_location=torch.device('cpu'))
                    if isinstance(d, torch.Tensor):
                        print(torch.mean(d))
                        d = {n: simple_bootstrap((np.log(2) + 2 * d.detach().numpy()) / np.log(2))}
                        print(d)
                except FileNotFoundError:
                    d = {}
                try:
                    for k, v in d.items():
                        super_dict[k] = v
                except AttributeError:
                    try:
                        unc = np.load("data/unc_{0}_{1}_{2}.npy".format(iteration, dist, n))
                    except FileNotFoundError:
                        unc = (0, 0)
                    super_dict[n] = (1 - 2 * d / np.log(2), 2 / np.log(2) * unc[0], 2 / np.log(2) * unc[1])
            torch.save(super_dict, "data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
    elif eval_what == 'log_error_rate':
        assert thresholds is not None
        for t in thresholds:
            for dist in distances:
                super_dict = {}
                for i, n in enumerate(noise_vals):
                    try:
                        d = torch.load("data/{0}_{1}_{2}_{3}_{4}".format(eval_what, iteration, dist, n, t),
                                       map_location=torch.device('cpu'))
                    except FileNotFoundError:
                        d = {}
                    for k, v in d.items():
                        super_dict[k] = v
                torch.save(super_dict, "data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
    else:
        raise ValueError(f'Quantity {eval_what} is not supported.')


def run_depolarizing(task, noise, distance, model_type, lr, device, num_epochs, batch_size, model_dict, data_size,
                     pretrained_model=None):
    noise_vals = [0.01, 0.02, 0.05, 0.08, 0.11, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39]
    distances = [3, 5, 7, 9, 11]
    thresholds = [0.5, 0.3, 0.15, 0.05, 0.01]

    assert noise in noise_vals
    assert distance in distances

    iteration = 'cc'
    eval_what = 'log_error_rate' if task == 400 else 'val_loss'
    mode = 'depolarizing'
    pretrained_model = iteration + '_' + str(distance) + '_' + str(noise_vals.index(noise) - 1)

    try:
        checkpoint = torch.load(f'data/checkpoint_{iteration}_{distance}_{noise}', map_location=device)
    except FileNotFoundError:
        checkpoint = None

    if model_type == 'qectransformer':
        model = QecTransformer(name=iteration + '_' + str(distance) + '_' + str(noise), distance=distance,
                               readout=readout,
                               penc_type='fixed', **model_dict).to(
            device)
    else:
        model = QecVT(name=iteration + '_' + str(distance) + '_' + str(noise), distance=distance,
                      pretrained_qec_name=iteration + '_' + str(patch_distance) + '_' + str(noise),
                      readout=readout, patch_distance=patch_distance, **model_dict)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    data = DepolarizingSurfaceData(distance=distance,
                                   noise=noise,
                                   name=f'{iteration}_{distance}_{noise}',
                                   load=False,
                                   device=device,
                                   logical=logical)

    if task == 1:
        if not online_learning:
            try:
                print('Generate dataset.')
                data = (data
                        .initialize(data_size)
                        .save())
                # data = (data
                #        .load())
                # print('Dataset loaded.')
            except FileNotFoundError:
                print('Generate dataset.')
                data = (data
                        .initialize(data_size)
                        .save())
            train, val = data.get_train_val_data()  # default ratio 80/20
        else:
            print('Online learning.')

        try:
            if checkpoint is not None:
                print('Load from checkpoint.')
                model.load_state_dict(checkpoint["model"])
            else:
                model = model.load()
        except FileNotFoundError:
            if checkpoint is not None:
                print('Load from checkpoint.')
            elif load_pretrained:
                try:
                    print('Load pretrained model.')
                    # Load weights from pretrained smaller model
                    pretrained_state_dict = torch.load("data/net_{}.pt".format(pretrained_model),
                                                       map_location=torch.device('cpu'))
                    # Interpolate or copy weights to larger model
                    for name, param in pretrained_state_dict.items():
                        if name in model.state_dict():
                            if len(param.size()) == 1:
                                model.state_dict()[name][:param.size(0)] = param
                            else:
                                model.state_dict()[name][:param.size(0), :param.size(1)] = param
                except FileNotFoundError:
                    try:
                        print('Load pretrained from checkpoint.')
                        pretrained_checkpoint = torch.load(f'data/checkpoint_{pretrained_model}',
                                                           map_location=torch.device('cpu'))
                        pretrained_state_dict = pretrained_checkpoint["model"]
                        # Interpolate or copy weights to larger model
                        for name, param in pretrained_state_dict.items():
                            if name in model.state_dict():
                                if len(param.size()) == 1:
                                    model.state_dict()[name][:param.size(0)] = param
                                else:
                                    model.state_dict()[name][:param.size(0), :param.size(1)] = param
                    except FileNotFoundError:
                        print('Load new model. File not found.')
                    pass
            else:
                print('Load new model.')
                pass
        except RuntimeError:
            if checkpoint is not None:
                print('Load from checkpoint.')
            else:
                print('Load new model. Runtime error.')
            pass

        if online_learning:
            model = online_training(model, data, make_optimizer(lr), device, epochs=num_epochs, batch_size=batch_size,
                                    num_data=data_size, from_checkpoint=from_checkpoint, checkpoint=checkpoint)
        else:
            assert train is not None and val is not None
            model = training_loop(model, train, val, make_optimizer(lr), device, epochs=num_epochs,
                                  batch_size=batch_size,
                                  mode=mode, activate_scheduler=True, load_pretrained=load_pretrained)

        # Final forward pass to get val loss including estimates
        checkpoint = torch.load(f'data/checkpoint_{iteration}_{distance}_{noise}', map_location=device)
        estimate_ci(checkpoint=checkpoint, model=model, noise=noise, device=device, mode=mode,
                    data=data, iteration=iteration, distance=distance)
    elif task == 4:  # Evaluate logical error rate
        data = DepolarizingSurfaceData(distance=distance,
                                       noise=noise,
                                       name='eval',
                                       load=False,
                                       device=device,
                                       only_syndromes=False)
        log_error_rate(checkpoint=checkpoint, model=model, thresholds=thresholds,
                       noise=noise, device=device, mode=mode,
                       data=data, iteration=iteration, distance=distance)
    elif task == 100:
        # Merge dictionaries
        merge_dictionaries(distances=distances, noise_vals=noise_vals, eval_what='val_loss', iteration=iteration)

        # Plot
        fig, ax = plt.subplots()
        coloring = ['blue', 'red', 'green', 'black', 'powderblue', 'orange']

        for i, dist in enumerate(distances[:-1]):
            dict = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())  # noises
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            if dist == 9:
                ax.errorbar(n[:-5], pr_m[:-5], yerr=(pr_d[:-5], pr_u[:-5]), marker='o', markersize=5,
                            linestyle='dotted', label='NN'.format(dist),
                            color=coloring[i])
            elif dist == 7:
                ax.errorbar(n[:-4], pr_m[:-4], yerr=(pr_d[:-4], pr_u[:-4]), marker='o', markersize=5,
                            linestyle='dotted', color=coloring[i])
            else:
                ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5, linestyle='dotted', color=coloring[i])

        if eval_what == 'result' or eval_what == 'val_loss':
            noises = np.arange(0.0, 0.4, 0.01)
            for d in [3]:
                g_stabilizer = np.loadtxt('src/qec_transformer/code/stabilizer_' + 'rsur' + '_d{}_k{}'.format(d, 1))
                logical_opt = np.loadtxt('src/qec_transformer/code/logical_' + 'rsur' + '_d{}_k{}'.format(d, 1))
                pr, entr, var = get_pr(d, noises, g_stabilizer, logical_opt, d ** 2)
                plt.plot(noises, pr, color=coloring[0])
            try:
                analytical_5 = np.loadtxt('src/qec_transformer/analytical/analytical_ci_d5')
                plt.plot(noises, analytical_5 / np.log(2), color=coloring[1])
            except FileNotFoundError:
                print('Analytical values for d=5 not found.')

        '''
        # Show conditional collapse data: Calculate p_lambda:
        
        p_lambda = np.zeros_like(analytical_5)
        
        noises = np.arange(0, 0.4, 0.01)
        patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        for k, p in enumerate(tqdm(noises)):
            sz = 10000
            data = (DepolarizingSurfaceData(distance=3,
                                            noise=p,
                                            name=f'{iteration}_{distance}_{noise}',
                                            load=False,
                                            device=torch.device('cpu'),
                                            logical=logical)
                    .initialize(sz)
                    .get_syndromes())
            frequencies = torch.zeros(len(patterns), dtype=torch.int32)
            for i, pattern in enumerate(patterns):
                frequencies[i] = torch.sum(torch.all(data[:, -2:, 0] == pattern, dim=1))
            frequencies = frequencies / sz
            print(frequencies)
            p_lambda[k] = 1 + torch.sum(frequencies * torch.log(frequencies + 1e-12) / torch.log(torch.tensor(2))).numpy()
        np.savetxt('plambda.txt', p_lambda)
        
        p_lambda = np.loadtxt('plambda.txt')
        plt.plot(noises, p_lambda, label=r'$p_7(\lambda | s) = p_7(\lambda)$', color='cyan')
        '''

        ax.set_xlabel('noise probability p')
        if eval_what == 'result' or eval_what == 'val_loss':
            ax.set_ylabel(r'$I / \log 2$')
        else:
            ax.set_ylabel(r'$p_L$')

        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances[:-1])])]

        style_legend = [
            Line2D([0], [0], color='black', marker=[None, 'o'][i], markersize=3, linestyle=ls,
                   label=f'{['exact', 'NN'][i]}')
            for i, ls in enumerate(['solid', 'dotted'])]

        legend_handles = color_legend + style_legend  # + auto_handles
        ax.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        if eval_what == 'result' or eval_what == 'val_loss':
            # Define inset
            inset = zoomed_inset_axes(ax, zoom=2, bbox_to_anchor=(0.4, 0.45),  # Adjust position, previous (0.4, 0.45)
                                      bbox_transform=ax.transAxes)
            for i, dist in enumerate(distances[:-1]):
                # iteration = 'ff14' if dist<= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
                n = list(dict.keys())  # noises
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                # plotting
                if dist == 9:
                    inset.errorbar(n[:-7], pr_m[:-7], yerr=(pr_d[:-7], pr_u[:-7]), marker='o', markersize=4,
                                   linestyle='dotted',
                                   color=coloring[i])
                elif dist == 7:
                    inset.errorbar(n[:-5], pr_m[:-5], yerr=(pr_d[:-5], pr_u[:-5]), marker='o', markersize=4,
                                   linestyle='dotted',
                                   color=coloring[i])
                else:
                    inset.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted',
                                   color=coloring[i])

            noises = np.arange(0.0, 0.4, 0.01)
            for i, d in enumerate([3]):
                g_stabilizer = np.loadtxt('src/qec_transformer/code/stabilizer_' + 'rsur' + '_d{}_k{}'.format(d, 1))
                logical_opt = np.loadtxt('src/qec_transformer/code/logical_' + 'rsur' + '_d{}_k{}'.format(d, 1))
                pr, entr, var = get_pr(d, noises, g_stabilizer, logical_opt, d ** 2)
                inset.plot(noises, pr, label='d={} num'.format(d), color=coloring[i])
            analytical_5 = np.loadtxt('src/qec_transformer/analytical/analytical_ci_d5')
            inset.plot(noises, analytical_5 / np.log(2), label='d=5 num', color=coloring[1])

            # Zoom into a specific region in the inset
            inset.set_xlim(0.15, 0.21)  # Adjust as needed
            inset.set_ylim(-0.15, 0.2)  # Adjust as needed

            # Optional: Add labels and ticks
            inset.tick_params(labelsize=9)
            # inset.set_title("Inset", fontsize=10)

            # Add a rectangle to show the zoomed-in area in the main plot
            ax.indicate_inset_zoom(inset, edgecolor="black")

            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black")

        plt.suptitle('code capacity, depolarizing')
        plt.tight_layout()

        # Get crossings
        dict1 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 3))
        n = list(dict1.keys())  # noises
        dict2 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 5))

        if finite_size_scaling:
            t = find_crossing(np.array(n), np.array([3, 5]), dict1, dict2, noise_model=mode, quant='ci')
            print(t)
            crossing = t['rho']
        else:
            t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='ci')
            crossing = t[0]

        ax.axvline(crossing, ymin=0, ymax=1, color='silver', linewidth=1)
        inset.axvline(crossing, ymin=0, ymax=1, color='silver', linewidth=1)
        plt.savefig('plots/CI_depolarizing_9.svg')
        plt.show()

        dict1 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 7))
        dict2 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 9))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='ci')
        print('Threshold 7,9 ', t)
    elif task == 400:
        thresholds = [0.5, 0.3, 0.15, 0.05, 0.01]
        distances = [3, 5, 7, 9]

        # get dictionaries
        merge_dictionaries(distances, noise_vals, 'log_error_rate', iteration, thresholds)

        # Plots
        # scaling error rate
        fig, ax = plt.subplots()
        # for i, dist in enumerate(distances):
        coloring = ['blue', 'red', 'green', 'black', 'powderblue', 'orange']

        # MWPM
        for i, dist in enumerate([3, 5, 7, 9]):
            dict = torch.load("data/{0}_{1}_{2}_mwpm.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n[1:], pr_m[1:], yerr=(pr_d[1:], pr_u[1:]), marker='v', markersize=5, linestyle='dashed',
                        label='d={}_mwpm'.format(dist), color=coloring[i])

        # NN decoder without post-selection, scaling
        t = 0.5
        for i, dist in enumerate([3, 5, 7, 9]):
            dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n[1:], pr_m[1:], yerr=(pr_d[1:], pr_u[1:]), marker='o', markersize=5,
                        linestyle='solid',
                        label='d={0}'.format(dist), color=coloring[i])

        # plt.grid()
        ax.set_xlabel("$p$", fontsize=14)
        ax.set_ylabel("$p_L$", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.suptitle("code capacity, depolarizing")
        plt.xlim([0.019, 0.25])
        plt.ylim([3e-5, 0.6])
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=['v', 'o'][i], markersize=4, linestyle=ls,
                               label=f'{['MWPM', 'NN'][i]}')
                        for i, ls in enumerate(['dashed', 'solid'])]

        legend_handles = color_legend + style_legend
        plt.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        plt.grid()

        plt.savefig('plots/log_error_rate_depolarizing_9_scaling.svg')
        plt.show()

        # NN decoder without post-selection, linear
        fig, ax = plt.subplots()

        for i, dist in enumerate([3, 5, 7, 9]):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_mwpm.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='v', markersize=5, linestyle='dashed',
                        label='d={}_mwpm'.format(dist), color=coloring[i])

        t = 0.5
        markers = ['o', 'x', 'v', '^']
        for i, dist in enumerate([3, 5, 7, 9]):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
            n = list(dict.keys())
            print(n)
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5,
                        linestyle='solid',
                        label='d={0}'.format(dist), color=coloring[i])

        # plt.grid()
        ax.set_xlabel("$p$", fontsize=14)
        ax.set_ylabel("$p_L$", fontsize=14)
        plt.suptitle("code capacity, depolarizing")
        # plt.xlim([0, 0.25])
        # plt.ylim([0, 0.6])
        plt.xlim([0., 0.25])
        plt.ylim([0., 0.6])
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=['v', 'o'][i], markersize=4, linestyle=ls,
                               label=f'{['MWPM', 'NN'][i]}')
                        for i, ls in enumerate(['dashed', 'solid'])]

        legend_handles = color_legend + style_legend
        plt.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig('plots/log_error_rate_depolarizing_9.svg')
        plt.show()

        crossings = np.zeros((3, len(thresholds)))
        crossings[0] = 1 - np.array(thresholds)

        # find crossing 3,5 and 7,9
        dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 3, 0.5))
        n = list(dict1.keys())  # noises
        dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 5, 0.5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model='depolarizing', quant='pl')
        print('Crossing 3,5: ', t)

        dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 7, 0.5))
        dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 9, 0.5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model='depolarizing', quant='pl')
        print('Crossing 7,9: ', t)
        crossings[1, 0] = t[0]
        crossings[2, 0] = t[1]

        # NN decoder, post-selection, linear
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        styles = ['dashed', 'dotted', 'dashdot', 'dotted']

        p0s = [[0.16, 0.2], [0.15, 0.185], [0.14, 0.18], [0.12, 0.18]]
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances[1:]):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                ax1.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i + 1])

            # crossings for p_L
            dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 7, t))
            n = list(dict1.keys())
            dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 9, t))

            crossing = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='pl', p0=p0s[j])
            print(f'For c={t} crossing:', crossing)
            crossings[1, j + 1] = crossing[0]
            crossings[2, j + 1] = crossing[1]

        inset_ax = ax1.inset_axes([0.42, 0.74, 0.3, 0.2])  # or loc=1
        for i in range(len(thresholds)):
            inset_ax.errorbar(crossings[0, i], crossings[1, i], yerr=crossings[2, i], linestyle='None',
                              marker='s' if i == 0 else markers[i - 1], color='black', markersize=4)
        inset_ax.set_title("Threshold", fontsize=11)
        inset_ax.set_xlabel('c', fontsize=11)
        inset_ax.set_ylabel(r"$p_{th}$", fontsize=11)
        inset_ax.tick_params(axis='both', labelsize=11)

        # plt.grid()
        ax1.set_xlabel("$p$", fontsize=14)
        ax1.set_ylabel("$p_L$", fontsize=14)
        plt.suptitle("code capacity, depolarizing")
        plt.tight_layout()
        ax1.set_xlim([0., 0.25])
        ax1.set_ylim([0., 0.3])
        # plt.savefig('plots/log_error_rate_threshold_depolarizing_9.svg')
        # plt.show()

        crossings_abort = np.zeros((3, len(thresholds[1:])))
        crossings_abort[0] = 1 - np.array(thresholds[1:])

        # fig, ax = plt.subplots()
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances[1:]):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = np.array(list(map(lambda x: x[3], pr)))

                ax2.errorbar(n, pr_m, yerr=(np.sqrt(pr_m / 100000)), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i + 1])

            # crossings for p_abort
            dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 7, t))
            n = list(dict1.keys())
            dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 9, t))

            crossing = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='abort', p0=[0.15, 0.2])
            print(f'For c={t} crossing:', crossing)

            crossings_abort[1, j] = crossing[0]
            crossings_abort[2, j] = crossing[1]

        inset_ax = ax2.inset_axes([0.67, 0.1, 0.3, 0.2])
        for i in range(len(thresholds[1:])):
            inset_ax.errorbar(crossings_abort[0, i], crossings_abort[1, i], yerr=crossings_abort[2, i],
                              linestyle='None',
                              marker=markers[i], color='black', markersize=4)
        inset_ax.set_title("Threshold", fontsize=11)
        inset_ax.set_xlabel('c', fontsize=11)
        inset_ax.set_yticks((0.172, 0.177))
        inset_ax.set_ylabel(r"$p_{th}$", fontsize=11)
        inset_ax.tick_params(axis='both', labelsize=11)

        # plt.grid()
        ax2.set_xlim(0., 0.3)
        ax2.set_ylim(0., 1)
        ax2.set_xlabel("$p$", fontsize=14)
        ax2.set_ylabel(r'$p_{abort}$', fontsize=14)

        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i + 1]}')
                        for i, color in enumerate(coloring[1:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                               label=f'c={1 - thresholds[i + 1]}')
                        for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax1.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax2.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.tight_layout()
        plt.savefig('plots/log_error_rate_abort_depolarizing_9.svg', format="svg")
        plt.show()

        # NN decoder with post-selection, scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        styles = ['dashed', 'dotted', 'dashdot', 'solid']

        # inset_ax = ax1.inset_axes([0.67, 0.1, 0.3, 0.2])
        # inset_ax.set_title(r"Scaling $\alpha p^{t+1}$", fontsize=11)
        # inset_ax.set_xlabel('c', fontsize=11)
        # inset_ax.set_yticks((0.172, 0.177))
        # inset_ax.set_ylabel(r"$t$", fontsize=11)
        # inset_ax.tick_params(axis='both', labelsize=11)
        from_to = np.array([[[1, 7], [1, 5], [1, 9], [2, 5]],
                            [[2, 5], [1, 5], [2, 4], [1, 4]],
                            [[2, 6], [1, 6], [2, 6], [1, 6]]])
        vals = np.zeros((len(thresholds[1:]), len(distances[1:])))

        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances[1:]):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                # pr_unc = 0.5 * (np.array(pr_u) + np.array(pr_d))

                if t == 0.01 and dist == 5:
                    n = n[:-1]
                    pr_m = pr_m[:-1]
                    pr_u = pr_u[:-1]
                    pr_d = pr_d[:-1]
                ax1.errorbar(n[1:], pr_m[1:], yerr=(pr_d[1:], pr_u[1:]), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i + 1])

                # scaling = get_scaling(n[from_to[i, j, 0]:from_to[i, j, 1]], pr_m[from_to[i, j, 0]:from_to[i, j, 1]],
                #                       pr_unc[from_to[i, j, 0]:from_to[i, j, 1]])
                # print(f'Scaling pL {dist}, {t}: ', scaling)
                # vals[j, i] = scaling[0][1]
                # ax1.plot(n[1:], function(n[1:], scaling[0][0], scaling[0][1]), color=coloring[i+1])
                # inset_ax.errorbar(1 - t, scaling[0][1], yerr=scaling[1][1], markersize=5,
                #                   linestyle=styles[j], marker=markers[j], color=coloring[i + 1])
        '''
        for i, d in enumerate(distances[1:]):
            # print(1 - np.array(thresholds[1:]))
            # print(vals[:, i])
            inset_ax.plot(1 - np.array(thresholds[1:]), vals[:, i], marker='None', linestyle='dotted', linewidth=2,
                          color=coloring[i + 1])
        '''
        plt.grid()
        ax1.set_xlabel("$p$", fontsize=14)
        ax1.set_ylabel("$p_L$", fontsize=14)
        # ax1.set_xscale("log")
        # ax1.set_yscale("log")
        plt.suptitle("code capacity, depolarizing")
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i + 1]}')
                        for i, color in enumerate(coloring[1:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                               label=f'c={1 - thresholds[i + 1]}')
                        for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax1.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax1.set_xlim([0.019, 0.25])
        ax1.set_ylim([1e-5, 0.3])
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        # plt.savefig('plots/log_error_rate_threshold_depolarizing_9.svg')
        # plt.show()

        # fig, ax = plt.subplots()
        # inset_ax = ax2.inset_axes([0.37, 0.1, 0.3, 0.2])
        # inset_ax.set_title(r"Scaling $\alpha p^{t+1}$", fontsize=11)
        # inset_ax.set_xlabel('c', fontsize=11)
        # inset_ax.set_yticks((0.172, 0.177))
        # inset_ax.set_ylabel(r"$t$", fontsize=11)
        # inset_ax.tick_params(axis='both', labelsize=11)
        from_to = np.array([[[1, 5], [1, 5], [1, 4], [1, 3]],
                            [[2, 5], [1, 5], [1, 4], [1, 4]],
                            [[2, 5], [2, 5], [2, 4], [2, 4]]])
        upto = [3, 4, 5, 5]
        vals = np.zeros((len(thresholds[1:]), len(distances[1:])))
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances[1:]):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = np.array(list(map(lambda x: x[3], pr)))

                ax2.errorbar(n[1:], pr_m[1:], yerr=(np.sqrt(pr_m[1:] / 100000)), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i + 1])

                # scaling = get_scaling(n[from_to[i, j, 0]:from_to[i, j, 1]], pr_m[from_to[i, j, 0]:from_to[i, j, 1]],
                #                       np.sqrt(pr_m[from_to[i, j, 0]:from_to[i, j, 1]] / 100000))
                # print(f'Scaling {dist}, {t}: ', scaling)
                # vals[j, i] = scaling[0][1]
                # ax2.plot(n[1:], function(n[1:], scaling[0][0], scaling[0][1]), color=coloring[i+1])
                # inset_ax.errorbar(1 - t, scaling[0][1], yerr=scaling[1][1], markersize=5,
                #                   linestyle=styles[j], marker=markers[j], color=coloring[i + 1])
        '''
        for i, d in enumerate(distances[1:]):
            inset_ax.plot(1 - np.array(thresholds[1:]), vals[:, i], marker='None', linestyle='dotted', linewidth=2,
                          color=coloring[i + 1])
        '''
        # noise, abort_prob = get_abort_probability(thresholds, 5)
        # for i in range(np.shape(abort_prob)[0]):
        #     ax.plot(noise, abort_prob[i], color='black')
        plt.grid()
        ax2.set_xlim(0.019, 0.3)
        ax2.set_ylim(1e-4, 1)
        ax2.set_xlabel("$p$", fontsize=14)
        ax2.set_ylabel(r'$p_{abort}$', fontsize=14)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i + 1]}')
                        for i, color in enumerate(coloring[1:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                               label=f'c={1 - thresholds[i + 1]}')
                        for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax2.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax1.grid()
        ax2.grid()
        plt.savefig('plots/log_error_rate_abort_depolarizing_9_scaling.svg')
        plt.show()
    elif task == 5:  # Plot attention
        # Create an input example
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
        else:
            model = model.load()
        model.eval()

        type = 'double'
        if type == 'zero':
            x = torch.as_tensor([[0, 0, 0, 0, 0, 0, 0, 0]], device=device)
            log = torch.as_tensor([[0, 0]], device=device)
        elif type == 'single':
            x = torch.as_tensor([[0, 0, 0, 0, 0, 0, 1, 1]], device=device)
            log = torch.as_tensor([[1, 0]], device=device)
        elif type == 'double':
            x = torch.as_tensor([[0, 1, 0, 0, 0, 1, 1, 0]], device=device)
            log = torch.as_tensor([[1, 0]], device=device)
        else:
            raise ValueError(f'Type {type} no valid error type specification.')
        log = torch.concatenate((torch.as_tensor([[2]], device=device), log[:, :-1]), dim=1)

        x = model.input_repr(x)
        x = model.res_net(x)

        self_attn_weights = []
        for layer in model.encoder.layers:
            # Self-attention in the decoder
            logits, weights = layer.self_attn(x, x, x,
                                              attn_mask=None,
                                              key_padding_mask=None,
                                              is_causal=False,
                                              need_weights=True,
                                              average_attn_weights=False)
            x = layer.norm1(x + logits)
            self_attn_weights.append(weights)
            x = layer.norm2(x + layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))
            x = layer._conv_block(x)

        log = model.logical_input_repr(log)
        log = model.res_net(log)

        cross_attention_weights = []

        seq_len = log.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
        mask = mask.masked_fill(mask == 0, -1e12)
        mask = mask.masked_fill(mask == 1, 0.0)
        mask = mask.to(model.device)

        # Pass through the encoder
        for layer in model.decoder.layers:
            # Self-attention in the decoder
            log = layer.norm1(log + layer.self_attn(log, log, log,
                                                    attn_mask=mask,
                                                    key_padding_mask=None,
                                                    is_causal=False,
                                                    need_weights=False)[0])
            logits, weights = layer.multihead_attn(log, x, x,
                                                   attn_mask=None,
                                                   key_padding_mask=None,
                                                   is_causal=False,
                                                   need_weights=True,
                                                   average_attn_weights=False)
            cross_attention_weights.append(weights)
            log = layer.norm2(log + logits)
            log = layer.norm3(log + layer.linear2(layer.dropout(layer.activation(layer.linear1(log)))))

        # Visualize self-attention weights for the first layer, first head
        layer, batch, head = (0, 0, 0)
        attn_weights = self_attn_weights[layer][batch, head].cpu().detach().numpy()  # Layer 0, Batch 0, Head 0
        output = model.fc_out(log)
        output = F.sigmoid(output).squeeze(2)
        print(output)
        plt.figure(figsize=(10, 10))
        sns.heatmap(attn_weights, cmap="viridis", annot=True, cbar=True)
        plt.title(f"Self Attention Weights - Layer {layer}, Head {head}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.tight_layout()
        plt.savefig(f'plots/Self_attention_{layer}_{type}.svg')
        plt.show()

        layer, batch, head = (2, 0, 0)
        attn_weights = self_attn_weights[layer][batch, head].cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        sns.heatmap(attn_weights, cmap="viridis", annot=True, cbar=True)
        plt.title(f"Self Attention Weights - Layer {layer}, Head {head}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.tight_layout()
        plt.savefig(f'plots/Self_attention_{layer}_{type}.svg')
        plt.show()

        layer, batch, head = (0, 0, 0)
        attn_weights = cross_attention_weights[layer][batch, head].cpu().detach().numpy()  # Layer 0, Batch 0, Head 0
        output = model.fc_out(log)
        output = F.sigmoid(output).squeeze(2)
        plt.figure(figsize=(10, 4))
        sns.heatmap(attn_weights, cmap="viridis", annot=True, cbar=True)
        plt.title(f"Attention Weights - Layer {layer}, Head {head}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.tight_layout()
        plt.savefig(f'plots/Cross_attention_{layer}_{type}.svg')
        plt.show()

        layer, batch, head = (2, 0, 0)
        attn_weights = cross_attention_weights[layer][
            batch, head].cpu().detach().numpy()  # Select layer 1, batch 0, head 2
        plt.figure(figsize=(10, 4))
        sns.heatmap(attn_weights, cmap="viridis", annot=True, cbar=True)
        plt.title(f"Attention Weights - Layer {layer}, Head {head}")
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")
        plt.tight_layout()
        plt.savefig(f'plots/Cross_attention_{layer}_{type}.svg')
        plt.show()
    elif task == 8:  # Statistic about distribution of syndromes for increasing noise
        noise = 0.4
        distance = 3
        data_size = 1000000
        data = (DepolarizingSurfaceData(distance=distance,
                                        noise=noise,
                                        name=f'{iteration}_{distance}_{noise}',
                                        load=False,
                                        device=device,
                                        logical=logical)
                .initialize(data_size)
                .get_syndromes())
        data = data[:, :, 0].cpu().numpy()
        data = data[:, :-2]
        # Convert binary sequences to decimal numbers for efficient counting
        data_decimal = np.zeros(data_size)
        for i in tqdm(range(data_size)):
            data_decimal[i] = binary_to_decimal(data[i])  # sample) for sample in data]

        # Count the occurrences of each unique sequence
        sequence_counts = Counter(data_decimal)

        # Prepare data for plotting
        unique_sequences = list(sequence_counts.keys())
        frequencies = list(sequence_counts.values())

        # Sort by sequence frequency for a clean plot
        sorted_indices = torch.argsort(torch.tensor(frequencies), descending=True)
        sorted_sequences = [unique_sequences[i] for i in sorted_indices]
        sorted_frequencies = [frequencies[i] / data_size for i in sorted_indices]
        sorted_frequencies = np.array(sorted_frequencies)

        # Plot the histogram
        # plt.figure(figsize=(10, 6))
        # bax = brokenaxes(ylims=((0, 0.05), (sorted_frequencies[0] - 0.01, sorted_frequencies[0] + 0.01)), hspace=0.05)

        # If we were to simply plot pts, we'd lose most of the interesting
        # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
        # into two portions - use the top (ax1) for the outliers, and the bottom
        # (ax2) for the details of the majority of our data
        if sorted_frequencies[0] > 0.07:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [20, 55]})
            fig.subplots_adjust(hspace=0.02)  # adjust space between Axes

            # plot the same data on both Axes
            ax1.bar(np.arange(len(sorted_frequencies)), sorted_frequencies)
            ax2.bar(np.arange(len(sorted_frequencies)), sorted_frequencies)

            # zoom-in / limit the view to different portions of the data
            ax1.set_ylim(sorted_frequencies[0] - 0.01, sorted_frequencies[0] + 0.01)  # outliers only
            ax2.set_ylim(0, .055)  # most of the data

            # hide the spines between ax and ax2
            ax1.spines.bottom.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.tick_params(labelbottom=False, bottom=False)
            # ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            # Now, let's turn towards the cut-out slanted lines.
            # We create line objects in axes coordinates, in which (0,0), (0,1),
            # (1,0), and (1,1) are the four corners of the Axes.
            # The slanted lines themselves are markers at those locations, such that the
            # lines keep their angle and position, independent of the Axes size or scale
            # Finally, we need to disable clipping.

            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
            ax2.set_xlabel('# occurring syndromes')
            ax2.set_ylabel('Relative Frequency')
            ax1.set_title(f'Distribution of Syndromes in Training Set: p={noise}, d={distance}')

        else:
            if distance == 3:
                plt.bar(range(len(sorted_frequencies)), sorted_frequencies, color='blue', alpha=0.7)
            else:
                plt.plot(range(len(sorted_frequencies))[1:], sorted_frequencies[1:], color='blue', alpha=0.7)
            plt.ylim(0, 0.07)
            plt.xlabel('# occurring syndromes')
            plt.ylabel('Relative Frequency')
            plt.title(f'Distribution of Syndromes in Training Set: p={noise}, d={distance}')

        # plt.xlabel('Unique Sequences (Encoded as Decimal)', fontsize=12)
        # plt.xticks(ticks=range(len(sorted_frequencies)), labels=sorted_sequences, rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'plots/syndrome_distribution_d{distance}_n{noise}.svg')
        plt.show()
    else:
        raise ValueError(f'Unsupported task: {task}')


def run_phenomenological(task, noise, distance, model_type, lr, device, num_epochs, batch_size, model_dict, data_size,
                         pretrained_model=None):
    noise_vals = [2e-3, 5e-3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1,
                  0.12]
    distances = [3, 5, 7]
    thresholds = [0.5, 0.3, 0.15, 0.05]

    assert noise in noise_vals
    assert distance in distances

    iteration = 'ph'
    eval_what = 'log_error_rate' if task == 400 else 'val_loss'
    mode = 'phenomenological'
    pretrained_model = iteration + '_' + str(distance) + '_' + str(noise_vals.index(noise) - 1)

    print(f'p={noise}, d={distance}, iteration={iteration}')

    # Specify model
    try:
        checkpoint = torch.load(f'data/checkpoint_{iteration}_{distance}_{noise}', map_location=device)
    except FileNotFoundError:
        checkpoint = None

    if model_type == 'qectransformer':
        model = RQecTransformer(name=iteration + '_' + str(distance) + '_' + str(noise), distance=distance,
                                readout=readout,
                                penc_type='fixed', every_round=False, **model_dict).to(
            device)
    else:
        model = RQecVT(name=iteration + '_' + str(distance) + '_' + str(noise), distance=distance,
                       readout=readout, patch_distance=patch_distance, **model_dict)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    data = PhenomenologicalSurfaceData(distance=distance,
                                       noise=noise,
                                       name=iteration + '_' + str(distance) + '_' + str(noise),
                                       load=False,
                                       device=device,
                                       only_syndromes=False,
                                       every_round=False)
    if task == 1:
        if not online_learning:
            try:
                print('Generate dataset.')
                data = (data
                        .initialize(data_size)
                        .save())
                # data = (data
                #        .load())
                # print('Dataset loaded.')
            except FileNotFoundError:
                print('Generate dataset.')
                data = (data
                        .initialize(data_size)
                        .save())
            train, val = data.get_train_val_data()  # default ratio 80/20
        else:
            print('Online learning.')

        try:
            model = model.load()
        except FileNotFoundError:
            if checkpoint is not None:
                print('Load from checkpoint.')
            elif load_pretrained:
                try:
                    print('Load pretrained model.')
                    # Load weights from pretrained smaller model
                    pretrained_state_dict = torch.load("data/net_{}.pt".format(pretrained_model),
                                                       map_location=torch.device('cpu'))
                    # Interpolate or copy weights to larger model
                    for name, param in pretrained_state_dict.items():
                        if name in model.state_dict():
                            if len(param.size()) == 1:
                                model.state_dict()[name][:param.size(0)] = param
                            else:
                                model.state_dict()[name][:param.size(0), :param.size(1)] = param
                except FileNotFoundError:
                    try:
                        print('Load pretrained from checkpoint.')
                        pretrained_checkpoint = torch.load(f'data/checkpoint_{pretrained_model}',
                                                           map_location=torch.device('cpu'))
                        pretrained_state_dict = pretrained_checkpoint["model"]
                        # Interpolate or copy weights to larger model
                        for name, param in pretrained_state_dict.items():
                            if name in model.state_dict():
                                if len(param.size()) == 1:
                                    model.state_dict()[name][:param.size(0)] = param
                                else:
                                    model.state_dict()[name][:param.size(0), :param.size(1)] = param
                    except FileNotFoundError:
                        print('Load new model. File not found.')
                    pass
            else:
                print('Load new model.')
                pass
        except RuntimeError:
            if checkpoint is not None:
                print('Load from checkpoint.')
            else:
                print('Load new model. Runtime error.')
            pass

        # Train
        if online_learning:
            model = online_training(model, data, make_optimizer(lr), device, epochs=num_epochs, batch_size=batch_size,
                                    num_data=data_size, from_checkpoint=from_checkpoint, checkpoint=checkpoint)
        else:
            assert train is not None and val is not None
            model = training_loop(model, train, val, make_optimizer(lr), device, epochs=num_epochs,
                                  batch_size=batch_size,
                                  mode=mode, activate_scheduler=True, load_pretrained=load_pretrained)

        # Final forward pass to get val loss including estimates
        checkpoint = torch.load(f'data/checkpoint_{iteration}_{distance}_{noise}', map_location=device)
        estimate_ci(checkpoint=checkpoint, model=model, noise=noise, device=device, mode=mode,
                    data=data, iteration=iteration, distance=distance)
    elif task == 4:  # Evaluate error rate as decoder
        data = PhenomenologicalSurfaceData(distance=distance,
                                           noise=noise,
                                           name='eval',
                                           load=False,
                                           device=device,
                                           only_syndromes=False,
                                           every_round=False)
        log_error_rate(checkpoint=checkpoint, model=model, thresholds=thresholds,
                       noise=noise, device=device, mode=mode,
                       data=data, iteration=iteration, distance=distance)
    elif task == 100:
        # Merge dictionaries
        merge_dictionaries(distances=distances, noise_vals=noise_vals, eval_what='val_loss', iteration=iteration)

        # Plot
        fig, ax = plt.subplots()
        coloring = ['blue', 'red', 'green', 'yellow']

        for i, dist in enumerate(distances):
            dict = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())  # noises
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5, linestyle='dotted', color=coloring[i])

        '''
        # To estimate conditional collapse
        p_lambda = np.zeros((len(noises), 4))
        patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        for k, p in enumerate(tqdm(noises)):
            sz = 10000
            data = (CircuitLevelSurfaceData(distance=3,
                                            noise=p,
                                            name=f'{iteration}_{distance}_{noise}',
                                            load=False,
                                            device=torch.device('cpu'),
                                            every_round=False)
                    .initialize(sz)
                    .get_syndromes())
            frequencies = torch.zeros(len(patterns), dtype=torch.int32)
            for i, pattern in enumerate(patterns):
                frequencies[i] = torch.sum(torch.all(data[:, -2:] == pattern, dim=1))
            frequencies = frequencies / sz
            print(frequencies)
            p_lambda[k] = 1 + torch.sum(frequencies * np.log(frequencies + 1e-12)).numpy() / np.log(2)
        # np.savetxt('plambda.txt', p_lambda)
        '''
        # p_lambda = np.loadtxt('plambda.txt')
        # plt.plot(noises, p_lambda, label=r'$p_3(\lambda | s) = p_3(\lambda)$', color='cyan')

        ax.set_xlabel('noise probability p')

        if eval_what == 'result' or eval_what == 'val_loss':
            ax.set_ylabel(r'$I / \log 2$')
        elif eval_what == 'log_error_rate':
            ax.set_ylabel(r'$p_L$')
        else:
            raise ValueError(f'{eval_what} is not supported.')

        if eval_what == 'val_loss':
            with h5py.File("data/surface_code_distance_three_pheno_noise-2.h5", "r") as f:
                # List all groups/datasets
                # print("Keys:", list(f.keys()))

                # Access a specific dataset
                ps = f['ps'][:]
                ci = f['ci'][:]

                ax.plot(ps, ci, marker='v', markersize=4, linestyle='dashed',
                        label='d=3 num', color='black')

                # Print dataset shape and type
                # print(f"Dataset ps: shape {ps.shape}, dtype {ps.dtype}")

        inset = zoomed_inset_axes(ax, zoom=2, bbox_to_anchor=(0.385, 0.49),  # Adjust position, previous (0.4, 0.45)
                                  bbox_transform=ax.transAxes)

        with h5py.File("data/surface_code_distance_three_pheno_noise-2.h5", "r") as f:
            # List all groups/datasets
            print("Keys:", list(f.keys()))

            # Access a specific dataset
            ps = f['ps'][:]
            ci = f['ci'][:]

            inset.plot(ps, ci, marker='v', markersize=4, linestyle='dashed',
                       label='d=3 num', color='black')

        for i, dist in enumerate(distances):
            dict = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())  # noises
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
            inset.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted',
                           label='d={}'.format(dist), color=coloring[i])
        # Zoom into a specific region in the inset
        inset.set_xlim(0.027, 0.046)  # Adjust as needed
        inset.set_ylim(0.55, 0.87)  # Adjust as needed

        # Optional: Add labels and ticks
        # inset.set_xticks((0.003, 0.00375, 0.0045))
        inset.tick_params(labelsize=8)
        # inset.set_title("Inset", fontsize=10)

        # Add a rectangle to show the zoomed-in area in the main plot
        ax.indicate_inset_zoom(inset, edgecolor="black")

        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black")

        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=['v', 'o'][i], markersize=3, linestyle=ls,
                               label=f'{['exact', 'NN'][i]}')
                        for i, ls in enumerate(['dashed', 'dotted'])]

        legend_handles = color_legend + style_legend  # + auto_handles
        ax.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.suptitle(f'{mode}')
        plt.tight_layout()

        # Get crossings
        dict1 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 3))
        n = list(dict1.keys())  # noises
        dict2 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='ci', p0=[0.03, 0.06])
        print(t)

        ax.axvline(t[0], ymin=0, ymax=1, color='silver', linewidth=1)
        inset.axvline(t[0], ymin=0, ymax=1, color='silver', linewidth=1)
        plt.savefig(f'plots/CI_{mode}.svg')
        plt.show()

        # Get crossings
        dict1 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 3))
        n = list(dict1.keys())  # noises
        dict2 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='ci', p0=[0.02, 0.05])
        print(t)
    elif task == 400:
        # get dictionaries
        merge_dictionaries(distances, noise_vals, 'log_error_rate', iteration, thresholds)

        # Plots
        # NN decoder without post-selection, linear
        fig, ax = plt.subplots()
        # for i, dist in enumerate(distances):
        coloring = ['blue', 'red', 'green', 'black', 'powderblue', 'orange']
        markers = ['o', 'x', 'v', '^']
        styles = ['dashed', 'dotted', 'dashdot', 'dotted']

        # MWPM
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_mwpm.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='v', markersize=5, linestyle='dashed',
                        label='d={}_mwpm'.format(dist), color=coloring[i])

        t = 0.5
        for i, dist in enumerate(distances):
            dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5,
                        linestyle='solid',
                        label='d={0}'.format(dist), color=coloring[i])

        # plt.grid()
        ax.set_xlabel("$p$")
        ax.set_ylabel("$p_L$")
        plt.suptitle(f'{mode}')
        plt.xlim([0., 0.1])

        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], marker=['v', 'o'][i], markersize=4, color='black', linestyle=ls,
                               label=f'{['MWPM', 'NN'][i]}')
                        for i, ls in enumerate(['dashed', 'solid'])]

        legend_handles = color_legend + style_legend
        plt.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        plt.savefig(f'plots/log_error_rate_{mode}.svg')
        plt.show()

        # find crossing
        dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 3, 0.5))
        n = list(dict1.keys())
        dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 5, 0.5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='pl', p0=[0.03, 0.05])
        print('Crossing 3,5: ', t)

        # scaling error rate
        fig, ax = plt.subplots()
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_mwpm.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='v', markersize=5, linestyle='dashed',
                        label='d={}_mwpm'.format(dist), color=coloring[i])
        t = 0.5
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
            if dist == 7:
                n = n[1:]
                pr_m = pr_m[1:]
                pr_u = pr_u[1:]
                pr_d = pr_d[1:]
            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5,
                        linestyle='solid',
                        label='d={0}'.format(dist), color=coloring[i])

        plt.grid()
        ax.set_xlabel("$p$")
        ax.set_ylabel("$p_L$")
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.suptitle(f'{mode}')
        plt.xlim([0.0019, 0.1])
        plt.ylim([0.000001, 0.8])

        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=['v', 'o'][i], markersize=4, linestyle=ls,
                               label=f'{['MWPM', 'NN'][i]}')
                        for i, ls in enumerate(['dashed', 'solid'])]

        legend_handles = color_legend + style_legend
        plt.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig(f'plots/log_error_rate_{mode}_scaling.svg')
        plt.show()

        distances = distances[:-1]

        # abort scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                ax1.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

        # plt.grid()
        ax1.set_xlabel("$p$")
        ax1.set_ylabel("$p_L$")
        plt.suptitle(f'{mode}')
        ax1.set_xlim([1.9e-3, 0.06])
        ax1.set_ylim([1e-5, 0.11])

        # inset_ax = ax2.inset_axes([0.67, 0.1, 0.3, 0.2])
        # inset_ax.set_title(r"Scaling $\alpha p^{t+1}$", fontsize=11)
        # inset_ax.set_xlabel('c', fontsize=11)
        # inset_ax.set_yticks((0.172, 0.177))
        # inset_ax.set_ylabel(r"$t$", fontsize=11)
        # inset_ax.tick_params(axis='both', labelsize=11)
        from_to = np.array([[[1, 4], [1, 5], [1, 4]],
                            [[2, 6], [2, 5], [2, 7]]])
        vals = np.zeros((len(thresholds[1:]), len(distances)))

        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = np.array(list(map(lambda x: x[3], pr)))

                ax2.errorbar(n, pr_m, yerr=(np.sqrt(np.array(pr_m) / 100000)), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

                # scaling = get_scaling(n[from_to[i, j, 0]:from_to[i, j, 1]], pr_m[from_to[i, j, 0]:from_to[i, j, 1]],
                #                       np.sqrt(pr_m[from_to[i, j, 0]:from_to[i, j, 1]] / 100000))
                # print(f'Scaling {dist}, {t}: ', scaling)
                # vals[j, i] = scaling[0][1]
                # ax2.plot(n[1:], function(n[1:], scaling[0][0], scaling[0][1]), color=coloring[i+1])
                # inset_ax.errorbar(1 - t, scaling[0][1], yerr=scaling[1][1], markersize=5,
                #                   linestyle=styles[j], marker=markers[j], color=coloring[i])
        '''
        for i, d in enumerate(distances):
            inset_ax.plot(1 - np.array(thresholds[1:]), vals[:, i], marker='None', linestyle='dotted', linewidth=2,
                          color=coloring[i])
        '''
        # noise, abort_prob = get_abort_probability(thresholds, 5)

        ax2.set_xlim(1.9e-3, 0.06)
        ax2.set_ylim(5e-5, 1)

        ax2.set_xlabel("$p$")
        ax2.set_ylabel(r'$p_{abort}$')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                               label=f'c={1 - thresholds[i + 1]}')
                        for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax1.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax2.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig(f'plots/log_error_rate_abort_{mode}_scaling.svg')
        plt.show()

        # THRESHOLD LINEAR

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                ax1.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

        ax1.set_xlabel("$p$")
        ax1.set_ylabel("$p_L$")
        plt.suptitle(f'{mode}')

        ax1.set_xlim([0, 0.06])
        ax1.set_ylim([0, 0.11])

        crossings = np.zeros((3, len(thresholds[1:])))
        crossings[0] = 1 - np.array(thresholds[1:])

        upto = [4, 4, 4]
        go = [1, 0]
        # fig, ax = plt.subplots()
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = np.array(list(map(lambda x: x[3], pr)))

                ax2.errorbar(n, pr_m, yerr=(np.sqrt(pr_m / 100000)), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

                # scaling = get_scaling(n[go[i]:upto[j] + 1], pr_m[go[i]:upto[j] + 1],
                #                       np.sqrt(pr_m[go[i]:upto[j] + 1] / 100000))
                # print(f'Scaling {dist}, {t}: ', scaling)

                # crossings for p_abort
                # crossings for p_abort
            dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 3, t))
            n = list(dict1.keys())
            dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 5, t))

            crossing = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='abort', p0=[0.02, 0.05])
            print(f'For c={t} crossing:', crossing)

            crossings[1, j] = crossing[0]
            crossings[2, j] = crossing[1]

        ax2.set_xlim(0, 0.06)
        ax2.set_ylim(0, 1)

        inset_ax = ax2.inset_axes([0.15, 0.75, 0.3, 0.2])  # or loc=1
        for i in range(len(thresholds[1:])):
            inset_ax.errorbar(crossings[0, i], crossings[1, i], yerr=crossings[2, i],
                              linestyle='None',
                              marker=markers[i], color='black', markersize=4)
        inset_ax.set_title("Threshold", fontsize=11)
        inset_ax.set_xlabel('c', fontsize=11)
        inset_ax.set_ylabel(r"$p_{th}$", fontsize=11)
        inset_ax.tick_params(axis='both', labelsize=11)

        ax2.set_xlabel("$p$")
        ax2.set_ylabel(r'$p_{abort}$')
        # plt.suptitle(f'{mode}')
        plt.tight_layout()
        # ax2.legend()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                               label=f'c={1 - thresholds[i + 1]}')
                        for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax1.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax2.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig(f'plots/log_error_rate_abort_{mode}.svg')
        plt.show()
    elif task == 8:  # Loss plots
        # File path (change if needed)
        try:
            file_path = "data/output_Qec_ST-Transformer_train_ph_40.txt"
        except FileNotFoundError:
            print('File not found. Save loss values first.')
            exit(-1)

        losses = []
        val_losses = []
        lrs = []

        loss_pattern = re.compile(r"Epoch (\d+), Loss: ([\d.eE+-]+)")
        val_loss_pattern = re.compile(r"Epoch (\d+), Validation Loss: ([\d.eE+-]+)")
        lr_pattern = re.compile(r"\[([\d.eE+-]+)]")  # Matches learning rate in square brackets

        # Read and extract values
        with open(file_path, "r") as f:
            for line in f:
                loss_match = loss_pattern.search(line)
                val_loss_match = val_loss_pattern.search(line)
                lr_match = lr_pattern.search(line)

                if loss_match:
                    losses.append(float(loss_match.group(2)))

                if val_loss_match:
                    val_losses.append(float(val_loss_match.group(2)))

                if lr_match:
                    lrs.append(float(lr_match.group(1)))

        # Convert to numpy arrays
        losses = np.array(losses)
        val_losses = np.array(val_losses)
        lrs = np.array(lrs)

        fig, ax = plt.subplots()

        ax.plot(losses, label='Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="black")
        ax.legend()
        plt.tight_layout(pad=2.0)
        plt.savefig('plots/loss.pdf')
        plt.show()


def run_circuit_level(task, noise, distance, model_type, lr, device, num_epochs, batch_size, model_dict, data_size,
                      pretrained_model=None):
    noise_vals = [1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 0.01, 0.014]
    distances = [3, 5]
    thresholds = [0.5, 0.3, 0.15, 0.05]

    assert noise in noise_vals
    assert distance in distances

    iteration = 'cl'
    eval_what = 'log_error_rate' if task == 400 else 'val_loss'
    mode = 'circuit-level'
    pretrained_model = iteration + '_' + str(distance) + '_' + str(noise_vals.index(noise) - 1)

    print(f'p={noise}, d={distance}, iteration={iteration}')

    # Specify model
    try:
        checkpoint = torch.load(f'data/checkpoint_{iteration}_{distance}_{noise}', map_location=device)
    except FileNotFoundError:
        checkpoint = None

    if model_type == 'qectransformer':
        model = RQecTransformer(name=iteration + '_' + str(distance) + '_' + str(noise), distance=distance,
                                readout=readout,
                                penc_type='fixed', every_round=False, **model_dict).to(
            device)
    else:
        model = RQecVT(name=iteration + '_' + str(distance) + '_' + str(noise), distance=distance,
                       readout=readout, patch_distance=patch_distance, **model_dict)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    data = CircuitLevelSurfaceData(distance=distance,
                                   noise=noise,
                                   name=iteration + '_' + str(distance) + '_' + str(noise),
                                   load=False,
                                   device=device,
                                   only_syndromes=False,
                                   every_round=False)

    if task == 1:
        if not online_learning:
            try:
                print('Generate dataset.')
                data = (data
                        .initialize(data_size)
                        .save())
                # data = (data
                #        .load())
                # print('Dataset loaded.')
            except FileNotFoundError:
                print('Generate dataset.')
                data = (data
                        .initialize(data_size)
                        .save())
            train, val = data.get_train_val_data()  # default ratio 80/20
        else:
            print('Online learning.')

        try:
            model = model.load()
        except FileNotFoundError:
            if checkpoint is not None:
                print('Load from checkpoint.')
            elif load_pretrained:
                try:
                    print('Load pretrained model.')
                    # Load weights from pretrained smaller model
                    pretrained_state_dict = torch.load("data/net_{}.pt".format(pretrained_model),
                                                       map_location=torch.device('cpu'))
                    # Interpolate or copy weights to larger model
                    for name, param in pretrained_state_dict.items():
                        if name in model.state_dict():
                            if len(param.size()) == 1:
                                model.state_dict()[name][:param.size(0)] = param
                            else:
                                model.state_dict()[name][:param.size(0), :param.size(1)] = param
                except FileNotFoundError:
                    try:
                        print('Load pretrained from checkpoint.')
                        pretrained_checkpoint = torch.load(f'data/checkpoint_{pretrained_model}',
                                                           map_location=torch.device('cpu'))
                        pretrained_state_dict = pretrained_checkpoint["model"]
                        # Interpolate or copy weights to larger model
                        for name, param in pretrained_state_dict.items():
                            if name in model.state_dict():
                                if len(param.size()) == 1:
                                    model.state_dict()[name][:param.size(0)] = param
                                else:
                                    model.state_dict()[name][:param.size(0), :param.size(1)] = param
                    except FileNotFoundError:
                        print('Load new model. File not found.')
                    pass
            else:
                print('Load new model.')
                pass
        except RuntimeError:
            if checkpoint is not None:
                print('Load from checkpoint.')
            else:
                print('Load new model. Runtime error.')
            pass

        if online_learning:
            model = online_training(model, data, make_optimizer(lr), device, epochs=num_epochs, batch_size=batch_size,
                                    num_data=data_size, from_checkpoint=from_checkpoint, checkpoint=checkpoint)
        else:
            assert train is not None and val is not None
            model = training_loop(model, train, val, make_optimizer(lr), device, epochs=num_epochs,
                                  batch_size=batch_size,
                                  mode=mode, activate_scheduler=True, load_pretrained=load_pretrained)
        checkpoint = torch.load(f'data/checkpoint_{iteration}_{distance}_{noise}', map_location=device)
        estimate_ci(checkpoint=checkpoint, model=model, noise=noise, device=device, mode=mode,
                    data=data, iteration=iteration, distance=distance)
    elif task == 4:  # Evaluate error rate as decoder
        data = CircuitLevelSurfaceData(distance=distance,
                                       noise=noise,
                                       name='eval',
                                       load=False,
                                       device=device,
                                       only_syndromes=False,
                                       every_round=False)
        log_error_rate(checkpoint=checkpoint, model=model, thresholds=thresholds,
                       noise=noise, device=device, mode=mode,
                       data=data, iteration=iteration, distance=distance)
    elif task == 100:
        # Merge dictionaries
        merge_dictionaries(distances=distances, noise_vals=noise_vals, eval_what='val_loss', iteration=iteration)

        # Plot
        fig, ax = plt.subplots()
        coloring = ['blue', 'red', 'green', 'yellow']

        for i, dist in enumerate(distances):
            # if dist > 5:
            #    break
            dict = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())  # noises
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5, linestyle='dotted', color=coloring[i])

        '''
        noises = np.arange(0.0, 0.03, 0.001)
        p_lambda = np.zeros((len(noises), 4))
        patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        for k, p in enumerate(tqdm(noises)):
            sz = 10000
            data = (CircuitLevelSurfaceData(distance=3,
                                            noise=p,
                                            name=f'{iteration}_{distance}_{noise}',
                                            load=False,
                                            device=torch.device('cpu'),
                                            every_round=False)
                    .initialize(sz)
                    .get_syndromes())
            frequencies = torch.zeros(len(patterns), dtype=torch.int32)
            for i, pattern in enumerate(patterns):
                frequencies[i] = torch.sum(torch.all(data[:, -2:] == pattern, dim=1))
            frequencies = frequencies / sz
            print(frequencies)
            p_lambda[k] = 1 + torch.sum(frequencies * np.log(frequencies + 1e-12)).numpy() / np.log(2)
        # np.savetxt('plambda.txt', p_lambda)
        '''
        # p_lambda = np.loadtxt('plambda.txt')
        # plt.plot(noises, p_lambda, label=r'$p_3(\lambda | s) = p_3(\lambda)$', color='cyan')
        ax.set_xlabel('noise probability p')
        # ax.set_ylabel(r'$\sum_s p(s) \sum_\lambda p(\lambda | s) \log p(\lambda | s)$')
        # ax.set_ylabel(r'$\sum_s p(s) \sum_\lambda p(\lambda | s)^2$')
        if eval_what == 'result' or eval_what == 'val_loss':
            ax.set_ylabel(r'$I / \log 2$')
        elif eval_what == 'log_error_rate':
            ax.set_ylabel(r'$p_L$')
        else:
            raise ValueError(f'{eval_what} is not supported.')

        inset = zoomed_inset_axes(ax, zoom=2, bbox_to_anchor=(0.405, 0.42),  # Adjust position, previous (0.4, 0.45)
                                  bbox_transform=ax.transAxes)

        for i, dist in enumerate(distances):
            # if dist > 5:
            #    break
            dict = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())  # noises
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
            inset.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted',
                           label='d={}'.format(dist), color=coloring[i])

        # Zoom into a specific region in the inset
        inset.set_xlim(0.0028, 0.0048)  # Adjust as needed
        inset.set_ylim(0.8, 1)  # Adjust as needed

        # Optional: Add labels and ticks
        inset.set_xticks((0.003, 0.00375, 0.0045))
        inset.tick_params(labelsize=8)
        # inset.set_title("Inset", fontsize=10)

        # Add a rectangle to show the zoomed-in area in the main plot
        ax.indicate_inset_zoom(inset, edgecolor="black")

        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black")

        '''
        # Define inset
        inset = zoomed_inset_axes(ax, zoom=2, bbox_to_anchor=(0.4, 0.45),  # Adjust position
                                  bbox_transform=ax.transAxes)
        for i, dist in enumerate(distances):
            dict = torch.load("data/result_{0}_{1}.pt".format(iteration, dist))
            n = list(dict.keys())  # noises
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
        
            # plotting
            if dist == 5:
                inset.errorbar(n[:-4], pr_m[:-4], yerr=(pr_d[:-4], pr_u[:-4]), marker='o', markersize=4, linestyle='dotted',
                               color=coloring[i])
        
            else:
                inset.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted', color=coloring[i])
        
        noises = np.arange(0.0, 0.4, 0.01)
        for i, d in enumerate([3]):  # , 5]:
            g_stabilizer = np.loadtxt('code/stabilizer_' + 'rsur' + '_d{}_k{}'.format(d, 1))
            logical_opt = np.loadtxt('code/logical_' + 'rsur' + '_d{}_k{}'.format(d, 1))
            pr, entr, var = get_pr(d, noises, g_stabilizer, logical_opt, d ** 2)
            inset.plot(noises, pr, label='d={}'.format(d), color=coloring[i])
        analytical_5 = np.loadtxt('analytical/analytical_d5')
        inset.plot(noises, analytical_5, label='d=5', color=coloring[1])
        
        # Zoom into a specific region in the inset
        inset.set_xlim(0.16, 0.22)  # Adjust as needed
        inset.set_ylim(0.55, 0.7)  # Adjust as needed
        
        # Optional: Add labels and ticks
        inset.tick_params(labelsize=8)
        # inset.set_title("Inset", fontsize=10)
        
        # Add a rectangle to show the zoomed-in area in the main plot
        ax.indicate_inset_zoom(inset, edgecolor="black")
        
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="black")
        '''

        # noises = np.arange(0.0, 0.12, 0.001)
        '''
        p_lambda = np.zeros_like(noises)
        patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        for k, p in enumerate(tqdm(noises)):
            sz = 100000
            data = (PhenomenologicalSurfaceData(distance=5,
                                                noise=p,
                                                name=f'{iteration}_{distance}_{noise}',
                                                load=False,
                                                device=torch.device('cpu'))
                    .initialize(sz)
                    .get_syndromes())
            frequencies = torch.zeros(len(patterns), dtype=torch.int32)
            for i, pattern in enumerate(patterns):
                frequencies[i] = torch.sum(torch.all(data[:, -2:] == pattern, dim=1))
            frequencies = frequencies / sz
            p_lambda[k] = 1 + torch.sum(frequencies * np.log(frequencies + 1e-12)).numpy() / np.log(2)
            np.savetxt('plambda_ci_5.txt', p_lambda)
        '''
        # p_lambda = np.loadtxt('plambda_ci_3.txt')
        # plt.plot(np.arange(0.0, 0.12, 0.001), p_lambda, label=r'$p_3(\lambda | s) = p_3(\lambda)$', color='burlywood')

        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [Line2D([0], [0], color='black', marker=['o'][i], markersize=4, linestyle=ls,
                               label=f'{['NN'][i]}')
                        for i, ls in enumerate(['dotted'])]

        legend_handles = color_legend + style_legend  # + auto_handles
        ax.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        # p_lambda = np.loadtxt('plambda_ci_5.txt')
        # plt.plot(np.arange(0.0, 0.12, 0.001), p_lambda, label=r'$p_5(\lambda | s) = p_5(\lambda)$', color='cyan')
        plt.suptitle(f'{mode}')
        plt.tight_layout()

        # Get crossings
        dict1 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 3))
        n = list(dict1.keys())  # noises
        dict2 = torch.load("data/{0}_{1}_{2}.pt".format(eval_what, iteration, 5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='ci', p0=[0.003, 0.005])
        print(t)
        ax.axvline(t[0], ymin=0, ymax=1, color='silver', linewidth=1)
        inset.axvline(t[0], ymin=0, ymax=1, color='silver', linewidth=1)
        plt.savefig(f'plots/CI_{mode}.svg')
        plt.show()
    elif task == 400:
        # get dictionaries
        merge_dictionaries(distances, noise_vals, 'log_error_rate', iteration, thresholds)

        # Plots
        fig, ax = plt.subplots()
        coloring = ['blue', 'red', 'green', 'black', 'powderblue', 'orange']
        markers = ['o', 'x', 'v', '^']
        styles = ['dashed', 'dotted', 'dashdot', 'dotted']

        # log error rate
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_mwpm.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='v', markersize=5, linestyle='dashed',
                        label='d={}_mwpm'.format(dist), color=coloring[i])

        t = 0.5
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5,
                        linestyle='solid',
                        label='d={0}'.format(dist), color=coloring[i])

        # plt.grid()
        ax.set_xlabel("$p$")
        ax.set_ylabel("$p_L$")
        plt.suptitle(f'{mode}')

        plt.xlim([0, 0.0145])
        plt.ylim([0, 0.55])
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [
            Line2D([0], [0], marker=['v', 'o'][i], markersize=4, color='black', linestyle=ls,
                   label=f'{['MWPM', 'NN'][i]}')
            for i, ls in enumerate(['dashed', 'solid'])]

        legend_handles = color_legend + style_legend
        plt.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        # plt.grid()
        plt.savefig(f'plots/log_error_rate_{mode}.svg')
        plt.show()

        dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 3, 0.5))
        n = list(dict1.keys())  # noises
        dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 5, 0.5))

        t = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='pl', p0=[0.001, 0.01])
        print('Crossing 3,5: ', t)

        # scaling error rate
        fig, ax = plt.subplots()
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_mwpm.pt".format(eval_what, iteration, dist))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))

            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='v', markersize=5, linestyle='dashed',
                        label='d={}_mwpm'.format(dist), color=coloring[i])
        t = 0.5
        for i, dist in enumerate(distances):
            # iteration = 'ff14' if dist <= 7 else 'ff15'
            dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
            n = list(dict.keys())
            pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
            # get here median, upper error bar, lower error bar
            pr_m = list(map(lambda x: x[0], pr))
            pr_u = list(map(lambda x: x[1], pr))
            pr_d = list(map(lambda x: x[2], pr))
            if dist == 7:
                n = n[1:]
                pr_m = pr_m[1:]
                pr_u = pr_u[1:]
                pr_d = pr_d[1:]
            ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=5,
                        linestyle='solid',
                        label='d={0}'.format(dist), color=coloring[i])

        plt.grid()
        ax.set_xlabel("$p$")
        ax.set_ylabel("$p_L$")
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.suptitle(f'{mode}')
        ax.set_xlim(9e-4, 0.0145)
        ax.set_ylim(1e-3, 0.55)

        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [
            Line2D([0], [0], color='black', marker=['v', 'o'][i], markersize=4, linestyle=ls,
                   label=f'{['MWPM', 'NN'][i]}')
            for i, ls in enumerate(['dashed', 'solid'])]

        legend_handles = color_legend + style_legend
        plt.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig(f'plots/log_error_rate_{mode}_scaling.svg')
        plt.show()

        # abort scaling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                ax1.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

        # plt.grid()
        ax1.set_xlabel("$p$")
        ax1.set_ylabel("$p_L$")
        plt.suptitle(f'{mode}')
        ax1.set_xlim([9e-4, 0.0145])
        ax1.set_ylim([1e-4, 0.3])

        # fig, ax = plt.subplots()
        # inset_ax = ax2.inset_axes([0.67, 0.1, 0.3, 0.2])
        # inset_ax.set_title(r"Scaling $\alpha p^{t+1}$", fontsize=11)
        # inset_ax.set_xlabel('c', fontsize=11)
        # inset_ax.set_ylabel(r"$t$", fontsize=11)
        # inset_ax.tick_params(axis='both', labelsize=11)
        from_to = np.array([[[1, 5], [1, 5], [1, 5]],
                            [[1, 4], [1, 4], [1, 4]]])
        vals = np.zeros((len(thresholds[1:]), len(distances)))

        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = np.array(list(map(lambda x: x[3], pr)))

                ax2.errorbar(n, pr_m, yerr=(np.sqrt(np.array(pr_m) / 100000)), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

                # scaling = get_scaling(n[from_to[i, j, 0]:from_to[i, j, 1]], pr_m[from_to[i, j, 0]:from_to[i, j, 1]],
                #                       np.sqrt(pr_m[from_to[i, j, 0]:from_to[i, j, 1]] / 100000))
                # print(f'Scaling {dist}, {t}: ', scaling)
                # vals[j, i] = scaling[0][1]
                # ax2.plot(n[1:], function(n[1:], scaling[0][0], scaling[0][1]), color=coloring[i+1])
                # inset_ax.errorbar(1 - t, scaling[0][1], yerr=scaling[1][1], markersize=5,
                #                   linestyle=styles[j], marker=markers[j], color=coloring[i])
        '''
        for i, d in enumerate(distances):
            inset_ax.plot(1 - np.array(thresholds[1:]), vals[:, i], marker='None', linestyle='dotted', linewidth=2,
                          color=coloring[i])
        '''
        # noise, abort_prob = get_abort_probability(thresholds, 5)
        # for i in range(np.shape(abort_prob)[0]):
        #     ax.plot(noise, abort_prob[i], color='black')
        # plt.grid()
        ax2.set_xlim(9e-4, 0.0145)
        ax2.set_ylim(9e-4, 1.1)
        ax2.set_xlabel("$p$")
        ax2.set_ylabel(r'$p_{abort}$')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        # ax2.legend()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [
            Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                   label=f'c={1 - thresholds[i + 1]}')
            for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax1.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax2.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig(f'plots/log_error_rate_abort_{mode}_scaling.svg')
        plt.show()

        # NN decoder with post-selection, linear
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = list(map(lambda x: x[0], pr))
                pr_u = list(map(lambda x: x[1], pr))
                pr_d = list(map(lambda x: x[2], pr))

                ax1.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

        ax1.set_xlabel("$p$")
        ax1.set_ylabel("$p_L$")
        plt.suptitle(f'{mode}')

        crossings = np.zeros((3, len(thresholds[1:])))
        crossings[0] = 1 - np.array(thresholds[1:])

        upto = [4, 4, 4]
        go = [1, 0]
        # fig, ax = plt.subplots()
        for j, t in enumerate(thresholds[1:]):
            for i, dist in enumerate(distances):
                # iteration = 'ff14' if dist <= 7 else 'ff15'
                dict = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, dist, t))
                n = list(dict.keys())
                pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
                # get here median, upper error bar, lower error bar
                pr_m = np.array(list(map(lambda x: x[3], pr)))

                ax2.errorbar(n, pr_m, yerr=(np.sqrt(pr_m / 100000)), marker=markers[j], markersize=5,
                             linestyle=styles[j],
                             label='d={0}_c={1}'.format(dist, 1 - t), color=coloring[i])

                # scaling = get_scaling(n[go[i]:upto[j] + 1], pr_m[go[i]:upto[j] + 1],
                #                       np.sqrt(pr_m[go[i]:upto[j] + 1] / 100000))
                # print(f'Scaling {dist}, {t}: ', scaling)

            # crossings for p_abort
            dict1 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 3, t))
            n = list(dict1.keys())
            dict2 = torch.load("data/{0}_{1}_{2}_{3}.pt".format(eval_what, iteration, 5, t))

            crossing = find_crossing2(np.array(n), dict1, dict2, noise_model=mode, quant='abort', p0=[0.001, 0.01])
            print(f'For c={t} crossing:', crossing)

            crossings[1, j] = crossing[0]
            crossings[2, j] = crossing[1]

        # noise, abort_prob = get_abort_probability(thresholds, 5)
        # for i in range(np.shape(abort_prob)[0]):
        #     ax.plot(noise, abort_prob[i], color='black')
        # plt.grid()
        inset_ax = ax2.inset_axes([0.43, 0.75, 0.3, 0.2])  # or loc=1
        for i in range(len(thresholds[1:])):
            inset_ax.errorbar(crossings[0, i], crossings[1, i], yerr=crossings[2, i],
                              linestyle='None',
                              marker=markers[i], color='black', markersize=4)
        inset_ax.set_title("Threshold", fontsize=11)
        inset_ax.set_xlabel('c', fontsize=11)
        inset_ax.set_ylabel(r"$p_{th}$", fontsize=11)
        inset_ax.tick_params(axis='both', labelsize=11)

        ax2.set_xlabel("$p$")
        ax2.set_ylabel(r'$p_{abort}$')
        plt.tight_layout()
        color_legend = [Patch(facecolor=color, edgecolor=color, label=f'd={distances[i]}')
                        for i, color in enumerate(coloring[:len(distances)])]

        style_legend = [
            Line2D([0], [0], color='black', marker=markers[i], markersize=4, linestyle=ls,
                   label=f'c={1 - thresholds[i + 1]}')
            for i, ls in enumerate(styles[:len(thresholds[1:])])]

        legend_handles = color_legend + style_legend
        ax1.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)
        ax2.legend(handles=legend_handles, loc='best', fontsize=12, markerscale=1.5)

        plt.savefig(f'plots/log_error_rate_abort_{mode}.svg')
        plt.show()
    elif task == 8:  # Loss plots
        # File path (change if needed)
        try:
            file_path = "data/output_Qec_ST-Transformer_train_cl_14.txt"
        except FileNotFoundError:
            print('File not found. Save loss values first.')
            exit(-1)

        losses = []
        val_losses = []
        lrs = []

        loss_pattern = re.compile(r"Epoch (\d+), Loss: ([\d.eE+-]+)")
        val_loss_pattern = re.compile(r"Epoch (\d+), Validation Loss: ([\d.eE+-]+)")
        lr_pattern = re.compile(r"\[([\d.eE+-]+)]")  # Matches learning rate in square brackets

        # Read and extract values
        with open(file_path, "r") as f:
            for line in f:
                loss_match = loss_pattern.search(line)
                val_loss_match = val_loss_pattern.search(line)
                lr_match = lr_pattern.search(line)

                if loss_match:
                    losses.append(float(loss_match.group(2)))

                if val_loss_match:
                    val_losses.append(float(val_loss_match.group(2)))

                if lr_match:
                    lrs.append(float(lr_match.group(1)))

        # Convert to numpy arrays
        losses = np.array(losses)
        val_losses = np.array(val_losses)
        lrs = np.array(lrs)

        fig, ax = plt.subplots()

        ax.plot(losses, label='Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="black")
        ax.legend()
        plt.tight_layout(pad=2.0)
        plt.savefig('plots/loss.pdf')
        plt.show()


if __name__ == '__main__':
    # hyperparameters

    # noise_model = 'depolarizing'
    noise_model = 'phenomenological'
    # noise_model = 'circuit-level'

    distance = 3
    task = 1
    noise = 0.04

    main_qec(distance, task, noise_model, noise)
