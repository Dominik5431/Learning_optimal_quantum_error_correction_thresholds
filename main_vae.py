from datetime import datetime
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchinfo import summary

import src.qec_vae.nn.utils.functions as functions
from src.qec_vae.nn.data.depolarizing import DepolarizingToricData
from src.qec_vae.nn.data.bitflip import BitFlipToricData
from src.qec_vae.nn.net.vae import VariationalAutoencoder
from src.qec_vae.nn.utils.plotter import plot_reconstruction_error, \
    plot_latent_susceptibility, plot_reconstruction_derivative, plot_reconstruction, plot_collapsed, \
    plot_mean_variance_samples, final_plot, final_plot2
from src.qec_vae.nn.train import train, train_TraVAE
from src.qec_vae.nn.test import test_model_latent_space, test_model_reconstruction_error, test_latent_space_TraVAE
from src.qec_vae.nn.data.results_wrapper import ResultsWrapper
from src.qec_vae.nn.utils.loss import loss_func
from src.qec_vae.nn.utils.optimizer import make_optimizer

import numpy as np
import logging

task_description = {0: "Create data", 1: "Train network", 2: "Evaluate latent space",
                    20: "Evaluate reconstruction error"}

# Distances to evaluate VAE
distances = [7, 9, 11, 15, 21, 27, 33]


def main_vae(task, distance, noise_model):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    ''' Hyperparameters '''

    random_flip = True  # Flips every second syndrome to assure Z2 symmetry in the data

    structures = ['conv-only', 'upsampling', 'skip']
    structure = structures[1]  # Select encoder-decoder structure

    lr = 1e-4
    assert noise_model in ['bitflip', 'depolarizing']

    # Training hyperparameters
    num_epochs = 10
    batch_size = 100
    data_size = 8000 if distance > 30 else 10000

    # Load samples / Save generated samples
    load_data = True
    save_data = True

    # Train on noise strengths sampled uniformly along the Nishimori temperature line coming out from the
    # statistical mechanical mapping of the Toric code to some random bond Ising models.
    if noise_model == 'depolarizing':
        noises_training = np.array(
            list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.1, 3, 0.05))))
    else:
        noises_training = np.array(
            list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 2, 0.05))))

    # Data for evaluating latent space and reconstruction error is set to be equal the training data
    if noise_model == 'depolarizing':
        noises_testing = np.array(
            list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.1, 3, 0.02))))
    else:
        noises_testing = np.array(
            list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 2, 0.02))))

    # Test/Val split
    ratio = 0.8
    # Dimension of the latent space
    latent_dims = 1

    if structure == 'upsampling' or structure == 'skip':
        assert distance in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    elif structure == 'conv-only':
        assert distance in [9, 17, 25, 33, 41, 49]

    only_syndromes = True

    plt.rcParams['xtick.labelsize'] = 14  # 16
    plt.rcParams['ytick.labelsize'] = 14  # 16
    plt.rcParams['axes.labelsize'] = 16

    iteration = "vae"

    name_data = str(noise_model) + "_" + ("rf_" if random_flip else "") + iteration + "-" + str(distance)
    name_NN = "net_" + str(noise_model) + "_" + structure + "_dim" + str(latent_dims) + (
        "_rf_" if random_flip else "") + iteration + "-" + str(distance)
    name_dict_recon = "reconstruction_" + str(noise_model) + "_" + structure + (
        "_rf_" if random_flip else "") + str(distance) + iteration
    name_dict_latent = "latents_" + str(noise_model) + "_" + structure + (
        "_rf_" if random_flip else "") + str(distance) + iteration

    if noise_model == 'depolarizing':
        b_recon = {17: 0.3, 25: 0.2, 33: 0.1}
        b_latent = {17: 0.02, 25: 0.01, 33: 0.01}
    else:
        b_recon = {17: 0.15, 25: 0.1, 33: 0.05}
        b_latent = {17: 0.01, 25: 0.005, 33: 0.005}  # better 3

    if task == 1:  # Create data
        # Generates data samples and
        #
        # saves them as .pt file.
        print("Create data.")
        try:
            if noise_model == 'bitflip':
                (BitFlipToricData(distance=distance, noises=noises_training, name=name_data.format(distance),
                                  load=True, random_flip=random_flip, device=device,
                                  only_syndromes=only_syndromes)
                 .training()
                 .initialize(num=data_size)
                 .save())
            elif noise_model == 'depolarizing':
                (DepolarizingToricData(distance=distance, noises=noises_training, name=name_data.format(distance),
                                       load=True, random_flip=random_flip, device=device,
                                       only_syndromes=only_syndromes)
                 .training()
                 .initialize(num=data_size)
                 .save())
        except FileNotFoundError:
            if noise_model == 'bitflip':
                (BitFlipToricData(distance=distance, noises=noises_training, name=name_data.format(distance),
                                  load=False, random_flip=random_flip, device=device,
                                  only_syndromes=only_syndromes)
                 .training()
                 .initialize(num=data_size)
                 .save())
            elif noise_model == 'depolarizing':
                (DepolarizingToricData(distance=distance, noises=noises_training, name=name_data.format(distance),
                                       load=False, random_flip=random_flip, device=device,
                                       only_syndromes=only_syndromes)
                 .training()
                 .initialize(num=data_size)
                 .save())

        #    elif task == 1:  # Training the network
        # Manages the training of the network.
        logging.debug("Get data.")
        data_train = None
        data_val = None

        if noise_model == 'bitflip':
            # data_train, data_val = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
            data_train, data_val = (
                BitFlipToricData(distance=distance, noises=noises_training, name=name_data.format(distance),
                                 load=load_data, random_flip=random_flip, device=device,
                                 only_syndromes=only_syndromes)
                .training()
                .initialize(num=data_size)
                .get_train_test_data(ratio))
        elif noise_model == 'depolarizing':
            data_train, data_val = (DepolarizingToricData(distance=distance, noises=noises_training,
                                                          name=name_data.format(distance),
                                                          load=load_data,
                                                          random_flip=random_flip,
                                                          device=device,
                                                          only_syndromes=only_syndromes)
                                    .training()
                                    .initialize(num=data_size)
                                    .get_train_test_data(ratio))

        print("Train nn.")
        assert data_train is not None
        assert data_val is not None
        model = VariationalAutoencoder(latent_dims, distance, name_NN.format(distance), structure=structure,
                                       noise=noise_model, device=device)
        #    elif task == 20:  # Evaluate reconstruction loss
        # Evaluates the model reconstruction error for different noise strengths. Saves the reconstruction error as
        # dictionary.

        model = train(model, make_optimizer(lr), loss_func, num_epochs, batch_size, data_train, data_val,
                      distance=distance, b=b_recon[distance])

        print("Evaluate reconstruction error.")

        # Use dictionary with noise value and return values to store return data from VAE while testing
        reconstructions = ResultsWrapper(name=name_dict_recon)
        reconstructions.load()
        results = {}
        for noise in tqdm(noises_testing):
            data_test = None
            if noise_model == 'bitflip':
                # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
                data_test = (BitFlipToricData(distance=distance, noises=[noise],
                                              name="BFS_Testing-{0}".format(distance),
                                              load=False, random_flip=random_flip,
                                              device=device, only_syndromes=only_syndromes)
                             .eval()
                             .initialize(num=1000))
            elif noise_model == 'depolarizing':
                data_test = (DepolarizingToricData(distance=distance, noises=[noise],
                                                   name="DS_Testing-{0}".format(distance),
                                                   load=False, random_flip=random_flip,
                                                   device=device, only_syndromes=only_syndromes)
                             .eval()
                             .initialize(num=1000))
            assert data_test is not None
            results[noise] = test_model_reconstruction_error(model, data_test,
                                                             torch.nn.MSELoss(
                                                                 reduction='none'))  # returns avg_loss, variance
        reconstructions.add(distance, results)
        reconstructions.save()

        print("Retrain.")
        model = train(model, make_optimizer(lr), loss_func, num_epochs, batch_size, data_train, data_val,
                      distance=distance, b=b_latent[distance])
        # elif task == 2:  # Evaluating the latent space
        # Evaluates the model latent space.
        # Creates dictionary containing the latent variables for specified noise strengths.
        print("Evaluate latent space.")

        # Use dictionary with noise value and return values to store return data from VAE while testing
        latents = ResultsWrapper(name=name_dict_latent)
        latents.load()
        results = {}
        for noise in tqdm(noises_testing):
            data_test = None
            if noise_model == 'bitflip':
                # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
                data_test = (BitFlipToricData(distance=distance, noises=[noise],
                                              name="BFS_Testing-{0}".format(distance),
                                              load=False, random_flip=random_flip,
                                              device=device, only_syndromes=only_syndromes)
                             .eval()
                             .initialize(num=1000))
            elif noise_model == 'depolarizing':
                data_test = (DepolarizingToricData(distance=distance, noises=[noise],
                                                   name="DS_Testing-{0}".format(distance),
                                                   load=False, random_flip=random_flip,
                                                   device=device, only_syndromes=only_syndromes)
                             .eval()
                             .initialize(num=1000))
            assert data_test is not None
            # res = test_model_latent_space(model, data_test)

            m = torch.abs(torch.mean(data_test.syndromes, dim=(1, 2, 3)))
            print(torch.mean(m))
            sus = (torch.mean(m ** 2) - torch.mean(torch.abs(m)) ** 2).cpu().detach().numpy()
            results[noise] = test_model_latent_space(model, data_test) + (
                m, sus,)  # z_mean, z_log_var, z, flips, mean
            latents.add(distance, results)
            latents.save()
            result = latents.get_dict()

        # plot_latent_susceptibility(result, RANDOM_FLIP, STRUCTURE, NOISE_MODEL, surface=surface)
    elif task == 3:  # Plot latent space, computed in task 2
        test = ResultsWrapper(name=name_dict_latent).load().get_dict()
        # plot_latent_mean(test, random_flip, STRUCTURE)

        plot_latent_susceptibility(test, random_flip, structure, noise_model)
        # scatter_latent_var(test, random_flip, STRUCTURE)
        # plot_binder_cumulant(test, random_flip, STRUCTURE, NOISE_MODEL)
    elif task == 5:  # Plot exemplary reconstruction

        # noise = 0.15
        # temperature = 0.9
        # noise = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
        noise = 0.1
        print(noise)
        if noise_model == 'bitflip':
            sample = BitFlipToricData(distance=distance, noises=[noise],
                                      name="BFS_Testing-{0}".format(distance),
                                      load=False, random_flip=False, device=device,
                                      only_syndromes=only_syndromes).training().initialize(
                10)
        else:
            sample = DepolarizingToricData(distance=distance, noises=[noise],
                                           name="BFS_Testing-{0}".format(distance),
                                           load=False, random_flip=False, device=device,
                                           only_syndromes=only_syndromes).training().initialize(
                10)
        model = VariationalAutoencoder(latent_dims, distance, name_NN.format(distance), structure=structure,
                                       noise=noise_model, device=device)
        model = model.to(device)
        plot_reconstruction(sample, noise, distance, model)
    elif task == 6:  # Get mean and variance of raw data samples
        # Calculates mean and variance of the raw syndrome samples.
        results = {}
        # for noise in tqdm(NOISES_TESTING):
        for noise in np.arange(0.01, 0.2, 0.01):
            if noise_model == 'bitflip':
                data_test = (BitFlipToricData(distance=distance, noises=[noise],
                                              name="BFS_Testing-{0}".format(distance),
                                              load=False, random_flip=False, only_syndromes=True, device=device)
                             .eval()
                             .initialize(num=1000))
                # mean_tot = torch.mean(data_test.syndromes[0], dim=(0, 1, 2, 3))
                mean = torch.mean(data_test.syndromes[0], dim=(1, 2, 3))
                var = torch.var(data_test.syndromes[0], dim=(1, 2, 3))
                # print(mean_tot)
                # print(var)
                results[noise] = (mean, var)
            elif noise_model == 'depolarizing':
                data_test = (DepolarizingToricData(distance=distance, noises=[noise],
                                                   name="DS_Testing-{0}".format(distance),
                                                   load=False, random_flip=random_flip,
                                                   device=device)
                             .eval()
                             .initialize(num=100))
                mean_tot = torch.mean(data_test.syndromes[0], dim=(0, 1, 2, 3))
                mean = torch.mean(data_test.syndromes[0], dim=(1, 2, 3))
                var = torch.var(mean)
                results[noise] = (mean, var)
        raw = ResultsWrapper(name="mean_variance_" + str(noise_model).lower() + "_2_" + str(distance))
        raw.add(distance, results)
        raw.save()
        raw = raw.get_dict()
        assert raw != {}

        plot_mean_variance_samples(raw, distance, noise_model)
    elif task == -1:
        bitflip_dict_latent = {}
        for dist in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]:
            name_dict = "latents_bitflip" + "_" + structure + (
                "_rf_" if random_flip else "") + str(dist) + iteration
            try:
                current_dict = ResultsWrapper(name=name_dict).load().get_dict()
            except FileNotFoundError:
                current_dict = {}
            bitflip_dict_latent.update(current_dict)
        bitflip_dict_recon = {}
        for dist in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]:
            name_dict = "reconstruction_bitflip" + "_" + structure + (
                "_rf_" if random_flip else "") + str(dist) + iteration
            try:
                current_dict = ResultsWrapper(name=name_dict).load().get_dict()
            except FileNotFoundError:
                current_dict = {}
            bitflip_dict_recon.update(current_dict)

        depolarizing_dict_latent = {}
        for dist in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]:
            name_dict = "latents_depolarizing" + "_" + structure + (
                "_rf_" if random_flip else "") + str(dist) + iteration
            try:
                current_dict = ResultsWrapper(name=name_dict).load().get_dict()
            except FileNotFoundError:
                current_dict = {}
            depolarizing_dict_latent.update(current_dict)
        depolarizing_dict_recon = {}
        for dist in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]:
            name_dict = "reconstruction_depolarizing" + "_" + structure + (
                "_rf_" if random_flip else "") + str(dist) + iteration
            try:
                current_dict = ResultsWrapper(name=name_dict).load().get_dict()
            except FileNotFoundError:
                current_dict = {}
            depolarizing_dict_recon.update(current_dict)

        final_plot(bitflip_dict_latent, depolarizing_dict_latent, bitflip_dict_recon, depolarizing_dict_recon)
    elif task == -2:
        dist = 17

        name_dict = "latents_bitflip" + "_" + structure + (
            "_rf_" if random_flip else "") + str(dist) + iteration
        bitflip_dict_latent = ResultsWrapper(name=name_dict).load().get_dict()

        name_dict = "latents_depolarizing" + "_" + structure + (
            "_rf_" if random_flip else "") + str(dist) + iteration
        depolarizing_dict_latent = ResultsWrapper(name=name_dict).load().get_dict()

        final_plot2(bitflip_dict_latent, depolarizing_dict_latent, dist)

    elif task == 9:  # Show network params
        net = VariationalAutoencoder(latent_dims, distance, name_NN.format(distance), structure=structure,
                                     noise=noise_model)
        summary(net, (1, 1, distance, distance))

    elif task == 11:  # Plot Ising data and Toric code data: Used for presentation and thesis
        # temperature = 0.9
        # noise = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
        distance = 29
        noise = 0.01

        # Generate exemplary sample
        sample = BitFlipToricData(distance=distance, noises=[noise],
                                  name="BFS_Testing-{0}".format(distance),
                                  load=False, random_flip=False, device=device,
                                  only_syndromes=only_syndromes).training().initialize(
            10)
        sample = sample[0]
        syndrome = sample[0][0].squeeze()
        print(syndrome.shape)

        # Plotting
        fig, ax = plt.subplots()
        im = ax.imshow(np.reshape(syndrome.cpu().numpy(), (distance, distance)), cmap='inferno', origin='upper',
                       vmin=-1, vmax=1)

        ax.set_title(r'$d=29$')
        bar = fig.colorbar(im, ticks=[-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)

        # Adjust the axis limits to center the image
        padding = 1  # Adjust this value to control how much space you want around the image
        ax.set_xlim(-padding, distance)
        ax.set_ylim(distance, -padding)  # Flip the y-axis to keep origin='upper'

        # Draw lines that extend beyond the image
        for i in range(-padding, distance + 1):
            ax.vlines(i - 0.5, -padding, distance, colors='black')
            ax.hlines(i - 0.5, -padding, distance, colors='black')

        plt.tight_layout()
        plt.savefig('plots/syndrome_Toric_low_T.svg')
        plt.show()
    else:
        print("Unknown task number.")
        exit(-1)


if __name__ == '__main__':
    # hyperparameters
    task = 1
    distance = 25
    noise_model = "depolarizing"
    main_vae(task=task, distance=distance, noise_model=noise_model)
