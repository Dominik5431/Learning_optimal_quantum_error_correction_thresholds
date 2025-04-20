# Machine Learning the Optimal Error Correction Threshold

This repository contains code for the master's thesis titled *"Machine Learning the Optimal Error Correction Threshold"*.

## Virtual Environment Setup

To activate the virtual environment, run:

```bash
source .venv/bin/activate
```

## Code Overview

The code is divided into two parts, one corresponding to Sec. 4, i.e. learning a latent space representation for toric code syndromes using a variational autoencoder, and the other corresponding to Sec. 5, i.e. learning to estimate the coherent information by estimating the densities $$p(l|s)$$ for syndromes $$s$$ and logical operators $$l$$.

## Running the scripts
To execute the scripts from the terminal, run e.g. the following command that trains a transformer model for the distance-three surface code.
```
python3 main.py --method "transformer" --distance 3 --noise 0.05 --noise_model "depolarizing" --task "train"
```

## Command-Line Arguments

Below are the available command-line arguments:

### Required Arguments:
- `--method`  
  Choose the method for model training. Options:
  - `"transformer"`
  - `"vae"`

- `--noise_model`  
  Select the noise model, which is simulated. Options:
  - `"bitflip"`
  - `"depolarizing"`
  - `"phenomenological"`
  - `"circuit-level"`

- `--task`  
  Define the task to be executed. Options:
  - `"train"`: Train the model
  - `"decoding"`: Perform decoding
  - `"plot_ci"`: Plot the coherent information estimates
  - `"plot_log_error"`: Plot logical error rate
  - `"attention"`: Visualize attention weights
  - `"loss"`: Plot loss function during training
  - `"plot-single-latent"`: Plot a single latent space representation
  - `"reconstruction"`: Visualize reconstructed data
  - `"raw-data"`: Visualize raw data
  - `"plot-latent-reconstruction"`: Plot latent space reconstruction
  - `"plot-vs-raw"`: Plot comparison between latent space and raw data

### Additional Options:
- `--distance`  
  The distance of the surface code (e.g., 3 for distance-three surface code).
  
- `--noise`  
  The noise rate (e.g., 0.05).

Available distances and noise rates can be read in the thesis.

## Data and Output
The main scripts for the representation learning task (coherent information estimation task) can be found in main_vae.py and main_qec.py. All model checkpoints and data is saved in data, and plots are saved in plots.


 
