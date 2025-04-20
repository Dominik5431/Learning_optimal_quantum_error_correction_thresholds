# Machine learning the optimal error correction threshold

Code for the master's thesis "Machine learning the optimal error correction threshold"

To activate virtual environment:
```
source .venv/bin/activate
```

The code is divided into two parts, one corresponding to Sec. 4, i.e. learning a latent space representation for toric code syndromes using a variational autoencoder, and the other corresponding to Sec. 5, i.e. learning to estimate the coherent information by estimating the densities $$p(l|s)$$ for syndromes $$s$$ and logical operators $$l$$.

To execute the scripts from the terminal, run e.g.
```
python3 main.py --method "transformer" --distance 3 --noise 0.05 --noise_model "depolarizing" --task "train"
```
to train a transformer model for  the distance-three surface code.

Possible command-line arguments:
--method: "transformer", "vae"
--noise_model: "bitflip", "depolarizing", "phenomenological", "circuit-level"
--task: "train", "decoding", "plot_ci", "plot_log_error", "attention", "loss", "plot-single-latent", "reconstruction", "raw-data", "plot-latent-reconstruction", "plot-vs-raw"

Available distances and noise rates can be read in the thesis.

The main scripts for the representation learning task (coherent information estimation task) can be found in main_vae.py and main_qec.py. All model checkpoints and data is saved in data, and plots are saved in plots.
 
