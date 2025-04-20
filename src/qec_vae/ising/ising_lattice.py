import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm


# Autocorrelation time isn't implemented here anymore. For implementation go back to old script.
# Follow sweep estimations in Civitcioglu to ensure independence of spin samples.

class IsingLattice:
    """
    Class to model an Ising lattice instance. Allows to sample configurations via the Metropolis algorithm.
    First the lattice is thermalized. Then configurations are sampled. Per sweep, L**2 flip attempts are performed.
    """
    def __init__(self, L, T_max, delta, n: int, initial='random'):
        """
        Parameters
        ----------
        L : int
            lattice size.
        n : int
            number of samples to collect
        initial : 'polarized' or 'random', optional
            Initial configuration of the lattice. The default is 'polarized'.

        Returns
        -------
        None.
        """
        self.L = L
        self.config = None
        self.T_max = T_max
        self.delta = delta
        self.data = np.zeros((int(self.T_max / self.delta), n, L, L))
        self.data_collection(n)

    def initialize(self):
        self.config = np.zeros((self.L, self.L), dtype=int)
        for i in np.arange(self.L):
            for j in np.arange(self.L):
                r = np.random.rand(1)
                self.config[i, j] = -1 if r < 0.5 else +1
        T = 104
        for i in np.arange(1000):
            self.mc_sweep(T)
            T -= 0.1

    def data_collection(self, n):
        self.initialize()
        print('initialized')
        T = self.T_max
        counter = 0
        while T > self.delta / 2:
            for _ in np.arange(3000):
                self.mc_sweep(T)
            for i in np.arange(n):
                for _ in range(1000):
                    self.mc_sweep(T)
                self.data[counter, i] = self.config.copy()
            counter += 1
            print('next temperature ', T, 'at position ', counter)
            T -= self.delta

    def pbint(self, n):
        return n % self.L

    def get_magnetization(self):
        return 1 / self.L ** 2 * np.sum(self.config)

    def next_config(self, T):
        """
        Performs on MC spin flip attempt.

        Returns
        -------
        None.

        """
        i = int(np.random.rand(1) * self.L)
        j = int(np.random.rand(1) * self.L)
        copy = self.config.copy()
        copy[i, j] *= -1
        # Consider only the energy difference: better code performance, only difference necessary for acceptance probability
        delta_energy = (
                copy[self.pbint(i), self.pbint(j)] * copy[self.pbint(i - 1), self.pbint(j)] +
                copy[self.pbint(i), self.pbint(j)] * copy[self.pbint(i + 1), self.pbint(j)] +
                copy[self.pbint(i), self.pbint(j)] * copy[self.pbint(i), self.pbint(j - 1)] +
                copy[self.pbint(i), self.pbint(j)] * copy[self.pbint(i), self.pbint(j + 1)] -
                self.config[self.pbint(i), self.pbint(j)] * self.config[self.pbint(i), self.pbint(j - 1)] -
                self.config[self.pbint(i), self.pbint(j)] * self.config[self.pbint(i), self.pbint(j + 1)] -
                self.config[self.pbint(i), self.pbint(j)] * self.config[self.pbint(i - 1), self.pbint(j)] -
                self.config[self.pbint(i), self.pbint(j)] * self.config[self.pbint(i + 1), self.pbint(j)])
        delta_energy *= -1
        if delta_energy <= 0:
            accepted = True
        else:
            accepted = (np.random.rand(1) < np.exp(-1 * delta_energy / T))
        if accepted:
            self.config = copy

    def mc_sweep(self, T):
        """
        Performs one Monte Carlo sweep.

        Returns
        -------
        Magnetization of configuration after one Monte Carlo sweep consisting of
        N = L**2 spin flip attempts.

        """
        for i in np.arange(self.L ** 2):
            self.next_config(T)
        return self.get_magnetization()

    def get_configs(self):
        return self.data
