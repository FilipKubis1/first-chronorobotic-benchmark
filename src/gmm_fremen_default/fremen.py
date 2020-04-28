import numpy as np
from src.gmm_fremen_default.cython_files import gammas_for_fremen, fremen_predict
from time import time


class Fremen:

    def __init__(self):
        pass


    def fit(self, times, values, no_freqs=5, longest=604800, shortest=3600):
        self.gamma_0 = np.mean(values)
        self.phis, self.alphas, self.omegas = self._get_model(times, values, no_freqs, longest, shortest)
        return self


    def predict(self, pred_times):
        try:
            len(pred_times)
        except:
            pred_times = np.array([pred_times])
        return fremen_predict.calculate(self.alphas, self.omegas, np.array(pred_times).astype(float), self.phis, self.gamma_0)


    def _get_model(self, times, values, no_freqs, longest, shortest):
        tested_omegas = self._build_frequencies(longest, shortest)
        gammas = self._complex_numbers_batch(times, values, tested_omegas)
        ids = np.argpartition(-np.absolute(gammas), no_freqs)[:no_freqs]
        best_gammas = gammas[ids]
        phis = np.angle(best_gammas)
        alphas = np.absolute(best_gammas)
        omegas = tested_omegas[ids]
        return phis, alphas, omegas


    def _build_frequencies(self, longest, shortest):
        """
        input: longest float, legth of the longest wanted period in default
                              units
               shortest float, legth of the shortest wanted period
                               in default units
        output: W numpy array Lx1, sequence of frequencies
        uses: np.arange()
        objective: to find frequencies w_0 to w_k
        """
        k = int(longest / shortest)  # + 1
        tested_omegas = (2.0*np.pi*np.float64(np.arange(k) + 1)) / float(longest) # removed zero periodicity
        return tested_omegas


    def _complex_numbers_batch(self, T, S, W):
        """
        input: T numpy array Nx1, time positions of measured values
               S numpy array Nx1, sequence of measured values
               W numpy array Lx1, sequence of reasonable frequencies
        output: G numpy array Lx1, sequence of complex numbers corresponding
                to the frequencies from W
        uses: np.e, np.newaxis, np.pi, np.mean()
        objective: to find sparse(?) frequency spectrum of the sequence S
        """
        Gs = gammas_for_fremen.calculate(S.astype(float), T, W, np.pi*2.0, self.gamma_0)
        gammas = np.empty(Gs.shape[0], dtype=complex)
        gammas.real = Gs[:,0]
        gammas.imag = Gs[:,1]
        return gammas


    def mutate(self, phis, alphas, omegas, gamma_0):
        """
        to load pretrained params of model
        """
        self.phis = phis
        self.alphas = alphas
        self.omegas = omegas
        self.gamma_0 = gamma_0
        return self


