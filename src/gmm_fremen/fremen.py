import numpy as np
from numpy import ndarray

# CONSTANTS
WEEK = 7 * 24 * 3600


# functions
def model_periodicity(omega: float, alpha0: float, times: ndarray, measurements: ndarray):
    """
    See fit model function for param description
    :return: alpha (amplitude), phi (phase shift)
    """
    complex_vectors = np.exp(-1j * omega * times)
    lengths = measurements - alpha0

    gamma = np.mean(complex_vectors * lengths)
    return np.abs(gamma), np.angle(gamma)


def fit_model(times: ndarray, measurements: ndarray, t: int, n_samples: int, pi : int, ps : int, shortest: float):
    """
    see fremen class for parameter description
    see Krajnas' paper on fremen for details on : phi (phase shift), alpha (amplitude) and omega (2pi / period)
    :return: model - a dictionary of {alpha_0, alphas, phis, omegas}
    """
    alpha_0 = measurements.mean()

    param = ps
    if shortest is not None:
        param = min(param, np.floor(t/shortest))

    phis = np.zeros(param)
    alphas = np.zeros(param)
    omegas = np.array([(i * 2 * np.pi) / t for i in range(1, param + 1)])

    for i in range(param):
        omega = omegas[i]
        alphas[i], phis[i] = model_periodicity(omega, alpha_0, times, measurements)

    alphas_sorted = np.sort(np.copy(alphas))

    pi = min(pi, n_samples)
    threshold = alphas_sorted[-pi]

    return {'alpha_0': alpha_0,
            'alphas': alphas[alphas >= threshold],
            'phis': phis[alphas >= threshold],
            'omegas': omegas[alphas >= threshold]}


# fremen model
class Fremen:
    """
    Non-binary fremen implementation
    """

    def __init__(self, measurement_period: float = WEEK, periods_to_include=7,
                 periods_to_consider: int = 70, shortest: float = None):
        """

        :param base_periodicity: in seconds, longest periodicity to be considered | one week by default
        :param periods_to_consider: how many periods will be considered
        """
        self.p0 = measurement_period
        self.pi = periods_to_include
        self.ps = periods_to_consider
        self.model = None
        self.shortest = shortest

    def fit(self, times: ndarray, measurements: ndarray):
        """

        :param times: ndarray (n, )| timestamp format of measured sample times
        :param measurements: ndarray (n, ) | measured samples
        :return: self
        """
        self.model = fit_model(times, measurements, self.p0, times.size, self.pi, self.ps, self.shortest)
        return self

    def predict(self, times: ndarray):
        """

        :param times: ndarray (n, ) | timestamps of times to predict
        :return: ndarray (n, ) | predicted values
        """
        alpha_0 = self.model['alpha_0']
        omegas = self.model['omegas']
        phis = self.model['phis']
        alphas = self.model['alphas']

        return alpha_0 + np.sum(np.cos(np.outer(times, omegas) + phis) * alphas, axis=1)