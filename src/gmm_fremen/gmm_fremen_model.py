import numpy as np
from sklearn.mixture import GaussianMixture
from config import config
from src.gmm_fremen.fremen import Fremen
from scipy.stats import multivariate_normal


class GMMFremenModel:

    def __init__(self, n_components: int = 5):
        self.gmm_model = GaussianMixture(n_components=n_components)
        self.fremen_models = np.array([Fremen(periods_to_include=5, periods_to_consider=100) for _ in range(n_components)])

    def fit(self, training_data, training_times, step):
        """

        :param training_data: numpy array of shape (n_samples, n_features)
        :param training_times: timestamps
        :param step: in seconds
        :return: self
        """
        self.gmm_model.fit(training_data)
        self.gmm_model.weights_ = np.ones(self.gmm_model.n_components) / self.gmm_model.n_components
        u_matrix = self.gmm_model.predict_proba(training_data)

        # for each training times: compute alphas
        t0 = training_times.min()
        tn = training_times.max()
        last_index = int((tn - t0) // step)
        alphas = np.zeros((last_index + 1, self.gmm_model.n_components))

        for i in range(len(training_times)):
            t = training_times[i]
            t_index = int((t - t0) // step)
            alphas[t_index, :] += u_matrix[i, :]

        # train fremen for each cluster
        training_times = np.array([t0 + (i * step) for i in range(last_index + 1)])
        for i in range(self.gmm_model.n_components):
            self.fremen_models[i].fit(training_times, alphas[:, i])

        return self

    def predict_densities(self, data, times):
        # get alpha of each fremen for each time
        alphas = np.array([model.predict(times) for model in self.fremen_models])

        # get pdf value from each cluster for each datapoint
        pdfs = np.array([multivariate_normal.pdf(data, mean=self.gmm_model.means_[i, :],
                                                 cov=self.gmm_model.covariances_[i, :, :])
                         for i in range(self.gmm_model.n_components)])

        # return sum of alpha_i * pdf_i for each data point
        return np.sum(alphas * pdfs, axis=0)


def main():
    arr = np.load(config.train_array)
    model = GMMFremenModel(5)
    model.fit(arr[:, 1:], arr[:, 0], step=10*60)

    
    densities = model.predict_densities(arr[:4200, 1:], arr[:4200, 0] + (3600 * 24 * 365))
    print(1)


if __name__ == '__main__':
    main()
