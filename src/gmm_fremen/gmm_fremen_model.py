import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from config import config
from src.gmm_fremen.fremen import Fremen
from scipy.stats import multivariate_normal
import scipy.stats as st


class GMMFremenModel:

    def __init__(self, n_components: int = 5, n_periodicities: int = 10):
        self.gmm_model = GaussianMixture(n_components=n_components)
        self.fremen_models = np.array([Fremen(periods_to_include=n_periodicities, periods_to_consider=100) for _ in range(n_components)])

    def fit(self, training_data, training_times, step):
        """

        :param training_data: numpy array of shape (n_samples, n_features)
        :param training_times: timestamps
        :param step: in seconds - discretization step of measurements to create time series for fremen
        :return: self
        """

        # fit gmm to get clusters
        self.gmm_model.fit(training_data)

        # get "correspondence matrix" of posterior probabilities of each point belonging to different clusters
        self.gmm_model.weights_ = np.ones(self.gmm_model.n_components) / self.gmm_model.n_components
        u_matrix = self.gmm_model.predict_proba(training_data)

        # for each training time: compute alphas
        t0 = training_times.min()
        tn = training_times.max()
        last_index = int((tn - t0) // step)
        alphas = np.zeros((last_index + 1, self.gmm_model.n_components))

        for i in range(len(training_times)):
            t = training_times[i]
            t_index = int((t - t0) // step)
            alphas[t_index, :] += u_matrix[i, :]

        # normalize alphas to correspond to one second
        alphas /= step

        # train fremen for each cluster
        training_times = np.array([t0 + (i * step) for i in range(last_index + 1)])
        for i in range(self.gmm_model.n_components):
            self.fremen_models[i].fit(training_times, alphas[:, i])

        return self

    def predict_densities(self, data, times, step=1):
        """

        :param data: np.array (n, 2) of data to predict
        :param times: np.array (n, ) of the times of data to predict
        :param step: in seconds, factor of alphas
        :return: np.array(n, ) densities
        """

        # get alpha of each fremen for each time and
        alphas = np.array([model.predict(times) for model in self.fremen_models]) * step

        # get pdf value from each cluster for each datapoint
        pdfs = np.array([multivariate_normal.pdf(data, mean=self.gmm_model.means_[i, :],
                                                 cov=self.gmm_model.covariances_[i, :, :])
                         for i in range(self.gmm_model.n_components)])

        # return sum of alpha_i * pdf_i for each data point
        return np.sum(alphas * pdfs, axis=0)

    def predict_chi_square(self, data, times, step=1):
        """

                :param data: np.array (n, 2) of data to predict
                :param times: np.array (n, ) of the times of data to predict
                :param step: in seconds, factor of alphas
                :return: np.array(n, ) densities
                """
        alphas = np.array([model.predict(times) for model in self.fremen_models]) * step

        chi_squared = np.zeros((self.gmm_model.n_components, times.size))
        for i in range(times.size):
            for j in range(self.gmm_model.n_components):
                PREC = self.gmm_model.precisions_[j, :, :]
                X_C = data[i, :] - self.gmm_model.means_[j, :]
                chi_squared[j, i] = self.get_chi_2(PREC, X_C)

    def get_chi_2(self, PREC, X_C):
        c_dist_x = np.sum(np.dot(X_C, PREC) * X_C)
        return st.chi2._sf(c_dist_x, 2)

    def predict_for_grid(self, times, grid_shape, first_cell, step, cell_dimensions=(1, 1), chi_sq=False):
        """

        :param times: np.array (t, ) of the times of the data to predict
        :param grid_shape: (n, m) tuple
        :param first_cell: (x0, y0) tuple
        :param step: in seconds, factor of alpha
        :param cell_dimensions: (x, y) tuple
        :return: np.array (n, m, t)
        """

        # create grid
        grid = np.array([[first_cell[0] + cell_dimensions[0] * x, first_cell[1] + cell_dimensions[1] * y]
                         for x in range(grid_shape[0]) for y in range(grid_shape[1])])

        # get alpha of each fremen for each time
        cell_volume = cell_dimensions[0] * cell_dimensions[1]
        alphas = np.array([model.predict(times) for model in self.fremen_models]) * step * cell_volume

        # get pdf value from each cluster for each grid cell
        if chi_sq:
            prob_densities = np.array([[self.get_chi_2(self.gmm_model.precisions_[j, :, :], grid[i, :] - self.gmm_model.means_[j, :])
                for i in range(grid_shape[0] * grid_shape[1])]for j in range(self.gmm_model.n_components)])
        else:
            prob_densities = np.array([multivariate_normal.pdf(grid, mean=self.gmm_model.means_[i, :],
                                                     cov=self.gmm_model.covariances_[i, :, :])
                             for i in range(self.gmm_model.n_components)])

        prob_densities_trans = prob_densities.T

        # create empty array for predictions
        predictions = np.zeros((*grid_shape, times.size))

        # make grid prediction for each time
        for i in range(times.size):
            alphas_t = alphas[:, i]
            pred_t = prob_densities_trans @ alphas_t
            predictions[:, :, i] = pred_t.reshape(grid_shape)

        # return predictions
        return predictions

    def load_model(self, f_name):
        with open(f_name + '_gmm.pickle', 'rb') as f:
            self.gmm_model = pickle.load(f)
        with open(f_name + '_fremen.pickle', 'rb') as f:
            self.fremen_models = pickle.load(f)

    def save_model(self, f_name):
        with open(f_name + '_gmm.pickle', 'wb') as f:
            pickle.dump(self.gmm_model, f)
        with open(f_name + '_fremen.pickle', 'wb') as f:
            pickle.dump(self.fremen_models, f)


def main():
    arr = np.load(config.train_array)
    model = GMMFremenModel(20)
    model.fit(arr[:, 1:], arr[:, 0], step=10*60)
    model.save_model(config.gmm_fremen + 'model')
    print(1)


if __name__ == '__main__':
    main()
