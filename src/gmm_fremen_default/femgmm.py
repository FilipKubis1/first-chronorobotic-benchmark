from sklearn.mixture import GaussianMixture
import scipy.stats as st
import numpy as np
import pickle
from pandas import read_csv
from sklearn.metrics import log_loss
from src.gmm_fremen_default import fremen
from config import config


class Model:
    """
    parameters:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
    attributes:
        clusters ... int, number of clusters created by method
        structure ... list, parameters of hypertime space
        C_1 ... np.array, centres of clusters from the detections's model
        Pi_1 ... np.array, weights of clusters from the detections's model
        PREC_1 ... np.array, precision matrices of clusters from the detections's model
        C_0 ... np.array, centres of clusters from the not-detections's model
        Pi_0 ... np.array, weights of clusters from the not-detections's model
        PREC_0 ... np.array, precision matrices of clusters from the not-detections's model
    methods:
        fit(training_path)
            objective:
                to train model
            input:
                training_path ... string, path to training dataset
            output:
                self
        transform_data(path)
            objective:
                return transformed data into warped hypertime space and also return target values (0 or 1)
            input:
                path ... string, path to the test dataset
            outputs:
                X ... np.array, test dataset transformed into the hypertime space
                target ... target values
        predict(X)
            objective:
                return predicted values
            input:
                X ... np.array, test dataset transformed into the hypertime space
            output:
                prediction ... probability of the occurrence of detections
        rmse(path)
            objective:
                return rmse between prediction and target of a test dataset
            inputs:
                path ... string, path to the test dataset
            output:
                err ... float, root of mean of squared errors
    """


    def __init__(self):
        pass


    def fit(self, T, X, clusters=5, periods=5):
        """
        objective:
            to train model
        input:
            training_path ... string, path to training dataset
        output:
            self
        """
        self.clusters = clusters
        self.periods = periods
        self.C, self.PREC, self.phis, self.alphas, self.omegas, self.gamma_0 = self._estimate_distribution(T, X)
        return self


    def _estimate_distribution(self, T, X):
        """
        objective:
            return parameters of the mixture of gaussian distributions of the data from one class projected into the warped hypertime space
        inputs:
            condition ... integer 0 or 1, labels of classes, 0 for non-occurrences and 1 for occurrences
            path ... string, path to the test dataset
        outputs:
            C ... np.array, centres of clusters, estimation of expected values of each distribution
            Pi ... np.array, weights of clusters
            PREC ... np.array, precision matrices of clusters, inverse matrix to the estimation of the covariance of the distribution
        """
        clf = GaussianMixture(n_components=self.clusters, max_iter=500).fit(X)
        C = clf.means_
        PREC = clf.precisions_
        frm = fremen.Fremen()
        phis = []
        alphas = []
        omegas = []
        gamma_0 = []
        for idx in range(self.clusters):
            probs = self._prob_of_belong(X, C[idx], PREC[idx])
            frm = frm.fit(times=T, values=probs, no_freqs=self.periods)
            phis.append(frm.phis)
            alphas.append(frm.alphas)
            omegas.append(frm.omegas)
            gamma_0.append(frm.gamma_0)
        return C, PREC, np.array(phis), np.array(alphas), np.array(omegas), np.array(gamma_0)



    def predict(self, T, X):
        """
        objective:
            return predicted values
        input:
            X ... np.array, test dataset transformed into the hypertime space
        output:
            prediction ... probability of the occurrence of detections
        """
        DISTR = []
        frm = fremen.Fremen()
        for idx in range(self.clusters):
            frm = frm.mutate(self.phis[idx], self.alphas[idx], self.omegas[idx], self.gamma_0[idx])
            DISTR.append(frm.predict(T) * self._prob_of_belong(X, self.C[idx], self.PREC[idx]))
        model = np.sum(np.array(DISTR), axis=0)
        model[model<0.0] = 0.0
        return model

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

        # create empty array for predictions
        predictions = np.zeros((*grid_shape, times.size))

        # make grid prediction for each time
        for i in range(times.size):
            pred_t = self.predict(np.ones(np.prod(grid_shape)), grid)
            predictions[:, :, i] = pred_t.reshape(grid_shape)

        # return predictions
        return predictions


    def _prob_of_belong(self, X, C, PREC):
        """
        massively inspired by:
        https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor

        objective:
            return 1 - "p-value" of the hypothesis, that values were "generated" by the (normal) distribution
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        imputs:
            X ... np.array, test dataset transformed into the hypertime space
            C ... np.array, centres of clusters, estimation of expected values of each distribution
            PREC ... numpy array, precision matrices of corresponding clusters
        output
            numpy array, estimation of probabilities for each tested vector
        """
        X_C = X - C
        c_dist_x = np.sum(np.dot(X_C, PREC) * X_C, axis=1)
        return st.chi2._sf(c_dist_x, len(C))


    def rmse(self, path):
        """
        objective:
            return rmse between prediction and target of a test dataset
        inputs:
            path ... string, path to the test dataset
        output:
            err ... float, root of mean of squared errors
        """
        X, target = self.transform_data(path)
        y = self.predict(X)
        return np.sqrt(np.mean((y - target) ** 2.0))

    def save_model(self, f_name):
        with open(f_name + '.pickle', 'wb') as f:
            pickle.dump(self, f)


if __name__ == '__main__':
    arr = np.load(config.train_array)
    model = Model()
    model.fit(arr[:, 0], arr[:, 1:])
    model.save_model(config.gmm_default_model)
    # with open(config.gmm_default_model + '.pickle', 'rb') as f:
    #     model = pickle.load(f)
