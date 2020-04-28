import pickle

import numpy as np
from config import config
from src.gmm_fremen.gmm_fremen_model import GMMFremenModel
from os import mkdir
from shapely.geometry import Point, Polygon
from src.gmm_fremen_default.femgmm import Model

x_low = -9.5
y_low = 0.25

x_low_england = -7
y_low_england = -2.5


def save_prediction_into_model_no_angle(pred, times, directory):
    x_max, y_max, n_times = pred.shape
    times = times.astype(int)
    mins_for_times = np.min(np.min(pred, axis=0), axis=0)
    for t in range(n_times):
        f = open(directory + str(times[t]) + "_model.txt", "w")
        for a in range(8):
            a_real = (a - 3) * (np.pi / 4)
            for y in range(y_max):
                y_real = y / 2 + y_low
                for x in range(x_max):
                    x_real = x / 2 + x_low
                    f.write("{} {} {} {}\n".format(x_real, y_real, a_real, pred[x, y, t] - mins_for_times[t]))


def save_predictions_in_polygon_into_model_no_angle(pred, times, directory, poly: Polygon):
    x_max, y_max, n_times = pred.shape
    times = times.astype(int)
    mins_for_times = np.min(np.min(pred, axis=0), axis=0)
    for t in range(n_times):
        f = open(directory + str(times[t]) + "_model.txt", "w")
        for a in range(8):
            a_real = (a - 3) * (np.pi / 4)
            for y in range(y_max):
                y_real = y + y_low
                for x in range(x_max):
                    x_real = x + x_low
                    if poly.contains(Point(x_real, y_real)):
                        f.write("{} {} {} {}\n".format(x_real, y_real, a_real, pred[x, y, t] - mins_for_times[t]))


def save_lot_of_models(clusters, periodicities, step=600):
    arr = np.load(config.train_array)
    test_timestamps = np.genfromtxt(config.resources + 'test_times.txt')

    for c in clusters:
        for p in periodicities:
            try:
                mkdir(config.models + 'gmm_fremen_c_{}_p_{}'.format(c, p))
            except:
                pass

            model = GMMFremenModel(n_components=c, n_periodicities=p)
            model.fit(arr[:, 1:], arr[:, 0], step=step)
            model.save_model(config.models + 'gmm_fremen_c_{}_p_{}/'.format(c, p))
            predictions = model.predict_for_grid(test_timestamps, (24, 33), (-9.5, 0.25), 40,
                                                 cell_dimensions=(0.5, 0.5), chi_sq=True)
            save_prediction_into_model_no_angle(predictions, test_timestamps, config.models + 'gmm_fremen_c_{}_p_{}/'.format(c, p))


def save_lot_of_models_england(clusters, periodicities, step=600):
    arr = np.load(config.england_train_arr)
    test_timestamps = np.genfromtxt(config.england_test_times)

    for c in clusters:
        for p in periodicities:
            try:
                mkdir(config.england_models + 'gmm_fremen_c_{}_p_{}'.format(c, p))
            except:
                pass

            model = GMMFremenModel(n_components=c, n_periodicities=p)
            model.fit(arr[:, 1:], arr[:, 0], step=step)
            model.save_model(config.england_models + 'gmm_fremen_c_{}_p_{}/'.format(c, p))
            predictions = model.predict_for_grid(test_timestamps, (18, 20), (-7, -2.5), 40,
                                                 cell_dimensions=(0.5, 0.5), chi_sq=True)
            poly = Polygon([(-7.5, 1), (-7.5, -3), (10.5, -3), (10.5, 1),
                                             (1.5, 1), (1.5, 17), (-0.5, 17), (-0.5, 1)])
            save_predictions_in_polygon_into_model_no_angle(predictions, test_timestamps,
                                                            config.england_models + 'gmm_fremen_c_{}_p_{}/'.format(c, p),
                                                            poly=poly)


def main():
    arr = np.load(config.train_array)
    model = GMMFremenModel(20)
    model.fit(arr[:, 1:], arr[:, 0], step=10 * 60)
    # model.load_model(config.gmm_fremen + 'model')

    test_timestamps = np.genfromtxt(config.resources + 'test_times.txt')
    predictions = model.predict_for_grid(test_timestamps, (24, 33), (-9.5, 0.25), 40, cell_dimensions=(0.5, 0.5))
    save_prediction_into_model_no_angle(predictions, test_timestamps, config.gmm_fremen)


def save_femgmm():
    with open(config.gmm_default_model + '.pickle', 'rb') as f:
        model = pickle.load(f)
    test_timestamps = np.genfromtxt(config.resources + 'test_times.txt')
    predictions = model.predict_for_grid(test_timestamps, (24, 33), (-9.5, 0.25), 40, cell_dimensions=(0.5, 0.5))
    save_prediction_into_model_no_angle(predictions, test_timestamps, config.fem_gmm)


if __name__ == '__main__':
    # main()
    # save_lot_of_models_england(clusters=[3, 5], periodicities=[3, 5], step=3600)
    save_femgmm()