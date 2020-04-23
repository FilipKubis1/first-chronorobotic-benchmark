import numpy as np
from config import config
from src.gmm_fremen.gmm_fremen_model import GMMFremenModel

x_low = -9.5
y_low = 0.25


def save_prediction_into_model_no_angle(pred, times, directory):
    x_max, y_max, n_times = pred.shape
    times = times.astype(int)
    for t in range(n_times):
        f = open(directory + str(times[t]) + "_model.txt", "w")
        for a in range(8):
            a_real = (a - 3) * (np.pi / 4)
            for y in range(y_max):
                y_real = y / 2 + y_low
                for x in range(x_max):
                    x_real = x / 2 + x_low
                    f.write("{} {} {} {}\n".format(x_real, y_real, a_real, max(0, pred[x, y, t])))


def main():
    model = GMMFremenModel(5)
    model.load_model(config.gmm_fremen + 'model')

    test_timestamps = np.genfromtxt(config.resources + 'test_times.txt')
    predictions = model.predict_for_grid(test_timestamps, (24, 33), (-9.5, 0.25), 40, cell_dimensions=(0.5, 0.5))
    save_prediction_into_model_no_angle(predictions, test_timestamps, config.gmm_fremen)


if __name__ == '__main__':
    main()
