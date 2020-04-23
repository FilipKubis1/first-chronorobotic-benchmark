import numpy as np
from config import config
from src.gmm_fremen.gmm_fremen_model import GMMFremenModel
from datetime import datetime


def main():
    # load training data and fit the model
    arr = np.load(config.train_array)
    model = GMMFremenModel(5)
    model.fit(arr[:, 1:], arr[:, 0], step=10 * 60)

    # generate test data (a random monday)
    t0 = int(datetime(2020, 4, 20, 0, 0, 0).timestamp())
    times = np.array([t0 + i for i in range(t0, t0 + 3600 * 24, 10 * 60)])

    predictions_for_times = np.zeros(len(times))
    densities = np.zeros((len(times), 15 * 17))

    x_s = np.array([[x, y] for x in range(-10, 5) for y in range(-2, 15)])
    n = x_s.shape[0]

    for i in range(len(times)):
        densities[i, :] = model.predict_densities(x_s, times[i] * np.ones(n))
        predictions_for_times[i] = densities[i, :].sum()

    print(0)


if __name__ == '__main__':
    main()