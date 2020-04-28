import numpy as np
from config import config


def main():
    arr = np.load(config.train_array)
    ts = arr[:, 0]
    t_min = ts.min()
    t_max = ts.max()

    t0s = np.array(range(int(t_min), int(t_max)))

    ones_and_zeros = np.concatenate((np.ones(ts.size), np.zeros(t0s.size)))
    times = np.concatenate((ts, t0s))

    order = np.argsort(times)

    times = times[order]
    ones_and_zeros = ones_and_zeros[order]

    np.save(config.fremen_times, times)
    np.save(config.fremen_vals, ones_and_zeros)


if __name__ == '__main__':
    main()
