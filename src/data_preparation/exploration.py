import numpy as np
from src.fremen.fremen_whyte import Fremen
from config import config
from datetime import datetime

IOTA = 1e-6
OBSERVATION_WINDOW = 7 * 60  # five minutes

def get_entropy(p):
    if p <= 0:
        return 0
    elif p >= 1:
        return 0
    else:
        return - p * np.log2(p)


def greedy_entropy(p, n):
    entropies = np.vectorize(get_entropy)(p)
    return np.argsort(entropies)[-n:]


def monte_carlo_entropy(p, n):
    entropies = np.vectorize(get_entropy)(p)
    entropies += IOTA
    return np.random.choice(range(len(entropies)), n, replace=False, p=entropies/entropies.sum())


def greedy_probability(p, n):
    return np.argsort(p)[-n:]


def monte_carlo_probability(p, n):
    p[p > 1] = 1
    p[p < 0] = 0
    p += IOTA
    return np.random.choice(range(len(p)), n, replace=False, p=p / p.sum())


def random(p, n):
    return np.random.choice(range(len(p)), n, replace=False)


def uniform(p, n):
    n_times = len(p)
    step = n_times / n
    return np.array([int(round(i * step)) for i in range(n)])


def in_measurement_window(t, step, measurement_times):
    t_datetime = datetime.fromtimestamp(t)
    t0 = datetime(t_datetime.year, t_datetime.month, t_datetime.day, 0, 0, 0).timestamp()
    rounded_time = t0 + (((t - t0) // step) * step)
    return int(rounded_time) in measurement_times


def monte_carlo_with_extremes(p, n):
    w = np.vectorize(get_p_sin)(p)
    return np.random.choice(range(len(p)), n, replace=False, p=w / w.sum())


def greedy_with_extremes(p, n):
    w = np.vectorize(get_p_sin)(p)
    return np.argsort(w)[-n:]


def get_p_sin(p):
    return 0.5 * (np.sin(p * 4 * np.pi) + 1)


def exploration(times_all, ones_and_zeros_all, days_t0s, strategy, n_daily_measurements=20):
    """
    exploration function, performs temporl exploration based on strategy
    :param times_all: array (n, ) - times of observations (interspaced with times of non-observations - zeros)
    :param ones_and_zeros_all: array (n, ) - for each time 0 or 1
    :param days_t0s: timestamps of beginnings of days
    :param strategy: a function that take 2 parameters (probabilities: array, n: int) and returns
    n indices based on probabilities (the indices are then used to select observation times)
    :param n_daily_measurements: int, how many observations should be made each day
    :return:  times of observation chosen based on strategy
    """

    # do a random exploration for the first day
    t0 = days_t0s[0]
    possible_measurement_times = np.array([t0 + step for step in range(0, 3600 * 24, OBSERVATION_WINDOW)])
    measurement_times = set(np.random.choice(possible_measurement_times, n_daily_measurements, replace=False))

    # get predicate to determine if times are in the measurement time windows
    filtering_predicate = np.vectorize(lambda x: in_measurement_window(x, OBSERVATION_WINDOW,measurement_times))(times_all)

    times = times_all[filtering_predicate]
    ones_and_zeros = ones_and_zeros_all[filtering_predicate]

    # repeat for all days, but create a fremen model each day and base your strategy on it
    for t0 in days_t0s[1:]:
        fremen = Fremen()
        fremen.fit(times, ones_and_zeros)

        # choose times based on strategy
        possible_measurement_times = np.array([t0 + step for step in range(0, 3600 * 24, OBSERVATION_WINDOW)])
        predictions_fremen = fremen.predict(possible_measurement_times)

        # choose times based on strategy
        chosen_times = strategy(predictions_fremen, n_daily_measurements)
        measurement_times.update(possible_measurement_times[chosen_times])

        # get predicate to determine if times are in the measurement time windows
        filtering_predicate = np.vectorize(lambda x: in_measurement_window(x, OBSERVATION_WINDOW, measurement_times))(times_all)

        times = times_all[filtering_predicate]
        ones_and_zeros = ones_and_zeros_all[filtering_predicate]

    return measurement_times


def save_train_data_and_times_for_exploration(path, measurement_times, train_array_all):
    times_all = train_array_all[:, 0]
    filtering_predicate = np.vectorize(lambda x: in_measurement_window(x, OBSERVATION_WINDOW, measurement_times))(times_all)

    np.save(path, train_array_all[filtering_predicate, :])


def main():
    days = [2, 4, 6, 12, 14, 18, 20, 22, 26, 28]
    days_t0s = [datetime(2019, 3, day, 0, 0, 0).timestamp() for day in days]
    times_all = np.load(config.fremen_times)
    vals_all = np.load(config.fremen_vals)
    measurement_times = exploration(times_all, vals_all, days_t0s, greedy_entropy)
    save_train_data_and_times_for_exploration('/home/filip/Desktop/data.npy',
                                              measurement_times, np.load(config.train_array))


if __name__ == '__main__':
    main()