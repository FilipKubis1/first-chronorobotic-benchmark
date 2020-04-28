import numpy as np
from src.fremen.fremen_whyte import Fremen
from config import config
from datetime import datetime


def get_entropy(p):
    if p <= 0:
        return 0
    elif p >= 1:
        return 0
    else:
        return p * np.log2(p)


def greedy(entropies, n):
    return np.argsort(entropies)[-n:]


def monte_carlo(entropies, n):
    return np.random.choice(range(len(entropies)), n, replace=False, p=entropies/entropies.sum())


def exploration(times_all, ones_and_zeros_all, days_t0s, strategy, n_daily_measurements=20):

    # do a random exploration for the first day
    t0 = days_t0s[0]
    possible_measurement_times = np.array([t0 + step for step in range(0, 3600 * 24, 5 * 60)])
    measurement_times = set(np.random.choice(possible_measurement_times, n_daily_measurements, replace=False))

    # get predicate to determine if times are in the measurement time windows
    filtering_predicate = np.vectorize(lambda x: int((x - t0) // (5 * 60) + t0) in measurement_times)(times_all)

    times = times_all[filtering_predicate]
    ones_and_zeros = ones_and_zeros_all[filtering_predicate]

    # repeat for all days, but create a fremen model each day and base your strategy on it
    for t0 in days_t0s[1:]:
        fremen = Fremen()
        fremen.fit(times, ones_and_zeros)

        # choose times based on strategy
        possible_measurement_times = np.array([t0 + step for step in range(0, 3600 * 24, 5 * 60)])
        predictions_fremen = fremen.predict(possible_measurement_times)
        entropies = np.vectorize(get_entropy)(predictions_fremen)

        # choose times based on strategy
        chosen_times = strategy(entropies, n_daily_measurements)
        measurement_times.update(possible_measurement_times[chosen_times])

        # get predicate to determine if times are in the measurement time windows
        filtering_predicate = np.vectorize(lambda x: int((x - t0) // (5 * 60) + t0) in measurement_times)(times_all)

        times = times_all[filtering_predicate]
        ones_and_zeros = ones_and_zeros_all[filtering_predicate]

    return measurement_times


def main():
    days = [2, 4, 6, 12, 14, 18, 20, 22, 26, 28]
    days_t0s = [datetime(2019, 3, day, 0, 0, 0).timestamp() for day in days]
    times_all = np.load(config.fremen_times)
    vals_all = np.load(config.fremen_vals)
    measurement_times = exploration(times_all, vals_all, days_t0s, greedy)
    print(0)


if __name__ == '__main__':
    main()