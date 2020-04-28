import numpy as np
from datetime import datetime
from config import config


def prepare_test_times(t0, tn, step=40):
    return np.array(range(t0, tn, step))


def fill_time_windows(test_data, t0, time_windows, step=40):

    for i in range(test_data.shape[0]):
        index = int((test_data[i, 0] - t0) // step)
        time_windows[index].append([test_data[i, 0], test_data[i, 1], test_data[i, 2], len(time_windows[index])])


def save_time_windows(test_times, time_windows, path):
    for i in range(len(test_times)):
        arr = np.array(time_windows[i])
        np.savetxt(path + "{}_test_data.txt".format(test_times[i]), arr)


def main():
    t0 = int(datetime(2017, 6, 7, 10, 20, 40).timestamp())
    tn = int(datetime(2017, 6, 8, 20, 13, 45).timestamp())
    test_times = prepare_test_times(t0, tn)
    np.savetxt(config.england_test_times, test_times)

    test_data = np.load(config.england_test_arr)
    time_windows = [[] for _ in range(len(test_times))]
    fill_time_windows(test_data, t0, time_windows)
    save_time_windows(test_times, time_windows, config.england + 'time_windows/')


def modify_france():
    test_times = np.loadtxt(config.resources + 'test_times.txt').astype(int)
    for t in test_times:
        arr = np.loadtxt(config.time_windows + "{}_test_data.txt".format(t))
        if arr.size == 0:
            continue
        if len(arr.shape) == 1:
            np.savetxt(config.time_windows + "{}_test_data.txt".format(t), arr[[0, 1, 2, -1]])
            continue
        arr[:, -1] = np.array(range(arr.shape[0]))
        np.savetxt(config.time_windows + "{}_test_data.txt".format(t), arr[:, [0, 1, 2, -1]])


if __name__ == '__main__':
    # main()
    modify_france()
