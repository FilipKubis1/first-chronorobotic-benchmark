import numpy as np
import pandas as pd
from config import config


def load_data(datafile):
    return pd.read_csv(datafile, sep=' ', header=None, engine='c', usecols=[0, 1, 2]).rename(
        columns={0:'t', 1:'x', 2:'y'}
    )


def transform_england():
    """
    1495411200 1496620800
    -8 12
    -3 17
    """
    arr_tr = np.loadtxt(config.england_train_txt)
    np.save(config.england_train_arr, arr_tr)
    arr_ts = np.loadtxt(config.england_test_txt)
    np.save(config.england_test_arr, arr_ts)


def transform_france():
    data = load_data(config.train_csv)
    array = np.array(data)
    np.save(config.train_array, array)


if __name__ == '__main__':
    transform_france()
    transform_england()
