import numpy as np
import pandas as pd
from config import config


def load_data(datafile):
    return pd.read_csv(datafile, sep=' ', header=None, engine='c', usecols=[0, 1, 2]).rename(
        columns={0:'t', 1:'x', 2:'y'}
    )


def main():
    data = load_data(config.train_csv)
    array = np.array(data)
    np.save(config.train_array, array)


if __name__ == '__main__':
    main()