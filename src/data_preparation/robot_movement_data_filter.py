import numpy as np
from config import config
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import Dict


def get_new_position(x, y, delta, border : Polygon):
    """

    :param x:
    :param y:
    :param delta: distance of robot movement
    :param border: shapely polygon
    :return: new position within borders
    """

    while True:
        # get random vector of length delta
        rand_vec = (np.random.random(2) * 2) - 1
        movement = delta * (rand_vec / np.linalg.norm(rand_vec))
        x_new = x + movement[0]
        y_new = y + movement[1]

        # return point moved by the vector if it is still within the borders
        if border.contains(Point(x_new, y_new)):
            return x_new, y_new


def filter_data(data, time_indices, robot_positions, radius):
    """

    :param data: all the measurements, np.array (n, 3) t,x,y
    :param time_indices: indices of robot position for each row
    :param robot_positions: a position for each time
    :param radius: max distance allowed from the robot to be visible
    :return: filtered data
    """
    is_visible = np.zeros(len(time_indices)).astype(np.bool)
    for i in range(len(time_indices)):
        t = time_indices[i]
        is_visible[i] = np.linalg.norm(robot_positions[t, :] - data[i, 1:]) <= radius

    return data[is_visible, :]


def main():
    # load data
    data_all = np.load(config.train_array)

    for r in [0.5, 1, 2]:
        # prepare the robot positions to filter the data #
        # positions, radius, and movement speed
        x, y, delta = -3, 6, 1

        # border of the corridor
        border = Polygon([(-9.25, 0.1), (-9.25, 12.8), (3, 12.8), (3, 0.1)])

        # times
        t0 = int(datetime(2019, 3, 2, 2, 19, 0).timestamp())
        tn = int(datetime(2019, 3, 28, 21, 13, 30).timestamp())
        n_times = (tn - t0) // 30

        # gets robot position for each time
        robot_positions = np.zeros((n_times, 2))
        for i in range(n_times):
            robot_positions[i] = x, y
            x, y = get_new_position(x, y, delta, border)

        # filter data based on robot positions and radius #
        # get time indices for each measurement
        time_indices = np.vectorize(lambda t: int(round((t - t0) / 30)))(data_all[:, 0])

        # filter data and save it
        data_filtered = filter_data(data_all, time_indices, robot_positions, r)
        np.save(config.resources + 'robot_data_random_r{}.npy'.format(r), data_filtered)


if __name__ == '__main__':
    main()