import numpy as np
from matplotlib import pyplot as plt
from config import config
from datetime import datetime
from shapely.geometry import Polygon, Point


week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def points_in_poly(arr, poly: Polygon):
    res = np.zeros(arr.shape[0]).astype(np.bool)
    for i in range(arr.shape[0]):
        res[i] = poly.contains(Point(arr[i, 1], arr[i, 2]))
    return res


def visualize_space_in_polygon(arr, poly: Polygon):
    plt.plot(arr[:, 1], arr[:, 2], 'b.')
    plt.plot(*poly.exterior.xy)
    plt.show()


def visualize_space(arr):
    plt.plot(arr[:, 1], arr[:, 2], 'b.')
    plt.show()


def visualize_time(arr):
    days = np.vectorize(lambda x: datetime.fromtimestamp(x).day)(arr[:, 0])
    plt.plot(days, np.zeros(days.size), 'r.')
    plt.show()


def show_max_min(arr):
    print('x_min: {}'.format(arr[:, 1].min()))
    print('x_max: {}'.format(arr[:, 1].max()))
    print('y_min: {}'.format(arr[:, 2].min()))
    print('y_max: {}'.format(arr[:, 2].max()))
    times = np.vectorize(lambda x: datetime.fromtimestamp(x))(arr[:, 0])  # returns array of datetimes
    print('t_min: {}'.format(times.min()))
    print('t_max: {}'.format(times.max()))
    print('t_min: {}'.format(arr[:, 0].min()))
    print('t_max: {}'.format(arr[:, 0].max()))


def analyze_measurement_times(arr):
    # print number of measurements for each weekday
    days_of_week = np.vectorize(lambda x: datetime.fromtimestamp(x).weekday())(arr[:, 0])
    unique_measurements = np.unique(days_of_week, return_counts=True)
    for i in range(len(unique_measurements[0])):
        print("Number of measurements for {}: {}".format(week_days[unique_measurements[0][i]], unique_measurements[1][i]))

    # print each measurement day with its weekday
    days_with_weekdays = np.array(
        [(datetime.fromtimestamp(t).day, week_days[datetime.fromtimestamp(t).weekday()]) for t in arr[:, 0]])
    unique_days = np.unique(days_with_weekdays, axis=0)
    for d in unique_days[np.argsort(unique_days[:, 0].astype(int))]:
        print("{}, {}. March".format(d[1], d[0]))


def main():
    arr = np.load(config.england_train_arr)
    # visualize_space_in_polygon(arr, Polygon([(-7.5, 1), (-7.5, -3), (10.5, -3), (10.5, 1),
    #                                          (1.5, 1), (1.5, 17), (-0.5, 17), (-0.5, 1)]))
    # visualize_space(arr)
    # visualize_time(arr)
    show_max_min(arr)
    analyze_measurement_times(arr)


if __name__ == '__main__':
    main()