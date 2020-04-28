import femgmm
import numpy as np
from pandas import read_csv
from time import time

path = '../data/python_training_two_weeks.txt'
data = read_csv(filepath_or_buffer=path, sep=' ', header=None, engine='c', memory_map=True).values

T = data[:, 0]
X = data[:, 1:]

model = femgmm.Model()

start = time()
model = model.fit(T, X)
finish = time()
print('fit: ' + str(finish-start))

start = time()
y = model.predict(T, X)
finish = time()
print('predict: ' + str(finish-start))
