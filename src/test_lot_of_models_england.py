import numpy as np
import tester
#import summarize
import summarize_new
import os
from config import config

"""
Before run this, please create new directories with your models name inside following directories; 'models' and 'results'
and, change the variable 'model' with the same name of the directories you created.

You can find the list of positions in '../data/positions.txt' (x, y, angle)
Model outputs should be same as this file with an additional column of weights (order of rows is not important for testing method).

Here, are the parameters of grid;

edges of cells: x...0.5 [m], y...0.5 [m], angle...pi/4.0 [rad]
number of cells: x...24, y...33, angles...8
center of "first" cell: (-9.5, 0.25, -3.0*pi/4.0)
center of "last" cell: (2.0, 16.25, pi) 

If you change the argument 'create_video' to True, there will be video of every time window in results

outputs will be written in ../results/$model/output.txt in following format;
list of values; [testing_time, number_of_detections_in_testing_data, interactions_of_dummy_model_clockwise, interactions_of_dummy_model_counterclockwise, interactions_of_real_model_clockwise, interactions_of_real_model_counterclockwise, total_weight_in_clockwise, total_weight_in_counterclockwise, total_interactions_of_chosen_trajectory]

Since this code is prepared in a short time for scientific reasons, sorry in advance for any ambiguity
"""

tester = tester.Tester(radius_of_robot=1.)
times = np.loadtxt(config.england_test_times, dtype='int')

# models, we need to compare
# models = ['gmm_fremen', 'gmm_fremen_r0.5', 'gmm_fremen_r1', 'gmm_fremen_r2', 'fremen', 'daily']
# models = ['gmm_fremen']

france_route = np.array([(-5.0, 9.75), (2.0, 2.75), (-7.0, 0.75), (-5.0, 9.75)])
england_route = np.array([(-6.0, -0.5), (10.0, -0.5), (1.0, 14.5), (-6.0, -0.5)])

models = ['gmm_fremen_c_{}_p_{}'.format(c, p) for c in [3, 5] for p in [3, 5]]

edges_of_cell = [1, 1]
speed = 1.

for model in models:
    print('testing  ' + model)
    # creating path for the outputs of planner
    try:
        os.mkdir('../results/lot_of_models_testing_eng')
    except OSError as error:
        pass

    output_path = '../results/lot_of_models_testing_eng/' + str(model) + '_output.txt'
    if os.path.exists(output_path):
        os.remove(output_path)

    for time in times:
        path_model = config.england_models + str(model) + '/' + str(time) + '_model.txt'
        test_data_path = config.england + 'time_windows/' + str(time) + '_test_data.txt'

        result = tester.test_model(path_model=path_model, path_data=test_data_path, testing_time=time,
                                   model_name=model, route=england_route,
                                   remove_walls=False, edges_of_cell=edges_of_cell,
                                   speed=speed, create_video=False)
        with open(output_path, 'a') as file:
            file.write(' '.join(str(value) for value in result) + '\n')

for model in models:

    print('\n statistics of ' + model)
    output_path = '../results/lot_of_models_testing_eng/' + str(model) + '_output.txt'
    summarize_new.summarize(output_path)