import numpy as np
from config import config
from src.gmm_fremen.gmm_fremen_model import GMMFremenModel
from src.gmm_fremen.save_model_for_experiment import save_prediction_into_model_no_angle


## TOTO JE POMOCNY A NEDULEZITY SKRIPT


def main():
    r_s = [2, 1, 0.5]
    for r in r_s:
        arr = np.load(config.filtered_train + '{}.npy'.format(r))
        model = GMMFremenModel(5)
        model.fit(arr[:, 1:], arr[:, 0], step=10 * 60)
        model.save_model(config.filtered_models + '{}/model'.format(r))

        test_timestamps = np.genfromtxt(config.resources + 'test_times.txt')
        predictions = model.predict_for_grid(test_timestamps, (24, 33), (-9.5, 0.25), 40, cell_dimensions=(0.5, 0.5))
        save_prediction_into_model_no_angle(predictions, test_timestamps, config.filtered_models + '{}/'.format(r))


if __name__ == '__main__':
    main()
