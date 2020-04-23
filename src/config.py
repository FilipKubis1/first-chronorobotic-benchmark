
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


filip_config = {
    'train_csv': '/home/filip/Documents/bakalarka/resources/robot_sim_data/training_dataset_all.txt',
    'train_array': '/home/filip/Documents/bakalarka/resources/robot_sim_data/training_dataset_t_x_y.npy',
    'resources': '/home/filip/Documents/bakalarka/resources/robot_sim_data/',
    'gmm_fremen': '/home/filip/Documents/bakalarka/resources/robot_sim_data/model_gmm_fremen/',
    'time_windows': '/home/filip/Documents/bakalarka/resources/robot_sim_data/time_windows/',
    'models': '/home/filip/Documents/bakalarka/resources/robot_sim_data/model_'
}

config = Struct(**filip_config)