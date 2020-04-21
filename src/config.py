
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


filip_config = {
    'train_csv': '/home/filip/Documents/bakalarka/resources/robot_sim_data/training_dataset_all.txt',
    'train_array': '/home/filip/Documents/bakalarka/resources/robot_sim_data/training_dataset_t_x_y.npy',
    'resources': '/home/filip/Documents/bakalarka/resources/robot_sim_data/'
}

config = Struct(**filip_config)