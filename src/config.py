
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


filip_config = {
    'train_csv': '/home/filip/Documents/bakalarka/resources/robot_sim_data/training_dataset_all.txt',
    'train_array': '/home/filip/Documents/bakalarka/resources/robot_sim_data/training_dataset_t_x_y.npy',
    'resources': '/home/filip/Documents/bakalarka/resources/robot_sim_data/',
    'gmm_fremen': '/home/filip/Documents/bakalarka/resources/robot_sim_data/model_gmm_fremen/',
    'time_windows': '/home/filip/Documents/bakalarka/resources/robot_sim_data/time_windows/',
    'models': '/home/filip/Documents/bakalarka/resources/robot_sim_data/model_',
    'filtered_train': '/home/filip/Documents/bakalarka/resources/robot_sim_data/robot_data_random_r',
    'filtered_models': '/home/filip/Documents/bakalarka/resources/robot_sim_data/model_gmm_fremen_r',
    'england_train_txt': '/home/filip/Documents/bakalarka/resources/robot_sim_data/trenovaci_dva_tydny.txt',
    'england_train_arr': '/home/filip/Documents/bakalarka/resources/robot_sim_data/trenovaci_dva_tydny.npy',
    'england_test_txt': '/home/filip/Documents/bakalarka/resources/robot_sim_data/testovaci_dva_dny.txt',
    'england_test_arr': '/home/filip/Documents/bakalarka/resources/robot_sim_data/testovaci_dva_dny.npy',
    'england_test_times': '/home/filip/Documents/bakalarka/resources/robot_sim_data/england/test_times.txt',
    'england': '/home/filip/Documents/bakalarka/resources/robot_sim_data/england/',
    'england_models': '/home/filip/Documents/bakalarka/resources/robot_sim_data/england/model_',
    'fremen_times': '/home/filip/Documents/bakalarka/resources/robot_sim_data/fremen_times.npy',
    'fremen_vals': '/home/filip/Documents/bakalarka/resources/robot_sim_data/fremen_vals.npy',
    'gmm_default_model': '/home/filip/Documents/bakalarka/resources/robot_sim_data/gmm_default_model',
    'fem_gmm': '/home/filip/Documents/bakalarka/resources/robot_sim_data/model_fem_gmm/'

}

config = Struct(**filip_config)