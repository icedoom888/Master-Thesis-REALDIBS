
import json


class Config:

    def import_jsonfile(self, f):

        # load file and parse
        with open(f) as f:
            json_dict = json.load(f)

        for key, value in json_dict.items():
            setattr(self, key, value)

    def export_jsonfile(self, f):

        # convert to json-able
        dt = {}
        dt.update(vars(self))

        # save to stream
        with open(f, "w") as file:
            json.dump(dt, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    """ Create config file.
    Change parameters for the various experiments """

    config = Config()
    base_path = '/media/alberto/DATA/project/realdibs/meshoptim/configs/'

    # method = 'mesh_optim_vertex_colors'
    method = 'MorphGAN_animation'
    dataset_name = 'sullens_animation'

    # f = base_path + method + '_' + dataset_name + '_epochs_' + str(config.epochs) + '_generate_at_' +
    # str(config.initial_epoch) + '.json'
    # f = base_path + method + '_' + dataset_name + '_epochs_' + str(config.epochs) + '.json'
    f = base_path + method + '_' + dataset_name + '_generate.json'

    # comment out if you want to create a specific config name
    # f = base_path + method + '_' + dataset_name + '.json'
    # f = base_path + 'config_to_gen_dataset.json'

    #############################

    # Dataset Flags
    config.dataset_name = dataset_name
    # config.dataset_version = '0.2.0'
    config.data_dir = '/media/alberto/DATA/project/realdibs/datasets/'
    config.buffer_size = 50
    config.prefetch_size = 100
    config.initial_epoch = 350

    # Training Flags
    gpu = "1"
    config.gpu = gpu
    config.epochs = 1000
    # config.batch_size = 3
    config.batch_size = 1
    config.checkpoint_freq = 100
    config.lr = 0.0002

    # Testing Flags
    config.test_one = True
    config.test_freq = 100

    # Input Flags
    config.height = 3000
    config.width = 4000

    # Output Flags
    config.out_dir = "/mnt/soarin/results/"
    config.task = method
    # config.task = 'morphGAN'
    config.method = method

    # Inference Flags
    config.ckpt_number = config.epochs

    # Rendering Flags
    # config.mesh_path = '/mnt/soarin/results/mesh_optim_vertex_colors/sullens/mesh_optim_vertex_colors/20191219-134502/mesh/morphed_mesh_499.tfrecords'
    # config.mesh_path = "/mnt/soarin/data/trinity/mesh/trinity_dense_mesh_simplified.tfrecords"
    # config.mesh_path = "/mnt/soarin/data/trinity_crop/trinity_crop.tfrecords"
    config.mesh_path = '/mnt/soarin/results/mesh_optim_vertex_colors/sullens/mesh_optim_vertex_colors/20191219-134502/mesh/morphed_mesh_499.tfrecords'
    # config.camera_path = "/mnt/soarin/data/sullens/mesh/trinity_opk.txt"
    config.camera_path = "/mnt/soarin/results/render/sullens/animation_1583930186.85482_opk_rotation_copied_from_opk.txt"
    config.mesh_freq = 100

    #############################

    config.export_jsonfile(f)
