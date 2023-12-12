import json

from absl import flags


class BaseOptions:
    """
    Class containing all possible flags for the model with some default values

    Methods
    -------
    initialize( config_path=None )
        Initializes all flags with the respective values of the passed config file in 'config_path'
    """

    def __init__(self):

        self.FLAGS = flags.FLAGS

        # Output Flags
        flags.DEFINE_string('out_dir', None, 'Output path')
        flags.DEFINE_string('task', None, 'Baseline or Progress')
        flags.DEFINE_string('method', None, 'Model name')

        # Dataset Flags
        flags.DEFINE_string('dataset_name', None, 'Dataset Name')
        flags.DEFINE_string('dataset_version', '0.1.0', 'Dataset Version')
        flags.DEFINE_string('data_dir', '/mnt/soarin/Datasets', 'Datasets Directory')
        flags.DEFINE_integer('buffer_size', 500, 'Buffer size used during Dataset Shuffling')
        flags.DEFINE_integer('prefetch_size', 100, 'Number of elements to prefetch')
        flags.DEFINE_integer('initial_epoch', 0, 'Starting epoch.')

        # Training Flags
        flags.DEFINE_string('gpu', "0", 'Active GPU to train the model on.')
        flags.DEFINE_integer('epochs', 2000, 'Number of training epochs.')
        flags.DEFINE_integer('patch_size', 512, 'Training pathces size. Used when random cropping during preprocessing.')
        flags.DEFINE_integer('batch_size', 1, 'Batch size used during training.')
        flags.DEFINE_integer('checkpoint_freq', 10, 'Frequency of epochs to store checkpoints.')
        flags.DEFINE_float('lr', 0.0002, 'Learning rate fo the optimizer')

        # Testing Flags
        flags.DEFINE_bool('test_one', True, 'True to use one image as test set. False the use the full test set.')
        flags.DEFINE_integer('test_freq', 100, 'Frequency with whom testing is carried out.')

        # Input Flags
        flags.DEFINE_integer('height', 3000, 'Input image height')
        flags.DEFINE_integer('width', 4000, 'Input image width')

        # Model Flags
        flags.DEFINE_integer('network_downscale', 256, 'Factor of dimentional reduction of the network.')
        flags.DEFINE_integer('LAMBDA', 100, 'Lambda parameter used in the Generator Loss')

        # Inference Flags
        flags.DEFINE_string('ii_dir', '', 'Inference images directory path.')
        flags.DEFINE_string('save_dir', '', 'Store inference results in this directory.')
        flags.DEFINE_integer('ckpt_number', None, 'Checkpoint to load.')

    def initialize(self, config_path):
        """
        Initializes all flags with the respective values of the passed config file in 'config_path'

        Parameters
        ----------
        config_path: str
            path to the config file

        Returns
        -------
        FLAGS:
            set of initialized flags

        """

        with open(config_path, 'r') as f:
            config = json.load(f)
            for name, value in config.items():
                try:
                    self.FLAGS[name].value = value
                except KeyError:
                    pass

        return self.FLAGS
