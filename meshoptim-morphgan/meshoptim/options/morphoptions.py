from absl import flags

from .baseoptions import BaseOptions


class MorphOptions(BaseOptions):
    """
    Class containing additional flags for the morpher application
    """

    def __init__(self, *args, **kwargs):
        super(MorphOptions, self).__init__(**kwargs)

        flags.DEFINE_string('mesh_path', '', 'Path to mesh .ply file.')
        flags.DEFINE_string('camera_path', '', 'Path to cameras .txt file')

        flags.DEFINE_integer('num_filters', 8, 'Rate of trials that discriminator will win on average.')
        flags.DEFINE_integer('encoder_filter_dims', 3, 'Number of filters in first conv layer of generator(encoder-decoder).')
        flags.DEFINE_integer('output_dim', 3, 'Number of filters in first conv layer of discriminator.')

        flags.DEFINE_integer('mesh_freq', 500, 'Frequency of mesh saving.')
