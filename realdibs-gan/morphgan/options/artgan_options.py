from .base_options import Base_options
from absl import flags


class Artgan_options(Base_options):
    """
    Class containing additional flags for the artgan model
    """

    def __init__(self, *args, **kwargs):
        super(Artgan_options, self).__init__(**kwargs)

        flags.DEFINE_float('discr_success', 0.8, 'Rate of trials that discriminator will win on average.')
        flags.DEFINE_integer('ngf', 32, 'Number of filters in first conv layer of generator(encoder-decoder).')
        flags.DEFINE_integer('ndf', 64, 'Number of filters in first conv layer of discriminator.')
        flags.DEFINE_float('dlw', 1., 'Weight of discriminator loss.')
        flags.DEFINE_float('tlw', 100., 'Weight of transformer loss.')
        flags.DEFINE_float('flw', 100., 'Weight of feature loss.')
