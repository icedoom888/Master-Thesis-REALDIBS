import tensorflow as tf
from .module import *


class Encoder(tf.keras.Model):
    """
    Encoder architecture used in the artgan model.
    """

    def __init__(self, gf_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        print(gf_dim)
        self.stack = [
            InstanceNorm(),
            PadReflect(pad_dim=15),
            DownsampleInst(filters=gf_dim, size=3, strides=1, padding='valid'),
            DownsampleInst(filters=gf_dim, size=3, strides=2, padding='valid'),
            DownsampleInst(filters=gf_dim * 2, size=3, strides=2, padding='valid'),
            DownsampleInst(filters=gf_dim * 4, size=3, strides=2, padding='valid'),
            DownsampleInst(filters=gf_dim * 8, size=3, strides=2, padding='valid'),
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x


class Decoder(tf.keras.Model):
    """
    Decoder architecture used in the artgan model.
    """

    def __init__(self, gf_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        res_dim = gf_dim * 8
        self.stack = [
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            UpsampleInst(filters=gf_dim * 8, size=3, strides=2, padding='same'),
            UpsampleInst(filters=gf_dim * 4, size=3, strides=2, padding='same'),
            UpsampleInst(filters=gf_dim * 2, size=3, strides=2, padding='same'),
            UpsampleInst(filters=gf_dim, size=3, strides=2, padding='same'),
            PadReflect(pad_dim=3),
            tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1,
                                   padding='valid',
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02, seed=1),
                                   activation='sigmoid')
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return (x * 2.) - 1.


class Discriminator(tf.keras.Model):
    """
    Multi-level Discriminator architecture used in the artgan model.
    """

    def __init__(self, df_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        self.h0 = DownsampleInstLeaky(filters=df_dim * 2, size=5, strides=2, padding='same')
        self.h0_pred = tf.keras.layers.Conv2D(1, kernel_size=5, strides=1, kernel_initializer=initializer,
                                              padding='same')

        self.h1 = DownsampleInstLeaky(filters=df_dim * 2, size=5, strides=2, padding='same')
        self.h1_pred = tf.keras.layers.Conv2D(1, kernel_size=10, strides=1, kernel_initializer=initializer,
                                              padding='same')

        self.h2 = DownsampleInstLeaky(filters=df_dim * 4, size=5, strides=2, padding='same')

        self.h3 = DownsampleInstLeaky(filters=df_dim * 8, size=5, strides=2, padding='same')
        self.h3_pred = tf.keras.layers.Conv2D(1, kernel_size=10, strides=1, kernel_initializer=initializer,
                                              padding='same')

        self.h4 = DownsampleInstLeaky(filters=df_dim * 8, size=5, strides=2, padding='same')

        self.h5 = DownsampleInstLeaky(filters=df_dim * 16, size=5, strides=2, padding='same')
        self.h5_pred = tf.keras.layers.Conv2D(1, kernel_size=6, strides=1, kernel_initializer=initializer,
                                              padding='same')

        self.h6 = DownsampleInstLeaky(filters=df_dim * 16, size=5, strides=2, padding='same')
        self.h6_pred = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, kernel_initializer=initializer,
                                              padding='same')

    def call(self, inputs, training=None, mask=None):
        # print('Discriminator:')
        x = inputs
        x = self.h0(x)
        # print(x.shape)
        scale_0 = self.h0_pred(x)
        # print('Scale_0:', scale_0.shape)

        # print(x.shape)
        x = self.h1(x)
        scale_1 = self.h1_pred(x)
        # print('Scale_1:', scale_1.shape)

        # print(x.shape)
        x = self.h2(x)

        # print(x.shape)
        x = self.h3(x)
        scale_3 = self.h3_pred(x)
        # print('Scale_3:', scale_3.shape)

        # print(x.shape)
        x = self.h4(x)

        # print(x.shape)
        x = self.h5(x)
        scale_5 = self.h5_pred(x)
        # print('Scale_5:', scale_5.shape)

        # print(x.shape)
        x = self.h6(x)

        # print(x.shape)
        scale_6 = self.h6_pred(x)
        # print('Scale_6:', scale_6.shape)

        return {"scale_0": scale_0,
                "scale_1": scale_1,
                "scale_3": scale_3,
                "scale_5": scale_5,
                "scale_6": scale_6}


class Generator(tf.keras.Model):
    """
    Generator architecture used in the artgan model.
    """

    def __init__(self, gf_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.encoder = Encoder(gf_dim)
        self.decoder = Decoder(gf_dim)

    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.decoder(x)
        return x
