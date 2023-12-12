import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    """
    Down-sampling layer

    Parameters
    ----------
    filters: int
        number on convolution filters

    size: int
            convolution kernel size

    apply_batchnorm: bool
        if apply batch normalisation or not

    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """
    Up-sampling layer

    Parameters
    ----------
    filters: int
        number on convolution filters

    size: int
        convolution kernel size

    apply_dropout: bool
        if apply dropout or not

    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class UNetGenerator(tf.keras.Model):
    """
    Generator network with Unet auto-encoder architecture.
    Used in pix2pix and cyclegan models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample(128, 4),  # (bs, 64, 64, 128)
            downsample(256, 4),  # (bs, 32, 32, 256)
            downsample(512, 4),  # (bs, 16, 16, 512)
            downsample(512, 4),  # (bs, 8, 8, 512)
            downsample(512, 4),  # (bs, 4, 4, 512)
            downsample(512, 4),  # (bs, 2, 2, 512)
            downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        self.up_stack = [
            upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4),  # (bs, 16, 16, 1024)
            upsample(256, 4),  # (bs, 32, 32, 512)
            upsample(128, 4),  # (bs, 64, 64, 256)
            upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        self.last = tf.keras.layers.Conv2DTranspose(3, 4,
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                                    activation='tanh')  # (bs, 256, 256, 3)

        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        skips = []
        x = inputs
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = self.concat([x, skip])

        x = self.last(x)
        return x


class UNetDiscriminator(tf.keras.Model):
    """
    Discriminator network with convolutional architecture.
    Used in pix2pix and cyclegan models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.arch = [
            downsample(64, 4, False),  # (bs, 128, 128, 64)
            downsample(128, 4),  # (bs, 64, 64, 128)
            downsample(256, 4),  # (bs, 32, 32, 256)

            tf.keras.layers.ZeroPadding2D(),  # (bs, 34, 34, 256)
            tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                   use_bias=False),  # (bs, 31, 31, 512)

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.ZeroPadding2D(),  # (bs, 33, 33, 512)

            tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02)),
            # (bs, 30, 30, 1)

        ]

    def call(self, inputs, training=None, mask=None):
        x = tf.concat(inputs, axis=0)
        for layer in self.arch:
            x = layer(x)
        return x
