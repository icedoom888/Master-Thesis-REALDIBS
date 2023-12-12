import tensorflow as tf


class PadReflect(tf.keras.layers.Layer):
    """
    Reflecting padding keras layer
    """
    def __init__(self, pad_dim, *args, **kwargs):
        """
        Parameters
        ----------
        pad_dim: int
            padding dimension
        """
        super(PadReflect, self).__init__(self, *args, **kwargs)
        self.p = pad_dim

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.p, self.p], [self.p, self.p], [0, 0]], "REFLECT")


class Upsample(tf.keras.layers.Layer):
    """
    Up-sampling keras layer. Image resizing + Convolution
    """

    def __init__(self, filters, ks, initializer, strides=2, padding='same', *args, **kwargs):
        """
        Parameters
        ----------
        filters: int
            number on convolution filters

        ks: int
            convolution kernel size

        initializer:
            convolution kernel initializer

        strides: int
            convolution stride

        padding:
            convolution padding dimension

        """

        super(Upsample, self).__init__(self, *args, **kwargs)

        self.strides = strides

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=ks,
                                           strides=1, padding=padding,
                                           kernel_initializer=initializer)

    def call(self, inputs):
        x = inputs
        x = tf.image.resize(images=inputs, size=tf.shape(x)[1:3] * self.strides,
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)

        return x


class InstanceNorm(tf.keras.layers.Layer):
    """
    Instance normalisation keras layer.
    """

    def __init__(self, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        depth = input_shape[3]
        self.scale = self.add_weight(name='scale', shape=[depth],
                                     initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02, dtype=tf.float32), trainable=True)
        self.offset = self.add_weight(name='offset', shape=[depth],
                                      initializer=tf.constant_initializer(0.0), trainable=True)
        self.epsilon = 1e-5
        super(InstanceNorm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2])
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x-mean)*inv
        return self.scale*normalized + self.offset

    def compute_output_shape(self, input_shape):
        return input_shape


class Resblock(tf.keras.Model):
    """
    Resblock keras layer.
    """

    def __init__(self, dim, ks=3, s=1, *args, **kwargs):
        """
        Parameters
        ----------
        dim: int
            number on convolution filters
        ks: int
            convolution kernel size
        s: int
            convolution stride

        """

        super(Resblock, self).__init__(self, *args, **kwargs)

        p = int((ks - 1) / 2)
        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)

        self.stack = [
            PadReflect(pad_dim=p),
            tf.keras.layers.Conv2D(dim, ks, s, padding='valid', kernel_initializer=initializer, use_bias=False),
            InstanceNorm(),
            tf.keras.layers.ReLU(),
            PadReflect(pad_dim=p),
            tf.keras.layers.Conv2D(dim, ks, s, padding='valid', kernel_initializer=initializer, use_bias=False),
            InstanceNorm()
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x + inputs


class DownsampleBatch(tf.keras.Model):
    """
    Down-sample keras layer with batch normalisation.
    """

    def __init__(self, filters, size, strides = 2, padding='same', apply_batchnorm=True, *args, **kwargs):
        """
        Parameters
        ----------
        filters: int
            number on convolution filters

        size: int
            convolution kernel size

        strides: int
            convolution stride

        padding: int
            convolution padding dimension

        apply_batchnorm: bool
            if apply batch normalisation

        """

        super(DownsampleBatch, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)

        self.stack = [
            tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding,
                                   kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x


class DownsampleInst(tf.keras.layers.Layer):
    """
    Down-sample keras layer with instance normalisation.
    """

    def __init__(self, filters, size, strides=2, padding='same', *args, **kwargs):
        """
        Parameters
        ----------
        filters: int
            number on convolution filters

        size: int
            convolution kernel size

        strides: int
            convolution stride

        padding: int
            convolution padding dimension

        """

        super(DownsampleInst, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)

        self.stack=[
            tf.keras.layers.Conv2D(filters=filters, kernel_size=size, strides=strides,
                                   padding=padding, kernel_initializer=initializer, bias_initializer=None),
            InstanceNorm(),
            tf.keras.layers.ReLU()
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x


class UpsampleBatch(tf.keras.Model):
    """
    Up-sampling keras layer with batch normalisation
    """

    def __init__(self, filters, size, strides=2, padding='same', apply_dropout=False, *args, **kwargs):
        """
        Parameters
        ----------
        filters: int
            number on convolution filters

        size: int
            convolution kernel size

        strides: int
            convolution stride

        padding: int
            convolution padding dimension

        apply_dropout: bool
            if apply dropout

        """

        super(UpsampleBatch, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)

        self.stack=[
            tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding=padding,
                                            kernel_initializer=initializer, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.ReLU()
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x


class UpsampleInst(tf.keras.layers.Layer):
    """
    Up-sampling keras layer with instance normalisation
    """

    def __init__(self, filters, size, strides=2, padding='same', *args, **kwargs):
        """
        Parameters
        ----------
        filters: int
            number on convolution filters

        size: int
            convolution kernel size

        strides: int
            convolution stride

        padding: int
            convolution padding dimension

        """

        super(UpsampleInst, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)

        self.stack=[
            Upsample(filters=filters, ks=size, strides=strides, padding=padding, initializer=initializer),
            InstanceNorm(),
            tf.keras.layers.ReLU()
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x


class DownsampleInstLeaky(tf.keras.Model):
    """
    Down-samling keras layer with instance normalisation and leaky-relu
    """

    def __init__(self, filters, size, strides=2, padding='same', *args, **kwargs):
        """
        Parameters
        ----------
        filters: int
            number on convolution filters

        size: int
            convolution kernel size

        strides: int
            convolution stride

        padding: int
            convolution padding dimension

        """

        super(DownsampleInstLeaky, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)

        self.stack=[
            tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding,
                                   kernel_initializer=initializer, use_bias=False),
            InstanceNorm(),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x
