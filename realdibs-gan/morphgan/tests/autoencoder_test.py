import tensorflow as tf
import time
import logging, os
import datetime
import sys
import numpy as np
from models.artgan import Transformer


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
logging.disable(logging.WARNING)

class Resblock(tf.keras.Model):
    def __init__(self, dim, ks=3, s=1, *args, **kwargs):
        super(Resblock, self).__init__(self, *args, **kwargs)

        p = int((ks - 1) / 2)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        self.stack=[
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

class PadReflect(tf.keras.layers.Layer):
    def __init__(self, pad_dim, *args, **kwargs):
        super(PadReflect, self).__init__(self, *args, **kwargs)
        self.p = pad_dim


    def call(self, inputs):
        #print('Encoder:')
        return tf.pad(inputs, [[0, 0], [self.p, self.p], [self.p, self.p], [0, 0]], "REFLECT")

class Upsample(tf.keras.layers.Layer):

    def __init__(self, filters, ks, initializer, strides=2, padding='same', *args, **kwargs):
        super(Upsample, self).__init__(self, *args, **kwargs)

        self.strides = strides

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=ks,
                                   strides=1, padding=padding,
                                   kernel_initializer=initializer)


    def call(self, inputs):
        x = inputs
        x = tf.image.resize(images=inputs, size=tf.shape(x)[1:3] * self.strides, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)

        return x

class InstanceNorm(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #print(input_shape)
        depth = input_shape[3]
        self.scale = self.add_weight(name='scale', shape=[depth], initializer=tf.compat.v1.random_normal_initializer(1.0, 0.02), trainable=True)
        self.offset = self.add_weight(name='offset', shape=[depth], initializer=tf.constant_initializer(0.0), trainable=True)
        self.epsilon = 1e-5
        super(InstanceNorm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #print(x.shape)
        mean, variance = tf.nn.moments(x, axes=[1, 2])
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x-mean)*inv
        return self.scale*normalized + self.offset

    def compute_output_shape(self, input_shape):
        return input_shape

class DownsampleInst(tf.keras.layers.Layer):
    def __init__(self, filters, size, strides=2, padding='same', *args, **kwargs):
        super(DownsampleInst, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

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

class UpsampleInst(tf.keras.layers.Layer):
    def __init__(self, filters, size, strides = 2, padding='same', *args, **kwargs):
        super(UpsampleInst, self).__init__(self, *args, **kwargs)

        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)

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

class Encoder(tf.keras.layers.Layer):
    def __init__(self, gf_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        print(gf_dim)
        self.stack=[
            InstanceNorm(),
            PadReflect(pad_dim=15),
            DownsampleInst(filters=gf_dim, size=3, strides=1, padding='valid'),
            DownsampleInst(filters=gf_dim, size=3, strides=2, padding='valid'),
            DownsampleInst(filters=gf_dim*2, size=3, strides=2, padding='valid'),
            DownsampleInst(filters=gf_dim*4, size=3, strides=2, padding='valid'),
            DownsampleInst(filters=gf_dim*8, size=3, strides=2, padding='valid'),
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, gf_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        res_dim = gf_dim*8
        self.stack=[
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            Resblock(dim=res_dim, ks=3, s=1),
            UpsampleInst(filters= gf_dim*8, size=3, strides=2, padding='same'),
            UpsampleInst(filters= gf_dim*4, size=3, strides=2, padding='same'),
            UpsampleInst(filters= gf_dim*2, size=3, strides=2, padding='same'),
            UpsampleInst(filters= gf_dim, size=3, strides=2, padding='same'),
            PadReflect(pad_dim=3),
            tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1,
             padding='valid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), activation='sigmoid')
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.stack:
            x = layer(x)
        return (x*2.) -1.

class Generator(tf.keras.Model):
    def __init__(self, gf_dim, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.encoder = Encoder(gf_dim)
        self.decoder = Decoder(gf_dim)

    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def transformer_loss(output_photo, input_photo, transformer):
    def mse_criterion(in_, target):
        return tf.reduce_mean((in_-target)**2)

    transformed_output = transformer(output_photo)
    transformed_input = transformer(input_photo)

    return mse_criterion(transformed_output, transformed_input)

generator = Generator(32, name='Generator')
transformer = Transformer()
generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("/mnt/soarin/results/", "baselines", "sullens","autoencoder", current_time)
logs_dir = os.path.join(run_dir, 'logs/')
writer = tf.summary.create_file_writer(logs_dir)

##################
in_path = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/in_imgs/'
tar_path = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/tar_imgs/'

#for step in range(27*5556):
for step in range(20):
    in_img = 'in_%06d.npy' %step

    start = time.time()
    print('Step: ', step)
    in_image = np.load(in_path + in_img)

    with tf.GradientTape() as gen_tape:
        output_photo = generator(in_image)
        with writer.as_default():

            tf.summary.image('in_out', tf.concat([tf.cast((in_image * 0.5 + 0.5)*255, tf.uint8),
                                        tf.cast((output_photo * 0.5 + 0.5)*255, tf.uint8)], axis=2),
                                        step=step)

        trans_loss = transformer_loss(output_photo, in_image, transformer)
        #trans_loss = tf.reduce_mean(tf.abs(output_photo-in_image))
        gen_loss = 100. * trans_loss

    '''
    for el in generator.trainable_variables:
        print(el._handle_name)
    '''


    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_weights)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_weights))

    print ('Time taken for step {} is {} sec\n'.format(step + 1, time.time()-start))
