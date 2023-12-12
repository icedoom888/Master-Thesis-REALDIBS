from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging, os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
logging.disable(logging.WARNING)
import time
import matplotlib.pyplot as plt
tf.executing_eagerly()

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
OUTPUT_CHANNELS = 3


def load2(image_file):
    image = tf.io.read_file(image_file)
    real_image = tf.image.decode_png(image)
    image_file = PATHfake + tf.strings.split(image_file,'/')[-1]

    image = tf.io.read_file(image_file)
    input_image = tf.image.decode_png(image, channels=3)

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):

  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def random_jitter(input_image, real_image):

    input_image, real_image = random_crop(input_image, real_image)
    import numpy as np
    if np.random.uniform(size=1) > 0.5: # random mirroring
        #print('Flipped!')4096
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = load2(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load2(image_file)
  input_image, real_image = pad_test(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

PATHfake = './dataset_try_fake/'
PATH = './dataset_try_real/'

train_dataset = tf.data.Dataset.list_files(PATH+'*.png')
#train_dataset = train_dataset.map(load_image_train, num_parallel_calls=1)
# Tried to Remove caching to fix memory issue
#train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
#train_dataset = train_dataset.shuffle(BUFFER_SIZE)
#train_dataset = train_dataset.batch(1)
#train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

for epoch in range(int(10e10)):
  print('Epoch: ', epoch)
  # Train
