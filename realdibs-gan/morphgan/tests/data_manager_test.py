import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import datetime
import os

#from .sullens_dataset import sullens_dataset
from data.data_manager import DataManager
from data.ruemonge_animation_dataset import ruemonge_animation_dataset
from data.sullens_dataset_v2 import sullens_dataset
from data.trinity_dataset import trinity_dataset
from data.ruemonge_dataset import ruemonge_dataset
from data.church_tower_dataset import church_tower_dataset

class Data_manager_test(DataManager):

    def initialize(self):

        if self.flags.test_one:
            split='TEST_ONE'
        else:
            split=tfds.Split.TEST
        self.test = tfds.load(name=self.flags.dataset_name+'_dataset:' + self.flags.dataset_version, data_dir=self.flags.data_dir, split=split)

        #Create Output and Checkpoint directories
        self.make_output_dirs()

        #Preprocess test set
        self.compute_paddings()
        if self.flags.test_one:
            self.prepare_test_one()
        else:
            self.prepare_test_full()


    def store_generated_img(self, prediction, img_name):
        p= tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
        p = self.remove_pad(p)
        img = tf.image.encode_png(p[0])

        tf.io.write_file(self.output_dir + img_name, img)

    def preprocess_test(self, datapoint):
        input_image = tf.cast(datapoint['in_img'], tf.float32)

        if self.flags.method == 'artgan':
            input_image = self.resize(input_image)
            input_image = self.pad_test(input_image)
            input_image = self.random_jitter(input_image)

        else:
            input_image = self.pad_test(input_image)

        input_image = self.normalize(input_image)

        datapoint['in_img'] = input_image

        return datapoint


    @tf.function
    def resize(self, input_image):

        if self.flags.resize:
            # ARTGAN uses 1800px in their impl
            factor = self.flags.resize_to  / self.flags.width
            height = tf.cast(factor * self.flags.height, dtype=tf.int32)
            width = tf.cast(factor * self.flags.width, dtype=tf.int32)

            input_image = tf.image.resize(input_image, (height, width),
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image

    def pad_test(self, input_image):
        padded_image = tf.pad([input_image], self.paddings, "CONSTANT")
        return padded_image[0]

    def random_crop(self, input_image):

      cropped_image = tf.image.random_crop(
          input_image, size=[1, self.flags.patch_size, self.flags.patch_size, 3])

      return cropped_image[0]

    # normalizing the images to [-1, 1]
    def normalize(self, input_image):
      input_image = (input_image / 127.5) - 1

      return input_image

    @tf.function
    def random_jitter(self, input_image):
        # randomly cropping to 512 x 512 x 3
        input_image = self.random_crop(input_image)
        if tf.random.uniform(()) > 0.5: # random mirroring
            input_image = tf.image.flip_left_right(input_image)
        return input_image
