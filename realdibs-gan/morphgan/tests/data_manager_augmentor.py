import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import datetime
import os
#from .sullens_dataset import sullens_dataset
from data.sullens_dataset_v2 import sullens_dataset
from data.trinity_dataset import trinity_dataset

import numpy as np
import scipy.misc
import cv2
from PIL import Image


class Augmentor():
    def __init__(self,
                 crop_size=(256, 256),
                 scale_augm_prb=0.5, scale_augm_range=0.2,
                 rotation_augm_prb=0.5, rotation_augm_range=0.15,
                 hsv_augm_prb=1.0,
                 hue_augm_shift=0.05,
                 saturation_augm_shift=0.05, saturation_augm_scale=0.05,
                 value_augm_shift=0.05, value_augm_scale=0.05,
                 affine_trnsfm_prb=0.5, affine_trnsfm_range=0.05,
                 horizontal_flip_prb=0.5,
                 vertical_flip_prb=0.5):

        self.crop_size = crop_size

        self.scale_augm_prb = scale_augm_prb
        self.scale_augm_range = scale_augm_range

        self.rotation_augm_prb = rotation_augm_prb
        self.rotation_augm_range = rotation_augm_range

        self.hsv_augm_prb = hsv_augm_prb
        self.hue_augm_shift = hue_augm_shift
        self.saturation_augm_scale = saturation_augm_scale
        self.saturation_augm_shift = saturation_augm_shift
        self.value_augm_scale = value_augm_scale
        self.value_augm_shift = value_augm_shift

        self.affine_trnsfm_prb = affine_trnsfm_prb
        self.affine_trnsfm_range = affine_trnsfm_range

        self.horizontal_flip_prb = horizontal_flip_prb
        self.vertical_flip_prb = vertical_flip_prb

    def __call__(self, image, is_inference=False):
        if is_inference:
            return cv2.resize(image, None, fx=self.crop_size[0], fy=self.crop_size[1], interpolation=cv2.INTER_CUBIC)

        # If not inference stage apply the pipeline of augmentations.
        if self.scale_augm_prb > np.random.uniform():
            image = self.scale(image=image,
                               scale_x=1. + np.random.uniform(low=-self.scale_augm_range, high=-self.scale_augm_range),
                               scale_y=1. + np.random.uniform(low=-self.scale_augm_range, high=-self.scale_augm_range)
                               )

        print(image.shape)
        rows, cols, ch = image.shape
        image = np.pad(array=image, pad_width=[[rows // 4, rows // 4], [cols // 4, cols // 4], [0, 0]], mode='reflect')
        if self.rotation_augm_prb > np.random.uniform():
            image = self.rotate(image=image,
                                angle=np.random.uniform(low=-self.rotation_augm_range*90.,
                                                        high=self.rotation_augm_range*90.)
                                )

        if self.affine_trnsfm_prb > np.random.uniform():
            image = self.affine(image=image,
                                rng=self.affine_trnsfm_range
                                )
        image = image[(rows // 4):-(rows // 4), (cols // 4):-(cols // 4), :]

        # Crop out patch of desired size.
        image = self.crop(image=image,
                          crop_size=self.crop_size
                          )

        if self.hsv_augm_prb > np.random.uniform():
            image = self.hsv_transform(image=image,
                                       hue_shift=self.hue_augm_shift,
                                       saturation_shift=self.saturation_augm_shift,
                                       saturation_scale=self.saturation_augm_scale,
                                       value_shift=self.value_augm_shift,
                                       value_scale=self.value_augm_scale)

        if self.horizontal_flip_prb > np.random.uniform():
            image = self.horizontal_flip(image)

        if self.vertical_flip_prb > np.random.uniform():
            image = self.vertical_flip(image)

        return image

    def scale(self, image, scale_x, scale_y):
        """
        Args:
            image:
            scale_x: float positive value. New horizontal scale
            scale_y: float positive value. New vertical scale
        Returns:
        """
        print(image)
        image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        return image

    def rotate(self, image, angle):
        """
        Args:
            image: input image
            angle: angle of rotation in degrees
        Returns:
        """
        rows, cols, ch = image.shape

        rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, rot_M, (cols, rows))
        return image

    def crop(self, image, crop_size=(256, 256)):
        rows, cols, chs = image.shape
        x = int(np.random.uniform(low=0, high=max(0, rows - crop_size[0])))
        y = int(np.random.uniform(low=0, high=max(0, cols - crop_size[1])))

        image = image[x:x+crop_size[0], y:y+crop_size[1], :]
        # If the input image was too small to comprise patch of size crop_size,
        # resize obtained patch to desired size.
        if image.shape[0] < crop_size[0] or image.shape[1] < crop_size[1]:
            image = scipy.misc.imresize(arr=image, size=crop_size)
        return image

    def hsv_transform(self, image,
                      hue_shift=0.2,
                      saturation_shift=0.2, saturation_scale=0.2,
                      value_shift=0.2, value_scale=0.2,
                      ):

        image = Image.fromarray(image)
        hsv = np.array(image.convert("HSV"), 'float64')

        # scale the values to fit between 0 and 1
        hsv /= 255.

        # do the scalings & shiftings
        hsv[..., 0] += np.random.uniform(-hue_shift, hue_shift)
        hsv[..., 1] *= np.random.uniform(1. / (1. + saturation_scale), 1. + saturation_scale)
        hsv[..., 1] += np.random.uniform(-saturation_shift, saturation_shift)
        hsv[..., 2] *= np.random.uniform(1. / (1. + value_scale), 1. + value_scale)
        hsv[..., 2] += np.random.uniform(-value_shift, value_shift)

        # cut off invalid values
        hsv.clip(0.01, 0.99, hsv)

        # round to full numbers
        hsv = np.uint8(np.round(hsv * 254.))

        # convert back to rgb image
        return np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))


    def affine(self, image, rng):
        rows, cols, ch = image.shape
        pts1 = np.float32([[0., 0.], [0., 1.], [1., 0.]])
        [x0, y0] = [0. + np.random.uniform(low=-rng, high=rng), 0. + np.random.uniform(low=-rng, high=rng)]
        [x1, y1] = [0. + np.random.uniform(low=-rng, high=rng), 1. + np.random.uniform(low=-rng, high=rng)]
        [x2, y2] = [1. + np.random.uniform(low=-rng, high=rng), 0. + np.random.uniform(low=-rng, high=rng)]
        pts2 = np.float32([[x0, y0], [x1, y1], [x2, y2]])
        affine_M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, affine_M, (cols, rows))

        return image

    def horizontal_flip(self, image):
        return image[:, ::-1, :]

    def vertical_flip(self, image):
        return image[::-1, :, :]

class Data_manager:
    def __init__(self, flags):
        self.flags = flags
        self.augmentor = Augmentor(crop_size=[self.flags.patch_size, self.flags.patch_size],
                                       vertical_flip_prb=0.,
                                       hsv_augm_prb=1.0,
                                       hue_augm_shift=0.05,
                                       saturation_augm_shift=0.05, saturation_augm_scale=0.05,
                                       value_augm_shift=0.05, value_augm_scale=0.05, )
        return

    def name(self):
        return 'Data_loader' + self.flags.dataset_name

    def initialize(self):

        if self.flags.dataset_name == 'sullens':
            if self.flags.test_one:
                self.train = tfds.load(name='sullens_dataset:0.2.0', data_dir=self.flags.data_dir, split=tfds.Split.TRAIN)
                self.test = tfds.load(name='sullens_dataset:0.2.0', data_dir=self.flags.data_dir, split='TEST_ONE')

            else:
                self.train = tfds.load(name='sullens_dataset', data_dir=self.flags.data_dir, split=tfds.Split.TRAIN)
                self.test = tfds.load(name='sullens_dataset', data_dir=self.flags.data_dir, split=tfds.Split.TEST)

        elif self.flags.dataset_name == 'trinity':
            self.train = tfds.load(name='trinity_dataset', data_dir=self.flags.data_dir, split=tfds.Split.TRAIN)
            self.train = tfds.load(name='trinity_dataset', data_dir=self.flags.data_dir, split=tfds.Split.TEST)

        else:
            print('Invalid Dataset name..\nDataset not prepared or incorrect.')

        #Create Output and Checkpoint directories
        self.make_output_dirs()

        #Preprocess training set
        self.prepare_train()

        #Preprocess test set
        self.compute_paddings()
        if self.flags.test_one:
            self.prepare_test_one()
        else:
            self.prepare_test_full()

    def prepare_train(self):
        #Cache before map, otherwise the same crops are cached
        self.train = self.train.cache()
        self.train = self.train.map(self.preprocess_train, num_parallel_calls=24)
        self.train = self.train.shuffle(self.flags.buffer_size)
        self.train = self.train.batch(self.flags.batch_size)
        return


    def prepare_test_one(self):
        self.test = self.test.map(self.preprocess_test, num_parallel_calls=8)
        self.test = self.test.batch(1)
        # Cache after because only one image
        self.test = self.test.take(1).cache().prefetch(10)
        return

    def prepare_test_full(self):
        self.test = self.test.map(self.preprocess_test, num_parallel_calls=8)
        self.test = self.test.batch(1)
        self.test = self.test.cache().prefetch(20)
        return

    @tf.function
    def preprocess_train(self, datapoint):
        input_image = tf.cast(datapoint['in_img'], tf.float32)
        real_image = tf.cast(datapoint['tar_img'], tf.float32)

        print(input_image.shape)

        input_image = self.augmentor(input_image)
        real_image = self.augmentor(real_image)

        datapoint['in_img'] = input_image
        datapoint['tar_img'] = real_image

        return datapoint

    @tf.function
    def preprocess_test(self, datapoint):
        input_image = tf.cast(datapoint['in_img'], tf.float32)
        real_image = tf.cast(datapoint['tar_img'], tf.float32)

        input_image = self.augmentor(input_image)
        real_image = self.augmentor(real_image)

        datapoint['in_img'] = input_image
        datapoint['tar_img'] = real_image

        return datapoint

    def compute_paddings(self):
        from math import ceil
        self.h_pad = int(((ceil(self.flags.width/self.flags.network_downscale) * self.flags.network_downscale) - self.flags.width)/2)
        self.v_pad = int(((ceil(self.flags.height/self.flags.network_downscale) * self.flags.network_downscale) - self.flags.height)/2)
        self.paddings = [[0,0],[self.v_pad, self.v_pad],[self.h_pad,self.h_pad],[0,0]]
        return

    def make_output_dirs(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.flags.out_dir, self.flags.task, self.flags.dataset_name, self.flags.method, current_time)

        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints/')
        self.output_dir = os.path.join(run_dir, 'output/')
        self.logs_dir = os.path.join(run_dir, 'logs/')


    def resize(self, input_image, real_image, height, width):
      input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      real_image = tf.image.resize(real_image, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      return input_image, real_image

    def random_crop(self, input_image, real_image):

      stacked_image = tf.stack([input_image, real_image], axis=0)
      cropped_image = tf.image.random_crop(
          stacked_image, size=[2, self.flags.patch_size, self.flags.patch_size, 3])

      return cropped_image[0], cropped_image[1]

    # normalizing the images to [-1, 1]
    def normalize(self, input_image, real_image):
      input_image = (input_image / 127.5) - 1
      real_image = (real_image / 127.5) - 1

      return input_image, real_image

    @tf.function
    def random_jitter(self, input_image, real_image):
        # randomly cropping to 512 x 512 x 3
        input_image, real_image = self.random_crop(input_image, real_image)
        if tf.random.uniform(()) > 0.5: # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
        return input_image, real_image

    def pad_test(self, input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        padded_image = tf.pad(stacked_image, self.paddings, "CONSTANT")
        return padded_image[0], padded_image[1]

    def remove_pad(self, image):
        return image[:, self.v_pad:image.shape[1]-self.v_pad, self.h_pad:image.shape[2]-self.h_pad, :]

    def store_generated_img(self, prediction, input, tar, epoch):
        p= tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
        p = self.remove_pad(p)
        img = tf.image.encode_png(p[0])
        tf.io.write_file(self.output_dir + 'gen_%05d' % epoch + '.png', img )

        if epoch == 0:
            p= tf.cast((tar * 0.5 + 0.5)*255,tf.uint8)
            p = self.remove_pad(p)
            img = tf.image.encode_png(p[0])
            tf.io.write_file(self.output_dir + 'tar' + '.png', img )

            p= tf.cast((input * 0.5 + 0.5)*255,tf.uint8)
            p = self.remove_pad(p)
            img = tf.image.encode_png(p[0])
            tf.io.write_file(self.output_dir + 'in' + '.png', img )

    def store_train_img(self, prediction, input, tar, epoch, batch):
        p= tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
        img = tf.image.encode_png(p[0])
        output_dir = '/media/alberto/DATA/project/realdibs/code/src/train_imgs/'
        tf.io.write_file(output_dir + 'epoch_%05d/' %epoch +'gen_%05d' % batch + '.png', img )

        if epoch == 0:
            p= tf.cast((tar * 0.5 + 0.5)*255,tf.uint8)
            p = self.remove_pad(p)
            img = tf.image.encode_png(p[0])
            tf.io.write_file(output_dir + 'tar' + '.png', img )

            p= tf.cast((input * 0.5 + 0.5)*255,tf.uint8)
            p = self.remove_pad(p)
            img = tf.image.encode_png(p[0])
            tf.io.write_file(output_dir + 'in' + '.png', img )
