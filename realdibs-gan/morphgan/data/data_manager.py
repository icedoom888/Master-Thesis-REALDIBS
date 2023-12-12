import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import datetime
import os

#from .sullens_dataset import sullens_dataset
from .sullens_dataset_v2 import sullens_dataset
from .trinity_dataset import trinity_dataset
from .ruemonge_dataset import ruemonge_dataset
from .church_tower_dataset import church_tower_dataset
from .merton3_dataset import merton3_dataset


class DataManager:
    """
    A class used to load and handle data during training and inference.

    Attributes
    ----------
    flags: dict
        a dictionary containing all parameters necessary for the script

    train: tfds.Dataset
        training split of the chosen dataset

    test: tfds.Dataset
        test split of the chosen dataset

    checkpoint_dir: str
        a path to the checkpoint directory

    output_dir: str
        a path to the output directory for generated images

    logs_dir: str
        a path to the logs directory


    Methods
    -------
    name()
        returns the class name

    initialize()
        initializes train, test and mesh attributes

    load_mesh( mesh_path=None )
        loads .tfrecords mesh into dictionary

    prepare_train()
        training set dataset operations to load data during training

    prepare_test_one()
        test set dataset operations to load data during testing, when only the test set has only one image

    prepare_test_full()
        test set dataset operations to load data during testing

    preprocess_train( datapoint=None )
        pre-processing operations to be done on each data-point during training

    preprocess_test( datapoint=None )
        pre-processing operations to be done on each data-point during testing

    compute_paddings()
        Computes padding to ensure network downscaling will work properly

    make_output_dirs()
        creates the following directories: checkpoint_dir, output_dir, logs_dir and mesh_dir

    resize( input_image=None, real_image=None, mask=None )
        resizes input, target and mask images to a given factor. Necessary for the artgan network

    random_crop( input_image=None, real_image=None, mask=None )
        randomly crops patches of 'flags.patch_size' from the given images

    normalize( input_image=None, real_image=None, mask=None )
        normalizes the given images

    random_jitter( input_image=None, real_image=None, mask=None )
        combines the pre-processing operations of random cropping and random horizontal flipping

    pad_test( input_image=None, real_image=None, mask=None )
        padding function to be used for pre-processing at test time

    remove_pad( image=None )
        removes padding from a given image

    store_generated_img( prediction=None, input=None, tar=None, epoch=None )
        stores the given 'prediction' image into the 'output_dir' directory based on the 'epoch' value

    store_train_img( prediction=None, input=None, tar=None, epoch=None, batch=None )
        Helper function to be used during debugging.
        Stores preprocessed training image batches to test if pr-processing works correctly.

    store_test_img( prediction=None, input=None, tar=None, epoch=None, batch=None )
        Helper function to be used during debugging.
        Stores preprocessed testing image batches to test if pr-processing works correctly.

    """

    def __init__(self, flags):
        """
        Parameters
        ----------
        flags: dict
            a dictionary containing all parameters necessary for the script
        """

        self.flags = flags
        return

    def name(self):
        return 'Data_loader' + self.flags.dataset_name

    def initialize(self):
        """
        initializes train and test attributes
        """

        split = tfds.Split.TRAIN
        self.train = tfds.load(name=self.flags.dataset_name+'_dataset:' + self.flags.dataset_version,
                               data_dir=self.flags.data_dir, split=split)

        if self.flags.test_one:
            split = 'TEST_ONE'
        else:
            split = tfds.Split.TEST
        self.test = tfds.load(name=self.flags.dataset_name+'_dataset:' + self.flags.dataset_version,
                              data_dir=self.flags.data_dir, split=split)

        # Pre-process training set
        self.prepare_train()

        # Pre-process test set
        self.compute_paddings()

        if self.flags.test_one:
            self.prepare_test_one()
        else:
            self.prepare_test_full()

        # Create output directories
        self.make_output_dirs()

    def prepare_train(self):
        """
        training set dataset operations to load data during training
        """

        # Warning:
        # The calling iterator did not fully read the dataset being cached.
        # In order to avoid unexpected truncation of the dataset,
        # the partially cached contents of the dataset will be discarded.
        # This can happen if you have an input pipeline similar to
        # `dataset.cache().take(k).repeat()`.
        # You should use `dataset.take(k).cache().repeat()` instead.

        self.train = self.train.map(self.preprocess_train, num_parallel_calls=24)
        self.train = self.train.shuffle(self.flags.buffer_size)
        self.train = self.train.batch(self.flags.batch_size)

        return

    def prepare_test_one(self):
        """
        test set dataset operations to load data during testing, when only the test set has only one image
        """
        self.test = self.test.map(self.preprocess_test, num_parallel_calls=8)
        self.test = self.test.batch(1)
        # Cache after because only one image
        self.test = self.test.take(1).cache().prefetch(10)

        return

    def prepare_test_full(self):
        """
        test set dataset operations to load data during testing
        """
        self.test = self.test.cache()
        self.test = self.test.map(self.preprocess_test, num_parallel_calls=8)
        self.test = self.test.batch(1).prefetch(20)

        return

    def preprocess_train(self, datapoint):
        """
        pre-processing operations to be done on each data-point during training

        Parameters
        ----------
        datapoint: dict
            a datapoint of the dataset, keys: in_img, tar_img, mask

        Returns
        -------
        datapoint: dict
            the processed datapoint

        """

        input_image = tf.cast(datapoint['in_img'], tf.float32)
        real_image = tf.cast(datapoint['tar_img'], tf.float32)
        try:
            mask = tf.cast(datapoint['mask'], tf.float32)
        except KeyError:
            mask = tf.zeros(input_image.shape)

        if self.flags.method == 'artgan':
            # Artgan requires smaller input images. We resize the fixed size.
            input_image, real_image, mask = self.resize(input_image, real_image, mask)

        else:
            pass

        # pre-process: random crop and jitter plus normalisation
        input_image, real_image, mask = self.random_jitter(input_image, real_image, mask)
        input_image, real_image, mask = self.normalize(input_image, real_image, mask)

        datapoint['in_img'] = input_image
        datapoint['tar_img'] = real_image
        datapoint['mask'] = mask

        return datapoint

    def preprocess_test(self, datapoint):
        """
        pre-processing operations to be done on each data-point during testing

        Parameters
        ----------
        datapoint: dict
            a datapoint of the dataset, keys: in_img, tar_img, mask

        Returns
        -------
        datapoint: dict
            the processed datapoint

        """

        input_image = tf.cast(datapoint['in_img'], tf.float32)
        real_image = tf.cast(datapoint['tar_img'], tf.float32)

        try:
            mask = tf.cast(datapoint['mask'], tf.float32)
        except KeyError:
            mask = tf.zeros(input_image.shape)

        # the artgan network requires cropping during testing as well
        if self.flags.method == 'artgan':
            input_image, real_image, mask = self.resize(input_image, real_image, mask)
            input_image, real_image, mask = self.pad_test(input_image, real_image, mask)
            input_image, real_image, mask = self.random_jitter(input_image, real_image, mask)
        else:
            input_image, real_image, mask = self.pad_test(input_image, real_image, mask)

        input_image, real_image, mask = self.normalize(input_image, real_image, mask)

        datapoint['in_img'] = input_image
        datapoint['tar_img'] = real_image
        datapoint['mask'] = mask

        return datapoint

    def compute_paddings(self):
        """
        computes padding to ensure network downscaling will work properly
        """

        from math import ceil
        self.h_pad = int(((ceil(self.flags.width/self.flags.network_downscale) * self.flags.network_downscale)
                          - self.flags.width)/2)
        self.v_pad = int(((ceil(self.flags.height/self.flags.network_downscale) * self.flags.network_downscale)
                          - self.flags.height)/2)
        self.paddings = [[0, 0], [self.v_pad, self.v_pad], [self.h_pad, self.h_pad], [0, 0]]

        return

    def make_output_dirs(self):
        """
        creates the following directories: checkpoint_dir, output_dir, logs_dir and mesh_dir
        """

        # Get current time to make experiment output folder
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.flags.out_dir, self.flags.task, self.flags.dataset_name,
                               self.flags.method, current_time)

        # make directory paths
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints/')
        self.output_dir = os.path.join(run_dir, 'output/')
        self.logs_dir = os.path.join(run_dir, 'logs/')

    @tf.function
    def resize(self, input_image, real_image, mask):
        """
        resizes input, target and mask images to a given factor. Necessary for the artgan network

        Parameters
        ----------
        input_image: tf.image
            input image

        real_image: tf.image
            target image

        mask: tf.image
            mask for both input and target image

        Returns
        -------
        resized received images

        """

        if self.flags.resize:
            # ARTGAN uses 1800px in their impl
            factor = self.flags.resize_to / self.flags.width
            height = tf.cast(factor * self.flags.height, dtype=tf.int32)
            width = tf.cast(factor * self.flags.width, dtype=tf.int32)

            input_image = tf.image.resize(input_image, (height, width),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_image = tf.image.resize(real_image, [height, width],
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            mask = tf.image.resize(mask, [height, width],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image, mask

    def random_crop(self, input_image, real_image, mask):
        """
        randomly crops patches of 'flags.patch_size' from the original images

        Parameters
        ----------
        input_image: tf.image
            input image

        real_image: tf.image
            target image

        mask: tf.image
            mask for both input and target image

        Returns
        -------
        cropped patches of each input image

        """

        stacked_image = tf.stack([input_image, real_image, mask], axis=0)  # stack 3 images
        # Crop the stacked image
        cropped_image = tf.image.random_crop(stacked_image, size=[3, self.flags.patch_size, self.flags.patch_size, 3])

        return cropped_image[0], cropped_image[1], cropped_image[2]

    def normalize(self, input_image, real_image, mask):
        """
        normalizes the given images

        Parameters
        ----------
        input_image: tf.image
            input image

        real_image: tf.image
            target image

        mask: tf.image
            mask for both input and target image

        Returns
        -------
        normalised images
        """

        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        mask = mask / 255.

        return input_image, real_image, mask

    @tf.function
    def random_jitter(self, input_image, real_image, mask):
        """
        combines the pre-processing operations of random cropping and random horizontal flipping.

        input_image: tf.image
            input image

        real_image: tf.image
            target image

        mask: tf.image
            mask for both input and target image

        Returns
        -------
        randomly cropped and flipped patches of all received images

        """

        input_image, real_image, mask = self.random_crop(input_image, real_image, mask)
        if tf.random.uniform(()) > 0.5:  # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
            mask = tf.image.flip_left_right(mask)

        return input_image, real_image, mask

    def pad_test(self, input_image, real_image, mask):
        """
        padding function to be used for pre-processing at test time

        Parameters
        ----------
        input_image: tf.image
            input image

        real_image: tf.image
            target image

        mask: tf.image
            mask for both input and target image

        Returns
        -------
        padded received images
        """

        stacked_image = tf.stack([input_image, real_image, mask], axis=0)
        padded_image = tf.pad(stacked_image, self.paddings, "CONSTANT")
        return padded_image[0], padded_image[1], padded_image[2]

    def remove_pad(self, image):
        """
        removes padding from a given image
        """
        return image[:, self.v_pad:image.shape[1]-self.v_pad, self.h_pad:image.shape[2]-self.h_pad, :]

    def store_generated_img(self, prediction, input, tar, epoch):
        """
        stores the given 'prediction' image into the 'output_dir' directory based on the 'epoch' value

        Parameters
        ----------
        prediction: tf.image
            image to save

        input: tf.image
            input image to save if 'epoch' is 0

        tar: tf.image
            target image to save if 'epoch' is 0

        epoch: int
            current epoch, used in the output image name
        """

        # Save prediction image in 'output_dir'
        # Image name is based on 'epoch'
        p = tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
        p = self.remove_pad(p)  # remove pad
        img = tf.image.encode_png(p[0])  # encode png
        tf.io.write_file(self.output_dir + 'gen_%05d' % epoch + '.png', img)  # writes the image to file

        if epoch == 0:  # store target and input images if epoch is 0

            p = tf.cast((tar * 0.5 + 0.5)*255, tf.uint8)
            p = self.remove_pad(p)  # remove pad
            img = tf.image.encode_png(p[0])  # encode png
            tf.io.write_file(self.output_dir + 'tar' + '.png', img)  # writes the image to file

            p = tf.cast((input * 0.5 + 0.5)*255, tf.uint8)
            p = self.remove_pad(p)  # remove pad
            img = tf.image.encode_png(p[0])  # encode png
            tf.io.write_file(self.output_dir + 'in' + '.png', img)  # writes the image to file

    def store_train_img(self, prediction, input, tar, epoch, batch):
        """
        Helper function to be used during debugging.
        Stores preprocessed training image batches to test if pre-processing works correctly.

        Parameters
        ----------
        prediction: tf.image
            image to save

        input: tf.image
            input image to save if 'epoch' is 0

        tar: tf.image
            target image to save if 'epoch' is 0

        epoch: int
            current epoch, used in the output image name

        batch: int
            current bacth, used in the output image name

        """
        # debug: write all train images to folder instead of tb

        p = tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
        img = tf.image.encode_png(p[0])

        tf.io.write_file(self.output_dir + 'epoch_%05d/' %epoch +'gen_%05d' % batch + '.png', img )

        if epoch == 0:
            p = tf.cast((tar * 0.5 + 0.5)*255,tf.uint8)
            p = self.remove_pad(p)
            img = tf.image.encode_png(p[0])
            tf.io.write_file(self.output_dir + 'tar' + '.png', img )

            p = tf.cast((input * 0.5 + 0.5)*255,tf.uint8)
            p = self.remove_pad(p)
            img = tf.image.encode_png(p[0])
            tf.io.write_file(self.output_dir + 'in' + '.png', img )

    def store_test_img(self, prediction, input, tar, img_name):
        """
        Helper function to be used during debugging.
        Stores preprocessed test image batches to test if pre-processing works correctly.

        Parameters
        ----------
        prediction: tf.image
            image to save

        input: tf.image
            input image to save if 'epoch' is 0

        tar: tf.image
            target image to save if 'epoch' is 0

        img_name: str
            name of the test image to save

        """

        p= tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
        p = self.remove_pad(p)
        img = tf.image.encode_png(p[0])
        tf.io.write_file(self.output_dir + 'test_' + img_name, img)

        p= tf.cast((tar * 0.5 + 0.5)*255,tf.uint8)
        p = self.remove_pad(p)
        img = tf.image.encode_png(p[0])
        tf.io.write_file(self.output_dir + 'tar_' + img_name, img)

        p= tf.cast((input * 0.5 + 0.5)*255,tf.uint8)
        p = self.remove_pad(p)
        img = tf.image.encode_png(p[0])
        tf.io.write_file(self.output_dir + 'in_' + img_name, img)
