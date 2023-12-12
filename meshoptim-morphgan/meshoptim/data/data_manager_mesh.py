import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import trimesh
from tensorflow_graphics.notebooks.mesh_segmentation_dataio import _parse_mesh_data, _parse_tfex_proto, \
    adjacency_from_edges
from .sullens_mesh_dataset import sullens_mesh_dataset
from .merton_mesh_dataset import merton_mesh_dataset
from .trinity_mesh_dataset import trinity_mesh_dataset
from .trinity_crop_mesh_dataset import trinity_crop_mesh_dataset

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

    mesh: dict
        a dictionary containing all mesh information loaded from .tfrecords file

    checkpoint_dir: str
        a path to the checkpoint directory

    output_dir: str
        a path to the output directory for generated images

    logs_dir: str
        a path to the logs directory

    mesh_dir: str
        a path to the mesh output directory

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
        pre-processing operations to be done on each data-point during testing,

    normalize( image=None )
        normalizes a given image

    make_output_dirs()
        creates the following directories: checkpoint_dir, output_dir, logs_dir and mesh_dir

    store_generated_img( prediction=None, tar=None, epoch=None )
        stores the given 'preciction' image into the 'output_dir' directory based on the 'epoch' value

    store_generated_img_str( prediction=None, name=None)
        stores the given 'preciction' image into the 'output_dir' directory based on the 'name' value

    store_mesh( mesh=None, epoch=None )
        stores the given 'mesh' into the 'mesh_dir' directory

    """

    def __init__(self, flags):
        """
        Parameters
        ----------
        flags: dict
            a dictionary containing all parameters necessary for the script
        """
        self.flags = flags

    def name(self):
        return 'Data_loader' + self.flags.dataset_name

    def initialize(self):
        """
        initializes train, test and mesh attributes
        """
        # Load train split of the dataset
        split = tfds.Split.TRAIN
        self.train = tfds.load(name=self.flags.dataset_name + '_mesh_dataset:' + self.flags.dataset_version,
                               data_dir=self.flags.data_dir, split=split)

        # Load test split of the dataset
        if self.flags.test_one:
            split = 'TEST_ONE'
        else:
            split = tfds.Split.TEST
        self.test = tfds.load(name=self.flags.dataset_name + '_mesh_dataset:' + self.flags.dataset_version,
                              data_dir=self.flags.data_dir, split=split)

        # Load mesh
        self.mesh = self.load_mesh(self.flags.mesh_path)

        # Preprocess training set
        self.prepare_train()

        # Preprocess test set
        if self.flags.test_one:
            self.prepare_test_one()
        else:
            self.prepare_test_full()

        # Create output directories
        self.make_output_dirs()

    def _parse_mesh_data_wrapper(self, mean_center):
        return partial(_parse_mesh_data, mean_center=mean_center)

    def load_mesh(self, mesh_path):
        """
        loads .tfrecords mesh into dictionary

        Parameters
        ----------
        mesh_path: str
            a path to the mesh .tfrecords file

        Returns
        -------
        mesh: dict
            a dictionary containing all mesh information loaded from .tfrecords file
        """

        # Load .tfrecords file
        raw_dataset = tf.data.TFRecordDataset(mesh_path)
        parse_mesh = self._parse_mesh_data_wrapper(mean_center=False)
        parsed_dataset = raw_dataset.map(_parse_tfex_proto)
        mesh_dataset = parsed_dataset.map(parse_mesh)

        for mesh_data in mesh_dataset:  # can only read tfrecords in a for loop, even if only one mesh is present

            # Compute neighbors using adjacency from edges function
            mesh_data['neighbors'] = adjacency_from_edges(tf.expand_dims(mesh_data['edges'], axis=0),
                                                          tf.expand_dims(mesh_data['edge_weights'], axis=0),
                                                          tf.expand_dims(mesh_data['num_edges'], axis=0),
                                                          tf.expand_dims(mesh_data['num_vertices'], axis=0))

            # Build mesh dictionary
            mesh = dict(
                  vertices=mesh_data['vertices'],
                  triangles=mesh_data['triangles'],
                  neighbors=mesh_data['neighbors'],
                  num_triangles=mesh_data['num_triangles'],
                  num_vertices=mesh_data['num_vertices'],
                  colors=tf.cast(mesh_data['labels'], dtype=tf.float32))

            break # break for loop because only one mesh

        return mesh

    def prepare_train(self):
        """
        training set dataset operations to load data during training
        """
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
        self.test = self.test.map(self.preprocess_test, num_parallel_calls=8)
        self.test = self.test.cache()
        self.test = self.test.batch(1).prefetch(20)

        return

    def preprocess_train(self, datapoint):
        """
        pre-processing operations to be done on each data-point during training

        Parameters
        ----------
        datapoint: dict
            a datapoint of the dataset, keys: tar_img, mask

        Returns
        -------
        datapoint: dict
            the processed datapoint

        """

        real_image = tf.cast(datapoint['tar_img'], tf.float32)
        mask = tf.cast(datapoint['mask'], tf.float32)/255.
        # datapoint['tar_img'] = tf.image.flip_left_right(real_image)
        # datapoint['mask'] = tf.image.flip_left_right(mask)

        # TODO: some datasets are flipped, make flag
        datapoint['tar_img'] = real_image
        datapoint['mask'] = mask

        return datapoint

    def preprocess_test(self, datapoint):
        """
        pre-processing operations to be done on each data-point during testing

        Parameters
        ----------
        datapoint: dict
            a datapoint of the dataset, keys: tar_img, mask

        Returns
        -------
        datapoint: dict
            the processed datapoint

        """

        return self.preprocess_train(datapoint)

    def normalize(self, image):
        """
        normalizes a given image
        """
        return (image / 127.5) - 1.

    def make_output_dirs(self):
        """
        creates the following directories: checkpoint_dir, output_dir, logs_dir and mesh_dir
        """

        # Get current time to make experiment output folder
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.flags.out_dir, self.flags.task, self.flags.dataset_name,
                               self.flags.method, current_time)

        # make directory paths
        # mesh_dir has to be created because of a specific mesh saving function which requires an existing directory
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints/')
        self.output_dir = os.path.join(run_dir, 'output/')
        self.logs_dir = os.path.join(run_dir, 'logs/')
        self.mesh_dir = os.path.join(run_dir, 'mesh/')
        os.makedirs(self.mesh_dir)

    def store_generated_img(self, prediction, tar, epoch):
        """
        stores the given 'preciction' image into the 'output_dir' directory based on the 'epoch' value

        Parameters
        ----------
        prediction: tf.image
            image to save

        tar: tf.image
            target image to save if 'epoch' is 0

        epoch: int
            current epoch, used in the output image name
        """

        # Save prediction image in 'output_dir'
        # Image name is based on 'epoch'
        prediction = tf.cast(prediction, tf.uint8)
        img = tf.image.encode_png(prediction)  # encodes into png
        tf.io.write_file(self.output_dir + 'gen_%05d' % epoch + '.png', img)   # writes the image to file

        if epoch == 0:  # store target image if epoch is 0
            tar = tf.cast(tar, tf.uint8)
            img = tf.image.encode_png(tar)  # encodes into png
            tf.io.write_file(self.output_dir + 'tar' + '.png', img)  # writes the target image to file

    def store_generated_img_str(self, prediction, name):
        """
        stores the given 'preciction' image into the 'output_dir' directory based on the 'name' value

        Parameters
        ----------
        prediction: tf.image
            image to save

        name: str
            name of the image to be saved
        """
        prediction = tf.cast(prediction, tf.uint8)
        prediction = tf.image.flip_left_right(prediction)
        img = tf.image.encode_png(prediction)  # encode into png
        tf.io.write_file(self.output_dir + name, img)  # writes the image to file

    def store_mesh(self, mesh, epoch):
        """
        stores the given 'mesh' into the 'mesh_dir' directory

        Parameters
        ----------
        mesh: trimesh
            mesh to be saved

        epoch: int
            current epoch, used in the output mesh name
        """
        file_name = self.mesh_dir + 'morphed_mesh_' + str(epoch) + '.ply'
        trimesh.exchange.export.export_mesh(mesh, file_name, file_type='ply')  # save trimesh object to .ply file
