import logging
import os
import sys

import tensorflow as tf
from absl import app
from data.data_manager_mesh import DataManager
from models.morpher import Morpher
from options.morphoptions import MorphOptions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logging.disable(logging.WARNING)


def generate_dataset(argv):
    """
    Reproduce the whole dataset after running Geometrical Morphing step.
    Produces new dataset in data_manager.output_path.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu  # set active GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)  # don't let tensorflow directly allocate all memory on GPU

    # Load Data
    data_manager = DataManager(FLAGS)
    data_manager.initialize()
    mesh = data_manager.mesh  # get mesh
    logs_dir = data_manager.logs_dir  # get logs directory

    # Build Model
    model = Morpher(FLAGS, mesh, logs_dir)

    # Loop over the Training set
    for train_el in data_manager.train:

        cam_names = train_el['img_name'].numpy()  # get the camera name of the current training picture

        generated_images = model.generate_images(cam_names)  # generate output images.

        # Store generated images in a "train" sub-folder in the output path.
        # We get element 0 because generated_images is a batch.
        output_str = 'train/' + train_el['img_name'].numpy()[0].decode()
        data_manager.store_generated_img_str(generated_images[0], output_str)

    # Loop over the Test set
    for test_el in data_manager.test:

        cam_names = test_el['img_name'].numpy()  # get the camera name of the current test picture

        generated_images = model.generate_images(cam_names)   # Generate output images using the model

        # store generated images in a "test" sub-folder in the output path
        # We get element 0 because generated_images is a batch
        output_str = 'test/' + test_el['img_name'].numpy()[0].decode()
        data_manager.store_generated_img_str(generated_images[0], output_str)


if __name__ == '__main__':

    config_path = sys.argv[1]  # get config name
    FLAGS = MorphOptions().initialize(config_path)  # load config details
    app.run(generate_dataset)  # run generate_dataset function
