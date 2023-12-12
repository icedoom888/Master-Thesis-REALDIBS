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


def render_animation(argv):
    """
    Inference function used to render animation sequences in the config file.
    This operations requires a camera path and a mesh path in the config file.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu  # set active GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)  # Don't let tensorflow directly allocate all memory on GPU

    # Load Data
    data_manager = DataManager(FLAGS)
    data_manager.make_output_dirs()  # make output directories

    mesh = data_manager.load_mesh(FLAGS.mesh_path)  # get mesh
    logs_dir = data_manager.logs_dir  # get logs directory

    # Build Model
    model = Morpher(FLAGS, mesh, logs_dir)
    cam_dict = model.renderer.cam_dict  # get list of cameras

    for cam_name in cam_dict.keys():  # render view for each camera in the sequence
        print('Rendering: ', cam_name)
        generated_images = model.generate_images([cam_name])  # Generate output image. [cam_name] for batch.

        # Store generated images in the output folder.
        # We get element 0 because generated_images is a batch.
        data_manager.store_generated_img_str(generated_images[0], None, cam_name + '.png')


if __name__ == '__main__':

    config_path = sys.argv[1]  # get config name
    FLAGS = MorphOptions().initialize(config_path)  # load config details
    app.run(render_animation)  # run render_animation function
