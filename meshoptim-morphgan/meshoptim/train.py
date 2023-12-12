import logging
import os
import sys
import time

import tensorflow as tf
from absl import app
from data.data_manager_mesh import DataManager
from models.morpher import Morpher
from options.morphoptions import MorphOptions
from metrics.compute_metrics import MetricsAnalyser

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
logging.disable(logging.WARNING)


def train(argv):
    """
    Mesh optimisation training function.
    Produces optimised mesh in data_manager.output_path
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

    # Create metric MetricsAnalyser
    metrics_analyzer = MetricsAnalyser(writer=model.writer)

    # Training loop
    for epoch in range(FLAGS.initial_epoch, FLAGS.epochs):
        start = time.time()
        print('Epoch: ', epoch)

        # Train
        for train_el in data_manager.train:  # for each batch in the training set

            cam_names = train_el['img_name'].numpy()  # get camera name
            targets = tf.math.multiply(train_el['tar_img'], train_el['mask'])  # mask out target images

            model.train_step(targets, cam_names)  # training step

        # Test
        if (epoch + 1) % FLAGS.test_freq == 0:  # test every 'test_freq' steps.

            for test_el in data_manager.test:  # for each batch in the test set

                cam_names = test_el['img_name'].numpy()  # get camera name
                targets = tf.math.multiply(test_el['tar_img'], test_el['mask'])  # mask out target images

                generated_images = model.generate_images(cam_names)  # generate images using the model

                if FLAGS.test_one:  # store generated images only when only one test image
                    data_manager.store_generated_img(generated_images[0], targets[0], epoch)  # save generated image
                    metrics_analyzer.compute_metrics(targets[0], generated_images[0], epoch)  # compute metrics

        # Save mesh
        if (epoch + 1) % FLAGS.mesh_freq == 0:  # store morphed mesh every 'mesh_freq' steps.
            modified_mesh = model.get_morphed_mesh()  # get morphed mesh from model
            data_manager.store_mesh(modified_mesh, epoch)  # store morphed mesh

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


if __name__ == '__main__':

    config_path = sys.argv[1]  # get config name
    FLAGS = MorphOptions().initialize(config_path)  # load config details
    app.run(train)  # run train function
