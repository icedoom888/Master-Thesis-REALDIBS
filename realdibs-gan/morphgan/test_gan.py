import tensorflow as tf
from data.data_manager import DataManager
import logging
import os
from absl import app
from options.artgan_options import Artgan_options
import sys
from metrics.compute_metrics import MetricsAnalyser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logging.disable(logging.WARNING)


# HOWTO
#
# create a config with make_config.py
# setup the data structure
    # file structure is always:
    # data/<dataset>
    # data/<dataset>/undistorted/test/
    # data/<dataset>/undistorted/test_one/
    # data/<dataset>/undistorted/train/
    #
    # results/render/<dataset>/paired/test/
    # results/render/<dataset>/paired/test_one/
    # results/render/<dataset>/paired/train/
    # results/render/<dataset>/masks/test/
    # results/render/<dataset>/masks/test_one/
    # results/render/<dataset>/masks/train/
# make sure your files are divisble by 2 (no 1067 stuff)
# run masks.py to create masks from RGBA renders
# create a <datasetname>_dataset.py
# run it :)
#

# KNOWN ISSUES:
#
# tensorflow/core/grappler/optimizers/meta_optimizer.cc:502]
# layout failed: Invalid argument: Size of values 0 does not match size of permutation 4.


def test_gan(argv):
    """
    Test routine for GAN models
    """

    os.system('clear')

    # Set active GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    # Load Data
    data_manager = DataManager(FLAGS)
    data_manager.initialize()
    checkpoint_dir = data_manager.checkpoint_dir  # get checkpoint dir
    logs_dir = data_manager.logs_dir  # get logs dir

    # Build Model
    if FLAGS.method == 'pix2pix':
        from models.pix2pix import Pix2pix
        model = Pix2pix(FLAGS, checkpoint_dir, logs_dir)

    elif FLAGS.method == 'cyclegan':
        from models.cyclegan import CycleGan
        model = CycleGan(FLAGS, checkpoint_dir, logs_dir)

    elif FLAGS.method == 'artgan':
        from models.artgan import Artgan
        model = Artgan(FLAGS, checkpoint_dir, logs_dir)

    else:
        print('No model available for the method choosen.')
        return

    # load model from checkpoint
    model.load_checkpoint()

    # Testing loop
    for test_el in data_manager.test:  # for each element in the test set

        in_img = tf.math.multiply(test_el['in_img'], test_el['mask'])  # mask out input images
        tar_img = tf.math.multiply(test_el['tar_img'], test_el['mask'])  # mask out target images

        generated_img = model.generate_images(in_img)  # generate images using the model
        generated_img = tf.math.multiply(generated_img, test_el['mask'])  # mask out output images

        data_manager.store_test_img(generated_img, in_img, tar_img, test_el['img_name'][0])  # save generated image

    print('Testing completed.')


if __name__ == '__main__':
    config_path = sys.argv[1]  # get config name
    FLAGS = Artgan_options().initialize(config_path)  # load config details
    app.run(test_gan)  # run test function
