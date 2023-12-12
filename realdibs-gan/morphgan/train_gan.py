import tensorflow as tf
from data.data_manager import DataManager
import time
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
# setup the data structure
    # filenames identical across folders
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
# create a config with make_config.py
# add <datasetname>_dataset.py to train_morphgan.py
# create tfds dataset and copy locally (for caching)
# run it :)
# edit spreadsheet runtimes/results

# KNOWN ISSUES:
#
# tensorflow/core/grappler/optimizers/meta_optimizer.cc:502]
# layout failed: Invalid argument: Size of values 0 does not match size of permutation 4.


def train_gan(argv):
    """
    Training routine for GAN models
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
        print('No model available for the method chosen.')
        return

    # Try loading from checkpoint
    try:
        model.load_checkpoint()
        print('Model loaded from: ', FLAGS.ckpt_path)

    except:
        print('No checkpoint loaded.')
        pass

    # Create metric MetricsAnalyser
    metrics_analyzer = MetricsAnalyser(writer=model.writer)

    # Training loop
    for epoch in range(FLAGS.initial_epoch, FLAGS.epochs):
        start = time.time()
        print(sys.argv[1], 'epoch: ', epoch)

        # Train
        for train_el in data_manager.train:  # for each batch in the training set

            in_img = tf.math.multiply(train_el['in_img'], train_el['mask'])  # mask out input images
            tar = tf.math.multiply(train_el['tar_img'], train_el['mask'])  # mask out target images

            # model training step
            model.train_step(in_img, tar, tf.constant(epoch, dtype=tf.int64), train_el['mask'])

        # Test
        if epoch % FLAGS.test_freq == 0:  # test every 'test_freq' steps

            for test_el in data_manager.test:

                in_img = tf.math.multiply(test_el['in_img'], test_el['mask'])  # mask out input images
                tar_img = tf.math.multiply(test_el['tar_img'], test_el['mask'])  # mask out target images

                generated_img = model.generate_images(in_img)  # generate images using the model
                generated_img = tf.math.multiply(generated_img, test_el['mask'])  # mask out output images

                data_manager.store_generated_img(generated_img, in_img, tar_img, epoch)  # save generated images
                # model.log_test(in_img, tar_img, generated_img, epoch) # log test step
                metrics_analyzer.compute_metrics(tar_img, generated_img, epoch)  # compute metrics

        # Save model checkpoint
        if (epoch + 1) % FLAGS.checkpoint_freq == 0:  # store model every checkpoint_freq' steps
            model.checkpoint.save(file_prefix=checkpoint_dir + '/' + str(epoch) + '/')

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


if __name__ == '__main__':

    config_path = sys.argv[1]  # get config name
    FLAGS = Artgan_options().initialize(config_path)  # load config details
    app.run(train_gan)  # run train function
