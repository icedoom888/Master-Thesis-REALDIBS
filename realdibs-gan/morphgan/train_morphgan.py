import tensorflow as tf
from data.data_manager_morpher import DataManager
import logging, os
from absl import app
from options.artgan_options import Artgan_options
import sys
from metrics.compute_metrics import MetricsAnalyser
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
logging.disable(logging.WARNING)


def train_morphgan(argv):
    """
    Training routine for MorphGAN model
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
        from models.pix2pix_morph import Pix2pix
        model = Pix2pix(FLAGS, checkpoint_dir, logs_dir)

    else:
        print('No model available for the method choosen.')
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

    with tqdm(total=FLAGS.epochs-FLAGS.initial_epoch) as progress_bar:

        # Training loop
        for epoch in range(FLAGS.initial_epoch, FLAGS.epochs):

            # Train
            for train_el in data_manager.train:  # for each batch in the training set

                in_img = tf.math.multiply(train_el['in_img'], train_el['mask'])  # mask out input images
                morph_img = tf.math.multiply(train_el['morph_img'], train_el['mask'])  # mask out morphed images
                tar = tf.math.multiply(train_el['tar_img'], train_el['mask'])  # mask out target images

                in_img = tf.concat([in_img, morph_img], axis=3)  # stack input and morphed images together

                # model training step
                model.train_step(in_img, tar, tf.constant(epoch, dtype=tf.int64), train_el['mask'])

            # Test
            if epoch % FLAGS.test_freq == 0:  # test every 'test_freq' steps

                for test_el in data_manager.test:

                    in_img = tf.math.multiply(test_el['in_img'], test_el['mask'])  # mask out input images
                    morph_img = tf.math.multiply(test_el['morph_img'], test_el['mask'])  # mask out morphed images
                    tar_img = tf.math.multiply(test_el['tar_img'], test_el['mask'])  # mask out target images

                    in_img = tf.concat([in_img, morph_img], axis=3)  # stack input and morphed images together

                    # Generate output images using the model
                    generated_img = model.generate_images(in_img)
                    generated_img = tf.math.multiply(generated_img, test_el['mask'])  # mask out output images

                    data_manager.store_generated_img(generated_img, in_img, tar_img, epoch)  # save generated images
                    metrics_analyzer.compute_metrics(tar_img, generated_img, epoch)  # compute metrics

            # Save model checkpoint
            if (epoch + 1) % FLAGS.checkpoint_freq == 0:  # store model every 'checkpoint_freq' steps
                model.checkpoint.save(file_prefix=checkpoint_dir + '/' + str(epoch) + '/')

            progress_bar.update(1)  # update progress


if __name__ == '__main__':

    config_path = sys.argv[1]  # get config name
    FLAGS = Artgan_options().initialize(config_path)  # load config details
    app.run(train_morphgan)  # run train function

