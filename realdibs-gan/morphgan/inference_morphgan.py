import tensorflow as tf
from data.data_manager_morpher import DataManager
import logging, os
from absl import app
from options.artgan_options import Artgan_options
import sys


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
logging.disable(logging.WARNING)


def test_morphgan(argv):
    """
    Inference routine for MorphGAN model
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

    # load model from checkpoint
    model.load_checkpoint()

    # Test or Inference
    for test_el in data_manager.test:  # for each element in the test (or inference) set

        name = test_el['img_name'].numpy()[0].decode()  # get and decode image name
        in_img = test_el['in_img']
        morph_img = test_el['morph_img']

        in_img = tf.concat([in_img, morph_img], axis=3)  # stack input and morphed images together

        # Generate output image using the model
        generated_img = model.generate_images(in_img)
        print('Generated..')

        # save generated images with respective name
        data_manager.store_generated_img_str(generated_img, name)

    print('Inference/Testing completed.')


if __name__ == '__main__':

    config_path = sys.argv[1]  # get config name
    FLAGS = Artgan_options().initialize(config_path)  # load config details
    app.run(test_morphgan)  # run train function
