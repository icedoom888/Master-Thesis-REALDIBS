import tensorflow as tf
from data.data_manager import DataManager
import matplotlib.pyplot as plt
import time
import logging, os
from absl import app
from options import Base_options

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
logging.disable(logging.WARNING)


def main(args):
    # Load Data
    data_manager = DataManager(FLAGS)
    data_manager.initialize()
    checkpoint_dir = data_manager.checkpoint_dir
    logs_dir = data_manager.logs_dir

    # Build Model
    if FLAGS.method == 'pix2pix':
        from models.pix2pix import Pix2pix
        model = Pix2pix(FLAGS, checkpoint_dir)

    elif FLAGS.method == 'cyclegan':
        from models.cyclegan import CycleGan
        model = CycleGan(FLAGS, checkpoint_dir, logs_dir)

    else:
        print('No model available for the method choosen.')
        return

    for epoch in range(FLAGS.epochs):
        start = time.time()
        print('Epoch: ', epoch)
        # Train
        for train_el in data_manager.train:
            model.train_step(train_el['in_img'], train_el['tar_img'], tf.constant(epoch, dtype=tf.int64))

        # saving (checkpoint) the model
        if (epoch + 1) % FLAGS.test_freq == 0:
            for test_el in data_manager.test:
                #print('Generating: ', test_el['img_name'])
                generated_img = model.generate_images(test_el['in_img'])
                data_manager.store_generated_img(generated_img, test_el['in_img'], test_el['tar_img'], epoch)

        # saving (checkpoint) the model
        if (epoch + 1) % FLAGS.checkpoint_freq == 0:
            model.checkpoint.save(file_prefix = checkpoint_dir)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


if __name__ == '__main__':

    config_path = 'configs/artgan_sullens.json'
    FLAGS = Base_options().initialize(config_path)

    app.run(main)
