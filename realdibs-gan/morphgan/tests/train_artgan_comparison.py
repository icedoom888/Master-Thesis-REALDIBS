import tensorflow as tf
from data.data_manager import DataManager
import matplotlib.pyplot as plt
import time
import logging, os
from absl import app
from options.artgan_options import Artgan_options
from options.base_options import Base_options
import datetime
import sys
import numpy as np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
logging.disable(logging.WARNING)

def store_generated_img(prediction, input, tar, epoch, output_dir):
    p= tf.cast((prediction * 0.5 + 0.5)*255, tf.uint8)
    img = tf.image.encode_png(p[0])
    tf.io.write_file(output_dir + 'gen_%05d' % epoch + '.png', img )

    if epoch == 0:
        p= tf.cast((tar * 0.5 + 0.5)*255,tf.uint8)
        img = tf.image.encode_png(p[0])
        tf.io.write_file(output_dir + 'tar' + '.png', img )

        p= tf.cast((input * 0.5 + 0.5)*255,tf.uint8)
        img = tf.image.encode_png(p[0])
        tf.io.write_file(output_dir + 'in' + '.png', img )

def main(args):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(FLAGS.out_dir, FLAGS.task, FLAGS.dataset_name, FLAGS.method, current_time)

    checkpoint_dir = os.path.join(run_dir, 'checkpoints/')
    output_dir = os.path.join(run_dir, 'output/')
    logs_dir = os.path.join(run_dir, 'logs/')

    from models.artgan import Artgan
    model = Artgan(FLAGS, checkpoint_dir, logs_dir)

    in_path = '/media/alberto/DATA/project/realdibs/code/tmp_dataset_scale/in_imgs/'
    tar_path = '/media/alberto/DATA/project/realdibs/code/tmp_dataset_scale/tar_imgs/'

    in_test = np.load('/media/alberto/DATA/project/realdibs/code/tmp_dataset_scale/in_test.npy')
    tar_test = np.load('/media/alberto/DATA/project/realdibs/code/tmp_dataset_scale/tar_test.npy')

    for step in range(205721):
    #for step in range(100):
        in_img = 'in_%06d.npy' %step
        tar_img = 'tar_%06d.npy' %step

        start = time.time()
        print('Step: ', step)
        in_image = np.load(in_path + in_img)
        tar_image = np.load(tar_path + tar_img)

        model.train_step(in_image, tar_image, tf.constant(step, dtype=tf.int64))
        print ('Time taken for step {} is {} sec\n'.format(step + 1, time.time()-start))

        if (step % 1000) == 0:
            generated_img = model.generate_images(in_test)
            store_generated_img(generated_img, in_test, tar_test, step, output_dir)


if __name__ == '__main__':

    config_path = 'configs/' + sys.argv[1] + '.json'
    FLAGS = Artgan_options().initialize(config_path)

    app.run(main)
