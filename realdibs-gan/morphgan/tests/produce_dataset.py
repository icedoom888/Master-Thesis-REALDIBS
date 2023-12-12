import tensorflow as tf
from data.data_manager import DataManager
import matplotlib.pyplot as plt
import time
import logging, os
from absl import app
from options.artgan_options import Artgan_options
from options.base_options import Base_options
from datetime import datetime
import sys
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
logging.disable(logging.WARNING)


def main(args):
    # Load Data
    data_manager = DataManager(FLAGS)
    data_manager.initialize()
    checkpoint_dir = data_manager.checkpoint_dir
    logs_dir = data_manager.logs_dir
    step = 0

    for test_el in data_manager.test:
        f = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/in_imgs/in_test.npy'
        g = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/tar_imgs/tar_test.npy'
        np.save(f, test_el['in_img'])
        np.save(g, test_el['tar_img'])
    

    #create 2 numpy arrays
    for epoch in range(FLAGS.epochs):
        start = time.time()
        for train_el in data_manager.train:
            f = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/in_imgs/in_%06d.npy' %step
            g = '/media/alberto/DATA/project/realdibs/code/tmp_dataset/tar_imgs/tar_%06d.npy' %step
            np.save(f, train_el['in_img'])
            np.save(g, train_el['tar_img'])
            step += 1
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


if __name__ == '__main__':

    config_path = 'configs/' + sys.argv[1] + '.json'
    FLAGS = Artgan_options().initialize(config_path)

    app.run(main)
