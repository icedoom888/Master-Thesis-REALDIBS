import tensorflow as tf
from data.data_manager import DataManager
import matplotlib.pyplot as plt
import time
import logging, os, sys
from absl import app
from options.artgan_options import Artgan_options
from options.base_options import Base_options
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
logging.disable(logging.WARNING)

import numpy as np

def denormalize(image):
    return tf.cast((image * 0.5 + 0.5)*255, tf.uint8)

def main(args):
    data_manager = DataManager(FLAGS)
    data_manager.initialize()

    for epoch in range(4):
        fig=plt.figure(figsize=(17, 14))
        columns = 3
        rows = 3
        i = 1
        for train_el in data_manager.train:
            fig.add_subplot(rows, columns, i)
            plt.imshow(denormalize(train_el['in_img'][0]))
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(denormalize(train_el['tar_img'][0]))
            fig.add_subplot(rows, columns, i+2)
            plt.imshow(denormalize(train_el['mask'][0]))

            i+=3
            if i == 10:
                break
        filename = 'fig_%d.png' %epoch
        plt.savefig(fname=filename)
        break


if __name__ == '__main__':

    config_path = 'configs/' + sys.argv[1] + '.json'
    FLAGS = Artgan_options().initialize(config_path)

    app.run(main)
