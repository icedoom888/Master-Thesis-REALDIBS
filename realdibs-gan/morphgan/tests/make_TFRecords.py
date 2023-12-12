
import tensorflow as tf
import os
import skimage.io as io


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Create a dictionary with features that may be relevant.
def image_features(real_image, rendered_image, name):
    image_shape = real_image.shape
    real_raw = real_image.tostring()
    rendered_raw = rendered_image.tostring()

    feature = {
        'filename': _bytes_feature(str.encode(name)),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'tar_image': _bytes_feature(real_raw),
        'render_image': _bytes_feature(rendered_raw),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def load2(image_file):
    print('##########################################################################')
    print(image_file)
    real_image = io.imread(image_file)
    f_name = os.path.split(image_file)
    image_file = PATHfake + os.path.split(f_name[0])[1] + '/' + f_name[1]
    input_image = io.imread(image_file)
    return input_image, real_image, f_name[1]


####################

dataset = input('Type dataset name..')
base_dir = '/mnt/soarin/TFRecords/'
train_record = 'train.tfrecords'
test_record = 'test.tfrecords'

if dataset == 'sullens':
    PATHfake = '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/'
    PATH = '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/'

elif dataset == 'trinity':
    PATHfake = '/media/alberto/DATA/project/realdibs/results/render/trinity/pairedSamename/'
    PATH = '/media/alberto/DATA/project/realdibs/data/trinity/undistorted/'

else:
    print('Invalid name provided. Terminated.')
    exit()

record_path = base_dir + dataset + '/'
if not os.path.exists(record_path):
    os.makedirs(record_path)

train_record = record_path + train_record
test_record = record_path + test_record

train_dataset = [PATH+'train/'+f for f in os.listdir(PATH+'train/')]
test_dataset = [PATH+'test/'+f  for f in os.listdir(PATH+'test/')]

with tf.io.TFRecordWriter(train_record) as writer:
    for el in train_dataset:
        input_image, real_image, name = load2(el)
        tf_example = image_features(real_image, input_image, name)
        writer.write(tf_example.SerializeToString())

with tf.io.TFRecordWriter(test_record) as writer:
    for el in test_dataset:
        input_image, real_image, name = load2(el)
        tf_example = image_features(real_image, input_image, name)
        writer.write(tf_example.SerializeToString())
