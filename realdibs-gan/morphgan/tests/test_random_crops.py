import tensorflow as tf

IMG_WIDTH = 512
IMG_HEIGHT = 512

def load2(image_file):
    global PATH, PATHfake
    # trainB
    image = tf.io.read_file(image_file)
    real_image = tf.image.decode_png(image)
    #image_file = PATHfake + 'trainA/sullens_cam_' +  image_file[-12:]

    image_file = PATHfake + tf.strings.split(image_file,'/')[-2] + '/' + tf.strings.split(image_file,'/')[-1]
    image = tf.io.read_file(image_file)
    input_image = tf.image.decode_png(image, channels=3)

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def random_crop(input_image, real_image):

  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load2(image_file)
  #input_image, real_image = normalize(input_image, real_image)
  input_image, real_image = random_crop(input_image, real_image)

  return input_image, real_image


PATHfake = '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/'
PATH = '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/'

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)
for i in range(30):
    for example_input, example_target in test_dataset.take(1):
        example_input = tf.image.encode_png(tf.cast(example_input, tf.uint8)[0])
        tf.io.write_file('./crops/' + str(i) + '_in.png', example_input)
        example_target = tf.image.encode_png(tf.cast(example_target, tf.uint8)[0])
        tf.io.write_file('./crops/' + str(i) + '_tar.png', example_target)
