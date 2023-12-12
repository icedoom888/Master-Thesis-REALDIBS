import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np


class sullens_dataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.4.0')
    SUPPORTED_VERSIONS = [
        # tfds.core.Version('0.1.0'), # Paired in and tar images
        # tfds.core.Version('0.2.0'), # Un-Paired in and tar images
        tfds.core.Version('0.3.0'),  # Paired in and tar images with masks available
        tfds.core.Version('0.4.0'),  # Un-Paired in and tar images with masks available
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("This is the Sullens dataset for RealDibs project. It contains 37 images. The "
                         "images are kept at their original dimensions of 4000x3000 px."),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                "img_name": tfds.features.Text(),
                "in_img": tfds.features.Image(),
                "tar_img": tfds.features.Image(),
                "mask": tfds.features.Image()
            }),
            supervised_keys=("rendered_image", "target_image"),
        )

    def _split_generators(self, dl_manager):
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/train/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/train/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/'
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/test/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/'
                },
            ),
            tfds.core.SplitGenerator(
                name='TEST_ONE',
                gen_kwargs={
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/'
                                   'test_one/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test_one/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/test_one/'
                },
            ),
        ]

    def _generate_examples(self, render_path, real_path, mask_path):
        # Read the input data out of the source files
        render_list = tf.io.gfile.listdir(render_path)
        real_list = tf.io.gfile.listdir(real_path)

        real_list = np.sort(real_list)
        render_list = np.sort(render_list)

        real_list = np.roll(real_list, 5)

        for f_name, g_name in zip(render_list, real_list):
            yield f_name, {
                "img_name": f_name,
                "in_img": "%s%s" % (render_path, f_name),
                "tar_img": "%s%s" % (real_path, g_name),
                "mask": "%s%s" % (mask_path, f_name)
            }


if __name__ == '__main__':
    sullens_train = tfds.load(name='sullens_dataset', data_dir='/mnt/soarin/Datasets', split=tfds.Split.TRAIN)
