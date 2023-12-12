import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np


class sullens_morphed_dataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.2.0')
    SUPPORTED_VERSIONS = [
        tfds.core.Version('0.1.0'),  # Paired input and target images with masks available.
                                     # in images are generated from per vertex direct optimization.
        tfds.core.Version('0.2.0'),  # Paired input, target and rendering images with masks available.
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("This is the morphed version of the Sullens dataset for RealDibs project."
                         "The images are obtained by per vertex optimization."),
            features=tfds.features.FeaturesDict({
                "img_name": tfds.features.Text(),
                "morph_img": tfds.features.Image(),
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
                    "morphed_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/morphed/train/',
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/train/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/train/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/'
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "morphed_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/morphed/test/',
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/test/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/'
                },
            ),
            tfds.core.SplitGenerator(
                name='TEST_ONE',
                gen_kwargs={
                    "morphed_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/morphed/test_one/',
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/'
                                   'test_one/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test_one/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/test_one/'
                },
            ),
        ]

    def _generate_examples(self, morphed_path, render_path, real_path, mask_path):
        # Read the input data out of the source files
        morph_list = tf.io.gfile.listdir(morphed_path)
        render_list = tf.io.gfile.listdir(render_path)
        real_list = tf.io.gfile.listdir(real_path)

        morph_list = np.sort(morph_list)
        render_list = np.sort(render_list)
        real_list = np.sort(real_list)

        for morph_name, in_name, tar_name in zip(morph_list, render_list, real_list):
            yield morph_name, {
                "img_name": tar_name,
                "morph_img": "%s%s" % (morphed_path, morph_name),
                "in_img": "%s%s" % (render_path, in_name),
                "tar_img": "%s%s" % (real_path, tar_name),
                "mask": "%s%s" % (mask_path, in_name)
            }


if __name__ == '__main__':
    sullens_train = tfds.load(name='sullens_morphed_dataset', data_dir='/mnt/soarin/Datasets', split=tfds.Split.TRAIN)
