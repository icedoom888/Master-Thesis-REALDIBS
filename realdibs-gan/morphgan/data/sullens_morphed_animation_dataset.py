import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np


class sullens_morphed_animation_dataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.2.0')
    SUPPORTED_VERSIONS = [
        # Paired input and target images with masks available.
        # in images are generated from per vertex direct optimization.
        tfds.core.Version('0.2.0'),
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("This is the morphed version of the animation over the Sullens dataset for RealDibs project. "
                         "The images are obtained by per vertex optimization."),
            features=tfds.features.FeaturesDict({
                "img_name": tfds.features.Text(),
                "morph_img": tfds.features.Image(),
                "in_img": tfds.features.Image(),
            }),
            supervised_keys=("rendered_image", "target_image"),
        )

    def _split_generators(self, dl_manager):
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "morphed_path": '/mnt/soarin/results/MorphGAN_animation/sullens_animation/MorphGAN_animation/'
                                    '20200312-124325/output/',
                    "render_path": '/mnt/soarin/results/render/sullens/animation_1583930186.85482/',
                },
            ),
        ]

    def _generate_examples(self, morphed_path, render_path):
        # Read the input data out of the source files
        morph_list = tf.io.gfile.listdir(morphed_path)
        render_list = tf.io.gfile.listdir(render_path)

        morph_list = np.sort(morph_list)
        render_list = np.sort(render_list)

        for morph_name, in_name in zip(morph_list, render_list):
            yield morph_name, {
                "img_name": morph_name,
                "morph_img": "%s%s" % (morphed_path, morph_name),
                "in_img": "%s%s" % (render_path, in_name),
            }


if __name__ == '__main__':
    sullens_train = tfds.load(name='sullens_morphed_animation_dataset', data_dir='/mnt/soarin/Datasets',
                              split=tfds.Split.TEST)
