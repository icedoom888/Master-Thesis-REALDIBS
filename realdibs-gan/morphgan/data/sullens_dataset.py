import tensorflow_datasets.public_api as tfds
import tensorflow as tf


class sullens_dataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

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
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/test/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test/',
                },
            ),
            tfds.core.SplitGenerator(
                name='TEST_ONE',
                gen_kwargs={
                    "render_path": '/media/alberto/DATA/project/realdibs/results/render/sullens/pairedSamename/'
                                   'test_one/',
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test_one/',
                },
            ),
        ]

    def _generate_examples(self, render_path, real_path):
        # Read the input data out of the source files
        for f_name in tf.io.gfile.listdir(render_path):
            yield f_name, {
                "img_name": f_name,
                "in_img": "%s%s" % (render_path, f_name),
                "tar_img": "%s%s" % (real_path, f_name),
            }


if __name__ == '__main__':
    sullens_train = tfds.load(name='sullens_dataset', data_dir='/mnt/soarin/Datasets', split=tfds.Split.TRAIN)
