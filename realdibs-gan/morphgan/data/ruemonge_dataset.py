import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import numpy as np

paired = True  # pix2pix needs paired data, the others not
base_path = '/mnt/soarin/'
version = '0.4.0'


class ruemonge_dataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version(version)
    SUPPORTED_VERSIONS = [
        # tfds.core.Version('0.1.0'), # Paired in and tar images
        # tfds.core.Version('0.2.0'), # Un-Paired in and tar images
        # tfds.core.Version('0.3.0'), # Paired in and tar images with masks available
        tfds.core.Version(version),  # Un-Paired in and tar images with masks available
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("This is the RueMonge dataset for RealDibs project. It contains 428 images. The "
                         "images are kept at their original dimensions of 800x1024 px."),
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

        # file structure is always
        # data/<dataset>
        # data/<dataset>/undistorted/test/
        # data/<dataset>/undistorted/test_one/
        # data/<dataset>/undistorted/train/
        #
        # results/render/<dataset>/paired/
        # results/render/<dataset>/masks/test/
        # results/render/<dataset>/masks/test_one/
        # results/render/<dataset>/masks/train/
        # <dataset>

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "render_path": base_path + 'results/render/ruemonge/paired/high-res/train/',
                    "real_path": base_path + 'data/ruemonge/undistorted/train/',
                    "mask_path": base_path + 'results/render/ruemonge/masks/train/'
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "render_path": base_path + 'results/render/ruemonge/paired/high-res/test/',
                    "real_path": base_path + 'data/ruemonge/undistorted/test/',
                    "mask_path": base_path + 'results/render/ruemonge/masks/test/'
                },
            ),
            tfds.core.SplitGenerator(
                name='TEST_ONE',
                gen_kwargs={
                    "render_path": base_path + 'results/render/ruemonge/paired/high-res/test_one/',
                    "real_path": base_path + 'data/ruemonge/undistorted/test_one/',
                    "mask_path": base_path + 'results/render/ruemonge/masks/test_one/'
                },
            ),
        ]

    def _generate_examples(self, render_path, real_path, mask_path):
        # Read the input data out of the source files
        render_list = tf.io.gfile.listdir(render_path)
        real_list = tf.io.gfile.listdir(real_path)
        real_list = np.sort(real_list)
        render_list = np.sort(render_list)

        if not paired:
            real_list = np.roll(real_list, 5)

        for f_name, g_name in zip(render_list, real_list):
            yield f_name, {
                "img_name": f_name,
                "in_img": "%s%s" % (render_path, f_name),
                "tar_img": "%s%s" % (real_path, g_name),
                "mask": "%s%s" % (mask_path, f_name)
            }


if __name__ == '__main__':

    ruemonge_train = tfds.load(name='ruemonge_dataset',
                               data_dir=base_path + 'datasets', split=tfds.Split.TEST)

    for sample in ruemonge_train:
        print(sample['img_name'])
