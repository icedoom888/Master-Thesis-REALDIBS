import tensorflow as tf
import tensorflow_datasets.public_api as tfds


class sullens_mesh_dataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.2.0')
    SUPPORTED_VERSIONS = [
        tfds.core.Version('0.1.0'),  # camera and target image
        tfds.core.Version('0.2.0'),  # camera and target images with masks available
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="This is the Sullens dataset for RealDibs project. It contains 37 target images. ",
            features=tfds.features.FeaturesDict({
                "img_name": tfds.features.Text(),
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
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/train/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/'
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/'
                },
            ),
            tfds.core.SplitGenerator(
                name='TEST_ONE',
                gen_kwargs={
                    "real_path": '/media/alberto/DATA/project/realdibs/data/sullens/undistorted/test_one/',
                    "mask_path": '/mnt/soarin/results/render/sullens/masks/test_one/'
                },
            ),
        ]

    def _generate_examples(self, real_path, mask_path):
        # Read the input data out of the source files
        real_list = tf.io.gfile.listdir(real_path)
        for f_name in real_list:
            # print(f_name)
            yield f_name, {
                "img_name": f_name,
                "tar_img": "%s%s" % (real_path, f_name),
                "mask": "%s%s" % (mask_path, f_name)}


if __name__ == '__main__':
    sullens_train = tfds.load(name='sullens_mesh_dataset', data_dir='/mnt/soarin/Datasets', split=tfds.Split.TRAIN)
