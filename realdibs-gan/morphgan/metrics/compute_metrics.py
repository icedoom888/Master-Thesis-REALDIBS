import tensorflow as tf


class MetricsAnalyser:
    """
    A class used to compute metrics during training and inference.

    Attributes
    ----------
    metrics: dict
        a dictionary of metrics functions to be computed

    writer: tf.Summary.FileWriter
        tf writer used to log during training

    Methods
    -------
    compute_metrics( tar_img=None, generated_img=None, epoch=None )
        computes each metric in 'metrics', between target image 'tar_img' and 'generated_img'
    """

    def __init__(self, writer):
        # add metrics here
        self.metrics = {
        'ssim_multiscale': tf.image.ssim_multiscale,
        'ssim': tf.image.ssim,
        #'total_variation': tf.image.total_variation,
        'psnr': tf.image.psnr
        }
        self.writer = writer

    def compute_metrics(self, tar_img, generated_img, epoch):
        """
        computes each metric in 'metrics', between target image 'tar_img' and 'generated_img'

        Parameters
        ----------
        tar_img: tf.image
            target image

        generated_img: tf.image
            generated image

        epoch: int
            current epoch
        """

        with self.writer.as_default():
            for m_name, m_func in zip(self.metrics.keys(), self.metrics.values()):  # loop over metrics
                result = m_func(generated_img, tar_img, max_val = 1)  # compute metric value
                tf.summary.scalar('Metrics/' + m_name, result[0], step=epoch)  # write value in the logs

            # total variation is handled differently: no need for target image
            result = tf.image.total_variation(generated_img)
            tf.summary.scalar('Metrics/total_variation', result[0], step=epoch)