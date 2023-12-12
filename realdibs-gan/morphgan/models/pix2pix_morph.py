import tensorflow as tf
from .unet import UNetGenerator, UNetDiscriminator


class Pix2pix:
    """
    Class for the Pix2pix model used in the Morphgan architecture.

    Attributes
    ----------

    flags: dict
        a dictionary containing all parameters necessary for the script

    generator: UNetGenerator
        generator network

    discriminator: UNetDiscriminator
        discriminator network

    loss_object: tf.keras.losses.BinaryCrossentropy
        binary cross entropy function

    generator_optimizer: tf.keras.optimizers.Adam
        optimiser for generator

    discriminator_optimizer: tf.keras.optimizers.Adam
        optimiser for discriminator

    checkpoint: tf.train.Checkpoint
        checkpoint for the model

    writer: tf.summary.SummaryWriter
        writer for logs

    step: tf.Variable
        step variable to track training

    Methods
    -------

    load_checkpoint()
        Loads a checkpoint into the model

    discriminator_loss( real=None, generated=None )
        Pix2pix discriminator loss function

    generator_loss( generated=None )
        Pix2pix generator loss function

    generate_images( test_input=None )
        Use generator to generate image

    log_test( input_image=None, target=None, gen_output=None, epoch=None )
        Logs test image every step

    log( input_image=None, target=None, gen_output=None, epoch=None )
        Logs losses during training every step

    train_step( real_x=None, real_y=None, epoch=None, mask=None )
        Training step for Pix2pix model.

    """

    def __init__(self, flags, checkpoint_dir, logs_dir):
        """
        Creates generator and discriminator networks.
        Sets up loss and optimizers.

        Parameters
        ----------
        flags: dict
            a dictionary containing all parameters necessary for the script

        checkpoint_dir: str
            path to the checkpoint directory

        logs_dir:
            path to the logs directory
        """

        self.flags = flags

        self.generator = UNetGenerator()
        self.discriminator = UNetDiscriminator()

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.writer = tf.summary.create_file_writer(logs_dir)
        self.step = tf.Variable(initial_value=0, dtype=tf.int64)

        return

    def load_checkpoint(self):
        """
        Loads a checkpoint into the model.
        Checkpoint can be found in 'flags.ckpt_path'.

        """

        print('Loading model from checkpoint..')

        self.checkpoint.restore(tf.train.latest_checkpoint(self.flags.ckpt_path))

        print('Model Loaded.')

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """
        Pix2pix discriminator loss function.

        Parameters
        ----------
        disc_real_output: tf.tensor
            predictions of the discriminator over real images

        disc_generated_output: tf.tensor
            predictions of the discriminator over fake images

        Returns
        -------
        total_disc_loss: tf.tensor
            total discriminator loss value
        """

        # discriminator predictions should be all ones for real images
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        # discriminator predictions should be all zeros for fake images
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        """
        Pix2pix generator loss function.

        Parameters
        ----------
        disc_generated_output: tf.tensor
            predictions of the discriminator over fake images

        gen_output: tf.image
            generated image by the generator network

        target: tf.image
            target image

        Returns
        -------
        total_gen_loss: tf.tensor
            total generator loss value

        """

        # discriminator predictions should be all ones for real images, if zeros then the generator is doing well
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # l1 loss
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        # combine both losses
        total_gen_loss = gan_loss + (self.flags.LAMBDA * l1_loss)

        return total_gen_loss

    def generate_images(self, test_input):
        """
        Use generator to generate image

        Parameters
        ----------
        test_input: tf.image
            input image for the generator network

        Returns
        -------
        prediction: tf.image
            generated image

        """

        prediction = self.generator(test_input, training=False)

        return prediction

    def log_test(self, input_image, target, gen_output, epoch):
        """
        Logs test image every step

        Parameters
        ----------
        input_image: tf.image
            input image for the generator network

        target: tf.image
            target image

        gen_output: tf.image
            generated image

        epoch: int
            current training epoch

        """

        with self.writer.as_default():
            # DEBUG: write input, generated, target images to TB
            in_ = tf.cast((input_image * 0.5 + 0.5)*255, tf.uint8)
            out_ = tf.cast((gen_output * 0.5 + 0.5)*255, tf.uint8)
            tar_ = tf.cast((target * 0.5 + 0.5)*255, tf.uint8)
            conc = tf.concat([in_, out_, tar_], axis=2)

            # DEBUG: disable tf.function if you want tf.summary.image
            # non-DMA-copy attempted of tensor type: string tensorflow summary image
            # batch is causing this, so only reduce the first image but keep 4 dims

            # image gets a batch now, but only writes one image?
            # conc = tf.expand_dims(conc[0,:,:,:],0)
            # print(conc.shape)
            tf.summary.image('in_out_tar_test', conc, step=epoch)

    def log(self, input_image, target, gen_output, epoch):
        """
        Logs losses during training every step

        Parameters
        ----------
        input_image: tf.image
            input image for the generator network

        target: tf.image
            target image

        gen_output: tf.image
            generated image

        epoch: tf.Variable
            current training epoch

        """
        with self.writer.as_default():
            if 0:
                # DEBUG: write input, generated, target images to TB
                in_ = tf.cast((input_image[:,:,:,:3] * 0.5 + 0.5)*255, tf.uint8)
                morph_ = tf.cast((input_image[:,:,:,3:] * 0.5 + 0.5)*255, tf.uint8)
                out_ = tf.cast((gen_output * 0.5 + 0.5)*255, tf.uint8)
                tar_ = tf.cast((target * 0.5 + 0.5)*255, tf.uint8)
                conc =  tf.concat([in_ , morph_, out_, tar_], axis=2)

                # DEBUG: disable tf.function if you want tf.summary.image
                # non-DMA-copy attempted of tensor type: string tensorflow summary image
                # batch is causing this, so only reduce the first image but keep 4 dims

                # image gets a batch now, but only writes one image?
                # conc = tf.expand_dims(conc[0,:,:,:],0)
                # print(conc.shape)
                tf.summary.image('in_morph_out_tar', conc, step=self.step)

            # Log generator and discriminator losses
            tf.summary.scalar('gen_loss', self.gen_loss, step=epoch)
            tf.summary.scalar('discr_loss', self.disc_loss, step=epoch)

            # Log all gradients and variables values
            '''
            for grad, name in zip(generator_gradients, generator.trainable_variables):
                tf.summary.histogram('Gen/' + name.name, grad, step=epoch)
            for grad, name in zip(discriminator_gradients, discriminator.trainable_variables):
                tf.summary.histogram('Disc/' + name.name, grad, step=epoch)
                '''

    # @tf.function
    def train_step(self, input_image, target, epoch, mask):
        """
        Training step for pix2pix model.

        Parameters
        ----------
        input_image: tf.image
            input image for the generator network

        target: tf.image
            target image to compute losses over

        epoch: int
            current training epoch

        mask: tf.image
            mask to mask out generated images

        """

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            gen_output = self.generator(input_image, training=True)  # use generator to generate images
            gen_output = tf.math.multiply(gen_output, mask)  # mask out generated images

            in_image = input_image[:, :, :, :3]  # get input image
            morph_image = input_image[:, :, :, 3:]  # get morphed image

            # feed real, input, morphed and generated images to discriminator
            disc_real_output = self.discriminator([in_image, morph_image, target], training=True)
            disc_generated_output = self.discriminator([in_image, morph_image, gen_output], training=True)

            # compute generator and discriminator losses
            self.gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
            self.disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        # compute gradients of the losses over trainable variables
        generator_gradients = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)

        # update trainable variables with gradients
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        # log losses over the current training step
        self.log(input_image, target, gen_output, epoch)

        # increment step variable
        self.step.assign_add(1)
