import tensorflow as tf
from .unet import UNetGenerator, UNetDiscriminator


class CycleGan:
    """
    Class for the Cycle-GAN model.

    Attributes
    ----------

    flags: dict
        a dictionary containing all parameters necessary for the script

    generator_g: UNetGenerator
        generator network for forward cycle

    generator_f: UNetGenerator
        generator network for backward cycle

    discriminator_x: UNetDiscriminator
        discriminator network for forward cycle

    discriminator_y: UNetDiscriminator
        discriminator network for backward cycle

    loss_obj: tf.keras.losses.BinaryCrossentropy
        binary cross entropy function

    generator_g_optimizer: tf.keras.optimizers.Adam
        optimiser for generator_g

    generator_f_optimizer: tf.keras.optimizers.Adam
        optimiser for generator_f

    discriminator_x_optimizer: tf.keras.optimizers.Adam
        optimiser for discriminator_x

    discriminator_y_optimizer: tf.keras.optimizers.Adam
        optimiser for discriminator_y

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
        Cycle-GAN discriminator loss function

    generator_loss( generated=None )
        Cycle-GAN generator loss function

    calc_cycle_loss( real_image=None, cycled_image=None )
        Cycle-GAN cycle loss function

    identity_loss( real_image=None, same_image=None )
        Identity loss function

    generate_images( test_input=None )
        Use generator to generate image

    train_step( real_x=None, real_y=None, epoch=None, mask=None )
        Training step for Cycle-GAN model.

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

        self.generator_g = UNetGenerator()
        self.generator_f = UNetGenerator()

        self.discriminator_x = UNetDiscriminator()
        self.discriminator_y = UNetDiscriminator()

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(generator_g=self.generator_g,
                                              generator_f=self.generator_f,
                                              discriminator_x=self.discriminator_x,
                                              discriminator_y=self.discriminator_y,
                                              generator_g_optimizer=self.generator_g_optimizer,
                                              generator_f_optimizer=self.generator_f_optimizer,
                                              discriminator_x_optimizer=self.discriminator_x_optimizer,
                                              discriminator_y_optimizer=self.discriminator_y_optimizer)

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

    def discriminator_loss(self, real, generated):
        """
        Cycle-GAN discriminator loss function.

        Parameters
        ----------
        real: tf.tensor
            predictions of the discriminator over real images

        generated: tf.tensor
            predictions of the discriminator over fake images

        Returns
        -------
        total_disc_loss: tf.tensor
            total discriminator loss value
        """

        # discriminator predictions should be all ones for real images
        real_loss = self.loss_obj(tf.ones_like(real), real)

        # discriminator predictions should be all zeros for fake images
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        """
        Cycle-GAN generator loss function.

        Parameters
        ----------
        generated: tf.tensor
            predictions of the discriminator over fake images

        Returns
        -------
        total_gen_loss: tf.tensor
            total generator loss value
        """
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        """
        Cycle-GAN cycle loss function.

        Parameters
        ----------
        real_image: tf.image
            target image

        cycled_image: tf.image
            generated image after cycle

        Returns
        -------
        cycle loss value

        """

        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return self.flags.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        """
        Identity loss function

        Parameters
        ----------
        real_image: tf.image
            target image

        same_image: tf.image
            other image to compare

        Returns
        -------
        identity loss value

        """

        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.flags.LAMBDA * 0.5 * loss

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

        prediction = self.generator_g(test_input)

        return prediction

    # @tf.function
    def train_step(self, real_x, real_y, epoch, mask):
        """
        Training step for Cycle-GAN model.

        Parameters
        ----------
        real_x: tf.image
            input image for the generator network

        real_y: tf.image
            target image

        epoch: int
            current training epoch

        mask: tf.image
            mask to mask out generated images

        """

        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:

            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x)
            # fake_y = tf.math.multiply(fake_y, mask)
            cycled_x = self.generator_f(fake_y)
            # cycled_x = tf.math.multiply(cycled_x, mask)

            fake_x = self.generator_f(real_y)
            # fake_x = tf.math.multiply(fake_x, mask)
            cycled_y = self.generator_g(fake_x)
            # cycled_y = tf.math.multiply(cycled_y, mask)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x)
            # same_x = tf.math.multiply(same_x, mask)
            same_y = self.generator_g(real_y)
            # same_y = tf.math.multiply(same_y, mask)

            disc_real_x = self.discriminator_x(real_x)
            disc_real_y = self.discriminator_y(real_y)

            disc_fake_x = self.discriminator_x(fake_x)
            disc_fake_y = self.discriminator_y(fake_y)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # Apply the gradients to the trainable variables
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

        # log during training
        with self.writer.as_default():
            tf.summary.scalar('total_gen_loss', total_gen_g_loss, step=self.step)
            tf.summary.scalar('disc_loss', disc_x_loss, step=self.step)
            if 0:
                # DEBUG: write input, generated, target images to TB
                in_ = tf.cast((real_x * 0.5 + 0.5) * 255, tf.uint8)
                out_ = tf.cast((fake_y * 0.5 + 0.5) * 255, tf.uint8)
                tar_ = tf.cast((real_y * 0.5 + 0.5) * 255, tf.uint8)
                conc = tf.concat([in_, out_, tar_], axis=2)

                img = tf.image.encode_png(conc[0])

                tf.io.write_file('train.png', img)

        # updat estep value
        self.step.assign_add(1)
