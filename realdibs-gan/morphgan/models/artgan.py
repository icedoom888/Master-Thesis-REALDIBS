from .resgan import *

tf.executing_eagerly()


class Transformer(tf.keras.Model):
    """
    Class for transformer used in artgan architecture
    """

    def __init__(self, kernel_size=10, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.out = tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=1, padding='same')

    def call(self, inputs):
        return self.out(inputs)


class Artgan:
    """
    Class for the Artgan model.

    Attributes
    ----------

    flags: dict
        a dictionary containing all parameters necessary for the script

    scale_weight: dict
        dictionary of scale for multi-layer discriminator

    discr_success: tf.Variable
        discriminator success rate

    win_rate: tf.constant
        win rate to switch between discriminator and generator training

    alpha: tf.constant
        update weight for discriminator success rate

    encoder: tf.keras.Models
        encoder of the generator network

    decoder: tf.keras.Models
        decoder of the generator network

    discriminator: tf.keras.Models
        discriminator network

    transformer: tf.keras.Models
        tranformer network

    loss: tf.keras.losses.BinaryCrossentropy
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

    show_summaries: bool
        flag to show model summaries

    Methods
    -------

    load_checkpoint()
        Loads a checkpoint into the model

    discriminator_loss( real_pred=None, render_pred=None, fake_pred=None )
        Artgan discriminator loss function

    discriminator_accuracy( real_pred=None, render_pred=None, fake_pred=None )
        Artgan discriminator accuracy

    generator_loss( fake_pred=None )
        Artgan generator loss function

    generator_accuracy( fake_pred=None )
        Artgan generator accuracy

    transformer_loss( output_photo=None, input_photo=None )
        Artgan transformer loss function

    feature_loss( output_photo_features=None, input_photo_features=None )
        Artgan feature loss function

    optimize_generator( generator_gradients=None, gen_acc=None )
        Optimisation step for the generator

    optimize_discriminator( discriminator_gradients=None, disc_acc=None )
        Optimisation step for the discriminator

    log_()
        Log function to use during training

    generate_images( test_input=None )
        Use generator to generate image

    train_step( render=None, real=None, epoch=None )
        Training step for Artgan model.

    """

    def __init__(self, flags, checkpoint_dir, logs_dir):
        self.flags = flags

        self.scale_weight = {"scale_0": 1.,
                             "scale_1": 1.,
                             "scale_3": 1.,
                             "scale_5": 1.,
                             "scale_6": 1.}

        self.discr_success = tf.Variable(initial_value=self.flags.discr_success,
                                         dtype=tf.float32)
        self.win_rate = tf.constant(self.flags.discr_success)
        self.alpha = tf.constant(0.05, dtype=tf.float32)

        self.encoder = Encoder(self.flags.ngf, name='Encoder')
        self.decoder = Decoder(self.flags.ngf, name='Decoder')
        self.discriminator = Discriminator(self.flags.ndf, name='Discriminator')
        self.transformer = Transformer()
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(flags.lr, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(flags.lr, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder,
                                              decoder=self.decoder,
                                              discriminator=self.discriminator,
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer)

        self.writer = tf.summary.create_file_writer(logs_dir)
        self.step = tf.Variable(initial_value=0, dtype=tf.int64)
        self.show_summaries = True

    def load_checkpoint(self):
        """
        Loads a checkpoint into the model

        """

        print('Loading model from checkpoint..')

        self.checkpoint.restore(tf.train.latest_checkpoint(self.flags.ckpt_path))

        print('Model Loaded.')

    def discriminator_loss(self, real_pred, render_pred, fake_pred):
        """
        Artgan discriminator loss function

        Parameters
        ----------
        real_pred: tf.tensor
            discriminator output over target images

        render_pred: tf.tensor
            discriminator output over input images

        fake_pred: tf.tensor
            discriminator output over generated images

        Returns
        -------
        discr_loss: tf.tensor
            discriminator loss value

        """

        real_loss = {key: self.loss(tf.ones_like(pred), pred) * self.scale_weight[key]
                     for key, pred in zip(real_pred.keys(),
                                          real_pred.values())}

        render_loss = {key: self.loss(tf.zeros_like(pred), pred) * self.scale_weight[key]
                       for key, pred in zip(render_pred.keys(),
                                            render_pred.values())}

        fake_loss = {key: self.loss(tf.zeros_like(pred), pred) * self.scale_weight[key]
                     for key, pred in zip(fake_pred.keys(),
                                          fake_pred.values())}

        with self.writer.as_default():
            tf.summary.scalar('Discriminator/real_loss', tf.add_n(list(real_loss.values())), step=self.step)
            tf.summary.scalar('Discriminator/render_loss', tf.add_n(list(render_loss.values())), step=self.step)
            tf.summary.scalar('Discriminator/fake_loss', tf.add_n(list(fake_loss.values())), step=self.step)

        discr_loss = tf.add_n(list(real_loss.values())) + \
                     tf.add_n(list(render_loss.values())) + \
                     tf.add_n(list(fake_loss.values()))

        return discr_loss

    def discriminator_accuracy(self, real_pred, render_pred, fake_pred):
        """
        Artgan discriminator accuracy

        Parameters
        ----------
        real_pred: tf.tensor
            discriminator output over target images

        render_pred: tf.tensor
            discriminator output over input images

        fake_pred: tf.tensor
            discriminator output over generated images

        Returns
        -------
        discriminator accuracy

        """

        # Compute discriminator accuracies.
        real_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)),
                                                      dtype=tf.float32)) * self.scale_weight[key]
                          for key, pred in zip(real_pred.keys(),
                                               real_pred.values())}
        render_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)),
                                                        dtype=tf.float32)) * self.scale_weight[key]
                            for key, pred in zip(render_pred.keys(),
                                                 render_pred.values())}
        fake_discr_acc = {key: tf.reduce_mean(tf.cast(x=(pred < tf.zeros_like(pred)),
                                                      dtype=tf.float32)) * self.scale_weight[key]
                          for key, pred in zip(fake_pred.keys(),
                                               fake_pred.values())}
        return (tf.add_n(list(real_discr_acc.values())) + \
                tf.add_n(list(render_discr_acc.values())) + \
                tf.add_n(list(fake_discr_acc.values()))) / float(len(self.scale_weight.keys()) * 3)

    def generator_loss(self, fake_pred):
        """
        Artgan generator loss function

        Parameters
        ----------
        fake_pred: tf.tensor
            discriminator output over generated images

        Returns
        -------
        gener_loss: tf.tensor
            generator loss value

        """

        fake_loss = {key: self.loss(tf.ones_like(pred), pred) * self.scale_weight[key]
                     for key, pred in zip(fake_pred.keys(),
                                          fake_pred.values())}

        gener_loss = tf.add_n(list(fake_loss.values()))

        return gener_loss

    def generator_accuracy(self, fake_pred):
        """
        Artgan generator accuracy

        Parameters
        ----------
        fake_pred: tf.tensor
            discriminator output over generated images

        Returns
        -------
        generator accuracy
        """

        output_photo_gener_acc = {key: tf.reduce_mean(tf.cast(x=(pred > tf.zeros_like(pred)),
                                                              dtype=tf.float32)) * self.scale_weight[key]
                                  for key, pred in zip(fake_pred.keys(),
                                                       fake_pred.values())}

        return tf.add_n(list(output_photo_gener_acc.values())) / float(len(self.scale_weight.keys()))

    def transformer_loss(self, output_photo, input_photo):
        """
        Artgan transformer loss function

        Parameters
        ----------
        output_photo: tf.image
            generated image

        input_photo: tf.image
            input image

        Returns
        -------
        transformer loss value

        """

        def mse_criterion(in_, target):
            return tf.reduce_mean((in_ - target) ** 2)

        transformed_output = self.transformer(output_photo)
        transformed_input = self.transformer(input_photo)

        '''
        with self.writer.as_default():
            tf.summary.image('transformed_in_out', tf.concat([tf.cast((transformed_input * 0.5 + 0.5)*255, tf.uint8),
                                        tf.cast((transformed_output * 0.5 + 0.5)*255, tf.uint8)], axis=2),
                                        step=self.step)
        '''
        return mse_criterion(transformed_output, transformed_input)

    def feature_loss(self, output_photo_features, input_photo_features):
        """
        Artgan feature loss function

        Parameters
        ----------
        output_photo_features: tf.tensor
            latent space representation of output image

        input_photo_features: tf.tensor
            latent space representation of input image

        Returns
        -------
        Artgan feature loss value

        """

        def abs_criterion(in_, target):
            return tf.reduce_mean(tf.abs(in_ - target))

        return abs_criterion(output_photo_features, input_photo_features)

    def optimize_generator(self, generator_gradients, gen_acc):
        """
        Optimisation step for the generator.

        Parameters
        ----------
        generator_gradients: tf.tensor
            generator gradients

        gen_acc: tf.tensor
            generator accuracy

        """

        print('\nOptimize generator ...')
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.encoder.trainable_weights + self.decoder.trainable_weights))

        # update discr_success
        self.discr_success.assign(self.discr_success * (1. - self.alpha) + self.alpha * (1. - gen_acc))

    def optimize_discriminator(self, discriminator_gradients, disc_acc):
        """
        Optimisation step for the discriminator.

        Parameters
        ----------
        discriminator_gradients: tf.tensor
            discriminator gradients

        disc_acc: tf.tensor
            discriminator accuracy

        """

        print('Optimize discriminator ...')
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_weights))

        # update discr_success
        self.discr_success.assign(self.discr_success * (1. - self.alpha) + self.alpha * disc_acc)

    def log_(self):
        """
        Logs losses during training every step
        """

        with self.writer.as_default():
            tf.summary.scalar('Total_Losses/total_gen_loss', self.total_gen_loss, step=self.step)
            tf.summary.scalar('Total_Losses/total_discr_loss', self.total_discr_loss, step=self.step)
            tf.summary.scalar('Generator/gen_loss', self.gen_loss, step=self.step)
            tf.summary.scalar('Discriminator/discr_loss', self.discr_loss, step=self.step)
            tf.summary.scalar('Generator/trans_loss', self.trans_loss, step=self.step)
            tf.summary.scalar('Generator/feat_loss', self.feat_loss, step=self.step)

            tf.summary.scalar('Accuracies/gen_acc', self.gen_acc, step=self.step)
            tf.summary.scalar('Accuracies/disc_acc', self.disc_acc, step=self.step)

            tf.summary.scalar('Success_rate/discr_success', self.discr_success, step=self.step)

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

        latent = self.encoder(test_input)
        prediction = self.decoder(latent)

        return prediction

    @tf.function
    def train_step(self, render, real, epoch):
        """
        Training step for Artgan model.

        render: tf.image
            input image for the generator network

        real: tf.image
            target image to compute losses over

        epoch: int
            current training epoch

        """

        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape() as disc_tape:

            input_photo_features = self.encoder(render)  # get latent space of input image using encoder
            output_photo = self.decoder(input_photo_features)  # use latent space to produce generated image
            output_photo_features = self.encoder(output_photo)  # get latent space of generated image using encoder

            '''
            with self.writer.as_default():
                in_ = tf.cast((render * 0.5 + 0.5)*255, tf.uint8)
                out_ = tf.cast((output_photo * 0.5 + 0.5)*255, tf.uint8)
                conc =  tf.concat([in_ , out_], axis=2)
                tf.summary.image('in_out', conc, step=self.step)
            '''

            real_pred = self.discriminator(real)  # discriminator output over target images
            render_pred = self.discriminator(render)  # discriminator output over input images
            fake_pred = self.discriminator(output_photo)  # discriminator output over generated images

            # Show model summaries
            if self.show_summaries:
                self.encoder.summary()
                print('\n')
                self.decoder.summary()
                print('\n')
                self.discriminator.summary()
                print('\n')
                self.show_summaries = False

            # compute specific losses
            self.discr_loss = self.discriminator_loss(real_pred, render_pred, fake_pred)
            self.gen_loss = self.generator_loss(fake_pred)
            self.trans_loss = self.transformer_loss(output_photo, render)
            self.feat_loss = self.feature_loss(output_photo_features, input_photo_features)

            # compute total losses
            self.total_gen_loss = self.flags.dlw * self.gen_loss + self.flags.tlw * self.trans_loss +\
                                  self.flags.flw * self.feat_loss
            self.total_discr_loss = self.flags.dlw * self.discr_loss

        # compute accuracies
        self.gen_acc = self.generator_accuracy(fake_pred)
        self.disc_acc = self.discriminator_accuracy(real_pred, render_pred, fake_pred)

        # optimise either generator or discriminator based on success rate
        if tf.math.greater_equal(self.discr_success, self.win_rate):
            generator_gradients = gen_tape.gradient(self.total_gen_loss,
                                                    self.encoder.trainable_weights + self.decoder.trainable_weights)
            self.optimize_generator(generator_gradients, self.gen_acc)

        else:
            discriminator_gradients = disc_tape.gradient(self.total_discr_loss, self.discriminator.trainable_weights)
            self.optimize_discriminator(discriminator_gradients, self.disc_acc)

        # log operation
        self.log_()

        # update step variable
        self.step.assign_add(1)
