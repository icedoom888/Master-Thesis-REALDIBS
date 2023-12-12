import matplotlib.pyplot as plt
import tensorflow as tf
import trimesh

from .renderer import Renderer


class Morpher:
    """
    A class to perform a morphing operation, or per vertex direct optimisation.

    Attributes
    ----------
    flags: dict
        a dictionary containing all parameters necessary for the script

    mesh: dict
        a dictionary containing all mesh information loaded from .tfrecords file

    renderer: Renderer object
        an instance of the differentiable renderer class Renderer

    optimizer: tf.keras.optimizers.Optimizer
        optimizer for the gradient descent operation

    colors_var: tf.Variable
        variable of vertex colors, to optimize during training

    vertex_var: tf.Variable
        variable of vertex locations, to optimize during training

    writer: tf.Summary.FileWriter
        tf writer used to log during training

    step: tf.Variable
        variable that tracks training steps

    total_loss = tf.tensor
        current loss value

    Methods
    -------
    l1_loss( target=None, generated=None )
        L1 distance between target and generated images, loss function

    optimize_mesh( gradients=None )
        Applies computed gradients to vertex locations and colors variables

    log_()
        Log function that writes the loss values during training

    render_batch( vertices=None, vertex_colors=None, cam_names=None )
        Renders images from a batch of cameras

    show_pair( res=None, tar=None )
        Shows rendered image, target image and difference between the two

    get_morphed_mesh()
        Makes a trimesh mesh object from the updated 'vertex_var' and 'colors_var'

    train_step( targets=None, cam_names=None )
        Per vertex optimisation step. Morphes the original mesh.


    """
    def __init__(self, flags, mesh, logs_dir):

        self.flags = flags
        self.mesh = mesh

        print('Mesh keys: ', mesh.keys())
        for key in mesh.keys():
            print(key, ' shape: ', mesh[key].shape)

        # define Naural renderer
        self.renderer = Renderer(flags)
        self.renderer.initialize()

        # define loss and optimizer
        self.optimizer = tf.keras.optimizers.Adam(flags.lr, beta_1=0.5)

        # define tf Variables for colors and vertices to be optimised
        self.colors_var = tf.Variable(initial_value=self.mesh['colors'], dtype=tf.float32)
        self.vertex_var = tf.Variable(initial_value=self.mesh['vertices'], dtype=tf.float32)

        # define summary writer
        self.writer = tf.summary.create_file_writer(logs_dir)

        # define step variable
        self.step = tf.Variable(initial_value=0, dtype=tf.int64)

    def l1_loss(self, target, generated):
        """
        L1 distance between target and generated images, loss function

        Parameters
        ----------
        target: tf.image
            target image

        generated: tf.image
            generated image

        Returns
        -------
        L1 distance between the two images
        """
        return tf.reduce_mean(tf.abs(target - generated))

    def optimize_mesh(self, gradients):
        """
        Applies computed gradients to vertex locations and colors variables

        Parameters
        ----------
        gradients: tf.tensor
            gradients computed from the loss over trainable variables

        """
        self.optimizer.apply_gradients(zip(gradients, [self.colors_var, self.vertex_var]))
        return

    def log_(self):
        """
        Log function that writes the loss values during training
        """
        with self.writer.as_default():
            tf.summary.scalar('Total_Losses/total_loss', self.total_loss, step=self.step)

        self.step.assign_add(1)  # increment step variable
        return

    def render_batch(self, vertices, vertex_colors, cam_names):
        """
        Renders images from a batch of cameras

        Parameters
        ----------
        vertices: tf.Variable
            variable of vertex locations

        vertex_colors:
            variable of vertex colors

        cam_names: list
            batch of camera names

        Returns
        -------
        tf.tensor of stacked rendered images

        """

        generated = []  # empty list
        for cam_name in cam_names:  # for each camera in the batch
            # Use the differentiable renderer to render an image
            generated.append(self.renderer(vertices, self.mesh['triangles'], vertex_colors, cam_name))

        return tf.stack(generated)

    def show_pair(self, res, tar):
        """
        Shows rendered image, target image and difference between the two

        Parameters
        ----------
        res: tf.image
            rendered image

        tar: tf.image
            target image

        """
        fig=plt.figure(figsize=(8, 8))

        res = tf.cast(res, tf.uint8)
        tar = tf.cast(tar, tf.uint8)

        fig.add_subplot(1, 3, 1)
        plt.imshow(res)
        fig.add_subplot(1, 3, 2)
        plt.imshow(tar)
        fig.add_subplot(1, 3, 3)
        plt.imshow(tar-res)
        plt.show()

        return

    def get_morphed_mesh(self):
        """
        Makes a trimesh mesh object from the updated 'vertex_var' and 'colors_var'

        Returns
        -------
        modified_mesh: trimesh.Trimesh
            trimesh mesh object representing the new morphed mesh
        """

        # get vertices, vertex colors and faces
        vertices = self.vertex_var.numpy()
        vertex_colors = self.colors_var
        faces = self.mesh['triangles']

        # trimesh need 4 channel colors
        tmp = tf.constant(255., shape=(vertex_colors.shape[0], 1), dtype=tf.float32)
        trimesh_colors = tf.cast(tf.concat([vertex_colors, tmp], -1), tf.uint8)

        # create trimesh object
        modified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=trimesh_colors)

        return modified_mesh

    def generate_images(self, cam_names):
        """
        Renders views from 'cam_names' cameras
        Used in Inference or Testing

        Parameters
        ----------
        cam_names: list
            batch of camera names

        Returns
        -------
        generated: tf.tensor
            tf.tensor of stacked rendered images

        """

        vertices = self.vertex_var
        vertex_colors = self.colors_var

        generated = self.render_batch(vertices, vertex_colors, cam_names)

        return generated

    def train_step(self, targets, cam_names):
        """
        Per vertex optimisation step.
        Morphes the original mesh.

        Parameters
        ----------
        targets: tf.tensor
            batch of target images

        cam_names: list
            batch of camera names

        """

        with tf.GradientTape(persistent=True) as tape:  # to compute gradients

            # get variables
            vertices = self.vertex_var
            vertex_colors = self.colors_var

            # Use the neural renderer to obtain rendered images from the batch of cameras
            generated = self.render_batch(vertices, vertex_colors, cam_names)
            
            # self.show_pair(generated[0], targets[0])

            # Compute the loss
            self.total_loss = self.l1_loss(targets, generated)

        # compute gradients of the loss over vertex locations and colors
        gradients = tape.gradient(self.total_loss, [self.colors_var, self.vertex_var])

        # apply gradients to the mesh
        self.optimize_mesh(gradients)

        # log operation
        self.log_()
