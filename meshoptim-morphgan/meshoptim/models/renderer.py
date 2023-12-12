import dirt
import tensorflow as tf

from .geometry import *


class Renderer:
    """
    The differentiable renderer

    Attributes
    ----------
    flags: dict
        a dictionary containing all parameters necessary for the script

    o_r_matr: np.array
        a matrix for world axis transformations, dataset specific

    cam_dict: dict
        a dictionary where all camera names are associated with the respective transformation matrix

    Methods
    -------
    initialize()
        Initializes the differentiable renderer

    prepare_cameras()
        Creates the 'cam_dict' dictionary by computing transformation matrices for each camera of the dataset

    render_cam( vertices=None, faces=None, colors=None, cam_name=None )
        Rendering function. Renders mesh described from 'vertices' and 'faces' from the view of camera 'cam_name'

    get_transformation_matrix( name=None )
        Extracts transformation matrix from 'cam_dict' given the key 'name'

    """

    def __init__(self, flags):
        self.flags = flags
        self.o_r_matr = np.atleast_2d([(1, 0, 0, 0),
                                       (0, -1, 0, 0),
                                       (0, 0, 1, 0),
                                       (0, 0, 0, 1)])

        # TODO: remove for sullens
        # self.o_r_matr = np.atleast_2d([(1, 0, 0, 0),
        #                                (0, 1, 0, 0),
        #                                (0, 0, 1, 0),
        #                                (0, 0, 0, 1)])

        return

    def initialize(self):
        """
        Initializes the differentiable renderer
        """
        self.prepare_cameras()
        return

    def prepare_cameras(self):
        """
        Creates the 'cam_dict' dictionary by computing transformation matrices for each camera of the dataset.
        DIRT requires the mesh to be expressed in camera clip space.
        """

        f = open(self.flags.camera_path, 'r')  # load .txt file containing all camera informations
        cam_list = f.readlines()
        f.close()

        self.cam_dict = {}

        # Compute transformation matrix for each camera
        for camera in range(2, len(cam_list)):  # Iterate through cameras
            # Read camera properties from .txt file
            cam_prop = cam_list[camera].split()
            cam_name = cam_prop[0]
            cam_xyz = [-float(i) for i in cam_prop[1:4]]  # camera xyz coordinates
            cam_angles = [float(i) for i in cam_prop[4:7]]  # camera rotation angles

            # Compute transformation matrix
            T = tf.cast(
                full_transormation_matrix(cam_xyz, cam_angles, self.o_r_matr, self.flags.width, self.flags.height),
                tf.float32)

            self.cam_dict[cam_name] = T  # assign matrix to respective camera name

        return

    def render_cam(self, vertices, faces, colors, cam_name):
        """
        Rendering function.
        Renders mesh described from 'vertices', 'faces' and 'colors' from the view of camera 'cam_name'

        Parameters
        ----------
        vertices: tf.Variable
            tensor of mesh vertices

        faces: tf.tensor
            tensor of mesh faces

        colors: tf.Variable
            tensor of mesh vertex colors

        cam_name: str
            name of the camera to render

        Returns
        -------
        pixels: tf.image
            rendered image

        """

        try:  # decode if tensor
            cam_name = cam_name.decode()[:-4]

        except AttributeError:  # if str
            cam_name = cam_name

        # Transform mesh vertices into camera view
        vertices = tf.concat([vertices, tf.ones((vertices.shape[0], 1), dtype=tf.float32)], axis=1)  # homogeneous coord
        T = self.get_transformation_matrix(cam_name)  # get transformation matrix from dict
        vertices = tf.transpose(tf.matmul(T, tf.transpose(vertices)))  # apply matrix tothe  vertices

        # Rasterization function
        pixels = dirt.rasterise(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors,
            background=tf.zeros([self.flags.height, self.flags.width, 3]),
            height=self.flags.height, width=self.flags.width, channels=3)

        return pixels

    def get_transformation_matrix(self, name):
        """
        Extracts transformation matrix from 'cam_dict' given the key 'name'

        Parameters
        ----------
        name: str
            name of the camera

        Returns
        -------
        transformation matrix of camera with name 'name'

        """
        return self.cam_dict[name]

    def __call__(self, vertices, faces, colors, cam_name):
        return self.render_cam(vertices, faces, colors, cam_name)
