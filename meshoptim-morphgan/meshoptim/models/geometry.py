import math
import numpy as np
import transformations as trans


def get_clip_matrix(width, height, near=0.01, far=1000000.):
    """
    Compute clip matrix: camera -> camera clip space

    Parameters
    ----------
    width: int
        image width

    height: int
        image height

    near: int
        near clipping plane

    far: int
        far clipping plane

    Returns
    -------
    clip_matrix: np.array
        clip matrix

    """
    aspectRatio = float(width)/height
    rangeInv = 1. / (near - far)
    fieldOfViewInRadians = np.pi * 0.305
    
    fov_degrees = 65
    # fieldOfViewInRadians = fov_degrees * np.pi/180.
    # print('Fov in radians: ', fieldOfViewInRadians)

    f = 1.0 / np.tan(fieldOfViewInRadians / 2)

    clip_matrix = [[f/aspectRatio, 0, 0, 0],
                   [0., f, 0, 0],
                   [0, 0, (near + far) * rangeInv,  near * far * rangeInv * 2],
                   [0, 0,  -1,   0]]
    return clip_matrix


def get_transf_matrix(cam_xyz, cam_angles):
    """
    Compute transformation matrix: world -> camera

    Parameters
    ----------
    cam_xyz: list
        list of camera xyz coordinates

    cam_angles: list
        lost of camera rotation angles

    Returns
    -------
    transformation matrix

    """

    alpha, beta, gamma = [math.radians(i) for i in cam_angles]

    R = np.transpose(trans.euler_matrix(alpha, beta, gamma, 'rxyz'))
    T = trans.translation_matrix(cam_xyz)

    return np.matmul(R, T)


def get_matrixes(cam_xyz, cam_angles, width, height):
    """
    Compute transformation and clip matrices

    Parameters
    ----------
    cam_xyz: list
        list of camera xyz coordinates

    cam_angles: list
        lost of camera rotation angles

    width: int
        image width

    height: int
        image height

    Returns
    -------
    rot: np.array
        transformation matrix

    clip: np.array
        clip matrix

    """

    rot = get_transf_matrix(cam_xyz, cam_angles)
    clip = get_clip_matrix(width, height)

    return rot, clip


def full_transormation_matrix(cam_xyz, cam_angles, o_r_matr, width, height):
    """
    Computes full transformation matrix: world -> camera clip space

    Parameters
    ----------
    cam_xyz: list
        list of camera xyz coordinates

    cam_angles: list
        lost of camera rotation angles

    o_r_matr: np.array
        a matrix for world axis transformations, dataset specific

    width: int
        image width

    height: int
        image height

    Returns
    -------
    T: np.array
        transformation matrix
    """
    # compute transformation and clip matrices
    rot, clip = get_matrixes(cam_xyz, cam_angles, width, height)

    # combine matrices
    T = np.array(clip).dot(o_r_matr).dot(rot)

    return T
