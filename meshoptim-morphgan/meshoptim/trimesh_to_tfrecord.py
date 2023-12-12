import tensorflow as tf
import trimesh


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    """Returns an float32_list."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord_from_obj(mesh, out_path):
    """
    Converts a trimesh mesh object to tfrecord file.

    Parameters
    ----------
    mesh: trimesh object
            trimesh mesh object loaded from a '.obj' file
    out_path: str
            output file path
    """

    num_vertices = mesh.vertices.shape[0]  # get number of mesh vertices
    num_triangles = mesh.faces.shape[0]  # get number of mesh triangles
    vertices = tf.io.serialize_tensor(tf.cast(mesh.vertices, dtype=tf.float32))  # serialize vertices
    triangles = tf.io.serialize_tensor(tf.cast(mesh.faces, dtype=tf.int32))  # serialize triangles

    # get mesh vertex colors and serialize them
    try:  # if 'mesh.visual.vertex_colors' exists
        labels = tf.io.serialize_tensor(tf.cast(mesh.visual.vertex_colors[:, :3], tf.int32))

    except AttributeError:  # otherwise get vertex colors
        colors = mesh.visual.to_color().vertex_colors
        labels = tf.io.serialize_tensor(tf.cast(colors[:, :3], tf.int32))

    # create feature dictionary with serialized objects
    feature = {
                'num_vertices': _int64_feature(num_vertices),
                'num_triangles': _int64_feature(num_triangles),
                'vertices': _bytes_feature(vertices),
                'triangles': _bytes_feature(triangles),
                'labels': _bytes_feature(labels)
            }

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))  # define tf.train.Example object with given features

    serialized_proto = example_proto.SerializeToString()  # serialize to string

    # save tfrecords file
    with tf.io.TFRecordWriter(out_path) as writer:
        writer.write(serialized_proto)


mesh_path = '/mnt/soarin/results/mesh_optim_vertex_colors/trinity_crop/' \
            'mesh_optim_vertex_colors/20200302-174539/mesh/morphed_mesh_349.ply'
tfrecord_path_out = '/mnt/soarin/results/mesh_optim_vertex_colors/trinity_crop/' \
                    'mesh_optim_vertex_colors/20200302-174539/mesh/morphed_mesh_349.tfrecords'

mesh = trimesh.load(mesh_path, process=False)
write_tfrecord_from_obj(mesh, tfrecord_path_out)
