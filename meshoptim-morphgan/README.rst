.. -*- mode: rst -*-

********************************************
Morpher: Geometrical Per Vertex Optimisation
********************************************


This repository covers the 'Per vertex optimisation' section of the REAL-DIBS project.

Differentiable renderer and direct vertex and color optimisation are used to optimise a
given 3d triangular mesh based on a set of target images.

|

Requirements Installation
#########################

This application requires the following packages:

- python=3.7
- tensorflow==2.1.0
- tensorflow_datasets==3.1.0
- scikit-image
- pygame
- sewar
- trimesh
- tensorflow-graphics

Also requires the following repositories:

- Dirt: https://github.com/pmh47/dirt
- Elpips: https://github.com/mkettune/elpips/


Run
###

To run the code, the user should follow the following instructions:

1. run 'trimesh_to_tfrecord.py' over original mesh file.

2. make dataset in data/ folder with img_name, tar_img and mask information.

3. make configuration with 'make_config.py' specifying the new tfrecords mesh made in 1) as "mesh_path", and the new dataset in "dataset_name".

4. run 'train.py' over the new configuration.

5. run 'trimesh_to_tfrecord.py' over the desired mesh outcome of 'train.py'.

6. generate every picture of the dataset by running 'generate_dataset.py' over a new configuration where "mesh_path" points at the results of 5).

|

A working training experiment on the sullens dataset can be launched by:
    python train.py config/MorphGAN_sullens_batch.json

|