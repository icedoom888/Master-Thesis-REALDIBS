import cv2
import os
import numpy as np

"""
Make mixed videos out of collection of images.
Mix 2 different videos with given masks.
"""

# image_folder = '/mnt/soarin/results/gan_models/sullens_morphed_animation/pix2pix/20200312-140514/output/'

# images folders
in_folder = '/mnt/soarin/results/render/sullens/animation_1583930186.85482/'
out_folder = '/mnt/soarin/results/gan_models/sullens_morphed_animation/pix2pix/20200312-140514/'
# output folder for frames
out_folder = out_folder + 'output/'
fps = 20  # frame per second
video_name = out_folder + 'mixed_colour_%dfps.mp4'%fps  # video output file

# load masks for mixed effect
mask_1 = np.uint8(cv2.imread('mask_1.png')/255.)
mask_2 = np.uint8(cv2.imread('mask_2.png')/255.)

# images lists
in_images = [img for img in os.listdir(in_folder) if img.endswith(".png")]
in_images.sort()
out_images = [img for img in os.listdir(out_folder) if img.endswith(".png")]
out_images.sort()

# make frame
frame = cv2.imread(os.path.join(out_folder, out_images[0]))
height, width, layers = frame.shape

# video writer
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, 0x7634706d, float(fps), (width, height))

# for each image of both folders create a mixed frame
for in_image, out_image in zip(in_images, out_images):
    print(in_image)
    out_image = cv2.imread(os.path.join(out_folder, out_image))
    # mean_out = np.mean(out_image, axis = (0, 1))
    # std_out = np.std(out_image, axis=(0, 1))

    in_image = cv2.imread(os.path.join(in_folder, in_image))
    # mean_in = np.mean(in_image, axis=(0, 1))
    # std_in = np.std(in_image, axis=(0, 1))
    # norm_in = (in_image - mean_in)/std_in
    # in_image = np.uint8((norm_in * std_out) + mean_out)

    # in_image[np.where(in_image < 0)] = 0
    # in_image[np.where(in_image > 255)] = 255

    in_image = in_image * mask_1  # mask out with mask 1
    out_image = out_image * mask_2  # mask out with mask 2

    final = in_image + out_image
    video.write(final)

video.release()