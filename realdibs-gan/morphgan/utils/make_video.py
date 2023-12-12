import cv2
import os


"""
Make videos out of collection of images
"""

# image_folder = '/mnt/soarin/results/gan_models/sullens_morphed_animation/pix2pix/20200312-140514/output/'
image_folder = '/mnt/soarin/results/render/sullens/animation_1583930186.85482/'
fps = 20
video_name = 'video_%dfps.avi'%fps
video_name = image_folder + video_name

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MP42')
video = cv2.VideoWriter(video_name, fourcc, float(fps), (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()
