
# tf13
# blender masks generation
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from pathlib import Path
#def mkdir(p): os.makedirs(p, exist_ok=True) # NOT recursive
def mkdir(p): 
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)

p = '/media/hayko/soarin/results/render/sullens/paired_samename/'
p = '/media/hayko/soarin/results/render/trinity/paired_samename/'
p = '/media/hayko/soarin/results/render/ruemonge/paired/high-res/'
po = p + '../masks/'
mkdir(po)

# PARSE FOLDER
files = os.listdir(p)
files = np.sort(files)
cnt = len(files)+1
# print(files)


from progress.bar import Bar
verbose = 1
if verbose < 2:
    bar = Bar('Processing', max=len(files), suffix='%(index)d/%(max)d %(eta)ds => %(elapsed)ds')

for fidx, file in enumerate(files):

    # READ RENDER MASK (RGBA) AND APPLY TO ORIGINAL
    mask = cv2.imread(p+file, cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
    # only 4th channel has alpha, needs 3 channels to multiply

    # CLEAN mask by erosion to ensure no border pixels are used
    kernel = np.ones((5,5), np.uint8) 
    img_morph = cv2.erode(mask[:,:,3], kernel, iterations=10) 

    # MAKE MASK BINARY AND DELETE FROM ORIGINAL
    ret, bw_img = cv2.threshold(img_morph,127,255,cv2.THRESH_BINARY)

    # SELECT LARGEST
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw_img, connectivity=8)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    bw_img8 = (output == max_label)
    #print(bw_img8.shape)

    bw_img8.dtype='uint8'
    bw_img8 = bw_img8*255
    #print(bw_img8.shape)

    cv2.imwrite(po+file, bw_img8)

    if 0:
        plt.imshow(mask[:,:,3])
        plt.axis("off")
        plt.show()
        plt.imshow(bw_img)
        plt.axis("off")
        plt.show()

    bar.next()
