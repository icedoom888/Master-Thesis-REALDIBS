# library of visual metric functions
#  author: hayko riemenschneider

# RUN IN conda: tf1.3 python metrics

###############################################################################
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from skimage.measure import compare_ssim as scikit_ssim_base
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as scikit_nrmse_base
from skimage.measure import compare_psnr as scikit_psnr_base
# from skimage.measure import shannon_entropy as shannon
#
from tensorflow.image import ssim_multiscale as tf_ssim_ms #no color space
from tensorflow.image import ssim as tf_ssim #no color space
from tensorflow.image import total_variation as tf_tv
from tensorflow.image import psnr as tf_psnr

#from tensorflow.image import mse as tf_mse
import tensorflow as tf

import sewar

import sys
# sys.path.append(os.path.abspath('./'))
# sys.path.append('/home/hayko/Dropbox/work/drz/code/20190429_realdibs/metrics/elpips/')
sys.path.append('/home/alberto/realdibs/code/elpips/')
import elpips

from pathlib import Path
#def mkdir(p): os.makedirs(p, exist_ok=True) # NOT recursive
def mkdir(p): 
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)




# make graph for elpips
tf.set_random_seed(42)

tf_image1 = tf.placeholder(tf.float32)
tf_image2 = tf.placeholder(tf.float32)

config = elpips.elpips_squeeze_maxpool()
config.average_over = 200
metric = elpips.Metric(config, back_prop=False)
squeeze_maxpool_distance = metric.forward(tf_image1, tf_image2)


config = elpips.elpips_vgg()
config.average_over = 200
metric = elpips.Metric(config, back_prop=False)
vgg_distance = metric.forward(tf_image1, tf_image2)


config = elpips.lpips_squeeze()
metric = elpips.Metric(config, back_prop=False)
lpips_squeeze_distance = metric.forward(tf_image1, tf_image2)


config = elpips.lpips_vgg()
metric = elpips.Metric(config, back_prop=False)
lpips_vgg_distance = metric.forward(tf_image1, tf_image2)

sess = tf.Session()

def run_elpips(reference, predicted, dist):

    batch_reference = np.expand_dims(reference, axis=0)
    batch_predicted = np.expand_dims(predicted, axis=0)

    errvalue = sess.run(dist, feed_dict={tf_image1: batch_predicted, tf_image2: batch_reference})

    # print(errvalue)
    return errvalue[0]


###############################################################################
# WRAPPERS
if 1:
    def tv_value (image, reference=None):
        #errvalue = tf.reduce_sum(tf.image.total_variation(image))
        #errvalue = tf.image.total_variation(image)
        pixel_dif1 = image[1:,:,:] - image[:-1,:,:]
        pixel_dif2 = image[:,1:,:] - image[:,:-1,:]
        errvalue = np.sum(abs(pixel_dif1)) + np.sum(abs(pixel_dif2)) 
        return errvalue, None

    def e2(reference, predicted):
        errimage = reference.astype("float") - predicted.astype("float")
        return errimage

    def mse2(reference, predicted):
        errimage = (e2(reference, predicted) ** 2) / reference.mean()**2
        errvalue = errimage.mean()
        return errvalue, errimage

    def me2(reference, predicted):
        errimage = abs(e2(reference, predicted)) / reference.mean()
        errvalue = errimage.mean()
        return errvalue, errimage

    def scikit_ssim(reference, predicted):
        v,i = scikit_ssim_base(reference, predicted, multichannel=True, full=True)
        return 1-v,i 


    def scikit_nrmse_value(reference, predicted):
        return scikit_nrmse_base(reference, predicted),None

    def scikit_psnr_value(reference, predicted):
        return scikit_psnr_base(reference, predicted),None


    def sewar_ergas_value(reference, predicted):
        return sewar.full_ref.ergas(reference, predicted),None

    def sewar_sam_value(reference, predicted):
        return sewar.full_ref.sam(reference, predicted),None


    def elpips_squeeze_maxpool(reference, predicted):
        # note: this code is stochastic and needs repeats
        errvalue = run_elpips(reference, predicted, squeeze_maxpool_distance)
        return errvalue, None

    def elpips_vgg(reference, predicted):
        # note: this code is stochastic and needs repeats
        errvalue = run_elpips(reference, predicted, vgg_distance)
        return errvalue, None

    def lpips_squeeze(reference, predicted):
        errvalue = run_elpips(reference, predicted, lpips_squeeze_distance)
        return errvalue, None

    def lpips_vgg(reference, predicted):
        errvalue = run_elpips(reference, predicted, lpips_vgg_distance)
        return errvalue, None


###############################################################################
# CONFIG
flag_heavyplot = True # whether to plot input images onto figure
flag_saveerrormap = False # whether to plot input images onto figure
flag_verbose = 1 # level of debuging output

pair = [mse2, me2, scikit_ssim, scikit_psnr_value, scikit_nrmse_value, tv_value, sewar_sam_value, sewar_ergas_value,
        elpips_squeeze_maxpool, elpips_vgg, lpips_squeeze, lpips_vgg]
# pair = [mse2, me2, scikit_ssim, scikit_psnr_value, scikit_nrmse_value, sewar_sam_value]
# pair = [mse2, me2,  scikit_nrmse_value, sewar_sam_value, scikit_ssim]

# pair = [elpips_squeeze_maxpool, elpips_vgg, lpips_squeeze, lpips_vgg]

pair = [mse2, me2, scikit_ssim, scikit_psnr_value, scikit_nrmse_value, sewar_sam_value,
        elpips_squeeze_maxpool, elpips_vgg, lpips_squeeze, lpips_vgg]

# Removed scikit_psnr_value, tv_value and sewar_ergas_value because too big
pair = [mse2, me2, scikit_ssim, scikit_nrmse_value, sewar_sam_value,
        elpips_squeeze_maxpool, elpips_vgg, lpips_squeeze, lpips_vgg]

columns = [metric.__name__ for metric in pair]



###############################################################################
# UTILS

def get_mask_from_alpha(p):
    # create RENDER MASK from INPUT (RGBA)

    # mask is hidden in alpha channel of the initialized/input image.
    mask = cv2.imread(p+"in.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)

    # clean mask by erosion to ensure no border pixels are used
    kernel = np.ones((5,5), np.uint8) 
    mask = cv2.erode(mask[:,:,3], kernel, iterations=5) 

    # make binary 
    ret, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    mask[mask==255] = 1

    # write out as 8bit 255 image to see
    mask_uint8 = mask
    mask_uint8.dtype = 'uint8'
    cv2.imwrite(p + 'mask_eroded.png', mask_uint8)

    if 0:
        #plt.imshow(mask[:,:,3])
        #plt.axis("off")
        #plt.show()
        plt.imshow(bw_img)
        plt.axis("off")
        plt.show()

    return mask

def compare_images(original, rendered, generated, title):
    # compares three images: original vs. rendered vs. generated
    # in all combinations and for a number of metric values
    # and plots the images, difference images and ssim images


    
    tvO,_ = tv_value(original)
    # compute the mean squared error and structural similarity
    # index for the images
    mseOR,mseOR_full = mse2(original, rendered)
    meOR,meOR_full = me2(original, rendered)
    ssimOR,ssimOR_full = scikit_ssim_base(original, rendered,multichannel=True, full=True)
    pnsrOR,_ = scikit_psnr_value(original, rendered)
    nrmseOR,_ = scikit_nrmse_value(original, rendered)
    ergasOR = sewar.full_ref.ergas(original, rendered)
    samOR = sewar.full_ref.sam(original, rendered)
    #s_ergas = sewar.no_ref.ergas(original, rendered)
    esqmOR, _ = elpips_squeeze_maxpool(original, rendered)
    evggOR, _ = elpips_vgg(original, rendered)
    lsqOR, _ = lpips_squeeze(original, rendered)
    lvggOR, _ = lpips_vgg(original, rendered)

    tvG,_ = tv_value(generated)
    mseOG,mseOG_full = mse2(original, generated)
    meOG,meOG_full = me2(original, generated)
    ssimOG, ssimOG_full = scikit_ssim_base(original,generated,multichannel=True, full=True)
    pnsrOG,_ = scikit_psnr_value(original, generated)
    nrmseOG,_ = scikit_nrmse_value(original, generated)
    ergasOG = sewar.full_ref.ergas(original, generated)
    samOG = sewar.full_ref.sam(original, generated)
    raseOG = sewar.full_ref.rase(original, generated)
    #s_ergas = sewar.no_ref.ergas(original, generated)
    esqmOG, _ = elpips_squeeze_maxpool(original, generated)
    evggOG, _ = elpips_vgg(original, generated)
    lsqOG, _ = lpips_squeeze(original, generated)
    lvggOG, _ = lpips_vgg(original, generated)

    tvR,_ = tv_value(rendered)
    mseRG,mseRG_full = mse2(rendered, generated)
    meRG,meRG_full = me2(rendered, generated)
    ssimRG, ssimRG_full = scikit_ssim_base(rendered,generated,multichannel=True, full=True)
    pRG = scikit_psnr_value(rendered, generated)
    rRG = scikit_nrmse_value(rendered, generated)
    ergasRG = sewar.full_ref.ergas(rendered, generated)
    samRG = sewar.full_ref.sam(rendered, generated)
    #s_ergas = sewar.no_ref.ergas(rendered, generated)
    esqmRG, _ = elpips_squeeze_maxpool(rendered, generated)
    evggRG, _ = elpips_vgg(rendered, generated)
    lsqRG, _ = lpips_squeeze(rendered, generated)
    lvggRG, _ = lpips_vgg(rendered, generated)

    #tfpsnr = psnr_tf(original, rendered, max_val=255)
    #tftv = tv(rendered)
    #tfssim = ssim_tf(original, rendered)
    #tfssimms = ssim_ms(original, rendered)

    
    np.set_printoptions(precision=4)
    print('tvO  %2.0f tvR  %2.0f tvG  %2.0f' % (tvO,tvR,tvG))
    print('ergasOR %2.0f ergasOG %2.0f' % (ergasOR,ergasOG))
    print('pnsrOR  %2.3f pnsrOG  %2.3f higher better' % (pnsrOR,pnsrOG))
    print('nrmseOR %2.4f nrmseOG %2.4f lower better' % (nrmseOR,nrmseOG))
    print('mseOR   %2.4f mseOG   %2.4f lower better' % (mseOR,mseOG))
    print('meOR    %2.4f meOG    %2.4f lower better' % (meOR,meOG))
    print('ssimOR  %2.4f ssimOG  %2.4f higher better' % (ssimOR,ssimOG))
    print('samOR   %2.4f samOG   %2.4f' % (samOR,samOG))
    print('esqmOR %2.4f esqmOG %2.4f lower better' % (esqmOR,esqmOG))
    print('evggOR %2.4f evggOG %2.4f lower better' % (evggOR,evggOG))
    print('lsqOR  %2.4f lsqOG  %2.4f lower better' % (lsqOR,lsqOG))
    print('lvggOR %2.4f lvggOG %2.4f lower better' % (lvggOR,lvggOG))

    #print(tfpsnr)
    #print(tftv)
    #print(tfssim)
    #print(tfssimms)

    # setup the figure
    fig = plt.figure(title)
    #plt.suptitle("MSE: %.2f, SSIM: %.2f     MSE: %.2f, SSIM: %.2f " % (m, s, m2, s2))
    plt.suptitle("original \t rendered \t L1 \t L2 \t SSIM")

    # show first image
    ax = fig.add_subplot(3, 5, 1)
    plt.imshow(original)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(3, 5, 2)
    plt.imshow(rendered)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 3)
    plt.imshow(meOR_full)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 4)
    plt.imshow(mseOR_full)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 5)
    plt.imshow(ssimOR_full)
    plt.axis("off")

    # show first image
    ax = fig.add_subplot(3, 5, 6)
    plt.imshow(original)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(3, 5, 7)
    plt.imshow(generated)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 8)
    plt.imshow(meOG_full)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 9)
    plt.imshow(mseOG_full)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 10)
    plt.imshow(ssimOG_full)
    plt.axis("off")

    # show first image
    ax = fig.add_subplot(3, 5, 11)
    plt.imshow(rendered)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(3, 5, 12)
    plt.imshow(generated)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 13)
    plt.imshow(meRG_full)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 14)
    plt.imshow(mseRG_full)
    plt.axis("off")

    ax = fig.add_subplot(3, 5, 15)
    plt.imshow(ssimRG_full)
    plt.axis("off")


    # show the images
    plt.show()


def create_crops(path_epoch_predictions):
    # crops images to create example images

    # TODO: use Path()
    p = path_epoch_predictions + '/'
    po = path_epoch_predictions + '/output/'
    if flag_saveerrormap:
        pe = path_epoch_predictions + '/errormap/'
        mkdir(pe)


    # READ ground truth original images
    original = cv2.imread(p + "gt.png")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = original[500:1012,2000:2512,:]
   
    if 1:
        plt.imshow(original)
        plt.axis("off")
        plt.show()

        
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pe + "gt.png", original)

    # PARSE FOLDER
    files = os.listdir(p)
    files = np.sort(files)
    cnt = len(files)+1
    # print(files)

    for fidx, file in enumerate(files):
        if file == 'in.png': fidx=-1
        if file == 'gt.png': continue
        if file == 'mask.png': continue
        if file == 'tar.png': continue

        print(file, fidx)

        generated = cv2.imread(p+file)
        # rendered input image has RGBA, all else are classic RGB
        if file == 'in.png':
            generated = cv2.cvtColor(generated, cv2.COLOR_BGRA2RGB)
        else:
            generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)

        generated = generated[500:1012,2000:2512,:]
        generated = cv2.cvtColor(generated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(pe + file, generated)

def process_folder(path_epoch_predictions):
    # compares all files in a folder to the groundtruth orginal images
    # also handles rendered image (in.png) and mask creation (mask_eroded.png)
    # skips the in output folder files: gt.png tar.png mask.png
    # special loading for RGBA rendered (in.png)

    # TODO: use Path()
    p = path_epoch_predictions + '/../'
    po = path_epoch_predictions

    # TODO: works?
    p = po

    if flag_saveerrormap:
        pe = path_epoch_predictions + '/../errormap/'
        mkdir(pe)


    title   = 'progress?'
    import matplotlib.gridspec as gridspec
    #fig = plt.figure(tight_layout=True)
    my_dpi = 100
    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi, tight_layout=True) 

    # read groundtruth reference (original photos)
    reference = cv2.imread(p + "tar.png")
    # TODO: test exist, in case file missing: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

    if 0: # DEBUG
        plt.imshow(reference)
        plt.axis("off")
        plt.show()

    # READ RENDER MASK (RGBA)
    mask = get_mask_from_alpha(p)

    # DELETE FROM reference
    for i in range(3):
        reference[:,:,i] = reference[:,:,i] * mask




    # PARSE FOLDER WITH PREDICTED CONTENT
    files = os.listdir(po)
    files = np.sort(files)
    # files = ['../in.png'] + list(files) + ['../gt.png']
    # cnt = len(files)
    new_files = []
    for file in files:
        #HACK: to skip filenames in case also in folder
        if file == 'in.png': continue
        if file == 'gt.png':  continue
        if file == 'mask.png': continue
        if file == 'mask_eroded.png': continue
        if file == 'tar.png':  continue
        new_files.append(file)
    new_files = new_files[::2]
    files = ['in.png'] + list(new_files) + ['tar.png']
    print(files)
    cnt = len(files)
    print(cnt)

    import pandas
    df_ = pandas.DataFrame(columns=columns, index=range(cnt))
    gs = gridspec.GridSpec(3, cnt)

    from progress.bar import Bar
    if flag_verbose < 2:
        bar = Bar('Processing', max=len(files), suffix='%(index)d/%(max)d %(eta)ds => %(elapsed)ds')


    # TODO HACK: skipping in.png causes a NaN wherever it was in the sorted list
    for fidx, file in enumerate(files):
        #HACK: to skip filenames in case also in folder
        # if file == 'in.png': bar.next(); continue
        # if file == 'gt.png': bar.next(); continue
        # if file == 'mask.png': bar.next(); continue
        # if file == 'mask_eroded.png': bar.next(); continue
        # if file == 'tar.png': bar.next(); continue
        if flag_verbose > 1:
            print(file, fidx)

        if 0 and fidx % 100 > 0:
            bar.next()
            continue

        predicted = cv2.imread(po+file)

        # rendered input image has RGBA, all else are classic RGB
        if file == 'in.png':
            predicted = cv2.cvtColor(predicted, cv2.COLOR_BGRA2RGB)
        else:
            predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)

        # DELETE MASK FROM predicted IMAGE
        for i in range(3):
            predicted[:,:,i] = predicted[:,:,i] * mask

        if flag_heavyplot:
            # show predicted image
            ax = fig.add_subplot(gs[1, fidx])
            plt.imshow(predicted)
            plt.axis("off")

        if 0: # check if image is correctly loaded
            title   = 'correct image?'
            fig = plt.figure(title)

            # show first image
            ax = fig.add_subplot(1, 4, 1)
            plt.imshow(predicted)
            plt.axis("off")
            predicted = predicted * mask
            predicted = predicted * mask
            ax = fig.add_subplot(1, 4, 2)
            plt.imshow(predicted)
            plt.axis("off")
            ax = fig.add_subplot(1, 4, 3)
            plt.imshow(mask)
            plt.axis("off")
            ax = fig.add_subplot(1, 4, 4)
            plt.imshow(reference)
            plt.axis("off")
            plt.show()


        # CALCULATE METRICS (error value, error image)
        evlist = []
        for metric in pair:
            ev, ei = metric(reference, predicted)
            if flag_verbose > 1:
                print(metric.__name__, ev)
            evlist.append(ev)
        df_.iloc[fidx] = evlist


        if flag_saveerrormap and fidx % 100 == 0: #output errormap
            #data = (255.0 * i) / np.max(i) # Now scale by 255
            
            # average the 3 channel error map (alternative max)
            ev, ei = mse2(reference, predicted)
            ei = np.mean(ei, axis=2)
            data = (255.0 * ei)  # Now scale by 255

            #img = data.astype(np.uint8)

            errbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(pe + file, errbgr)

        if flag_heavyplot:
            # average the 3 channel error map (alternative max)
            ev, ei = mse2(reference, predicted)
            ei = np.mean(ei, axis=2)
            ei = np.stack((ei,ei,ei), axis=2)
            ei_data = (255.0 * ei)  # Now scale by 255
            ei_img = ei_data.astype(np.uint8)

            # show error map image below
            ax = fig.add_subplot(gs[2, fidx])
            plt.imshow(ei_img)
            plt.axis("off")

            # last column is ground truth reference
            #ax = fig.add_subplot(gs[1, cnt-2])
            #plt.imshow(reference)
            #plt.axis("off")


        if not flag_heavyplot: # INTERMEDIATE GRAPH AS PNG
            df_.plot()
            plt.savefig(p + 'metrics.png')
            plt.close()
            plt.close()
            #fig = plt.figure(tight_layout=True)

        bar.next()

    if flag_heavyplot:
        ax = fig.add_subplot(gs[0,:])
        df_.plot(ax=ax)
        plt.savefig(p + 'metrics_heavy.png')
        # plt.show()
    else:
        my_dpi = 100
        fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi, tight_layout=True) 
        ax = fig.add_subplot(1,1,1)
        df_.plot(ax=ax)
        #plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode
        plt.savefig(p + 'metrics.png')
        plt.show()

    # debug metric errors in table format
    print(df_)
    df_.to_csv(p + 'metrics_heavy.csv', index=False)

###############################################################################
# EXAMPLES
def example_single_image():
    ''' read in one image (predicted) and compare it to
        its groundtruth (reference) and its input (initialized).

        in this case the predicted is the gan result, the
        initialized is the blender rendered version, and the
        grountruth is the original photo.
    '''
    
    example = 0
    if example==0:
        reference = cv2.imread('example_single/gt.png')
        initialized = cv2.imread('example_single/in.png')
        predicted = cv2.imread('example_single/pred.png')

    if example==1:
        reference = cv2.imread("/media/hayko/DATA/output/agisoft/trinity/crop/trinity_10x_out_3/0493.JPG")
        initialized = cv2.imread("/media/hayko/DATA/output/agisoft/trinity/crop/trinity_10x_out_3/0493_trinity_10x_nshiftx_shifty.png")
        # missing predicted

    if example==2:
        reference = cv2.imread("/media/hayko/DATA/project/realdibs/results/render/trinity/0001.png")
        initialized = cv2.imread("/media/hayko/DATA/project/realdibs/results/render/trinity/trinity_cam_0001.png")
        # missing predicted

    if example==3:
        reference = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_512x512.png")
        initialized = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_input_alowres.png")
        #initialized = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_input_highres.png")
        predicted = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_output1_highres.png")

    
    if flag_verbose > 1:
        print(type(reference), (reference.shape))
        print(type(initialized), (initialized.shape))

    # convert the images to grayscale
    # error if file is missing: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    initialized = cv2.cvtColor(initialized, cv2.COLOR_BGR2RGB)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2RGB)

    compare_images(reference, initialized, predicted, 'reference vs. initialized')

def create_noisy_images(path_base_folder):
    ''' create example noise images ontop a given image
        a) use lena (path_base_folder)
        b) use a gt image from the datasets
    '''

    # TODO: use from pathlib import Path()
    pn = path_base_folder + '/examples_noise/'
    po = path_base_folder + '/examples_noise/output/'
    mkdir(pn)
    mkdir(po)

    image_url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    import requests
    img_data = requests.get(image_url).content
    with open(pn + "gt.png", 'wb') as handler:
        handler.write(img_data)


    from PIL import Image
    import numpy as np
    from skimage.util import random_noise

    # read groundtruth reference (original photos)
    reference = Image.open(pn + "gt.png")
    reference.thumbnail((400, 400)) 
    reference_arr = np.asarray(reference)
    reference.save(pn+'gt.png') 

    # random_noise() method will convert image in [0, 255] to [0, 1.0],
    # inherently it use np.random.normal() to create normal distribution
    # and adds the generated noised back to image
    sigma = 0.5
    noise_arr = random_noise(reference_arr, mode='gaussian', var=sigma**2)
    noise_arr = (255*noise_arr).astype(np.uint8)

    noise = Image.fromarray(noise_arr)
    #noise.show()
    noise.save(pn+'in.png') 
    noise.save(po+'in.png') 

    for sigma in [0, 0.05, 0.1, 0.2, 0.3]:
        noise_arr = random_noise(reference_arr, mode='gaussian', var=sigma**2)
        noise_arr = (255*noise_arr).astype(np.uint8)

        noise = Image.fromarray(noise_arr)
        noise.save(po + 'generated_sigma_' + str(1-sigma) + '.png') 


###############################################################################
if __name__== "__main__":

    # unstructured filenames, just set filename inside code (TODO: fix configurable)
    # see './example_single/
    # <basefolder>/pred.png
    # <basefolder>/in.png
    # <basefolder>/gt.png
    if 0:
        example_single_image()

    # create example noise data
    # see './example_noise/
    if 0:
        #path_reference = '/media/hayko/soarin/metrics/basefolder/'
        path_base_folder = '/'
        create_noisy_images(path_base_folder)

    # structured filenames according realdibs project
    # see './example/
    # <basefolder>/
    # <basefolder>/output/gen_00000.png
    # <basefolder>/output/gen_00xxx.png
    # <basefolder>/output/in.png/gt.png/tar.png are skipped
    # <basefolder>/in.png
    # <basefolder>/gt.png
    if 0:
        # folder with iteration output per method
        # p = '/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/'
        # p = '/media/hayko/soarin/results/transfer/hayko/Pix2Pix_Test_sullens_batch1_epoch2000/'
        # p = '/media/hayko/soarin/results/baselines/sullens/cycleGAN/20191001-135653/output/'
        # p = '/media/hayko/soarin/results/baselines/sullens/pix2pix/20191001-125359/output/'
        # p = '/media/hayko/soarin/results/baselines/sullens/cycleGAN/20191001-144556/output/'
        # p = '/media/hayko/soarin/results/baselines/sullens/pix2pix/20191002-172459/output/'
        # p = '/media/hayko/soarin/results/baselines/sullens/cycleGAN/20191003-130953/output/'
        # p = '/mnt/soarin/results/baselines/sullens/pix2pix/20191002-103829_full_run/output/'
        # p = '/mnt/soarin/results/baselines/sullens/cycleGAN/20191014-192858/output2/'
        # p = '/mnt/soarin/results/baselines/sullens/pix2pix/20191002-172459_hayko/output2/'
        # p = '/mnt/soarin/results/baselines/sullens/pix2pix/20191002-172459_hayko/examples/'

        path_epoch_predictions = '/media/hayko/soarin/metrics/basefolder/output/'
        path_epoch_predictions = 'examples_noise/output/'
        path_epoch_predictions = 'examples/output/'
        path_epoch_predictions = '/media/hayko/soarin/metrics/examples_fullresolution/output/'
        path_epoch_predictions = '/media/hayko/soarin/metrics/x_3037_y_835_zoom_420/output/'
        path_epoch_predictions = '/media/hayko/soarin/metrics/20191002-172459_hayko/output2/'
        process_folder(path_epoch_predictions)

    if 1:
        path_epoch_predictions = sys.argv[1]
        process_folder(path_epoch_predictions)


    # TODO: create zoom images semi-automatically
    if 0: # example crops
        pe = p + '../examples/'
        mkdir(pe)
        create_crops()
