# library of visual metric functions
#  author: hayko riemenschneider

    # DONE: COLOR IMAGES
    # DONE: work on 3 images: input, generated, rendered
    # DONE: automatize new error measures
    # DONE: loop across folder and plot
    # TODO: aziz video, perceptual
    # TODO: check mse normalization
    # TODO: single image / no reference values
    # TODO: tensorflow metrics need a tensor (tensorboard)

# tf1.3 python metrics

# shiftx from neural paper?
# photogrammetry error paper

# https://github.com/NVIDIA/partialconv
# perceptual loss and more

# pip install sewar
# https://sewar.readthedocs.io/en/latest/

# https://github.com/opencv/opencv_contrib/tree/master/modules/quality
# https://github.com/krshrimali/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model

'''
    impnsrOrt cv2
    # read image
    img = cv2.imread(img_path, 1) # mention img_path
    # compute brisque quality score via static method
    score = cv2.quality.QualityBRISQUE_compute(img, model_path, range_path) # specify model_path and range_path
    # compute brisque quality score via instance
    # specify model_path and range_path
    obj = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    score = obj.compute(img)
'''

# https://github.com/Netflix/vmaf
# https://github.com/Netflix/vmaf/blob/master/resource/doc/VMAF_Python_library.md


# https://github.com/richzhang/PerceptualSimilarity


# https://scikit-learn.org/stable/modules/metrics.html#metrics


import os
from pathlib import Path

import cv2
###############################################################################
# IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import sewar
from skimage.measure import compare_nrmse as scikit_nrmse_base
from skimage.measure import compare_psnr as scikit_psnr_base
from skimage.measure import compare_ssim as scikit_ssim_base


#  https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
# from tensorflow.image import ssim_multiscale as tf_ssim_ms #no color space
# from tensorflow.image import ssim as tf_ssim #no color space
# from tensorflow.image import total_variation as tf_tv
# from tensorflow.image import psnr as tf_psnr
#def mkdir(p): os.makedirs(p, exist_ok=True) # NOT recursive
def mkdir(p):
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)


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


###############################################################################
# CONFIG
flag_heavyplot = False # whether to plot input images onto figure
flag_saveerrormap = False # whether to plot input images onto figure

pair = [mse2, me2, scikit_ssim, scikit_psnr_value, scikit_nrmse_value, tv_value, sewar_sam_value, sewar_ergas_value]
pair = [mse2, me2, scikit_ssim, scikit_psnr_value, scikit_nrmse_value, sewar_sam_value]
pair = [mse2, me2,  scikit_nrmse_value, sewar_sam_value, scikit_ssim]
columns = [metric.__name__ for metric in pair]

# base folder for dataset (gt.png and in.png)
pb = '/mnt/soarin/results/baselines/sullens/'
pb = '/mnt/soarin/results/mesh_optim_vertex_colors/sullens/mesh_optim_vertex_colors/20191219-134502/output/'


# folder with iteration output per method
p = '/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/'
p = '/media/hayko/soarin/results/transfer/hayko/Pix2Pix_Test_sullens_batch1_epoch2000/'
p = '/media/hayko/soarin/results/baselines/sullens/cycleGAN/20191001-135653/output/'
p = '/media/hayko/soarin/results/baselines/sullens/pix2pix/20191001-125359/output/'
p = '/media/hayko/soarin/results/baselines/sullens/cycleGAN/20191001-144556/output/'
p = '/media/hayko/soarin/results/baselines/sullens/pix2pix/20191002-172459/output/'
p = '/media/hayko/soarin/results/baselines/sullens/cycleGAN/20191003-130953/output/'
p = '/mnt/soarin/results/baselines/sullens/pix2pix/20191002-103829_full_run/output/'

p = '/mnt/soarin/results/baselines/sullens/cycleGAN/20191014-192858/output2/'
p = '/mnt/soarin/results/baselines/sullens/pix2pix/20191002-172459_hayko/output2/'

p = '/mnt/soarin/results/baselines/sullens/pix2pix/20191002-172459_hayko/examples/'

p = './examples/output/'; pb = p + '../'


###############################################################################
# CODE

def run_metrics(generated, rendered, original , epoch, metrics_dir):
    base = metrics_dir + 'epoch_%05d/'%epoch
    os.makedirs(base)

    # compares three images: original vs. rendered vs. generated
    # in all combinations and for a number of metric values
    # and plots the images, difference images and ssim images



    tvO = tv_value(original)
    # compute the mean squared error and structural similarity
    # index for the images
    mseOR,mseOR_full = mse2(original, rendered)
    meOR,meOR_full = me2(original, rendered)
    ssimOR,ssimOR_full = scikit_ssim_base(original, rendered,multichannel=True, full=True)
    pnsrOR = scikit_psnr_value(original, rendered)
    nrmseOR = scikit_nrmse_value(original, rendered)
    ergasOR = sewar.full_ref.ergas(original, rendered)
    samOR = sewar.full_ref.sam(original, rendered)
    #s_ergas = sewar.no_ref.ergas(original, rendered)

    tvG = tv_value(generated)
    mseOG,mseOG_full = mse2(original, generated)
    meOG,meOG_full = me2(original, generated)
    ssimOG, ssimOG_full = scikit_ssim_base(original,generated,multichannel=True, full=True)
    pnsrOG = scikit_psnr_value(original, generated)
    nrmseOG = scikit_nrmse_value(original, generated)
    ergasOG = sewar.full_ref.ergas(original, generated)
    samOG = sewar.full_ref.sam(original, generated)
    raseOG = sewar.full_ref.rase(original, generated)
    #s_ergas = sewar.no_ref.ergas(original, generated)

    tvR = tv_value(rendered)

    mseRG,mseRG_full = mse2(rendered, generated)
    meRG,meRG_full = me2(rendered, generated)
    ssimRG, ssimRG_full = scikit_ssim_base(rendered,generated,multichannel=True, full=True)
    pRG = scikit_psnr_value(rendered, generated)
    rRG = scikit_nrmse_value(rendered, generated)
    ergasRG = sewar.full_ref.ergas(rendered, generated)
    samRG = sewar.full_ref.sam(rendered, generated)
    #s_ergas = sewar.no_ref.ergas(rendered, generated)

    #tfpsnr = psnr_tf(original, rendered, max_val=255)
    #tftv = tv(rendered)
    #tfssim = ssim_tf(original, rendered)
    #tfssimms = ssim_ms(original, rendered)

    # Save metrics text file
    with open(base + "metrics.txt", "w") as text_file:
        text_file.write('tvO  %2.0f tvR  %2.0f tvG  %2.0f' % (tvO,tvR,tvG))
        text_file.write('ergasOR %2.0f ergasOG %2.0f' % (ergasOR,ergasOG))
        text_file.write('pnsrOR  %2.3f pnsrOG  %2.3f higher better' % (pnsrOR,pnsrOG))
        text_file.write('nrmseOR %2.4f nrmseOG %2.4f lower better' % (nrmseOR,nrmseOG))
        text_file.write('mseOR   %2.4f mseOG   %2.4f lower better' % (mseOR,mseOG))
        text_file.write('meOR    %2.4f meOG    %2.4f lower better' % (meOR,meOG))
        text_file.write('ssimOR  %2.4f ssimOG  %2.4f higher better' % (ssimOR,ssimOG))
        text_file.write('samOR   %2.4f samOG   %2.4f' % (samOR,samOG))

    # Save metrics images

    # setup the figure
    fig = plt.figure('metrics_%05d/'%epoch)
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

    plt.savefig(base + 'metrics_imgs.png')

def compare_images(original, rendered, generated, title):
    # compares three images: original vs. rendered vs. generated
    # in all combinations and for a number of metric values
    # and plots the images, difference images and ssim images



    tvO = tv_value(original)
    # compute the mean squared error and structural similarity
    # index for the images
    mseOR,mseOR_full = mse2(original, rendered)
    meOR,meOR_full = me2(original, rendered)
    ssimOR,ssimOR_full = scikit_ssim_base(original, rendered,multichannel=True, full=True)
    pnsrOR = scikit_psnr_value(original, rendered)
    nrmseOR = scikit_nrmse_value(original, rendered)
    ergasOR = sewar.full_ref.ergas(original, rendered)
    samOR = sewar.full_ref.sam(original, rendered)
    #s_ergas = sewar.no_ref.ergas(original, rendered)

    tvG = tv_value(generated)
    mseOG,mseOG_full = mse2(original, generated)
    meOG,meOG_full = me2(original, generated)
    ssimOG, ssimOG_full = scikit_ssim_base(original,generated,multichannel=True, full=True)
    pnsrOG = scikit_psnr_value(original, generated)
    nrmseOG = scikit_nrmse_value(original, generated)
    ergasOG = sewar.full_ref.ergas(original, generated)
    samOG = sewar.full_ref.sam(original, generated)
    raseOG = sewar.full_ref.rase(original, generated)
    #s_ergas = sewar.no_ref.ergas(original, generated)

    tvR = tv_value(rendered)

    mseRG,mseRG_full = mse2(rendered, generated)
    meRG,meRG_full = me2(rendered, generated)
    ssimRG, ssimRG_full = scikit_ssim_base(rendered,generated,multichannel=True, full=True)
    pRG = scikit_psnr_value(rendered, generated)
    rRG = scikit_nrmse_value(rendered, generated)
    ergasRG = sewar.full_ref.ergas(rendered, generated)
    samRG = sewar.full_ref.sam(rendered, generated)
    #s_ergas = sewar.no_ref.ergas(rendered, generated)

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



def create_examples():
    # crops images to create example images

    # READ ground truth original images
    original = cv2.imread(pb + "gt.png")
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


def process_folder():
    # compares all files in a folder to the groundtruth orginal images
    # also handles rendered image (in.png) and mask creation (mask_eroded.png)
    # skips the in folder files: gt.png tar.png mask.png
    # special loading for RGBA rendered (in.png)

    if flag_saveerrormap:
        po = p + '../errormap/'
        mkdir(po)

    title   = 'progress?'
    import matplotlib.gridspec as gridspec
    #fig = plt.figure(tight_layout=True)
    my_dpi = 100
    fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi, tight_layout=True)

    # READ ground truth original images
    original = cv2.imread(pb + "gt.png")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    if 0:
        plt.imshow(original)
        plt.axis("off")
        plt.show()

    # READ RENDER MASK (RGBA) AND APPLY TO ORIGINAL
    mask = cv2.imread(pb+"in.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA)
    # only 4th channel has alpha, needs 3 channels to multiply

    # CLEAN mask by erosion to ensure no border pixels are used
    kernel = np.ones((5,5), np.uint8)
    img_morph = cv2.erode(mask[:,:,3], kernel, iterations=5)

    # MAKE MASK BINARY AND DELETE FROM ORIGINAL
    ret, bw_img = cv2.threshold(img_morph,127,255,cv2.THRESH_BINARY)
    bw_img[bw_img==255] = 1
    for i in range(3):
        original[:,:,i] = original[:,:,i] * bw_img
    bw_img8 = bw_img
    bw_img8.dtype='uint8'
    cv2.imwrite(pb+'mask_eroded.png', bw_img8)

    if 0:
        plt.imshow(mask[:,:,3])
        plt.axis("off")
        plt.show()
        plt.imshow(bw_img)
        plt.axis("off")
        plt.show()

    # PARSE FOLDER
    files = os.listdir(p)
    files = np.sort(files)
    cnt = len(files)+1
    # print(files)

    import pandas
    df_ = pandas.DataFrame(columns=columns, index=range(cnt))
    gs = gridspec.GridSpec(3, cnt)

    from progress.bar import Bar
    verbose = 1
    if verbose < 2:
        bar = Bar('Processing', max=len(files), suffix='%(index)d/%(max)d %(eta)ds => %(elapsed)ds')


    for fidx, file in enumerate(files):
        if file == 'in.png': fidx=-1
        if file == 'gt.png': bar.next(); continue
        if file == 'mask.png': bar.next(); continue
        if file == 'mask_eroded.png': bar.next(); continue
        if file == 'tar.png': bar.next(); continue
        if verbose > 1:
            print(file, fidx)

        if 0 and fidx % 100 > 0:
            bar.next()
            continue

        generated = cv2.imread(p+file)

        # rendered input image has RGBA, all else are classic RGB
        if file == 'in.png':
            generated = cv2.cvtColor(generated, cv2.COLOR_BGRA2RGB)
        else:
            generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)

        # DELETE MASK FROM RENDERED IMAGE
        for i in range(3):
            generated[:,:,i] = generated[:,:,i] * bw_img

        if flag_heavyplot:
            # show generated image
            ax = fig.add_subplot(gs[1, fidx+1])
            plt.imshow(generated)
            plt.axis("off")

        if 0: # check if image is correctly loaded
            title   = 'correct image?'
            fig = plt.figure(title)

            # show first image
            ax = fig.add_subplot(1, 4, 1)
            plt.imshow(generated)
            plt.axis("off")
            generated = generated * mask
            generated = generated * mask
            ax = fig.add_subplot(1, 4, 2)
            plt.imshow(generated)
            plt.axis("off")
            ax = fig.add_subplot(1, 4, 3)
            plt.imshow(mask)
            plt.axis("off")
            ax = fig.add_subplot(1, 4, 4)
            plt.imshow(original)
            plt.axis("off")
            plt.show()


        # CALCULATE METRICS (error value, error image)
        evlist = []
        for metric in pair:
            ev, ei = metric(original, generated)
            if verbose > 1:
                print(metric.__name__, ev)
            evlist.append(ev)
        df_.iloc[fidx+1] = evlist

        if flag_saveerrormap and fidx % 100 == 0: #output errormap
            #data = (255.0 * i) / np.max(i) # Now scale by 255

            # average the 3 channel error map (alternative max)
            ei = np.mean(ei, axis=2)
            data = (255.0 * ei)  # Now scale by 255

            #img = data.astype(np.uint8)

            errbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(po + file, errbgr)

        if flag_heavyplot:
            # average the 3 channel error map (alternative max)
            ei = np.mean(ei, axis=2)
            ei = np.stack((ei,ei,ei), axis=2)
            data = (255.0 * ei)  # Now scale by 255
            img = data.astype(np.uint8)

            # show error map image below
            ax = fig.add_subplot(gs[2, fidx+1])
            plt.imshow(img)
            plt.axis("off")

            # last column is ground truth original
            ax = fig.add_subplot(gs[1, cnt-2])
            plt.imshow(original)
            plt.axis("off")


        if not flag_heavyplot: # INTERMEDIATE GRAPH AS PNG
            df_.plot()
            plt.savefig(p+'../metrics.png')
            plt.close()
            plt.close()
            #fig = plt.figure(tight_layout=True)

        bar.next()

    if flag_heavyplot:
        ax = fig.add_subplot(gs[0,:])
        df_.plot(ax=ax)
        plt.savefig(p+'../metrics_heavy.png')
        plt.show()
    else:
        my_dpi = 100
        fig = plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi, tight_layout=True)
        ax = fig.add_subplot(1,1,1)
        df_.plot(ax=ax)
        #plt.get_current_fig_manager().full_screen_toggle() # toggle fullscreen mode
        plt.savefig(p+'../metrics.png')
        plt.show()

    # debug metric errors in table format
    print(df_)

    if 0:
        fig = plt.figure(tight_layout=True)
        original = cv2.imread(p+'gt.png')
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        plt.imshow(original)
        plt.axis("off")
        plt.show()


def main():


    if 0: #check individual images
        original = cv2.imread("/media/hayko/DATA/output/agisoft/trinity/crop/trinity_10x_out_3/0493.JPG")
        rendered = cv2.imread("/media/hayko/DATA/output/agisoft/trinity/crop/trinity_10x_out_3/0493_trinity_10x_nshiftx_shifty.png")

        original = cv2.imread("/media/hayko/DATA/project/realdibs/results/render/trinity/0001.png")
        rendered = cv2.imread("/media/hayko/DATA/project/realdibs/results/render/trinity/trinity_cam_0001.png")

        original = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_512x512.png")
        rendered = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_input_alowres.png")
        #rendered = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_input_highres.png")
        generated = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/camera/out_ruemonge/00365_output1_highres.png")

        print(type(original))
        print(type(rendered))
        print((original.shape))
        print((rendered.shape))

        # convert the images to grayscale
        # file is missing if error (-215:Assertion failed) !_src.empty() in function 'cvtColor'
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)

        compare_images(original, rendered,generated, 'original vs. rendered')

    if 1: # process folders
        process_folder()

    if 0: # example crops
        pe = p + '../examples/'
        mkdir(pe)
        create_examples()

if __name__== "__main__":
  main()










###############################################################################
# EXTRA CODE




# class _Loss:
#     def __init__(name):
#         self.name = name
#     def loss(target, input, **kwargs):
#         pass

# class MSE(_Loss):
#     def __init__():
#         super().__init__(__name__)
#     def loss(target, input, h):
#         return target


# class MSE(_Loss):
#     def __init__():
#         super().__init__(__name__)
#     def loss(target, input, h):
#         return target

#losses = [
    #MSE(),
    #L1()
#]


if 0: # ADD MASK TO RGBA
    rendered = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/in.png")
    rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGBA)
    mask = cv2.imread("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/mask.png")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    rendered[:,:,3] = mask[:,:,0]
    rendered = cv2.cvtColor(rendered, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite("/home/hayko/Dropbox/work/drz/code/20190429_realdibs/transfer/pix2pix2_dithering/in2.png",rendered)
