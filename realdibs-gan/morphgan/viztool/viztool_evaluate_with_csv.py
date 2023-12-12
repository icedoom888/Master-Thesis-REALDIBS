import os
import sys
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from imagemetrics.metrics import process_folder
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_crop(x_image, y_image, zooming, im_size):
    crop_dim = [x_image - (zooming / 2), y_image - (zooming / 2), x_image + (zooming / 2), y_image + (zooming / 2)]

    if crop_dim[0] < 0:
        crop_dim[0] = 0
    if crop_dim[1] < 0:
        crop_dim[1] = 0
    if crop_dim[2] > im_size[0]:
        crop_dim[2] = im_size[0]
    if crop_dim[3] > im_size[1]:
        crop_dim[3] = im_size[1]

    return crop_dim


def crop_and_save(in_path, out_path, click):
    file_list = os.listdir(in_path)

    for img in file_list:
        image = Image.open(os.path.join(in_path, img))
        im_size = image.size
        crop_dim = compute_crop(click['x'], click['y'], click['zoom'], im_size)
        crop = image.crop(crop_dim)
        try:
            crop.save(out_path + img)

        except:
            print('Error with crop: ', crop_dim)
            return 1
            break
    return 0


def compute_clicks(crop_path):
    dir_list = os.listdir(crop_path)
    tmp = []
    for el in dir_list:
        split = el.split(sep='_')
        tmp.append({
            split[0]: int(split[1]),
            split[2]: int(split[3]),
            split[4]: int(split[5]),
        })
    return tmp


def e2(reference, predicted):
    errimage = reference.astype("float") - predicted.astype("float")
    return errimage


def mse2(reference, predicted):
    errimage = (e2(reference, predicted) ** 2) / reference.mean() ** 2
    errvalue = errimage.mean()
    return errvalue, errimage


def run_diff(reference, predicted):
    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    ev, ei = mse2(reference, predicted)
    ei = np.mean(ei, axis=2)
    ei = np.stack((ei, ei, ei), axis=2)
    ei_data = (255.0 * ei)  # Now scale by 255
    ei_img = ei_data.astype(np.uint8)
    return ei_img


def evaluate_with_csv(data_path, crop_path, output_path, config):
    clicks_list = compute_clicks(crop_path)
    tmp = np.array(clicks_list)
    np.save('clicks.npy', tmp)
    model_list = sorted(os.listdir(data_path))

    if config == 'comparison_transpose':
        save_image = False
        difference = False
        transpose = True

    elif config == 'comparison_with_image':
        save_image = True
        difference = False
        transpose = False

    elif config == 'comparison_with_difference':
        save_image = True
        difference = True
        transpose = False

    output_path = output_path + config + '/'

    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass

    for click in clicks_list:
        out_path = os.path.join(output_path, 'comparison_cropped_x_%d_y_%d_zoom_%d' %
                                (click['x'], click['y'], click['zoom']))
        values = []
        my_dpi = 100
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi, tight_layout=True)
        if save_image:
            gs = gridspec.GridSpec(2, len(model_list) + 2)

        else:
            gs = gridspec.GridSpec(1, 1)

        try:
            for idx, model_dir in enumerate(model_list):
                model_path = os.path.join(data_path, model_dir)
                exp_list = os.listdir(model_path)
                exp_name = exp_list[0].split(sep='_')[0]
                csv_path = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                        (click['x'], click['y'], click['zoom'])) + '/output/metrics_heavy.csv'
                print(csv_path)

                df = pd.read_csv(csv_path)

                # append input
                if len(values) == 0:
                    value = df.iloc[0, :]
                    value.name = 'input'
                    values.append(value)

                    # show input image
                    if save_image:
                        if difference:
                            ax = fig.add_subplot(gs[1, 0])

                            # load img
                            img = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                               (click['x'], click['y'], click['zoom'])) + '/output/in.png'
                            image = Image.open(img)

                            # load target image
                            tar = Image.open(os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                                          (click['x'], click['y'], click['zoom'])) + '/output/tar.png')

                            # compute mse
                            res = run_diff(tar, image)
                            plt.imshow(res)
                            plt.axis("off")

                        else:
                            ax = fig.add_subplot(gs[1, 0])
                            img = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                               (click['x'], click['y'], click['zoom'])) + '/output/in.png'
                            image = Image.open(img)
                            plt.imshow(image)
                            plt.axis("off")

                value = df.iloc[-3, :]
                value.name = model_dir
                # print(value)
                values.append(value)

                if save_image:
                    # show predicted image
                    if difference:
                        ax = fig.add_subplot(gs[1, idx + 1])

                        # load img
                        img = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                           (click['x'], click['y'], click['zoom'])) + '/output/gen_09400.png'
                        image = Image.open(img)

                        # load target image
                        tar = Image.open(os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                                      (click['x'], click['y'], click['zoom'])) + '/output/tar.png')

                        # compute mse
                        res = run_diff(tar, image)
                        plt.imshow(res)
                        plt.axis("off")

                    else:
                        ax = fig.add_subplot(gs[1, idx + 1])
                        img = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' %
                                           (click['x'], click['y'], click['zoom'])) + '/output/gen_09400.png'
                        image = Image.open(img)
                        plt.imshow(image)
                        plt.axis("off")

            # append target
            value = df.iloc[-1, :]
            value.name = 'target'
            # print(value)
            values.append(value)

            if save_image:
                # show target image
                if difference:
                    ax = fig.add_subplot(gs[1, idx + 2])
                    img = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' % (
                        click['x'], click['y'], click['zoom'])) + '/output/tar.png'
                    image = Image.open(img)
                    tar = Image.open(os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' % (
                        click['x'], click['y'], click['zoom'])) + '/output/tar.png')

                    res = run_diff(tar, image)
                    plt.imshow(res)
                    plt.axis("off")

                else:
                    ax = fig.add_subplot(gs[1, idx + 2])
                    img = os.path.join(model_path, exp_name + '_cropped_x_%d_y_%d_zoom_%d' % (
                    click['x'], click['y'], click['zoom'])) + '/output/tar.png'
                    image = Image.open(img)
                    plt.imshow(image)
                    plt.axis("off")

            # print(values)
            values = pd.DataFrame(values)
            # values = values.reset_index()
            # values = values.drop('index', axis=1)
            # print(values)

            if transpose:
                values = values.transpose()

            ax = fig.add_subplot(gs[0, :])
            values.plot(ax=ax, kind='bar', alpha=0.75, rot=0, grid=True)
            # plt.show()
            plt.savefig(out_path + '.png')

        except (FileNotFoundError, NotADirectoryError) as e:
            pass


if __name__ == '__main__':

    # format of output folders
    # /media/hayko/soarin/results/gan_models/ruemonge
    # reads all methods
    # read 1st experiment in fodler (random sort)

    data_path = sys.argv[1]
    crop_path = sys.argv[1][:-7] + '_viztool_screenshots/'
    output_path = sys.argv[1][:-7] + '_metrics/'
    if os.path.isdir(crop_path):
        evaluate_with_csv(data_path, crop_path, output_path, 'comparison_with_image')
        evaluate_with_csv(data_path, crop_path, output_path, 'comparison_with_difference')
        evaluate_with_csv(data_path, crop_path, output_path, 'comparison_transpose')

    else:
        print('\nYou have to run viztool_pygame on %s first.' % data_path)
