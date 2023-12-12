import os
import sys
from PIL import Image

from viztool.imagemetrics.metrics import process_folder


def compute_crop(x_image, y_image, zooming, im_size):
    """
    Compute image crop dimensions given x,y coordinates and zooming level

    Parameters
    ----------
    x_image: int
        x coordinate

    y_image: int
        y coordinate

    zooming: int
        zoom level, defines crop dimension

    im_size: (int, int)
        image size

    Returns
    -------
    image crop dimensions

    """

    crop_dim = [x_image-(zooming/2), y_image-(zooming/2), x_image+(zooming/2), y_image+(zooming/2)]

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
    """
    Crops around given click and saves generated cropped patch

    Parameters
    ----------
    in_path: str
        path for input images

    out_path: str
        path for output clicks

    click: dict
        click x, y, zoom coordinates

    """

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
    """
    Parser fucntion for saved clicks file

    Parameters
    ----------
    crop_path: str
        path to clicks file

    Returns
    -------
    dictionary of click details

    """

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


def crop_and_evaluate(data_path, crop_path, output_path):
    """

    Parameters
    ----------
    data_path
    crop_path
    output_path

    Returns
    -------

    """
    clicks_list = compute_clicks(crop_path)
    
    for model_dir in os.listdir(data_path):
        model_path = os.path.join(data_path, model_dir)
        out_p = os.path.join(output_path, model_dir)
        print('Model path: ', model_path)
        exp_list = os.listdir(model_path)
        for exp_dir in exp_list:

            # Skip because it's already a crop folder
            print('Experiment dir: ', exp_dir)
            if len(exp_dir.split(sep='_')) > 1:
                print('Pass this experiment.')
                pass

            else:
                in_path = os.path.join(model_path, exp_dir) + '/output/'
                for click in clicks_list:
                    out_path = os.path.join(out_p, exp_dir + '_cropped_x_%d_y_%d_zoom_%d' % (click['x'], click['y'], click['zoom'])) + '/output/'
                    try:
                        os.makedirs(out_path)
                        print('Input path: ', in_path)
                        print('Output path: ', out_path)
                        print('Cropping..')
                        error = crop_and_save(in_path, out_path, click)

                        if not error:
                            print('Running Metrics..')
                            process_folder(out_path)
                            print('Done.')

                    except FileExistsError:
                        print('Already there.')
                        pass

        print('\n')
    return


if __name__ == '__main__':

    # format of output folders
    # /media/hayko/soarin/results/gan_models/ruemonge
    # reads all methods
    # read 1st experiment in folder (random sort)

    data_path = sys.argv[1]
    crop_path = sys.argv[1][:-1] + '_viztool_screenshots/'
    output_path = sys.argv[1][:-1] + '_crops/'

    if os.path.isdir(crop_path):
        crop_and_evaluate(data_path, crop_path, output_path)

    else:
        print('\nYou have to run viztool_pygame on %s first.'%data_path)
