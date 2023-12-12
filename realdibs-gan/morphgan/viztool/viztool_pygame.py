import os
import pygame
import sys
from PIL import Image
import numpy as np
import glob

# HOW TO Use

# Click on any of the above image to zoom into detailed bb shown below
# Regulate the zoom level by using up and down arrows on your keyboard
# Move through training epochs by by using left and right arrows on your keyboard
# Save screenshot by pressing key: S
# Get directly to MAX available Epoch with key: T
# Get directly to MIN available Epoch with key: F

tar = True


def get_data(data_path):
    """
    Load data from data path
    Parameters
    ----------
    data_path: str
        path to data folder

    Returns
    -------
    sorted(result_paths): list
        list of all models paths present in data folder

    num_models: int
        number of models

    models: list
        list of all models present in data folder

    """


    models = [dI for dI in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,dI))]
    tmp = [os.path.join(data_path,dI) for dI in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,dI))]
    result_paths = []

    use_one_run_only = 0
    for model in tmp:
        result_paths.append(os.path.join(model, os.listdir(model)[use_one_run_only], 'output'))

    num_models = len(result_paths)

    return sorted(result_paths), num_models, models


def initialize_grid_tar(result_paths, num_models):
    """
    Set up a grid of images to show once pygame starts, with target and input images

    Parameters
    ----------
    result_paths: str
        path to model experiments to show

    num_models: int
        number of models (columns) to show

    Returns
    -------

    """

    grid = []
    for dir in range(num_models):
        tmp = sort_names(glob.glob(result_paths[dir] + '/*0.png'))
        grid.append(tmp)

    if os.path.isfile(os.path.join(result_paths[dir], 'in.png')):
        grid.append(glob.glob(result_paths[dir] + '/in.png'))
        num_models += 1

    if os.path.isfile(os.path.join(result_paths[dir], 'tar.png')):
        grid.append(glob.glob(result_paths[dir] + '/tar.png'))
        num_models += 1

    return grid, num_models


def initialize_grid(result_paths, num_models):
    """
    Set up a grid of images to show once pygame starts

    Parameters
    ----------
    result_paths: str
        path to model experiments to show

    num_models: int
        number of models (columns) to show

    Returns
    -------

    """
    grid = []
    for dir in range(num_models):
        tmp = sort_names(glob.glob(result_paths[dir] +  '/*0.png'))
        grid.append(tmp)
    return grid


def sort_names(names_list):
    """
    Sorts list of names

    """
    new_names = {}
    for name in names_list:
        try:
            new_names[name] = int(name[-9:-4])
        except ValueError:
            pass
    new_names = {k: v for k, v in sorted(new_names.items(), key=lambda item: item[1])}

    return list(new_names.keys())


def PIL_to_pygame(image):
    """
    PIL image to pygame image

    """
    mode = image.mode
    size = image.size
    data = image.tobytes()

    return pygame.image.fromstring(data, size, mode)


def compute_coord(pos, column, row, cell_width, cell_height, margin, box_height, ratio, fit_width, im_size):
    """
    Computes image coordinates from screen coordinates of the user's click

    """
    if fit_width:
        x_image = int((pos[0] - (column*(cell_width+margin)) - margin) / ratio)
        y_image = int((pos[1] - (row*(cell_height+margin)) - margin - box_height) / ratio)
    else:
        indent = int((cell_width - (ratio * im_size[1]))/2)

        x_image = int((pos[0] - (column*(cell_width+margin)) - margin - indent) / ratio)
        y_image = int((pos[1] - (row*(cell_height+margin)) - margin - box_height) / ratio)

    return x_image, y_image


def compute_bb(pic_x, pic_y, x_image, y_image, ratio, zooming):
    """
    Computes bounding box coordinates to be drawn on screen

    """

    left = pic_x + x_image*ratio - int(ratio*zooming/2)
    top = pic_y + y_image*ratio - int(ratio*zooming/2)
    right = pic_x + x_image*ratio + int(ratio*zooming/2)
    bottom = pic_y + y_image*ratio + int(ratio*zooming/2)

    bb = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return bb


def compute_crop(x_image, y_image, zooming, im_size):
    """
    compute image crop dimentions

    """

    crop_dim = [x_image-(zooming/2), y_image-(zooming/2), x_image+(zooming/2), y_image+(zooming/2)]

    if crop_dim[0] < 0:
        crop_dim[0] = 0
    if crop_dim[1] < 0:
        crop_dim[1] = 0
    if crop_dim[2] > im_size[1]:
        crop_dim[2] = im_size[1]
    if crop_dim[3] > im_size[0]:
        crop_dim[3] = im_size[0]

    return crop_dim


def get_img_size(grid):
    """
    Get image size

    """

    img = Image.open(grid[0][0])
    width, height = img.size
    return height, width


def get_epoch_ratio(grid):
    """
    Get epoch ratio from images in the experiment folders

    """
    str_1 = int(grid[0][0][-9:-4])
    str_2 = int(grid[0][1][-9:-4])
    return str_2-str_1


def get_max_epoch(grid, diff):
    """
    Get final epoch of the experiments

    """
    tmp = []
    for model in range(len(grid) - diff):
        tmp.append(int(grid[model][-1][-9:-4]))
    return max(tmp)


def main(data_path):

    pics = []

    def load_pictures(pics):
        #print('loading PICS')
        pics = []
        #print(epoch_index)
        for column in range(num_models):

            if column >= num_models-diff:
                picture = Image.open(grid[column][0])
            else:
                try:
                    picture = Image.open(grid[column][int(epoch_index/epoch_ratio)])
                except:
                    picture = Image.open(grid[column][-1])
            pics.append(picture)

        return pics

    result_paths, og_models, models = get_data(data_path)

    # grid
    if tar:
        grid, num_models = initialize_grid_tar(result_paths, og_models)

    else:
        num_models = og_models
        grid = initialize_grid(result_paths, num_models)

    print('Number of Models: ', num_models)

    models_string = '/'
    for model in models:
        models_string = models_string + str(model) + '_'

    # epoch
    epoch_index = 0
    epoch_ratio = get_epoch_ratio(grid)
    # epoch_ratio = 1
    print('Epoch ratio: ', epoch_ratio)
    diff = num_models - og_models
    max_epoch = get_max_epoch(grid, diff)
    print('Max epoch: ', max_epoch)

    # zoom
    zooming = 240
    zoom_active = False

    # assigning values to X and Y variable
    X = 1920
    Y = 1170
    WINDOW_SIZE = [X, Y]
    im_size = get_img_size(grid)
    print('Image size: ', im_size)

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # This sets the margin between each cell
    margin = 5

    # This sets the cell_width and cell_height of each grid location
    box_height = 25
    box_width = 30
    grid_area_height = Y-box_height
    cell_width = int((X-(margin*(num_models +1)))/num_models)
    cell_height = int((grid_area_height-(margin*3))/2)

    # decide indentation based on greatest image side
    fit_width = True

    if im_size[1] <= im_size[0]:
        ratio = cell_height/im_size[0]
        fit_width = False

    else:
        ratio = cell_width/im_size[1]

    print('Fit width: ', fit_width)

    # Initialise pygame
    pygame.init()
    pygame.key.set_repeat(0)

    # Set the cell_height and cell_width of the screen
    screen = pygame.display.set_mode(WINDOW_SIZE)
    screen.fill(WHITE)

    # Set title of screen
    pygame.display.set_caption("GANgster Explorer")

    # Create font object
    font = pygame.font.Font('freesansbold.ttf', 25)
    epoch_text = font.render('Epoch: %d' %(epoch_index), True, BLACK, WHITE)
    zoom_text = font.render('Zooming on %dx%d area.' % (zooming, zooming), True, BLACK, WHITE)

    image_update = True
    pics = load_pictures(pics)

    # Loop until the user clicks the close button.
    done = False
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # set save directory
    save_dir = data_path[:-1] + '_viztool_screenshots/'
    models.append('input')
    models.append('target')

    # -------- Main Program Loop --------
    while not done:
        events = pygame.event.get()
        for event in events:  # User did something
            # print(events)
            # print('looping events')

            if event.type == pygame.KEYUP and event.key == pygame.K_s:
                print('Saving screen...')
                # set saving path
                save_path = save_dir + 'x_' + str(x_image) + '_y_' + str(y_image) + '_zoom_' + str(zooming) + '/'
                os.makedirs(save_path)

                # save each image shown at the moment
                for model, column in zip(models, range(num_models)):
                    pic = pics[column]
                    zoom_pic = pic.crop(crop_dim)
                    pic.save(save_path + str(model) + '.png')
                    zoom_pic.save(save_path + str(model) + '_zoomed.png')

                # also save screenshot of the pygame screen
                screen_path = save_path + models_string + 'epoch%d.png'%(epoch_index)
                pygame.image.save(screen, screen_path)
                print('Done.')

            elif event.type == pygame.QUIT:
                # goal: quit, if user clicked close
                done = True  # Flag that we are done so we exit this loop

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # goal: place the zoom box

                # user clicks the mouse, get the position
                pos = pygame.mouse.get_pos()
                if pos[1] <= box_height:
                    pass
                elif pos[1] >= margin + box_height + int(ratio*im_size[1]):
                    pass
                else:
                    # Change the x/y screen coordinates to grid coordinates
                    column = pos[0] // (cell_width + margin)
                    row = (pos[1] - box_height) // (cell_height + margin)

                    x_image, y_image = compute_coord(pos, column, row, cell_width, cell_height, margin, box_height, ratio, fit_width, im_size)

                    if x_image > 0 and y_image > 0:
                        zoom_active=True

                    print("Click ", pos, ", Image coordinates: ", x_image, y_image)
                    image_update = True

            elif event.type == pygame.KEYUP:
                # goal: changing images which are load
                image_update = True

                # MOVING EPOCHS
                if event.key == pygame.K_RIGHT:
                    epoch_index += epoch_ratio
                    if epoch_index > max_epoch:
                        epoch_index = max_epoch
                    pics = load_pictures(pics)

                if event.key == pygame.K_LEFT:
                    epoch_index -= epoch_ratio
                    if epoch_index < 0:
                        epoch_index = 0
                    pics = load_pictures(pics)

                if event.key == pygame.K_t:
                    epoch_index = max_epoch
                    pics = load_pictures(pics)

                if event.key == pygame.K_f:
                    epoch_index = 0
                    pics = load_pictures(pics)

                # ADJUSTING ZOOMING
                if event.key == pygame.K_UP:
                    zooming += 20
                if event.key == pygame.K_DOWN:
                    zooming -= 20
                    if zooming < 20:
                        zooming = 20




                zoom_text = font.render('Zooming on %dx%d area.' % (zooming, zooming), True, BLACK, WHITE)
                epoch_text = font.render('Epoch: %d' %(epoch_index), True, BLACK, WHITE)

            # Remove everything on screen
            # screen.fill(WHITE)

            # Blit onto the screen
            screen.blit(epoch_text, (2, 2))
            screen.blit(zoom_text, (X-330, 2))

            if zoom_active:
                crop_dim = compute_crop(x_image, y_image, zooming, im_size)

            pygame.display.update()

        # print('looping')
        # Draw the grid

        for row in range(2):
            # print('looping rows',str(row))
            for column in range(num_models):
                # print('looping cols',str(column))

                if row == 0:

                    if image_update:

                        # get pics per column and fit to screen
                        # print('col', column)
                        picture = pics[column]
                        picture = PIL_to_pygame(picture)
                        picture = pygame.transform.scale(picture, (int(ratio*im_size[1]), int(ratio*im_size[0])))

                        if fit_width:
                            pic_x = (margin + cell_width) * column + margin
                            pic_y = (margin + cell_height) * row + margin + box_height
                            screen.blit(picture, [pic_x, pic_y])

                        else:
                            indent = int((cell_width - (ratio * im_size[1]))/2)
                            pic_x = (margin + cell_width) * column + margin + indent
                            pic_y = (margin + cell_height) * row + margin + box_height
                            screen.blit(picture, [pic_x, pic_y])

                        # draw bb
                        if zoom_active:
                            bb = compute_bb(pic_x, pic_y, x_image, y_image, ratio, zooming)
                            pygame.draw.aalines(screen, GREEN, True, bb)

                else:
                    if zoom_active:
                        pic = pics[column]
                        pic = pic.crop(crop_dim)
                        pic_size = pic.size
                        if num_models >= 4:
                            rt = cell_width/pic_size[0]
                        else:
                            rt = cell_height/pic_size[0]

                        picture = PIL_to_pygame(pic)
                        picture = pygame.transform.scale(picture, (int(rt * pic_size[0]), int(rt * pic_size[1])))
                        indent = int((cell_width - (rt * pic_size[0]))/2)
                        pic_x = (margin + cell_width) * column + margin + indent
                        pic_y = (margin + cell_height) * row + margin + box_height
                        screen.blit(picture, [pic_x, pic_y])
                        image_update = True


                    else:
                        pass
            if image_update:
                pygame.display.update()

            image_update = False

        # Limit to 60 frames per second
        # clock.tick(60)

    pygame.quit()


if __name__ == '__main__':

    # format of output folders
    # /media/hayko/soarin/results/gan_models/ruemonge
    # reads all methods
    # read 1st experiment in fodler (random sort)
    # last two columns are input image and target if available

    data_path = sys.argv[1]
    main(data_path)
