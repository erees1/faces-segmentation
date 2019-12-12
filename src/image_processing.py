'''
A collection of functions used for my project
'''
# Imports
import numpy as np
import matplotlib as plt
import re
from tqdm import tqdm
import os

# Image imports
from skimage import io, util
from skimage.color import hsv2rgb, rgb2gray
import skimage.transform


def load_images(RGB_directory='',
                label_directory=None,
                dataset='FASSEG',
                max_imgs=None,
                verbose=0):
    '''
    Helper function to load images from a dataset. Loads images from
    RGB_directory only if label_directory empty, otherwise loads both
    RGB images and correpsonding semgentation labels. Note that the RGB
    images and the labels must have the same file name.
    '''

    dir_list = os.listdir(RGB_directory)
    imgs = []

    if max_imgs is not None:
        dir_list = dir_list[:max_imgs]
    else:
        max_imgs = len(dir_list)

    if label_directory is not None:
        imgs_labels = []

    if dataset == 'fake_vs_real':
        fake_labels = np.zeros(shape=(max_imgs, 4), dtype=np.uint8)
        cats = [None] * max_imgs

    for i, img in enumerate(tqdm(dir_list, desc='Loading Images')):

        # Load RGB image
        x = io.imread(RGB_directory + '/' + img)
        imgs.append(x)

        # Load correpsonding labels if directory specified
        if label_directory is not None:
            y = io.imread(label_directory + '/' + img)
            imgs_labels.append(y)

        # If fake vs real dataset need to extract the labels from the filename
        if dataset == 'fake_vs_real':
            if 'real' in img:
                label = np.array([0, 0, 0, 0])
                cat = 'real'
            else:
                label = img[-8:-4]
                try:
                    label = [int(d) for d in label]
                except Exception:
                    print('failed on', img)
                cat = re.search('([a-zA-Z]+)(?=\_)', img)
                cat = cat[0]

            fake_labels[i] = label
            cats[i] = cat

    if label_directory is not None:
        return imgs, imgs_labels

    if dataset == 'fake_vs_real':
        return imgs, fake_labels, cats

    else:
        return imgs


def convert_images_to_array(images_list, img_shape, method='zero_pad'):
    '''
    Convert a list of images to a numpy array
    Size indicates the desired size of the images, the images are resized
    such that the height matches the size specified
    '''

    # If only one image
    if not isinstance(images_list, list):
        single_image = True
        images_list = [images_list]
    else:
        single_image = False

    # Images are first resized then added to a list
    n_images = len(images_list)
    n_channels = images_list[0].shape[2]

    new_imgs = np.zeros((n_images, img_shape[0], img_shape[1], n_channels),
                        dtype=np.uint8)
    for i, img in enumerate(
            tqdm(images_list, desc='Converting images to np array')):
        new_img = resize_image(img, img_shape, method)
        new_imgs[i] = new_img

    if single_image:
        return new_imgs[0]

    else:
        return new_imgs


def resize_images(array,
                  img_shape,
                  method='zero_pad',
                  convert_to_ubyte=False,
                  preserve_range=True):

    resized_array = np.zeros((array.shape[0], ) + img_shape +
                             (array.shape[-1], ))

    for i, img in enumerate(tqdm(array, desc='Resizing images')):
        resized_array[i] = resize_image(img, img_shape, method,
                                        convert_to_ubyte, preserve_range)
    return resized_array


def resize_image(image,
                 img_shape,
                 method='zero_pad',
                 convert_to_ubyte=False,
                 preserve_range=True):

    # Caclulate the new width based on the current aspect ratio
    old_height = image.shape[0]
    old_width = image.shape[1]
    scale = max(img_shape) / max(old_height, old_width)
    new_width = int(old_width * scale)
    new_height = int(old_height * scale)

    # Convert image data type to uint8

    img = skimage.transform.resize(image, (new_height, new_width),
                                   preserve_range=preserve_range)

    if convert_to_ubyte:
        if img.dtype != np.uint8:
            img = util.img_as_ubyte(img)

    # Pad width
    img = pad_img(img, img_shape, method)

    return img


def pad_img(img, new_shape, method='extend'):
    '''
    Function to pad an image to make it square without changing the
    aspect ratio
    '''

    if img.shape[0] >= img.shape[1]:
        pad = 'width'
    else:
        pad = 'height'

    if pad == 'width':
        total_padding = new_shape[1] - img.shape[1]
    elif pad == 'height':
        total_padding = new_shape[0] - img.shape[0]
        img = np.moveaxis(img, 0, 1)

    if total_padding % 2 == 0:
        padding = [int(total_padding / 2), int(total_padding / 2)]

    else:
        padding = [total_padding // 2 + 1, total_padding // 2]

    if method == 'extend':
        l_padding = np.array([img[:, 0]] * padding[0], dtype=np.uint8)
        r_padding = np.array([img[:, img.shape[1] - 1]] * padding[1],
                             dtype=np.uint8)
        l_padding = np.moveaxis(l_padding, 0, 1)
        r_padding = np.moveaxis(r_padding, 0, 1)

    elif method == 'zero_pad':
        if len(img.shape) == 3:
            zero_array = np.array([[0, 0, 0]], dtype=np.uint8)
        elif len(img.shape) == 2:
            zero_array = np.array([0], dtype=np.uint8)

        l_padding = np.array([
            zero_array.repeat(padding[0], axis=0),
        ] * img.shape[0])
        r_padding = np.array([
            zero_array.repeat(padding[1], axis=0),
        ] * img.shape[0])

    else:
        raise Exception('method not valid')

    padded_img = np.hstack((l_padding, img, r_padding))

    if pad == 'height':
        padded_img = np.moveaxis(padded_img, 0, 1)

    return padded_img


class MakeFlat():
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def transform(self, img):
        y = np.array([
            img[:, :, 0].flatten(), img[:, :, 1].flatten(),
            img[:, :, 2].flatten()
        ]).T
        return y

    def inverse_transform(self, flat_img):
        img_shape = self.img_shape[0:2]

        if len(flat_img.shape) > 1:
            y = np.array([
                flat_img[:, 0].reshape(img_shape),
                flat_img[:, 1].reshape(img_shape),
                flat_img[:, 2].reshape(img_shape)
            ])
            y = np.moveaxis(y, 0, 2)
        else:
            y = flat_img.reshape(img_shape)

        return y


def show_image(img, cmap='rgb'):
    if cmap == 'rgb':
        plt.imshow(img)
    elif cmap == 'hsv':
        y = hsv2rgb(img)
        y = util.img_as_ubyte(y)
        plt.imshow(y)


def convert_to_gray(array):
    if len(array.shape) == 4:
        gray_array = np.empty(shape=array.shape[:3])
        for i, img in enumerate(array):
            gray_img = rgb2gray(img)
            gray_array[i] = gray_img
    elif len(array.shape) == 3:
        gray_array = rgb2gray(array)
    else:
        print('Unexpected dimensions')

    return gray_array
