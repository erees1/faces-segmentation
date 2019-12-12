'''
A collection of functions used for processing the labels of the input images
'''

# Depenancies
from image_processing import MakeFlat
from sklearn.cluster import KMeans
import numpy as np
from feature_processing import create_pixel_loc
from tqdm import tqdm
from skimage.measure import block_reduce
from scipy.stats import mode


def segment_labeled_images(imgs,
                           n_clusters=6,
                           label_dict=[],
                           split_eyes=True,
                           desired_labels=['eyes', 'mouth', 'nose']):
    '''
    Utilility function for segmenting an array of images at once
    '''

    imgs_out = np.empty(imgs.shape[:3], dtype=np.uint8)

    for i, img in enumerate(tqdm(imgs, desc='Segmenting labels')):
        seg_img = segment_labeled_image(img, n_clusters, label_dict,
                                        split_eyes, desired_labels)
        imgs_out[i] = seg_img

    return imgs_out


def segment_labeled_image(img,
                          n_clusters,
                          label_dict={},
                          split_eye_labels=True,
                          desired_labels=['eyes', 'mouth', 'nose'],
                          return_mapping=False):

    img_labels = run_kmeans(img, n_clusters)

    # Translate the labels into labels
    desired_labels = desired_labels.copy()
    if split_eyes:
        desired_labels.append('eyes')

    img_labels = convert_labels(img_labels, img, label_dict, desired_labels)
    img_labels = img_labels.astype(np.uint8)

    mapping = {i: label_dict[i][1] for i in desired_labels
               if i in label_dict}
    mapping['nc'] = 0

    if split_eye_labels:
        img_labels = split_eyes(img_labels, label_dict)
        mapping['left_eye'] = mapping.pop('eyes')
        mapping['right_eye'] = mapping['left_eye'] + 1

    if return_mapping:
        return img_labels, mapping
    else:
        return img_labels


def run_kmeans(img, n_clusters):

    # create features
    make_flat = MakeFlat(img.shape)
    flat_image = make_flat.transform(img)

    # kmeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(flat_image)
    labels = kmeans.labels_ + 1

    img_labels = make_flat.inverse_transform(labels)

    return img_labels


def convert_labels(seg_img, labeled_img, label_dict, desired_labels):

    map_dict = {}
    for i in label_dict.keys():
        mask = mask_image(labeled_img,
                          i,
                          label_dict,
                          method='color_label',
                          return_type='bool')

        masked_seg = seg_img[mask[:, :, 0]]
        u = np.unique(masked_seg)
        try:
            max_ = u[0]
        except Exception:
            print('label error')

        if i in desired_labels:
            map_dict[max_] = [label_dict[i][1], i]
        else:
            # if not in desired label make 0
            map_dict[max_] = [0, i]

    seg_img = np.vectorize(map_label)(seg_img, map_dict)
    return seg_img


def map_label(x, map_dict):
    y = map_dict[x][0]
    return y


def mask_image(img, criteria, label_dict, method='color', return_type='bool'):

    if method == 'color':
        make_flat = MakeFlat(img.shape)
        flat_image = make_flat.transform(img)
        pixel_values = np.array(
            [f'{pix[0]}-{pix[1]}-{pix[2]}' for pix in flat_image[:, :3]])
        mask = pixel_values == criteria
        mask3 = np.moveaxis(np.array([mask, mask, mask]), 0, 1)
        if return_type == 'bool':
            masked_img = make_flat.inverse_transform(mask3)
        if return_type == 'original':
            masked_img = make_flat.inverse_transform(
                np.multiply(flat_image, mask3))
        if return_type == 'white':
            masked_img = make_flat.inverse_transform(mask3 * 255)

    elif method == 'color_label':
        color = label_dict[criteria][0]
        masked_img = mask_image(img,
                                color,
                                label_dict,
                                method='color',
                                return_type=return_type)

    elif method == 'label':
        mask = masked_img == criteria
        if return_type == 'bool':
            masked_img = mask
        if return_type == 'original':
            masked_img = np.multiple(img, mask)
        if return_type == 'white':
            masked_img = img * 255

    else:
        raise Exception('Method invalid')

    return masked_img


def split_eyes(segmented_image, label_dict):
    eye_label = label_dict['eyes'][1]
    nose_label = label_dict['nose'][1]
    eye_mask = segmented_image == eye_label
    nose_mask = segmented_image == nose_label

    (x, y) = create_pixel_loc(segmented_image.shape)
    x = x.reshape(segmented_image.shape)
    y = y.reshape(segmented_image.shape)
    nose_x = np.median(x[nose_mask].ravel())
    right_side = x > nose_x

    segmented_image[np.multiply(right_side, eye_mask)] += 1

    return segmented_image


def max_pool_labels(imgs, kernel=(2, 2), single_img=False):
    if single_img:
        imgs = np.array([imgs])

    out_imgs = np.zeros((imgs.shape[0], imgs.shape[1] // kernel[0],
                         imgs.shape[2] // kernel[1]),
                        dtype=np.uint8)

    for i, img in enumerate(imgs):
        b = block_reduce(img, kernel, np.max)
        out_imgs[i] = b

    if single_img:
        return out_imgs[0]
    return out_imgs


def mode_(x, **kwargs):
    return mode(x, axis=kwargs['axis']).mode[0]
