'''
Utility functions for viewing images
'''

# Imports
import matplotlib.pyplot as plt
import numpy as np


def view_prediction(pipe, X, **kwargs):
    '''

    Dispalys the image along with the associated model prediction,

    Parameters
        model (sklearn model instance)
        img (np.array)
        kwargs to pass onto view_data

    the model used must be specified. ground_truth is an optional argument
    to pass the original label. overlap=True will dispaly the predicted
    labels overlaid on the original image.
    '''

    y_real_predict = pipe.predict(X).reshape(X.shape[:2])

    view_data(img=X, pred=y_real_predict, **kwargs)


def view_data(img=None, pred=None, gt=None, to_overlap=False, ax=None, axis=1):
    '''
    Shows images.

    Arguments
        img
        pred
        gt
        to_overlap
        ax
        axis
        num_to_show
    '''


    # Color map for plotting labels
    newcmp = plt.cm.get_cmap('Set1', 8)
    to_show = []

    if img is None and pred is None and gt is None:
        raise Exception('Must specify an image to view')

    if img is not None:
        to_show.append((img, 'Image'))

    if pred is not None:
        to_show.append((pred, 'Prediction'))

    if gt is not None:
        to_show.append((gt, 'Ground Truth'))

    if to_overlap is not False:
        if to_overlap == 'gt':
            overlap = gt
        elif to_overlap == 'pred':
            overlap = pred
        to_show.append((overlap, 'Overlap'))

    if ax is None:
        if axis == 1:
            kwargs = {'ncols': len(to_show),
                      'nrows': 1,
                      'figsize': (4 * len(to_show), 4)}
        elif axis == 0:
            kwargs = {'ncols': 1,
                      'nrows': len(to_show),
                      'figsize': (4, 4 * len(to_show))}

        fig, ax = plt.subplots(**kwargs)

    if not isinstance(ax, np.ndarray):
        ax = [ax]


    for i, e in enumerate(to_show):
        if axis == 0:
            row = i
        elif axis == 1:
            col = i            

        ax[i].axis('off')
        if e[1] == 'Image':
            ax[i].imshow(e[0])
        elif e[1] == 'Overlap':
            ax[i].imshow(img, alpha=0.5)
            ax[i].imshow(e[0], alpha=0.5, cmap=newcmp)
        else:
            ax[i].imshow(e[0], cmap=newcmp)

        ax[i].set_title(e[1])
