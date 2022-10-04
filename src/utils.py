"""
Written by Kuangdai Leng
"""

import numpy as np
import torch
from skimage.transform import rotate
from sklearn import metrics


def im_crop_multiple(image, unit):
    """ crop image so that H and W are multiples of unit """
    shape = np.array(image.shape) // unit * unit
    start0 = (image.shape[0] - shape[0]) // 2
    start1 = (image.shape[1] - shape[1]) // 2
    image = image[start0:start0 + shape[0], start1:start1 + shape[1]]
    return image


def sample_patch_datasets(image, shapes_and_counts, n_batches=1,
                          returns_image_tensor=True, seed=0):
    """ sample patches from an image """
    np.random.seed(seed)
    image_patches = []
    for dx, dy, count in shapes_and_counts:
        count *= n_batches
        # locations
        try:
            x0s = np.random.randint(0, image.shape[0] - dx, count)
            y0s = np.random.randint(0, image.shape[1] - dy, count)
        except:
            # patch size too large
            continue
        # transform choices
        trans_pool = ['keep', 'h-flip', 'v-flip', 'hv-flip']
        if dx == dy:
            trans_pool += ['r-rotate', 'l-rotate']
        trans = np.random.choice(trans_pool, count)
        # sample
        patch_set = []
        for i in range(count):
            patch = image[x0s[i]:x0s[i] + dx, y0s[i]:y0s[i] + dy]
            # transform
            if trans[i] == 'h-flip':
                patch = patch[::-1, :]
            elif trans[i] == 'v-flip':
                patch = patch[:, ::-1]
            elif trans[i] == 'hv-flip':
                patch = patch[::-1, ::-1]
            elif trans[i] == 'r-rotate':
                patch = rotate(patch, 90)
            elif trans[i] == 'l-rotate':
                patch = rotate(patch, -90)
            else:
                assert trans[i] == 'keep'
            patch_set.append(np.moveaxis(patch, -1, -3))
        image_patches.append(torch.from_numpy(np.array(patch_set)))

    if returns_image_tensor:
        # original image tensor
        image_tensor = torch.from_numpy(np.moveaxis(image, -1, -3))
        return image_patches, image_tensor
    else:
        return image_patches


def compute_seg_metrics(seg_true, seg_pred):
    """ compute segmentation metric """
    assert seg_pred.shape == seg_true.shape
    seg_true = seg_true.reshape(-1)
    seg_pred = seg_pred.reshape(-1)

    # metrics
    metric_list = [metrics.rand_score,
                   metrics.adjusted_mutual_info_score,
                   metrics.homogeneity_score,
                   metrics.completeness_score,
                   metrics.v_measure_score,
                   metrics.fowlkes_mallows_score]
    metric_dict = {
        m.__name__: m(seg_true, seg_pred) for m in metric_list
    }
    return metric_dict
