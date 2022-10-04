"""
Written by Kuangdai Leng
based on
https://github.com/AsWali/WNet/blob/master/utils/soft_n_cut_loss.py
"""

import torch
import torch.nn.functional as F


def calculate_pixel_weights(image, radius=5, oi=10, ox=4):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    assert image.size(0) == 1

    image = torch.mean(image, dim=1, keepdim=True)
    batch_size, channels, h, w = \
        image.shape[0], image.shape[1], image.shape[2], image.shape[3]

    kh, kw = radius * 2 + 1, radius * 2 + 1
    dh, dw = 1, 1
    image = F.pad(input=image, pad=(radius, radius, radius, radius),
                  mode='constant', value=0)
    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
    patches = patches.permute(0, 2, 1, 3, 4)
    patches = patches.view(-1, channels, kh, kw)
    center_values = patches[:, :, radius, radius]
    center_values = center_values[:, :, None, None]
    center_values = center_values.expand(-1, -1, kh, kw)
    patches = torch.exp(
        torch.div(-1 * ((patches - center_values) ** 2), oi ** 2))

    k_row = (torch.arange(1, kh + 1) -
             torch.arange(1, kh + 1)[radius]).expand(kh, kw)
    k_row = k_row.to(image.device)
    distance_weights = (k_row ** 2 + k_row.T ** 2)
    distance_weights = torch.exp(torch.div(-distance_weights, ox ** 2))
    mask = distance_weights.le(radius)
    distance_weights = torch.mul(mask, distance_weights)
    pixel_weights = torch.mul(patches, distance_weights)
    return pixel_weights


def _soft_n_cut_loss_single_k(weights, enc, radius=5):
    batch_size, channels, h, w = \
        enc.shape[0], enc.shape[1], enc.shape[2], enc.shape[3]
    assert channels == 1

    kh, kw = radius * 2 + 1, radius * 2 + 1
    dh, dw = 1, 1
    encoding = F.pad(input=enc, pad=(radius, radius, radius, radius),
                     mode='constant', value=0)
    seg = encoding.unfold(2, kh, dh).unfold(3, kw, dw)
    seg = seg.contiguous().view(batch_size, channels, -1, kh, kw)
    seg = seg.permute(0, 2, 1, 3, 4)
    seg = seg.view(-1, channels, kh, kw)

    nominator = torch.sum(
        enc *
        torch.sum(weights * seg, dim=(1, 2, 3)).reshape(batch_size, 1, h, w),
        dim=(1, 2, 3))
    denominator = torch.sum(
        enc *
        torch.sum(weights, dim=(1, 2, 3)).reshape(batch_size, 1, h, w),
        dim=(1, 2, 3))
    return torch.div(nominator, denominator)


def soft_n_cut_loss_fn(pixel_weights, enc):
    """ soft N-cut loss function """
    if enc.ndim == 3:
        enc = enc.unsqueeze(0)
    assert enc.size(0) == 1

    # losses for each feature
    losses = []
    k = enc.shape[1]
    for i in range(0, k):
        losses.append(_soft_n_cut_loss_single_k(pixel_weights,
                                                enc[:, (i,), :, :]))
    # average over features
    return 1. - torch.stack(losses).mean()
