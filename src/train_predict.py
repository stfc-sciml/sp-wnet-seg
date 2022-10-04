"""
Written by Kuangdai Leng
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from skimage.future.graph import rag_mean_color
from torch.utils.data import DataLoader
from tqdm import trange

from src.soft_n_cut_loss import calculate_pixel_weights, soft_n_cut_loss_fn
from src.utils import compute_seg_metrics


def _compute_mean_by_sp(features, sp_indices):
    """ compute mean within superpixels """
    C = features.size(0)
    sp_mean = torch.zeros((C, len(sp_indices)), device=features.device)
    ft_flat = features.reshape((C, -1))
    for i, idx in enumerate(sp_indices):
        sp_mean[:, i] = ft_flat[:, idx].mean(dim=-1)
    return sp_mean


def _get_seg_label_by_sp(features, sp_indices, sp_seg_mode, sp_mean=None):
    """ get segmentation labels for superpixels """
    assert sp_seg_mode in ['argmax_of_mean', 'max_count_of_argmax']
    H, W = features.size(1), features.size(2)
    if sp_seg_mode == 'argmax_of_mean':
        # mean
        if sp_mean is None:
            sp_mean = _compute_mean_by_sp(features, sp_indices)
        # argmax of mean
        label = torch.zeros(H * W, dtype=torch.long, device=features.device)
        for i, idx in enumerate(sp_indices):
            label[idx] = sp_mean[:, i].argmax()
    else:
        # argmax
        label = features.argmax(dim=0).reshape(-1)
        # max count of argmax
        for idx in sp_indices:
            label_uni, counts = torch.unique(label[idx], return_counts=True)
            label[idx] = label_uni[counts.argmax()]
    return label.reshape((H, W)), sp_mean


def _prepare_sp_data(image_tensor, image_sp,
                     requires_adj_cut=False, tau_cut=-1.,
                     requires_adj_con=False, tau_con=0.,
                     use_sparse_adj=False):
    """ prepare superpixel data """
    # graph
    rag = rag_mean_color(
        image_tensor.permute(1, 2, 0).cpu().numpy(), image_sp)

    # indices
    sp_indices = []
    image_sp_flat = torch.from_numpy(image_sp.reshape(-1))
    for i_node, attr_node in rag.nodes.items():
        idx = torch.where(image_sp_flat == attr_node['labels'][0])[0]
        sp_indices.append(idx.to(image_tensor.device))

    # compute adj for n-cut
    N = len(rag.nodes)
    adj_cut = None
    adj_cut_sum = None
    if requires_adj_cut:
        if use_sparse_adj:
            idx, val = [], []
            for node_i in range(N):
                for node_j in rag.adj[node_i].keys():
                    dist = rag.adj[node_i][node_j]['weight']
                    idx.append([node_i, node_j])
                    val.append(math.exp(tau_cut * dist))
            Aij = torch.sparse_coo_tensor(torch.tensor(idx).t(), val, (N, N))
            adj_cut = Aij.to(image_tensor.device)
            adj_cut_sum = adj_cut.to_dense().sum(dim=1)
        else:
            Aij = torch.zeros((N, N))
            for node_i in range(N):
                for node_j in rag.adj[node_i].keys():
                    dist = rag.adj[node_i][node_j]['weight']
                    Aij[node_i, node_j] = math.exp(tau_cut * dist)
            adj_cut = Aij.to(image_tensor.device)
            adj_cut_sum = adj_cut.sum(dim=1)

    # compute adj for continuity
    adj_con = None
    if requires_adj_con:
        if use_sparse_adj:
            idx, val = [], []
            for node_i in range(N):
                dist_to_i = torch.zeros(len(rag.adj[node_i]))
                for j, node_j in enumerate(rag.adj[node_i].keys()):
                    dist_to_i[j] = rag.adj[node_i][node_j]['weight']
                # tau_con = -inf => use the closest neighbor
                # tau_con = 0    => use ave of all neighbors
                # tau_con = inf  => use the farthest neighbor
                con_fact = torch.softmax(tau_con * dist_to_i, dim=0)
                for j, node_j in enumerate(rag.adj[node_i].keys()):
                    idx.append([node_i, node_j])
                    val.append(con_fact[j])
            Aij = torch.sparse_coo_tensor(torch.tensor(idx).t(), val, (N, N))
        else:
            Aij = torch.zeros((N, N))
            for node_i in range(N):
                dist_to_i = torch.zeros(len(rag.adj[node_i]))
                for j, node_j in enumerate(rag.adj[node_i].keys()):
                    dist_to_i[j] = rag.adj[node_i][node_j]['weight']
                # tau_con = -inf => use the closest neighbor
                # tau_con = 0    => use ave of all neighbors
                # tau_con = inf  => use the farthest neighbor
                con_fact = torch.softmax(tau_con * dist_to_i, dim=0)
                for j, node_j in enumerate(rag.adj[node_i].keys()):
                    Aij[node_i, node_j] = con_fact[j]
        adj_con = Aij.to(image_tensor.device)
    return sp_indices, adj_cut, adj_cut_sum, adj_con


def predict_sp_wnet(wnet, image_tensor_list, image_sp_list=None,
                    sp_seg_mode='argmax_of_mean',
                    device='cuda', make_label_continuous=False,
                    returns_in_numpy=True):
    """ predict, returns reconstruction, features and labels """
    # input check
    if image_sp_list is not None:
        assert len(image_sp_list) == len(image_tensor_list)

    # nn
    wnet.to(device)
    training = wnet.training
    wnet.eval()

    # results
    ft_img_list, rc_img_list, label_img_list = [], [], []
    for i_image, image_tensor in enumerate(image_tensor_list):
        # forward
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            ft_img, rc_img = wnet.forward(image_tensor)

        # segmentation
        if image_sp_list is None:
            label_img = ft_img.argmax(dim=0)
        else:
            sp_indices, _, _, _ = _prepare_sp_data(image_tensor,
                                                   image_sp_list[i_image])
            label_img, _ = _get_seg_label_by_sp(ft_img, sp_indices, sp_seg_mode)
        # sort label as 0, 1, 2, ...
        if make_label_continuous:
            for i, lab in enumerate(torch.unique(label_img)):
                label_img[label_img == lab] = i

        # usually more convenient in numpy
        if returns_in_numpy:
            ft_img = ft_img.permute(1, 2, 0).cpu().numpy()
            rc_img = rc_img.permute(1, 2, 0).cpu().numpy()
            label_img = label_img.cpu().numpy()
        ft_img_list.append(ft_img)
        rc_img_list.append(rc_img)
        label_img_list.append(label_img)
    if training:
        wnet.train()
    return ft_img_list, rc_img_list, label_img_list


def train_sp_wnet(wnet,
                  # image data
                  image_tensor_list, image_patches_list=None, n_batches=None,
                  # superpixel
                  image_sp_list=None, sp_seg_mode='argmax_of_mean',
                  tau_cut=-1., tau_con=0.,
                  use_sparse_adj=False,
                  # beta values
                  beta_rc_image=1., beta_rc_patches=None,
                  beta_cut=None, beta_sim=None, beta_con=None,
                  # results after each epoch
                  plot_epoch=False, truth_for_metrics_epoch=None,
                  save_epoch_results_to='screen',
                  # others
                  epochs=10, lr=0.001, device='cuda', progress_bar=True):
    """ train WNet """
    ################
    # prepare data #
    ################
    # input check
    with_patches = False
    if image_patches_list is not None:
        assert len(image_patches_list) == len(image_tensor_list)
        with_patches = True
    with_sp = False
    if image_sp_list is not None:
        assert len(image_sp_list) == len(image_tensor_list)
        with_sp = True

    # patch data
    loaders_list = []
    if not with_patches:
        assert n_batches is None
        assert beta_rc_patches is None
        n_batches = 1
    else:
        if beta_rc_patches is None:
            # ignore patches
            n_batches = 1
        else:
            assert n_batches is not None
            for image_patches in image_patches_list:
                loaders = []
                for patches in image_patches:
                    batch_size = patches.size(0) // n_batches
                    loader = DataLoader(patches, batch_size=batch_size,
                                        shuffle=True)
                    loaders.append(loader)
                loaders_list.append(loaders)

    # sp data
    n_cut_pixel_weights_list = []
    sp_indices_list, adj_cut_list, adj_cut_sum_list, adj_con_list = \
        [], [], [], []
    if with_sp:
        for image_tensor, image_sp in zip(image_tensor_list, image_sp_list):
            sp_indices, adj_cut, adj_cut_sum, adj_con = _prepare_sp_data(
                image_tensor.to(device), image_sp,
                requires_adj_cut=beta_cut is not None, tau_cut=tau_cut,
                requires_adj_con=beta_con is not None, tau_con=tau_con,
                use_sparse_adj=use_sparse_adj)
            sp_indices_list.append(sp_indices)
            adj_cut_list.append(adj_cut)
            adj_cut_sum_list.append(adj_cut_sum)
            adj_con_list.append(adj_con)
    else:
        if beta_cut is not None:
            for image_tensor in image_tensor_list:
                n_cut_pixel_weights = calculate_pixel_weights(
                    image_tensor.to(device))
                n_cut_pixel_weights_list.append(n_cut_pixel_weights)

    ######################
    # prepare nn and log #
    ######################
    wnet.to(device)
    wnet.train()
    optimizer = torch.optim.Adam(wnet.parameters(), lr=lr)
    rc_loss_fn = torch.nn.MSELoss(reduction='mean')
    sim_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    con_loss_fn = torch.nn.L1Loss(reduction='mean')

    # training history
    keys = ['loss', 'loss_rc_image', 'loss_rc_patches',
            'loss_cut', 'loss_sim', 'loss_con']
    hist = {key: [] for key in keys}

    # epoch results
    need_epoch_res = (plot_epoch or truth_for_metrics_epoch is not None)
    out_dir = None
    if need_epoch_res and save_epoch_results_to.lower() != 'screen':
        out_dir = Path(save_epoch_results_to)
        out_dir.mkdir(parents=True, exist_ok=True)
    if plot_epoch:
        assert wnet.in_chs == 1 or wnet.in_chs == 3

    ############
    # training #
    ############
    for epoch in range(epochs):
        # logger
        logger = {key: 0. for key in keys}
        logger['n'] = 0

        # patch data iters
        iters_list = []
        for loaders in loaders_list:
            iters = []
            for loader in loaders:
                iters.append(iter(loader))
            iters_list.append(iters)

        # batch loop
        batches = trange(0, n_batches, desc=f'Epoch {epoch + 1}',
                         unit='batch', disable=not progress_bar,
                         position=0, file=sys.stdout, ascii=True)
        for _ in batches:
            # zero grad for each batch
            optimizer.zero_grad()

            # total loss
            loss = torch.tensor(0., device=device)

            # patch reconstruction loss
            loss_rc_patches = torch.tensor(0., device=device)
            if beta_rc_patches is not None:
                for iters in iters_list:
                    loss_rc_patch = torch.tensor(0., device=device)
                    for it in iters:
                        x = next(it).to(device)
                        _, rc = wnet.forward(x)
                        loss_rc_patch += rc_loss_fn(rc, x)
                    loss_rc_patches += loss_rc_patch / len(iters)
                loss += loss_rc_patches * beta_rc_patches

            # loop over images
            loss_rc_image = torch.tensor(0., device=device)
            loss_cut = torch.tensor(0., device=device)
            loss_sim = torch.tensor(0., device=device)
            loss_con = torch.tensor(0., device=device)
            for i_image, image_tensor in enumerate(image_tensor_list):
                image_tensor = image_tensor.to(device)

                # image reconstruction loss
                ft_img, rc_img = wnet.forward(image_tensor)
                if beta_rc_image is not None:
                    loss_rc_image += rc_loss_fn(rc_img, image_tensor)
                    loss += loss_rc_image * beta_rc_image

                if not with_sp:
                    ###############
                    # pixel-space #
                    ###############
                    # cut loss
                    if beta_cut is not None:
                        loss_cut += soft_n_cut_loss_fn(
                            n_cut_pixel_weights_list[i_image],
                            torch.softmax(ft_img, dim=0))
                        loss += loss_cut * beta_cut

                    # similarity loss
                    if beta_sim is not None:
                        label = ft_img.argmax(dim=0)
                        loss_sim += sim_loss_fn(ft_img.unsqueeze(0),
                                                label.unsqueeze(0))
                        loss += loss_sim * beta_sim

                    # continuity loss
                    if beta_con is not None:
                        loss_con += con_loss_fn(ft_img[:, :-1, :],
                                                ft_img[:, 1:, :]) / 2.
                        loss_con += con_loss_fn(ft_img[:, :, :-1],
                                                ft_img[:, :, 1:]) / 2.
                        loss += loss_con * beta_con
                else:
                    ft_sp_mean = None  # to avoid repeated computation
                    sp_indices = sp_indices_list[i_image]
                    ####################
                    # superpixel-space #
                    ####################
                    # cut loss
                    if beta_cut is not None:
                        ft_sp_mean = _compute_mean_by_sp(ft_img, sp_indices)
                        prob = torch.softmax(ft_sp_mean, dim=0)
                        # loop over each class
                        total = torch.tensor(0., device=image_tensor.device)
                        K = ft_sp_mean.size(0)
                        for k in range(K):
                            no = torch.dot(torch.mv(adj_cut_list[i_image],
                                                    prob[k]), prob[k])
                            de = torch.dot(adj_cut_sum_list[i_image], prob[k])
                            total += no / de
                        loss_cut += (K - total) / K
                        loss += loss_cut * beta_cut

                    # similarity loss
                    if beta_sim is not None:
                        labels, ft_sp_mean = _get_seg_label_by_sp(
                            ft_img, sp_indices, sp_seg_mode=sp_seg_mode,
                            sp_mean=ft_sp_mean)
                        loss_sim += sim_loss_fn(ft_img.unsqueeze(0),
                                                labels.unsqueeze(0))
                        loss += loss_sim * beta_sim

                    # continuity loss
                    if beta_con is not None:
                        if ft_sp_mean is None:
                            ft_sp_mean = _compute_mean_by_sp(ft_img, sp_indices)
                        ft_sp_adj = torch.mm(adj_con_list[i_image],
                                             ft_sp_mean.t()).t()
                        loss_con += con_loss_fn(ft_sp_mean, ft_sp_adj)
                        loss += loss_con * beta_con

            # backprop
            loss.backward()
            optimizer.step()

            # log
            for key in keys:
                logger[key] += eval(f'{key}.item()')
            logger['n'] += 1
            batches.set_postfix_str(', '.join(
                [f"{key}={logger[key] / logger['n']:.2e}" for key in
                 keys]).replace('0.00e+00', 'unused'))

        # history
        for key in keys:
            hist[key].append(logger[key] / logger['n'])

        # process epoch results
        if need_epoch_res:
            # predict
            ft_img_list, rc_img_list, label_img_list = predict_sp_wnet(
                wnet, image_tensor_list,
                image_sp_list=image_sp_list, sp_seg_mode=sp_seg_mode,
                device=device)

            # plots
            if plot_epoch:
                _, ax = plt.subplots(len(image_tensor_list), 2, dpi=200,
                                     squeeze=False)
                for i_image in range(len(image_tensor_list)):
                    ax[i_image, 0].imshow(rc_img_list[i_image].clip(0., 1.))
                    ax[i_image, 1].imshow(label_img_list[i_image],
                                          cmap='tab20')
                    ax[i_image, 0].axis('off')
                    ax[i_image, 1].axis('off')
                if save_epoch_results_to == 'screen':
                    plt.show()
                else:
                    plt.savefig(out_dir / f'rc_seg_epoch{epoch}.jpg')
                    plt.close()

            # metrics
            if truth_for_metrics_epoch is not None:
                metrics = []
                for label_img in label_img_list:
                    metric_dict = compute_seg_metrics(
                        truth_for_metrics_epoch, label_img)
                    metrics.append(metric_dict)
                if save_epoch_results_to == 'screen':
                    print(metrics)
                else:
                    with open(out_dir / f'metrics_epoch{epoch}.json', 'w') as f:
                        f.write(json.dumps(metrics))

    # clean up
    wnet.cpu()
    return hist
