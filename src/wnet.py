"""
Written by Kuangdai Leng
based on
https://github.com/AsWali/WNet/blob/master/WNet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """ convolutional block of UNet """

    def __init__(self, in_filters, out_filters, separable=True):
        super(Block, self).__init__()
        if separable:
            self.spatial1 = nn.Conv2d(in_filters, in_filters, kernel_size=3,
                                      padding=1, groups=in_filters)
            self.depth1 = nn.Conv2d(in_filters, out_filters, kernel_size=1)
            self.conv1 = lambda x: self.depth1(self.spatial1(x))

            self.spatial2 = nn.Conv2d(out_filters, out_filters, kernel_size=3,
                                      padding=1, groups=out_filters)
            self.depth2 = nn.Conv2d(out_filters, out_filters, kernel_size=1)
            self.conv2 = lambda x: self.depth2(self.spatial2(x))
        else:
            self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3,
                                   padding=1)
            self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3,
                                   padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_filters)
        self.batchnorm2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x)).clamp(0)  # clamp = relu
        x = self.batchnorm2(self.conv2(x)).clamp(0)
        return x

    def __call__(self, x):
        return self.forward(x)


class UNet(nn.Module):
    """ UNet, half of WNet """

    def __init__(self, in_chs, out_chs, ch_mul=64, n_blocks=4):
        super(UNet, self).__init__()
        # encoder
        self.enc1 = Block(in_chs, ch_mul, separable=False)
        self.enc234 = nn.ModuleList()
        for i in range(0, n_blocks - 1):
            self.enc234.append(Block(ch_mul * 2 ** i,
                                     ch_mul * 2 ** (i + 1)))
        # middle
        self.middle = Block(ch_mul * 2 ** (n_blocks - 1),
                            ch_mul * 2 ** n_blocks)
        # decoder
        self.up123 = nn.ModuleList()
        self.dec123 = nn.ModuleList()
        for i in range(n_blocks, 1, -1):
            self.up123.append(nn.ConvTranspose2d(
                ch_mul * 2 ** i,
                ch_mul * 2 ** (i - 1),
                kernel_size=3, stride=2, padding=1, output_padding=1))
            self.dec123.append(Block(ch_mul * 2 ** i,
                                     ch_mul * 2 ** (i - 1)))
        self.up4 = nn.ConvTranspose2d(ch_mul * 2, ch_mul, kernel_size=3,
                                      stride=2, padding=1, output_padding=1)
        self.dec4 = Block(ch_mul * 2, ch_mul, separable=False)
        # final
        self.final = nn.Conv2d(ch_mul, out_chs, kernel_size=(1, 1))

    def forward(self, x):
        # encoder
        enc1 = self.enc1(x)
        enc234 = []
        enc_res = enc1
        for enc_layer in self.enc234:
            enc_res = enc_layer(F.max_pool2d(enc_res, (2, 2)))
            enc234.append(enc_res)
        # middle
        middle = self.middle(F.max_pool2d(enc_res, (2, 2)))
        # decoder
        dec_res = middle
        for i, (up_layer, dec_layer) in enumerate(zip(self.up123, self.dec123)):
            up_res = torch.cat([enc234[::-1][i], up_layer(dec_res)], 1)
            dec_res = dec_layer(up_res)
        up4 = torch.cat([enc1, self.up4(dec_res)], 1)
        dec4 = self.dec4(up4)
        # final
        final = self.final(dec4)
        return final


class WNet(nn.Module):
    """ WNet """

    def __init__(self, in_chs, n_features, ch_mul=64, n_blocks=4):
        super(WNet, self).__init__()
        self.UEnc = UNet(in_chs, n_features, ch_mul=ch_mul, n_blocks=n_blocks)
        self.UDec = UNet(n_features, in_chs, ch_mul=ch_mul, n_blocks=n_blocks)
        self.in_chs = in_chs
        self.n_features = n_features

    def forward(self, x):
        if x.ndim == 3:
            enc = self.UEnc.forward(x.unsqueeze(0))
            dec = self.UDec.forward(F.softmax(enc, 1))
            return enc.squeeze(0), dec.squeeze(0)
        else:
            enc = self.UEnc.forward(x)
            dec = self.UDec.forward(F.softmax(enc, 1))
            return enc, dec
