# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
from torch import nn as nn
from torch.nn import functional as F
import math

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros, Linear, LinearZeros
from utils.util import opt_get

NUM_SminusR = 8 #8 for 11,12,13 # 8 for 14
NUM_freeT = NUM_SminusR - 4
NUM_freeA = NUM_SminusR - 2

class CondBsplineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt, std_mode=True, bias=True):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 320
        self.kernel_hidden = 1
        #self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        self.std_channels = opt_get(opt, ['network_G', 'flow', 'std_channels'])
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)
        self.bias = bias
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn
        
        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fBspline = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * (NUM_freeT+NUM_freeA+self.bias), # 2 for (shift, scale)
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * (NUM_freeT+NUM_freeA+self.bias), # 2 for (shift, scale)
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

        self.fBspline_std = self.F(in_channels=self.channels_for_nn + self.std_channels,
                                  out_channels=self.channels_for_co * (NUM_freeT+NUM_freeA+self.bias), # 2 for (shift, scale)
                                  hidden_channels=self.hidden_channels,
                                  kernel_hidden=self.kernel_hidden,
                                  n_hidden_layers=self.n_hidden_layers)

        self.stdmode = std_mode
        # +-0.5 for 11,12,13, +-1.0 for 14
        self._left = -0.5
        self._right = 0.5
        self._top = 0.5
        self._bottom = -0.5

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None, std=None):
        if std is not None:
            h, w = input.size(2), input.size(3)
            if self.std_channels == 3 and std.size(1) == 3:
                std_expand = torch.nn.functional.interpolate(std, size=(h, w), mode='bicubic')
            elif self.std_channels == 3 and std.size(1) == 1:
                std_expand = std.expand(-1, 3, h, w)
            elif self.std_channels == 1:
                std_expand = std.expand(-1, -1, h, w)

        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            # Feature Conditional
            # scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            # z = z + shiftFt
            # z = z * scaleFt
            # logdet = logdet + self.get_logdet(scaleFt)
            if not self.bias:
                tFt, alphaFt = self.feature_extract(ft, self.fFeatures)
                z, logdetFt = cubic_B_spline(
                    z,
                    tFt,
                    alphaFt,
                    inverse=False,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                logdet = logdet + thops.sum(logdetFt, dim=[1,2,3])

                if self.stdmode and std is not None:
                    # Std Conditional
                    z1, z2 = self.split(z)
                    #scaleFt, shiftFt = self.feature_extract_aff(z1, std_expand, self.fAffine_std)
                    tFt, alphaFt = self.feature_extract_Bspline(z1, std_expand, self.fBspline_std)
                    #self.asserts(scaleFt, shiftFt, z1, z2)
                    #z2 = z2 + shiftFt
                    #z2 = z2 * scaleFt
                    #logdet = logdet + self.get_logdet(scaleFt)
                    z2, logdet2 = cubic_B_spline(
                        z2,
                        tFt,
                        alphaFt,
                        inverse=False,
                        left=self._left,
                        right=self._right,
                        top=self._top,
                        bottom=self._bottom,
                    )
                    z = thops.cat_feature(z1, z2)
                    logdet = logdet + thops.sum(logdet2, dim=[1,2,3])
                # Self Conditional
                z1, z2 = self.split(z)
                # scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
                t, alpha = self.feature_extract_Bspline(z1, ft, self.fBspline)
                #self.asserts(scale, shift, z1, z2)
                #z2 = z2 + shift
                #z2 = z2 * scale

                z2, logdet2 = cubic_B_spline(
                    z2,
                    t,
                    alpha,
                    inverse=False,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                output = thops.cat_feature(z1, z2)
                logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])
            else:
                tFt, alphaFt, shiftFt= self.feature_extract(ft, self.fFeatures)
                z, logdetFt = cubic_B_spline(
                    z,
                    tFt,
                    alphaFt,
                    inverse=False,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                z = z + shiftFt
                logdet = logdet + thops.sum(logdetFt, dim=[1, 2, 3])

                if self.stdmode and std is not None:
                    # Std Conditional
                    z1, z2 = self.split(z)
                    # scaleFt, shiftFt = self.feature_extract_aff(z1, std_expand, self.fAffine_std)
                    tFt, alphaFt, shiftFt = self.feature_extract_Bspline(z1, std_expand, self.fBspline_std)
                    # self.asserts(scaleFt, shiftFt, z1, z2)
                    # z2 = z2 + shiftFt
                    # z2 = z2 * scaleFt
                    # logdet = logdet + self.get_logdet(scaleFt)
                    z2, logdet2 = cubic_B_spline(
                        z2,
                        tFt,
                        alphaFt,
                        inverse=False,
                        left=self._left,
                        right=self._right,
                        top=self._top,
                        bottom=self._bottom,
                    )
                    z2 = z2 + shiftFt
                    z = thops.cat_feature(z1, z2)
                    logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])
                # Self Conditional
                z1, z2 = self.split(z)
                # scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
                t, alpha, shift = self.feature_extract_Bspline(z1, ft, self.fBspline)
                # self.asserts(scale, shift, z1, z2)
                # z2 = z2 + shift
                # z2 = z2 * scale

                z2, logdet2 = cubic_B_spline(
                    z2,
                    t,
                    alpha,
                    inverse=False,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                z2 = z2 + shift
                output = thops.cat_feature(z1, z2)
                logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])

        else:
            if not self.bias:
                z = input
                # Self Conditional
                z1, z2 = self.split(z)
                t, alpha = self.feature_extract_Bspline(z1, ft, self.fBspline)
                # self.asserts(scale, shift, z1, z2)
                # z2 = z2 / scale
                # z2 = z2 - shift

                z2, logdet2 = cubic_B_spline(
                    z2,
                    t,
                    alpha,
                    inverse=True,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                z = thops.cat_feature(z1, z2)
                logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])

                if self.stdmode and std is not None:
                    # Std Conditional
                    z1, z2 = self.split(z)
                    # scaleFt, shiftFt = self.feature_extract_aff(z1, std_expand, self.fAffine_std)
                    tFt, alphaFt = self.feature_extract_Bspline(z1, std_expand, self.fBspline_std)
                    #z2 = z2 / scaleFt
                    #z2 = z2 - shiftFt
                    z2, logdet2 = cubic_B_spline(
                        z2,
                        tFt,
                        alphaFt,
                        inverse=True,
                        left=self._left,
                        right=self._right,
                        top=self._top,
                        bottom=self._bottom,
                    )
                    z = thops.cat_feature(z1, z2)
                    #logdet = logdet - self.get_logdet(scaleFt)
                    logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])

                # Feature Conditional
                tFt, alphaFt = self.feature_extract(ft, self.fFeatures)
                z, logdetFt = cubic_B_spline(
                    z,
                    tFt,
                    alphaFt,
                    inverse=True,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                logdet = logdet + thops.sum(logdetFt, dim=[1, 2, 3])
                output = z
            else:
                z = input
                # Self Conditional
                z1, z2 = self.split(z)
                t, alpha, shift = self.feature_extract_Bspline(z1, ft, self.fBspline)
                # self.asserts(scale, shift, z1, z2)
                # z2 = z2 / scale
                z2 = z2 - shift

                z2, logdet2 = cubic_B_spline(
                    z2,
                    t,
                    alpha,
                    inverse=True,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                z = thops.cat_feature(z1, z2)
                logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])

                if self.stdmode and std is not None:
                    # Std Conditional
                    z1, z2 = self.split(z)
                    # scaleFt, shiftFt = self.feature_extract_aff(z1, std_expand, self.fAffine_std)
                    tFt, alphaFt, shiftFt = self.feature_extract_Bspline(z1, std_expand, self.fBspline_std)
                    # z2 = z2 / scaleFt
                    z2 = z2 - shiftFt
                    z2, logdet2 = cubic_B_spline(
                        z2,
                        tFt,
                        alphaFt,
                        inverse=True,
                        left=self._left,
                        right=self._right,
                        top=self._top,
                        bottom=self._bottom,
                    )
                    z = thops.cat_feature(z1, z2)
                    # logdet = logdet - self.get_logdet(scaleFt)
                    logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])

                # Feature Conditional
                tFt, alphaFt, shiftFt = self.feature_extract(ft, self.fFeatures)
                z = z - shiftFt
                z, logdetFt = cubic_B_spline(
                    z,
                    tFt,
                    alphaFt,
                    inverse=True,
                    left=self._left,
                    right=self._right,
                    top=self._top,
                    bottom=self._bottom,
                )
                logdet = logdet + thops.sum(logdetFt, dim=[1, 2, 3])
                output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        if self.bias:
            h = f(z)
            newh = h.view(h.shape[0], self.in_channels, -1, h.shape[2], h.shape[3])
            newh = torch.permute(newh, (0, 1, 3, 4, 2))
            t = newh[..., :NUM_freeT]
            alpha = newh[..., NUM_freeT:-1]
            shift = newh[..., -1]
            return t, alpha, shift
        else:
            h = f(z)
            newh = h.view(h.shape[0], self.in_channels, -1, h.shape[2], h.shape[3])
            newh = torch.permute(newh, (0, 1, 3, 4, 2))
            t = newh[..., :NUM_freeT]
            alpha = newh[..., NUM_freeT:]
            return t, alpha

    def feature_extract_Bspline(self, z1, ft, f):
        if self.bias:
            z = torch.cat([z1, ft], dim=1)
            h = f(z) # B (C_co * (n1+n2+1)) H W
            newh = h.view(h.shape[0],self.channels_for_co,-1,h.shape[2],h.shape[3])
            newh = torch.permute(newh, (0,1,3,4,2))
            t = newh[...,:NUM_freeT]
            alpha = newh[...,NUM_freeT:-1]
            shift = newh[...,-1]
            return t, alpha, shift
        else:
            z = torch.cat([z1, ft], dim=1)
            h = f(z) # B (C_co * (n1+n2+1)) H W
            newh = h.view(h.shape[0],self.channels_for_co,-1,h.shape[2],h.shape[3])
            newh = torch.permute(newh, (0,1,3,4,2))
            t = newh[...,:NUM_freeT]
            alpha = newh[...,NUM_freeT:]
            return t, alpha

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

    # actually FL is not used.
    def FL(self, in_channels, out_channels, hidden_channels, n_hidden_layers=1):
        layers = [Linear(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU(inplace=False))
        layers.append(LinearZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[...,-1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def cbrt(x, eps=0):
    ans = torch.sign(x)*torch.exp(torch.log(torch.abs(x))/3.0)
    return ans


def sqrt(x, eps=1e-9):
    ans = torch.exp((torch.log(torch.abs(x))) / 2.0)
    return ans


def cubic_B_spline(
        inputs_whole,
        unnormalized_dt_whole,
        unnormalized_dalpha_whole,
        inverse=False,
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
        min_bin_width=0.01,
        min_bin_height=0.01,
        eps=1e-4,
        quadratic_threshold = 1e-7,
        linear_threshold = 1e-7
):
    # inputs
    # inputs_whole = B (C_co or 2*C_co) H W

    # assume that we deal with the last dimension
    # unnormalized_t = B C_co H W (s-r-4)
    # unnormalized_alpha = B C_co H W (s-r-2)

    # outputs
    # return : outputs, logabsdet
    outputs_whole = torch.zeros_like(inputs_whole)
    logabsdet_whole = torch.zeros_like(inputs_whole)

    # outliers : Identity
    if inverse:
        bottom_mask = inputs_whole <= bottom
        top_mask = inputs_whole >= top
        outputs_whole[bottom_mask] = inputs_whole[bottom_mask]
        outputs_whole[top_mask] = inputs_whole[top_mask]
        logabsdet_whole[bottom_mask] = 0.
        logabsdet_whole[top_mask] = 0.
        inside_mask = ~torch.logical_or(bottom_mask, top_mask)
    else:
        left_mask = inputs_whole <= left
        right_mask = inputs_whole >= right
        outputs_whole[left_mask] = inputs_whole[left_mask]
        outputs_whole[right_mask] = inputs_whole[right_mask]
        logabsdet_whole[left_mask] = 0.
        logabsdet_whole[right_mask] = 0.
        inside_mask = ~torch.logical_or(left_mask, right_mask)

    inputs = inputs_whole[inside_mask]
    unnormalized_dt = unnormalized_dt_whole[inside_mask]
    unnormalized_dalpha = unnormalized_dalpha_whole[inside_mask]
    # Non-uniform B-spline parameter generation

    num_d = unnormalized_dt.shape[-1] # s-r-4
    # we know C_co = 1 in 2dToyExperiment
    assert (num_d + 2 == unnormalized_dalpha.shape[-1])

    # generate t
    dt = torch.softmax(unnormalized_dt, dim=-1)
    dt = dt * (1 - 4 * min_bin_width)
    dt = min_bin_width + (1 - dt.shape[-1] * min_bin_width / (1 - 4 * min_bin_width)) * dt
    dt = F.pad(dt, pad=(4, 4), mode='constant', value=min_bin_width)

    t = torch.cumsum(dt, dim=-1)
    t = F.pad(t, pad=(1, 0), mode='constant', value=0.0)
    t = t - 2 * min_bin_width

    # generate alpha
    dalpha = torch.softmax(unnormalized_dalpha, dim=-1)
    dalpha = dalpha * (1 - 2 * min_bin_height)
    dalpha = min_bin_height + (1 - dalpha.shape[-1] * min_bin_height / (1 - 2 * min_bin_height)) * dalpha
    dalpha = F.pad(dalpha, pad=(2, 2), mode='constant', value=min_bin_height)

    alpha = torch.cumsum(dalpha, dim=-1)
    alpha = F.pad(alpha, pad=(1, 0), mode='constant', value=0.0)
    alpha = alpha - min_bin_height

    # t = torch.roll(t, shifts=-2, dims=-1)
    # alpha = torch.roll(alpha, shifts=-3, dims=-1)

    widths2 = dt
    knots3 = alpha
    num_bins = NUM_SminusR

    cumwidths = t[..., 2:num_bins + 3]

    cumheights = knots3[..., 0:num_bins + 1] * (torch.square(widths2[..., 2:num_bins + 3]) / (
            (widths2[..., 0:num_bins + 1] + widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3])
            * (widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3])
            )
                                                ) \
                 + knots3[..., 1:num_bins + 2] * (
                         (widths2[..., 2:num_bins + 3] * (widths2[..., 0:num_bins + 1] + widths2[..., 1:num_bins + 2]))
                         / ((widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3]) * (
                         widths2[..., 0:num_bins + 1] + widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3]))
                         + (widths2[..., 1:num_bins + 2] * (
                         widths2[..., 2:num_bins + 3] + widths2[..., 3:num_bins + 4]))
                         / ((widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3]) * (
                         widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3] + widths2[..., 3:num_bins + 4]))
                 ) \
                 + knots3[..., 2:num_bins + 3] * (
                         torch.square(widths2[..., 1:num_bins + 2]) / (
                         (widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3] + widths2[..., 3:num_bins + 4])
                         * (widths2[..., 1:num_bins + 2] + widths2[..., 2:num_bins + 3])
                 )
                 )

    t = torch.roll(t, shifts=-2, dims=-1)
    knots = torch.roll(alpha, shifts=-3, dims=-1)

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
        bin_idx = searchsorted(cumheights, inputs)[..., None]

        i0 = bin_idx
        im1 = torch.remainder(bin_idx - 1, num_bins + 3)
        im2 = torch.remainder(bin_idx - 2, num_bins + 3)
        im3 = torch.remainder(bin_idx - 3, num_bins + 3)

        j3 = bin_idx + 3
        j2 = bin_idx + 2
        j1 = bin_idx + 1
        j0 = bin_idx
        jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
        jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

        km0 = knots.gather(-1, i0)[..., 0]
        km1 = knots.gather(-1, im1)[..., 0]
        km2 = knots.gather(-1, im2)[..., 0]
        km3 = knots.gather(-1, im3)[..., 0]

        t3 = t.gather(-1, j3)[..., 0]
        t2 = t.gather(-1, j2)[..., 0]
        t1 = t.gather(-1, j1)[..., 0]
        t0 = t.gather(-1, j0)[..., 0]
        tm1 = t.gather(-1, jm1)[..., 0]
        tm2 = t.gather(-1, jm2)[..., 0]

        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]

        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]


        inputs_a1 = km0 * (
                1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            - 1 / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            - 1 / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            - 1 / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            1 / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + 1 / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + 1 / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            -1 / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_b1 = km0 * (
                (-3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (2 * tm1 + t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (tm1 + t2 + t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (t3 + 2 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            (-2 * t1 - tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + (-2 * t2 - t0) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + (-t2 - t1 - tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (3 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_c1 = km0 * (
                (3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (- tm1 * tm1 - 2 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (- tm1 * t2 - tm1 * t0 - t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (- t0 * t0 - 2 * t3 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            (t1 * t1 + 2 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            + (t2 * t2 + 2 * t0 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            + (t2 * t1 + tm1 * t1 + t2 * tm1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (-3 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs_d1 = km0 * (
                (- t0 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
        ) + km1 * (
                            (tm1 * tm1 * t1) / ((t2 - tm1) * (t1 - tm1) * (t1 - t0))
                            + (tm1 * t2 * t0) / ((t2 - tm1) * (t2 - t0) * (t1 - t0))
                            + (t3 * t0 * t0) / ((t3 - t0) * (t2 - t0) * (t1 - t0))
                    ) + km2 * (
                            - (t1 * t1 * tm2) / ((t1 - t0) * (t1 - tm2) * (t1 - tm1))
                            - (t0 * t2 * t2) / ((t1 - t0) * (t2 - t0) * (t2 - tm1))
                            - (t2 * tm1 * t1) / ((t1 - t0) * (t1 - tm1) * (t2 - tm1))
                    ) + km3 * (
                            (t1 * t1 * t1) / ((t1 - tm2) * (t1 - tm1) * (t1 - t0))
                    )

        inputs = torch.as_tensor(inputs, dtype=torch.double)
        outputs = torch.zeros_like(inputs)
        inputs_b_ = torch.as_tensor(inputs_b1 / inputs_a1 / 3., dtype=torch.double, device=inputs_b1.device)
        inputs_c_ = torch.as_tensor(inputs_c1 / inputs_a1 / 3., dtype=torch.double, device=inputs_b1.device)
        inputs_d_ = torch.as_tensor((inputs_d1 - inputs) / inputs_a1, dtype=torch.double, device=inputs_b1.device)
        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        # Deal with one root cases.
        p_ = torch.zeros_like(inputs)
        p_[one_root_mask] = cbrt((-depressed_1[one_root_mask] + sqrt(-discriminant[one_root_mask])) / 2.)

        p = p_[one_root_mask]
        q = cbrt((-depressed_1[one_root_mask] - sqrt(-discriminant[one_root_mask])) / 2.)

        outputs_one_root = ((p + q) - inputs_b_[one_root_mask])

        outputs[one_root_mask] = torch.as_tensor(outputs_one_root, dtype=outputs.dtype)

        # Deal with three root cases.

        theta = torch.atan2(sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

        root_scale = 2 * sqrt(-depressed_2[three_roots_mask])
        root_shift = -inputs_b_[three_roots_mask]

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
        root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
        root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
        root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        roots = torch.stack([root_1, root_2, root_3], dim=-1)

        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
        output_three_roots = torch.gather(roots, dim=-1, index=mask_index).view(-1)
        outputs[three_roots_mask] = torch.as_tensor(output_three_roots, dtype=outputs.dtype)

        # Deal with a -> 0 (almost quadratic) cases.

        quadratic_mask = inputs_a1.abs() < quadratic_threshold
        a = inputs_b1[quadratic_mask]
        b = inputs_c1[quadratic_mask]
        c = (inputs_d1[quadratic_mask] - inputs[quadratic_mask])
        alpha = (-b + sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
        outputs[quadratic_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)  # + input_left_cumwidths[quadratic_mask]

        # Deal with b-> 0 (almost linear) cases.
        linear_mask = inputs_b1.abs() < linear_threshold
        linear_mask = linear_mask * quadratic_mask
        b = inputs_c1[linear_mask]
        c = (inputs_d1[linear_mask] - inputs[linear_mask])
        alpha = c / b
        outputs[linear_mask] = torch.as_tensor(alpha, dtype=outputs.dtype)
        outputs = torch.clamp(outputs, input_left_cumwidths, input_right_cumwidths)
        logabsdet = -torch.log(
            (torch.abs(
                (3 * inputs_a1 * outputs.pow(2)
                 + 2 * inputs_b1 * outputs
                 + inputs_c1))
            )
        )
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)

    else:
        inputs = (inputs - left) / (right - left)
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

        i0 = bin_idx
        im1 = torch.remainder(bin_idx - 1, num_bins + 3)
        im2 = torch.remainder(bin_idx - 2, num_bins + 3)
        im3 = torch.remainder(bin_idx - 3, num_bins + 3)

        j3 = bin_idx + 3
        j2 = bin_idx + 2
        j1 = bin_idx + 1
        j0 = bin_idx
        jm1 = torch.remainder(bin_idx - 1, num_bins + 5)
        jm2 = torch.remainder(bin_idx - 2, num_bins + 5)

        km0 = knots.gather(-1, i0)[..., 0]
        km1 = knots.gather(-1, im1)[..., 0]
        km2 = knots.gather(-1, im2)[..., 0]
        km3 = knots.gather(-1, im3)[..., 0]

        t3 = t.gather(-1, j3)[..., 0]
        t2 = t.gather(-1, j2)[..., 0]
        t1 = t.gather(-1, j1)[..., 0]
        t0 = t.gather(-1, j0)[..., 0]
        tm1 = t.gather(-1, jm1)[..., 0]
        tm2 = t.gather(-1, jm2)[..., 0]

        w_j_2 = (inputs - t0) / (t1 - t0)
        w_j_3 = (inputs - t0) / (t2 - t0)  # (x-t_j)/(t_j+2 - t_j)
        w_jm1_3 = (inputs - tm1) / (t1 - tm1)

        B_jm2_3 = (1 - w_jm1_3) * (1 - w_j_2)
        B_jm1_3 = w_jm1_3 * (1 - w_j_2) + (1 - w_j_3) * w_j_2
        B_j_3 = w_j_3 * w_j_2
        D_jm2_3 = (km2 - km3) / (t1 - tm2)
        D_jm1_3 = (km1 - km2) / (t2 - tm1)
        D_j_3 = (km0 - km1) / (t3 - t0)

        absdet = 3 * (D_jm2_3 * B_jm2_3 + D_jm1_3 * B_jm1_3 + D_j_3 * B_j_3)
        logabsdet = torch.log(absdet)

        outputs = (km3 + (inputs - tm2) * D_jm2_3) * B_jm2_3 + (km2 + (inputs - tm1) * D_jm1_3) * B_jm1_3 + (
                    km1 + (inputs - t0) * D_j_3) * B_j_3
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)

    outputs = torch.as_tensor(outputs, dtype=torch.float)
    logabsdet = torch.as_tensor(logabsdet, dtype=torch.float)

    outputs_whole[inside_mask] = outputs
    logabsdet_whole[inside_mask] = logabsdet
    return outputs_whole, logabsdet_whole