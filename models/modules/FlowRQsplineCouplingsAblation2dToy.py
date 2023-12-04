import torch
from torch import nn as nn
from torch.nn import functional as F
import math

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get

NUM_BINS = 1

class CondRQsplineSeparatedAndCond2dToy(nn.Module):
    def __init__(self, in_channels, opt, std_mode=True, bias=True):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = 2
        self.kernel_hidden = 1
        # self.affine_eps = 0.0001
        self.n_hidden_layers = 4
        # self.std_channels = opt_get(opt, ['network_G', 'flow', 'std_channels'])
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)
        self.bias = bias
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fRQspline = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                               out_channels=self.channels_for_co * (3+self.bias),
                               # 2 for (shift, scale)
                               hidden_channels=self.hidden_channels,
                               kernel_hidden=self.kernel_hidden,
                               n_hidden_layers=self.n_hidden_layers)
        
        self.stdmode = std_mode
        self._left = -0.5
        self._right = 0.5
        self._top = 0.5
        self._bottom = -0.5

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None, std=None):
        assert self.bias # I didn't implement the case when self.bias==False
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

            # Self Conditional
            z1, z2 = self.split(z)
            # scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            width, height, derivative, shift = self.feature_extract_RQspline(z1, ft, self.fRQspline)
            # self.asserts(scale, shift, z1, z2)
            # z2 = z2 + shift
            # z2 = z2 * scale

            z2, logdet2 = rational_quadratic_spline(
                z2,
                width,
                height,
                derivative,
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
            z = input
            # Self Conditional
            z1, z2 = self.split(z)
            width, height, derivative, shift = self.feature_extract_RQspline(z1, ft, self.fRQspline)
            # self.asserts(scale, shift, z1, z2)
            # z2 = z2 / scale
            z2 = z2 - shift

            z2, logdet2 = rational_quadratic_spline(
                z2,
                width,
                height,
                derivative,
                inverse=True,
                left=self._left,
                right=self._right,
                top=self._top,
                bottom=self._bottom,
            )
            z = thops.cat_feature(z1, z2)
            logdet = logdet + thops.sum(logdet2, dim=[1, 2, 3])

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
        h = f(z)
        newh = h.view(h.shape[0], self.in_channels, -1, h.shape[2], h.shape[3])
        newh = torch.permute(newh, (0, 1, 3, 4, 2))
        width = newh[..., :NUM_BINS]
        height = newh[..., NUM_BINS:2*NUM_BINS]
        derivative = newh[..., 2*NUM_BINS:-1]
        shift = newh[..., -1]
        return width, height, derivative, shift

    def feature_extract_RQspline(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)  # B (C_co * (n1+n2+1)) H W
        newh = h.view(h.shape[0], self.channels_for_co, -1, h.shape[2], h.shape[3])
        newh = torch.permute(newh, (0, 1, 3, 4, 2))
        width = newh[..., :NUM_BINS]
        height = newh[..., NUM_BINS:2*NUM_BINS]
        derivative = newh[..., 2*NUM_BINS:-1]
        shift = newh[..., -1]
        return width, height, derivative, shift

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


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def searchsorted2(bin_locations, inputs):
    return (inputs >= bin_locations[..., 1]).to(torch.int8)


def rational_quadratic_spline(
    inputs_whole,
    width_whole,
    height_whole,
    derivative_whole,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=0.001,
    min_bin_height=0.001,
    min_derivative=0.001,
):
    """
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    """
    outputs_whole = torch.zeros_like(inputs_whole)
    logabsdet_whole = torch.zeros_like(inputs_whole)
    
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
    width = width_whole[inside_mask]
    height = height_whole[inside_mask]
    derivative = derivative_whole[inside_mask]
    """
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)
    """
    num_bins = 2
    
    width = torch.sigmoid(width)*(1-min_bin_width*2) + min_bin_width
    widths = torch.cat([width, 1-width], dim=-1)
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    
    derivatives = torch.exp(derivative)*(1-min_derivative) + min_derivative
    derivatives = F.pad(derivatives, pad=(1,1), mode="constant", value=1.)
    
    height = torch.sigmoid(height)*(1-min_bin_height*2) + min_bin_height
    heights = torch.cat([height, 1-height], dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    #print(cumwidths)
    #print(bin_idx)
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        outputs = torch.clamp(outputs, min=left, max=right)
        
        #outputs = outputs * (right - left) + left
        #logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)
        logabsdet = -logabsdet
        
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        outputs = torch.clamp(outputs,min=bottom,max=top)
        #outputs = outputs * (top - bottom) + bottom
        #logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)
        
    #outputs = torch.as_tensor(outputs, dtype=torch.float)
    #logabsdet = torch.as_tensor(logabsdet, dtype=torch.float)

    outputs_whole[inside_mask] = outputs
    logabsdet_whole[inside_mask] = logabsdet
    return outputs_whole, logabsdet_whole