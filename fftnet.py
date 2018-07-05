from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F

class FFTNetBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 shift,
                 local_condition_channels=None): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shift = shift
        self.local_condition_channels = local_condition_channels
        self.x_l_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.x_r_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        if local_condition_channels is not None:
            self.h_l_conv = nn.Conv1d(local_condition_channels, out_channels, kernel_size=1)
            self.h_r_conv = nn.Conv1d(local_condition_channels, out_channels, kernel_size=1)
        self.output_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, h=None):
        x_l = self.x_l_conv(x[:, :, :-self.shift]) 
        x_r = self.x_r_conv(x[:, :, self.shift:]) 
        if h is None:
            z = F.relu(x_l + x_r)
        else:
            h = h[:, :, -x.size(-1):]
            h_l = self.h_l_conv(h[:, :, :-self.shift])
            h_r = self.h_r_conv(h[:, :, self.shift:]) 
            z_x = x_l + x_r
            z_h = h_l + h_r
            z = F.relu(z_x + z_h)
        output = F.relu(self.output_conv(z))

        return output


class FFTNet(nn.Module):
    """Implements the FFTNet for vocoder

    Reference: FFTNet: a Real-Time Speaker-Dependent Neural Vocoder. ICASSP 2018

    Args:
        n_stacks: the number of stacked fft layer
        fft_channels:
        quantization_channels:
        local_condition_channels:
    """
    def __init__(self, 
                 n_stacks=11, 
                 fft_channels=256, 
                 quantization_channels=256, 
                 local_condition_channels=None):
        super().__init__()
        self.n_stacks = n_stacks
        self.fft_channels = fft_channels
        self.quantization_channels = quantization_channels
        self.local_condition_channels = local_condition_channels
        self.window_shifts = [2 ** i for i in range(self.n_stacks)]
        self.receptive_field = sum(self.window_shifts) + 1
        self.linear = nn.Linear(fft_channels, quantization_channels)
        self.layers = nn.ModuleList()

        for shift in reversed(self.window_shifts):
            if shift == self.window_shifts[-1]:
                in_channels = 1
            else:
                in_channels = fft_channels
            fftlayer = FFTNetBlock(in_channels, fft_channels,
                                   shift, local_condition_channels)
            self.layers.append(fftlayer)

    def forward(self, x, h):
        output = x.transpose(1, 2)
        for fft_layer in self.layers:
            output = fft_layer(output, h)
        output = self.linear(output.transpose(1, 2))
        return output.transpose(1, 2)

