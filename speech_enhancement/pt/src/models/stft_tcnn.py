# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
'''STFT-TCNN model : TCNN-type denoising network without the convolutional encoder and decoders
, using STFT preprocessing instead.'''
import torch.nn as nn
from collections import OrderedDict


###########################################################
# TCNN blocks and layers, taken straight from the TCNN repo
# https://github.com/LXP-Never/TCNN
###########################################################

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                  causal=False, layer_activation="relu"):
        super(DepthwiseSeparableConv, self).__init__()
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = dilation

        if layer_activation == "prelu":
            act = nn.PReLU()
        elif layer_activation == "relu":
            act = nn.ReLU()

            
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=False)

        pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if causal:
            self.net = nn.Sequential(depthwise_conv,
                                     Chomp1d(padding),
                                     nn.BatchNorm1d(in_channels),
                                     act,
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv,
                                     nn.BatchNorm1d(in_channels),
                                     act,
                                     pointwise_conv)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, layer_activation="relu"):
        super(ResBlock, self).__init__()
        if layer_activation == "prelu":
            act = nn.PReLU(num_parameters=1)
        elif layer_activation == "relu":
            act = nn.ReLU()

        self.TCM_net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            act,
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=1, dilation=dilation, causal=False, layer_activation=layer_activation)
        )

    def forward(self, input):
        x = self.TCM_net(input)
        return x + input


class TCNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_dilation=3, num_layers=6,
                 layer_activation="relu"):
        super(TCNN_Block, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i

            layers += [ResBlock(in_channels, out_channels,
                                kernel_size, dilation=dilation_size,
                                layer_activation=layer_activation)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class STFTTCNN(nn.Module):
    '''
    STFT-TCNN model : TCNN-type denoising network without the convolutional encoder and decoders
    , using STFT preprocessing instead.
    Takes magnitude spectrogram columns as input, and outputs columns of a mask to be applied 
    to the complex spectrogram.
    Input shape is (batch, in_channels, sequence_length)
    Output shape is (batch, in_channels, sequence_length)
    '''
    def __init__(self,
                 in_channels=257,
                 tcn_latent_dim=512,
                 n_blocks=3,
                 kernel_size=3,
                 num_layers=6,
                 mask_activation="tanh",
                 layer_activation="relu",
                 init_dilation=2,
                 **kwargs
                 ):
        '''
        Parameters
        ----------
        in_channels, int : Number of channels in input and output tensors. Should be n_fft // 2 + 1
        tcn_latent_dim, int : Latent dimension of the temporal convolutional blocks
        n_blocks, int : Number of temporal convolutional blocks
        n_layers, int : Number of layers per temporal convolutional block
        mask_activation, str : One of "tanh", "sigmoid". Final activation applied to model output.
            "tanh" means output coeffs are in [-1, 1], and "sigmoid" means they are in [0, 1]
        NOTE : Additional kwargs are discarded, this is for convenience to allow user to pass
        invalid model-specific kwargs without raising an exceptions
        '''
        super().__init__()
        self.in_channels = in_channels
        self.tcn_latent_dim = tcn_latent_dim
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.mask_activation = mask_activation
        self.layer_activation = layer_activation
        self.init_dilation = init_dilation

        self.tcn_block_dict = OrderedDict()
        for k in range(n_blocks):
            self.tcn_block_dict[f"tcn_block_{k}"] = TCNN_Block(
                           in_channels=self.in_channels,
                           out_channels=self.tcn_latent_dim,
                           kernel_size=self.kernel_size,
                           init_dilation=self.init_dilation,
                           num_layers=self.num_layers,
                           layer_activation=self.layer_activation)
            
        self.tcn = nn.Sequential(self.tcn_block_dict)
        if self.mask_activation == "tanh":
            self.activation = nn.Tanh()
        elif self.mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("mask_activation must be one of ['tanh', 'sigmoid']")
        
    def forward(self, input):
        # No encoder or decoder 
        tcn_out = self.tcn(input)
        out = self.activation(tcn_out)
        return out