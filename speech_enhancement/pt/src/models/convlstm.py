# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

""" Implementation of Conv + LSTM denoiser model"""
import torch.nn as nn
from einops import rearrange

class ConvLSTMDenoiser(nn.Module):
    '''
    LSTM-based recurrent denoiser model.
       Consists of a 1D pointwise convolutional encoder, a block of consecutive LSTM layers,
       and a 1D pointwise convolutional decoder.
       Takes magnitude spectrogram columns as input, and outputs a mask that should be applied
       to the complex spectrogram.
       Input shape : (batch, in_channels, sequence_length)
       Output shape : (batch, out_channels, sequence_length)
       NOTE : Additional kwargs are discarded, this is for convenience to allow user to pass
       invalid model-specific kwargs without raising an exception
    '''
    def __init__(self, in_channels=257, out_channels=257, lstm_hidden_size=256, num_lstm_layers=2, mask_activation="sigmoid", **kwargs):
        '''
        Parameters
        ----------
        in_channels, int : Number of input channels, should correspond to n_fft // 2 + 1
            e.g., if n_fft = 512, in_channels = 257
            Input shape of the model is : (batch, in_channels, sequence_length)
        out_channels, int : Number of output channels, should correspond to n_fft // 2 + 1
            e.g., if n_fft = 512, out_channels = 257
            Output shape of the model is : (batch, out_channels, sequence_length)
        lstm_hidden_size, int : Number of hidden units in LSTM layers
            Corresponds to the hidden_size parameter of torch.nn.LSTM
        num_lstm_layers, int : Number of consecutive LSTM layers to include in the middle LSTM block.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = lstm_hidden_size
        self.lstm_layers = num_lstm_layers
        self.conv_block_1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=1),
                                          nn.BatchNorm1d(num_features=self.out_channels),
                                          nn.ReLU()
                                          )
        self.lstm_block = nn.LSTM(input_size=out_channels,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.lstm_layers,
                                  batch_first=True)
        self.conv_block_2 = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                                     out_channels=self.out_channels,
                                                     kernel_size=1),
                                          nn.BatchNorm1d(num_features=self.out_channels),
                                          nn.ReLU())
        self.conv_3 = nn.Conv1d(in_channels=self.out_channels,
                                out_channels=self.out_channels,
                                kernel_size=1)
        if mask_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif mask_activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Mask activation must be one of ['sigmoid', 'tanh'], but was {mask_activation}")


    def forward(self, x):
        # x should be of shape (batch, in_channels, seq_length) or in our case (batch, n_fft // 2 + 1, n_frames)
        conv_1_out = self.conv_block_1(x)
        lstm_in = rearrange(conv_1_out, "b c l -> b l c")
        lstm_out, _ = self.lstm_block(lstm_in)
        conv_2_in = rearrange(lstm_out, "b l c -> b c l")
        conv_2_out = self.conv_block_2(conv_2_in)
        conv_3_out = self.conv_3(conv_2_out)
        output = self.activation(conv_3_out)

        return output