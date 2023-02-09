#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_type='avg', pool_size=(2, 2)):

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'frac':
            fractional_maxpool2d = nn.FractionalMaxPool2d(kernel_size=pool_size, output_ratio=1 / np.sqrt(2))
            x = fractional_maxpool2d(x)

        return x



class Net(nn.Module):
    def __init__(self, pool_type='avg'):
        super(Net, self).__init__()

        self.pool_type = pool_type

        self.conv_block1 = ConvBlock(in_channels=16, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru_block = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(512, 32) # then concat one-hot vector
        self.fc2 = nn.Linear(43, 2)



    def forward(self, x, one_hot_tensor):
        '''input: (batch_size, channels, time_steps, mel_bins) e.g.(64, 15, 960, 64)'''

        x = self.conv_block1(x, self.pool_type)
        x = self.conv_block2(x, self.pool_type)
        x = self.conv_block3(x, self.pool_type)
        x = self.conv_block4(x, self.pool_type)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.transpose(1, 2)
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru_block(x)

        x = self.fc1(x)

        one_hot_tmp = one_hot_tensor
        for i in range(x.shape[1]-1):
            one_hot_tensor = torch.cat((one_hot_tensor, one_hot_tmp), dim= 1)

        x = torch.cat((x.float(), one_hot_tensor.float()), dim=2)
        output = torch.sigmoid(self.fc2(x))

        return output