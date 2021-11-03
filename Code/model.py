#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' This is the code of CNN model'

__author__ = 'Quan Wang'

from torch import nn
import torch
from torch.nn import functional as F

class DoubleConv(nn.Module):
    """ the convolutional blok: (convolution => [BN] => ReLU) * 2 """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """ down-sample block: maxpool => DoubleConv block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """up-sample block: ConvTrans => [BN] => Relu => concat"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size= 2, stride=2),
            nn.BatchNorm2d(in_channels//2),
            nn.LeakyReLU(inplace=True)
        )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet_MTL(nn.Module):
    """
    UNet of Multi Task Learning Model.
    """

    def __init__(self, input_channels=3, num_classes=3, filters=(32, 64, 128, 256)):
        super(UNet_MTL, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.inc = DoubleConv(input_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])
        self.up1 = Up(filters[3], filters[2])
        self.up2 = Up(filters[2], filters[1])
        self.up3 = Up(filters[1], filters[0])
        self.out = nn.Sequential(
            nn.Conv2d(filters[0], self.num_classes, kernel_size=1)
            )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        out = self.out(x)
        return out