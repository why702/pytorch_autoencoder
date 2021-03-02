# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Discriminator class part of the GAN.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, num_gpu, latent_vector_size, ndf, num_channels):
        super(Discriminator, self).__init__()
        self.ngpu = num_gpu
        self.main = nn.Sequential(

            # input is (num_channels) x 216 x 216 (height goes first, when specifying tuples)
            spectral_norm(nn.Conv2d(num_channels, ndf, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # When dilation and padding is 1: ((in + 2p - (k - 1) - 1) / s) + 1

            # state: (ndf*2) x 108 x 108
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # state: (ndf*4) x 54 x 54
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True),
            # state:  (ndf*4) x 27 x 27
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=3, padding=0)),
            nn.LeakyReLU(0.1, inplace=True),
            # state:  (ndf*8) x 9 x 9
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=3, stride=3, padding=0)),
            nn.LeakyReLU(0.1, inplace=True),
            # state:  (ndf*8) x 3 x 3
            spectral_norm(nn.Conv2d(ndf * 16, latent_vector_size, kernel_size=3, stride=1)),
            # Output is 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, discriminator_input):
        return self.main(discriminator_input)
