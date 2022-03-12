#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, filters, padding=1, kernel_size=5):
        super(CNN, self).__init__()

        self.e_char = e_char
        self.filters = filters
        self.padding = padding
        self.k = kernel_size

        self.conv1d = None
        self.maxpool = None

        self.conv1d = nn.Conv1d(
            in_channels=self.e_char,
            out_channels=self.filters,
            kernel_size=self.k,
            stride=1,
            padding=self.padding,
            padding_mode='zeros',
            bias=True
        )


    def forward(self, xemb: torch.Tensor):
        # xemb #shape (batch, m_word, e_char)
        # xreshape #shape (batch, e_char, m_word)
        # W #shape (filter, e_char, k)

        m_word = xemb.shape[1]
        x_reshaped = xemb.permute(0, 2, 1)
        x_conv = self.conv1d(x_reshaped)
        x_conv = F.relu(x_conv)

        maxpool = nn.MaxPool1d(kernel_size=m_word+2*self.padding-self.k+1)
        x_conv_out = maxpool(x_conv).squeeze(2)
        # print('in CNN x_conv_out', x_conv_out.shape)

        return x_conv_out

    ### END YOUR CODE

