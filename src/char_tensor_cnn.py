#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch.nn as nn


class CharTensorConvNet(nn.Module):
    def __init__(self, n_classes=3, input_dim=68, word_length=30, char_length=10,
                 n_conv_filters=50, n_fc_neurons=256):
        super(CharTensorConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(char_length, n_conv_filters, kernel_size=(input_dim, 3)),
                                   nn.ReLU())  # (N, 50, 28)
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=1),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=1),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=1),
                                   nn.ReLU(), nn.MaxPool1d(2))

        dimension = n_conv_filters * int((word_length - 2) / 2)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = output.squeeze()
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output
