#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


class CharLevelConvNet(nn.Module):
    def __init__(self, input_length=285, n_classes=3, input_dim=68, n_conv_filters=256, n_fc_neurons=1024):
        super(CharLevelConvNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7),
                                   nn.ReLU(), nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7),
                                   nn.ReLU(), nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3),
                                   nn.ReLU(), nn.MaxPool1d(3))

        dimension = int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        # output = self.log_softmax(output)
        return output
