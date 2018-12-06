#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import csv
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path, class_path):
        self.data_path = data_path
        self.num_classes = sum(1 for _ in open(class_path, 'r'))
        self.texts, self.dates = [], []
        with open(data_path, 'r') as infile:
            for line in infile:
                words = line.split()
                date = words[0]
                text = " ".join(words[1:]).lower()
                self.dates.append(date)
                self.texts.append(text)
        self.length = len(self.dates)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pass


class CharDataset(MyDataset):
    def __init__(self, data_path, class_path, alphabet, max_length):
        super(CharDataset, self).__init__(data_path, class_path)
        self.max_length = max_length
        self.alphabet = list(alphabet)
        self.identity_mat = np.identity(len(self.alphabet))

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.alphabet.index(i)] for i in list(raw_text)
                         if i in self.alphabet], dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.alphabet)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.alphabet)), dtype=np.float32)
        date = self.dates[index]
        data = data.T       # shape (len(self.alphabet), max_length)
        return data, date


class CharTensorDataset(MyDataset):
    def __init__(self, data_path, class_path, alphabet, word_length, char_length):
        super(CharTensorDataset, self).__init__(data_path, class_path)
        self.alphabet = list(alphabet)
        self.identity_mat = np.identity(len(self.alphabet))
        self.word_length = word_length
        self.char_length = char_length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        words = raw_text.split()
        data = np.zeros((self.char_length, len(self.alphabet), self.word_length), dtype=np.float32)

        num_words = min(self.word_length, len(words))
        for i in range(num_words):
            word = words[i]
            num_chars = min(self.char_length, len(word))
            for j in range(num_chars):
                if word[j] in self.alphabet:
                    data[j, :, i] = self.identity_mat[self.alphabet.index(word[j])]
        date = self.dates[index]
        return data, date
