#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import *
from src.dataset import *
from src.char_tensor_cnn import CharTensorConvNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alphabet", type=str,
                        default="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
    parser.add_argument("-w", "--word_length", type=int, default=30)
    parser.add_argument("-r", "--char_length", type=int, default=10)
    parser.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-n", "--num_epochs", type=int, default=10)
    parser.add_argument("-l", "--lr", type=float, default=0.001)
    parser.add_argument("-c", "--n_conv_filters", type=int, default=50)
    parser.add_argument("-f", "--n_fc_neurons", type=int, default=256)
    parser.add_argument("-i", "--input", type=str, default="data", help="path to input folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    args = parser.parse_args()

    return args


def train(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    outfile = open(os.path.join(args.output, "logs.txt"), "w")
    outfile.write("Model's parameters: {}".format(vars(args)))

    # Prepare the training and testing data
    training_set = CharTensorDataset(os.path.join(args.input, "train.csv"),
                                     os.path.join(args.input, "classes.txt"),
                                     args.alphabet, args.word_length, args.char_length)
    test_set = CharTensorDataset(os.path.join(args.input, "test.csv"),
                                 os.path.join(args.input, "classes.txt"),
                                 args.alphabet, args.word_length, args.char_length)

    training_generator = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_generator = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    data, label = training_set[0]

    # Set the model
    model = CharTensorConvNet(n_classes=training_set.num_classes, input_dim=len(args.alphabet),
                              word_length=args.word_length, char_length=args.char_length,
                              n_conv_filters=args.n_conv_filters, n_fc_neurons=args.n_fc_neurons)
    model = model.to(device)

    criterion = nn.NLLLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    model.train()
    num_iter_per_epoch = len(training_generator)
    best_accuracy = 0

    for epoch in range(args.num_epochs):
        for itr, batch in enumerate(training_generator):
            _, n_true_label = batch         # numpy format
            batch = [record.to(device) for record in batch]
            t_data, t_true_label = batch    # tensor format

            optimizer.zero_grad()
            t_predicted_prob = model(t_data)
            loss = criterion(t_predicted_prob, t_true_label)
            n_predicted_prob = t_predicted_prob.cpu().data.numpy()  # numpy format
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(n_true_label, n_predicted_prob, list_metrics=["acc", "loss"])
            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {} Accuracy: {}".format(
                itr + 1, num_iter_per_epoch, epoch + 1, args.num_epochs,
                training_metrics["loss"], training_metrics["acc"]))

        model.eval()
        test_true = []
        test_prob = []
        for batch in test_generator:
            _, n_true_label = batch         # numpy format
            batch = [record.to(device) for record in batch]
            t_data, _ = batch               # tensor format

            t_predicted_prob = model(t_data)
            test_prob.append(t_predicted_prob)
            test_true.extend(n_true_label)

        test_prob = torch.cat(test_prob, 0).cpu().data.numpy()      # numpy format
        test_true = np.array(test_true)
        model.train()

        test_metrics = get_evaluation(test_true, test_prob, list_metrics=["acc", "loss", "confusion_matrix"])
        outfile.write(
            "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, args.num_epochs,
                training_metrics["loss"],
                training_metrics["acc"],
                test_metrics["loss"],
                test_metrics["acc"],
                test_metrics["confusion_matrix"]))
        print ("\tTest:Epoch: {}/{} Loss: {} Accuracy: {}\r".format(epoch + 1, args.num_epochs,
                                                                    test_metrics["loss"], test_metrics["acc"]))
        if test_metrics["acc"] > best_accuracy:
            best_accuracy = test_metrics["acc"]
            torch.save(model, os.path.join(args.output, "char_tensor_trained_model"))


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
