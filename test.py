#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : test.py

import argparse
import matplotlib.pyplot as plt
import torch
import math
import time
import numpy as np
from modules import *


def str2bool(v):
    """
    Convert string to boolean
    :param v: string
    :return: boolean True or False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_model():
    """
    Create a MLP model with two input units, two output units, three hidden layers of 25 units.
    :return: neural network
    """
    return Sequential((Linear(2, 25),
                       ReLU(),
                       Linear(25, 25),
                       ReLU(),
                       Linear(25, 25),
                       ReLU(),
                       Linear(25, 2),
                       Tanh()))


def generate_disc_set(nb):
    """
    Generates data set of nb points sampled uniformly in [0,1]^2, each with a label 0 if outside the disk of
    radius 1/sqrt(2*pi) and 1 inside,
    :param nb: number of samples
    :return: input, target
    """
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = torch.LongTensor([1 if (i - 0.5).pow(2).sum().item() < 1.0 / (2.0 * math.pi) else 0 for i in input])
    return input, target


def train_model(args, model, train_input, train_target, test_input, test_target, logs, plot):
    """
    Train the MLP model, log and display results
    :param args: experiment setup parameter
    :param model: model need to be train
    :param train_input: train set input data 1000x2
    :param train_target: train set target data 1000
    :param test_input: test set input data 1000x2
    :param test_target: test set target data 1000
    :param logs: logs.txt file to record results
    :param plot: a boolean, if true, record and display loss, error rate v.s. epoch
    :return: number of errors of final model on test set
    """
    criterion = LossMSE()

    train_error = []
    train_loss = []
    test_error = []
    test_loss = []
    for epoch in range(1, args.epoch_num + 1):

        for batch_input, batch_target in zip(train_input.split(args.batch_size),
                                             train_target.split(args.batch_size)):
            pred = model.forward(batch_input)
            labels = torch.ones(batch_input.size(0), 2) * -1
            labels.scatter_(1, batch_target.unsqueeze(1), 1)

            # mini-batch SGD
            criterion.forward(pred, labels)
            model.backward(criterion.backward())
            param = model.param()
            grad = model.gard()
            update_param = []
            for p, g in zip(param, grad):
                update_param.append(p - args.lr * g)
            model.update(update_param)

        if plot:
            # record train loss
            labels = torch.ones(train_input.size(0), 2) * -1
            labels.scatter_(1, train_target.unsqueeze(1), 1)
            pred = model.forward(train_input)
            loss = criterion.forward(pred, labels).item()
            print('epoch %d: train loss = %.6f,' % (epoch, loss), end=' ')
            logs.write('epoch %d: train loss = %.6f, ' % (epoch, loss))
            train_loss.append(loss)

            # record test loss
            labels = torch.ones(test_input.size(0), 2) * -1
            labels.scatter_(1, test_target.unsqueeze(1), 1)
            pred = model.forward(test_input)
            loss = criterion.forward(pred, labels).item()
            print('test loss = %.6f.' % loss)
            logs.write('test loss = %.6f, ' % loss)
            test_loss.append(loss)

            # record train error rate
            nb_train_errors = compute_nb_errors(args, model, train_input, train_target)
            logs.write('train error rate = %.4f%%, ' % (100.0 * nb_train_errors / args.sample_num))
            train_error.append(nb_train_errors)

            # record test error rate
            nb_test_errors = compute_nb_errors(args, model, test_input, test_target)
            logs.write('test error rate = %.4f%%.\n' % (100.0 * nb_test_errors / args.sample_num))
            test_error.append(nb_test_errors)

    if plot:
        # plot loss figure
        plt.figure()
        plt.plot(range(1, args.epoch_num+1), train_loss, label='Train loss', color='r')
        plt.plot(range(1, args.epoch_num+1), test_loss, label='Test loss', color='b')
        plt.title('Train and test loss v.s. epoch')
        plt.xlabel('epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss.pdf')
        plt.show()

        # plot error rate figure
        plt.figure()
        train_error = (np.array(train_error) / args.sample_num) * 100.0
        test_error = (np.array(test_error) / args.sample_num) * 100.0
        plt.plot(range(1, args.epoch_num + 1), train_error, label='Train error rate(%)', color='r')
        plt.plot(range(1, args.epoch_num + 1), test_error, label='Test error rate(%)', color='b')
        plt.title('Train and test error rate(%) v.s. epoch')
        plt.xlabel('epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig('error_rate.pdf')
        plt.show()

    return compute_nb_errors(args, model, test_input, test_target)


def compute_nb_errors(args, model, test_input, test_target):
    """
    compute the number of errors of given model on given dataset
    :param args: experiment setup parameters
    :param model: a model
    :param test_input: input data of dataset
    :param test_target: target data of dataset
    :return: number of error
    """
    nb_data_errors = 0
    for batch_input, batch_target in zip(test_input.split(args.batch_size),
                                         test_target.split(args.batch_size)):
        output = model.forward(batch_input)
        _, predicted_classes = torch.max(output, 1)
        for k in range(len(predicted_classes)):
            if batch_target[k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_num', default=1000, type=int)  # train and test set sample number
    parser.add_argument('--batch_size', default=10, type=int)  # train mini-batch size
    parser.add_argument('--epoch_num', default=200, type=int)  # train epoch number
    parser.add_argument('--lr', default=1e-3, type=float)  # learning rate
    parser.add_argument('--round_num', default=1, type=int)  # learning rate
    parser.add_argument('--plot', default=True, type=str2bool)
    args = parser.parse_args()

    # Prepare settings #
    # auto-grad globally off
    torch.set_grad_enabled(False)
    # logging results
    logs = open('logs.txt', mode='w')

    # Prepare data #
    # load data
    train_input, train_target = generate_disc_set(args.sample_num)
    test_input, test_target = generate_disc_set(args.sample_num)
    # plot data
    if args.plot:
        plt.figure(figsize=(8, 8))
        for d, t in zip(train_input, train_target):
            if t == 0:
                plt.scatter(d[0], d[1], color='r')
            else:
                plt.scatter(d[0], d[1], color='b')
        circle = plt.Circle((0.5, 0.5), 1.0/math.sqrt(2.0 * math.pi), color='g', alpha=0.5)
        plt.gcf().gca().add_artist(circle)
        plt.tight_layout()
        plt.savefig('dataset.pdf')
        plt.show()
    # normalize data
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Train model #
    nbs_errors, times = [], []
    plot = args.plot
    for r in range(1, args.round_num + 1):
        print('Round %d:' % r)
        model = create_model()
        t = time.time()
        nb_errors = train_model(args, model, train_input, train_target, test_input, test_target, logs, plot)
        times.append(time.time()-t)
        nbs_errors.append(nb_errors)
        del model
        plot = False

    # Record results #
    # record error rate
    error_rates = np.array(nbs_errors) / args.sample_num
    info = 'Average test error rate: %.2f%%, standard deviation: %.4e.' \
           % (100 * (error_rates.mean()), error_rates.std())
    print(info)
    logs.write('%s\n' % info)
    # record times
    times = np.array(times)
    info = 'Average time: %.2fs, standard deviation: %.4e.' \
           % (times.mean(), times.std())
    print(info)
    logs.write('%s\n\n' % info)

    print('Done.')
