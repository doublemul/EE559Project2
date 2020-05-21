#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : test.py
# @Description  :

import argparse
import matplotlib.pyplot as plt
import torch
import math
import time
import numpy as np
from modules import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(2, 25),
        torch.nn.ReLU(),
        torch.nn.Linear(25, 25),
        torch.nn.ReLU(),
        torch.nn.Linear(25, 25),
        torch.nn.ReLU(),
        torch.nn.Linear(25, 2),
        torch.nn.Tanh())


def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = torch.LongTensor([1 if (i - 0.5).pow(2).sum().item() < 1.0 / (2.0 * math.pi) else 0 for i in input])
    return input, target


def train_model(args, model, train_input, train_target, test_input, test_target, logs, plot):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    train_error = []
    train_loss = []
    test_error = []
    test_loss = []
    for epoch in range(1, args.epoch_num + 1):

        for batch_input, batch_target in zip(train_input.split(args.batch_size),
                                             train_target.split(args.batch_size)):
            pred = model(batch_input)
            labels = torch.ones(batch_input.size(0), 2) * -1
            labels.scatter_(1, batch_target.unsqueeze(1), 1)
            loss = criterion(pred, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if plot:
            # record train loss
            labels = torch.ones(train_input.size(0), 2) * -1
            labels.scatter_(1, train_target.unsqueeze(1), 1)
            pred = model.forward(train_input)
            loss = criterion.forward(pred, labels).item()
            print('epoch %d: train loss = %.6f.' % (epoch, loss))
            logs.write('epoch %d: train loss = %.6f. ' % (epoch, loss))
            train_loss.append(loss)

            # record test loss
            labels = torch.ones(test_input.size(0), 2) * -1
            labels.scatter_(1, test_target.unsqueeze(1), 1)
            pred = model.forward(test_input)
            loss = criterion.forward(pred, labels).item()
            print('epoch %d: loss = %.6f.' % (epoch, loss))
            logs.write('test loss = %.6f. ' % loss)
            test_loss.append(loss)

            # record train error rate
            nb_train_errors = compute_nb_errors(args, model, train_input, train_target)
            logs.write('train error rate = %.4f%%. ' % (100.0 * nb_train_errors / args.sample_num))
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

    nb_test_errors = compute_nb_errors(args, model, test_input, test_target)
    return nb_test_errors


def compute_nb_errors(args, model, test_input, test_target):
    nb_data_errors = 0

    for batch_input, batch_target in zip(test_input.split(args.batch_size),
                                         test_target.split(args.batch_size)):
        output = model(batch_input)
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
    parser.add_argument('--round_num', default=20, type=int)  # learning rate
    parser.add_argument('--plot', default=False, type=str2bool)
    args = parser.parse_args()

    # Prepare settings #
    # auto-grad globally off
    # torch.set_grad_enabled(False)
    # logging results
    logs = open('torch_logs.txt', mode='w')

    # Prepare data #
    # load data
    train_input, train_target = generate_disc_set(args.sample_num)
    test_input, test_target = generate_disc_set(args.sample_num)
    # plot data
    if args.plot:
        plt.figure(figsize=(10, 10))
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
        train_input, train_target = generate_disc_set(args.sample_num)
        test_input, test_target = generate_disc_set(args.sample_num)
        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)
        model = create_model()
        t = time.time()
        nb_errors = train_model(args, model, train_input, train_target, test_input, test_target, logs, plot)
        times.append(time.time()-t)
        nbs_errors.append(nb_errors)
        del model
        plot = False

    # Record results #
    error_rates = np.array(nbs_errors) / args.sample_num
    info = 'Average test error rate: %.2f%%, standard deviation: %.4e.' \
           % (100 * (error_rates.mean()), error_rates.std())
    print(info)
    logs.write('%s\n' % info)
    times = np.array(times)
    info = 'Average time: %.2fs, standard deviation: %.4e.' \
           % (times.mean(), times.std())
    print(info)
    logs.write('%s\n\n' % info)

    print('Done.')
