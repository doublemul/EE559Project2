#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : test.py
# @Description  :

import argparse
import torch
import math
import numpy as np
from modules import *


def create_model():
    return Sequential((Linear(2, 25),
                       Tanh(),
                       Linear(25, 25),
                       ReLU(),
                       Linear(25, 2)))


def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = torch.LongTensor([[1] if i.pow(2).sum().item() < 1.0 / (2.0 * math.pi) else [0] for i in input])
    return input, target


def train_model(args, model, train_input, train_target, logs):

    criterion = LossMSE()
    for epoch in range(1, args.epoch_num + 1):

        for batch_input, batch_target in zip(train_input.split(args.batch_size),
                                             train_target.split(args.batch_size)):
            pred = model.forward(batch_input)
            one_hot = torch.zeros(batch_input.size(0), 2).scatter_(1, batch_target, 1)
            loss = criterion.forward(pred, one_hot)
            model.backward(criterion.backward())
            param = model.param()
            grad = model.gard()
            update_param = []
            for p, g in zip(param, grad):
                update_param.append(p - args.lr * g)
            model.update(update_param)

        # record loss
        pred = model.forward(train_input)
        train_loss = criterion.forward(pred, train_target).item()
        print('epoch %d: loss = %.6f.' % (epoch, train_loss))
        # logs.write('epoch %d: loss = %.6f.\n' % (epoch, train_loss))


def compute_nb_errors(args, model, test_input, test_target):
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
    parser.add_argument('--batch_size', default=100, type=int)  # train mini-batch size
    parser.add_argument('--epoch_num', default=100, type=int)  # train epoch number
    parser.add_argument('--lr', default=1e-4, type=float)  # learning rate
    parser.add_argument('--rounds_num', default=5, type=int)  # round number
    args = parser.parse_args()

    # Prepare settings #
    # auto-grad globally off
    torch.set_grad_enabled(False)
    # logging results
    logs = open('logs.txt', mode='a')

    # Prepare data #
    # load data
    train_input, train_target = generate_disc_set(args.sample_num)
    test_input, test_target = generate_disc_set(args.sample_num)
    # normalize data
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)


    model = create_model()
    train_model(args, model, train_input, train_target, logs)
    nb_errors = compute_nb_errors(args, model, test_input, test_target)
    print('test error rate: %.2f%%' % (100 * nb_errors / args.sample_num))

    # # Train model #
    # nbs_errors = []
    # for r in range(1, args.rounds_num + 1):
    #     logs.write('Round %d:\n' % r)
    #     model = create_model()
    #     train_model(args, model, train_input, train_target, logs)
    #     nb_errors = compute_nb_errors(args, model, test_input, test_target)
    #     nbs_errors.append(nb_errors)
    #     del model
    #
    # # Record results #
    # print(nbs_errors)
    # error_rates = np.array(nbs_errors) / args.sample_num
    # info = 'Average test error rate: %.2f%%, standard deviation: %.4e.' \
    #        % (100 * (error_rates.mean()), error_rates.std())
    # print(info)
    # logs.write('%s\n\n' % info)

    print('Done.')
