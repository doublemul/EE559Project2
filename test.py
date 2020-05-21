#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : test.py

from modules import *


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
    radius 1/sqrt(2*pi) and 1 inside.
    :param nb: number of samples
    :return: input, target
    """
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = torch.LongTensor([1 if (i - 0.5).pow(2).sum().item() < 1.0 / (2.0 * math.pi) else 0 for i in input])
    return input, target


def train_model(sample_num, batch_size, epoch_num, lr, model, train_input, train_target, test_input, test_target, logs):
    """
    Train the MLP model, log and display results
    :param sample_num: sampler number
    :param batch_size: batch size
    :param epoch_num: epoch number
    :param lr: learning rate
    :param model: model need to be train
    :param train_input: train set input data 1000x2
    :param train_target: train set target data 1000
    :param test_input: test set input data 1000x2
    :param test_target: test set target data 1000
    :param logs: logs.txt file to record results
    """
    criterion = LossMSE()

    for epoch in range(1, epoch_num + 1):

        for batch_input, batch_target in zip(train_input.split(batch_size),
                                             train_target.split(batch_size)):
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
                update_param.append(p - lr * g)
            model.update(update_param)

        # record train loss
        labels = torch.ones(train_input.size(0), 2) * -1
        labels.scatter_(1, train_target.unsqueeze(1), 1)
        pred = model.forward(train_input)
        loss = criterion.forward(pred, labels).item()
        print('epoch %d: train loss = %.6f,' % (epoch, loss), end=' ')
        logs.write('epoch %d: train loss = %.6f, ' % (epoch, loss))

        # record test loss
        labels = torch.ones(test_input.size(0), 2) * -1
        labels.scatter_(1, test_target.unsqueeze(1), 1)
        pred = model.forward(test_input)
        loss = criterion.forward(pred, labels).item()
        print('test loss = %.6f,' % loss, end=' ')
        logs.write('test loss = %.6f, ' % loss)

        # record train error rate
        nb_train_errors = compute_nb_errors(batch_size, model, train_input, train_target)
        print('train error rate = %.2f%%, ' % (100.0 * nb_train_errors / sample_num), end=' ')
        logs.write('train error rate = %.2f%%, ' % (100.0 * nb_train_errors / sample_num))

        # record test error rate
        nb_test_errors = compute_nb_errors(batch_size, model, test_input, test_target)
        print('test error rate = %.2f%%.\n' % (100.0 * nb_test_errors / sample_num), end='')
        logs.write('test error rate = %.2f%%.\n' % (100.0 * nb_test_errors / sample_num))


def compute_nb_errors(batch_size, model, test_input, test_target):
    """
    compute the number of errors of given model on given dataset
    :param batch_size: test batch size
    :param model: a model
    :param test_input: input data of dataset
    :param test_target: target data of dataset
    :return: number of error
    """
    nb_data_errors = 0
    for batch_input, batch_target in zip(test_input.split(batch_size),
                                         test_target.split(batch_size)):
        output = model.forward(batch_input)
        _, predicted_classes = torch.max(output, 1)
        for k in range(len(predicted_classes)):
            if batch_target[k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


if __name__ == '__main__':

    # Set Hyper Parameters #
    sample_num = 1000
    batch_size = 10
    epoch_num = 200
    lr = 1e-3

    # Prepare settings #
    # auto-grad globally off
    torch.set_grad_enabled(False)
    # logging results
    logs = open('logs.txt', mode='w')

    # Prepare data #
    # load data
    train_input, train_target = generate_disc_set(sample_num)
    test_input, test_target = generate_disc_set(sample_num)
    # normalize data
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Train model #
    model = create_model()
    train_model(sample_num, batch_size, epoch_num, lr, model, train_input, train_target, test_input, test_target, logs)

    logs.close()
    print('Done.')
