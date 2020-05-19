#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : modules.py
# @Description  :
import torch
import math

class Linear(object):
    """
    Fully connected layer
    """

    def __init__(self, in_dim, out_dim):
        self.parameters = [torch.empty(out_dim, in_dim), torch.empty(out_dim)]
        self.parameters[0].normal_(0, math.sqrt(2/out_dim))
        self.parameters[1].zero_()

    def forward(self, input):
        """

        :param input:
        :return:
        """
        self.input = input
        output = input.matmul(self.parameters[0].t()) + self.parameters[1]
        return output

    def backward(self, gradwrtoutput):
        """

        :param gradwrtoutput:
        :return:
        """
        self.gradwrtoutput = gradwrtoutput
        return gradwrtoutput.matmul(self.parameters[0])

    def grad(self):

        grad_wrt_weights = self.gradwrtoutput.t().matmul(self.input)
        grad_wrt_biases = self.gradwrtoutput.sum(0)
        return [grad_wrt_weights, grad_wrt_biases]

    def param(self):
        """

        :return:
        """
        return self.parameters


class Tanh(object):
    """
    Tanh function
    """

    def forward(self, input):
        self.input = input
        return torch.tanh(input)

    def backward(self, gradwrtoutput):
        return gradwrtoutput * (1 - (self.input.tanh()) ** 2)

    def param(self):
        return None


class ReLU(object):
    """
    ReLu function
    """

    def forward(self, input):
        self.input = input
        return input.relu()

    def backward(self, gradwrtoutput):
        grad = self.input
        grad[grad > 0.0] = 1.0
        grad[grad <= 0.0] = 0.0
        return gradwrtoutput * grad

    def param(self):
        return None


class Sequential(object):

    def __init__(self, *layers):

        self.model = []
        for layer in layers:
            self.model.extend(layer)

    def forward(self, input):
        output = input
        for layer in self.model:
            output = layer.forward(output)
        return output

    def backward(self, gradwrtoutput):
        output = gradwrtoutput
        for layer in reversed(self.model):
            output = layer.backward(output)

    def param(self):
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.param())
        return output

    def gard(self):
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.grad())
        return output

    def update(self, parameters):
        i = 0
        for layer in self.model:
            if layer.param() is not None:
                for t in range(len(layer.param())):
                    layer.parameters[t] = parameters[i]
                    i += 1


class LossMSE(object):
    """
    compute the MSE loss.
    """

    def forward(self, pred, label):
        self.pred = pred
        self.label = label
        return (pred - label).pow(2).sum(-1).mean()

    def backward(self):
        return 2 * (self.pred - self.label).mean(0)



