#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : modules.py
# @Description  :
import torch


class Linear(object):
    """
    Fully connected layer
    """

    def __init__(self, in_dim, out_dim, epsilon):
        self.parameters = [torch.empty(out_dim, in_dim), torch.empty(out_dim)]
        self.parameters[0].normal_(0, epsilon)
        self.parameters[1].normal_(0, epsilon)

    def forward(self, input):
        """

        :param input:
        :return:
        """
        self.input = input
        output = input.matmul(self.parameters[0].t())
        output += self.parameters[1]
        return output

    def backward(self, gradwrtoutput):
        """

        :param gradwrtoutput:
        :return:
        """
        self.gradwrtoutput = gradwrtoutput
        return self.parameters[0].t().mv(gradwrtoutput)

    def grad(self):

        grad_wrt_weight = self.gradwrtoutput.view(-1, 1).mm(self.input.mean(0).view(1, -1))
        grad_wrt_bias = self.gradwrtoutput
        return [grad_wrt_weight, grad_wrt_bias]

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
        return input.tanh()

    def backward(self, gradwrtoutput):
        grad = 4 * (self.input.exp() + self.input.mul(-1).exp()).pow(-2).mean(0)
        return grad * gradwrtoutput

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
        grad = torch.where(self.input > 0, torch.ones_like(self.input), torch.zeros_like(self.input)).mean(0)
        return grad * gradwrtoutput

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
        return (pred - label).pow(2).mean(0).sum()

    def backward(self):
        return 2 * (self.pred - self.label).mean(0)

    def param(self):
        return None


