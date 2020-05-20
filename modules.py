#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : modules.py

import torch
import math


class Linear(object):
    """
    Fully connected layer: y = Wx + b
    """
    def __init__(self, in_dim, out_dim, bias=True):
        """
        Initialize liner layer:
        Weight parameter is initialized by Normal distribution N(0, sqrt(out_dim))
        bias parameter is initialized by zero vector
        :param in_dim: input dimension (n)
        :param out_dim: output dimension (m)
        :param bias: if True, add bias item
        """
        self.parameters = [torch.empty(out_dim, in_dim)]
        self.parameters[0].normal_(0, math.sqrt(2 / out_dim))
        if bias:
            self.parameters.append(torch.empty(out_dim))
            self.parameters[-1].zero_()

        self.bias = bias
        self.input = None
        self.gradwrtoutput = None

    def forward(self, input):
        """
        Forward path Y = XW + b, X(Nxn), W(nxm) b(m) , Y(Nxm)
        :param input: input data x
        :return: y = wx + b
        """
        self.input = input
        output = input.matmul(self.parameters[0].t())
        if self.bias:
            output += self.parameters[1]
        return output

    def backward(self, gradwrtoutput):
        """
        Backward path to calculate gradient
        :param gradwrtoutput: backward output from last layer, gradient w.r.t layer output
        :return: gradient
        """
        self.gradwrtoutput = gradwrtoutput
        return gradwrtoutput.matmul(self.parameters[0])

    def grad(self):
        """
        Compute gradient w.r.t each parameter, since we use mini-batch SGD, take the mean value
        :return: gradient w.r.t each parameter
        """
        grad_wrt_weights = self.gradwrtoutput.t().matmul(self.input) / self.input.size(0)
        if self.bias:
            grad_wrt_biases = self.gradwrtoutput.mean(0)
            return [grad_wrt_weights, grad_wrt_biases]
        else:
            return grad_wrt_weights

    def param(self):
        """
        :return: all parameter in this layer
        """
        return self.parameters


class Tanh(object):
    """
    Tanh activation function
    """
    def forward(self, input):
        """
        Forward path: apply Tanh() to each element in input data
        :param input: input data x
        :return: tanh(x) point-wise
        """
        self.input = input
        return torch.tanh(input)

    def backward(self, gradwrtoutput):
        """
        Backward path apply gradient of Tanh from input data on gradwrtoutput
        :param gradwrtoutput: output from last layer in backward path
        :return: apply gradient of Tanh from input data on gradwrtoutput
        """
        return gradwrtoutput * (1 - (self.input.tanh()) ** 2)

    def param(self):
        """
        No parameter in this function.
        """
        return None


class ReLU(object):
    """
    ReLU activation function
    """
    def forward(self, input):
        """
        Forward path: apply ReLU() to each element in input data
        :param input: input data x
        :return: ReLU(x) point-wise
        """
        self.input = input
        return input.relu()

    def backward(self, gradwrtoutput):
        """
        Backward path apply gradient of ReLU from input data on gradwrtoutput
        :param gradwrtoutput: output from last layer in backward path
        :return: apply gradient of ReLU from input data on gradwrtoutput
        """
        grad = self.input
        grad[grad > 0.0] = 1.0
        grad[grad <= 0.0] = 0.0
        return gradwrtoutput * grad

    def param(self):
        """
        No parameter in this function.
        """
        return None


class Sequential(object):

    def __init__(self, *layers):
        """
        Create model
        :param layers: define each layer in model in sequential
        """
        self.model = []
        for layer in layers:
            self.model.extend(layer)

    def forward(self, input):
        """
        Foword path: propagate input in forward path in each layer sequentially
        :param input: input data X
        :return: model output M(X)
        """
        output = input
        for layer in self.model:
            output = layer.forward(output)
        return output

    def backward(self, gradwrtoutput):
        output = gradwrtoutput
        for layer in reversed(self.model):
            output = layer.backward(output)

    def param(self):
        """
        Give all parameter in the model
        :return: a list of all parameter in model
        """
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.param())
        return output

    def gard(self):
        """
        Compute the gradient of final loss w.r.t each parameter
        :return: gradient w.r.t each parameter
        """
        output = []
        for layer in self.model:
            if layer.param() is not None:
                output.extend(layer.grad())
        return output

    def update(self, parameters):
        """
        Update the parameter in model
        :param parameters: new parameters
        """
        i = 0
        for layer in self.model:
            if layer.param() is not None:
                for t in range(len(layer.param())):
                    layer.parameters[t] = parameters[i]
                    i += 1

    def zero_grad(self):
        for layer in self.model:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()


class LossMSE(object):
    """
    Compute the Mean Square Error loss.
    """
    def forward(self, pred, label):
        """
        Compute the MSE= mean(L2norm(pred-label))
        :param pred: prediction given by model
        :param label: label converted by ground-truth target
        :return: MSE
        """
        self.pred = pred
        self.label = label
        return (pred - label).pow(2).sum(-1).mean()

    def backward(self):
        """
        Backward path gives gradient w.r.t model output
        :return: gradient w.r.t (Nx2)
        """
        return 2.0 * (self.pred - self.label)
