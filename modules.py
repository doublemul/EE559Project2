#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : DeepLearningProject2
# @Author       : Xiaoyu LIN
# @File         : modules.py
# @Description  :


class Linear(object):
    """
    Fully connected layer
    """

    def forward(self, *input):
        """

        :param input:
        :return:
        """
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        """

        :param gradwrtoutput:
        :return:
        """
        raise NotImplementedError

    def param(self):
        """

        :return:
        """
        return []


class Tanh(object):
    """
    Tanh function
    """

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(object):
    """
    ReLu function
    """

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class LossMSE(object):
    """
    compute the MSE loss.
    """

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []