# Mini deep-learning framework

**EPFL | [Deep Learning (EE559)](https://fleuret.org/ee559/) (Spring 2020) | Mini-project 2**  

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![Pytorch 1.13.1](https://img.shields.io/badge/pytorch-1.4.0-orange.svg)

## About
- This is our implementaion for the mini-project 2 in the Deep leaning course at EPFL.
  - **Team member**: Pengkang Guo, Xiaoyu Lin
- [[report](report.pdf)]

## Project Discription
The objective of this project is to design a mini "deep learning framework" using only pytorch's
tensor operations and the standard math library, hence in particular **without using autograd or the
neural-network modules**. 

## Requirements
Pytorch

## Run
From the root of the project: `python test.py`

## Description of the files
* module.py: the implementation of the modules
  * Includes `Linear`, `Relu`, `Tanh`, `LossMSE` and `Sequential`.
* test.py: the required basic Python script using our framework `module.py`.  <br>
  * Generates the dataset, initializes and trains the required model with three hidden layers of 25 units.
  * Generates an output file, `logs.out`, logging the loss and error rate of each epoch. Calculates and prints the average test error rate, average time and their standard deviations.
* test_plot.py: an upgraded version of `test.py`.   <br>
  * Has all the functions of `test.py`. <br>
  * Generates the images of the dataset, training error rate and test error rate.
* test_torch.py: an Pytorch version of `test_plot.py`.   <br>
  * has all the functions of `test_plot.py` but is implemented using Pytorch
