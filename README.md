# Lightweight-Deep-Convolutional-Network-for-Tiny-Object-Recognition

Implementation of paper :Lightweight Deep Convolutional Network for Tiny Object Recognition

### Introduction

In this paper,a Lightweight Deep Convolutional Neural Network architecture is proposed for tiny images codenamed “DCTI” to reduce significantly a number of parameters for such datasets. Additionally, we use batch-normalization to deal with
the change in distribution each layer. To demonstrate the efficiency of the proposed method, we conduct exper-
iments on two popular datasets: CIFAR-10 and CIFAR-100. The results show that the proposed network not
only significantly reduces the number of parameters but also improves the performance. The number of pa-
rameters in our method is only 21.33% the number of parameters of Wide ResNet but our method achieves up
to 94.34% accuracy on CIFAR-10, comparing to 96.11% of Wide ResNet. Besides, our method also achieves
the accuracy of 73.65% on CIFAR-100.


### Net Architecture

![](http://www.mediafire.com/convkey/8abb/6q2a2edfrx0c4a6zg.jpg)


### Dataset

[Link to dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Pictures are collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.


### Dependencies

- Pytorch 0.3.0.post4

- Python 3.6

### Setting up enviroment
------------------------
The easiest way to install the required dependencies is to use conda package manager.

- Install Anaconda with Python 3 - [Ananconda installation](https://docs.anaconda.com/anaconda/install/)

- Install pytorch 
