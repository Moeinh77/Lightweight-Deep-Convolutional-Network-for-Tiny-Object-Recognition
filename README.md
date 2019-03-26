# Lightweight Deep Convolutional Network for Tiny Object Recognition

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

DCTI has 5 phases of convolutional layers. We use all filters with receptive field 3x3 for all convolutional layers. All hidden layers are equipped with the rectification (ReLUs (Krizhevsky et al., 2017b)) non-linearity. **We use dropout and batch-normalization after each convolutional layer.**

Instead of using one convolutional layer with the kernel size 5 × 5, in this architecture two convolutional layers with kernel size 3 × 3 is used. Using two convolutional filers size 3 × 3 is equivalent to one convolutional filter size 5 × 5. By this way, **we reduce parameters and push network going deeper.**

In final phase, we use global average pooling layer to feed directly feature maps into feature vectors. From feature vectors, we apply fully connected and softmax to calculate probability each class.

Dataset is relatively small, we use a different way of dropout setting. We set dropout for convolution lay- ers, too. Specifically, we set dropout rate as 0.3 for the first and second group of convolution layers, 0.4 for the third and fourth group. The dropout rate for the feature vectors 512D layer was set to be 0.5.

### Dataset
[Link to dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Pictures are collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

### Results
![](http://www.mediafire.com/convkey/a87b/8a4acb8yobxq5mnzg.jpg)

### Dependencies
- Pytorch 0.3.0.post4

- Python 3.6

### Setting up enviroment
The easiest way to install the required dependencies is to use conda package manager.

- Install Anaconda with Python 3 - [Ananconda installation](https://docs.anaconda.com/anaconda/install/)

- Install pytorch 
