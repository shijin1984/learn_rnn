# Learning RNN

## Overview
The goal is to learn about RNN, by building it from scratch.

Goals:
 * Build a RNN specific to MNIST.
 * Both training and inferencing

Non-goals:
 * Configurable. Just to start with a fixed network.
 * Performance. A single-threaded implementation is enough.
 * GPU.

## The network structure

### The MNIST in RNN

This example is using MNIST handwritten digits. The dataset contains 60,000
examples for training and 10,000 examples for testing. The digits have been
size-normalized and centered in a fixed-size image (28x28 pixels) with values
from 0 to 1. For simplicity, each image has been flattened and converted to a
1-D numpy array of 784 features 28x28.

To classify images using a recurrent neural network, we consider every image row
as a sequence of pixels. Because MNIST image shape is 28x28px, we will then
handle 28 sequences of 28 timesteps for every sample.

[source](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb)


### The RNN cell

Just to start with the simplest ones, for example, Elman network with ReLU:

```
h_t = ReLU(W_h x_t + U_h h_{t-1} + b_h)
y_t = ReLU(W_y h_t + b_y)
```

[source](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)
