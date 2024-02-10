# MNIST-Handwritten-Digit-Recognition-Using-Neural-Network-from-Scratch

# Handwritten Digit Recognition using Neural Networks

## Overview

Handwritten digit recognition is a classic problem in the field of machine learning and computer vision. It involves the task of classifying images of handwritten digits into their respective numerical values. One of the most popular datasets used for this task is the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).

## Neural Network Approach

Neural networks have been widely used for handwritten digit recognition due to their ability to learn complex patterns from data. In this project, a feedforward neural network is trained to classify handwritten digits using the MNIST dataset. The neural network architecture consists of an input layer, one or more hidden layers, and an output layer. Each neuron in the network applies a linear transformation followed by a non-linear activation function, such as the Rectified Linear Unit (ReLU) and softmax.

Our NN will have a simple two-layer architecture. Input layer $a^{[0]}$ will have 784 units corresponding to the 784 pixels in each 28x28 input image. A hidden layer $a^{[1]}$ will have 10 units with ReLU activation, and finally our output layer $a^{[2]}$ will have 10 units corresponding to the ten digit classes with softmax activation.

**Forward propagation**

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$
$$A^{[1]} = g_{\text{ReLU}}(Z^{[1]}))$$
$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$
$$A^{[2]} = g_{\text{softmax}}(Z^{[2]})$$

**Backward propagation**

$$dZ^{[2]} = A^{[2]} - Y$$
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$$
$$dB^{[2]} = \frac{1}{m} \Sigma {dZ^{[2]}}$$
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$$
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$$
$$dB^{[1]} = \frac{1}{m} \Sigma {dZ^{[1]}}$$

**Parameter updates**

$$W^{[2]} := W^{[2]} - \alpha dW^{[2]}$$
$$b^{[2]} := b^{[2]} - \alpha db^{[2]}$$
$$W^{[1]} := W^{[1]} - \alpha dW^{[1]}$$
$$b^{[1]} := b^{[1]} - \alpha db^{[1]}$$

**Vars and shapes**

Forward prop

- $A^{[0]} = X$: 784 x m
- $Z^{[1]} \sim A^{[1]}$: 10 x m
- $W^{[1]}$: 10 x 784 (as $W^{[1]} A^{[0]} \sim Z^{[1]}$)
- $B^{[1]}$: 10 x 1
- $Z^{[2]} \sim A^{[2]}$: 10 x m
- $W^{[1]}$: 10 x 10 (as $W^{[2]} A^{[1]} \sim Z^{[2]}$)
- $B^{[2]}$: 10 x 1

Backprop

- $dZ^{[2]}$: 10 x m ($~A^{[2]}$)
- $dW^{[2]}$: 10 x 10
- $dB^{[2]}$: 10 x 1
- $dZ^{[1]}$: 10 x m ($~A^{[1]}$)
- $dW^{[1]}$: 10 x 10
- $dB^{[1]}$: 10 x 1

### Training Process

The training process involves several steps:

1. **Data Preprocessing:** The dataset is preprocessed by normalizing the pixel values to a range between 0 and 1.

2. **Initialization:** The parameters of the neural network, including weights and biases, are initialized randomly.

3. **Forward Propagation:** The input data is fed forward through the network, computing the output of each neuron using the current parameters.

4. **Prediction:** The output layer applies a softmax function to produce a probability distribution over the possible classes.

5. **Backward Propagation:** The gradients of the parameters are computed using backpropagation and the chain rule.

6. **Parameter Update:** The parameters are updated using gradient descent to minimize a loss function, such as cross-entropy.

7. **Iteration:** Steps 3-6 are repeated for multiple iterations or epochs until convergence.

### Evaluation

After training the neural network, it is evaluated on a separate testing dataset to assess its performance. The accuracy of the model is computed by comparing the predicted labels with the ground truth labels.

### Conclusion

Handwritten digit recognition using neural networks demonstrates the power of machine learning techniques in solving real-world problems. With advances in neural network architectures and optimization algorithms, state-of-the-art models have achieved high accuracy rates on the MNIST dataset and beyond, paving the way for applications in digit recognition, optical character recognition (OCR), and more.
