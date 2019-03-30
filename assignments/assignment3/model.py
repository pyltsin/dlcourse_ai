import numpy as np
import math

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, n_input, n_output, conv1_size, conv2_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        conv1_size, int - number of filters in the 1st conv layer
        conv2_size, int - number of filters in the 2nd conv layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = [
            ConvolutionalLayer(3, conv1_size, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_size, conv2_size, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(math.floor(n_input/16)**2*conv2_size,n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        X_next = X.copy()
        for layer in self.layers:
            X_next = layer.forward(X_next)
        
        loss, grad = softmax_with_cross_entropy(X_next, y)
#         print(grad)
#         loss_l2 = 0
#         for params in self.params():
#             w = self.params()[params]
#             loss_d, grad_d = l2_regularization(w.value, self.reg)
#             loss_l2+=0
        
#         loss+=loss_l2
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
#             grad_l2 = 0
            for params in layer.params():
                w = layer.params()[params]
#                 loss_d, grad_d = l2_regularization(w.value, self.reg)
#                 w.grad+=10
#             grad+=grad_l2
            
        return loss

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        y_pred = np.argmax(X, axis=1)
        return y_pred

    def params(self):
        result = {}

        for layer_num in range(len(self.layers)):
            for i in self.layers[layer_num].params():
                result[str(layer_num) + "_" + i] = self.layers[layer_num].params()[i]

        return result
