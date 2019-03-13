import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, hidden_layer_size, i, o, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.hidden_layer_size = hidden_layer_size
        self.layers = []        
        for num_layer in hidden_layer_size:
            self.layers.append(FullyConnectedLayer(i,i))
            self.layers.append(ReLULayer())
        
        self.layers.append(FullyConnectedLayer(i,o))
        
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: use self.params()
        

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        for layer in self.layers:
            X = layer.forward(X)
        y_pred = np.argmax(X, axis=1)
        return y_pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        raise Exception("Not implemented!")

        return result
