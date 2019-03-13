import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W*W)
    grad = 2 * W * reg_strength
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    f = predictions.copy()
    f -= np.max(f, axis=1).reshape([f.shape[0], 1]) # f becomes [-666, -333, 0]
    p = np.exp(f) / np.sum(np.exp(f), axis=1).reshape([f.shape[0], 1]) # safe to do, gives the correct answer
    loss_target = p[np.arange(p.shape[0]),target_index]
    loss = -np.log(loss_target)
    dprediction = p
    dprediction[np.arange(p.shape[0]), target_index] = dprediction[np.arange(p.shape[0]),target_index]-1
    dprediction = dprediction / loss.shape[0]
    loss_mean = np.mean(loss)
    return loss_mean, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.grad = X.copy()
        self.grad[self.grad<0] = 0
        self.grad[self.grad>0] = 1
        X_solve = X.copy()
        X_solve[X_solve<0]=0
        return X_solve
    
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return self.grad * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        m = np.dot(X, self.W.value)
        out = m + self.B.value
        #print(out)
        return out 

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        #print('d_out')
        #print(d_out)
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out)
        self.W.grad = np.dot(np.transpose(self.X), d_out)
        out = np.dot(d_out, np.transpose(self.W.value))
        #print(out)
        return out

    def params(self):
        return {'W': self.W, 'B': self.B}
