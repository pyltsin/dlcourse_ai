import numpy as np
import math

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W*W)
    grad = 2 * W * reg_strength
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    f = preds.copy()
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

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        padding = self.padding
        X = np.insert(X, 0, np.zeros([ padding]), axis=2)
        X = np.insert(X, X.shape[2], np.zeros([ padding]), axis=2)
        X = np.insert(X, 0, np.zeros([ padding]), axis=1)
        X = np.insert(X, X.shape[1], np.zeros([ padding]), axis=1)
        self.X = X
        batch_size, height, width, channels = X.shape

        out_height = height - (self.filter_size - 1)
        out_width = width - (self.filter_size - 1)
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        w_trans = self.W.value.reshape([-1, self.out_channels])
        out = np.zeros([batch_size, out_height, out_width, self.out_channels])
        for y in range(out_height):
            for x in range(out_width):
                X_cl = X[: , y:y + self.filter_size , x:x + self.filter_size , :]
                X_cl_tr = X_cl.reshape([batch_size, -1])
                b_out = np.dot(X_cl_tr, w_trans)
                b_out = b_out + self.B.value
                out[:, y, x, :] = b_out
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        padding = self.padding

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
       
        ## self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out)
        ## self.W.grad = np.dot(np.transpose(self.X), d_out)
        ## out = np.dot(d_out, np.transpose(self.W.value))
        #print(out)
        ## return out

        X = np.zeros_like(self.X)
        #[batch_seize, y, x, input_channels]
        
        #d_out
        #[batch_seize, out_height, out_width, out_channels]
        
        #W
        #[filter, filter, in, out]
        
        #B
        #[out_channels]
        
        w_reshape = self.W.value.reshape([-1, self.out_channels])
        # [filter * filter * in, out ]
        
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                X_filter = self.X[: , y:y + self.filter_size , x:x + self.filter_size , :]
                #[batch_size, filter, filter, in_channel]
                X_filter_reshape = X_filter.reshape([batch_size, -1])
                #[batch_size, filter * filter * in_channel]
                
                point = d_out[:, y, x, :]
                #[batch_size, out_channel]
                
                delta_w_grad =  np.dot(np.transpose(X_filter_reshape), point)
                #[filter * filter * in_channel, out_channel]
                delta_w_grad_reshape = delta_w_grad.reshape([ self.filter_size, self.filter_size, self.in_channels, self.out_channels])
                
                self.W.grad = self.W.grad  + delta_w_grad_reshape 
                
                
                delta_b_grad = np.dot(np.ones([1, batch_size]), point)
                delta_b_grad_reshape = delta_b_grad[0]

                self.B.grad = self.B.grad  + delta_b_grad_reshape
                delta_x = np.dot(point, np.transpose(w_reshape))
                #[batch_size, out_channel] *[out, filter * filter * in] = [batch_size, filter * filter * in]
                delta_x_reshape = delta_x.reshape([batch_size, self.filter_size, self.filter_size, self.in_channels])
                X[: , y:y + self.filter_size , x:x + self.filter_size , :] =  \
                X[: , y:y + self.filter_size , x:x + self.filter_size , :] +  \
                delta_x_reshape
                
                

                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
        return X[:, padding: height - padding,  padding: width - padding ,:]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape

        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = math.floor((height-self.pool_size)/self.stride)+1
        out_width = math.floor((width-self.pool_size)/self.stride)+1
        
        out = np.zeros([batch_size, out_height, out_width, channels])
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(channels):
                        x_source = x  * self.stride
                        y_source = y * self.stride
                        pool = X[batch, x_source: x_source+self.pool_size, y_source: y_source+self.pool_size,channel]
                        pool_reshape = pool.reshape([-1])
                        maximum = np.max(pool_reshape)
                        out[batch,x, y, channel] = maximum
        return out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        out = np.zeros_like( self.X)
        
        _, out_height, out_width, out_channels = d_out.shape
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(channels):
                        zeros = np.zeros([self.pool_size, self.pool_size])
                        x_source = x  * self.stride
                        y_source = y * self.stride
                        pool = self.X[batch, x_source: x_source+self.pool_size, y_source: y_source+self.pool_size,channel]

                        pool_reshape = pool.reshape([-1])
                        maximum = np.max(pool_reshape)
                        count = np.count_nonzero(pool==maximum)
                        argmax = np.argwhere(pool==maximum)
                        zeros[argmax[:,0], argmax[:,1]] = d_out[batch,x, y, channel] /count
                        out[batch, x_source: x_source+self.pool_size, y_source: y_source+self.pool_size,channel] = out[batch, x_source: x_source+self.pool_size, y_source: y_source+self.pool_size,channel] +  zeros
        return out
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape([batch_size, -1])

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape

        return d_out.reshape([batch_size, height, width, channels])

    def params(self):
        # No params!
        return {}
