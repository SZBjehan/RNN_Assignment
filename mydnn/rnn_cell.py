import numpy as np
from mydnn.activation import *

#################################################################################
# input_size: H_in, scalar, the number of expected features in the input x
# hidden_size: H_out, scalar, the number of features in the hidden state
# -------------------------------------------------------------------------------
# x: x_t,               maxtrix, N x H_in, input at current time step
# h_prev_t: h_{t-1,l},  maxtrix, N x H_out, previous time step hidden state of current layer
# h_t: h_{t,l},         maxtrix, N x H_out, current time step hidden state of current layer
# -------------------------------------------------------------------------------
# W_ih: maxtrix, H_out x H_in,  weight between input and hidden
# b_ih: vector,  H_out,         bias between input and hidden
# W_hh: maxtrix, H_out x H_out, weight between previous hidden and current hidden
# b_hh: vector,  H_out,         weight between previous hidden and current hidden
# ----USED in BACKWARD calculation ----------------------------------------------
# delta: maxtrix,   N x H_out,      gradient w.r.t. current hidden layer
# dx: matrix,       N x H_in,       gradient w.r.t. input layer
# -------------------------------------------------------------------------------
# dh_prev_t: maxtrix, N x H_out,    gradient w.r.t. hidden state at previous time step
# dW_ih: maxtrix,   H_out x H_in,   gradient of weight between input and hidden
# db_ih: vector,  H_out,            gradient of bias between input and hidden
# dW_hh: maxtrix, H_out x H_out,    gradient of weight between previous hidden and current hidden
# db_hh: vector,  H_out,             gradient of bias between previous hidden and current hidden
#################################################################################

class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # Weight definitions, TODO: add your code
        # ---------------------------start
        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)
        # ---------------------------end

        # Gradient definitions, TODO: add necessary code
        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_ih = np.zeros_like(self.b_ih)
        self.db_hh = np.zeros_like(self.b_hh)
        
    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        #TODO add necessary code
        self.W_ih = W_ih.astype(float).copy()
        self.W_hh = W_hh.astype(float).copy()
        # PyTorch gives 1D biases
        self.b_ih = b_ih.astype(float).copy().reshape(-1)
        self.b_hh = b_hh.astype(float).copy().reshape(-1)

        # Reset grads
        self.zero_grad()
        

    # DO NOT change this method
    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    # DO NOT change this method
    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input 
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(W_ih x_t + b_ih + W_hh h_t−1 + b_hh) 
        """

        # TODO, add necessary code to calculate h_t
        # Ensure batch dimension
        if x.ndim == 1:
            x_b = x.reshape(1, -1)
        else:
            x_b = x
        if h_prev_t.ndim == 1:
            h_prev = h_prev_t.reshape(1, -1)
        else:
            h_prev = h_prev_t
        
        z = x_b @ self.W_ih.T + h_prev @ self.W_hh.T
        z = z + self.b_ih.reshape(1, -1) + self.b_hh.reshape(1, -1)
        h_t = np.tanh(z)

        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t. the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t. the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # Add necessary code to calculate dz 
        # dz = #TODO
        dz = delta * (1.0 - np.square(h_t))

        # Parameter grads (average across batch to match PyTorch mean reduction)
        self.dW_ih += (dz.T @ h_prev_l) / batch_size
        self.dW_hh += (dz.T @ h_prev_t) / batch_size
        # Both biases receive the same dz
        db = dz.sum(axis=0) / batch_size
        self.db_ih += db
        self.db_hh += db

        # Input and h_prev grads
        dx = dz @ self.W_ih
        dh_prev_t = dz @ self.W_hh
        
        # Add necessary code to compute the averaged gradients (per batch) of the weights and biases
        # self.dW_ih = TODO
        # self.dW_hh = TODO
        # self.db_ih = TODO
        # self.db_hh = TODO
        

        # Add necessary code to compute dx, dh_prev_t
        # dx = TODO
        # dh_prev_t = TODO
        
        return dx, dh_prev_t
