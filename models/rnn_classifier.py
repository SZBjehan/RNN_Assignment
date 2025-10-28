import numpy as np
import sys

sys.path.append("mydnn")
from mydnn.linear import Linear
from mydnn.rnn_cell import RNNCell

#################################################################################
# input_size: H_in, scalar, the number of expected features in the input x
# hidden_size: H_out, scalar, the number of features in the hidden state
# -------------------------------------------------------------------------------
# x:  maxtrix,      N x seq_len x H_in, Input sequence
# h_0: maxtrix,     num_layers x H_out, Initial hidden states
# delta: maxtrix,   N x H_out,          gradient w.r.t. current hidden layer
#################################################################################

class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [
            (
                RNNCell(input_size, hidden_size)
                if i == 0
                else RNNCell(hidden_size, hidden_size)
            )
            for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    # DO NOT Change this method
    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size)

        Output (y): logits

        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # add necessary code
        # hidden = TODO
        
        # TODO self. x = 
        # TODO update h_prev, rnn_cell, hidden[l], etc. for each time step t and each layer l
        # TODO logits = 
        
        return logits

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        # add necessary code
        # TODO implement the backward pass algorithm here
        # ---------------------------start CODE adding
        for t in reversed(range(seq_len)):
            for l in reversed(range(self.num_layers)):
                h_t = self.hiddens[t + 1][l]

                if l == 0:
                    h_prev_l = self.x[:, t]
                else:
                    h_prev_l = self.hiddens[t + 1][l - 1]

                h_prev_t = self.hiddens[t][l]

                rnn_cell = self.rnn[l]
                dx, dh_prev_t = rnn_cell.backward(dh[l], h_t, h_prev_l, h_prev_t)

                dh[l] = dh_prev_t

                if l > 0:
                    dh[l - 1] += dx
        # ---------------------------end CODE adding
        return dh / batch_size
